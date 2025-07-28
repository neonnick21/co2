import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
# Import GIoU loss for robust bounding box cost
from torchvision.ops.boxes import generalized_box_iou_loss

class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(RFDETR, self).__init__()
        
        # Backbone: ResNet
        self.backbone = resnet50(weights='DEFAULT')
        # Use layers up to `layer4` (the last convolutional block before avgpool/fc)
        # ResNet50 typically outputs 2048 channels from layer4
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Convolutional layer to reduce channels from 2048 to 256 (common DETR feature dim)
        self.channel_reduction = nn.Conv2d(2048, 256, kernel_size=1)

        # Transformer Encoder
        # d_model=256: Dimension of the input features to the transformer
        # nhead=8: Number of attention heads
        # batch_first=True: Input and output tensors have batch dimension first
        encoder_layers = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        
        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)
        
        # Receptive Field Enhancement Module
        self.rfe_module = self.create_rfe_module()
        
        # Prediction Heads
        # num_classes includes the 'no-object' class (e.g., 4 real classes + 1 no-object = 5)
        self.class_head = nn.Linear(256, num_classes)
        # Bounding box head outputs 4 coordinates (x_center, y_center, width, height) or (x1, y1, x2, y2)
        # DETR typically normalizes coordinates to [0, 1]
        self.bbox_head = nn.Linear(256, 4)

        # Query embeddings: Learnable embeddings that act as object queries for the decoder
        self.query_embeddings = nn.Embedding(num_queries, 256)

        # Initialize weights for new layers (optional, but good practice)
        # For Conv2d
        nn.init.xavier_uniform_(self.channel_reduction.weight)
        nn.init.constant_(self.channel_reduction.bias, 0)
        # For Linear heads
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0)
        nn.init.xavier_uniform_(self.bbox_head.weight)
        nn.init.constant_(self.bbox_head.bias, 0)

    def create_rfe_module(self):
        """
        Creates the Receptive Field Enhancement (RFE) module.
        A simple 2-layer CNN block to process backbone features before transformer.
        """
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, images):
        """
        Forward pass of the RFDETR model.

        Args:
            images (torch.Tensor): Input image tensor(s). Shape: [batch_size, C, H, W]

        Returns:
            tuple: A tuple containing:
                - class_logits (torch.Tensor): Predicted class logits. Shape: [batch_size, num_queries, num_classes]
                - bbox_preds (torch.Tensor): Predicted bounding box coordinates (normalized [0,1]). 
                                             Shape: [batch_size, num_queries, 4]
        """
        # 1. Backbone: Extract features from images
        features = self.backbone_layers(images) # Output: [bs, 2048, H_feat, W_feat]
        
        # 2. Channel Reduction: Reduce feature channels
        reduced_features = self.channel_reduction(features) # Output: [bs, 256, H_feat, W_feat]
        
        # 3. RFE Module: Enhance receptive field
        enhanced_features = self.rfe_module(reduced_features) # Output: [bs, 256, H_feat, W_feat]
        
        # 4. Prepare for Transformer: Flatten spatial dimensions and permute
        bs, c, h_feat, w_feat = enhanced_features.shape
        # Flatten spatial (H_feat * W_feat) into a sequence dimension
        enhanced_features = enhanced_features.flatten(2).permute(0, 2, 1) # Output: [bs, H_feat*W_feat, 256]
        
        # 5. Transformer Encoder: Process visual features
        memory = self.transformer_encoder(enhanced_features) # Output: [bs, H_feat*W_feat, 256]
        
        # 6. Prepare Queries: Repeat learnable queries for each image in the batch
        # query_embeddings.weight: [num_queries, 256]
        queries = self.query_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1) # Output: [bs, num_queries, 256]
        
        # 7. Transformer Decoder: Decode object queries using encoded features
        outputs = self.transformer_decoder(queries, memory) # Output: [bs, num_queries, 256]
        
        # 8. Prediction Heads: Convert decoder outputs to class logits and bbox predictions
        class_logits = self.class_head(outputs) # Output: [bs, num_queries, num_classes]
        # Apply sigmoid to bounding box predictions to normalize them to [0, 1] range
        bbox_preds = self.bbox_head(outputs).sigmoid() # Output: [bs, num_queries, 4]
        
        return class_logits, bbox_preds

def compute_loss(pred_logits, pred_boxes, targets, num_classes, device, 
                 cost_class_weight=1.0, cost_bbox_weight=5.0, cost_giou_weight=2.0):
    """
    Computes the total loss for a batch of predictions and targets using bipartite matching.

    Args:
        pred_logits (torch.Tensor): Predicted class logits. Shape: [batch_size, num_queries, num_classes]
        pred_boxes (torch.Tensor): Predicted bounding box coordinates (normalized [0,1]). Shape: [batch_size, num_queries, 4]
        targets (list[dict]): List of dictionaries, each containing 'labels' and 'boxes' for a target image.
        num_classes (int): Total number of classes including the 'no-object' class.
        device (torch.device): The device (CPU or CUDA) to perform calculations on.
        cost_class_weight (float): Weight for the classification cost in matching.
        cost_bbox_weight (float): Weight for the L1 bounding box cost in matching.
        cost_giou_weight (float): Weight for the GIoU bounding box cost in matching.

    Returns:
        torch.Tensor: The total computed loss for the batch.
    """
    total_class_loss = torch.tensor(0.0, device=device)
    total_bbox_l1_loss = torch.tensor(0.0, device=device)
    total_giou_loss = torch.tensor(0.0, device=device)
    
    batch_size, num_queries = pred_logits.shape[:2]
    no_object_class_label = num_classes - 1 # Index of the 'no-object' class

    # Iterate through each image in the batch
    for i in range(batch_size):
        # Get predictions and targets for the current image
        pred_logits_i = pred_logits[i] # [num_queries, num_classes]
        pred_boxes_i = pred_boxes[i]   # [num_queries, 4]
        target_labels_i = targets[i]['labels'] # [num_targets]
        target_boxes_i = targets[i]['boxes']   # [num_targets, 4]

        num_targets = len(target_labels_i)
        
        # If there are no ground truth objects in this image, all predictions should be 'no-object'
        if num_targets == 0:
            # Classification loss for 'no-object' class for all queries
            class_loss = F.cross_entropy(pred_logits_i, torch.full((num_queries,), no_object_class_label, device=device))
            total_class_loss += class_loss
            continue # No bbox or giou loss if no targets

        # --- Construct the Cost Matrix for Hungarian Matching ---

        # 1. Classification Cost: Negative log-likelihood between predicted logits and target labels
        # Reshape target_labels_i to [1, num_targets] for broadcasting
        # We compute -log(P(class | query)) for each query and each target class
        # pred_logits_i.softmax(-1): [num_queries, num_classes] probabilities
        # pred_logits_i.softmax(-1)[:, target_labels_i]: [num_queries, num_targets] probabilities for target classes
        # .log(): then take log
        cost_class = -pred_logits_i.softmax(-1)[:, target_labels_i].log()
        # This will result in a [num_queries, num_targets] matrix where
        # cost_class[q, t] = -log(P(target_label_t | query_q_prediction))

        # 2. L1 Bounding Box Cost: Absolute difference between predicted and target boxes
        # torch.cdist computes pairwise distances. p=1 for L1 distance.
        # pred_boxes_i: [num_queries, 4], target_boxes_i: [num_targets, 4]
        # Result: [num_queries, num_targets]
        cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1)
        
        # 3. GIoU Bounding Box Cost: Measures spatial overlap and alignment
        # Requires boxes in (x1, y1, x2, y2) format, which is assumed for pred_boxes_i and target_boxes_i (normalized [0,1])
        # Returns a [num_queries, num_targets] matrix where lower values are better (closer to 1)
        # We use (1 - giou_score) as a cost, so lower is better.
        cost_giou = 1 - generalized_box_iou_loss(pred_boxes_i, target_boxes_i, reduction='none')

        # Combine costs for bipartite matching using weights
        # All cost matrices should have shape [num_queries, num_targets]
        C = cost_class_weight * cost_class + \
            cost_bbox_weight * cost_bbox + \
            cost_giou_weight * cost_giou
        
        # Move the cost matrix to CPU and convert to numpy for linear_sum_assignment
        # This is the unavoidable bottleneck for DETR's Hungarian matching.
        C = C.detach().cpu().numpy()
        
        # Find optimal matches using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(C) # row_ind are query indices, col_ind are target indices

        # --- Compute Losses based on Found Matches ---

        # 1. Classification Loss (matched queries)
        # For matched queries, compute cross-entropy loss against their assigned target labels
        matched_pred_logits = pred_logits_i[row_ind]
        matched_target_labels = target_labels_i[col_ind]
        class_loss_matched = F.cross_entropy(matched_pred_logits, matched_target_labels)
        total_class_loss += class_loss_matched

        # 2. Classification Loss (unmatched queries - background)
        # Queries not in row_ind are considered unmatched. They should predict 'no-object'.
        unmatched_preds_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
        unmatched_preds_mask[row_ind] = False
        
        # Labels for unmatched predictions are all 'no-object'
        num_unmatched_preds = torch.sum(unmatched_preds_mask).item()
        if num_unmatched_preds > 0:
            class_loss_unmatched = F.cross_entropy(
                pred_logits_i[unmatched_preds_mask], 
                torch.full((num_unmatched_preds,), no_object_class_label, device=device)
            )
            total_class_loss += class_loss_unmatched
        
        # 3. Bounding Box L1 Loss (only for matched queries)
        matched_pred_boxes = pred_boxes_i[row_ind]
        matched_target_boxes = target_boxes_i[col_ind]
        # Reduction='mean' means average over matched boxes
        bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
        total_bbox_l1_loss += bbox_l1_loss

        # 4. GIoU Loss (only for matched queries)
        giou_loss = generalized_box_iou_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
        total_giou_loss += giou_loss

    # Average total losses over the batch size
    # Apply coefficients to the total average losses
    final_loss = (total_class_loss / batch_size) + \
                 (cost_bbox_weight * total_bbox_l1_loss / batch_size) + \
                 (cost_giou_weight * total_giou_loss / batch_size)

    return final_loss

