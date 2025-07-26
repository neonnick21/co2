import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment

# Note: The RFDETR class itself is already structured correctly and does not require changes.
class RFDETR(nn.Module):
    def __init__(self, num_classes, num_queries):
        super(RFDETR, self).__init__()
        
        # Backbone: ResNet
        self.backbone = resnet50(weights='DEFAULT')
        self.backbone_layers = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Convolutional layer to reduce channels from 2048 to 256
        self.channel_reduction = nn.Conv2d(2048, 256, kernel_size=1)

        # Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=6)
        
        # Transformer Decoder
        decoder_layers = nn.TransformerDecoderLayer(d_model=256, nhead=8, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=6)
        
        # Receptive Field Enhancement Module
        self.rfe_module = self.create_rfe_module()
        
        # Prediction Heads
        self.class_head = nn.Linear(256, num_classes)
        self.bbox_head = nn.Linear(256, 4)

        # Query embeddings
        self.query_embeddings = nn.Embedding(num_queries, 256)

    def create_rfe_module(self):
        return nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, images):
        features = self.backbone_layers(images)
        reduced_features = self.channel_reduction(features)
        enhanced_features = self.rfe_module(reduced_features)
        
        bs, c, h, w = enhanced_features.shape
        enhanced_features = enhanced_features.flatten(2).permute(0, 2, 1)
        
        memory = self.transformer_encoder(enhanced_features)
        queries = self.query_embeddings.weight.unsqueeze(0).repeat(bs, 1, 1)
        
        outputs = self.transformer_decoder(queries, memory)
        
        class_logits = self.class_head(outputs)
        bbox_preds = self.bbox_head(outputs).sigmoid() # Sigmoid for normalized box coords
        
        return class_logits, bbox_preds

def compute_loss(pred_logits, pred_boxes, targets, num_classes, device):
    """
    Computes the total loss for a batch of predictions and targets.
    This function replaces hungarian_matcher and the old compute_loss.
    """
    total_class_loss = torch.tensor(0.0, device=device)
    total_bbox_loss = torch.tensor(0.0, device=device)
    
    batch_size, num_queries = pred_logits.shape[:2]

    # Iterate through each image in the batch
    for i in range(batch_size):
        # Get predictions and targets for the current image
        pred_logits_i = pred_logits[i] # [num_queries, num_classes]
        pred_boxes_i = pred_boxes[i]   # [num_queries, 4]
        target_labels_i = targets[i]['labels'] # [num_targets]
        target_boxes_i = targets[i]['boxes']   # [num_targets, 4]

        num_targets = len(target_labels_i)
        
        # If no targets, compute classification loss for 'no-object' class
        if num_targets == 0:
            no_object_class = num_classes - 1
            class_loss = F.cross_entropy(pred_logits_i, torch.full((num_queries,), no_object_class, device=device))
            total_class_loss += class_loss
            continue

        # Compute classification cost: The negative log likelihood of the predictions
        # against the target labels. We broadcast this over all predictions.
        cost_class = -pred_logits_i[:, target_labels_i].softmax(-1).log()

        # Compute bounding box cost: The L1 distance between predicted and target boxes
        # We need to normalize the target boxes to be in the range [0, 1] if they aren't already.
        # This assumes the model outputs normalized box coordinates.
        cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1)
        
        # Combine costs and find optimal matches using the Hungarian algorithm (on CPU)
        C = cost_class + cost_bbox
        C = C.detach().cpu().numpy()
        
        # This is the bottleneck! It requires CPU and will be slow.
        row_ind, col_ind = linear_sum_assignment(C)

        # Matched Predictions: Compute losses for the pairs found
        class_loss_matched = F.cross_entropy(pred_logits_i[row_ind], target_labels_i[col_ind], reduction='mean')
        bbox_loss_matched = F.l1_loss(pred_boxes_i[row_ind], target_boxes_i[col_ind], reduction='mean')
        
        # Unmatched Predictions: Compute classification loss for 'no-object' class
        # We create a mask for unmatched queries and compute the loss on them
        unmatched_preds_mask = torch.ones(num_queries, dtype=torch.bool, device=device)
        unmatched_preds_mask[row_ind] = False
        
        no_object_class = num_classes - 1
        class_loss_unmatched = F.cross_entropy(
            pred_logits_i[unmatched_preds_mask], 
            torch.full((torch.sum(unmatched_preds_mask).item(),), no_object_class, device=device)
        )
        
        total_class_loss += class_loss_matched + class_loss_unmatched
        total_bbox_loss += bbox_loss_matched
    
    # Average the losses over the batch
    return (total_class_loss + total_bbox_loss) / batch_size