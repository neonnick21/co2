import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment

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
        bbox_preds = self.bbox_head(outputs)
        
        return class_logits, bbox_preds

def compute_loss(pred_logits, pred_boxes, targets):
    total_class_loss = 0
    total_bbox_loss = 0
    num_matches = 0

    # Iterate through each image in the batch
    for i in range(len(targets)):
        # Get predictions and targets for the current image
        pred_logits_i = pred_logits[i]
        pred_boxes_i = pred_boxes[i]
        target_labels_i = targets[i]['labels']
        target_boxes_i = targets[i]['boxes']

        # Skip images with no annotations
        if len(target_labels_i) == 0:
            continue

        # Compute cost matrices
        cost_class = -pred_logits_i[:, target_labels_i].softmax(-1).log()
        cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1)

        # Combine costs and use Hungarian algorithm
        total_cost = cost_class + cost_bbox
        row_ind, col_ind = linear_sum_assignment(total_cost.detach().cpu().numpy())

        # Compute losses for matched predictions
        for pred_idx, target_idx in zip(row_ind, col_ind):
            # Classification loss
            pred_label = pred_logits_i[pred_idx].unsqueeze(0)
            target_label = target_labels_i[target_idx].unsqueeze(0)
            class_loss = F.cross_entropy(pred_label, target_label)
            total_class_loss += class_loss

            # Bounding box loss (L1 loss)
            pred_box = pred_boxes_i[pred_idx]
            target_box = target_boxes_i[target_idx]
            bbox_loss = F.l1_loss(pred_box, target_box, reduction='sum')
            total_bbox_loss += bbox_loss

            num_matches += 1

    # Average losses over the number of matches found across the batch
    if num_matches > 0:
        total_loss = (total_class_loss + total_bbox_loss) / num_matches
    else:
        total_loss = torch.tensor(0.0, device=pred_logits.device)
        
    return total_loss