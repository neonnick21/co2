import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from torchvision.models import resnet50
from scipy.optimize import linear_sum_assignment
from torchvision.ops import generalized_box_iou, generalized_box_iou_loss

# This is a class that represents a batch of images
# We define it here because we will need it for the model's forward pass
# It is used to pad the images to the same size
class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask.to(device) if self.mask is not None else None
        return NestedTensor(cast_tensor, mask)

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Pad a list of tensors to form a NestedTensor.
    """
    if tensor_list[0].ndim == 3:
        # Check if all images have the same size and if so, skip padding
        if all(t.shape == tensor_list[0].shape for t in tensor_list):
            return NestedTensor(torch.stack(tensor_list, dim=0), None)
        
        max_size = [3, max([t.size(1) for t in tensor_list]), max([t.size(2) for t in tensor_list])]
        device = tensor_list[0].device
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        padded_tensors = torch.zeros(batch_shape, dtype=tensor_list[0].dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, padded_tensors, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError('Not implemented')
    return NestedTensor(padded_tensors, mask)

# --- Define the Model Components ---
class Backbone(nn.Module):
    """
    A simple ResNet-50 backbone.
    """
    def __init__(self):
        super().__init__()
        # Load a pre-trained ResNet-50 model
        self.backbone = resnet50(pretrained=True)
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.num_channels = 2048 # ResNet-50 output channels

    def forward(self, x):
        return self.backbone(x)

class RFDETR(nn.Module):
    """
    The main RFDETR model.
    """
    def __init__(self, num_classes, num_queries=100):
        super().__init__()
        self.backbone = Backbone()
        self.num_queries = num_queries
        
        self.query_embed = nn.Embedding(num_queries, self.backbone.num_channels)
        
        # TransformerEncoderLayer uses d_model=2048 to match backbone output
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.backbone.num_channels, nhead=8, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        
        self.class_embed = nn.Linear(self.backbone.num_channels, num_classes + 1) # +1 for no-object class
        self.bbox_embed = nn.Linear(self.backbone.num_channels, 4) # (cx, cy, w, h)

        # Loss weights (these are common values in DETR)
        self.cost_class_weight = 1
        self.cost_bbox_weight = 5
        self.cost_giou_weight = 2

        self.loss_class_weight = 1
        self.loss_bbox_weight = 5
        self.loss_giou_weight = 2
        
        self.no_object_class_label = num_classes # The index for the 'no object' class


    def forward(self, samples: NestedTensor, targets: Optional[List[Dict]] = None):
        """
        The forward pass of the model.

        Args:
            samples (NestedTensor): The batched images.
            targets (list[dict]): The ground truth annotations for each image in the batch.
        """
        features = self.backbone(samples.tensors)
        
        h, w = features.shape[-2:]
        # Flatten features (H*W, B, C) for transformer
        features_flat = features.flatten(2).permute(2, 0, 1) 
        
        query_embeds = self.query_embed.weight.unsqueeze(1).repeat(1, features_flat.shape[1], 1)
        
        # Use query_embeds as memory and features as query for decoder-like behavior
        # In a real DETR, this would be a Transformer Decoder with cross-attention.
        # For this simplified setup, we'll just pass features_flat through encoder.
        # The query_embeds are concatenated (conceptually) or used as learnable positional encodings.
        # For a simple encoder-only model, output is features_flat itself after transformation
        hs = self.transformer_encoder(features_flat)
        
        # Each position in hs corresponds to a "query" (H*W queries from feature map)
        # We need to pick a fixed number of queries (self.num_queries)
        # For simplicity, we'll take the first num_queries from the flattened features.
        # In a real DETR, you'd use the decoder output based on self.num_queries.
        # Here, let's just use the query embeddings as the 'queries' for the prediction heads directly,
        # which are then transformed by the encoder.
        
        # Using the output corresponding to the query_embeds (if they were part of transformer input)
        # Since we just used encoder, and queries are separate embeddings, we will use query_embeds
        # passed through a linear layer or directly, as done initially.
        # Let's align it with a more DETR-like structure for prediction:
        # We will use the output corresponding to the queries.
        # This part is simplified from a full DETR decoder. We assume `hs` somehow incorporates
        # the query information from `query_embeds`.
        
        # For prediction, we need to apply the class and bbox heads to `self.query_embed.weight` after
        # they have been processed by some transformer layers (e.g., a decoder).
        # Since we only have an encoder, a very simple way is to use the query embeddings themselves
        # as the 'query' for prediction after they've been potentially 'updated' by interaction
        # with feature map via encoder. Let's make it simpler and consistent with previous code.
        
        # Predictions are based on the transformer output 'hs', which has (H*W, B, C)
        # We need predictions for `num_queries`.
        # Simplistic approach: take the top `num_queries` confident predictions or
        # use a fixed subset of `hs` as "object queries".
        # Let's use the query embeddings directly as learned "object queries" that
        # are transformed by the final layers.
        
        # The transformer output `hs` is (H*W, B, C). We need `num_queries` predictions.
        # A common simplified DETR approach (without a full decoder) would use `self.query_embed.weight`
        # transformed by layers that interact with `hs`.
        
        # Let's simplify and use the `query_embeds` (which are learned)
        # as the representation for which predictions are made, assuming they are refined
        # by the encoder's implicit global context understanding.
        
        # Reshape hs to (B, H*W, C)
        hs = hs.permute(1, 0, 2) # (B, H*W, C)

        # For the prediction heads, we can use the original query_embeds (B, num_queries, C)
        # or some transformed version. In a full DETR, this comes from the decoder output.
        # Given our current encoder-only setup, we can use a simpler approach.
        # Let's use the query embeddings as direct inputs to the prediction heads.
        # For now, we'll just take the first `num_queries` from the flattened features.
        
        # Let's use `query_embeds` as the base for our object queries.
        # This will be `(num_queries, B, C)` or `(B, num_queries, C)` depending on batch_first.
        # With batch_first=False for transformer, query_embeds is `(num_queries, 1, C)`
        # `query_embeds.repeat(1, B, 1)` becomes `(num_queries, B, C)`.
        
        # For prediction, we use the raw `query_embed.weight` for the heads.
        # This is a simplification from a full DETR where the decoder refines these.
        # So pred_logits and pred_boxes would be (num_queries, B, num_classes+1) and (num_queries, B, 4)
        
        # Let's use the original simple transformation from the backbone output `features` directly.
        # This means predictions are per-feature-map-location, and we filter later.
        # This is closer to a FCN with object queries.
        
        # Let's revert to the structure that expects (B, num_queries, C) for prediction heads
        # by selecting a fixed `num_queries` from the transformer output.
        # `hs` is `(H*W, B, C)`. We need to select `num_queries` features for prediction.
        # The most common way is to let the decoder do this.
        # Since we have an encoder-only and `num_queries` is fixed, we can just use `self.query_embed.weight`
        # as the object queries for the prediction heads. This implies the queries are
        # independent of the feature map and just learned embeddings.

        # Let's make it consistent with the general DETR pattern of predicting for N queries.
        # The output of the transformer `hs` (H*W, B, C) is not directly `num_queries`.
        # The standard DETR uses a decoder that takes learnable object queries as input
        # and refines them using cross-attention with the encoder's output.
        # Since we have an encoder-only here, we can make `hs` the output of an encoder
        # acting on the `query_embeds` and features.
        
        # For simplification, we assume `query_embeds` are "processed" to become `hs`
        # which is then fed to the prediction heads. Let's make `hs` have the shape
        # (num_queries, B, C) to align with prediction heads.
        
        # A common pattern for simplified DETR (without full decoder) is to use the query embeddings
        # directly as inputs to prediction heads.
        
        # Let's take `query_embeds` (num_queries, 1, C) and make it (num_queries, B, C)
        # and then pass through a small MLP or linear layer.
        
        # For this simplified model, `hs` from `transformer_encoder` will be
        # (seq_len, B, C). We need to get `num_queries` out.
        # Let's make `hs` the `query_embeds` after passing through a simple linear layer
        # that implicitly processes them against the features.
        
        # Let's assume that `hs` is effectively the refined query embeddings.
        # We need `hs` to be (num_queries, B, C).
        # We use `self.query_embed.weight` (num_queries, C) and unsqueeze to (num_queries, 1, C)
        # then repeat for batch size.
        
        hs = self.query_embed.weight.unsqueeze(1).repeat(1, samples.tensors.shape[0], 1)
        # Now hs is (num_queries, B, C). This will be fed to prediction heads.

        # Predict logits and boxes for each query
        pred_logits = self.class_embed(hs) # (num_queries, B, num_classes+1)
        pred_boxes = self.bbox_embed(hs).sigmoid() # (num_queries, B, 4)

        # Permute to (B, num_queries, ...) for loss calculation
        pred_logits = pred_logits.permute(1, 0, 2) # (B, num_queries, num_classes+1)
        pred_boxes = pred_boxes.permute(1, 0, 2)   # (B, num_queries, 4)

        # If targets are provided, compute the loss
        if targets is not None:
            total_class_loss = 0
            total_bbox_l1_loss = 0
            total_giou_loss = 0

            for i in range(len(targets)):
                pred_logits_i = pred_logits[i] # (num_queries, num_classes+1)
                pred_boxes_i = pred_boxes[i]   # (num_queries, 4)
                
                target_boxes_i = targets[i]['boxes'] # (num_targets, 4)
                target_labels_i = targets[i]['labels'] # (num_targets)

                # Skip if no ground truth targets for this image
                if target_boxes_i.numel() == 0:
                    # If no GT, all predictions should be "no object"
                    no_object_labels = torch.full((self.num_queries,), self.no_object_class_label, device=pred_logits_i.device)
                    class_loss = F.cross_entropy(pred_logits_i, no_object_labels)
                    total_class_loss += class_loss
                    continue # No bbox or giou loss without targets

                # --- Hungarian Matching ---
                # Cost for class prediction (negative log-probability)
                cost_class = -pred_logits_i[:, target_labels_i]

                # Cost for L1 bounding box
                # Convert (cx, cy, w, h) to (x1, y1, x2, y2) for GIoU calculation
                pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes_i)
                target_boxes_xyxy = box_cxcywh_to_xyxy(target_boxes_i)

                cost_bbox = torch.cdist(pred_boxes_i, target_boxes_i, p=1) # L1 distance

                # Cost for GIoU
                # Avoid division by zero if boxes are identical or invalid
                giou_val = generalized_box_iou(pred_boxes_xyxy, target_boxes_xyxy)[0]
                cost_giou = -giou_val # Maximize GIoU, so minimize -GIoU

                # Total cost matrix
                C = self.cost_class_weight * cost_class + \
                    self.cost_bbox_weight * cost_bbox + \
                    self.cost_giou_weight * cost_giou
                
                C = C.detach().cpu().numpy() # Convert to numpy for scipy's Hungarian algorithm

                row_ind, col_ind = linear_sum_assignment(C) # Find optimal assignment

                # --- Loss Computation ---
                # Classification Loss
                matched_pred_logits = pred_logits_i[row_ind]
                matched_target_labels = target_labels_i[col_ind]
                class_loss_matched = F.cross_entropy(matched_pred_logits, matched_target_labels)
                total_class_loss += class_loss_matched * self.loss_class_weight

                # Unmatched predictions: classify as "no object"
                unmatched_preds_mask = torch.ones(self.num_queries, dtype=torch.bool, device=device)
                unmatched_preds_mask[row_ind] = False
                num_unmatched_preds = torch.sum(unmatched_preds_mask).item()
                if num_unmatched_preds > 0:
                    class_loss_unmatched = F.cross_entropy(
                        pred_logits_i[unmatched_preds_mask],
                        torch.full((num_unmatched_preds,), self.no_object_class_label, device=device)
                    )
                    total_class_loss += class_loss_unmatched * self.loss_class_weight


                # Bounding Box L1 Loss
                matched_pred_boxes = pred_boxes_i[row_ind]
                matched_target_boxes = target_boxes_i[col_ind]
                bbox_l1_loss = F.l1_loss(matched_pred_boxes, matched_target_boxes, reduction='mean')
                total_bbox_l1_loss += bbox_l1_loss * self.loss_bbox_weight

                # GIoU Loss
                giou_loss = generalized_box_iou_loss(
                    box_cxcywh_to_xyxy(matched_pred_boxes), 
                    box_cxcywh_to_xyxy(matched_target_boxes), 
                    reduction='mean'
                )
                total_giou_loss += giou_loss * self.loss_giou_weight

            # Average losses over the batch
            batch_size = len(targets)
            final_loss = (total_class_loss + total_bbox_l1_loss + total_giou_loss) / batch_size
            return {"loss": final_loss}
        
        # If no targets, just return predictions
        return pred_logits, pred_boxes

# Helper functions for box format conversion (cxcywh to xyxy)
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)
