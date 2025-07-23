import torch
import torchvision
from torch import nn
from transformers import DetrConfig, DetrModel, DetrForObjectDetection

class ReceptiveFieldEnhancement(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dilated_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.dilated_conv(x)

class RFDetr(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Backbone with dilated convolutions
        backbone = torchvision.models.resnet50(weights="DEFAULT")
        
        # Replace first convolution with dilated version
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, dilation=1)
        
        # Add RFE modules after each block
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            ReceptiveFieldEnhancement(256, 256),
            backbone.layer2,
            ReceptiveFieldEnhancement(512, 512),
            backbone.layer3,
            ReceptiveFieldEnhancement(1024, 1024),
            backbone.layer4,
            ReceptiveFieldEnhancement(2048, 2048)
        )
        
        # FPN for multi-scale features
        self.fpn = torchvision.ops.FeaturePyramidNetwork(
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        # DETR configuration
        config = DetrConfig(
            num_labels=num_classes,
            num_queries=100,
            d_model=256
        )
        
        # DETR model
        self.detr = DetrForObjectDetection(config)
        
        # Prediction heads
        self.classifier = nn.Linear(256, num_classes + 1)  # +1 for "no object"
        self.bbox_predictor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )

    def forward(self, pixel_values):
        # Extract features
        features = self.backbone(pixel_values)
        
        # FPN requires features as a dictionary
        fpn_features = {
            '0': features[0],  # layer1 output
            '1': features[1],  # layer2 output
            '2': features[2],  # layer3 output
            '3': features[3]   # layer4 output
        }
        
        # FPN processing
        fpn_output = self.fpn(fpn_features)
        
        # DETR forward pass
        outputs = self.detr(
            pixel_values=None, 
            pixel_mask=None, 
            features=fpn_output['0']  # Use highest resolution feature map
        )
        
        # Get predictions
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        return {
            "logits": logits,
            "pred_boxes": pred_boxes
        }