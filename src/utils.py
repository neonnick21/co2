import torch
import numpy as np
from pycocotools import mask as maskUtils

def generalized_iou(boxes1, boxes2):
    """
    Compute Generalized IoU between two sets of boxes
    Args:
        boxes1: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (N, 4) in [x1, y1, x2, y2] format
    Returns:
        giou: Tensor of shape (N,)
    """
    # Calculate intersection area
    inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
    inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
    inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
    inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
    
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    
    # Calculate union area
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area1 + area2 - inter_area
    
    # Calculate enclosing box area
    enclose_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
    enclose_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
    enclose_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
    enclose_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
    enclose_area = torch.clamp(enclose_x2 - enclose_x1, min=0) * torch.clamp(enclose_y2 - enclose_y1, min=0)
    
    # Calculate IoU and GIoU
    iou = inter_area / (union_area + 1e-6)
    giou = iou - (enclose_area - union_area) / (enclose_area + 1e-6)
    
    return giou

def coco_to_pytorch(target, height, width):
    """
    Convert COCO annotations to PyTorch format
    """
    boxes = []
    labels = []
    for obj in target:
        # Convert from [x, y, w, h] to [x_min, y_min, x_max, y_max]
        x, y, w, h = obj['bbox']
        boxes.append([x, y, x + w, y + h])
        labels.append(obj['category_id'])
    
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    
    return {
        'boxes': boxes,
        'labels': labels
    }

def compute_cost_matrix(pred_logits, pred_boxes, target_boxes, target_labels):
    """
    Compute cost matrix for Hungarian matching
    """
    # Classification cost
    cls_cost = -pred_logits[:, target_labels]
    
    # L1 cost
    box_cost = torch.cdist(pred_boxes, target_boxes, p=1)
    
    # GIoU cost
    giou = generalized_iou(pred_boxes, target_boxes)
    giou_cost = -giou
    
    # Total cost
    total_cost = cls_cost + 5 * box_cost + 2 * giou_cost
    return total_cost

def save_training_plot(log_path, output_path):
    """
    Generate and save training loss plot
    """
    with open(log_path, 'r') as f:
        log = json.load(f)
    
    epochs = [entry['epoch'] for entry in log]
    train_loss = [entry['train_loss'] for entry in log]
    val_loss = [entry['val_loss'] for entry in log]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()