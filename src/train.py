import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from data_loader import get_data_loaders
from model import RFDetr
import argparse
import os
import time
import numpy as np
from datetime import datetime
import json  # Added missing import
from scipy.optimize import linear_sum_assignment  # Added for Hungarian matcher

# You need to define or import generalized_iou and compute_cost_matrix
# Placeholder implementations below (replace with your actual functions)
def generalized_iou(boxes1, boxes2):
    # Dummy implementation, replace with your actual GIoU function
    # boxes1, boxes2: (N, 4) tensors
    return torch.ones(boxes1.shape[0], device=boxes1.device)

def compute_cost_matrix(pred_boxes, target_boxes):
    # Dummy implementation, replace with your actual cost computation
    # Here, we use L1 distance as a simple cost
    if pred_boxes.shape[0] == 0 or target_boxes.shape[0] == 0:
        return torch.zeros((pred_boxes.shape[0], target_boxes.shape[0]), device=pred_boxes.device)
    cost = torch.cdist(pred_boxes, target_boxes, p=1)
    return cost

def compute_loss(outputs, targets):
    # Classification loss (Focal loss)
    cls_logits = outputs["logits"]
    cls_targets = [t["labels"] for t in targets]
    cls_loss = torch.nn.functional.cross_entropy(
        cls_logits.permute(0, 2, 1), 
        cls_targets, 
        reduction='mean'
    )
    
    # Bounding box loss (L1 + GIoU)
    box_preds = outputs["pred_boxes"]
    box_targets = [t["boxes"] for t in targets]
    
    # Hungarian matching
    matched_indices = hungarian_matcher(box_preds, box_targets)
    
    # Compute losses for matched pairs
    l1_loss = 0
    giou_loss = 0
    for i, (pred_idx, target_idx) in enumerate(matched_indices):
        pred_boxes = box_preds[i][pred_idx]
        target_boxes = box_targets[i][target_idx]
        
        # L1 loss
        l1_loss += torch.nn.functional.l1_loss(pred_boxes, target_boxes, reduction='sum')
        
        # GIoU loss
        giou_loss += (1 - generalized_iou(pred_boxes, target_boxes)).sum()
    
    total_loss = cls_loss + l1_loss + giou_loss
    return total_loss, cls_loss, l1_loss, giou_loss

def hungarian_matcher(preds, targets):
    # Simplified Hungarian matcher implementation
    # In practice, use a proper implementation like from DETR
    matched_indices = []
    for i in range(len(preds)):
        # Compute pairwise cost matrix (distance + classification)
        # This is a simplified version - actual implementation is more complex
        cost_matrix = compute_cost_matrix(preds[i], targets[i])
        row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().numpy())
        matched_indices.append((row_ind, col_ind))
    return matched_indices

def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    cls_loss = 0.0
    box_loss = 0.0
    giou_loss = 0.0
    
    for i, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        outputs = model(images)
        
        # Compute loss
        loss, cls, box, giou = compute_loss(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        cls_loss += cls.item()
        box_loss += box.item()
        giou_loss += giou.item()
        
        if i % 10 == 0:
            print(f"Epoch {epoch}, Batch {i}/{len(dataloader)}: "
                  f"Total Loss: {loss.item():.4f}, "
                  f"Cls Loss: {cls.item():.4f}, "
                  f"Box Loss: {box.item():.4f}, "
                  f"GIoU Loss: {giou.item():.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_cls = cls_loss / len(dataloader)
    avg_box = box_loss / len(dataloader)
    avg_giou = giou_loss / len(dataloader)
    
    return avg_loss, avg_cls, avg_box, avg_giou

def validate(model, dataloader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(images)
            loss, _, _, _ = compute_loss(outputs, targets)
            val_loss += loss.item()
    
    return val_loss / len(dataloader)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Data loaders
    train_loader, val_loader, _, classes = get_data_loaders(
        args.data_dir, 
        batch_size=args.batch_size
    )
    num_classes = len(classes)
    print(f"Dataset contains {num_classes} classes")
    
    # Model
    model = RFDetr(num_classes).to(device)
    
    # Optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    train_log = []
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_cls, train_box, train_giou = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        # Logging
        epoch_time = time.time() - start_time
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_cls': train_cls,
            'train_box': train_box,
            'train_giou': train_giou,
            'val_loss': val_loss,
            'time': epoch_time
        }
        train_log.append(log_entry)
        
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            ckpt_path = os.path.join(args.save_dir, f"rfdetr_epoch{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "rfdetr_best.pth")
            torch.save(model.state_dict(), best_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Save final model and log
    torch.save(model.state_dict(), os.path.join(args.save_dir, "rfdetr_final.pth"))
    log_path = os.path.join(args.log_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    with open(log_path, 'w') as f:
        json.dump(train_log, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF-DETR Model')
    parser.add_argument('--data_dir', type=str, default='data/', help='Dataset directory')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--step_size', type=int, default=30, help='LR scheduler step size')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--save_dir', type=str, default='saved_models/', help='Model save directory')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Training log directory')
    parser.add_argument('--save_interval', type=int, default=5, help='Model save interval (epochs)')
    
    args = parser.parse_args()
    main(args)