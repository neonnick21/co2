import torch
from data_loader import get_data_loaders
from model import RFDetr
import argparse
import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import box_iou

def calculate_iou(pred_boxes, true_boxes):
    return box_iou(pred_boxes, true_boxes)

def calculate_metrics(predictions, ground_truth, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for img_preds, img_gts in zip(predictions, ground_truth):
        pred_boxes = img_preds['boxes']
        pred_labels = img_preds['labels']
        pred_scores = img_preds['scores']
        
        gt_boxes = img_gts['boxes']
        gt_labels = img_gts['labels']
        
        # Match predictions to ground truth
        matched = np.zeros(len(gt_boxes))
        for i, (pred_box, pred_label, pred_score) in enumerate(zip(
            pred_boxes, pred_labels, pred_scores
        )):
            best_iou = 0
            best_idx = -1
            
            for j, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                iou = calculate_iou(
                    pred_box.unsqueeze(0), 
                    gt_box.unsqueeze(0)
                ).item()
                
                if iou > best_iou and gt_label == pred_label:
                    best_iou = iou
                    best_idx = j
            
            if best_iou >= iou_threshold and not matched[best_idx]:
                true_positives += 1
                matched[best_idx] = 1
            else:
                false_positives += 1
        
        false_negatives += (len(gt_boxes) - np.sum(matched))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def evaluate_coco(model, dataloader, device, class_names):
    model.eval()
    results = []
    coco_predictions = []
    
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        
        with torch.no_grad():
            outputs = model(images)
        
        for i, (output, target) in enumerate(zip(outputs, targets)):
            # Process predictions
            logits = output['logits'].softmax(-1)
            scores = logits.max(-1).values
            labels = logits.argmax(-1)
            boxes = output['pred_boxes']
            
            # Filter out background predictions
            keep = labels < len(class_names)  # Remove "no object" class
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            
            # Convert to COCO format
            image_id = target['image_id'].item()
            for score, label, box in zip(scores, labels, boxes):
                x, y, w, h = box[0], box[1], box[2]-box[0], box[3]-box[1]  # xywh
                coco_pred = {
                    'image_id': image_id,
                    'category_id': label.item() + 1,  # COCO starts at 1
                    'bbox': [x.item(), y.item(), w.item(), h.item()],
                    'score': score.item()
                }
                coco_predictions.append(coco_pred)
    
    # Save predictions to JSON
    with open('predictions.json', 'w') as f:
        json.dump(coco_predictions, f)
    
    # Load ground truth
    coco_gt = COCO(os.path.join(args.data_dir, 'test', '_annotations.coco.json'))
    coco_dt = coco_gt.loadRes('predictions.json')
    
    # Run COCO evaluation
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    return coco_eval.stats[0]  # mAP@0.5:0.95

def visualize_predictions(model, dataloader, device, class_names, output_dir, num_images=5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    for i, (images, targets) in enumerate(dataloader):
        if i >= num_images:
            break
            
        images = [img.to(device) for img in images]
        image = images[0].permute(1, 2, 0).cpu().numpy()
        image = (image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
        image = image.astype(np.uint8)
        
        with torch.no_grad():
            output = model(images)[0]
        
        # Process output
        logits = output['logits'].softmax(-1)
        scores = logits.max(-1).values
        labels = logits.argmax(-1)
        boxes = output['pred_boxes']
        
        # Filter predictions
        keep = scores > 0.5
        scores = scores[keep]
        labels = labels[keep]
        boxes = boxes[keep]
        
        # Create plot
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        # Draw predictions
        for box, label, score in zip(boxes, labels, scores):
            if label >= len(class_names):
                continue  # Skip background
                
            x1, y1, x2, y2 = box.cpu().numpy()
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, 
                linewidth=2, edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)
            ax.text(
                x1, y1, 
                f"{class_names[label]}: {score:.2f}", 
                color='r', fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7)
            )
        
        # Save visualization
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'pred_{i}.jpg'), bbox_inches='tight')
        plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    _, _, test_loader, classes = get_data_loaders(
        args.data_dir, 
        batch_size=1
    )
    class_names = {i: name for i, name in enumerate(classes.values())}
    
    # Load model
    model = RFDetr(len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model_path))
    
    # Evaluate
    mAP = evaluate_coco(model, test_loader, device, class_names)
    print(f"mAP@0.5:0.95: {mAP:.4f}")
    
    # Visualize
    visualize_predictions(
        model, test_loader, device, class_names, 
        args.output_dir, num_images=10
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RF-DETR Model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--data_dir', type=str, default='data/', help='Dataset directory')
    parser.add_argument('--output_dir', type=str, default='results/predictions/', help='Output directory')
    
    args = parser.parse_args()
    main(args)