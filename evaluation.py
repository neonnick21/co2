import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw, ImageFont
from model import RFDETR # Ensure RFDETR is importable from model.py
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics import MetricCollection # Import MetricCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches # Added this import

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets within a batch.
    It stacks images and keeps targets as a list of dictionaries.
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0]) # Image tensor
        targets.append(item[1]) # Target dictionary
    
    # Stack images into a single tensor for batch processing
    images = torch.stack(images)
    
    return images, targets

def post_process_predictions(pred_logits, pred_boxes, threshold=0.5):
    """
    Converts raw model outputs (logits and boxes) into a format suitable for
    torchmetrics, filtering predictions based on a confidence threshold.

    Args:
        pred_logits (torch.Tensor): Predicted class logits from the model.
                                    Shape: [batch_size, num_queries, num_classes]
        pred_boxes (torch.Tensor): Predicted bounding box coordinates (normalized).
                                   Shape: [batch_size, num_queries, 4]
        threshold (float): Confidence threshold to filter predictions.

    Returns:
        list[dict]: A list of dictionaries, one per image in the batch, where
                    each dictionary contains 'boxes', 'scores', and 'labels'
                    for the detected objects.
    """
    results = []
    
    for logits, boxes in zip(pred_logits, pred_boxes):
        # Apply softmax to get probabilities for each class
        probs = F.softmax(logits, dim=-1)
        # Get the highest score and corresponding label for each query,
        # excluding the 'no-object' class (assumed to be the last class)
        scores, labels = probs[:, :-1].max(-1) 

        # Filter out predictions below the confidence threshold
        keep = scores > threshold
        
        results.append({
            "boxes": boxes[keep],    # Bounding boxes for kept predictions
            "scores": scores[keep],  # Confidence scores for kept predictions
            "labels": labels[keep],  # Class labels for kept predictions
        })
    return results

def evaluate(model, data_loader, num_classes, device):
    """
    Evaluates the model's performance on a given data loader.

    Args:
        model (nn.Module): The RFDETR model to evaluate.
        data_loader (DataLoader): DataLoader for the validation/test dataset.
        num_classes (int): Total number of classes including the 'no-object' class.
        device (torch.device): The device (CPU or CUDA) to run evaluation on.

    Returns:
        dict: A dictionary containing computed metrics (mAP, mAP_50, mAP_75, etc.).
    """
    model.eval() # Set the model to evaluation mode
    
    # Initialize MeanAveragePrecision metric from torchmetrics
    metric = MetricCollection({ # Use MetricCollection for a single metric with class_metrics
        'map': MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)
    }).to(device)


    with torch.no_grad(): # Disable gradient calculations for inference
        for images, targets in data_loader:
            # Move images and targets to the specified device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get model predictions
            pred_logits, pred_boxes = model(images)
            
            # Post-process predictions into the format required by torchmetrics
            preds = post_process_predictions(pred_logits, pred_boxes)
            
            # Update the metric with the current batch's predictions and targets
            metric.update(preds, targets)
            
    # Compute the final metrics across all batches
    metrics_result = metric.compute()
    
    # Extract the map metric from the MetricCollection result
    map_metric = metrics_result['map']
    
    # Torchmetrics 0.11.0 and later returns a dict of tensors for mAP components
    # We want to return a flat dictionary for easier printing/access
    # Also, ensure 'map_per_class' and 'mar_100' are present for older versions of torchmetrics if needed
    # (though newer versions might structure these differently within the 'map' dict itself)
    
    # Flatten the map_metric dictionary for consistent access
    final_metrics = {
        'map': map_metric['map'].item(),
        'map_50': map_metric['map_50'].item(),
        'map_75': map_metric['map_75'].item(),
        'map_per_class': map_metric['map_per_class'].mean().item() if 'map_per_class' in map_metric else torch.tensor(0.0).item(), # Handle if not directly present or if it's a list
        'mar_100': map_metric['mar_100'].item() if 'mar_100' in map_metric else torch.tensor(0.0).item(),
        # Add other relevant metrics if needed
    }
    
    return final_metrics


def visualize_predictions(model, dataset, device, num_images=3, threshold=0.7):
    """
    Visualizes predictions on a few sample images from the dataset.

    Args:
        model (nn.Module): The RFDETR model.
        dataset (Dataset): The dataset to draw images from (e.g., validation set).
        device (torch.device): The device (CPU or CUDA) where the model is.
        num_images (int): Number of random images to visualize.
        threshold (float): Confidence threshold for displaying predicted boxes.
    """
    model.eval() # Set model to evaluation mode
    
    plt.figure(figsize=(15, 5 * num_images))
    with torch.no_grad(): # Disable gradient calculations
        # Select random indices for visualization
        # Ensure that random_indices is a list of Python integers
        random_indices = torch.randint(0, len(dataset), (num_images,)).tolist()

        for i, idx in enumerate(random_indices):
            # Get image and target from the dataset
            image, target = dataset[idx] # image is already a torch.Tensor, [C, H, W] float.
            
            # Prepare image for model input: add batch dimension and move to device
            image_tensor = image.unsqueeze(0).to(device)

            # Get predictions
            pred_logits, pred_boxes = model(image_tensor)
            # Post-process predictions for the single image (batch size 1)
            # [0] to get the dictionary for the first (and only) image in the batch
            preds = post_process_predictions(pred_logits, pred_boxes, threshold=threshold)[0]
            
            # Denormalize bounding box coordinates to pixel values
            _, h, w = image.shape # Get height and width from the image tensor
            
            # Ensure target['boxes'] is on the same device as the multiplier
            # and then move all to CPU for plotting
            preds_boxes_denorm = (preds['boxes'] * torch.tensor([w, h, w, h], device=device)).cpu().numpy()
            target_boxes_denorm = (target['boxes'].to(device) * torch.tensor([w, h, w, h], device=device)).cpu().numpy()

            # Convert image tensor to PIL Image for drawing
            original_image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8'))
            draw = ImageDraw.Draw(original_image)
            
            # Get class names for displaying labels
            category_names = dataset.cat_id_to_name # Corrected attribute name

            # Draw ground truth bounding boxes in green
            for j in range(len(target_boxes_denorm)): # Use target_boxes_denorm_cpu for iteration
                box = target_boxes_denorm[j].tolist() # Convert numpy array to list
                label = target['labels'][j].item() # Use original target labels, then .item()
                
                label_name = category_names.get(label, f"Unknown_{label}")
                
                # Bbox format for PIL.ImageDraw.rectangle is (x_min, y_min, x_max, y_max)
                draw.rectangle(box, outline="green", width=2)
                
                try: # Basic font handling, if font not found, it will skip
                    font = ImageFont.truetype("arial.ttf", 15) # Example font
                except IOError:
                    font = ImageFont.load_default() # Fallback to default
                
                draw.text(
                    (box[0], box[1] - 15), # Slightly higher than the box
                    f"GT: {label_name}", 
                    fill="green",
                    font=font
                )

            # Draw predicted bounding boxes in red
            for j in range(len(preds_boxes_denorm)):
                box = preds_boxes_denorm[j].tolist()
                label = preds['labels'][j].item()
                score = preds['scores'][j].item()

                label_name = category_names.get(label, f"Unknown_{label}")

                draw.rectangle(box, outline="red", width=2)
                
                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except IOError:
                    font = ImageFont.load_default()

                draw.text(
                    (box[0], box[1] + 5), # Offset text slightly to avoid overlap with GT
                    f"Pred: {label_name} ({score:.2f})", 
                    fill="red",
                    font=font
                )
            
            # Display the image using matplotlib
            plt.subplot(num_images, 1, i + 1)
            plt.imshow(original_image)
            plt.title(f"Image {idx} - Predictions (red) vs Ground Truth (green)")
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# The main execution block will only run if this script is executed directly,
# preventing it from running when imported by hyperparameter_tuning.py
if __name__ == '__main__':
    # --- Configuration & Automated Setup ---
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    VAL_DATA_ROOT = DATASET_BASE_DIR / "valid"
    VAL_ANNOTATION_FILE = VAL_DATA_ROOT / "_annotations.coco.json"
    
    MODEL_PATH = Path("rfd_et-r_model.pth")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("A trained model file 'rfd_et-r_model.pth' not found. Please train the model first using train.py or hyperparameter_tuning.py.")

    # --- Dataset and DataLoader ---
    val_dataset = BccdDataset(
        root_dir=VAL_DATA_ROOT,
        annotation_file=VAL_ANNOTATION_FILE,
        transforms=get_transform(train=False)
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # --- Model and Device Setup ---
    num_classes = len(val_dataset.cat_id_to_name) + 1 # +1 for 'no-object' class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    model = RFDETR(num_classes=num_classes, num_queries=100)
    # Load model weights, mapping to the correct device
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device) # Move model to device

    # --- Run Evaluation ---
    print(">>> Starting evaluation on the validation set...")
    metrics = evaluate(model, val_loader, num_classes, device)

    print("\n--- Evaluation Results ---")
    # Access individual items for printing, now that evaluate returns a flattened dict
    print(f"Mean Average Precision (mAP): {metrics['map']:.4f}")
    print(f"mAP@0.50 IoU: {metrics['map_50']:.4f}")
    print(f"mAP@0.75 IoU: {metrics['map_75']:.4f}")
    print(f"Mean Precision: {metrics['map_per_class']:.4f}") # Now directly a scalar
    print(f"Mean Recall: {metrics['mar_100']:.4f}") # Now directly a scalar
    print("--------------------------")
    
    # --- Visualize Predictions ---
    print("\n>>> Visualizing predictions on sample images...")
    visualize_predictions(model, val_dataset, device, num_images=3, threshold=0.7) # Added threshold
