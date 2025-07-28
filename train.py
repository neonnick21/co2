import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from pathlib import Path
from model import RFDETR, nested_tensor_from_tensor_list
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset
import matplotlib.pyplot as plt
import os
import evaluation # This import is to reuse the dataset class from the same project

# Set a random seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized targets within a batch.
    It keeps images as a list of tensors and targets as a list of dictionaries.
    """
    images = []
    targets = []
    
    for item in batch:
        images.append(item[0]) # Image tensor
        targets.append(item[1]) # Target dictionary
    
    return images, targets

def train_model(model, train_loader, val_loader, optimizer, num_epochs, device, model_save_path):
    """
    Trains the RFDETR model for a specified number of epochs.
    """
    print(">>> Preparing dataset for training...")
    model.to(device)
    
    best_loss = float('inf')

    for epoch in range(num_epochs):
        model.train() # Set the model to training mode
        total_loss = 0
        for i, (images, targets) in enumerate(train_loader):
            # Convert the list of image tensors into a NestedTensor object
            images = nested_tensor_from_tensor_list(images)
            images = images.to(device)

            # Ensure targets are on the correct device
            targets_processed = []
            for t in targets:
                targets_processed.append({
                    "boxes": t['boxes'].to(device),
                    "labels": t['labels'].to(device)
                })

            optimizer.zero_grad()
            
            # The model returns a dictionary of losses.
            loss_dict = model(images, targets_processed)
            loss = sum(loss_dict[k] for k in loss_dict.keys())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
        
        # Save the model if the current loss is the best we've seen so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print("Model saved to", model_save_path)

    print("Training complete.")

# Main execution block
if __name__ == '__main__':
    # --- Configuration & Automated Setup ---
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
    
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)
    
    TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
    TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"
    VAL_DATA_ROOT = DATASET_BASE_DIR / "valid"
    VAL_ANNOTATION_FILE = VAL_DATA_ROOT / "_annotations.coco.json"

    # --- Dataset and DataLoader ---
    train_dataset = BccdDataset(
        root_dir=TRAIN_DATA_ROOT, 
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    val_dataset = evaluation.BccdDataset( # Re-using the dataset class from evaluation
        root_dir=VAL_DATA_ROOT,
        annotation_file=VAL_ANNOTATION_FILE,
        transforms=evaluation.get_transform(train=False)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # --- Model and Device Setup ---
    # The number of classes in the dataset, plus 1 for the 'no-object' class
    num_classes = len(train_dataset.cat_id_to_name) + 1 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the RFDETR model
    model = RFDETR(num_classes=num_classes, num_queries=100)
    
    # Load pre-trained ResNet-50 weights if available
    try:
        model.backbone.backbone.load_state_dict(torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True).state_dict(), strict=False)
        print("Loaded pre-trained ResNet-50 weights.")
    except Exception as e:
        print(f"Could not load pre-trained weights: {e}")

    # --- Optimizer and Training ---
    # We use AdamW optimizer which is generally a good choice for Transformer models
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # We now train for 100 epochs instead of 50
    NUM_EPOCHS = 100
    MODEL_SAVE_PATH = "rfd_et-r_model.pth"
    
    train_model(model, train_loader, val_loader, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH)
