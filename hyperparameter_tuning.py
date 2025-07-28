import torch
from pathlib import Path
from torch.utils.data import DataLoader
import torch.optim as optim
from model import RFDETR, compute_loss # Import RFDETR model and compute_loss function
from data_preprocessing import BccdDataset, get_transform, download_and_extract_dataset
from evaluation import evaluate, custom_collate_fn # Import evaluation functions
from tqdm import tqdm # For progress bars during training

# --- Configuration ---
DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")
TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"
VAL_DATA_ROOT = DATASET_BASE_DIR / "valid"
VAL_ANNOTATION_FILE = VAL_DATA_ROOT / "_annotations.coco.json"

# Define the hyperparameter grid to search over
# Each dictionary in the list represents a single experiment.
# You can add more combinations or adjust values here.
hyperparameter_grid = [
    # Baseline configuration (similar to your previous train.py setup)
    {'lr': 1e-4, 'batch_size': 8, 'num_queries': 100, 'num_epochs': 50},
    
    # Experiment 1: Higher learning rate
    {'lr': 5e-4, 'batch_size': 8, 'num_queries': 100, 'num_epochs': 50},
    
    # Experiment 2: Larger batch size (if GPU memory allows)
    {'lr': 1e-4, 'batch_size': 16, 'num_queries': 100, 'num_epochs': 50},
    
    # Experiment 3: Fewer queries (suitable for datasets with fewer objects per image)
    {'lr': 1e-4, 'batch_size': 8, 'num_queries': 50, 'num_epochs': 50},
    
    # Experiment 4: Combination of higher LR and larger batch size
    {'lr': 2e-4, 'batch_size': 16, 'num_queries': 100, 'num_epochs': 50},

    # Experiment 5: Even fewer queries
    {'lr': 1e-4, 'batch_size': 8, 'num_queries': 25, 'num_epochs': 50},

    # Experiment 6: Smaller learning rate
    {'lr': 5e-5, 'batch_size': 8, 'num_queries': 100, 'num_epochs': 50},
]

def run_training_and_evaluation(hparams, train_dataset, val_dataset, num_classes, device):
    """
    Runs a single training and evaluation experiment with the given hyperparameters.

    Args:
        hparams (dict): Dictionary of hyperparameters for the current run.
        train_dataset (Dataset): The training dataset.
        val_dataset (Dataset): The validation dataset.
        num_classes (int): Total number of classes including the 'no-object' class.
        device (torch.device): The device (CPU or CUDA) to run the experiment on.

    Returns:
        dict: A dictionary containing the evaluation metrics for this run.
        dict: The state_dict of the trained model, if it's the best so far for this run.
    """
    print(f"\n--- Running experiment with hyperparameters: {hparams} ---")
    
    # Create DataLoaders for current batch size
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=custom_collate_fn)

    # Initialize model with current hyperparameters
    model = RFDETR(num_classes=num_classes, num_queries=hparams['num_queries'])
    model.to(device) # Move model to device
    model.train()    # Set model to training mode

    # Initialize optimizer
    optimizer = optim.AdamW(model.parameters(), lr=hparams['lr'])

    # Training loop
    # Using tqdm for a progress bar for better visualization of long runs
    for epoch in tqdm(range(hparams['num_epochs']), desc=f"Training (LR: {hparams['lr']}, BS: {hparams['batch_size']}, Q: {hparams['num_queries']})"):
        total_loss = 0
        for images, targets in train_loader:
            optimizer.zero_grad() # Zero gradients before backward pass
            
            # Move data to the specified device
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass
            pred_logits, pred_boxes = model(images)
            
            # Compute loss
            loss = compute_loss(pred_logits, pred_boxes, targets, num_classes, device)
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            total_loss += loss.item() # Accumulate loss

        avg_loss = total_loss / len(train_loader)
        # print(f"Epoch [{epoch + 1}/{hparams['num_epochs']}], Loss: {avg_loss:.4f}") # Optional: Print epoch loss

    print("Training complete. Evaluating model...")
    
    # Evaluate the trained model on the validation set
    metrics = evaluate(model, val_loader, num_classes, device)
    
    # Print results for the current hyperparameter combination
    print(f"Evaluation Results for this run:")
    print(f"  mAP: {metrics['MulticlassMeanAveragePrecision/map']:.4f}")
    print(f"  mAP@0.50: {metrics['MulticlassMeanAveragePrecision/map_50']:.4f}")
    print(f"  mAP@0.75: {metrics['MulticlassMeanAveragePrecision/map_75']:.4f}")
    print(f"  Mean Precision: {metrics['MulticlassMeanAveragePrecision/macro_avg_precision']:.4f}")
    print(f"  Mean Recall: {metrics['MulticlassMeanAveragePrecision/macro_avg_recall']:.4f}")
    
    # Return metrics and the model's state_dict
    return metrics, model.state_dict()

if __name__ == '__main__':
    # --- Dataset Preparation (Download and Load) ---
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    # Verify annotation files exist
    if not TRAIN_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"Train annotation file not found at '{TRAIN_ANNOTATION_FILE.resolve()}'")
    if not VAL_ANNOTATION_FILE.exists():
        raise FileNotFoundError(f"Validation annotation file not found at '{VAL_ANNOTATION_FILE.resolve()}'")

    print(">>> Preparing datasets for tuning...")
    # Load training and validation datasets
    train_dataset = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    val_dataset = BccdDataset(
        root_dir=VAL_DATA_ROOT,
        annotation_file=VAL_ANNOTATION_FILE,
        transforms=get_transform(train=False)
    )
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Validation Dataset size: {len(val_dataset)}")

    # Determine the number of classes dynamically (+1 for 'no-object')
    num_classes = len(train_dataset.cat_id_to_name) + 1
    print(f"Number of classes (including no-object): {num_classes}")

    # Set up device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hyperparameter Tuning Loop ---
    all_results = []
    best_overall_map = -1.0 # Initialize with a low value
    best_overall_hparams = None
    best_overall_model_state = None

    for hparams in hyperparameter_grid:
        current_metrics, current_model_state = run_training_and_evaluation(hparams, train_dataset, val_dataset, num_classes, device)
        all_results.append({'hparams': hparams, 'metrics': current_metrics})

        # Check if current model is the best so far based on mAP
        current_map = current_metrics['MulticlassMeanAveragePrecision/map'].item()
        if current_map > best_overall_map:
            best_overall_map = current_map
            best_overall_hparams = hparams
            best_overall_model_state = current_model_state # Store the state_dict of the best model

    # --- Tuning Summary and Best Model Saving ---
    print("\n" + "="*40)
    print("--- Hyperparameter Tuning Complete ---")
    print("="*40)
    
    print("\nSummary of all experiment results:")
    for result in all_results:
        print(f"  HParams: {result['hparams']}")
        print(f"    mAP: {result['metrics']['MulticlassMeanAveragePrecision/map']:.4f}")
        print(f"    mAP@0.50: {result['metrics']['MulticlassMeanAveragePrecision/map_50']:.4f}")
        print(f"    mAP@0.75: {result['metrics']['MulticlassMeanAveragePrecision/map_75']:.4f}")
        print("-" * 30)

    # Save the best model found during the tuning process
    if best_overall_model_state:
        torch.save(best_overall_model_state, 'best_tuned_rfd_et-r_model.pth')
        print(f"\nBest model (mAP: {best_overall_map:.4f}) saved to 'best_tuned_rfd_et-r_model.pth'")
        print("Corresponding best hyperparameters:")
        for k, v in best_overall_hparams.items():
            print(f"  {k}: {v}")
    else:
        print("\nNo best model saved. This might indicate an issue with training or evaluation.")

