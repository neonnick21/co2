import json
import random
from pathlib import Path
import requests
import zipfile
import io

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def download_and_extract_dataset(url: str, dest_path: Path):
    """
    Downloads a zip file from a URL and extracts it to a destination path.
    Skips the process if the destination directory already exists.
    """
    if dest_path.exists():
        print(f"Dataset already found at '{dest_path}'. Skipping download.")
        return

    print(f"Downloading dataset from Roboflow...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download file: {e}") from e

    print("Download complete. Extracting files...")
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(dest_path)
    print(f"Dataset successfully extracted to '{dest_path}'.")


def get_transform(train: bool):
    """
    Defines the transformations to be applied to the images and targets.
    - For training, it includes data augmentation (horizontal flip).
    - For validation/testing, it only includes type conversion.
    """
    transforms = []

    # The v2 transforms expect a PIL image, and convert it to a v2.Image tensor.
    transforms.append(T.ToImage())

    if train:
        # This now operates on the v2.Image tensor and its corresponding bounding boxes.
        # The README mentions the original dataset was augmented with flips and rotations.
        # Adding them here can further improve model generalization.
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomVerticalFlip(0.5))
        transforms.append(T.RandomRotation(degrees=90))

    # Sanitize bounding boxes to ensure they are valid after transformations.
    # This can remove boxes that are too small or fall outside the image, which is
    # a good practice when applying geometric augmentations.
    transforms.append(T.SanitizeBoundingBoxes())

    # ToDtype converts the image dtype and scales the values.
    # If scale=True, the output is a plain torch.Tensor with values in [0.0, 1.0].
    transforms.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transforms)


class BccdDataset(Dataset):
    """
    A PyTorch Dataset for the BCCD dataset in COCO format.
    This class handles both data preparation for the model and visual exploration.
    """

    def __init__(self, root_dir: str, annotation_file: str, transforms=None):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotation_file (str): Path to the COCO format JSON annotation file.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = Path(root_dir)
        self.transforms = transforms

        print(f"Loading annotations from: {annotation_file}")
        with open(annotation_file, 'r') as f:
            coco_data = json.load(f)

        # Create helpful mappings for quick lookups
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.img_id_to_annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.img_id_to_annotations:
                self.img_id_to_annotations[img_id] = []
            self.img_id_to_annotations[img_id].append(ann)

        # We only want to work with images that have annotations
        annotated_image_ids = set(self.img_id_to_annotations.keys())
        self.images = [img for img in coco_data['images'] if img['id'] in annotated_image_ids]

        print(f"Found {len(self.images)} images with annotations.")
        print(f"Categories: {self.cat_id_to_name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 1. Get image information and load the image
        image_info = self.images[idx]
        image_path = self.root_dir / image_info['file_name']
        image = Image.open(image_path).convert("RGB")
        
        # 2. Get all annotations for this image
        image_id = image_info['id']
        annotations = self.img_id_to_annotations.get(image_id, [])

        # 3. Extract bounding boxes and labels
        boxes = [ann['bbox'] for ann in annotations]
        labels = [ann['category_id'] for ann in annotations]

        # Convert to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # The COCO format is [x, y, width, height].
        # We need to convert it to [x_min, y_min, x_max, y_max] for torchvision.
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        # The v2 transforms require bounding boxes to be in a specific format
        # to automatically apply transformations. We wrap the tensor in
        # tv_tensors.BoundingBoxes and provide the canvas size.
        boxes = tv_tensors.BoundingBoxes(
            boxes,
            format="XYXY",
            canvas_size=image.size[::-1]  # (height, width)
        )

        # 4. Construct the target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id])
        }

        # 5. Apply transformations
        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

    def visualize_random_sample(self):
        """
        Selects a random raw image from the dataset and displays it with its
        bounding box annotations for verification.
        """
        random_image_info = random.choice(self.images)
        image_id = random_image_info['id']
        file_name = random_image_info['file_name']
        image_path = self.root_dir / file_name

        print(f"\n--- Visualizing Random Sample ---")
        print(f"Displaying: {file_name} (Image ID: {image_id})")

        image = Image.open(image_path).convert("RGB")
        image_annotations = self.img_id_to_annotations.get(image_id, [])

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        ax.axis('off')

        for ann in image_annotations:
            bbox = ann['bbox']  # COCO format: [x_min, y_min, width, height]
            category_id = ann['category_id']
            category_name = self.cat_id_to_name[category_id]

            rect = patches.Rectangle(
                (bbox[0], bbox[1]), bbox[2], bbox[3],
                linewidth=2, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                bbox[0], bbox[1] - 10, category_name,
                color='black', backgroundcolor='cyan', fontsize=10
            )
        
        print("---------------------------------")
        plt.show()


if __name__ == '__main__':
    # --- Configuration & Automated Setup ---
    DATASET_URL = "https://public.roboflow.com/ds/GVJCultPuQ?key=0AVhhCEQpy"
    DATASET_BASE_DIR = Path("BCCD.v3-raw.coco")

    # Step 1: Automatically download and extract the dataset if it doesn't exist.
    download_and_extract_dataset(url=DATASET_URL, dest_path=DATASET_BASE_DIR)

    # --- Path Definitions ---
    TRAIN_DATA_ROOT = DATASET_BASE_DIR / "train"
    TRAIN_ANNOTATION_FILE = TRAIN_DATA_ROOT / "_annotations.coco.json"

    # --- Verification Step ---
    if not TRAIN_ANNOTATION_FILE.exists():
        error_msg = (
            f"ERROR: Annotation file not found at '{TRAIN_ANNOTATION_FILE.resolve()}'\n\n"
            "This can happen if the downloaded zip file has an unexpected structure.\n"
            "Please check the contents of the 'BCCD.v3-raw.coco' directory."
        )
        raise FileNotFoundError(error_msg)

    # Step 2: Test the data preparation pipeline for the model
    print(">>> Testing data preparation for PyTorch...")
    dataset_for_model = BccdDataset(
        root_dir=TRAIN_DATA_ROOT,
        annotation_file=TRAIN_ANNOTATION_FILE,
        transforms=get_transform(train=True)
    )
    print(f"Dataset size: {len(dataset_for_model)}")
    image, target = dataset_for_model[0]
    print("\n--- Processed Sample ---")
    print(f"Image shape: {image.shape}, type: {image.dtype}")
    print("Target dict:", {k: v.shape for k, v in target.items()})
    print("--------------------------\n")

    # Step 3: Test the visual exploration functionality
    print(">>> Testing visual exploration...")
    # We create a new instance without transforms to see the original image
    dataset_for_viz = BccdDataset(root_dir=TRAIN_DATA_ROOT, annotation_file=TRAIN_ANNOTATION_FILE)
    dataset_for_viz.visualize_random_sample()