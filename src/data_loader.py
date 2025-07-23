import os
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.v2 as T
from PIL import Image
import json

class CocoDetection(Dataset):
    def __init__(self, root, ann_file, transforms=None):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        self.classes = {k: v['name'] for k, v in self.coco.cats.items()}

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        img_path = os.path.join(self.root, self.coco.loadImgs(img_id)[0]['file_name'])
        img = Image.open(img_path).convert('RGB')

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to [x_min, y_min, x_max, y_max]
            labels.append(ann['category_id'])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([img_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id
        }

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)

def get_transform(train):
    transforms = []
    transforms.append(T.Resize((800, 800)))
    transforms.append(T.ToImage())
    transforms.append(T.ToDtype(torch.float32, scale=True))
    
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotation(15))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225]))
    
    return T.Compose(transforms)

def collate_fn(batch):
    return tuple(zip(*batch))

def get_data_loaders(data_dir, batch_size=4):
    train_dataset = CocoDetection(
        root=os.path.join(data_dir, 'train', 'images'),
        ann_file=os.path.join(data_dir, 'train', '_annotations.coco.json'),
        transforms=get_transform(train=True)
    )
    
    val_dataset = CocoDetection(
        root=os.path.join(data_dir, 'val', 'images'),
        ann_file=os.path.join(data_dir, 'val', '_annotations.coco.json'),
        transforms=get_transform(train=False)
    )
    
    test_dataset = CocoDetection(
        root=os.path.join(data_dir, 'test', 'images'),
        ann_file=os.path.join(data_dir, 'test', '_annotations.coco.json'),
        transforms=get_transform(train=False)
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader, train_dataset.classes