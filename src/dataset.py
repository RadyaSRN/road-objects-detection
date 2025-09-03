import os
import cv2
import torch

from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

CLASS_MAPPING = {
  0: 0,  # Car
  1: 3,  # Different-Traffic-Sign
  2: 4,  # Red-Traffic-Light
  3: 1,  # Pedestrian
  4: 3,  # Warning-Sign
  6: 4,  # Green-Traffic-Light
  7: 3,  # Prohibition-Sign
  8: 0,  # Truck
  9: 3,  # Speed-Limit-Sign
  10: 2  # Motorcycle
}


class ObjectionDetectionDataset(Dataset):
  def __init__(self, images_dir, labels_dir, transforms=None):
    self.images_dir = images_dir
    self.labels_dir = labels_dir
    self.transforms = transforms
    self.image_files = sorted(os.listdir(images_dir))

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    image_name = self.image_files[idx]
    image_path = os.path.join(self.images_dir, image_name)
    label_path = os.path.join(self.labels_dir, os.path.splitext(image_name)[0] + '.txt')

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    boxes = []
    labels = []
    with open(label_path) as f:
      for line in f:
        parts = line.strip().split()
        class_id = int(parts[0])
        if class_id not in CLASS_MAPPING:
          continue
        mapped_class = CLASS_MAPPING[class_id]
    
        x_center, y_center, box_w, box_h = map(float, parts[1:])
        xmin = (x_center - box_w / 2) * w
        ymin = (y_center - box_h / 2) * h
        xmax = (x_center + box_w / 2) * w
        ymax = (y_center + box_h / 2) * h
    
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(mapped_class)
    
    if self.transforms:
      transformed = self.transforms(image=image, bboxes=boxes, labels=labels)
      image = transformed['image']
      boxes = torch.tensor(transformed['bboxes'], dtype=torch.float32).reshape(-1, 4)
      labels = torch.tensor(transformed['labels'], dtype=torch.int64)
    else:
      image = ToTensorV2()(image=image)['image']
      boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
      labels = torch.tensor(labels, dtype=torch.int64)
    
    target = {'boxes': boxes, 'labels': labels}
    return image, target