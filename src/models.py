import torch
import torch.optim
import pytorch_lightning as pl

import torchvision.models as models

from torchmetrics.detection.mean_ap import MeanAveragePrecision


class RetinaNetLightningModel(pl.LightningModule):
  """RetinaNet PyTorch Lightning wrapper"""
  def __init__(self, lr=1e-4, weight_decay=1e-2):
    super().__init__()
    weights_backbone = models.ResNet50_Weights.IMAGENET1K_V2
    self.model = models.detection.retinanet_resnet50_fpn_v2(weights_backbone=weights_backbone, num_classes=5)
    self.lr = lr
    self.weight_decay = weight_decay
    self.train_map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

  def forward(self, images, targets=None):
    return self.model(images, targets)
      
  def training_step(self, batch, batch_idx):
    images, targets = batch
    self.model.train()
    loss_dict = self(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    self.model.eval()
    with torch.no_grad():
      preds = self(images)
    self.train_map.update(preds, targets)

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    images, targets = batch
    self.model.train()
    loss_dict = self(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    self.model.eval()
    preds = self(images)
    self.val_map.update(preds, targets)

    self.log('val_loss', loss)
    return loss

  def on_train_epoch_end(self):
    metrics = self.train_map.compute()
    map_50 = metrics['map_50']
    self.log('train_map_50', map_50)
    self.train_map.reset()

  def on_validation_epoch_end(self):
    metrics = self.val_map.compute()
    map_50 = metrics['map_50']
    self.log('val_map_50', map_50)
    self.val_map.reset()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


class SSDLightningModel(pl.LightningModule):
  """SSD PyTorch Lightning wrapper"""
  def __init__(self, lr=1e-4, weight_decay=1e-2):
    super().__init__()
    weights_backbone = models.VGG16_Weights.IMAGENET1K_FEATURES
    self.model = models.detection.ssd300_vgg16(weights_backbone=weights_backbone, num_classes=5)
    self.lr = lr
    self.weight_decay = weight_decay
    self.train_map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    self.val_map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')

  def forward(self, images, targets=None):
    return self.model(images, targets)
      
  def training_step(self, batch, batch_idx):
    images, targets = batch
    self.model.train()
    loss_dict = self(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    self.model.eval()
    with torch.no_grad():
      preds = self(images)
    self.train_map.update(preds, targets)

    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    images, targets = batch
    self.model.train()
    loss_dict = self(images, targets)
    loss = sum(loss for loss in loss_dict.values())

    self.model.eval()
    preds = self(images)
    self.val_map.update(preds, targets)

    self.log('val_loss', loss)
    return loss

  def on_train_epoch_end(self):
    metrics = self.train_map.compute()
    map_50 = metrics['map_50']
    self.log('train_map_50', map_50)
    self.train_map.reset()

  def on_validation_epoch_end(self):
    metrics = self.val_map.compute()
    map_50 = metrics['map_50']
    self.log('val_map_50', map_50)
    self.val_map.reset()

  def configure_optimizers(self):
    return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)