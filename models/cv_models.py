"""
Computer Vision models using PyTorch Lightning.
Includes image classification and object detection models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchvision import models
from torchmetrics import Accuracy, F1Score, Precision, Recall
import logging
from typing import Dict, Any, Optional, List
import numpy as np

from config import config

logger = logging.getLogger(__name__)


class ImageClassifier(pl.LightningModule):
    """PyTorch Lightning module for image classification"""

    def __init__(self, num_classes: int, backbone: str = "resnet18",
                 pretrained: bool = True, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Initialize backbone
        self.backbone = self._create_backbone(backbone, pretrained, num_classes)

        # Initialize metrics
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")
        self.val_precision = Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = Recall(task="multiclass", num_classes=num_classes, average="macro")

        logger.info(f"Initialized ImageClassifier with {backbone} backbone, {num_classes} classes")

    def _create_backbone(self, backbone: str, pretrained: bool, num_classes: int):
        """Create backbone model"""
        if backbone == "resnet18":
            model = models.resnet18(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == "resnet34":
            model = models.resnet34(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == "resnet50":
            model = models.resnet50(pretrained=pretrained)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif backbone == "efficientnet_b0":
            model = models.efficientnet_b0(pretrained=pretrained)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        elif backbone == "mobilenet_v3_small":
            model = models.mobilenet_v3_small(pretrained=pretrained)
            model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        return model

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy(preds, y)

        # Log metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy(preds, y)
        self.val_f1(preds, y)
        self.val_precision(preds, y)
        self.val_recall(preds, y)

        # Log metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy(preds, y)

        # Log metrics
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("test_accuracy", self.test_accuracy, on_step=False, on_epoch=True)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        probs = F.softmax(logits, dim=1)

        return {
            "predictions": preds,
            "probabilities": probs,
            "logits": logits
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_accuracy",
                "interval": "epoch",
                "frequency": 1,
            },
        }


class SimpleObjectDetector(pl.LightningModule):
    """Simplified object detection model using YOLO-style approach"""

    def __init__(self, num_classes: int, learning_rate: float = 1e-3,
                 weight_decay: float = 1e-4, confidence_threshold: float = 0.5):
        super().__init__()
        self.save_hyperparameters()

        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.confidence_threshold = confidence_threshold

        # Simple backbone for demonstration
        self.backbone = models.resnet18(pretrained=True)
        backbone_features = self.backbone.fc.in_features

        # Remove final classification layer
        self.backbone.fc = nn.Identity()

        # Detection head: predicts [x, y, w, h, confidence, class_probs...]
        self.detection_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(backbone_features * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 5 + num_classes)  # bbox (4) + confidence (1) + classes
        )

        logger.info(f"Initialized SimpleObjectDetector with {num_classes} classes")

    def forward(self, x):
        features = self.backbone(x)
        detections = self.detection_head(features)
        return detections

    def training_step(self, batch, batch_idx):
        # Note: This is a simplified implementation
        # Real object detection would require proper loss functions and target formatting
        x, targets = batch
        predictions = self(x)

        # Simplified loss - in practice, you'd use proper YOLO/FCOS loss
        loss = F.mse_loss(predictions, targets.float())

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, targets = batch
        predictions = self(x)

        loss = F.mse_loss(predictions, targets.float())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        return optimizer

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, _ = batch if isinstance(batch, (tuple, list)) else (batch, None)
        detections = self(x)

        # Parse detections (simplified)
        batch_size = detections.shape[0]
        results = []

        for i in range(batch_size):
            detection = detections[i]
            bbox = detection[:4]
            confidence = torch.sigmoid(detection[4])
            class_probs = F.softmax(detection[5:], dim=0)

            if confidence > self.confidence_threshold:
                class_id = torch.argmax(class_probs)
                results.append({
                    "bbox": bbox.cpu().numpy(),
                    "confidence": confidence.item(),
                    "class_id": class_id.item(),
                    "class_prob": class_probs[class_id].item()
                })
            else:
                results.append(None)

        return results


class CVModelFactory:
    """Factory class for creating computer vision models"""

    @staticmethod
    def create_model(task_type: str, num_classes: int, **kwargs) -> pl.LightningModule:
        """Create a computer vision model based on task type"""

        if task_type == "classification":
            return ImageClassifier(
                num_classes=num_classes,
                backbone=kwargs.get("backbone", config.model.cv_backbone),
                pretrained=kwargs.get("pretrained", config.model.cv_pretrained),
                learning_rate=kwargs.get("learning_rate", config.training.learning_rate),
                weight_decay=kwargs.get("weight_decay", config.training.weight_decay)
            )

        elif task_type == "object_detection":
            return SimpleObjectDetector(
                num_classes=num_classes,
                learning_rate=kwargs.get("learning_rate", config.training.learning_rate),
                weight_decay=kwargs.get("weight_decay", config.training.weight_decay),
                confidence_threshold=kwargs.get("confidence_threshold", 0.5)
            )

        else:
            raise ValueError(f"Unsupported CV task type: {task_type}")

    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get available models for each task type"""
        return {
            "classification": [
                "resnet18", "resnet34", "resnet50",
                "efficientnet_b0", "mobilenet_v3_small"
            ],
            "object_detection": [
                "simple_detector"  # In practice, you'd add YOLO, FCOS, etc.
            ]
        }

    @staticmethod
    def get_model_info(model_name: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        model_info = {
            "resnet18": {
                "parameters": 11.7e6,
                "description": "ResNet-18 architecture, good balance of speed and accuracy",
                "recommended_for": "Small to medium datasets"
            },
            "resnet34": {
                "parameters": 21.8e6,
                "description": "ResNet-34 architecture, more capacity than ResNet-18",
                "recommended_for": "Medium datasets"
            },
            "resnet50": {
                "parameters": 25.6e6,
                "description": "ResNet-50 architecture, higher capacity with bottleneck blocks",
                "recommended_for": "Large datasets"
            },
            "efficientnet_b0": {
                "parameters": 5.3e6,
                "description": "EfficientNet-B0, optimized for efficiency",
                "recommended_for": "Resource-constrained environments"
            },
            "mobilenet_v3_small": {
                "parameters": 2.5e6,
                "description": "MobileNet-V3 Small, very lightweight",
                "recommended_for": "Mobile/edge deployment"
            },
            "simple_detector": {
                "parameters": "Variable",
                "description": "Simplified object detection model for demonstration",
                "recommended_for": "Learning and prototyping"
            }
        }

        return model_info.get(model_name, {"description": "Unknown model"})


# Convenience function for quick model creation
def create_cv_model(task_type: str, num_classes: int, **kwargs) -> pl.LightningModule:
    """Create a computer vision model"""
    return CVModelFactory.create_model(task_type, num_classes, **kwargs)
