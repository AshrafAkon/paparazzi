import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from papzi.constants import (
    BATCH_SIZE,
    BASE_DIR,
    learning_rate,
    train_dir,
    val_dir,
    num_epochs,
)
from papzi.image_dataset import ImageDataset
from papzi.models import ResNet50
from papzi.utils import torch_device
from torch.utils.tensorboard.writer import SummaryWriter


class Trainer:
    def __init__(self, writer: SummaryWriter, train_num_images=200) -> None:
        self.writer = writer

        self.train_dataset = ImageDataset(
            train_dir, num_images=train_num_images
        )

        with open(BASE_DIR / "classes.json", "w") as f:
            json.dump(self.train_dataset.label_map, f)
        print("Saved classes")

        self.val_dataset = ImageDataset(val_dir, num_images=30)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=BATCH_SIZE,
            num_workers=4,
            shuffle=False,
        )

        self.device = torch_device()

        self.model = ResNet50(num_classes=self.train_dataset.classes).to(
            self.device
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.tq_postfix = {"val": "0.0", "loss": "0.0"}
        self.previous_epoch = 0

    def train(self, tq: tqdm, epoch: int) -> float:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_loss = 0.0
        for i, (images, labels) in enumerate(self.train_loader):

            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Log running loss and accuracy to TensorBoard every 100 batches
            if (i + 1) % 10 == 0 or (i + 1) == len(self.train_loader):
                writer_index = epoch * len(self.train_loader) + i
                self.writer.add_scalar(
                    "training/loss",
                    running_loss
                    / (
                        10
                        if (i + 1) % 10 == 0
                        else len(self.train_loader) - i + 1
                    ),
                    writer_index,
                )
                self.writer.add_scalar(
                    "training/accuracy",
                    100 * correct / total,
                    writer_index,
                )
                self.writer.add_scalar(
                    "Learning Rate",
                    self.optimizer.param_groups[0]["lr"],
                    writer_index,
                )

                tqdm.write(
                    f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(self.train_loader)}], "  # noqa: E501
                    + f"Loss: {running_loss / 100:.4f}, Accuracy: {100 * correct / total:.2f}%"  # noqa: E501
                )
                running_loss = 0.0
            tq.update()

        epoch_loss = epoch_loss / len(self.train_loader)
        epoch_accuracy = 100 * correct / total

        self.tq_postfix["loss"] = "%.2f" % epoch_loss
        self.writer.add_scalar("epoch/loss", epoch_loss, epoch + 1)
        self.writer.add_scalar("epoch/accuracy", epoch_accuracy, epoch + 1)
        return epoch_loss

    def evaluate(self, tq: tqdm) -> float:
        tq.set_description("validating")
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tq.update()
        val_acc = 100 * correct / total
        self.tq_postfix["val"] = "%.2f" % val_acc
        return val_acc

    def save(self, MODEL_PATH: Path, epoch: int):
        # Save the trained model and optimizer state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            },
            MODEL_PATH,
        )

    def load(self, MODEL_PATH: Path):
        # Load the trained model and optimizer state
        checkpoint = torch.load(MODEL_PATH)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.previous_epoch = checkpoint.get("epoch", 0)
