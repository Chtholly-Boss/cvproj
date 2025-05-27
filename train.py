import torch
from torch import Tensor
from torch.utils.data import DataLoader
import dataset
import net
from torchvision import transforms
import numpy as np
import os
from score import batch_pix_accuracy, batch_intersection_union

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = dataset.VocDataSet(split='val', transform=transform)
train_dataset = dataset.VocDataSet(split='train', transform=transform)

val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)

class Trainer:
    def __init__(self, name, model):
        # environment setting
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.model_path = f'model_{name}.pth'
        # training parameters
        self.lr = 1e-3
        self.epoch = 0
        # training modules
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,[30, 50, 100, 200, 300, 600])
        # Scores 
        self.pixacc = 0.0
        self.mIoU = 0.0
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.epoch = checkpoint['epoch']
            self.mIoU = checkpoint['mIoU']
            self.pixacc = checkpoint['pixacc']
            print(f"Resuming training from epoch {self.epoch}")
    def validate(self):
        self.model.eval()
        pixel_correct = 0
        pixel_labeled = 0
        total_inter = 0
        total_union = 0
        with torch.no_grad():
            for i, (images, targets, _) in enumerate(val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(images)
                correct, labeled = batch_pix_accuracy(outputs, targets)
                inter, union = batch_intersection_union(outputs, targets, 21)
                pixel_labeled += labeled
                pixel_correct += correct
                total_inter += inter
                total_union += union
        pixacc = pixel_correct / pixel_labeled
        iou = total_inter / (total_union + 1e-10)  # Avoid division by zero
        mIoU = iou.mean().item()
        if mIoU > self.mIoU:
            self.mIoU = mIoU
            print("\033[1;32mBetter IoU Get!\033[0m")
            torch.save({
                'epoch': self.epoch,
                'pixacc': self.pixacc,
                'mIoU': self.mIoU,
                'model_state_dict': model.state_dict(),
            }, self.model_path)
        print(f'Epoch: {self.epoch} PixAcc: {pixacc:.4f}, mIoU: {mIoU:.4f}')
        self.model.train()

    def train(self):
        self.model.train()
        while True:
            self.epoch += 1
            for batch, (images, targets, _) in enumerate(train_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)

                # forward
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if batch % 10 == 0:
                    print(f'Batch {batch}, Loss: {loss.item():.4f}')
            self.validate()
            self.lr_scheduler.step()

if __name__ == "__main__":
    model = net.UNet(n_channels=3, n_classes=21, bilinear=False)
    trainer = Trainer('unet', model)
    trainer.train()
