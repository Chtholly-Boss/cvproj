import torch
from torch import Tensor
from torch.utils.data import DataLoader
import dataset
import net
from torchvision import transforms
import numpy as np
import os

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_dataset = dataset.VocDataSet(split='val', transform=transform)
train_dataset = dataset.VocDataSet(split='train', transform=transform)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=1)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

lr = 1e-4

def intersectionAndUnion(imPred, imLab, numClass):
    """
    This function takes the prediction and label of a single image,
    returns intersection and union areas for each class
    To compute over many images do:
    for i in range(Nimages):
        (area_intersection[:,i], area_union[:,i]) = intersectionAndUnion(imPred[i], imLab[i])
    IoU = 1.0 * np.sum(area_intersection, axis=1) / np.sum(np.spacing(1)+area_union, axis=1)
    """
    # Remove classes from unlabeled pixels in gt image.
    # We should not penalize detections in unlabeled portions of the image.
    imPred = imPred * (imLab >= 0)

    # Compute area intersection:
    intersection = imPred * (imPred == imLab)
    (area_intersection, _) = np.histogram(intersection, bins=numClass, range=(1, numClass))

    # Compute area union:
    (area_pred, _) = np.histogram(imPred, bins=numClass, range=(1, numClass))
    (area_lab, _) = np.histogram(imLab, bins=numClass, range=(1, numClass))
    area_union = area_pred + area_lab - area_intersection
    return (area_intersection, area_union)

def batch_pix_accuracy(output, target):
    """PixAcc"""
    # inputs are numpy array, output 4D, target 3D
    predict = torch.argmax(output.long(), 1) + 1
    target = target.long() + 1

    pixel_labeled = torch.sum(target > 0).item()
    pixel_correct = torch.sum((predict == target) * (target > 0)).item()
    assert pixel_correct <= pixel_labeled, "Correct area should be smaller than Labeled"
    return pixel_correct, pixel_labeled

def batch_intersection_union(output, target, nclass):
    """mIoU"""
    # inputs are numpy array, output 4D, target 3D
    mini = 1
    maxi = nclass
    nbins = nclass
    predict = torch.argmax(output, 1) + 1
    target = target.float() + 1

    predict = predict.float() * (target > 0).float()
    intersection = predict * (predict == target).float()
    # areas of intersection and union
    # element 0 in intersection occur the main difference from np.bincount. set boundary to -1 is necessary.
    area_inter = torch.histc(intersection.cpu(), bins=nbins, min=mini, max=maxi)
    area_pred = torch.histc(predict.cpu(), bins=nbins, min=mini, max=maxi)
    area_lab = torch.histc(target.cpu(), bins=nbins, min=mini, max=maxi)
    area_union = area_pred + area_lab - area_inter
    assert torch.sum(area_inter > area_union).item() == 0, "Intersection area should be smaller than Union area"
    return area_inter.float(), area_union.float()

def validate(model):
    pixel_correct = 0
    pixel_labeled = 0
    total_inter = 0
    total_union = 0
    model.eval()
    with torch.no_grad():
        for i, (images, targets, filenames) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            correct, labeled = batch_pix_accuracy(outputs, targets)
            inter, union = batch_intersection_union(outputs, targets, 21)
            pixel_labeled += labeled
            pixel_correct += correct
            total_inter += inter
            total_union += union
        model.train()
    pixacc = pixel_correct / pixel_labeled
    iou = total_inter / (total_union + 1e-10)  # Avoid division by zero
    print(f'PixAcc: {pixacc:.4f}')
    print(f'mIoU: {iou.mean().item():.4f}')

def train(model):
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for batch, (images, targets, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 10 == 0:
            print(f'Batch {batch}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    # model = net.ConvNet(21)
    # model = net.get_net(21)
    model = net.FCN(21)
    name = 'fcn'
    model_path = f'model_{name}.pth'
    start_epoch = 0
    
    # Load model checkpoint if it exists
    if os.path.exists(model_path):
        print(f"Loading model checkpoint from {model_path}")
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, start_epoch + 10):
        train(model)
        validate(model)
    
    # Save the model after training
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
    }, model_path)
    
    print(f"Training complete. Model saved to {model_path}")
