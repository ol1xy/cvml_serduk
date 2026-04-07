import torch_directml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from multiprocessing import freeze_support
from torchvision.datasets import ImageFolder
import torchvision
import torch.nn as nn

train_transform = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3,
                           contrast=0.3,
                           saturation=0.2),
    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225],)
])

val_transform = transforms.Compose([
    transforms.Resize((240, 240)),

    transforms.ToTensor(), 
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225],)
])

train_ds = ImageFolder("dataset/dataset/train",
                       transform=train_transform)
val_ds = ImageFolder("dataset/dataset/val",
                       transform=val_transform)

train_loader = DataLoader(train_ds, 16,
                          shuffle=True,
                          num_workers=6)
val_loader = DataLoader(val_ds, 16,
                          shuffle=False,
                          num_workers=6)

def build_model():
    weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = torchvision.models.efficientnet_b0(weights=weights)
    for p in model.features.parameters():
        p.requires_grad = False
    
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 3)
    )
    return model 

model = build_model()

if __name__ == "__main__":
    print(train_ds.classes)
    trainable = sum(p.numel()
                    for p in model.parameters()
                    if p.requires_grad)
    print(trainable)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters())
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=10)
    
    def run(model, loader, criterion, 
            optimizer=None):
        trained = False

        if optimizer is not None:
            model.train()
            trained = True
        else:
            model.eval()
            trained = False
        total_loss, correct, total = 0, 0, 0

        with torch.set_grad_enabled(trained):
            for images, labels in loader:
                images = images
                label = labels
                logits = model(images)
                loss = criterion(logits, labels)
                if trained:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)
        return total_loss / total, correct / total
    
    best_acc = 0.0
    for epoch in range(0, 10):
        train_loss, train_acc = run(model,
                                    train_loader,
                                    criterion, 
                                    optimizer)
        val_loss, val_acc = run(model,
                                val_loader, 
                                criterion)
        scheduler.step()
        print(f"Epoch = {epoch}, {train_loss=}, {train_acc=}, {val_loss=}, {val_acc=}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "efnet.pth")