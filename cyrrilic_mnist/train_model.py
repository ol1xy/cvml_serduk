from torch.utils.data import Dataset, DataLoader
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class CyrrilicDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        self.labels = []

        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in
                             enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if os.path.isdir(cls_dir):
                for img_name in os.listdir(cls_dir):
                    self.image_paths.append(os.path.join(cls_dir,
                                                        img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert("L")

        if self.transform:
            image = self.transform(image)
        return image, label
    
augments = transforms.Compose([
    transforms.Resize((28, 28)),
    # transforms.RandomAffine(5, (0.1, 0.1), (0.5, 1), 10),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

class CyrillicCNN(nn.Module):
    def __init__(self, num_classes):
        super(CyrillicCNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x  = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(-1, 32 * 7 * 7)

        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    dataset = CyrrilicDataset(root_dir = "cyrillic/Cyrillic",
                              transform=augments)
    dataloader = DataLoader(dataset, batch_size = 32,
                            shuffle = True)
    num_classes = len(dataset.classes)
    model = CyrillicCNN(num_classes)

    loss_fn = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr = 0.001
        )
    
    loss_history = []

    epochs = 15
    for epoch in range(epochs):
        total_loss = 0.0

        for images, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"Epoch: {epoch+1}, loss: {avg_loss:.4f}")

    print("end")

    torch.save(model.state_dict(), "cyrillic_model.pth")

    plt.plot(loss_history, label="training loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("train.png")
