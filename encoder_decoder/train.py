import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import torch_directml

class ImageDataset(Dataset):

    def __init__(self, n=200, size=128, mode=1):
        super().__init__()
        self.n = n
        self.size = size
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        image = Image.new('L', (self.size, self.size),
                          color = 255)
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()

        text = "ABD"
        x = 30
        y = 30

        if self.mode == 1:
            x = np.random.randint(10, self.size - 40)
            y = np.random.randint(10, self.size - 20)

        elif self.mode == 2:
            text = ''.join(np.random.choices(np.string.ascii_uppercase, k = 3))

        elif self.mode == 3:
            length = np.randint(1, 9)
            text = ''.join(np.random.choices(np.string.ascii_uppercase, k = length))

        elif self.mode == 4:
            length = np.randint(1, 9)
            x = np.random.randint(10, self.size - (10 * length) - 10)
            y = np.random.randint(10, self.size - 20)
            text = ''.join(np.random.choices(np.string.ascii_uppercase, k = length))  

        draw.text((x, y), text, fill = 0, font = font)

        tensor = self.transform(image)

        return tensor, tensor
    
ds = ImageDataset(2000, 256)
print(ds[0][0].shape)

plt.imshow(ds[0][0][0])
plt.show()

class Encoder(nn.Module):
    def __init__(self, latent=512):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, stride=2, kernel_size = 4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, stride=2, kernel_size = 4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, stride=2, kernel_size = 4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, stride=2, kernel_size = 4, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.bottleneck = nn.Linear(256 * 16 * 16, latent)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bottleneck(x)
        return x
        
class Decoder(nn.Module):

    def __init__(self, latent_size = 512):
        super().__init__()
        self.bottleneck = nn.Linear(latent_size, 256 * 16 * 16)
        self.features = nn.Sequential(
            nn.ConvTranspose2d(256, 128, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=4, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 1, stride=2, kernel_size=4, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.bottleneck(x)
        x = x.view(x.size(0), 256, 16, 16)
        x = self.features(x)
        return x

if __name__ == "__main__":
   
    device = torch_directml.device()
    epochs = 10
    losses = {}

    for cur_mode in range(1, 5):
        print(f"training model in {cur_mode} mode")
        encoder = Encoder()
        decoder = Decoder()


        dataset = ImageDataset(2000, 256, mode=cur_mode)
        dataloader = DataLoader(dataset, batch_size=16,
                                shuffle = True)

        encoder.to(device)
        decoder.to(device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(list(encoder.parameters()) + 
                                    list(decoder.parameters()))

        encoder.train()
        decoder.train()

        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for imgs, _ in dataloader:
                imgs = imgs.to(device)
                optimizer.zero_grad()
                latent = encoder(imgs)
                output = decoder(latent)
                loss = criterion(imgs, output)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(dataloader)
            print(f"{cur_mode=}, {epoch=}, {avg_loss=:.2f}")
        losses[cur_mode] = avg_loss

        torch.save(encoder.state_dict(), f"encoder_mode_{cur_mode}.pth")
        torch.save(decoder.state_dict(), f"decoder_mode_{cur_mode}.pth")

    print(losses)


