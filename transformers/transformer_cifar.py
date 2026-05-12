import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch_directml

class ViT(nn.Module):
    def __init__(self, image_size = 32, patch_size = 4, channels = 3,
                 num_classes = 10, embed_size=192, depth = 6,
                 num_heads = 3, mlp_coeff = 4, drop_rate = 0.1):
        super().__init__()
        assert image_size % patch_size == 0, "wrong patch size"
        num_patches = (image_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(channels, embed_size,
                                     patch_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_size))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, 
                                                  embed_size))
        self.pos_drop = nn.Dropout()

        encoder = nn.TransformerEncoderLayer(d_model=embed_size,
                                             nhead = num_heads, 
                                             dim_feedforward=int(embed_size * mlp_coeff),
                                             dropout=drop_rate,
                                             activation='gelu'
                                             )
        
        self.blocks = nn.TransformerEncoder(encoder, num_layers=depth)
        self.norm = nn.LayerNorm(embed_size)
        self.head = nn.Sequential(
            nn.Linear(embed_size, num_classes)
        )



if __name__ == "__main__":
    device = torch_directml.device() 

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2435, 0.2616))

    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.247, 0.2435, 0.2616))
    ])

    train_ds = datasets.CIFAR10("./", train=True,
                            download=True, transform = train_transforms)   
    test_ds = datasets.CIFAR10("./", train=False,
                            download=True, transform = test_transforms)

    train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=True)

    model = ViT()
    print(sum(p.numel() for p in model.parameters()))