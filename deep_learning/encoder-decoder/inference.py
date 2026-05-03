from train import (Decoder, Encoder,
                   ImageDataset)
import torch
import matplotlib.pyplot as plt

encoder = Encoder()
decoder = Decoder()

encoder.load_state_dict(torch.load("encoder.pth"))
decoder.load_state_dict(torch.load("decoder.pth"))

with torch.no_grad():
    dataset = ImageDataset(10, 256)
    image, _  = dataset[0]

    latent  = encoder(image.unsqueeze(0))
    result = decoder(latent)

    plt.subplot(131)
    plt.imshow(image.squeeze())
    plt.subplot(132)
    plt.imshow(image.squeeze().detach().numpy())
    plt.subplot(133)
    plt.imshow(image.squeeze() - result.squeeze())
    plt.show()