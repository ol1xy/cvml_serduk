
from train import Decoder, Encoder, ImageDataset
import torch
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(12, 16))

axes[0, 0].set_title("orig", fontsize=14)
axes[0, 1].set_title("result", fontsize=14)
axes[0, 2].set_title("delta", fontsize=14)

for mode in range(1, 5):
    row = mode - 1 
    
    encoder = Encoder()
    decoder = Decoder()

    try:
        encoder.load_state_dict(torch.load(f"encoder_mode_{mode}.pth"))
        decoder.load_state_dict(torch.load(f"decoder_mode_{mode}.pth"))

    except FileNotFoundError:
        print(f"not found lol")
        continue

    encoder.eval()
    decoder.eval()

    dataset = ImageDataset(10, 256, mode=mode)
    image, _  = dataset[0]

    with torch.no_grad():
        latent  = encoder(image.unsqueeze(0))
        result = decoder(latent)


    axes[row, 0].imshow(image.squeeze().numpy())
    axes[row, 0].set_ylabel(f"mode {mode}", fontsize=16)
    
    axes[row, 1].imshow(result.squeeze().numpy())
    
    diff = image.squeeze() - result.squeeze()
    axes[row, 2].imshow(diff.numpy())

plt.tight_layout()
plt.show()