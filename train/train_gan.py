
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

from models.gan import Generator, Discriminator, weights_init, train_gan


def main():
    # Hyperparameters
    z_dim = 100
    batch_size = 128
    lr = 2e-4
    num_epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create output directories
    os.makedirs("outputs/samples", exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model initialization
    generator = Generator(z_dim=z_dim, img_channels=1).to(device)
    discriminator = Discriminator(img_channels=1).to(device)

    generator.apply(weights_init)
    discriminator.apply(weights_init)

    # Train the GAN
    train_gan(generator, discriminator, dataloader, device, z_dim, lr, num_epochs, save_path="outputs/samples")

    # Save models
    torch.save(generator.state_dict(), "outputs/checkpoints/generator.pth")
    torch.save(discriminator.state_dict(), "outputs/checkpoints/discriminator.pth")

if __name__ == "__main__":
    main()
