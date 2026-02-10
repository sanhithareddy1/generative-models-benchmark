
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
from torchvision.models.inception import inception_v3
import os
import pandas as pd
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.transforms import Resize, Normalize, Compose
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

# Generator Model
# ✅ Define Generator and Discriminator (DCGAN)
class Generator(nn.Module):
    def __init__(self, z_dim=100, img_channels=1, feature_g=64):
        super().__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g * 4, 7, 1, 0, bias=False),  # 1x1 -> 7x7
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),  # 7x7 -> 14x14
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g * 2, img_channels, 4, 2, 1, bias=False),   # 14x14 -> 28x28
            nn.Tanh()
        )
    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels=1, feature_d=64):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d, feature_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_d * 2, 1, 7, 1, 0, bias=False)  # <- no Sigmoid
        )
    def forward(self, x):
        return self.disc(x).view(-1)


# Weight Initialization
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# FID Score Calculation (naive version using InceptionV3 features)
from scipy.linalg import sqrtm
import numpy as np

@torch.no_grad()
def calculate_fid(fake_images, real_images, device):
    model = inception_v3(pretrained=True, transform_input=False).to(device).eval()
    model.fc = nn.Identity()
    model.Mixed_7c = nn.Identity()

    def get_features(imgs):
        if imgs.shape[1] == 1:
            imgs = imgs.expand(-1, 3, -1, -1)
        imgs = Resize((299, 299))(imgs)
        return model(imgs).cpu().numpy()

    f1 = get_features(real_images)
    f2 = get_features(fake_images)

    mu1, mu2 = f1.mean(axis=0), f2.mean(axis=0)
    sigma1, sigma2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)

    # Matrix square root
    covmean = sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # Discard imaginary part due to numerical errors

    fid = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)
  

# Training Loop
def train_gan(generator, discriminator, dataloader, device, z_dim, lr, num_epochs, save_path):
    criterion = nn.BCEWithLogitsLoss()
    opt_gen  = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_disc = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))


    generator.train()
    discriminator.train()

    os.makedirs("outputs/logs", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    loss_log = []
    fid_log = []

    for epoch in range(num_epochs):
        for batch_idx, (real, _) in enumerate(dataloader):
            real = real.to(device)
            batch_size = real.size(0)

            
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake = generator(noise)   # shape (N,1,28,28)


            # Train Discriminator
            disc_real = discriminator(real).view(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = discriminator(fake.detach()).view(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2

            discriminator.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            # Train Generator
            output = discriminator(fake).view(-1)
            loss_gen = criterion(output, torch.ones_like(output))

            generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

        print(f"Epoch [{epoch+1}/{num_epochs}]  Loss D: {loss_disc.item():.4f}, loss G: {loss_gen.item():.4f}")

        # Save sample images
        with torch.no_grad():
            fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)
            fake_images = generator(fixed_noise)                    # (64,1,28,28)
            save_image(fake_images, os.path.join(save_path, f"epoch_{epoch+1:03d}.png"), normalize=True)
            
            real_images = real[:64]                                 # (64,1,28,28)
            fid = calculate_fid(fake_images, real_images, device)
            fid_log.append({"epoch": epoch + 1, "fid": fid})


        loss_log.append({
            "epoch": epoch + 1,
            "gen_loss": loss_gen.item(),
            "disc_loss": loss_disc.item()
        })

    # Save logs to CSV
    pd.DataFrame(loss_log).to_csv("outputs/logs/gan_loss.csv", index=False)
    pd.DataFrame(fid_log).to_csv("outputs/logs/fid_scores.csv", index=False)
