import os
import random
import argparse
import shutil
import imageio
import textwrap
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
from PIL import Image, ImageDraw, ImageFont, ImageFilter

##############################################
# Global Paths and Hyperparameters
##############################################
ROOMS_OUT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\rooms"
os.makedirs(ROOMS_OUT_DIR, exist_ok=True)

# Dataset directories (images stored directly)
DOOR_DIR = r"D:\project\haunted_escape_room_backend\assets\door"
ATMOSPHERE_DIR = r"D:\project\haunted_escape_room_backend\assets\atmosphere"
GHOST_DIR = r"D:\project\haunted_escape_room_backend\assets\ghost"
HALLOWEAN_DIR = r"D:\project\haunted_escape_room_backend\assets\hallowean"

# Pre-generated text image directories
RIDDLE_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\riddle"
SUDOKU_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\sudoku"
VALID_GUESSES_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\valid_guesses"

# VAE model save paths
DOOR_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\door_vae.pth"
ATMOS_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\atmosphere_vae.pth"
GHOST_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\ghost_vae.pth"
HALLOWEAN_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\hallowean_vae.pth"

# Target resolution: 720p (1280×720)
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

# VAE/training hyperparameters
LATENT_DIM = 256
nc = 3
num_epochs = 800   # Maximum training epochs (or until avg loss < threshold)
BATCH_SIZE = 1     # Due to high resolution, use small batch size
LR = 0.0002
BETA_VAE = 4.0
MAX_EPOCHS = 2000  # Hard cap on epochs
LOSS_THRESHOLD = 1.1
POOL_SIZE = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##############################################
# 1. Custom SingleFolderDataset
##############################################
class SingleFolderDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.image_paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0

##############################################
# 2. Beta-VAE Model with 10-Layer Encoder/Decoder
##############################################
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, beta=BETA_VAE):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder: 10 layers (5 with stride 2 and 5 with stride 1)
        self.encoder = nn.Sequential(
            # Layer 1: stride 2
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (3,720,1280)->(64,360,640)
            nn.ReLU(True),
            # Layer 2: stride 1
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),   # (64,360,640)->(64,360,640)
            nn.ReLU(True),
            # Layer 3: stride 2
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # ->(128,180,320)
            nn.ReLU(True),
            # Layer 4: stride 1
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),  # ->(128,180,320)
            nn.ReLU(True),
            # Layer 5: stride 2
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # ->(256,90,160)
            nn.ReLU(True),
            # Layer 6: stride 1
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),  # ->(256,90,160)
            nn.ReLU(True),
            # Layer 7: stride 2
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # ->(512,45,80)
            nn.ReLU(True),
            # Layer 8: stride 1
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # ->(512,45,80)
            nn.ReLU(True),
            # Layer 9: stride 2
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # ->(512,22,40)  [*]
            nn.ReLU(True),
            # Layer 10: stride 1
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),  # ->(512,22,40)
            nn.ReLU(True)
        )
        # Note: The final output shape is (512,22,40) because:
        # 720->360->180->90->45->22, 1280->640->320->160->80->40
        self.flat_dim = 512 * 22 * 40  # 512*880 = 450560
        
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)
        
        # Decoder: 10 layers
        self.fc_decode = nn.Linear(latent_dim, self.flat_dim)
        self.decoder = nn.Sequential(
            # Layer 1: stride 1
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),  # (512,22,40)
            nn.ReLU(True),
            # Layer 2: stride 2
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # ->(512,44,80)
            nn.ReLU(True),
            # Layer 3: stride 1
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1),  # ->(512,44,80)
            nn.ReLU(True),
            # Layer 4: stride 2
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # ->(256,88,160)
            nn.ReLU(True),
            # Layer 5: stride 1
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1),  # ->(256,88,160)
            nn.ReLU(True),
            # Layer 6: stride 2
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # ->(128,176,320)
            nn.ReLU(True),
            # Layer 7: stride 1
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # ->(128,176,320)
            nn.ReLU(True),
            # Layer 8: stride 2
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   # ->(64,352,640)
            nn.ReLU(True),
            # Layer 9: stride 1
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),    # ->(64,352,640)
            nn.ReLU(True),
            # Layer 10: stride 2
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),      # ->(3,704,1280) approx
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        B, C, H, W = h.size()
        h_flat = h.view(B, -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(-1, 512, 22, 40)
        x_recon = self.decoder(h_dec)
        # Interpolate to exactly TARGET_HEIGHT x TARGET_WIDTH (720p)
        x_recon = torch.nn.functional.interpolate(x_recon, size=(TARGET_HEIGHT, TARGET_WIDTH), mode='bilinear', align_corners=False)
        return x_recon
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        recon_loss = nn.MSELoss(reduction='sum')(x_recon, x)
        loss = recon_loss + self.beta * kl_loss
        return x_recon, mu, logvar, loss

##############################################
# 3. Training Function with Early Stop
##############################################
def train_vae(folder, device, model_save_path):
    transform = transforms.Compose([
        transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = SingleFolderDataset(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    model = BetaVAE(latent_dim=LATENT_DIM, beta=BETA_VAE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Training VAE on {folder} (up to {MAX_EPOCHS} epochs or until avg loss < {LOSS_THRESHOLD})")
    model.train()
    for epoch in range(MAX_EPOCHS):
        train_loss = 0.0
        count = 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar, loss = model(x)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            count += x.size(0)
        avg_loss = train_loss / count if count > 0 else 9999
        print(f"[{os.path.basename(folder)}] Epoch {epoch+1}/{MAX_EPOCHS} - Avg Loss: {avg_loss:.4f}")
        if avg_loss < LOSS_THRESHOLD:
            print(f"Early stopping: avg loss {avg_loss:.4f} < {LOSS_THRESHOLD}")
            break
    
    torch.save(model.state_dict(), model_save_path)
    print(f"VAE model saved to: {model_save_path}")
    return model

##############################################
# 4. Sampling Function from VAE
##############################################
def sample_from_vae(model, num_samples=POOL_SIZE):
    model.eval()
    samples = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, LATENT_DIM, device=device)
            out = model.decode(z)
            out = (out + 1) / 2  # Denormalize to [0,1]
            samples.append(out.cpu())
    return samples

##############################################
# 5. Utility: Convert Tensor to PIL Image & Save
##############################################
def tensor_to_pil(tensor):
    arr = tensor.squeeze(0).permute(1, 2, 0).numpy()  # (H,W,3)
    arr = (arr * 255).clip(0,255).astype(np.uint8)
    return Image.fromarray(arr)

def save_tensor_image(tensor, output_path):
    pil_img = tensor_to_pil(tensor)
    pil_img = post_process_image(pil_img)
    pil_img.save(output_path)
    print(f"Saved image: {output_path}")

##############################################
# 6. Post-processing: Gaussian Blur & Bilateral Filter
##############################################
def post_process_image(image):
    image = image.filter(ImageFilter.GaussianBlur(radius=1))
    np_img = np.array(image)
    np_img = cv2.bilateralFilter(np_img, 9, 75, 75)
    return Image.fromarray(np_img)

##############################################
# 7. Create Animated GIF
##############################################
def create_animated_gif(frames, output_path, duration=0.2):
    gif_frames = []
    for f in frames:
        img = tensor_to_pil(f)
        img = post_process_image(img)
        gif_frames.append(img)
    imageio.mimsave(output_path, gif_frames, duration=duration)
    print(f"GIF saved at {output_path}")

##############################################
# 8. Assemble Room Folders Using VAE Assets and Pre-generated Text Images
##############################################
def assemble_rooms(num_rooms=70, door_pool=None, atmosphere_pool=None, ghost_pool=None, hallowean_model=None):
    base_dir = os.path.join(ROOMS_OUT_DIR, "rooms")
    os.makedirs(base_dir, exist_ok=True)
    
    riddle_imgs = [os.path.join(RIDDLE_TEXT_DIR, f) for f in os.listdir(RIDDLE_TEXT_DIR) if f.lower().endswith(".png")]
    sudoku_imgs = [os.path.join(SUDOKU_TEXT_DIR, f) for f in os.listdir(SUDOKU_TEXT_DIR) if f.lower().endswith(".png")]
    valid_guess_imgs = [os.path.join(VALID_GUESSES_TEXT_DIR, f) for f in os.listdir(VALID_GUESSES_TEXT_DIR) if f.lower().endswith(".png")]
    
    for rid in range(1, num_rooms+1):
        room_path = os.path.join(base_dir, f"room_{rid:03d}")
        os.makedirs(room_path, exist_ok=True)
        
        # 2 Door images
        if door_pool and len(door_pool) >= 2:
            chosen = random.sample(door_pool, 2)
            for i, d in enumerate(chosen, 1):
                save_tensor_image(d, os.path.join(room_path, f"door_{i}.png"))
        
        # 4 Atmosphere images
        if atmosphere_pool and len(atmosphere_pool) >= 4:
            chosen = random.sample(atmosphere_pool, 4)
            for i, a in enumerate(chosen, 1):
                save_tensor_image(a, os.path.join(room_path, f"atmosphere_{i}.png"))
        
        # 2 Riddle images
        if riddle_imgs:
            chosen = random.sample(riddle_imgs, min(2, len(riddle_imgs)))
            for i, src in enumerate(chosen, 1):
                shutil.copy(src, os.path.join(room_path, f"riddle_{i}.png"))
        
        # 2 Sudoku images
        if sudoku_imgs:
            chosen = random.sample(sudoku_imgs, min(2, len(sudoku_imgs)))
            for i, src in enumerate(chosen, 1):
                shutil.copy(src, os.path.join(room_path, f"sudoku_{i}.png"))
        
        # 1 Valid guess image
        if valid_guess_imgs:
            src = random.choice(valid_guess_imgs)
            shutil.copy(src, os.path.join(room_path, "valid_guesses.png"))
        
        # Ghost GIF from 10 frames
        if ghost_pool and len(ghost_pool) >= 10:
            chosen = random.sample(ghost_pool, 10)
            gif_path = os.path.join(room_path, "ghost.gif")
            create_animated_gif(chosen, gif_path, duration=0.2)
        
        # Hallowean image generated on the fly
        if hallowean_model:
            with torch.no_grad():
                z = torch.randn(1, LATENT_DIM, device=device)
                out = hallowean_model.decode(z)
                out = (out + 1)/2
            save_tensor_image(out.cpu(), os.path.join(room_path, "hallowean.png"))
        
        print(f"Room {rid} assembled.")
    print("All rooms assembled successfully.")

##############################################
# 9. Main Routine
##############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Beta-VAE for 720p images (stop early if avg loss < 1.1) and assemble room folders using VAE-generated assets and pre-generated text images.")
    parser.add_argument('--mode', type=str, default='all', choices=['train','generate','all'],
                        help="Choose 'train' to train VAEs, 'generate' to assemble rooms using saved VAE models, or 'all' to do both.")
    parser.add_argument('--rooms', type=int, default=70, help="Number of room folders to generate.")
    parser.add_argument('--epochs', type=int, default=num_epochs, help="Number of epochs for VAE training (default 800).")
    args = parser.parse_args()
    
    # Training phase
    if args.mode in ['train','all']:
        print("=== Training VAE for DOOR images ===")
        door_vae = train_vae(DOOR_DIR, device, DOOR_VAE_PATH)
        
        print("=== Training VAE for ATMOSPHERE images ===")
        atmos_vae = train_vae(ATMOSPHERE_DIR, device, ATMOS_VAE_PATH)
        
        print("=== Training VAE for GHOST images ===")
        ghost_vae = train_vae(GHOST_DIR, device, GHOST_VAE_PATH)
        
        print("=== Training VAE for HALLOWEAN images ===")
        hallowean_vae = train_vae(HALLOWEAN_DIR, device, HALLOWEAN_VAE_PATH)
    
    # Generation phase
    if args.mode in ['generate','all']:
        if args.mode == 'generate':
            door_vae = BetaVAE(latent_dim=LATENT_DIM, beta=BETA_VAE).to(device)
            atmos_vae = BetaVAE(latent_dim=LATENT_DIM, beta=BETA_VAE).to(device)
            ghost_vae = BetaVAE(latent_dim=LATENT_DIM, beta=BETA_VAE).to(device)
            hallowean_vae = BetaVAE(latent_dim=LATENT_DIM, beta=BETA_VAE).to(device)
            
            door_vae.load_state_dict(torch.load(DOOR_VAE_PATH, map_location=device))
            atmos_vae.load_state_dict(torch.load(ATMOS_VAE_PATH, map_location=device))
            ghost_vae.load_state_dict(torch.load(GHOST_VAE_PATH, map_location=device))
            hallowean_vae.load_state_dict(torch.load(HALLOWEAN_VAE_PATH, map_location=device))
        
        print("Sampling images from each VAE to create pools...")
        door_pool = sample_from_vae(door_vae, POOL_SIZE)
        atmosphere_pool = sample_from_vae(atmos_vae, POOL_SIZE)
        ghost_pool = sample_from_vae(ghost_vae, POOL_SIZE)
        
        print("Assembling rooms...")
        assemble_rooms(num_rooms=args.rooms,
                       door_pool=door_pool,
                       atmosphere_pool=atmosphere_pool,
                       ghost_pool=ghost_pool,
                       hallowean_model=hallowean_vae)
