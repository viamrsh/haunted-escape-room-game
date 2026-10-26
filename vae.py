# D:\project\haunted_escape_room_backend\generate_image_for_vae.py

import os
import random
import argparse
import shutil
import imageio
import textwrap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

##############################################
# Global Paths and Hyperparameters
##############################################
ROOMS_OUT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\rooms"
os.makedirs(ROOMS_OUT_DIR, exist_ok=True)

# Single-folder datasets for VAE training
DOOR_DIR = r"D:\project\haunted_escape_room_backend\assets\door"
ATMOSPHERE_DIR = r"D:\project\haunted_escape_room_backend\assets\atmosphere"
GHOST_DIR = r"D:\project\haunted_escape_room_backend\assets\ghost"
HALLOWEAN_DIR = r"D:\project\haunted_escape_room_backend\assets\hallowean"

# Directories for pre-generated text images
RIDDLE_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\riddle"
SUDOKU_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\sudoku"
VALID_GUESSES_TEXT_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\texts\valid_guesses"

# VAE model save paths
DOOR_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\door_vae.pth"
ATMOS_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\atmosphere_vae.pth"
GHOST_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\ghost_vae.pth"
HALLOWEAN_VAE_PATH = r"D:\project\haunted_escape_room_backend\assets\ghost\hallowean_vae.pth"

# VAE hyperparameters
latent_dim = 100
nc = 3           # number of channels (RGB)
image_size = 64  # Increase if you want bigger images (e.g. 128)
num_epochs = 500 # Train for 500 epochs
batch_size = 64
lr = 0.0002

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
POOL_SIZE = 100

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
# 2. VAE Model Definition
##############################################
class VAE(nn.Module):
    def __init__(self, latent_dim=latent_dim, nc=nc):
        super(VAE, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, nc, 4, 2, 1),
            nn.Tanh()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        h_dec = self.fc_decode(z)
        h_dec = h_dec.view(x.size(0), 512, 4, 4)
        x_recon = self.decoder(h_dec)
        return x_recon, mu, logvar

##############################################
# 3. VAE Training Function
##############################################
def train_vae(folder, epochs, batch_size, device, model_save_path):
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])
    dataset = SingleFolderDataset(folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    model = VAE(latent_dim=latent_dim, nc=nc).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    reconstruction_loss_fn = nn.MSELoss(reduction='sum')

    print(f"Training VAE on {folder} for {epochs} epochs... (# images: {len(dataset)})")
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for i, data in enumerate(dataloader):
            x, _ = data
            x = x.to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            recon_loss = reconstruction_loss_fn(x_recon, x)
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"[{os.path.basename(folder)}] Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(dataset):.4f}")
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
            z = torch.randn(1, latent_dim, device=device)
            h_dec = model.fc_decode(z)
            h_dec = h_dec.view(1, 512, 4, 4)
            sample = model.decoder(h_dec)
            sample = sample.detach().cpu()
            sample = (sample + 1) / 2  # Denormalize to [0,1]
            samples.append(sample)
    return samples

##############################################
# 5. Utility: Convert Tensor to PIL Image & Save
##############################################
def tensor_to_pil(tensor):
    image = tensor.squeeze(0).permute(1, 2, 0)
    image = image.mul(255).clamp(0,255).byte().cpu().numpy()
    return Image.fromarray(image)

def save_tensor_image(tensor, output_path):
    pil_img = tensor_to_pil(tensor)
    pil_img.save(output_path)
    print(f"Saved image: {output_path}")

##############################################
# 6. Create Animated GIF using imageio
##############################################
def create_animated_gif(image_list, output_path, duration=0.2):
    if image_list:
        imageio.mimsave(output_path, image_list, duration=duration)
        print(f"Created animated GIF: {output_path}")

##############################################
# 7. Assemble Room Folders (using VAE-generated images + pre-generated text images)
##############################################
def assemble_rooms(num_rooms=70, door_pool=None, atmosphere_pool=None, ghost_pool=None, hallowean_model=None):
    room_base = os.path.join(ROOMS_OUT_DIR, "rooms")
    os.makedirs(room_base, exist_ok=True)
    
    # Load pre-generated text images from your directories
    riddle_images = [os.path.join(RIDDLE_TEXT_DIR, f) for f in os.listdir(RIDDLE_TEXT_DIR) if f.lower().endswith(".png")]
    sudoku_images = [os.path.join(SUDOKU_TEXT_DIR, f) for f in os.listdir(SUDOKU_TEXT_DIR) if f.lower().endswith(".png")]
    valid_guess_images = [os.path.join(VALID_GUESSES_TEXT_DIR, f) for f in os.listdir(VALID_GUESSES_TEXT_DIR) if f.lower().endswith(".png")]
    
    for room_id in range(1, num_rooms+1):
        room_folder = os.path.join(room_base, f"room_{room_id:03d}")
        os.makedirs(room_folder, exist_ok=True)
        
        # 2 door images
        if door_pool and len(door_pool) >= 2:
            door_samples = random.sample(door_pool, 2)
            for i, tensor_img in enumerate(door_samples, 1):
                door_path = os.path.join(room_folder, f"door_{i}.png")
                save_tensor_image(tensor_img, door_path)
        
        # 4 atmosphere images
        if atmosphere_pool and len(atmosphere_pool) >= 4:
            atmos_samples = random.sample(atmosphere_pool, 4)
            for i, tensor_img in enumerate(atmos_samples, 1):
                atmos_path = os.path.join(room_folder, f"atmosphere_{i}.png")
                save_tensor_image(tensor_img, atmos_path)
        
        # 2 riddle images
        if riddle_images:
            selected = random.sample(riddle_images, min(2, len(riddle_images)))
            for i, src in enumerate(selected, 1):
                dst = os.path.join(room_folder, f"riddle_{i}.png")
                shutil.copy(src, dst)
        else:
            print("No riddle images found.")
        
        # 2 sudoku images
        if sudoku_images:
            selected = random.sample(sudoku_images, min(2, len(sudoku_images)))
            for i, src in enumerate(selected, 1):
                dst = os.path.join(room_folder, f"sudoku_{i}.png")
                shutil.copy(src, dst)
        else:
            print("No sudoku images found.")
        
        # 1 valid guess image
        if valid_guess_images:
            src = random.choice(valid_guess_images)
            dst = os.path.join(room_folder, "valid_guesses.png")
            shutil.copy(src, dst)
        else:
            print("No valid guess images found.")
        
        # Ghost GIF from 10 frames
        if ghost_pool and len(ghost_pool) >= 10:
            ghost_samples = random.sample(ghost_pool, 10)
            ghost_frames = [tensor_to_pil(t) for t in ghost_samples]
            ghost_gif_path = os.path.join(room_folder, "ghost.gif")
            create_animated_gif(ghost_frames, ghost_gif_path, duration=0.2)
        else:
            print("Insufficient ghost samples to create GIF.")
        
        # Generate 1 hallowean image on the fly
        if hallowean_model:
            with torch.no_grad():
                z = torch.randn(1, latent_dim, device=device)
                h_dec = hallowean_model.fc_decode(z)
                h_dec = h_dec.view(1, 512, 4, 4)
                hallowean_tensor = hallowean_model.decoder(h_dec)
                hallowean_tensor = hallowean_tensor.detach().cpu()
            hallowean_path = os.path.join(room_folder, "hallowean.png")
            save_tensor_image(hallowean_tensor, hallowean_path)
        
        print(f"Room {room_id} assembled.")
    
    print(f"All {num_rooms} rooms have been assembled successfully.")

##############################################
# 8. Main Routine
##############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE models on image datasets for 500 epochs and assemble room folders using VAE-generated assets + pre-generated text images.")
    parser.add_argument('--mode', type=str, default='all', choices=['train','generate','all'],
                        help="Choose 'train' to train VAEs, 'generate' to assemble rooms using saved VAE models, or 'all' to do both.")
    parser.add_argument('--rooms', type=int, default=70, help="Number of room folders to generate.")
    parser.add_argument('--epochs', type=int, default=num_epochs, help="Number of epochs for VAE training on each dataset (default 500).")
    args = parser.parse_args()
    
    # Train VAEs if needed
    if args.mode in ['train','all']:
        print("Training VAE for door images...")
        door_vae = train_vae(DOOR_DIR, epochs=args.epochs, batch_size=batch_size, device=device, model_save_path=DOOR_VAE_PATH)
        
        print("Training VAE for atmosphere images...")
        atmos_vae = train_vae(ATMOSPHERE_DIR, epochs=args.epochs, batch_size=batch_size, device=device, model_save_path=ATMOS_VAE_PATH)
        
        print("Training VAE for ghost images...")
        ghost_vae = train_vae(GHOST_DIR, epochs=args.epochs, batch_size=batch_size, device=device, model_save_path=GHOST_VAE_PATH)
        
        print("Training VAE for hallowean images...")
        hallowean_vae = train_vae(HALLOWEAN_DIR, epochs=args.epochs, batch_size=batch_size, device=device, model_save_path=HALLOWEAN_VAE_PATH)
    
    # Generate rooms if needed
    if args.mode in ['generate','all']:
        # If we didn't just train, load from disk
        if args.mode == 'generate':
            door_vae = VAE(latent_dim=latent_dim, nc=nc).to(device)
            atmos_vae = VAE(latent_dim=latent_dim, nc=nc).to(device)
            ghost_vae = VAE(latent_dim=latent_dim, nc=nc).to(device)
            hallowean_vae = VAE(latent_dim=latent_dim, nc=nc).to(device)
            door_vae.load_state_dict(torch.load(DOOR_VAE_PATH, map_location=device))
            atmos_vae.load_state_dict(torch.load(ATMOS_VAE_PATH, map_location=device))
            ghost_vae.load_state_dict(torch.load(GHOST_VAE_PATH, map_location=device))
            hallowean_vae.load_state_dict(torch.load(HALLOWEAN_VAE_PATH, map_location=device))
        
        print("Sampling images from each VAE to create a pool...")
        door_pool = sample_from_vae(door_vae, num_samples=POOL_SIZE)
        atmosphere_pool = sample_from_vae(atmos_vae, num_samples=POOL_SIZE)
        ghost_pool = sample_from_vae(ghost_vae, num_samples=POOL_SIZE)
        
        # hallowean_vae used on the fly in assemble_rooms
        print("Assembling rooms...")
        assemble_rooms(num_rooms=args.rooms,
                       door_pool=door_pool,
                       atmosphere_pool=atmosphere_pool,
                       ghost_pool=ghost_pool,
                       hallowean_model=hallowean_vae)
