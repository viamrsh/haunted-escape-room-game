import os
import pytesseract
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from dotenv import load_dotenv
import pandas as pd
import json
import numpy as np

# Load environment variables
load_dotenv()

# Dataset Paths from .env
GHOST_IMAGES_DIR = os.getenv("GHOST_IMAGES_DIR")
ATMOSPHERE_IMAGES_DIR = os.getenv("ATMOSPHERE_IMAGES_DIR")
DOOR_IMAGES_DIR = os.getenv("DOOR_IMAGES_DIR")
HALLOWEAN_IMAGES_DIR = os.getenv("HALLOWEAN_IMAGES_DIR")
HAUNTED_TEXT_DIR = os.getenv("HAUNTED_TEXT_DIR")
RIDDLES_DIR = os.getenv("RIDDLES_DIR")
PUZZLES_GF_DIR = os.getenv("PUZZLES_GF_DIR")
PUZZLES_LETTERS_DIR = os.getenv("PUZZLES_LETTERS_DIR")
PUZZLES_SODUKU_DIR = os.getenv("PUZZLES_SODUKU_DIR")

# Specify the Tesseract executable path explicitly (if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'  # Update this if your installation path is different

# Function to extract text from haunted text images using Tesseract OCR
def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

# Extract haunted text from image files in the HAUNTED_TEXT_DIR
text_data = []
for filename in os.listdir(HAUNTED_TEXT_DIR):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(HAUNTED_TEXT_DIR, filename)
        text_data.append(extract_text_from_image(image_path))

# Tokenization with Byte Pair Encoding (BPE) using Tokenizers
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=50000, min_frequency=2)
tokenizer.train_from_iterator(text_data, trainer=trainer)

# Save tokenizer for later use
tokenizer.save("haunted_tokenizer.json")

# LLM Training (Simplified Transformer-based Model)
class CustomLLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(CustomLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=8, num_encoder_layers=num_layers, num_decoder_layers=num_layers, batch_first=True)
        self.fc_out = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x):
        # Embedding the input sequence
        embedded = self.embedding(x)
        # For language modeling, the target sequence (tgt) is the input sequence shifted by 1 position
        tgt = torch.roll(x, shifts=-1, dims=1)  # Shift the input to create the target
        # Feed into the transformer
        transformer_out = self.transformer(embedded, embedded)  # source (src) and target (tgt) are the same in language modeling
        output = self.fc_out(transformer_out)
        return output

# Hyperparameters for LLM
vocab_size = 50000  # Based on tokenizer
embedding_dim = 256
hidden_dim = 512
num_layers = 6

# Instantiate and initialize the model
model = CustomLLM(vocab_size, embedding_dim, hidden_dim, num_layers)

# Loss function and optimizer for LLM
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Convert tokenized text data into tensors for training
def encode_text_data(text_data, tokenizer):
    encoded_data = tokenizer.encode_batch(text_data)
    
    # Extract tokenized inputs and convert them into tensors
    input_ids = [encoding.ids for encoding in encoded_data]
    
    # Pad sequences to the same length
    max_length = max(len(seq) for seq in input_ids)
    input_ids_padded = [seq + [0] * (max_length - len(seq)) for seq in input_ids]
    
    # Convert the padded sequences to tensor
    input_tensor = torch.tensor(input_ids_padded)
    return input_tensor

# Tokenize the text data
train_data = encode_text_data(text_data, tokenizer)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Training loop for LLM
def train_llm(model, data_loader, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}, Loss: {total_loss / len(data_loader)}")

# Start LLM Training
train_llm(model, train_loader)

# GAN Training (for Ghost Image Generation)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 3 * 64 * 64)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        return x.view(-1, 3, 64, 64)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(3 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)

    def forward(self, img):
        x = img.view(-1, 3 * 64 * 64)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

# Initialize models
G = Generator()
D = Discriminator()

# Optimizers and loss function for GAN
optimizer_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
criterion = nn.BCELoss()

# Custom Dataset for Ghost Images
class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx])

        # Convert RGBA to RGB if the image has 4 channels
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        # If the image is grayscale (1 channel), convert it to 3 channels (RGB)
        if img.mode == 'L':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        return img

# Transformations for images (resize and normalization)
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the ghost images dataset
dataset = CustomImageDataset(GHOST_IMAGES_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training the GAN
def train_gan(num_epochs=50):
    for epoch in range(num_epochs):
        for real_imgs in dataloader:
            batch_size = real_imgs.size(0)
            valid_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)

            # Train the Discriminator
            optimizer_D.zero_grad()
            real_imgs = real_imgs.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            real_preds = D(real_imgs)
            real_loss = criterion(real_preds, valid_labels)

            z = torch.randn(batch_size, 100).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            fake_imgs = G(z)
            fake_preds = D(fake_imgs.detach())
            fake_loss = criterion(fake_preds, fake_labels)

            d_loss = real_loss + fake_loss
            d_loss.backward()
            optimizer_D.step()

            # Train the Generator
            optimizer_G.zero_grad()
            fake_preds = D(fake_imgs)
            g_loss = criterion(fake_preds, valid_labels)
            g_loss.backward()
            optimizer_G.step()

        print(f"Epoch [{epoch}/{num_epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")

# Start training the GAN
train_gan()

# Save the trained models
torch.save(G.state_dict(), "generator.pth")
torch.save(D.state_dict(), "discriminator.pth")
