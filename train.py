# D:\project\haunted_escape_room_backend\run_all.py

import os
import json
import random
import textwrap
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import argparse
from PIL import Image, ImageDraw, ImageFont

# Import configuration and model modules
from app.config import Config
from app.ai.llm_module import LLMModel  # Ensure LLMModel uses PositionalEncoding(max_len=5000) with batch_first=True
from app.ai.gan_module import GANGenerator, GANDiscriminator

##########################################
# Helper Functions: Cleaning and Tokenizing
##########################################
def clean_text(sentence: str) -> str:
    return sentence.replace("<color=", "").replace(">", " ").replace("</color>", " ")

def tokenize(sentence, word2idx):
    return [word2idx.get(word.lower(), word2idx["<UNK>"]) for word in clean_text(sentence).split()]

##########################################
# Data Loading Functions
##########################################
def load_riddles_from_csv(csv_path):
    """
    Load texts from a CSV file using the 'Riddle' column.
    (If your CSV uses a different column such as 'clue', adjust accordingly.)
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    print(f"CSV columns in {os.path.basename(csv_path)}: {df.columns.tolist()}")
    # Change "Riddle" to "clue" if needed.
    if "Riddle" in df.columns:
        return df["Riddle"].astype(str).tolist()
    elif "clue" in df.columns:
        return df["clue"].astype(str).tolist()
    else:
        print(f"Neither 'Riddle' nor 'clue' found in {csv_path}.")
        return []

def load_sudoku_from_csv(csv_path):
    """
    Load sudoku puzzles from a CSV file using the 'puzzle' column.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "puzzle" not in df.columns:
        print(f"Column 'puzzle' not found in {csv_path}.")
        return []
    return df["puzzle"].astype(str).tolist()

def load_mix_letters_from_csv(csv_path):
    """
    Load words from a CSV file using the 'word' column.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    if "word" not in df.columns:
        print(f"Column 'word' not found in {csv_path}.")
        return []
    return df["word"].astype(str).tolist()

def load_all_training_texts():
    texts = []
    # Riddles from two CSV files.
    riddles_dir = os.path.join(Config.BASE_DIR, "data", "riddles")
    times_csv = os.path.join(riddles_dir, "times.csv")
    big_dave_csv = os.path.join(riddles_dir, "big_dave.csv")
    texts += load_riddles_from_csv(times_csv)
    texts += load_riddles_from_csv(big_dave_csv)
    
    # Sudoku puzzles.
    sudoku_csv = os.path.join(Config.BASE_DIR, "data", "puzzles", "Gravity Falls", "sudoku.csv")
    texts += load_sudoku_from_csv(sudoku_csv)
    
    # Mix-of-letters.
    mix_folder = os.path.join(Config.BASE_DIR, "data", "puzzles", "Gravity Falls", "mixing of letters")
    valid_guesses_csv = os.path.join(mix_folder, "valid_guesses.csv")
    valid_solutions_csv = os.path.join(mix_folder, "valid_solutions.csv")
    texts += load_mix_letters_from_csv(valid_guesses_csv)
    texts += load_mix_letters_from_csv(valid_solutions_csv)
    return texts

##########################################
# Vocabulary and Dataset Setup
##########################################
def build_vocab(sentences, min_freq=1):
    word_freq = {}
    for sentence in sentences:
        c = clean_text(sentence)
        for word in c.split():
            word = word.lower()
            word_freq[word] = word_freq.get(word, 0) + 1
    vocab = {word for word, freq in word_freq.items() if freq >= min_freq}
    word2idx = {"<PAD>": 0, "<UNK>": 1}
    for word in sorted(vocab):
        word2idx[word] = len(word2idx)
    return word2idx

class TextDataset(Dataset):
    def __init__(self, sentences, word2idx, max_seq_len=20):
        self.data = []
        for sentence in sentences:
            tokens = self.tokenize(sentence, word2idx)
            if len(tokens) < max_seq_len:
                tokens += [word2idx["<PAD>"]] * (max_seq_len - len(tokens))
            else:
                tokens = tokens[:max_seq_len]
            self.data.append(torch.tensor(tokens, dtype=torch.long))
    
    def tokenize(self, sentence, word2idx):
        c = clean_text(sentence)
        return [word2idx.get(word.lower(), word2idx["<UNK>"]) for word in c.split()]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

##########################################
# Training Routines: LLM and GAN
##########################################
def train_llm():
    texts = load_all_training_texts()
    print(f"Loaded {len(texts)} training texts.")

    vocab = build_vocab(texts)
    vocab_path = os.path.join(Config.GHOST_DIR, "vocab.json")
    os.makedirs(Config.GHOST_DIR, exist_ok=True)
    with open(vocab_path, "w") as f:
        json.dump(vocab, f)
    print(f"Vocabulary saved with size: {len(vocab)}")

    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]

    dataset = TextDataset(texts, vocab, max_seq_len=20)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LLMModel(
        vocab_size=vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_length=20  # Training on sequences of length 20.
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    num_epochs = 5
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            optimizer.zero_grad()
            outputs = model(inputs)  # shape: (batch, seq_len-1, vocab_size)
            outputs = outputs.reshape(-1, vocab_size)
            targets = targets.reshape(-1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"[LLM] Epoch {epoch+1}/10 - Loss: {avg_loss:.4f}", flush=True)

    model_path = os.path.join(Config.GHOST_DIR, "llm_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"LLM model saved to: {model_path}")
    return model, vocab

def train_gan():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load vocabulary.
    vocab_path = os.path.join(Config.GHOST_DIR, "vocab.json")
    if not os.path.exists(vocab_path):
        print("Vocabulary file not found. Please run train_llm() first.")
        exit(1)
    with open(vocab_path, "r") as f:
        vocab = json.load(f)
    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]

    noise_dim = 100
    gan_generator = GANGenerator(noise_dim=noise_dim, vocab_size=vocab_size, llm_d_model=128).to(device)
    gan_discriminator = GANDiscriminator().to(device)

    llm_weights_path = os.path.join(Config.GHOST_DIR, "llm_model.pth")
    if not os.path.exists(llm_weights_path):
        print("LLM weights not found. Please run train_llm() first.")
        exit(1)
    gan_generator.llm.load_state_dict(torch.load(llm_weights_path, map_location=device))
    print("Loaded trained LLM weights into GAN generator.", flush=True)

    # Dummy image dataset.
    class DummyImageDataset(Dataset):
        def __init__(self, num_samples=500):
            self.num_samples = num_samples
        def __len__(self):
            return self.num_samples
        def __getitem__(self, idx):
            return torch.randn(3, 64, 64)
    image_dataset = DummyImageDataset(num_samples=500)
    image_dataloader = DataLoader(image_dataset, batch_size=16, shuffle=True)

    # Prepare dummy text data for GAN conditioning using vocabulary keys.
    dummy_texts = list(vocab.keys())
    def tokenize_text(text):
        return [vocab.get(word.lower(), vocab["<UNK>"]) for word in clean_text(text).split()]
    text_data_list = [torch.tensor(tokenize_text(word), dtype=torch.long) for word in dummy_texts]
    max_seq_len = 20
    padded_texts = []
    for tokens in text_data_list:
        if len(tokens) < max_seq_len:
            tokens = torch.cat([tokens, torch.zeros(max_seq_len - len(tokens), dtype=torch.long)])
        else:
            tokens = tokens[:max_seq_len]
        padded_texts.append(tokens)
    text_data = torch.stack(padded_texts)
    num_texts = text_data.size(0)

    adversarial_loss = nn.BCELoss()
    optimizer_G = torch.optim.Adam(gan_generator.parameters(), lr=1e-3)
    optimizer_D = torch.optim.Adam(gan_discriminator.parameters(), lr=1e-3)

    num_epochs = 5
    gan_generator.train()
    gan_discriminator.train()
    for epoch in range(num_epochs):
        for real_images in image_dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            valid = torch.ones(batch_size, 1).to(device)
            fake = torch.zeros(batch_size, 1).to(device)

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(gan_discriminator(real_images), valid)

            noise = torch.randn(batch_size, noise_dim).to(device)
            indices = torch.randint(0, num_texts, (batch_size,))
            text_batch = text_data[indices].to(device)

            fake_images = gan_generator(noise, text_batch)
            fake_loss = adversarial_loss(gan_discriminator(fake_images.detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            gen_loss = adversarial_loss(gan_discriminator(fake_images), valid)
            gen_loss.backward()
            optimizer_G.step()
        print(f"[GAN] Epoch {epoch+1}/10 - D Loss: {d_loss.item():.4f}, G Loss: {gen_loss.item():.4f}", flush=True)
    
    gan_gen_save_path = os.path.join(Config.GHOST_DIR, "gan_generator.pth")
    gan_disc_save_path = os.path.join(Config.GHOST_DIR, "gan_discriminator.pth")
    torch.save(gan_generator.state_dict(), gan_gen_save_path)
    torch.save(gan_discriminator.state_dict(), gan_disc_save_path)
    print(f"GAN generator saved to: {gan_gen_save_path}", flush=True)
    print(f"GAN discriminator saved to: {gan_disc_save_path}", flush=True)
    return gan_generator, gan_discriminator

##########################################
# Rendering Functions for Dataset Images
##########################################
def render_text_image(text, output_path, wrap_width=50, image_size=(800,600), font_size=32, bg_color="black"):
    COLOR_CODES = [
        "#FF0000", "#DC143C", "#B22222", "#FFD700", "#FFDF00",
        "#DAA520", "#FF8C00", "#FF4500", "#FF6347", "#00FF7F"
    ]
    text_color = random.choice(COLOR_CODES)
    lines = textwrap.wrap(text, width=wrap_width)
    font_files = [os.path.join(HAUNTED_FONT_DIR, f) for f in os.listdir(HAUNTED_FONT_DIR) if f.endswith(".ttf")]
    if font_files:
        font_path = random.choice(font_files)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    image = Image.new("RGB", image_size, bg_color)
    draw = ImageDraw.Draw(image)
    x, y = 10, 10
    for line in lines:
        draw.text((x, y), line, fill=text_color, font=font)
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y += line_height + 5
    image.save(output_path)
    print(f"Saved text image: {output_path}")

def render_sudoku_table(sudoku_str, output_path):
    sudoku_str = sudoku_str.strip()
    if len(sudoku_str) < 81:
        sudoku_str = sudoku_str.ljust(81, '0')
    else:
        sudoku_str = sudoku_str[:81]
    cell_size = 50
    grid_size = cell_size * 9
    bg_color = "black"
    line_color = "#FFFFFF"
    text_color = "#FFA500"
    image = Image.new("RGB", (grid_size + 1, grid_size + 1), bg_color)
    draw = ImageDraw.Draw(image)
    font_size = 32
    font_files = [os.path.join(HAUNTED_FONT_DIR, f) for f in os.listdir(HAUNTED_FONT_DIR) if f.endswith(".ttf")]
    if font_files:
        font_path = random.choice(font_files)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        y_line = i * cell_size
        draw.line([(0, y_line), (grid_size, y_line)], fill=line_color, width=thickness)
        x_line = i * cell_size
        draw.line([(x_line, 0), (x_line, grid_size)], fill=line_color, width=thickness)
    for row in range(9):
        for col in range(9):
            idx = row * 9 + col
            ch = sudoku_str[idx]
            if ch in ['0', '.', ' ']:
                continue
            x_center = col * cell_size + cell_size // 2
            y_center = row * cell_size + cell_size // 2
            w, h = draw.textsize(ch, font=font)
            draw.text((x_center - w/2, y_center - h/2), ch, fill=text_color, font=font)
    image.save(output_path)
    print(f"Saved sudoku image: {output_path}")

##########################################
# Generation Routine: Render Dataset Images
##########################################
def generate_dataset_images():
    # Create output subfolders for each dataset.
    times_out_dir = os.path.join(OUTPUT_BASE_DIR, "times")
    bigdave_out_dir = os.path.join(OUTPUT_BASE_DIR, "big_dave")
    sudoku_out_dir = os.path.join(OUTPUT_BASE_DIR, "sudoku")
    guesses_out_dir = os.path.join(OUTPUT_BASE_DIR, "valid_guesses")
    solutions_out_dir = os.path.join(OUTPUT_BASE_DIR, "valid_solutions")
    for d in [times_out_dir, bigdave_out_dir, sudoku_out_dir, guesses_out_dir, solutions_out_dir]:
        os.makedirs(d, exist_ok=True)

    # 1. Riddles: times.csv
    times_texts = load_riddles(TIMES_CSV)
    print(f"Generating images for {len(times_texts)} riddles from times.csv...", flush=True)
    for i, text in enumerate(times_texts):
        out_file = os.path.join(times_out_dir, f"times_{i+1}.png")
        render_text_image(text, out_file, wrap_width=50)

    # 2. Riddles: big_dave.csv
    bigdave_texts = load_riddles(BIGDAVE_CSV)
    print(f"Generating images for {len(bigdave_texts)} riddles from big_dave.csv...", flush=True)
    for i, text in enumerate(bigdave_texts):
        out_file = os.path.join(bigdave_out_dir, f"bigdave_{i+1}.png")
        render_text_image(text, out_file, wrap_width=50)

    # 3. Sudoku: select 50 random puzzles.
    sudoku_list = load_sudoku(SUDOKU_CSV)
    if sudoku_list:
        if len(sudoku_list) >= 50:
            selected_sudokus = random.sample(sudoku_list, 50)
        else:
            selected_sudokus = sudoku_list
        print(f"Generating images for {len(selected_sudokus)} sudoku puzzles...", flush=True)
        for i, puzzle in enumerate(selected_sudokus):
            out_file = os.path.join(sudoku_out_dir, f"sudoku_{i+1}.png")
            render_sudoku_table(puzzle, out_file)
    else:
        print("Sudoku dataset not found or empty.", flush=True)

    # 4. Mix-of-letters: valid_guesses.csv
    guesses = load_mix_letters(VALID_GUESSES_CSV)
    print(f"Generating images for {len(guesses)} words from valid_guesses.csv...", flush=True)
    for i, word in enumerate(guesses):
        out_file = os.path.join(guesses_out_dir, f"guess_{i+1}.png")
        render_text_image(word, out_file, wrap_width=20)

    # 5. Mix-of-letters: valid_solutions.csv
    solutions = load_mix_letters(VALID_SOLUTIONS_CSV)
    print(f"Generating images for {len(solutions)} words from valid_solutions.csv...", flush=True)
    for i, word in enumerate(solutions):
        out_file = os.path.join(solutions_out_dir, f"solution_{i+1}.png")
        render_text_image(word, out_file, wrap_width=20)

    print("All dataset images have been generated successfully.", flush=True)

##########################################
# Main Routine: Train then Generate Dataset Images
##########################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    
    parser = argparse.ArgumentParser(
        description="Train LLM and GAN, then generate separate images for each dataset."
    )
    parser.add_argument('--mode', type=str, default='all', choices=['llm', 'gan', 'all'],
                        help="Mode: 'llm' for only LLM training, 'gan' for only GAN training, 'all' for both then generate images.")
    args = parser.parse_args()
    
    if args.mode == 'llm':
        train_llm()
    elif args.mode == 'gan':
        train_gan()
    else:
        # Mode 'all': Train LLM then GAN sequentially.
        llm_model, vocab = train_llm()
        train_gan()
    
    # Generate dataset images.
    generate_dataset_images()
