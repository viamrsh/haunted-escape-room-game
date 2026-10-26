# D:\project\haunted_escape_room_backend\generate_dialogue_image.py

import os
import json
import random
import torch
from PIL import Image, ImageDraw, ImageFont

from app.config import Config
from app.ai.llm_module import LLMModel

def clean_text(sentence: str) -> str:
    return sentence.replace("<color=", "").replace(">", " ").replace("</color>", " ")

def tokenize(sentence, word2idx):
    clean = clean_text(sentence)
    return [word2idx.get(word.lower(), word2idx["<UNK>"]) for word in clean.split()]

# Load saved vocabulary.
vocab_path = os.path.join(Config.GHOST_DIR, "vocab.json")
if not os.path.exists(vocab_path):
    print("Vocabulary file not found. Please run train.py first.")
    exit(1)
with open(vocab_path, "r") as f:
    VOCAB = json.load(f)
print("Loaded vocabulary from file. Size:", len(VOCAB))

VOCAB_SIZE = len(VOCAB)
PAD_IDX = VOCAB["<PAD>"]

# Build reverse mapping.
IDX2WORD = {int(idx): word for word, idx in VOCAB.items()}

def generate_text(prompt, model, vocab, idx2word, max_length=50):
    tokens = tokenize(prompt, vocab)
    input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
    generated = tokens[:]
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_tensor)  # (1, seq_len, vocab_size)
            next_token_logits = outputs[0, -1]
            next_token = torch.argmax(next_token_logits).item()
            generated.append(next_token)
            input_tensor = torch.cat(
                [input_tensor, torch.tensor([[next_token]], dtype=torch.long).to(device)], dim=1
            )
            if next_token == PAD_IDX:
                break
    words = [idx2word.get(token, "<UNK>") for token in generated]
    return " ".join(words)

COLOR_CODES = [
    "#FF0000", "#DC143C", "#B22222", "#FFD700", "#FFDF00",
    "#DAA520", "#FF8C00", "#FF4500", "#FF6347", "#00FF7F"
]

def render_dialogue_image(text, output_path, font_folder=Config.HAUNTED_TEXT_DIR, image_size=(800,600), font_size=32):
    font_files = [os.path.join(font_folder, f) for f in os.listdir(font_folder) if f.endswith(".ttf")]
    if font_files:
        font_path = random.choice(font_files)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    text_color = random.choice(COLOR_CODES)
    lines = text.split("\n")
    y = 10
    for line in lines:
        draw.text((10, y), line, fill=text_color, font=font)
        bbox = font.getbbox(line)
        line_height = bbox[3] - bbox[1]
        y += line_height + 5
    image.save(output_path)
    return image

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = LLMModel(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        max_seq_length=20
    ).to(device)

    llm_weights_path = os.path.join(Config.GHOST_DIR, "llm_model.pth")
    if not os.path.exists(llm_weights_path):
        print("Trained LLM weights not found. Please run train.py first.")
        exit(1)
    model.load_state_dict(torch.load(llm_weights_path, map_location=device))
    print("Trained LLM weights loaded.")

    prompt = "Room 1: Welcome to the haunted escape room. The ghost asks:"
    generated_text = generate_text(prompt, model, VOCAB, IDX2WORD, max_length=50)
    print("Generated Text:")
    print(generated_text)

    output_image_path = os.path.join(Config.GHOST_DIR, "generated_dialogue.png")
    render_dialogue_image(generated_text, output_image_path)
    print(f"Dialogue image saved to: {output_image_path}")
