# D:\project\haunted_escape_room_backend\generate_image.py

import os
import json
import random
import textwrap
import argparse
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn

##########################################
# 1. Global Paths
##########################################
# Output directory for generated images
OUTPUT_BASE_DIR = r"D:\project\haunted_escape_room_backend\models\haunted_text_gan\text"

# CSV file paths
# Use the "Riddles" file (assumed to be a CSV) from the riddles folder
RIDDLES_FILE = r"D:\project\haunted_escape_room_backend\data\riddles\Riddles.csv"
# Sudoku file (all puzzles will be generated)
SUDOKU_FILE = r"D:\project\haunted_escape_room_backend\data\puzzles\Gravity Falls\sudoku.csv"

# Haunted font folder – using the folder that contains your 253 fonts
HAUNTED_FONT_DIR = r"D:\project\haunted_escape_room_backend\assets\haunted text"

##########################################
# 2. Pixel-Based Wrapping using textbbox
##########################################
def wrap_text_by_pixels(text, font, draw, max_width):
    """
    Splits 'text' into multiple lines so that each line's pixel width
    does not exceed 'max_width' when rendered with 'font'.
    Uses draw.textbbox for measurement.
    """
    words = text.split()
    lines = []
    current_line = []
    for word in words:
        test_line = " ".join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        line_width = bbox[2] - bbox[0]
        if line_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
    if current_line:
        lines.append(" ".join(current_line))
    return lines

##########################################
# 3. Data Loading Functions
##########################################
def load_riddles(csv_path):
    """
    Loads riddle texts from a CSV file.
    Checks for a 'clue' column first; if not present, uses 'Riddle'.
    """
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return []
    df = pd.read_csv(csv_path)
    df.columns = [col.strip() for col in df.columns]
    print(f"CSV columns in {os.path.basename(csv_path)}: {df.columns.tolist()}")
    if "clue" in df.columns:
        return df["clue"].astype(str).tolist()
    elif "Riddle" in df.columns:
        return df["Riddle"].astype(str).tolist()
    else:
        print(f"Neither 'clue' nor 'Riddle' found in {csv_path}.")
        return []

def load_sudoku(csv_path):
    """
    Loads sudoku puzzles from a CSV file using the 'puzzle' column.
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

##########################################
# 4. Rendering Functions
##########################################
def render_text_image(text, output_path, image_size=(800,600), font_size=32, bg_color="black"):
    """
    Renders the provided text as multiline text on a black background,
    using pixel-based wrapping so that each line fits within the image width.
    """
    COLOR_CODES = [
        "#FF0000", "#DC143C", "#B22222", "#FFD700", "#FFDF00",
        "#DAA520", "#FF8C00", "#FF4500", "#FF6347", "#00FF7F"
    ]
    text_color = random.choice(COLOR_CODES)
    
    # Attempt to load a random .ttf font from HAUNTED_FONT_DIR.
    if os.path.exists(HAUNTED_FONT_DIR):
        font_files = [os.path.join(HAUNTED_FONT_DIR, f) for f in os.listdir(HAUNTED_FONT_DIR) if f.endswith(".ttf")]
    else:
        print(f"Haunted font directory not found: {HAUNTED_FONT_DIR}. Using default font.")
        font_files = []
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
    
    margin_x = 10
    max_line_width = image_size[0] - 2 * margin_x
    
    # Use pixel-based wrapping
    lines = wrap_text_by_pixels(text, font, draw, max_line_width)
    
    x = margin_x
    y = 10
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        draw.text((x, y), line, fill=text_color, font=font)
        y += line_height + 5
    
    image.save(output_path)
    print(f"Saved text image: {output_path}")

def render_sudoku_table(sudoku_str, output_path):
    """
    Renders a sudoku puzzle as a 9x9 grid on a black background.
    Expects sudoku_str to be at least 81 characters (row-wise).
    Blank cells should be represented by '0', '.', or spaces.
    """
    sudoku_str = sudoku_str.strip()
    if len(sudoku_str) < 81:
        sudoku_str = sudoku_str.ljust(81, '0')
    else:
        sudoku_str = sudoku_str[:81]
    
    cell_size = 50
    grid_size = cell_size * 9
    bg_color = "black"
    line_color = "#FFFFFF"  # white grid lines
    text_color = "#FFA500"  # orange digits
    
    image = Image.new("RGB", (grid_size+1, grid_size+1), bg_color)
    draw = ImageDraw.Draw(image)
    
    font_size = 32
    if os.path.exists(HAUNTED_FONT_DIR):
        font_files = [os.path.join(HAUNTED_FONT_DIR, f) for f in os.listdir(HAUNTED_FONT_DIR) if f.endswith(".ttf")]
    else:
        print(f"Haunted font directory not found: {HAUNTED_FONT_DIR}. Using default font.")
        font_files = []
    if font_files:
        font_path = random.choice(font_files)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # Draw grid lines
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        y_line = i * cell_size
        draw.line([(0, y_line), (grid_size, y_line)], fill=line_color, width=thickness)
        x_line = i * cell_size
        draw.line([(x_line, 0), (x_line, grid_size)], fill=line_color, width=thickness)
    
    # Place digits
    for row in range(9):
        for col in range(9):
            idx = row * 9 + col
            ch = sudoku_str[idx]
            if ch in ['0', '.', ' ']:
                continue
            x_center = col * cell_size + cell_size // 2
            y_center = row * cell_size + cell_size // 2
            bbox = draw.textbbox((0,0), ch, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            draw.text((x_center - text_w/2, y_center - text_h/2), ch, fill=text_color, font=font)
    
    image.save(output_path)
    print(f"Saved sudoku image: {output_path}")

##############################################
# 5. Generation Routine: Generate Images for Riddles and Sudoku
##############################################
def generate_dataset_images():
    # Create output subfolders.
    riddles_out_dir = os.path.join(OUTPUT_BASE_DIR, "riddles")
    sudoku_out_dir = os.path.join(OUTPUT_BASE_DIR, "sudoku")
    for d in [riddles_out_dir, sudoku_out_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Riddles: load from Riddles.csv (using load_riddles_from_csv)
    riddles_texts = load_riddles_from_csv(RIDDLES_FILE)
    print(f"Generating images for {len(riddles_texts)} riddles from Riddles.csv...", flush=True)
    for i, text in enumerate(riddles_texts):
        out_file = os.path.join(riddles_out_dir, f"riddle_{i+1}.png")
        render_text_image(text, out_file, wrap_width=50)
    
    # Sudoku: load all puzzles from sudoku.csv and generate image for each puzzle.
    sudoku_list = load_sudoku_from_csv(SUDOKU_FILE)
    print(f"Generating images for {len(sudoku_list)} sudoku puzzles...", flush=True)
    for i, puzzle in enumerate(sudoku_list):
        out_file = os.path.join(sudoku_out_dir, f"sudoku_{i+1}.png")
        render_sudoku_table(puzzle, out_file)

    print("All dataset images have been generated successfully.", flush=True)

##############################################
# Main Routine
##############################################
if __name__ == "__main__":
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    parser = argparse.ArgumentParser(
        description="Generate images for riddles and sudoku from CSV files."
    )
    parser.add_argument('--mode', type=str, default='all', choices=['data'],
                        help="Currently only 'data' mode is supported for dataset images.")
    args = parser.parse_args()
    
    generate_dataset_images()
