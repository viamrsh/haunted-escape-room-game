# D:\project\haunted_escape_room_backend\generate_rooms.py

import os
import random
from PIL import Image, ImageDraw, ImageFont

# List of colors for rendering text.
COLOR_CODES = [
    "#FF0000", "#DC143C", "#B22222", "#FFD700", "#FFDF00",
    "#DAA520", "#FF8C00", "#FF4500", "#FF6347", "#00FF7F"
]

def render_text_image(text, image_size=(800, 600), font_size=24, bg_color="white"):
    """
    Render the given multi-line text onto an image.
    Each line is drawn in a random color.
    """
    image = Image.new("RGB", image_size, bg_color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    lines = text.split("\n")
    y = 10
    for line in lines:
        color = random.choice(COLOR_CODES)
        draw.text((10, y), line, fill=color, font=font)
        y += font.getsize(line)[1] + 5
    return image

def generate_room_assets(root_folder="generated_rooms", num_rooms=50, fighting_room_ids=None):
    """
    Generate assets for each room. For each of the num_rooms, create a folder that contains:
    - room_info.txt: Text file with room details.
    - room_info.png: An image rendering the room details.
    - Additional asset images: atmosphere.png, npc_ghost.png, halloween.png, doors.png, texts.png.
    - For fighting rooms, a weapon.png image is generated.
    """
    if fighting_room_ids is None:
        # Designate the first 10 rooms as fighting rooms.
        fighting_room_ids = set(range(1, 11))
    
    os.makedirs(root_folder, exist_ok=True)
    
    for room_id in range(1, num_rooms + 1):
        room_folder = os.path.join(root_folder, f"Room_{room_id}")
        os.makedirs(room_folder, exist_ok=True)
        
        welcome = f"Welcome to Room {room_id}"
        if room_id in fighting_room_ids:
            fight_points = random.randint(50, 100)
            key_location = "Hidden under the broken door"
        else:
            fight_points = "N/A"
            key_location = "On the antique table"
        lifetime = random.randint(1, 100)
        conversation = "Ghost: Beware the darkness!\nPlayer: I will find the key."
        
        room_details = (
            f"Room Number: {room_id}\n"
            f"{welcome}\n"
            f"Fight Score Points: {fight_points}\n"
            f"Key Location: {key_location}\n"
            f"Lifetime: {lifetime}\n"
            f"Conversation:\n{conversation}\n"
        )
        
        # Save text details to a file.
        text_file = os.path.join(room_folder, "room_info.txt")
        with open(text_file, "w") as f:
            f.write(room_details)
        
        # Generate an image of the text details.
        text_image = render_text_image(room_details, image_size=(800, 600), font_size=24)
        text_image.save(os.path.join(room_folder, "room_info.png"))
        
        # Generate additional dummy asset images.
        asset_names = ["atmosphere", "npc_ghost", "halloween", "doors", "texts"]
        for asset in asset_names:
            asset_text = f"Room {room_id} {asset.capitalize()} Asset"
            asset_image = render_text_image(asset_text, image_size=(400, 300), font_size=20)
            asset_image.save(os.path.join(room_folder, f"{asset}.png"))
        
        # For fighting rooms, generate a weapon image.
        if room_id in fighting_room_ids:
            weapon_text = f"Room {room_id} Weapon Asset"
            weapon_image = render_text_image(weapon_text, image_size=(300, 300), font_size=20)
            weapon_image.save(os.path.join(room_folder, "weapon.png"))
        
        print(f"Assets generated for Room {room_id}")

if __name__ == "__main__":
    generate_room_assets()
