import os
from PIL import Image, ImageDraw

def generate_icons():
    src_path = "C:/Users/robin/.gemini/antigravity/brain/36288f45-1c77-4c4b-a035-241d03dd62c6/.user_uploaded/media_1790062603140.jpg"
    img = Image.open(src_path).convert("RGBA")
    
    # Flood-fill outer white canvas with transparency from all four corners
    w, h = img.size
    ImageDraw.floodfill(img, (0, 0), (0, 0, 0, 0), thresh=25)
    ImageDraw.floodfill(img, (w - 1, 0), (0, 0, 0, 0), thresh=25)
    ImageDraw.floodfill(img, (0, h - 1), (0, 0, 0, 0), thresh=25)
    ImageDraw.floodfill(img, (w - 1, h - 1), (0, 0, 0, 0), thresh=25)

    # Destination directories
    public_dir = os.path.join("frontend", "public")
    electron_dir = os.path.join("frontend", "electron")
    assets_dir = os.path.join("frontend", "src", "assets")

    os.makedirs(public_dir, exist_ok=True)
    os.makedirs(electron_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. High-res 512x512 PNG
    img_512 = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_512.save(os.path.join(public_dir, "icon.png"), format="PNG")
    img_512.save(os.path.join(electron_dir, "icon.png"), format="PNG")
    img_512.save(os.path.join(assets_dir, "logo.png"), format="PNG")

    # 2. Multi-resolution ICO (16, 24, 32, 48, 64, 128, 256)
    ico_sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    img.save(os.path.join(public_dir, "icon.ico"), format="ICO", sizes=ico_sizes)
    img.save(os.path.join(electron_dir, "icon.ico"), format="ICO", sizes=ico_sizes)
    
    print("Successfully generated icon.png, icon.ico, and logo.png in public, electron, and assets folders.")

if __name__ == "__main__":
    generate_icons()
