"""Draws jarvis.ico (the desktop shortcut and window icon). Run: python make_icon.py"""
from pathlib import Path

from PIL import Image, ImageDraw


def make_icon(path: Path | None = None) -> Path:
    path = path or Path(__file__).resolve().parent / "jarvis.ico"
    size = 256
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, size - 8, size - 8), fill=(15, 23, 42, 255), outline=(56, 189, 248, 255), width=14)
    d.ellipse((58, 58, size - 58, size - 58), outline=(125, 211, 252, 255), width=10)
    d.ellipse((100, 100, size - 100, size - 100), fill=(186, 230, 253, 255))  # glowing core
    img.save(path, sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    return path


if __name__ == "__main__":
    print(f"Wrote {make_icon()}")
