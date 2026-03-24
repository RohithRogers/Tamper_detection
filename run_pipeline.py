import os
import subprocess
from PIL import Image, ImageDraw

print("--- Running Embed ---")
res = subprocess.run(["python", "main.py", "embed", "-i", "../hares_new.jpg", "-o", "watermarked_hares.png"], capture_output=True, text=True)
print("STDOUT:", res.stdout)
print("STDERR:", res.stderr)

print("--- Tampering ---")
try:
    img = Image.open('watermarked_hares.png').convert('RGB')
    draw = ImageDraw.Draw(img)
    # Draw a 100x100 black rectangle at (50, 50)
    draw.rectangle([50, 50, 150, 150], fill=(0, 0, 0))
    img.save('suspect_hares.png')
    print("Tampered image saved as suspect_hares.png")
except Exception as e:
    print(f"Failed to tamper: {e}")

print("--- Running Verify ---")
res = subprocess.run(["python", "main.py", "verify", "-i", "suspect_hares.png", "-o", "tamper_map.png", "-r", "recovered_hares.png"], capture_output=True, text=True)
print(res.stdout)
if res.stderr: print("ERR:", res.stderr)
