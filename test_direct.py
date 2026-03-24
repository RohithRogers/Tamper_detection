import torch
from core.watermark import embed_watermark
from core.autoencoder import BlockAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BlockAutoencoder(block_size=4, channels=3, latent_bits=8).to(device)
# Try to load if exists, else it might just fail at forward pass, which is fine for shape testing
try:
    model.load_state_dict(torch.load("models/autoencoder_8bit.pth", map_location=device, weights_only=True))
except:
    pass

print("Starting direct embed test...")
embed_watermark("../hares_new.jpg", "test_out.png", model)
print("Finished direct embed test.")
