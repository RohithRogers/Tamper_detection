import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from glob import glob
from core.autoencoder import BlockAutoencoder

# Create a dataset for training the autoencoder on random 4x4 patches
class ImagePatchDataset(Dataset):
    def __init__(self, image_paths, num_samples_per_image=100, block_size=4):
        self.image_paths = image_paths
        self.num_samples_per_image = num_samples_per_image
        self.block_size = block_size
        
        self.valid_images = []
        for p in self.image_paths: 
            try:
                # Basic check to avoid corrupted files
                img = Image.open(p)
                img.verify()
                self.valid_images.append(p)
            except Exception:
                pass
                
        print(f"Found {len(self.valid_images)} valid images for training.")

    def __len__(self):
        return len(self.valid_images) * self.num_samples_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.num_samples_per_image
        img_path = self.valid_images[img_idx]
        
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        
        # Sample a random 4x4 patch
        if w <= self.block_size or h <= self.block_size:
            patch = img.resize((self.block_size, self.block_size))
        else:
            x = np.random.randint(0, w - self.block_size)
            y = np.random.randint(0, h - self.block_size)
            patch = img.crop((x, y, x + self.block_size, y + self.block_size))
            
        # Convert to tensor [0, 1]
        patch_np = np.array(patch).astype(np.float32) / 255.0
        # HWC -> CHW
        patch_tensor = torch.from_numpy(patch_np).permute(2, 0, 1)
        
        return patch_tensor

def train():
    # 1. Gather some images for training
    # Looking into the parent directory for any images to train the autoencoder
    data_dir = r"d:\Copy_Move_Forgery"
    image_paths = glob(os.path.join(data_dir, "**", "*.jpg"), recursive=True) + \
                  glob(os.path.join(data_dir, "**", "*.png"), recursive=True)
                  
    if not image_paths:
        print("No images found! Using random noise as fallback (only for testing the script)...")
        image_paths = ["dummy"]
        
        class DummyDataset(Dataset):
            def __len__(self): return 1000
            def __getitem__(self, idx): return torch.rand(3, 4, 4)
        dataset = DummyDataset()
    else:
        dataset = ImagePatchDataset(image_paths, num_samples_per_image=1000)
        
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BlockAutoencoder(block_size=4, channels=3, latent_bits=8).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    epochs = 10
    print(f"Starting training for {epochs} epochs...")
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            
            optimizer.zero_grad()
            x_hat, z_discrete, z_continuous = model(batch)
            loss = criterion(x_hat, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/autoencoder_8bit.pth")
    print("Training complete! Model saved to models/autoencoder_8bit.pth")

if __name__ == "__main__":
    train()
