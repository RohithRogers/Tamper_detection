# Image Tamper Detection and Recovery

This project implements a self-embedding fragile watermarking system for image authentication and recovery using a Deep Learning Autoencoder and SHA-256 cryptographic hashing.

## 🚀 Key Features
- **Blind Verification**: Detects tampering **without** needing the original image.
- **Self-Recovery**: Reconstructs tampered regions using embedded 8-bit compressed latents.
- **High Sensitivity**: Detects even single-pixel modifications using SHA-256 hashes of block MSBs.
- **Secure Mapping**: Uses a secret key for pseudo-random block mapping, ensuring that recovery data is stored in distant blocks.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   cd tamper_detection
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision pillow numpy matplotlib
   ```

3. **Check for Pre-trained Model**:
   Ensure `models/autoencoder_8bit.pth` exists. If not, train it using `train_autoencoder.py`.

## 📖 Usage

### 1. Watermark an Image (Embed)
Embeds authentication and recovery data into the LSBs of an image.
```bash
python main.py embed -i input.png -o watermarked.png
```

### 2. Verify and Recover (Verify)
Detects tampered regions and attempts to recover them.
```bash
python main.py verify -i suspect.png -o tamper_map.png -r recovered.png
```

- `-i`: Path to the suspect/watermarked image.
- `-o`: Output path for the binary tamper map (white = tampered).
- `-r`: Output path for the recovered image.

### 3. Retrain the Autoencoder (Optional)
If you want to train the model on a different dataset:
```bash
python train_autoencoder.py
```

## 📂 Project Structure
- `core/autoencoder.py`: Neural network for block compression.
- `core/watermark.py`: Watermark embedding logic (LSB, Hashing).
- `core/verification.py`: Tamper detection and reconstruction.
- `main.py`: Main CLI entry point.
- `models/`: Contains pre-trained weights.
- `run_pipeline.py`: Comprehensive test script for the entire flow.

## 📸 Results & Example Images

### Original Image
![Original Image](test_images/tree.jpg)

### Watermarked Image
![Watermarked Image](test_images/treenew.png)

### Tampered Image
![Tampered Image](test_images/suspect.png)

### Tamper Detection Map
![Tamper Detection Map](test_images/tampered_map.png)

### Recovered Image(s)
![Recovered Image](test_images/recovered.png)

---
**Note**: This system is designed for **blind detection**. The original image is **NOT** required for verification or recovery, as all necessary data is embedded within the watermarked image itself.
