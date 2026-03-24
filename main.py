import argparse
import sys
import torch
from core.autoencoder import BlockAutoencoder
from core.watermark import embed_watermark
from core.verification import verify_and_recover

import core.watermark
print(f"DEBUG: core.watermark file: {core.watermark.__file__}")

def main():
    parser = argparse.ArgumentParser(description="Image Tamper Detection and Recovery Pipeline")
    parser.add_argument("mode", choices=["embed", "verify"], help="Mode: embed or verify")
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output image/map path")
    parser.add_argument("-r", "--recovered", help="Recovered image path (only for verify mode)", default="recovered.png")
    parser.add_argument("-k", "--key", default="secret", help="Secret key for mapping and hashing")
    parser.add_argument("-m", "--model", default="models/autoencoder_8bit.pth", help="Path to Pre-trained Autoencoder Model weights")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BlockAutoencoder(block_size=4, channels=3, latent_bits=8).to(device)
    
    try:
        model.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
        print(f"Loaded Autoencoder from {args.model}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"DEBUG: Mode={args.mode}, Input={args.input}, Output={args.output}")
    sys.stdout.flush()

    import traceback
    try:
        if args.mode == "embed":
            embed_watermark(args.input, args.output, model, key=args.key)
        elif args.mode == "verify":
            verify_and_recover(args.input, args.output, args.recovered, model, key=args.key)
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    main()
