import hashlib
import torch
import sys
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

def embed_watermark(image_path, out_path, model, key="secret"):
    sys.stderr.write(f"DEBUG: Entering embed_watermark with {image_path}\n")
    sys.stderr.flush()
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img) # [0, 1] Range

    _, h, w = tensor.shape
    new_h = h if h % 4 == 0 else h + (4 - h % 4)
    new_w = w if w % 4 == 0 else w + (4 - w % 4)
    
    if new_h != h or new_w != w:
        pad_transform = torch.nn.ZeroPad2d((0, new_w - w, 0, new_h - h))
        tensor = pad_transform(tensor)
        h, w = new_h, new_w

    blocks = tensor.unfold(1, 4, 4).unfold(2, 4, 4) 
    blocks = blocks.contiguous().view(3, -1, 4, 4).permute(1, 0, 2, 3) 
    num_blocks = blocks.shape[0]
    print(f"DEBUG: h={h}, w={w}, blocks.shape={blocks.shape}, num_blocks={num_blocks}")
    sys.stdout.flush()
    
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        _, latents_discrete = model.encode(blocks.to(device))
    
    latents = latents_discrete.cpu().to(torch.uint8)

    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    mapping = np.arange(num_blocks)
    np.random.shuffle(mapping)

    inverse_mapping = np.zeros(num_blocks, dtype=int)
    for i, j in enumerate(mapping):
        inverse_mapping[j] = i

    blocks_uint8 = (blocks * 255.0).clamp(0, 255).to(torch.uint8)
    watermarked_blocks = blocks_uint8.clone()

    for j in range(num_blocks):
        block = blocks_uint8[j]
        block_msb = block & 0xFC
        
        hasher = hashlib.sha256()
        hasher.update(block_msb.numpy().tobytes())
        hasher.update(j.to_bytes(4, 'big'))
        hasher.update(key.encode())
        digest = hasher.digest() 
        hash_88bits = digest[:11] 

        i = inverse_mapping[j]
        latent_8bits = latents[i] 
        latent_byte = 0
        for bit_idx in range(8):
            latent_byte |= (latent_8bits[bit_idx].item() << (7 - bit_idx))

        payload_bytes = bytearray([latent_byte]) + bytearray(hash_88bits)
        
        block_flat = block_msb.view(-1) 
        for byte_idx in range(12):
            val = payload_bytes[byte_idx]
            part0 = (val >> 6) & 0x3
            part1 = (val >> 4) & 0x3
            part2 = (val >> 2) & 0x3
            part3 = (val >> 0) & 0x3
            
            block_flat[byte_idx*4 + 0] |= part0
            block_flat[byte_idx*4 + 1] |= part1
            block_flat[byte_idx*4 + 2] |= part2
            block_flat[byte_idx*4 + 3] |= part3
            
        watermarked_blocks[j] = block_flat.view(3, 4, 4)

    rows = h // 4
    cols = w // 4
    
    sys.stderr.write(f"DEBUG: Before reshape, watermarked_blocks.shape={watermarked_blocks.shape}\n")
    sys.stderr.flush()
    watermarked_blocks = watermarked_blocks.reshape(rows, cols, 3, 4, 4)
    sys.stderr.write(f"DEBUG: After reshape, watermarked_blocks.shape={watermarked_blocks.shape}\n")
    sys.stderr.flush()
    watermarked_blocks = watermarked_blocks.permute(2, 0, 3, 1, 4).contiguous()
    sys.stderr.write(f"DEBUG: After permute, watermarked_blocks.shape={watermarked_blocks.shape}\n")
    sys.stderr.flush()
    out_img_tensor = watermarked_blocks.reshape(3, h, w)
    sys.stderr.write(f"DEBUG: After final reshape, out_img_tensor.shape={out_img_tensor.shape}\n")
    sys.stderr.flush()
    
    out_img = Image.fromarray(out_img_tensor.permute(1, 2, 0).numpy(), 'RGB')
    out_img.save(out_path, format="PNG")
    sys.stderr.write(f"Watermarked image saved to {out_path}\n")
    sys.stderr.flush()
