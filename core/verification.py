import hashlib
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image

def verify_and_recover(image_path, out_map_path, out_recovered_path, model, key="secret"):
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img)
    _, h, w = tensor.shape
    
    blocks = tensor.unfold(1, 4, 4).unfold(2, 4, 4)
    blocks = blocks.contiguous().view(3, -1, 4, 4).permute(1, 0, 2, 3)
    num_blocks = blocks.shape[0]
    
    blocks_uint8 = (blocks * 255.0).clamp(0, 255).to(torch.uint8)

    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    mapping = np.arange(num_blocks)
    np.random.shuffle(mapping)

    inverse_mapping = np.zeros(num_blocks, dtype=int)
    for i, j in enumerate(mapping):
        inverse_mapping[j] = i

    tamper_map = np.zeros(num_blocks, dtype=bool)
    extracted_latents = torch.zeros((num_blocks, 8), dtype=torch.float32)

    for j in range(num_blocks):
        block = blocks_uint8[j]
        block_flat = block.reshape(-1)
        
        payload_bytes = bytearray(12)
        for byte_idx in range(12):
            part0 = block_flat[byte_idx*4 + 0] & 0x3
            part1 = block_flat[byte_idx*4 + 1] & 0x3
            part2 = block_flat[byte_idx*4 + 2] & 0x3
            part3 = block_flat[byte_idx*4 + 3] & 0x3
            
            val = (part0 << 6) | (part1 << 4) | (part2 << 2) | part3
            payload_bytes[byte_idx] = val
            
        latent_byte = payload_bytes[0]
        hash_88bits_extracted = bytes(payload_bytes[1:])
        
        block_msb = block & 0xFC
        hasher = hashlib.sha256()
        hasher.update(block_msb.numpy().tobytes())
        hasher.update(j.to_bytes(4, 'big'))
        hasher.update(key.encode())
        digest = hasher.digest()
        hash_88bits_recomputed = digest[:11]
        
        if hash_88bits_extracted != hash_88bits_recomputed:
            tamper_map[j] = True
            
        i = inverse_mapping[j]
        for bit_idx in range(8):
            bit_val = (latent_byte >> (7 - bit_idx)) & 1
            extracted_latents[i, bit_idx] = float(bit_val)

    rows = h // 4
    cols = w // 4
    tamper_map_2d = tamper_map.reshape(rows, cols)
    
    try:
        import matplotlib.pyplot as plt
        plt.imsave(out_map_path, tamper_map_2d, cmap='gray')
        print(f"Tamper map saved to {out_map_path}")
    except ImportError:
        map_img = Image.fromarray((tamper_map_2d * 255).astype(np.uint8), mode='L')
        map_img.save(out_map_path)
        print(f"Tamper map saved to {out_map_path}")

    # Recovery
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        recovered_blocks_tensor = model.decode(extracted_latents.to(device))
        
    recovered_blocks_tensor = (recovered_blocks_tensor.cpu() * 255.0).clamp(0, 255).to(torch.uint8)
    recovered_blocks = blocks_uint8.clone()
    
    for b_idx in range(num_blocks):
        j = mapping[b_idx]
        if tamper_map[b_idx]: 
            if not tamper_map[j]:
                recovered_blocks[b_idx] = recovered_blocks_tensor[b_idx]
            else:
                # Fill with black block indicating unrecoverable
                recovered_blocks[b_idx] = torch.zeros_like(recovered_blocks[b_idx])

    recovered_blocks_final = recovered_blocks.reshape(rows, cols, 3, 4, 4)
    out_recovered_tensor = recovered_blocks_final.permute(2, 0, 3, 1, 4).reshape(3, h, w)
    
    out_recovered_img = Image.fromarray(out_recovered_tensor.numpy().transpose(1, 2, 0), 'RGB')
    out_recovered_img.save(out_recovered_path, format="PNG")
    print(f"Recovered image saved to {out_recovered_path}")

