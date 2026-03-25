import hashlib
import torch
import sys
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def compute_chunk_hashes(chunk_indices, blocks_msb_bytes, key_bytes):
    results = []
    # Pre-encode key and constant parts
    for idx in range(len(chunk_indices)):
        j = chunk_indices[idx]
        msb_bytes = blocks_msb_bytes[idx]
        
        hasher = hashlib.sha256()
        hasher.update(msb_bytes)
        hasher.update(int(j).to_bytes(4, 'big'))
        hasher.update(key_bytes)
        digest = hasher.digest()
        results.append(digest[:11])
    return results

def embed_watermark(image_path, out_path, model, key="secret"):
    sys.stderr.write(f"DEBUG: Entering embed_watermark with {image_path}\n")
    sys.stderr.flush()
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img) 

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
    
    latents = latents_discrete.cpu()

    seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    np.random.seed(seed)
    mapping = np.arange(num_blocks)
    np.random.shuffle(mapping)

    inverse_mapping = np.zeros(num_blocks, dtype=int)
    for i, j in enumerate(mapping):
        inverse_mapping[j] = i

    blocks_uint8 = (blocks * 255.0).clamp(0, 255).to(torch.uint8)
    blocks_np = blocks_uint8.numpy().reshape(num_blocks, 48)
    blocks_msb_bytes = (blocks_np & 0xFC).tobytes() # All blocks' MSBs in one go
    # Split into list of 48-byte chunks for faster access in the loop
    msb_list = [blocks_msb_bytes[i:i+48] for i in range(0, len(blocks_msb_bytes), 48)]

    # Fully vectorized latent byte calculation
    print("Preparing recovery latents...")
    latents_np = latents.numpy().astype(np.uint8)
    weights = np.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=np.uint8)
    latent_bytes = np.sum(latents_np * weights, axis=1).astype(np.uint8) # (num_blocks,)

    # Hashing is the remaining bottleneck
    print("Calculating authentication hashes (chunked)...")
    all_hashes = [None] * num_blocks
    key_bytes = key.encode()
    
    chunk_size = 20000
    num_chunks = (num_blocks + chunk_size - 1) // chunk_size
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, num_blocks)
            
            chunk_indices = np.arange(start, end)
            msb_chunk = msb_list[start:end]
            
            futures.append(executor.submit(compute_chunk_hashes, chunk_indices, msb_chunk, key_bytes))
        
        for idx, future in enumerate(tqdm(futures, total=num_chunks)):
            chunk_hashes = future.result()
            start = idx * chunk_size
            end = start + len(chunk_hashes)
            all_hashes[start:end] = chunk_hashes

    print("Embedding watermarks into LSBs...")
    # Partner latents (j must store latent for mapping[i] where j=mapping[i]? No, i=inverse_mapping[j])
    # The partner for block j is block i = inverse_mapping[j]
    # So block j stores latent_bytes[i]
    partner_latents = latent_bytes[inverse_mapping]
    
    # Combined payload: partner_latent (1 byte) + authentication_hash (11 bytes)
    # all_hashes is list of 11-byte bytes objects
    # This part is also a bit slow in Python if not vectorized
    payload_arrays = []
    for j in range(num_blocks):
        payload_arrays.append(np.frombuffer(partner_latents[j].tobytes() + all_hashes[j], dtype=np.uint8))
    all_payloads = np.stack(payload_arrays) # (num_blocks, 12)

    payload_bits = np.unpackbits(all_payloads, axis=1).reshape(num_blocks, 48, 2)
    payload_2bits = (payload_bits[:, :, 0] << 1) | payload_bits[:, :, 1] # (num_blocks, 48)
    
    blocks_np = (blocks_np & 0xFC) | payload_2bits
    
    watermarked_blocks = torch.from_numpy(blocks_np).reshape(num_blocks, 3, 4, 4)

    rows = h // 4
    cols = w // 4
    watermarked_blocks = watermarked_blocks.reshape(rows, cols, 3, 4, 4)
    watermarked_blocks = watermarked_blocks.permute(2, 0, 3, 1, 4).reshape(3, h, w)
    
    out_img = Image.fromarray(watermarked_blocks.permute(1, 2, 0).numpy(), 'RGB')
    out_img.save(out_path, format="PNG")
    print(f"Watermarked image saved to {out_path}")
