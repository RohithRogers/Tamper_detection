import hashlib
import torch
import numpy as np
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def verify_chunk_hashes(chunk_indices, blocks_msb_chunk, payloads_chunk, key_bytes):
    # blocks_msb_chunk: (chunk_size, 48) bytes
    # payloads_chunk: (chunk_size, 12) bytes
    
    results = []
    for idx in range(len(chunk_indices)):
        j = chunk_indices[idx]
        msb_bytes = blocks_msb_chunk[idx]
        payload = payloads_chunk[idx]
        
        extracted_hash = payload[1:] # 11 bytes
        latent_byte = payload[0]
        
        hasher = hashlib.sha256()
        hasher.update(msb_bytes)
        hasher.update(int(j).to_bytes(4, 'big'))
        hasher.update(key_bytes)
        digest = hasher.digest()
        recomputed_hash = digest[:11]
        
        tampered = (extracted_hash != recomputed_hash)
        results.append((j, tampered, latent_byte))
    return results

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

    print("Extracting payloads from LSBs (vectorized)...")
    blocks_np = blocks_uint8.numpy().reshape(num_blocks, 48)
    lsbs = blocks_np & 0x03
    lsbs_reshaped = lsbs.reshape(num_blocks, 12, 4)
    payloads_np = (lsbs_reshaped[:, :, 0] << 6) | (lsbs_reshaped[:, :, 1] << 4) | \
                  (lsbs_reshaped[:, :, 2] << 2) | lsbs_reshaped[:, :, 3]
    # payloads_np: (num_blocks, 12) uint8
    
    msb_list = [(blocks_np[j] & 0xFC).tobytes() for j in range(num_blocks)]
    key_bytes = key.encode()

    tamper_map = np.zeros(num_blocks, dtype=bool)
    latent_bytes_extracted = np.zeros(num_blocks, dtype=np.uint8)

    print("Verifying blocks (chunked)...")
    chunk_size = 20000
    num_chunks = (num_blocks + chunk_size - 1) // chunk_size
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for c in range(num_chunks):
            start = c * chunk_size
            end = min((c + 1) * chunk_size, num_blocks)
            
            chunk_indices = np.arange(start, end)
            msb_chunk = msb_list[start:end]
            payloads_chunk = [p.tobytes() for p in payloads_np[start:end]]
            
            futures.append(executor.submit(verify_chunk_hashes, chunk_indices, msb_chunk, payloads_chunk, key_bytes))
            
        for future in tqdm(futures, total=num_chunks):
            chunk_results = future.result()
            for j, tampered, latent_byte in chunk_results:
                tamper_map[j] = tampered
                latent_bytes_extracted[j] = latent_byte

    # Convert latent bytes back to bits for recovery
    # Each latent_byte is 8 bits for a recovery block i = inverse_mapping[j]
    print("Preparing latent bits for recovery...")
    extracted_latents = torch.zeros((num_blocks, 8), dtype=torch.float32)
    
    # Vectorize latent byte -> bits conversion
    # latent_bytes_extracted[j] contains latent for block i = inverse_mapping[j]
    # So extracted_latents[i] comes from latent_bytes_extracted[j]
    # bits = np.unpackbits(latent_bytes_extracted).reshape(num_blocks, 8)
    # Then map them to i
    all_bits = np.unpackbits(latent_bytes_extracted).reshape(num_blocks, 8).astype(np.float32)
    # extracted_latents[i] is latent for block i, which was stored in block j=mapping[i]
    # So we need to index all_bits with mapping? 
    # Let's re-read: block j stores latent for block i = inverse_mapping[j].
    # So i = inverse_mapping[j]. We want extracted_latents[i] = all_bits[j].
    # This means extracted_latents[inverse_mapping] = torch.from_numpy(all_bits).
    # Correct.
    extracted_latents[inverse_mapping] = torch.from_numpy(all_bits)

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

    device = next(model.parameters()).device
    model.eval()
    print("Recovering tampered blocks...")
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
                recovered_blocks[b_idx] = torch.zeros_like(recovered_blocks[b_idx])

    recovered_blocks_final = recovered_blocks.reshape(rows, cols, 3, 4, 4)
    out_recovered_tensor = recovered_blocks_final.permute(2, 0, 3, 1, 4).reshape(3, h, w)
    
    out_recovered_img = Image.fromarray(out_recovered_tensor.numpy().transpose(1, 2, 0), 'RGB')
    out_recovered_img.save(out_recovered_path, format="PNG")
    print(f"Recovered image saved to {out_recovered_path}")
