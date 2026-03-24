import torch

h, w = 400, 400
tensor = torch.randn(3, h, w)
blocks = tensor.unfold(1, 4, 4).unfold(2, 4, 4)
print(f"Blocks shape after unfold: {blocks.shape}")

# Current line 21 logic
blocks_v1 = blocks.contiguous().view(3, -1, 4, 4).permute(1, 0, 2, 3)
print(f"Blocks shape after view/permute: {blocks_v1.shape}")

# Reconstruct
rows, cols = h // 4, w // 4
num_blocks = blocks_v1.shape[0]
watermarked_blocks = blocks_v1.clone()

# Line 80-82 logic
print(f"Before view: {watermarked_blocks.shape}")
watermarked_blocks_v2 = watermarked_blocks.view(rows, cols, 3, 4, 4)
print(f"After view: {watermarked_blocks_v2.shape}")
watermarked_blocks_v3 = watermarked_blocks_v2.permute(2, 0, 3, 1, 4).contiguous()
print(f"After permute: {watermarked_blocks_v3.shape}")
out_img_tensor = watermarked_blocks_v3.view(3, h, w)
print(f"Final tensor shape: {out_img_tensor.shape}")
print("SUCCESS!")
