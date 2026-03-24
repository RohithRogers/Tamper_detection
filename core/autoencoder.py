import torch
import torch.nn as nn

class BinarizeSTE(torch.autograd.Function):
    """
    Binarization with Straight-Through Estimator (STE).
    Forward pass: threshold at 0.5 to output 0 or 1.
    Backward pass: identity matrix, allowing gradients to flow through.
    """
    @staticmethod
    def forward(ctx, x):
        return (x > 0.5).float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class BlockAutoencoder(nn.Module):
    def __init__(self, block_size=4, channels=3, latent_bits=8):
        super(BlockAutoencoder, self).__init__()
        self.block_size = block_size
        self.channels = channels
        self.input_dim = block_size * block_size * channels
        self.latent_bits = latent_bits
        
        # Lightweight Encoder
        # 48 -> 32 -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.latent_bits),
            nn.Sigmoid() # Push values between 0 and 1
        )
        
        # Lightweight Decoder
        # 8 -> 16 -> 32 -> 48
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_bits, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, self.input_dim),
            nn.Sigmoid() # Reconstruct pixel values in [0, 1] range
        )

    def encode(self, x):
        """
        x: (Batch, Channels, H, W) where H=W=block_size
        Returns: (Batch, latent_bits) of discrete 0 / 1 values.
        """
        # Flatten the blocks
        b = x.size(0)
        x_flat = x.reshape(b, -1)
        
        # Forward through encoder
        z_continuous = self.encoder(x_flat)
        
        # Binarize to exact 8 bits (0 or 1) using STE
        z_discrete = BinarizeSTE.apply(z_continuous)
        
        return z_continuous, z_discrete

    def decode(self, z):
        """
        z: (Batch, latent_bits)
        Returns: (Batch, Channels, block_size, block_size)
        """
        b = z.size(0)
        x_hat_flat = self.decoder(z)
        # Ensure dimensions match requested Shape (N, C, H, W)
        x_hat = x_hat_flat.view(b, self.channels, self.block_size, self.block_size)
        return x_hat

    def forward(self, x):
        z_continuous, z_discrete = self.encode(x)
        # In training, reconstruct from the discrete bits
        # to ensure the decoder learns to handle 8-bit quantized inputs
        x_hat = self.decode(z_discrete)
        return x_hat, z_discrete, z_continuous
