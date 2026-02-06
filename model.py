import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # First convolution block (encoder)
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Pool to downsample the feature map (100x100 -> 50x50)
        self.pool1 = nn.MaxPool2d(2)
        
        # Second convolution block (encoder)
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Transposed convolution to upsample the feature map (50x50 -> 100x100)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        # Final convolution to produce the output with one channel
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # x has shape [B, 1, 100, 100]
        x1 = self.down1(x)       # Output shape: [B, 64, 100, 100]
        x2 = self.pool1(x1)      # Downsampling: [B, 64, 50, 50]
        x3 = self.down2(x2)      # [B, 128, 50, 50]
        x4 = self.up1(x3)        # Upsampling: [B, 64, 100, 100]
        out = self.final(x4)     # Final output: [B, 1, 100, 100]
        return out

if __name__ == '__main__':
    model = UNet()
    sample_input = torch.randn(1, 1, 100, 100)
    output = model(sample_input)
    print("Input shape:", sample_input.shape, "Output shape:", output.shape)