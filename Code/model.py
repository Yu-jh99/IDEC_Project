import torch.nn as nn

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # (B,64,128,128)
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # (B,128,64,64)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # (B,256,32,32)
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Sigmoid()  # 0~1 값으로 복원
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

