from torch import nn
from torchvision import transforms

class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim,fc2_input_dim):
        super().__init__()
        
        self.transform = transforms.Compose([
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(2, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(
            nn.Linear(3 * 3 * 32, 128),
            nn.ReLU(True),
            nn.Linear(128, encoded_space_dim),
            nn.ReLU(True),
            nn.Linear(encoded_space_dim, encoded_space_dim)
        )
        
    def forward(self, x):
        x = self.transform(x)
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x
