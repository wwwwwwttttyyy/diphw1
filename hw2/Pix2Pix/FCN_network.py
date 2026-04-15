import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # originally: 3 -> 8
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        ### FILL: add more CONV Layers,训练一次时间要半天,,新参数没有训练,目前的数据是老参数的,有过拟合
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # originally: 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # originally: 16 -> 32
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # originally: 32 -> 64
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # originally: not present
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # originally: 64 -> 32
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # originally: 32 -> 16
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # originally: 16 -> 8
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # originally: 8 -> 3
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.deconv5 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # originally: not present
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder forward pass
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        # Decoder forward pass
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        
        ### FILL: encoder-decoder forward pass

        output = x
        
        return output
    
