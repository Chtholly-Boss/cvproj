import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

def get_net(num_classes=21):
    pretrained_net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    num_classes = 21
    net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                        kernel_size=64, padding=16, stride=32))
    W = bilinear_kernel(num_classes, num_classes, 64)
    net.transpose_conv.weight.data.copy_(W)
    return net

class FCN(nn.Module):
    def __init__(self, num_classes=21, input_channels=3):
        super().__init__()
        
        # Stage 1: Downsample to 1/4 of original size (320x320 -> 80x80)
        self.stage1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 320x320 -> 160x160
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 160x160 -> 80x80
        )
        
        # Stage 2: Downsample to 1/2 of previous (80x80 -> 40x40)
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 80x80 -> 40x40
        )
        
        # Stage 3: Downsample to 1/8 of previous (40x40 -> 10x10)
        # This makes it 1/32 of the original size
        self.stage3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 40x40 -> 20x20
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)   # 20x20 -> 10x10
        )
        
        # Reduce channels before upsampling
        self.conv_out = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Transposed convolution to upsample back to original size (10x10 -> 320x320)
        self.trans_conv = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=64, stride=32, padding=16
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        # Initialize convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for convolutional layers
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            
            # Special initialization for transposed convolution
            elif isinstance(m, nn.ConvTranspose2d):
                # Xavier initialization works well for transposed convolutions
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
    
    def forward(self, x: Tensor) -> Tensor:
        # Store original size
        input_size = x.size()[2:]
        
        # Downsampling path
        x = self.stage1(x)  # 320x320 -> 80x80
        x = self.stage2(x)  # 80x80 -> 40x40
        x = self.stage3(x)  # 40x40 -> 10x10
        
        # Apply 1x1 convolution to get class predictions
        x = self.conv_out(x)
        
        # Upsample back to original size
        x = self.trans_conv(x)
        
        # Ensure output size matches input size exactly
        if x.size()[2:] != input_size:
            x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=True)
        
        return x