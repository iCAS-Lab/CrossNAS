import numpy as np
import numpy as np
import torch
import torch.nn as nn

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.maxpool(out)
        return out

class VGGBlock_Nomax(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(VGGBlock_Nomax, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)
        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out2 = self.conv3(residual)
        out2 = self.bn3(out2)
        out = out1 + out2
        out = self.relu(out)
        return out

class SinglePath_Network(nn.Module):
    def __init__(self, dataset, classes, layers, choice, kchoice):
        super(SinglePath_Network, self).__init__()
        self.classes = classes
        self.layers = layers
        self.kernel_list = [32, 64, 128]
        # choice_block
        self.fixed_block = nn.ModuleList([])
        for i,j in enumerate(choice):
            if i==0:
                if j==0:
                    self.fixed_block.append(VGGBlock(in_channels=3, out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                elif j==1:
                    self.fixed_block.append(VGGBlock_Nomax(in_channels=3, out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                elif j==2:
                    self.fixed_block.append(ResidualBlock(in_channels=3, out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                else:
                    kchoice[i] = kchoice[i-1]
                    continue
            else:
                if j==0:
                    self.fixed_block.append(VGGBlock(in_channels=self.kernel_list[kchoice[i-1]], out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                elif j==1:
                    self.fixed_block.append(VGGBlock_Nomax(in_channels=self.kernel_list[kchoice[i-1]], out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                elif j==2:
                    self.fixed_block.append(ResidualBlock(in_channels=self.kernel_list[kchoice[i-1]], out_channels=self.kernel_list[kchoice[i]], kernel_size=3))
                else:
                    kchoice[i] = kchoice[i-1]
                    continue

        self.global_pooling = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Linear(2*2*self.kernel_list[kchoice[layers-1]], 200)
        self.relu1 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(200, 50)
        self.relu2 = nn.ReLU(inplace=True)
        self.dp2 = nn.Dropout(p=0.2)
        self.out = nn.Linear(50, 10)
        #self._initialize_weights()


    def forward(self, x):
        # repeat
        for i in range(len(self.fixed_block)):
            x = self.fixed_block[i](x)
        #x = self.last_conv(x)
        x = self.global_pooling(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dp2(x)
        x = self.out(x)
        return x
