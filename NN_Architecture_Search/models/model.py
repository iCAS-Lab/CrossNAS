import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x, kin, kout):
        out = F.conv2d(x, self.conv1.weight[:kout,:kin,:,:], self.conv1.bias[:kout], padding=1)
        out = F.batch_norm(out, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training = True)
        out = self.relu1(out)
        out = F.conv2d(out, self.conv2.weight[:kout,:kout,:,:], self.conv2.bias[:kout], padding=1)
        out = F.batch_norm(out, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training = True)
        out = self.relu2(out)
        out = self.maxpool(out)
        return out

class VGGBlock_Nomax(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(VGGBlock_Nomax, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, kin, kout):
        out = F.conv2d(x, self.conv1.weight[:kout,:kin,:,:], self.conv1.bias[:kout], padding=1)
        out = F.batch_norm(out, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training = True)
        out = self.relu1(out)
        out = F.conv2d(out, self.conv2.weight[:kout,:kout,:,:], self.conv2.bias[:kout], padding=1)
        out = F.batch_norm(out, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training = True)
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

    def forward(self, x, kin, kout):
        residual = x
        out1 = F.conv2d(x, self.conv1.weight[:kout,:kin,:,:], padding=1)
        out1 = F.batch_norm(out1, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training = True)
        out1 = self.relu(out1)
        out1 = F.conv2d(out1, self.conv2.weight[:kout,:kout,:,:], padding=1)
        out1 = F.batch_norm(out1, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training = True)
        out2 = F.conv2d(residual, self.conv3.weight[:kout,:kin,:,:])
        out2 = F.batch_norm(out2, self.bn3.running_mean[:kout], self.bn3.running_var[:kout], self.bn3.weight[:kout], self.bn3.bias[:kout], training = True)
        out = out1 + out2
        out = self.relu(out)
        return out


class SinglePath_Search(nn.Module):
    def __init__(self, dataset, classes, layers):
        super(SinglePath_Search, self).__init__()
        self.classes = classes
        self.layers = layers
        self.kernel_list = [32, 64, 128]
        # choice_block
        self.fixed_block = nn.ModuleList([])
        for i in range(layers):
            layer_cb = nn.ModuleList([])
            if i==0:
                layer_cb.append(VGGBlock(in_channels=3, out_channels=128, kernel_size=3))
                layer_cb.append(VGGBlock_Nomax(in_channels=3, out_channels=128, kernel_size=3))
                layer_cb.append(ResidualBlock(in_channels=3, out_channels=128, kernel_size=3))
            else:
                layer_cb.append(VGGBlock(in_channels=128, out_channels=128, kernel_size=3))
                layer_cb.append(VGGBlock_Nomax(in_channels=128, out_channels=128, kernel_size=3))
                layer_cb.append(ResidualBlock(in_channels=128, out_channels=128, kernel_size=3))
            self.fixed_block.append(layer_cb)
        
        self.global_pooling = nn.AdaptiveAvgPool2d((2,2))
        self.fc1 = nn.Parameter(torch.randn(200, 2*2*128))
        self.fc2 = nn.Linear(200, 50)
        self.out = nn.Linear(50, self.classes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)
        #self._initialize_weights()

    def forward(self, x, choice, kchoice):
        for i, j in enumerate(choice):
            if j==3:
                kchoice[i]=kchoice[i-1]
                continue
            else:
                if i==0:
                    x=self.fixed_block[i][j](x, 3, self.kernel_list[kchoice[i]])
                else:
                    x=self.fixed_block[i][j](x, self.kernel_list[kchoice[i-1]], self.kernel_list[kchoice[i]])

        x = self.global_pooling(x)
        x = x.view(-1, 2*2*self.kernel_list[kchoice[self.layers-1]])
        x = F.linear(x, self.fc1[:,:self.kernel_list[kchoice[self.layers-1]]*2*2])
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dp2(x)
        x = self.out(x)
        return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

