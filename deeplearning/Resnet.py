import torch
import torch.nn as nn

#定义一个基本块类，适用于resnet18,34
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.baseblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                    kernel_size=3, stride=stride, bias=False, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if downsample is None:
            if stride != 1 or in_channel != out_channel * self.expansion:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channel)
                )
            else:
                self.downsample = downsample
        else:
            self.downsample = downsample

    def forward(self, x):
        # 保存输入数据，以便后续残差连接
        identity = x
        out = self.baseblock(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out


class DeepResBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super().__init__()
        self.downsample = downsample
        self.bottleblock = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // self.expansion,
                               kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel // self.expansion),
            nn.ReLU(inplace=True),  # 使用inplace=True可以减少内存占用
            nn.Conv2d(in_channels=out_channel // self.expansion, out_channels=out_channel // self.expansion,
                               kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel // self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channel // self.expansion, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if downsample is None:
            if stride != 1 or in_channel != out_channel:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channel, out_channel,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channel)
                )
            else:
                self.downsample = downsample
        else:
            self.downsample = downsample



    def forward(self, x):
        identity = x

        out = self.bottleblock(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = nn.ReLU(inplace=True)(out)
        return out


class MyResnet(nn.Module):
    def __init__(self, block, blocks_num, num_class, include_top=True):
        super().__init__()
        self.include_top = include_top
        self.in_channel = 64
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, bias=False
                               , padding=3)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_class)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, bias=False, stride=stride),
                nn.BatchNorm2d(channel * block.expansion)
            )
        layer = []
        layer.append(block(
            self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion
        for _ in range(1, block_num):
            layer.append(block(self.in_channel, channel))
        return nn.Sequential(*layer)



    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        return x
def Resnet18(num_class, include_top=True, pretrained=False):
    return MyResnet(BasicBlock, [2, 2, 2, 2], num_class, include_top=include_top)

def Resnet34(num_class, include_top=True, pretrained=False):
    return MyResnet(BasicBlock, [3, 4, 6, 3], num_class, include_top=include_top)

def Resnet50(num_class, include_top=True, pretrained=False):
    return MyResnet(DeepResBlock, [3, 4, 6, 3], num_class, include_top=include_top)

def Resnet101(num_class, include_top=True, pretrained=False):
    return MyResnet(DeepResBlock, [3, 4, 23, 3], num_class, include_top=include_top)