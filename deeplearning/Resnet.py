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


x = torch.rand(1, 64, 224, 224)
y = torch.rand(1, 256, 224, 224)
baseblock = BasicBlock(64, 64)
deepblock = DeepResBlock(256, 256)

output1 = baseblock(x)
print(output1.shape)  # 应该输出 torch.Size([1, 64, 224, 224])

output2 = deepblock(y)
print(output2.shape)  # 应该输出 torch.Size([1, 256, 224, 224])