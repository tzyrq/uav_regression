import torch
import torch.nn as nn
from torchsummary import summary


class MaxModel(nn.Module):
    def __init__(self):
        super(MaxModel, self).__init__()

        self.out1 = 64
        self.out2 = 128
        self.out3 = 256

        self.relu = nn.ReLU(inplace=True)

        self.initMaxConv1 = nn.Conv2d(in_channels=1, out_channels=self.out1, kernel_size=2)
        self.initMaxConv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=2)
        self.initMaxConv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=2)

        self.initMaxBN1 = nn.BatchNorm2d(self.out1)
        self.initMaxBN2 = nn.BatchNorm2d(self.out2)
        self.initMaxBN3 = nn.BatchNorm2d(self.out3)

        self.initMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.netConv1 = nn.Conv2d(in_channels=60, out_channels=self.out1, kernel_size=4, stride=2)
        self.netConv2 = nn.Conv2d(in_channels=self.out1, out_channels=self.out2, kernel_size=2, stride=1)
        self.netConv3 = nn.Conv2d(in_channels=self.out2, out_channels=self.out3, kernel_size=3, stride=1)

        self.netBN1 = nn.BatchNorm2d(self.out1)
        self.netBN2 = nn.BatchNorm2d(self.out2)
        self.netBN3 = nn.BatchNorm2d(self.out3)

        self.netMaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.deconvMax1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3)
        self.deconvMax2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2)
        self.deconvMax3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.deconvMax4 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

        self.deconvMaxBN1 = nn.BatchNorm2d(256)
        self.deconvMaxBN2 = nn.BatchNorm2d(128)
        self.deconvMaxBN3 = nn.BatchNorm2d(64)

    def initNet(self, xMax):
        xMax = self.initMaxConv1(xMax)
        xMax = self.initMaxBN1(xMax)
        xMax = self.relu(xMax)
        xMax = self.initMaxPool(xMax)

        xMax = self.initMaxConv2(xMax)
        xMax = self.initMaxBN2(xMax)
        xMax = self.relu(xMax)
        xMax = self.initMaxPool(xMax)

        xMax = self.initMaxConv3(xMax)
        xMax = self.initMaxBN3(xMax)
        xMax = self.relu(xMax)

        return xMax

    def net(self, x):
        x = self.netConv1(x)
        x = self.netBN1(x)
        x = self.relu(x)

        x = self.netConv2(x)
        x = self.netBN2(x)
        x = self.relu(x)

        x = self.netMaxPool(x)

        x = self.netConv3(x)
        x = self.netBN3(x)
        x = self.relu(x)

        return x

    def deconv(self, xMax):
        xMax = self.deconvMax1(xMax)
        xMax = self.deconvMaxBN1(xMax)
        xMax = self.relu(xMax)

        xMax = self.deconvMax2(xMax)
        xMax = self.deconvMaxBN2(xMax)
        xMax = self.relu(xMax)

        xMax = self.deconvMax3(xMax)
        xMax = self.deconvMaxBN3(xMax)
        xMax = self.relu(xMax)

        xMax = self.deconvMax4(xMax)

        return xMax

    def forward(self, initMax, x):
        initMax = self.initNet(xMax=initMax)
        x = self.net(x)
        xMax = torch.cat((x, initMax), 1)
        xMax = self.deconv(xMax=xMax)
        xMax = torch.squeeze(xMax)
        return xMax


if __name__ == "__main__":
    m = MaxModel()
    summary(m, input_size=[(1, 98, 98), (60, 100, 100)],
            device="cpu")
"""
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 97, 97]             320
       BatchNorm2d-2           [-1, 64, 97, 97]             128
              ReLU-3           [-1, 64, 97, 97]               0
         MaxPool2d-4           [-1, 64, 48, 48]               0
            Conv2d-5          [-1, 128, 47, 47]          32,896
       BatchNorm2d-6          [-1, 128, 47, 47]             256
              ReLU-7          [-1, 128, 47, 47]               0
         MaxPool2d-8          [-1, 128, 23, 23]               0
            Conv2d-9          [-1, 256, 22, 22]         131,328
      BatchNorm2d-10          [-1, 256, 22, 22]             512
             ReLU-11          [-1, 256, 22, 22]               0
           Conv2d-12           [-1, 64, 49, 49]          61,504
      BatchNorm2d-13           [-1, 64, 49, 49]             128
             ReLU-14           [-1, 64, 49, 49]               0
           Conv2d-15          [-1, 128, 48, 48]          32,896
      BatchNorm2d-16          [-1, 128, 48, 48]             256
             ReLU-17          [-1, 128, 48, 48]               0
        MaxPool2d-18          [-1, 128, 24, 24]               0
           Conv2d-19          [-1, 256, 22, 22]         295,168
      BatchNorm2d-20          [-1, 256, 22, 22]             512
             ReLU-21          [-1, 256, 22, 22]               0
  ConvTranspose2d-22          [-1, 256, 24, 24]       1,179,904
      BatchNorm2d-23          [-1, 256, 24, 24]             512
             ReLU-24          [-1, 256, 24, 24]               0
  ConvTranspose2d-25          [-1, 128, 49, 49]         295,040
      BatchNorm2d-26          [-1, 128, 49, 49]             256
             ReLU-27          [-1, 128, 49, 49]               0
  ConvTranspose2d-28           [-1, 64, 98, 98]         131,136
      BatchNorm2d-29           [-1, 64, 98, 98]             128
             ReLU-30           [-1, 64, 98, 98]               0
  ConvTranspose2d-31            [-1, 1, 98, 98]              65
================================================================
Total params: 2,162,945
Trainable params: 2,162,945
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 21981.81
Forward/backward pass size (MB): 62.95
Params size (MB): 8.25
Estimated Total Size (MB): 22053.01
----------------------------------------------------------------

"""
