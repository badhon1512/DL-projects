import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.batch_norm_1 = nn.BatchNorm2d(out_channels)
        self.conv2d_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.batch_norm_2 = nn.BatchNorm2d(out_channels)
        self.conv_1_1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        output = self.relu(self.batch_norm_2(self.conv2d_2(self.relu(self.batch_norm_1(self.conv2d_1(x))))))
        # making the same b and c
        x = self.batch_norm(self.conv_1_1(x))
        output = torch.add(output, x)
        return output


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv2d_1 = nn.Conv2d(3, 64, 7, 2)
        self.batch_norm_1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(3, 2)
        self.res_block_1 = ResBlock(64, 64, 1)
        self.res_block_2 = ResBlock(64, 128, 2)
        self.res_block_3 = ResBlock(128, 256, 2)
        self.res_block_4 = ResBlock(256, 512, 2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512, 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # according to arch
        m_p_a = self.max_pool(self.relu(self.batch_norm_1(self.conv2d_1(x))))
        r_b_a = self.res_block_4(self.res_block_3(self.res_block_2(self.res_block_1(m_p_a))))
        g_a = self.global_avg_pool(r_b_a)
        f = self.flatten(g_a)
        fc_a = self.fc(f)

        return self.sigmoid(fc_a)



