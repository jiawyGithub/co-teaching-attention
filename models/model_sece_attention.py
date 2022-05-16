import math
from operator import attrgetter
import torch
import torch.nn as nn
import torch.nn.init as init 
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class CALayer(nn.Module):  # Channel Attention (CA) Layer
    def __init__(self, in_channels, reduction=16, pool_types=['avg', 'max']):
        super().__init__()
        self.pool_list = ['avg', 'max']
        self.pool_types = pool_types
        self.in_channels = in_channels
        self.Pool = [nn.AdaptiveAvgPool2d(
            1), nn.AdaptiveMaxPool2d(1, return_indices=False)]
        self.conv_ca = nn.Sequential(
            nn.Conv2d(in_channels, in_channels //
                      reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction,
                      in_channels, 1, padding=0, bias=True)
        )

    def forward(self, x):
        for (i, pool_type) in enumerate(self.pool_types):
            pool = self.Pool[self.pool_list.index(pool_type)](x)
            channel_att_raw = self.conv_ca(pool)
            if i == 0:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum += channel_att_raw
        scale = F.sigmoid(channel_att_sum)
        return x * scale

class SALayer(nn.Module):  # Spatial Attention Layer
    def __init__(self):
        super().__init__()
        self.conv_sa = nn.Sequential(
            nn.Conv2d(2, 1, 3, 1, 1, bias=False),
            nn.BatchNorm2d(1, momentum=0.01),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_compress = torch.cat(
            (torch.max(x, 1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)), dim=1)
        scale = self.conv_sa(x_compress)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=2, pool_types=['avg', 'max'], attention_type = None):
        super().__init__()
        self.CALayer = CALayer(in_channels, reduction, pool_types)
        self.SALayer = SALayer()
        
        self.attention_type = attention_type

    def forward(self, x):
        x_out = x
        if self.attention_type == None:
            x_out = self.CALayer(x_out)
            x_out = self.SALayer(x_out)
        if self.attention_type == "ca":
            x_out = self.CALayer(x_out)
        if self.attention_type == "sa":
            x_out = self.SALayer(x_out)
        return x_out

class CNN_CBAM(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False, attention_type=None):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN_CBAM, self).__init__()
        self.c1 = nn.Conv2d(input_channel, 128, kernel_size=3, stride=1, padding=1)
        self.se = CBAM(in_channels=128, attention_type=attention_type)
        self.c2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.se1 = CBAM(in_channels=128, attention_type=attention_type)
        self.c4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.c5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.c6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.se2 = CBAM(in_channels=256, attention_type=attention_type)
        self.c7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.se3 = CBAM(in_channels=512, attention_type=attention_type)
        self.c8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
        self.se4 = CBAM(in_channels=256, attention_type=attention_type)
        self.c9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)
        self.se5 = CBAM(in_channels=128, attention_type=attention_type)

        self.l_c1 = nn.Linear(128, n_outputs)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(256)
        self.bn7 = nn.BatchNorm2d(512)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(128)

    def forward(self, x, ):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = self.c3(h)
        h = self.se1(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c4(h)
        h = F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h = self.c5(h)
        h = F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h = self.c6(h)
        # h = self.se2(h)
        h = F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=2, stride=2)
        h = F.dropout2d(h, p=self.dropout_rate)

        h = self.c7(h)
        # h = self.se3(h)
        h = F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h = self.c8(h)
        # h = self.se4(h)
        h = F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h = self.c9(h)
        # h = self.se5(h)
        h = F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h = F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit = self.l_c1(h)
        if self.top_bn:
            logit = call_bn(self.bn_c1, logit)
        return logit


