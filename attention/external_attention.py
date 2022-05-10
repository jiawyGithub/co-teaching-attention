# Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks (CVMJ2021)
from torch import nn
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class External_attention(nn.Module):
    '''
    Arguments:
        c (int): The input and output channel number.
    '''

    def __init__(self, c):
        super(External_attention, self).__init__()

        self.conv1 = nn.Conv2d(c, c, 1)

        self.k = 64
        self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)

        self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
        self.linear_1.weight = Parameter(self.linear_0.weight.permute(1, 0, 2))

        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            nn.BatchNorm2d(c))

        self.relu = nn.ReLU()

    def forward(self, x):
        idn = x
        x = self.conv1(x)

        b, c, h, w = x.size()
        n = h*w
        x = x.view(b, c, h*w)   # b * c * n

        attn = self.linear_0(x)  # b, k, n
        attn = F.softmax(attn, dim=-1)  # b, k, n

        attn = attn / (1e-9 + attn.sum(dim=1, keepdims=True))  # b, k, n
        x = self.linear_1(attn)  # b, c, n

        x = x.view(b, c, h, w)
        x = self.conv2(x)
        x = x + idn
        x = self.relu(x)
        return x


# def main():
#     attention_block = External_attention(64)
#     input = torch.randn(4, 64, 32, 32)
#     output = attention_block(input)
#     print(input.size(), output.size())


# if __name__ == '__main__':
#     main()
