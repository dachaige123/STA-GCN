import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()

        self.tokenConv = nn.Conv2d(in_channels=c_in, out_channels=d_model,
                                    kernel_size=(1,1), padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

    def forward(self, x):

        x = self.tokenConv(x.permute(0, 3, 1,2))
        return x

