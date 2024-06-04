
from models.Graph_convolution import GraphConvLayer
from models.attn import ProbAttention,FullAttention,AttentionLayer
from models.Temporal_gated_convolution import TemporalConvLayer
import torch.nn as nn

class ST_Block(nn.Module):


    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, graph_conv_type, gso,
                 bias, droprate, factor, scale, atten_s, atten_t, seq, atten_dropout, d_model,black_num,ko,d_model_t):
        super(ST_Block, self).__init__()


        if black_num!=1:
            d_model_t=channels[1]
        else:
            d_model_t = d_model

        attention_t = ProbAttention(factor=factor, attention_dropout=atten_dropout,
                                      scale=scale) if atten_t == 'prob' else FullAttention(
            factor=factor, scale=scale, attention_dropout=atten_dropout)
        attention_s = ProbAttention(factor=factor, attention_dropout=atten_dropout,
                                      scale=scale) if atten_s == 'prob' else FullAttention(
            factor=factor, scale=scale, attention_dropout=atten_dropout)
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.attention_s = AttentionLayer(attention=attention_s, d_model=channels[0], )
        self.graph_conv = GraphConvLayer(graph_conv_type, channels[0], channels[1], Ks, gso, bias)

        self.attention_t = AttentionLayer(attention=attention_t, d_model= d_model_t, )

        #self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        # self.tc1_ln = nn.LayerNorm([n_vertex, channels[2]])
        # self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])

        self.dropout = nn.Dropout(p=droprate)
        #self.norm_s = nn.LayerNorm([ko, channels[2]])
        #self.norm_t = nn.LayerNorm([n_vertex, d_model_t])
        self.norm_s = nn.LayerNorm([ko,n_vertex, channels[0]])
        self.norm_t = nn.LayerNorm([n_vertex,ko+2, d_model_t])
        self.relu_t=nn.ReLU()
        self.relu_s = nn.ReLU()
        self.relu = nn.ReLU()


    def forward(self, x_t):

        if len(x_t)==3:
            x_t=x_t[0]

        x_t_a, att_t = self.attention_t(x_t, x_t, x_t)
        x_t_a=self.dropout(x_t_a) + x_t
        #x_t_a = self.norm_t(x_t_a.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        #x_t_a = self.norm_t(x_t_a)
        x_t_a=self.relu_t(x_t_a)
        x_t_a = x_t_a.permute(0, 3, 2, 1)

        #x_t_a= self.tmp_conv1(x_t.permute(0, 3, 2, 1))####
        x_t_a = self.tmp_conv1(x_t_a)##

        x_t_a = x_t_a.permute(0, 2, 3, 1)

        x_s_a, att_s = self.attention_s(x_t_a, x_t_a, x_t_a)
        x_s_a=self.dropout(x_s_a) + x_t_a

        #x_s_a = self.norm_s(x_s_a)
        x_s_a = self.relu_s(x_s_a)
        x_s_a = x_s_a.permute(0, 3, 1, 2)

        #x_s_a = self.graph_conv(x_t_a.permute(0, 3, 1, 2)).permute(0, 3, 2, 1)####
        x_s_a = self.graph_conv(x_s_a).permute(0, 3, 2, 1)##
        # x_s_a = self.relu(x_s_a)
        # x_s_a = self.tmp_conv2(x_s_a)
        # x_s_a = self.tc2_ln(x_s_a.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x_s_a = self.dropout(x_s_a)
        # x_s_a= x_s_a.permute(0,3,2,1)

        return x_s_a, att_t, att_s
class OutputBlock(nn.Module):


    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()

        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()
        self.silu = nn.SiLU()
        self.dropout = nn.Dropout(p=droprate)
        self.last_block_channel = last_block_channel

    def forward(self, x):

        x = self.tmp_conv1(x)

        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)

        x = self.relu(x)
        x = self.fc2(x).permute(0, 3, 1, 2)

        return x