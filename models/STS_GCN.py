from models.model_layer import ST_Block,OutputBlock
from models.Input_layer import TokenEmbedding
import torch.nn as nn

def count_blocks(seq,pre,Kt,block_num,d_model):

    Ko = seq - (Kt - 1) *1* block_num  # seq

    blocks = []
    blocks.append([d_model])
    for l in range(block_num):

        blocks.append([64, 16,])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([pre])
    return blocks



class STS_GCN(nn.Module):
    def __init__(self, args,):
        super(STA_GCN, self).__init__()
        blocks=count_blocks(args.seq, args.pre, args.Kt, args.block_num,args.d_model)
        if args.time_enconding==True:
            c_in=4
        else:
            c_in = 1
        self.TokenEmbedding=TokenEmbedding(c_in=c_in,d_model=args.d_model)
        Ko_ls=[]
        seq_leno=args.seq-args.Kt+1
        Ko_ls.append(seq_leno)
        next_seq=seq_leno
        for k in range(1,len(blocks) - 2):

            next_seq = next_seq - args.Kt + 1
            Ko_ls.append(next_seq)

        model_t_ls=[]
        model_t_ls.append(args.d_model)
        for l in range(len(blocks) - 3):
            model_t_ls.append(blocks[l + 1][-1])
        modules = []
        for l in range(len(blocks) - 3):

            modules.append(ST_Block(args.Kt, args.Ks, args.n_vertex, blocks[l][-1], blocks[l + 1], args.act_func,
            args.graph_conv_type,  args.gso, args.enable_bias, args.dropout,
            args.factor,args.scale,args.atten_s,args.atten_t,args.seq,args.atten_dropout,
            args.d_model,args.block_num,Ko_ls[l],model_t_ls[l]))
        self.blocks = nn.Sequential(*modules)

        Ko = args.seq - (len(blocks) - 3) * 1 * (args.Kt - 1)

        self.Ko = Ko
        if self.Ko > 1:
            self.output = OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], args.n_vertex, args.act_func,
                                             args.enable_bias, args.dropout)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, x):

        x=self.TokenEmbedding(x)
        x=x.permute(0,3,2,1)
        x,atten_t,atten_s = self.blocks(x)
        x = x.permute(0, 3, 2, 1)

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x.squeeze(-2),atten_t,atten_s