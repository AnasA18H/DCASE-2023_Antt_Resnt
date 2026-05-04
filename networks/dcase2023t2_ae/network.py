import torch
from torch import nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim, momentum=0.01, eps=1e-03)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim, momentum=0.01, eps=1e-03)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        res = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.leaky_relu(out, 0.2)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        
        out += res
        return F.leaky_relu(out, 0.2)

class AENet(nn.Module):
    def __init__(self,input_dim, block_size):
        super(AENet,self).__init__()
        self.input_dim = input_dim
        self.cov_source = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)
        self.cov_target = nn.Parameter(torch.zeros(block_size, block_size), requires_grad=False)

        # Encoder
        self.enc_in = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2)
        )
        
        self.enc_res1 = ResBlock(128)
        self.enc_res2 = ResBlock(128)
        
        # Self-Attention before bottleneck
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        self.enc_out = nn.Sequential(
            nn.Linear(128, 8),
            nn.BatchNorm1d(8, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.dec_in = nn.Sequential(
            nn.Linear(8, 128),
            nn.BatchNorm1d(128, momentum=0.01, eps=1e-03),
            nn.LeakyReLU(0.2)
        )
        
        self.dec_res1 = ResBlock(128)
        self.dec_res2 = ResBlock(128)
        
        self.dec_out = nn.Linear(128, self.input_dim)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        
        # Encoder
        e = self.enc_in(x)
        e = self.enc_res1(e)
        e = self.enc_res2(e)
        
        # Self-Attention (Requires shape: Batch, SeqLen, EmbedDim)
        e_seq = e.unsqueeze(1) # shape: (batch_size, 1, 128)
        attn_out, _ = self.attention(e_seq, e_seq, e_seq)
        e = attn_out.squeeze(1) # shape: (batch_size, 128)
        
        z = self.enc_out(e)
        
        # Decoder
        d = self.dec_in(z)
        d = self.dec_res1(d)
        d = self.dec_res2(d)
        out = self.dec_out(d)
        
        return out, z
