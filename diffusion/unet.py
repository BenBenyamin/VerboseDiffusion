import torch
import torch.nn as nn
from typing import List

class SinusoidalEmbedding(nn.Module):

    def __init__(self, noise_embedding_size):
        super().__init__()

        self.embed_size = noise_embedding_size
    
    def forward(self,x):

        frequencies = torch.exp(torch.linspace( torch.log(1.0), torch.log(1000.0), self.embed_size // 2))
        angular_speeds = 2.0 * torch.pi * frequencies
        embeddings = torch.concat([torch.sin(angular_speeds * x), torch.cos(angular_speeds * x)], axis=3)
        return embeddings


class ResidualBlock(nn.Module):

    def __init__(self, in_channels,out_channels):
        super().__init__()

        if (in_channels == out_channels):
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        
        self.batch_norm = nn.BatchNorm2d(in_channels,affine=False)
        
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)
        self.activation = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)


    def forward(self,x):

        res = self.res(x)
        x = self.batch_norm(x)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)

        return x + res

    
class DownBlock(nn.Module):

    def __init__(self, in_channels:int,out_channels:int,depth:int):
        super().__init__()

        self.depth = depth

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualBlock(in_channels,out_channels))

        for _ in range(depth-1):
            self.res_blocks.append(ResidualBlock(out_channels,out_channels))
        
        self.pool = nn.AvgPool2d(kernel_size=2)


    def forward(self,x):

        skips = []
        
        for res_block in self.res_blocks:

            x = res_block(x)

            skips.append(x)

        x = self.pool(x)
        
        return x, skips


class UpBlock(nn.Module):

    def __init__(self, in_channels:int,out_channels:int,depth:int):
        super().__init__()

        self.depth = depth

        # Make it 2x bigger
        self.up_sample = nn.ConvTranspose2d(in_channels,out_channels, kernel_size=2, stride=2)

        self.res_blocks = nn.ModuleList()

        for _ in range(depth):
            self.res_blocks.append(ResidualBlock(2*out_channels,out_channels))
    
    def forward(self,x):

        # Unpack the skips from the down block
        x, skips = x

        x = self.up_sample(x)

        for res_block in self.res_blocks:

            x = torch.cat((x,skips.pop()),dim =1)
            x = res_block(x)
        
        return x
    

class UNet(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 depth:int, 
                 time_const:int,
                 res_block_sizes:List[int] = [32,64,96],
                 noise_embed_dim:int = 32,
                 ):

        self.T = time_const
        self.noise_embed_dim = noise_embed_dim
        # Note: Usually, this is what is going on:
        # 1) Get t ∈[0,T]
        # 2) Do sin embedding to get a vector of size embed_dim
        # 3) Pass it through a MLP: embed_dim -> 4*embed_dim
        # However, in this implemintation 
        self.pos_embed = nn.Embedding(self.T,4*self.noise_embed_dim)

        self.image_conv = nn.Conv2d(in_channels,noise_embed_dim,kernel_size=1)





            
