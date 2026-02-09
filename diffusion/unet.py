"""
TODO:
1) Add sin embedding into res block
2) Attention blocks in each ResBlock

"""

import torch
import torch.nn as nn
from typing import List

class SinusoidalEmbedding(nn.Module):

    def __init__(self, noise_embedding_size:int):
        ## TODO: Add options for the log levels
        super().__init__()

        self.embed_size = noise_embedding_size
        # Between log(1) to log(1000)
        frequencies = torch.exp(torch.linspace(0, 3, self.embed_size // 2))
        self.angular_speeds = 2.0 * torch.pi * frequencies
    
    def forward(self,x):

        x/= self.embed_size

        embeddings = torch.concat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)
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
        self.up_sample = nn.ConvTranspose2d(in_channels,in_channels, kernel_size=2, stride=2)

        self.res_blocks = nn.ModuleList()
        for _ in range(depth-1):
            self.res_blocks.append(ResidualBlock(2*in_channels,in_channels))
        
        self.res_blocks.append(ResidualBlock(2*in_channels,out_channels))
        
    
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
                 time_const:int,
                 block_depth:int = 2,
                 block_sizes:List[int] = [32,64,96],
                 n_res_blocks:int = 2,
                 noise_embed_dim:int = 128,
                 ):

        super().__init__()
        self.T = time_const
        self.noise_embed_dim = noise_embed_dim
        
        # 1) Get t ∈[0,T]        
        self.noise_pipeline = nn.Sequential(
        # 2) Do sin embedding to get a vector of size embed_dim
        SinusoidalEmbedding(noise_embed_dim),
        # 3) Pass it through a MLP: embed_dim -> 4*embed_dim
        nn.Linear(self.noise_embed_dim,4*self.noise_embed_dim),
        # 4) Activation
        nn.SiLU(inplace=True),
        # 5) 2nd MLP 4*self.noise_embed_dim->in_channels
        nn.Linear(4*self.noise_embed_dim,self.noise_embed_dim),
        )
        self.embed_mlp = nn.Linear(self.noise_embed_dim,block_sizes[0])
        self.input_conv = nn.Conv2d(in_channels,block_sizes[0],kernel_size=3,padding=1)

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(block_sizes[i],block_sizes[i+1],block_depth)
                for i in range(len(block_sizes)-1)
            ]
        )

        horz_channels = [block_sizes[-1]]*n_res_blocks
        self.horz_blocks = nn.Sequential(
            *[
                ResidualBlock(horz_channels[i],horz_channels[i+1])
                for i in range(len(horz_channels)-1)
            ]
        )
        
        up_channels =  list(reversed(block_sizes))
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(up_channels[i],up_channels[i+1],block_depth)
                for i in range(len(up_channels)-1)
            ]
        )

        self.out_conv = nn.Conv2d(block_sizes[-1],in_channels,kernel_size=3,padding=1)
        
    def forward(self,img,noise_level):

        noise = self.noise_pipeline(noise_level)
        x = self.input_conv(img) + self.embed_mlp(noise)[:,:,None,None]

        skips = []
        for db in self.down_blocks:

            x , level_skips = db(x)
            skips.append(level_skips)
        
        x = self.horz_blocks(x)

        for ub, level_skips in zip(self.up_blocks, reversed(skips)):
            x = ub((x, level_skips))

        x = self.out_conv(x)
        
        return x


model = UNet(
    in_channels=3,
    time_const= 1,
    block_depth=2,
)
print(model)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)
x = model(torch.zeros((10,3,64,64)),torch.zeros((10,1)))

print(x.shape)






            
