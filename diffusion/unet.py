"""
TODO:
0) Maybe debug guided diffusion? Diffusers looks more readable
1) Attention blocks in each ResBlock?
2) Learn sigma?

Orginal paper:
https://arxiv.org/pdf/2006.11239

Improved paper:
https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf

Group norm paper:
https://arxiv.org/pdf/1803.08494
"""

import torch
import torch.nn as nn
from typing import List

class SinusoidalEmbedding(nn.Module):

    def __init__(self, 
                 noise_embedding_size:int,
                 n_timesteps:int,
                 min_freq:float = 1.0,
                 max_freq:float = 1000.0,
                 ):
        super().__init__()

        min_freq = torch.log10(torch.tensor(min_freq,dtype=torch.float16))
        max_freq = torch.log10(torch.tensor(max_freq,dtype=torch.float16))

        self.embed_size = noise_embedding_size
        self.T = n_timesteps

        frequencies = torch.pow(10,torch.linspace(min_freq,max_freq , self.embed_size // 2))
        self.angular_speeds = 2.0 * torch.pi * frequencies
    
    def forward(self,x):

        x/= self.T

        embeddings = torch.concat([torch.sin(self.angular_speeds * x), torch.cos(self.angular_speeds * x)], dim=-1)
        return embeddings


class AdditiveTimeConditioning(nn.Module):

    def __init__(self,temb_channels: int,out_channels:int,n_groups:int):
        super().__init__()
        
        self.temb_channels = temb_channels
        self.out_channels = out_channels

        self.norm = nn.GroupNorm(n_groups,out_channels, eps= 1e-6,affine=True)
        self.proj = nn.Linear(temb_channels,out_channels)
    
    def forward(self,x,temb):
        
        x = self.proj(temb)[:, :, None, None] + x
        x = self.norm(x)

        return x

        

class ScaleShiftConditioning(nn.Module):
    """
    https://arxiv.org/abs/1709.07871
    https://github.com/caffeinism/FiLM-pytorch?tab=readme-ov-file
    """
    def __init__(self,temb_channels: int,out_channels:int,n_groups:int):
        super().__init__()
        
        self.temb_channels = temb_channels
        self.out_channels = out_channels

        self.proj = nn.Linear(temb_channels,2*out_channels)
        self.norm = nn.GroupNorm(n_groups,out_channels, eps= 1e-6,affine=True)

    def forward(self,x,temb):

        proj_temb = self.proj(temb)

        scale, shift = torch.chunk(proj_temb,2,dim = 1)

        x = self.norm(x)
        
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        
        return x*(1 + scale) + shift


class ResidualBlock(nn.Module):

    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 time_embedding_norm:str,  ## additive or scale-shift/ FiLM
                 noise_embed_dim:int = 128,
                 n_groups:int = 32,
                 dropout:float = 0.0
                 ):
        super().__init__()

        if (in_channels == out_channels):
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_channels,out_channels,kernel_size=1)
        
        self.norm1 = nn.GroupNorm(n_groups,in_channels, eps= 1e-6,affine=True) 

        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1)

        if time_embedding_norm == "additive":
            self.temb_pipeline = AdditiveTimeConditioning(noise_embed_dim,out_channels,n_groups)
        
        elif time_embedding_norm == "film":
            self.temb_pipeline = ScaleShiftConditioning(noise_embed_dim,out_channels,n_groups)
        
        else:
            raise ValueError(f"Unkown time_embedding_norm, got {time_embedding_norm}, expected: 'additive',film")

        self.activation = nn.SiLU(inplace=True)

        self.dropout = torch.nn.Dropout(dropout)
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)


    def forward(self,x,temb):

        res = self.res(x)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        temb = self.activation(temb)
        x = self.temb_pipeline(x,temb)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + res

class AttentionBlock(nn.Module):

    def __init__(self,  
                 channels:int = 3,
                 num_heads:int = 4,
                 dropout:float = 0.0,
                 norm:bool = True,
                 n_groups:int = 32,
                 ):
        
        super().__init__()

        if norm:
            self.norm = nn.GroupNorm(n_groups,channels,eps=1e-6,affine=True)

        else:
            self.norm = None
        
        self.attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.linear = nn.Linear(channels,channels)
    
    def forward(self,x):

        B, C, H, W = x.shape

        if self.norm is not None:
        
            x = self.norm(x)

        x = x.view(B, C, H * W).transpose(1, 2).contiguous()

        x_attn, _ = self.attn(x, x, x, need_weights=False)
        
        x_attn = self.linear(x_attn)

        x_attn = x_attn.transpose(1, 2).contiguous().view(B, C, H, W)


        return x_attn
        
class DownBlock(nn.Module):

    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 depth:int,
                 time_embedding_norm:str,
                 noise_embed_dim:int,
                 n_groups:int = 32,
                 num_heads:int = 0,
                 dropout:float = 0.0,
                 norm:bool = True, # group
                 ):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.res_blocks = nn.ModuleList()
        self.res_blocks.append(ResidualBlock(in_channels,
                                             out_channels,
                                             time_embedding_norm,
                                             noise_embed_dim,
                                             n_groups,
                                             dropout,
                                             ))

        for _ in range(depth-1):
            self.res_blocks.append(ResidualBlock(out_channels,
                                                 out_channels,
                                                 time_embedding_norm,
                                                 noise_embed_dim,
                                                 n_groups,
                                                 dropout,
                                                 ))
        
        self.downsample = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)

        self.has_attn = num_heads > 0

        if self.has_attn  :
            self.attn = AttentionBlock(
                out_channels,
                num_heads,
                dropout,
                norm,
                n_groups
            )


    def forward(self,x,temb):

        skips = []
        
        for res_block in self.res_blocks:

            x = res_block(x,temb)

            skips.append(x)

        x = self.downsample(x)

        if self.has_attn:
            x = self.attn(x)
        
        return x, skips


class UpBlock(nn.Module):

    def __init__(self, in_channels:int,
                 out_channels:int,
                 depth:int,
                 time_embedding_norm:str,
                 noise_embed_dim:int,
                 n_groups:int = 32,
                 num_heads:int = 0,
                 dropout:float = 0.0,
                 norm:bool = True, # Group
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth

        # Make it 2x bigger
        self.up_sample = nn.Upsample(scale_factor=2, mode="nearest")

        self.res_blocks = nn.ModuleList()
        for _ in range(depth-1):
            self.res_blocks.append(ResidualBlock(2*in_channels,
                                                 in_channels,
                                                 time_embedding_norm,
                                                 noise_embed_dim,
                                                 n_groups,
                                                 dropout,
                                                 ))
            # self.res_blocks.append(ResidualBlock(2*in_channels,in_channels,time_embedding_norm,noise_embed_dim))
        
            
        
        self.res_blocks.append(ResidualBlock(2*in_channels,
                                             out_channels,
                                             time_embedding_norm,
                                             noise_embed_dim,
                                             n_groups,
                                             dropout,
                                             ))

        self.has_attn = num_heads > 0

        if self.has_attn  :
            self.attn = AttentionBlock(
                out_channels,
                num_heads,
                dropout,
                norm,
                n_groups
            )
    def forward(self,x, temb):

        # Unpack the skips from the down block
        x, skips = x

        x = self.up_sample(x)

        for res_block in self.res_blocks:

            x = torch.cat((x,skips.pop()),dim =1)
            x = res_block(x,temb)
        
        if self.has_attn:
            x = self.attn(x)
        
        return x


class UNet(nn.Module):

    def __init__(self, 
                 in_channels:int, 
                 time_const:int,
                 time_embedding_norm:str = "additive",
                 block_depth:int = 2,
                 block_sizes:List[int] = [32,64,96],
                 n_res_blocks:int = 2,
                 noise_embed_dim:int = 128,
                 n_groups:int = 32,
                 n_attn_heads:int = 4,
                 attn_levels:List[bool] = [False,True,True],
                 norm_attn:bool = True,
                 dropout:float = 0.0,
                 ):

        super().__init__()
        self.T = time_const
        self.noise_embed_dim = noise_embed_dim
        
        # 1) Get t ∈[0,T]        
        self.noise_pipeline = nn.Sequential(
        # 2) Do sin embedding to get a vector of size embed_dim
        SinusoidalEmbedding(noise_embed_dim,time_const),
        # 3) Pass it through a MLP: embed_dim -> 4*embed_dim
        nn.Linear(self.noise_embed_dim,4*self.noise_embed_dim),
        # 4) Activation
        nn.SiLU(inplace=True),
        # 5) 2nd MLP 4*self.noise_embed_dim->in_channels
        nn.Linear(4*self.noise_embed_dim,self.noise_embed_dim),
        )
        self.input_conv = nn.Conv2d(in_channels,block_sizes[0],kernel_size=3,padding=1)

        self.down_blocks = nn.ModuleList(
            [
                DownBlock(
                    block_sizes[i],
                    block_sizes[i+1],
                    block_depth,time_embedding_norm,
                    noise_embed_dim,
                    n_groups,
                    norm=norm_attn,
                    num_heads=n_attn_heads if attn_levels[i] else 0,
                    dropout=dropout,
                    )
                for i in range(len(block_sizes)-1)
            ]
        )

        horz_channels = [block_sizes[-1]]*n_res_blocks

        self.horz_blocks = nn.Sequential(
            *[
                ResidualBlock(horz_channels[i],horz_channels[i+1],time_embedding_norm,noise_embed_dim)
                for i in range(len(horz_channels)-1)
            ]
        )
        
        up_channels =  list(reversed(block_sizes))
        rev_attn_lvl = list(reversed(attn_levels))
        self.up_blocks = nn.ModuleList(
            [
                UpBlock(
                    up_channels[i],
                    up_channels[i+1],
                    block_depth,
                    time_embedding_norm,
                    noise_embed_dim,
                    n_groups,
                    norm=norm_attn,
                    num_heads=n_attn_heads if rev_attn_lvl[i] else 0,
                    dropout=dropout,
                    )
                for i in range(len(up_channels)-1)
            ]
        )

        self.out_conv = nn.Conv2d(block_sizes[0],in_channels,kernel_size=3,padding=1)
        
    def forward(self,img,noise_level):

        temb = self.noise_pipeline(noise_level)
        x = self.input_conv(img) 

        skips = []
        for db in self.down_blocks:

            x , level_skips = db(x,temb)
            skips.append(level_skips)
        
        for block in self.horz_blocks:
            x = block(x, temb)

        for ub, level_skips in zip(self.up_blocks, reversed(skips)):
            x = ub((x, level_skips),temb)

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






            
