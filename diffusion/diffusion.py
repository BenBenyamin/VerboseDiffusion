"""
TODO: Condition the model

It seems like the after this everything else stays the same (from diffusers):

class_emb = self.get_class_embed(sample=sample, class_labels=class_labels)
if class_emb is not None:
    if self.config.class_embeddings_concat:
        emb = torch.cat([emb, class_emb], dim=-1)
    else:
        emb = emb + class_emb


if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim, act_fn=act_fn)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        elif class_embed_type == "projection":
---

Orginal paper:
https://arxiv.org/pdf/2006.11239

Improved paper:
https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf

Group norm paper:
https://arxiv.org/pdf/1803.08494
"""

import torch
from torch.optim.swa_utils import AveragedModel , get_ema_multi_avg_fn
from torch.optim import AdamW
import torch.nn as nn

import os

from typing import List , Literal

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class SinusoidalEmbedding(nn.Module):

    def __init__(self,
                 device:str,
                 noise_embedding_size:int,
                 n_timesteps:int,
                 min_freq:float = 1.0,
                 max_freq:float = 1000.0,
                 ):
        super().__init__()

        min_freq = torch.log10(torch.tensor(min_freq,dtype=torch.float16,device=device))
        max_freq = torch.log10(torch.tensor(max_freq,dtype=torch.float16,device=device))

        self.embed_size = noise_embedding_size
        self.T = n_timesteps

        frequencies = torch.pow(10,torch.linspace(min_freq,max_freq , self.embed_size // 2)).to(device)
        self.angular_speeds = 2.0 * torch.pi * frequencies
    
    def forward(self,x):

        x = x.float() / float(self.T)

        x = x[:,None]

        embeddings = torch.concat([torch.sin(self.angular_speeds[None, :] * x), torch.cos(self.angular_speeds[None, :] * x)], dim=-1)
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
                 time_embedding_norm:Literal["additive","film"],  ## additive or scale-shift/ FiLM
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

        self.activation = nn.SiLU()

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

        x_attn, _ = self.attn(x, x, x, need_weights=False) #Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
        
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
                 device:str,
                 in_channels:int, 
                 time_const:int,
                 conditional:bool,
                 n_classes:int,
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

        self.device = device

        if conditional:
            self.class_embd = nn.Embedding(n_classes,noise_embed_dim)
        else:
            self.class_embd = None
        
        # 1) Get t ∈[0,T]        
        self.noise_pipeline = nn.Sequential(
        # 2) Do sin embedding to get a vector of size embed_dim
        SinusoidalEmbedding(self.device, noise_embed_dim,time_const),
        # 3) Pass it through a MLP: embed_dim -> 4*embed_dim
        nn.Linear(self.noise_embed_dim,4*self.noise_embed_dim),
        # 4) Activation
        nn.SiLU(),
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
        
    def forward(self,img,noise_level:int,class_idx = None):

        temb = self.noise_pipeline(noise_level)

        if self.class_embd is not None and class_idx is not None:
            temb += self.class_embd(class_idx)

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

## TODO: Write this->
class DiffusionModel:

    def __init__(self,
                 device:str,
                 in_channels:int,
                 time_const:int,
                 conditional:bool,
                 n_classes:int,
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
                 beta_schedule:Literal["linear","cosine"] = "cosine",
                 sched_min:float = 0.02,
                 sched_max:float = 0.95,
                 prediction_type:Literal["epsilon","sample","v_prediction"] = "epsilon", # https://medium.com/@zljdanceholic/three-stable-diffusion-training-losses-x0-epsilon-and-v-prediction-126de920eb73
                 ema_decay:float = 0.999,
                 cond_pred:bool = True,
                 ):    
        
        
        if cond_pred:
            n_classes +=1 # add null class

        self.model = UNet(
                 device,
                 in_channels, 
                 time_const,
                 conditional,
                 n_classes,
                 time_embedding_norm,
                 block_depth,
                 block_sizes,
                 n_res_blocks,
                 noise_embed_dim,
                 n_groups,
                 n_attn_heads,
                 attn_levels,
                 norm_attn,
                 dropout,
        ).to(device)

        self.n_classes = n_classes

        self.in_channels = in_channels

        #Get a copy for the EMA
        self.ema = AveragedModel(
            self.model,
            multi_avg_fn= get_ema_multi_avg_fn(ema_decay)
            )

        self.T = time_const
        self.device = device
        
        sched_min = torch.tensor(sched_min, dtype=torch.float32)
        sched_max = torch.tensor(sched_max, dtype=torch.float32)

        if beta_schedule == "linear":
            
            betas = torch.linspace(sched_min,sched_max,self.T, dtype=torch.float32,device=device)
            alphas = torch.cumprod(1-betas,dim=0)
            self.signal_rates = torch.sqrt(alphas)
            self.noise_rates = torch.sqrt(1-alphas)
        
        elif beta_schedule == "cosine":
            # TODO: Check why this formula is equivalent to the orginal one
            # And why is it flipped with the angles 
            start_angle = torch.acos(sched_max)
            end_angle = torch.acos(sched_min)
            angles = torch.linspace(start_angle,end_angle,self.T, dtype=torch.float32,device=device)
            self.signal_rates = torch.cos(angles)
            self.noise_rates = torch.sin(angles)
        
        self.pred_type = prediction_type
        self.ema_decay = ema_decay
        self.cond_pred = cond_pred

        self.loss = nn.MSELoss(reduction="none")

    def train(
        self,
        steps,
        train_dataloader,
        val_dataloader,
        lr = 0.001,
        uncond_prob:float = 0.1,
        grad_norm: float | None = 1.0,
        log_dir:str = "./runs/",
        log_every:int = 10_000,
        ):
        
        # TODO: 1) Add mixed precision

        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda")

        self.writer = SummaryWriter(log_dir)

        steps_cnt = 0

        while True:

            running_loss = 0
            n_examples = 0

            self.model.train()

            for batch_idx, (img, class_idx) in enumerate(train_dataloader):

                batch_size = img.shape[0]

                optimizer.zero_grad()
                img = img.to(self.device)
                class_idx = class_idx.to(self.device)
                ts = torch.randint(0, self.T, size=(batch_size,), device=img.device)
                epsilon = torch.randn_like(img)

                nr = self.noise_rates[ts].view(batch_size, 1, 1, 1)
                sr = self.signal_rates[ts].view(batch_size, 1, 1, 1)

                noisy_img = sr * img + nr * epsilon

                # mask random class indexes
                if self.cond_pred:
                    class_idx +=1
                    mask = torch.rand(batch_size, device=class_idx.device) < uncond_prob
                    class_idx_input = class_idx.clone()
                    class_idx_input[mask] = 0
                else:
                    class_idx_input = class_idx

                with torch.autocast("cuda", dtype=torch.float16):

                    model_output = self.model(noisy_img,ts,class_idx_input)

                    # TODO: Check why snr is needed
                    snr = (sr/nr)**2
                    if self.pred_type == "epsilon":
                        loss = self.loss(epsilon,model_output)
                        weight = 1.0
                    
                    elif self.pred_type == "sample":
                        loss = self.loss(img,model_output)
                        weight = snr
                    
                    elif self.pred_type == "v_prediction":
                        # https://arxiv.org/pdf/2102.09672
                        v_target = sr * epsilon - nr * img
                        loss = self.loss(v_target,model_output)
                        weight = snr/(snr+1)
                
                
                loss = (loss * weight).mean()

                scaler.scale(loss).backward()

                if grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm)

                scaler.step(optimizer)
                scaler.update()

                self.ema.update_parameters(self.model)

                running_loss += loss.item()*batch_size
                n_examples += batch_size
                steps_cnt +=1
            
                if steps_cnt % log_every == 0:
                    self._log_metrics(steps_cnt,running_loss/n_examples,val_dataloader,img.shape[2:])
                
                if steps_cnt >= steps:
                    return # Stop training

    
    @torch.inference_mode()
    def validate(self, val_dataloader , mode:Literal["raw","ema"]):
        
        running_loss = 0
        n_examples = 0
        
        if mode == "raw":
            model = self.model
            
        elif mode == "ema":
            model = self.ema.module

        model.eval()


        for batch_idx, (img, class_idx) in enumerate(val_dataloader):
                
                batch_size = img.shape[0]

                img = img.to(self.device)
                class_idx = class_idx.to(self.device)

                ts = torch.randint(0, self.T, size=(batch_size,), device=img.device)
                epsilon = torch.randn_like(img)

                nr = self.noise_rates[ts].view(batch_size, 1, 1, 1)
                sr = self.signal_rates[ts].view(batch_size, 1, 1, 1)

                noisy_img = sr * img + nr * epsilon

                # if self.cond_pred:
                #     class_idx +=1

                model_output = model(noisy_img,ts,class_idx)

                snr = (sr/nr)**2

                if self.pred_type == "epsilon":
                    loss = self.loss(epsilon,model_output)
                    weight = 1.0
                
                elif self.pred_type == "sample":
                    loss = self.loss(img,model_output)
                    weight = snr
                
                elif self.pred_type == "v_prediction":
                    # https://arxiv.org/pdf/2102.09672
                    v_target = sr * epsilon - nr * img
                    loss = self.loss(v_target,model_output)
                    weight = snr/(snr+1)
                
                loss = (loss*weight).mean()
                
                running_loss += loss.item()*batch_size
                n_examples += batch_size
        
        if mode == "raw":
            self.model.train()
        return running_loss/n_examples
    
    @torch.inference_mode()
    def sample(self,
               n_samples:int, 
               shape: tuple, 
               class_labels = None, 
               added_noise_weight:float = 0.0, 
               guidance_scale:float = 1.0
               ):
        
        #generate the noise

        self.ema.module.eval()
        
        device = next(self.model.parameters()).device

        size = (n_samples,self.in_channels,*shape)

        x_t = torch.randn(size, device=device)

        if class_labels is None and self.cond_pred:
            
            class_labels = torch.randint(0, self.n_classes, (n_samples,), device=device)

        # if self.cond_pred:
        #     class_labels +=1

        for t in reversed(range(1,self.T)):
            
            ts = torch.full((n_samples,), t, device=device, dtype=torch.long)
            
            nr = self.noise_rates[ts].view(n_samples,1,1,1)
            sr = self.signal_rates[ts].view(n_samples,1,1,1)

            # use CFG?
            if self.cond_pred and guidance_scale != 1.0:
                uncond_labels = torch.zeros_like(class_labels)  # 0 is your unconditional token
                out_uncond = self.ema.module(x_t, ts, class_idx=uncond_labels)
                out_cond   = self.ema.module(x_t, ts, class_idx=class_labels)
                model_out  = out_uncond + guidance_scale * (out_cond - out_uncond)
            else:
                model_out  = self.ema.module(x_t, ts, class_idx=class_labels)
            

            if self.pred_type == "epsilon":

                pred_eps = model_out
                x_0 = (x_t-nr*pred_eps)/sr
            
            elif self.pred_type == "sample":

                x_0 = model_out
                pred_eps = (x_t - x_0*sr)/nr
            
            elif self.pred_type == "v_prediction":

                v_pred = model_out 
                
                # v_target = sr * epsilon - nr * x_0
                pred_eps = sr * v_pred + nr * x_t
                x_0 = sr * x_t - nr * v_pred
            
            # sr = sqrt(alpha_t), nr = sqrt(1- alpha_t)
            prev_sr = self.signal_rates[ts-1].view(n_samples,1,1,1)
            prev_nr = self.noise_rates[ts-1].view(n_samples,1,1,1)
            
            # https://arxiv.org/pdf/2010.02502#page=6 eq 16 , added_noise_weight = eta
            sigma_t = added_noise_weight * (prev_nr / nr) * torch.sqrt(torch.abs(1 - sr**2/prev_sr**2))

            x_t = prev_sr * x_0 + torch.sqrt(1 - prev_sr**2 - sigma_t**2) * pred_eps + sigma_t * torch.randn_like(x_t)

        x_0 = (x_0.clamp(-1,1) + 1) / 2  # normalize to [0,1]
        return x_0
            

    @torch.inference_mode()
    def _log_metrics(self, epoch, train_loss , val_dataloader, shape):
        
        val_loss = self.validate(val_dataloader,mode = "raw")
        ema_val_loss = self.validate(val_dataloader,mode = "ema")

        self.writer.add_scalar("Loss/train",train_loss,global_step=epoch)
        self.writer.add_scalar("Loss/val",val_loss,global_step=epoch)
        self.writer.add_scalar("Loss/ema_val",ema_val_loss,global_step=epoch)


        # Set the seed in this particular instance
        with torch.random.fork_rng(enabled=True):
            torch.manual_seed(0)

            samples = self.sample(
                n_samples=self.n_classes,
                shape=shape,
                class_labels=torch.arange(self.n_classes, device=self.device, dtype=torch.long),
                added_noise_weight=0.0,
                guidance_scale=1.0,
            )

        self.writer.add_image(
            "Samples",
            make_grid(samples,nrow=self.n_classes),
            global_step=epoch,
                        )
    
    def generate(self, class_idx:List, shape: tuple, guidance_scale:float = 1.0):

        class_labels = torch.tensor(class_idx,dtype=torch.long,device=self.device)
        return self.sample(
            n_samples=len(class_idx),
            shape=shape,
            class_labels=class_labels,
            guidance_scale=guidance_scale
        )

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "ema": self.ema.module.state_dict(),
            },
            path,
        )

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model"])
        self.ema.module.load_state_dict(ckpt["ema"])

  

# df = DiffusionModel("cuda",
#                     3,
#                     1000,
#                     conditional=True,
#                     n_classes=10,
#                     beta_schedule="linear",
#                     sched_max=1.0,sched_min=0)

# x = df.sample(10,(32,32),class_labels=torch.arange(10, dtype=torch.long,device="cuda"),guidance_scale=1.1).to("cuda")

# print(x.shape)
# model = df.model
# print(model.T)
# total_params = sum(p.numel() for p in model.parameters())
# print(f"{total_params:,}")
# x = model(torch.zeros((10,3,32,32)),torch.zeros((10,1)),torch.zeros((10,),dtype=torch.int))


# print(x.shape)






            
