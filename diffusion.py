## TODO: Delete one keep the other condtional , cond_pred

import torch
from torch.optim.swa_utils import AveragedModel , get_ema_multi_avg_fn
from torch.optim import AdamW
import torch.nn as nn

import os

from typing import List , Literal

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from unet import UNet

from diffusers import AutoencoderKL

class DiffusionModel:

    def __init__(self,
                 device:str,
                 in_channels:int,
                 time_const:int,
                 conditional:bool,
                 n_classes:int,
                 time_embedding_norm:Literal["additive","film"] = "additive",
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

        self.model = torch.compile(self.model)

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

        self.steps_cnt = 0

    def train(
        self,
        steps,
        train_dataloader,
        val_dataloader,
        lr = 0.0001,
        uncond_prob:float = 0.1,
        grad_norm: float | None = 1.0,
        log_dir:str = "./runs/",
        log_every:int = 10_000,
        ):
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler("cuda")

        self.writer = SummaryWriter(log_dir)

        steps = self.steps_cnt + steps
        
        pbar = tqdm(total=steps)
        
        epochs = 0

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
                self.steps_cnt +=1
                pbar.update(1)
            
                if self.steps_cnt % log_every == 0:
                    self._log_metrics(self.steps_cnt,running_loss/n_examples,val_dataloader,img.shape[2:])
                
                if self.steps_cnt >= steps:
                    # Stop training
                    pbar.close()
                    return
                
                pbar.set_postfix(
                    loss=loss.item(),
                    lr=optimizer.param_groups[0]["lr"],
                    epoch = epochs,
                )
        
            epochs+=1


    
    @torch.inference_mode()
    def validate(self, val_dataloader , mode:Literal["raw","ema"]):
        
        running_loss = 0
        n_examples = 0
        
        if mode == "raw":
            model = self.model
            
        elif mode == "ema":
            model = self.ema.module

        model.eval()


        for batch_idx, (img, class_idx) in enumerate(tqdm(val_dataloader,desc="Validating...")):
                
                batch_size = img.shape[0]

                img = img.to(self.device)
                class_idx = class_idx.to(self.device)

                ts = torch.randint(0, self.T, size=(batch_size,), device=img.device)
                epsilon = torch.randn_like(img)

                nr = self.noise_rates[ts].view(batch_size, 1, 1, 1)
                sr = self.signal_rates[ts].view(batch_size, 1, 1, 1)

                noisy_img = sr * img + nr * epsilon


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
    
    def generate(self, 
                 class_idx:List, 
                 shape: tuple, 
                 guidance_scale:float = 1.0,
                 seed:int = 0,
                 ):

        with torch.random.fork_rng(enabled=True):
            torch.manual_seed(seed)

            class_labels = torch.tensor(class_idx,dtype=torch.long,device=self.device) +1
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
                "steps": self.steps_cnt,
            },
            path,
        )

    def load(self, path: str, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        self.model.load_state_dict(ckpt["model"])
        self.ema.module.load_state_dict(ckpt["ema"])
        self.steps_cnt = ckpt.get("steps_cnt",0)



## TODO: Use AutoencoderKL from diffusers to load a VAE
##  1) Get latent channels: AutoencoderKL.config.latent_channels
##  2) Use them instead of in_channels
##  3) Ask chat what is the most elgeant to insert the VAE before and after 
class StableDiffusionModel(DiffusionModel):


    def __init__(self,
                device:str,
                pretrained_vae_name:str,
                time_const:int,
                conditional:bool,
                n_classes:int,
                time_embedding_norm:Literal["additive","film"] = "additive",
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
                prediction_type:Literal["epsilon","sample","v_prediction"] = "epsilon",
                ema_decay:float = 0.999,
                cond_pred:bool = True,
                scaling_const:float = 0.1825,
                ):

        
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(pretrained_vae_name).to(self.device)
        self.vae.eval()


        in_channels = self.vae.config.latent_channels

        self.scaling_const = scaling_const

        super().__init__(device, 
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
                         beta_schedule, 
                         sched_min, 
                         sched_max, 
                         prediction_type,
                         ema_decay, 
                         cond_pred)


    @torch.inference_mode()
    def _preprocess_dataloader(self, dataloader, microbatch_size, cache_name:str):

        # If cache exists, load it
        if os.path.exists(cache_name):
            print(f"Loading cached latents from {cache_name}")
            return torch.load(cache_name)

        lat_dataloader = []

        for (images, labels) in tqdm(dataloader, total=len(dataloader), desc="Encoding images with VAE"):

            lat_chunks = []

            for i in range(0, images.shape[0], microbatch_size):

                img_chunk = images[i:i+microbatch_size].to(self.device)

                lat = self.vae.encode(img_chunk).latent_dist.sample()
                lat = lat * self.scaling_const

                lat_chunks.append(lat.cpu())

            lats = torch.cat(lat_chunks, dim=0)

            lat_dataloader.append((lats, labels.cpu()))

            torch.cuda.empty_cache()

        # Save cache
        os.makedirs(os.path.dirname(cache_name), exist_ok=True)
        torch.save(lat_dataloader, cache_name)

        print(f"Saved latent cache to {cache_name}")

        return lat_dataloader

    @torch.inference_mode()
    def sample(self,
               n_samples:int, 
               shape: tuple, 
               class_labels = None, 
               added_noise_weight:float = 0.0, 
               guidance_scale:float = 1.0
               ):
        
        lats = super().sample(
            n_samples,
            shape,
            class_labels,
            added_noise_weight,
            guidance_scale,
        )

        # Undo normalization
        lats = lats * 2 - 1
        # undo latent scaling used during training
        lats = lats / self.scaling_const

        # Decode
        imgs = self.vae.decode(lats).sample

        # convert decoded images to display range
        imgs = (imgs.clamp(-1,1) + 1) / 2

        return imgs

    def train(
        self,
        steps,
        train_dataloader,
        val_dataloader,
        image_load_microbatch:int,
        lr = 0.0001,
        uncond_prob:float = 0.1,
        grad_norm: float | None = 1.0,
        log_dir:str = "./runs/",
        log_every:int = 10_000,
        ):

        lat_train_dataloader = self._preprocess_dataloader(
            train_dataloader,
            image_load_microbatch,
            cache_name="./dataset/afhq/train_latents.pt"
            )
        lat_val_dataloader = self._preprocess_dataloader(
            val_dataloader,
            image_load_microbatch,
            cache_name="./dataset/afhq/val_latents.pt"
            )

        super().train(
            steps,
            lat_train_dataloader,
            lat_val_dataloader,
            lr,
            uncond_prob,
            grad_norm,
            log_dir,
            log_every,
        )        






            
