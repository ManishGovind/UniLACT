"""UniLARN model."""
import torch
import torch.nn.functional as F
from torch import nn
import lpips
from einops import rearrange
from transformers import ViTMAEModel
from PIL import Image
from torchvision import transforms as T
import time
from collections import OrderedDict


class UniLARN(nn.Module):
    def __init__(
            self,
            image_encoder,
            m_former,
            vector_quantizer,
            m_former_depth=None,
            vector_quantizer_uni=None,
            decoder=None,
            depth_decoder=None,
            hidden_state_decoder=None,
            codebook_dim=32,
            commit_loss_w=1.,
            recon_loss_w=1.,
            recon_hidden_loss_w=1.,
            recon_depth_loss_w=1.,
            perceptual_loss_w=1.,
            use_abs_recons_loss=False,
    ):
        super().__init__()

        codebook_embed_dim = codebook_dim
        decoder_hidden_size = decoder.config.hidden_size
        m_former_hidden_size = m_former.config.hidden_size

        if isinstance(image_encoder, ViTMAEModel):
            image_encoder.config.mask_ratio = 0.0

        self.image_encoder = image_encoder.requires_grad_(False).eval()
        

        self.m_former = m_former
        
        self.m_former_depth = m_former_depth if m_former_depth else m_former

        self.vector_quantizer = vector_quantizer
        self.vq_unified = vector_quantizer_uni
        self.vq_down_resampler = nn.Sequential(
            nn.Linear(m_former_hidden_size, decoder_hidden_size),
            nn.Tanh(),
            nn.Linear(decoder_hidden_size, codebook_embed_dim)
        )
        self.vq_up_resampler = nn.Sequential(
            nn.Linear(codebook_embed_dim, codebook_embed_dim),
            nn.Tanh(),
            nn.Linear(codebook_embed_dim, decoder_hidden_size)
        )

        self.decoder = decoder
        self.depth_decoder = depth_decoder if depth_decoder else decoder
        self.hidden_state_decoder = hidden_state_decoder
        
        self.cat_fuser = nn.Linear(2 * codebook_dim, codebook_dim)
        self.uni_backprop_into_modalities = True 
        self.enable_unified = True
       
        self.commit_loss_w = commit_loss_w
        self.recon_loss_w = recon_loss_w
        self.recon_hidden_loss_w = recon_hidden_loss_w
        self.recon_depth_loss_w = recon_depth_loss_w
        self.perceptual_loss_w = perceptual_loss_w
        self.loss_fn_lpips = lpips.LPIPS(net='vgg').requires_grad_(False).eval()
        self.use_abs_recons_loss = use_abs_recons_loss

    @property
    def device(self):
        return next(self.parameters()).device


    def get_state_dict_to_save(self):
        modules_to_exclude = ['loss_fn_lpips', 'image_encoder']
        state_dict = {k: v for k, v in self.state_dict().items() if
                      not any(module_name in k for module_name in modules_to_exclude)}
        return state_dict


    @torch.no_grad()
    def decode_image(self, cond_pixel_values, given_action_token_ids):
        quant = self.vector_quantizer.get_codebook_entry(given_action_token_ids)
        latent_action_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_action_tokens=latent_action_tokens_up)
        return  {
            "recons_pixel_values": recons_pixel_values
        }
    
    @torch.no_grad()
    def decode_image_with_unified_rep(self, cond_pixel_values, given_action_token_ids):
        quant = self.vq_unified.get_codebook_entry(given_action_token_ids)
        latent_action_tokens_up = self.vq_up_resampler(quant)
        recons_pixel_values = self.decoder(cond_input=cond_pixel_values, latent_action_tokens=latent_action_tokens_up)
        return  {
            "recons_pixel_values": recons_pixel_values
        }    

    @torch.no_grad()
    def embed(self, cond_pixel_values, target_pixel_values ,pool=False, before_vq=False, avg=False):
        quant, *_ = self.tokenize(cond_pixel_values, target_pixel_values, before_vq=before_vq)
        if pool:
            latent_action_tokens_up = self.vq_up_resampler(quant)
            flat_latent_action_tokens_up = latent_action_tokens_up.reshape(latent_action_tokens_up.shape[0], -1)
            pooled_embeddings = self.decoder.transformer.embeddings.query_pooling_layer(flat_latent_action_tokens_up)
            return pooled_embeddings
        elif avg:
            return quant.mean(dim=1)
        else:
            return quant.reshape(quant.shape[0], -1)

    def tokenize(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_action_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_action_tokens, None, None
        else:
            latent_action_tokens_down = self.vq_down_resampler(latent_action_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_action_tokens_down)
            return quant, indices, commit_loss
        
    def tokenize_rgb_depth_unified(self, cond_pixel_values, target_pixel_values, before_vq=False):
        with torch.no_grad():
            cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
            target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state

        query_num = self.m_former.query_num
        latent_action_tokens = self.m_former(
            cond_hidden_states=cond_hidden_states,
            target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]

        if before_vq:
            return latent_action_tokens, None, None
        else:
            latent_action_tokens_down = self.vq_down_resampler(latent_action_tokens)
            quant, indices, commit_loss = self.vector_quantizer(latent_action_tokens_down)
            return quant, indices, commit_loss    
        
        


    def forward(self, cond_pixel_values=None, target_pixel_values=None, 
                cond_depth_values=None,target_depth_values=None,
                cond_pixel_feats=None, target_pixel_feats=None, 
                cond_depth_feats=None,target_depth_feats=None,
                return_recons_only=False, 
                return_action_token_ids_only=False): 
        
        use_rgb   = (cond_pixel_values  is not None) and (target_pixel_values  is not None)
        use_depth = (cond_depth_values  is not None) and (target_depth_values  is not None)
        assert use_rgb or use_depth, "Provide RGB and/or Depth inputs."
        
        # Tokenization
        rgb_quant = rgb_indices = rgb_commit_loss = None
        depth_quant = depth_indices = depth_commit_loss = None
        uni_quant =  uni_indices = uni_commit_loss = None
        
        if use_rgb :
           
            if cond_pixel_feats is not None :
                cond_hidden_states = cond_pixel_feats
                target_hidden_states = target_pixel_feats
            else :     
                with torch.no_grad():
                    # print("from rgb")
                    cond_hidden_states = self.image_encoder(cond_pixel_values).last_hidden_state
                    target_hidden_states = self.image_encoder(target_pixel_values).last_hidden_state
            query_num = self.m_former.query_num
            latent_action_tokens = self.m_former(
                cond_hidden_states=cond_hidden_states,
                target_hidden_states=target_hidden_states).last_hidden_state[:, :query_num]
            latent_action_tokens_down = self.vq_down_resampler(latent_action_tokens)
            rgb_quant, rgb_indices, rgb_commit_loss = self.vector_quantizer(latent_action_tokens_down)  
              
        if use_depth : 
                    
            if cond_pixel_feats is not None :
                cond_depth_hidden_states = cond_depth_feats
                target_depth_hidden_states = target_depth_feats 
            else :          
                with torch.no_grad(): 
                    # print("from depth")
                    cond_depth_hidden_states = self.image_encoder(cond_depth_values).last_hidden_state
                    target_depth_hidden_states = self.image_encoder(target_depth_values).last_hidden_state
    

            depth_query_num = self.m_former_depth.query_num
            depth_latent_action_tokens = self.m_former_depth(
                cond_hidden_states=cond_depth_hidden_states,
                target_hidden_states=target_depth_hidden_states).last_hidden_state[:, :depth_query_num]
            depth_latent_action_tokens_down = self.vq_down_resampler(depth_latent_action_tokens)
            depth_quant, depth_indices, depth_commit_loss = self.vector_quantizer(depth_latent_action_tokens_down)
        
        # quant, indices, commit_loss = self.tokenize(cond_pixel_values, target_pixel_values)
        
        if self.enable_unified and use_rgb and use_depth:
        
            fused_in = self.cat_fuser(torch.cat([rgb_quant, depth_quant], dim=-1))      # (B,T,D)
            if not self.uni_backprop_into_modalities:
                fused_in = fused_in.detach()
            uni_quant, uni_indices, uni_commit_loss = self.vq_unified(fused_in)    
                
    
        if return_action_token_ids_only:
            out = {'indices' : uni_indices }
            if use_rgb:   out["indices_rgb"] = rgb_indices
            if use_depth: out["indices_depth"] = depth_indices
            
            return out 



        # Detokenization
      
        uni_latent_action_tokens_up = self.vq_up_resampler(uni_quant)
        recons_pixel_values = None
        if self.decoder is not None:
            recons_pixel_values = self.decoder(
                cond_input=cond_pixel_values,
                latent_action_tokens=uni_latent_action_tokens_up
            )
      
        #Auxiliairy depth decoder 
        recons_depth_values = None
        if self.depth_decoder is not None:
            recons_depth_values = self.depth_decoder(
                cond_input=cond_depth_values,
                latent_action_tokens=uni_latent_action_tokens_up
            )
                
    
            
        if return_recons_only:
            out = {"indices" : uni_indices ,"indices_rgb": rgb_indices , "indices_depth": depth_indices}                   
            if recons_pixel_values is not None:
                out["recons_pixel_values"] = recons_pixel_values
            if recons_depth_values is not None:
                out["recons_depth_values"] = recons_depth_values
            return out    

        # Compute loss
        outputs = {
            "loss": torch.zeros_like(rgb_commit_loss),
            "rgb_commit_loss": rgb_commit_loss,
            "depth_commit_loss" :  depth_commit_loss,
            "uni_commit_loss" : uni_commit_loss,
            "recons_loss": torch.zeros_like(rgb_commit_loss),
            "recons_hidden_loss": torch.zeros_like(rgb_commit_loss),
            "recons_depth_loss" : torch.zeros_like(rgb_commit_loss),
            "perceptual_loss": torch.zeros_like(rgb_commit_loss)
        }

        if recons_pixel_values is not None:
            if self.use_abs_recons_loss:
                recons_loss = torch.abs(recons_pixel_values - target_pixel_values).mean()
            else:
                recons_loss = F.mse_loss(target_pixel_values, recons_pixel_values) 
                    
            outputs["recons_loss"] = recons_loss
       
        if self.perceptual_loss_w > 0:
            with torch.no_grad():
                perceptual_loss = self.loss_fn_lpips.forward(
                    target_pixel_values, recons_pixel_values, normalize=True).mean()
        else:
            perceptual_loss = torch.zeros_like(commit_loss)
            
        outputs["perceptual_loss"] = perceptual_loss
        #Add depth_commit loss and unicommit loss  here 
        loss =  self.commit_loss_w * outputs["rgb_commit_loss"] + self.recon_loss_w * outputs["recons_loss"] + \
                self.perceptual_loss_w * outputs["perceptual_loss"] 
        
        if self.depth_decoder is not None:
            recons_depth_loss = F.mse_loss(target_depth_values, recons_depth_values)
            outputs["recons_depth_loss"] = recons_depth_loss
            loss += self.recon_depth_loss_w * outputs["recons_depth_loss"]
            
        if depth_commit_loss is not None:
            loss = loss + self.commit_loss_w * depth_commit_loss
        if uni_commit_loss is not None:
            loss = loss + self.commit_loss_w * uni_commit_loss
        
       
        outputs["loss"] = loss

        # active_code_num = torch.tensor(len(set(indices.long().reshape(-1).cpu().numpy().tolist()))).float().to(loss.device)
        if uni_indices is not None :
            active_code_num = torch.tensor(torch.unique(uni_indices).shape[0]).float().to(loss.device)
            outputs["unified_active_code_num"] = active_code_num
        if rgb_indices is not None :
            rgb_active_code_num = torch.tensor(torch.unique(rgb_indices).shape[0]).float().to(loss.device)
            outputs["rgb_active_code_num"] = rgb_active_code_num  
        
        if depth_indices is not None :
            depth_active_code_num = torch.tensor(torch.unique(depth_indices).shape[0]).float().to(loss.device)
            outputs["depth_active_code_num"] = depth_active_code_num      
                
        return outputs
