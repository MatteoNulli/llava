import torch
import torch.nn.functional as F
import os
from pathlib import Path
from transformers import AutoProcessor, AutoModel
from open_clip import create_model_from_pretrained, create_model_and_transforms
import os.path
import sys
import numpy as np
from PIL import Image
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig



from .base_encoder import ProcessorWrapper
from .clip_encoder import ClipVisionTower

# 3. Function to convert the weights
def convert_siglip_weights(google_state_dict):
    openclip_state_dict = {}
    
    for google_key, param in google_state_dict.items():
        # Map the keys to OpenCLIP format
        openclip_key = google_key
        for google_pattern, openclip_pattern in state_dict_mapping.items():
            if google_pattern in google_key:
                openclip_key = google_key.replace(google_pattern, openclip_pattern)
                break
        
        # Handle specific layer name differences
        if 'self_attn.q_proj' in openclip_key:
            openclip_key = openclip_key.replace('self_attn.q_proj', 'attn.q_proj')
        elif 'self_attn.k_proj' in openclip_key:
            openclip_key = openclip_key.replace('self_attn.k_proj', 'attn.k_proj')
        elif 'self_attn.v_proj' in openclip_key:
            openclip_key = openclip_key.replace('self_attn.v_proj', 'attn.v_proj')
        elif 'self_attn.out_proj' in openclip_key:
            openclip_key = openclip_key.replace('self_attn.out_proj', 'attn.out_proj')
            
        openclip_state_dict[openclip_key] = param
    
    return openclip_state_dict

class MKH_SIGLIP(nn.Module):
   
    def __init__(self,pre_train_config):
        super().__init__()
        
        self.emb_size_final=2048
        self.encoder = AutoModel.from_pretrained(pre_train_config)
        self.image_reducer = nn.Linear(768, self.emb_size_final)
        
        self.title_hidden = nn.Sequential(
            nn.Linear(768, self.emb_size_final),
            nn.ReLU(),
            nn.LayerNorm(self.emb_size_final),
        )
        self.image_hidden = nn.Sequential(
            nn.BatchNorm1d(self.emb_size_final),
            nn.ReLU(),
        )
        
    def forward(self, input_x):
        
        outputs=self.encoder(**{'pixel_values':input_x['pixel_values'],'input_ids':input_x['input_ids']})
        title_emb=outputs.text_embeds
        image_emb=outputs.image_embeds
        
        image_emb = self.image_reducer(image_emb) # BSx2048
        
        title_emb = self.title_hidden(title_emb)
        title_emb=title_emb/title_emb.norm(p=2, dim=-1, keepdim=True)
        image_emb = self.image_hidden(image_emb)  ### BN + RELU for V2 (Exp 1)
        image_emb=image_emb/image_emb.norm(p=2, dim=-1, keepdim=True)
        
        logits_per_text = (
            torch.matmul(title_emb, image_emb.t().to(title_emb.device)) * self.encoder.logit_scale.exp()
            + self.encoder.logit_bias
        )
        logits_per_image = logits_per_text.t()

        eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
        m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        loglik = torch.nn.functional.logsigmoid(m1_diag1 * logits_per_text)
        nll = -torch.sum(loglik, dim=-1)
        loss = nll.mean()
        
        return loss
    


def extract_res_interp(model_name):
    valid_model_prefixes = {
        "siglip/CLIP-ViT-SO400M-14-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "timm/ViT-SO400M-14-SigLIP-384":"hf-hub:timm/ViT-SO400M-14-SigLIP-384",
        "siglip/CLIP-ViT-SO400M-14":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "timm/ViT-SO400M-14-SigLIP":"hf-hub:timm/ViT-SO400M-14-SigLIP",
        "/mnt/nushare2/data/mnulli/model_zoos/SigLIP_SO400M-14_384":"hf-hub:/mnt/nushare2/data/mnulli/model_zoos/SigLIP_SO400M-14_384",
        "/mnt/nushare2/data/mnulli/model_zoos/siglip/models--timm--ViT-SO400M-14-SigLIP-384": "hf-hub:/mnt/nushare2/data/mnulli/model_zoos/siglip/models--timm--ViT-SO400M-14-SigLIP-384",
        "/mnt/nushare2/data/mnulli/model_zoos/cvteam-ve/siglip-b-16-384-cvteam-2":"/mnt/nushare2/data/mnulli/model_zoos/cvteam-ve/siglip-b-16-384-cvteam-2",
        "/mnt/nushare2/data/mnulli/model_zoos/cvteam-ve/siglip-b-16-384-cvteam":"/mnt/nushare2/data/mnulli/model_zoos/cvteam-ve/siglip-b-16-384-cvteam",
        '/mnt/nushare2/data/mnulli/model_zoos/siglip/models--timm--ViT-B-16-SigLIP-384': '/mnt/nushare2/data/mnulli/model_zoos/siglip/models--timm--ViT-B-16-SigLIP-384/snapshots/9477b5b38a19eb7ff44123329ee6eaf1fcc724e3',  
        '/mnt/nushare2/data/mnulli/model_zoos/siglip/S0400Mv2/models--timm--ViT-SO400M-14-SigLIP-384/snapshots/ac16108d567c4389e6cd2b11c9b8585f7474435b':'/mnt/nushare2/data/mnulli/model_zoos/siglip/S0400Mv2/models--timm--ViT-SO400M-14-SigLIP-384/snapshots/ac16108d567c4389e6cd2b11c9b8585f7474435b'
    } 
    

    res = 384 if '384' in model_name else 224
    interp = None

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = valid_model_prefixes[prefix]
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class SiglipVisionTower(ClipVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, res, interp = extract_res_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._image_size = res if res is not None else 512
        self._interp_size = interp
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self._hidden_size = 1152

    def load_model(self, device_map=None):
        self.vision_model = "siglip"
        print('vision_tower_name', self.vision_tower_name)
        
        if "/mnt" in self.vision_tower_name.lower() and 'timm' in self.vision_tower_name.lower():
            if 'timm' in self.vision_tower_name.lower():
                m_name = self.vision_tower_name.split('models--timm--')[1]
                if 'hf-hub:' in self.vision_tower_name.lower():
                    pretrained = self.vision_tower_name.split('hf-hub:')[1]
                else:
                    pretrained = self.vision_tower_name
            # elif "cvteam-ve/" in  self.vision_tower_name.lower():
            #     m_name = self.vision_tower_name.split('-ve/')[1]
            #     pretrained = self.vision_tower_name
            if 'snapshots' in m_name:
                m_name = m_name.split('/snapshots')[0]
            print("m_name", m_name)
            # print('pretrained before bin', pretrained)
            
            pretrained = pretrained + '/open_clip_pytorch_model.bin'
            # print('pretrained after bin', pretrained)
            # path_obj = Path(pretrained)
            if not os.path.exists(pretrained):
                print(f"This path {pretrained} does not exist")
                print(f'Trying to access safetensors file...')
                pretrained = pretrained.replace('open_clip_pytorch_model.bin', 'open_clip_model.safetensors')
                print('pretrained', pretrained)
                if not os.path.exists(pretrained):               
                    raise FileNotFoundError(f"This path {pretrained} does not exist")
                
            clip_model, processor = create_model_from_pretrained(m_name, pretrained=pretrained)
            self.vision_tower = clip_model.visual.trunk
            self.vision_tower.output_tokens = True

            self._hidden_size = self.vision_tower.embed_dim
            self._image_size = self.vision_tower.patch_embed.img_size[0]
            self._patch_size = self.vision_tower.patch_embed.patch_size[0]

            ##try
            # self.image_processor = processor

            ##actual
            self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

            self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
            self.is_loaded = True
        
            
        # elif "/mnt" in self.vision_tower_name.lower() and 'cvteam' in self.vision_tower_name.lower():
        # #  do what you were doing in the ipynb
        #     # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     # print(f'using device: {device}')

        #     m_name = 'ViT-B-16-SigLIP-384'
        #     pretrained = self.vision_tower_name
        #     pretrained = pretrained + '/model.safetensors'
        #     # pretrained = pretrained + '/open_clip_pytorch_model.bin'
        #     clip_model, processor = create_model_from_pretrained(m_name, pretrained=pretrained)
        #     # clip_model.to(device_map)

        #     ##define model
        #     # pre_train_config = '/mnt/nushare2/data/mnulli/model_zoos/cvteam-ve/models--google--siglip-base-patch16-384/snapshots/41aec1c83b32e0a6fca20ad88ba058aa5b5ea394'  #download it from https://huggingface.co/google/siglip-base-patch16-384
        #     # SigLip_model=MKH_SIGLIP(pre_train_config)
        #     # # SigLip_model.to(device)

        #     # ##load weights
        #     # path = '/mnt/nushare2/data/xiaogli/mkh_siglip.pth'
        #     # weights = torch.load(path)

        #     # modified_weights = {}
        #     # for key in weights['state_dict'].keys():
        #     #     new_key = key[9:]  # adjust this as needed
        #     #     modified_weights[new_key] = weights['state_dict'][key]
            
        #     # SigLip_model.load_state_dict(modified_weights, strict=True)

        #     # ##convert to openclip
        #     # # 1. Load OpenCLIP's SigLip
        #     # model_name = "ViT-B-16-SigLIP-384"
        #     # clip_model, _, processor = create_model_and_transforms(model_name)

        #     # # 2. Create a weight mapping dictionary
        #     # state_dict_mapping = {
        #     #     # Vision Model mappings
        #     #     'vision_model.embeddings.patch_embedding': 'visual.conv1',
        #     #     'vision_model.embeddings.position_embedding': 'visual.positional_embedding',
        #     #     'vision_model.encoder.layers': 'visual.transformer.resblocks',
        #     #     'vision_model.post_layernorm': 'visual.ln_post',
                
        #     #     # Text Model mappings
        #     #     'text_model.embeddings.token_embedding': 'text.token_embedding',
        #     #     'text_model.embeddings.position_embedding': 'text.positional_embedding',
        #     #     'text_model.encoder.layers': 'text.transformer.resblocks',
        #     #     'text_model.final_layer_norm': 'text.ln_final',
        #     # }

        #     # # 3. Convert and load the weights
        #     # google_state_dict = SigLip_model.state_dict()
        #     # openclip_state_dict = convert_siglip_weights(google_state_dict)
        #     # clip_model.load_state_dict(openclip_state_dict, strict=False)

        #     ## finally
        #     self.vision_tower = clip_model.visual.trunk
        #     self.vision_tower.output_tokens = True

        #     self._hidden_size = self.vision_tower.embed_dim
        #     self._image_size = self.vision_tower.patch_embed.img_size[0]
        #     self._patch_size = self.vision_tower.patch_embed.patch_size[0]
        #     self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

        #     self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        #     self.is_loaded = True


            

        #     m_name = self.vision_tower_name.split('model_zoos/')[1]
        #     # print(f'Loading SigLIP model from cvteam {pretrained}...')
        #     print("m_name", m_name)
        #     pretrained = self.vision_tower_name
        #     # pretrained = pretrained + '/open_clip_pytorch_model.bin'
        #     print('pretrained', pretrained)
        #     # path_obj = Path(pretrained)
        #     if not os.path.exists(pretrained):
        #         # print(f"This path {pretrained} does not exist")
        #         raise FileNotFoundError(f"This path {pretrained} does not exist")
            
        #     model = AutoModel.from_pretrained(pretrained)
        #     processor = AutoProcessor.from_pretrained(pretrained)
            
        #     self.vision_tower = model.vision_model
        #     self.vision_tower.output_tokens = True

        #     self._hidden_size = self.vision_tower.embeddings.embed_dim
        #     self._image_size = model.config.vision_config.image_size
        #     self._patch_size = self.vision_tower.embeddings.patch_embedding.kernel_size[0]
        #     self.image_processor = ProcessorWrapper(processor.image_processor, height=self._image_size, width=self._image_size)

        #     self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        #     self.is_loaded = True

        else:
            clip_model, processor = create_model_from_pretrained(self.vision_tower_name)
        
        
            self.vision_tower = clip_model.visual.trunk
            self.vision_tower.output_tokens = True

            self._hidden_size = self.vision_tower.embed_dim
            self._image_size = self.vision_tower.patch_embed.img_size[0]
            self._patch_size = self.vision_tower.patch_embed.patch_size[0]
            self.image_processor = ProcessorWrapper(processor, height=self._image_size, width=self._image_size)

            self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
            self.is_loaded = True


    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images, interpolate_token = 576):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_features = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            interp_features = self.interpolate(image_features)
            return interp_features
