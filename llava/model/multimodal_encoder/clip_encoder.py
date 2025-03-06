import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, create_model_and_transforms, get_tokenizer
# from ezcolorlog import root_logger as logger ##logger.debug --> print for non ezcolorlog support

from .base_encoder import BaseVisionTower
from llava.utils import IS_XLA_AVAILABLE


def extract_interp(model_name):
    interp = None
    base_model_name = model_name

    if "interp" in model_name:
        base_model_name = model_name.split('-interp')[0]

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, interp


class ClipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, interp = extract_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        if hasattr(args, 'vision_tower_base'):        
            self.vision_tower_base = args.vision_tower_base
        else:
            self.vision_tower_base = None
        self._interp_size = interp 
        self.args = args
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return

        
        destination_folder=self.vision_tower_name+'/preprocessor_config.json'

        # print('destination_folder', destination_folder)
        if os.path.exists(destination_folder):
            print('Loading image preprocessor from the same directory...')
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        
        else:
            # print(self.args._name_or_path)
            # print(self.args.model_name_or_path)
            try:
                image_processor_dir = str(self.args.model_name_or_path).split('model_zoos')[0] + 'model_zoos/openai/clip-vit-large-patch14-336'
            except:
                image_processor_dir = str(self.args._name_or_path).split('model_zoos')[0] + 'model_zoos/openai/clip-vit-large-patch14-336'
                
            if os.path.exists(image_processor_dir):
                print(f'Loading image preprocessor from {image_processor_dir}...')
                self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_dir)
            else:
                image_processor_dir = '/mnt/nushare2/data/baliao/multimodal/model_zoos/openai/clip-vit-large-patch14-336'
                print(f'Loading image preprocessor from {image_processor_dir}...')
                self.image_processor = CLIPImageProcessor.from_pretrained(image_processor_dir, device_map=device_map)
        
        ##problem is that vision_tower_base is none., We need to instanciate it to solve the issue. 
        # print('self.vision_tower_base', self.vision_tower_base)
        # print('self.vision_tower_name', self.vision_tower_name)

        if self.vision_tower_base is not None and self.vision_tower_base != self.vision_tower_name:
            print(f'Loading vision tower backbone from pre-trained checkpoint {self.vision_tower_base}...')
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_base)

        elif self.vision_tower_base is None and 'openclip-lora' in self.vision_tower_name:
            self.vision_tower_base = '/mnt/nushare2/data/baliao/multimodal/model_zoos/openai/clip-vit-large-patch14-336'

            print(f'Loading vision tower backbone from pre-trained checkpoint {self.vision_tower_base}...')
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_base)

            # print(f'Adding pre-trained lora weights to the vision tower...')
            # non_lora_path = self.vision_tower_name + '/non_lora_trainables.bin'
            # if non_lora_path:
            #     non_lora_weights = torch.load(non_lora_path)
            #     self.vision_tower.load_state_dict(non_lora_weights, strict=False)
            
        else:
            self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        
        try:
            if self.args.unfreeze_vision_encoder == True:
                self.unfreeze_mm_vision_tower = True
        except:
            print(f'No argument self.args.unfreeze_vision_encoder found, leaving unfreeze_mm_vision_tower = {self.unfreeze_mm_vision_tower}')
        
            
        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':
            features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

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

    def _forward(self, images):
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features
