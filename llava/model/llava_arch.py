#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)

from llava.mm_utils import get_anyres_image_grid_shape


class BOMMaskToken(nn.Module):
    def __init__(self, input_features, dtype=torch.float32, device="cuda"):
        super().__init__()
        # Create the token parameter as part of this module
        self.mm_bom_mask_token = nn.Parameter(
            torch.randn((1, input_features), dtype=dtype, device=device)
        )

    def forward(self, x=None):
        return self.mm_bom_mask_token


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if "unpad" in getattr(config, "mm_patch_merge_type", ""):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        pretrain_mm_bom_mask_token = model_args.pretrain_mm_bom_mask_token
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.sam2_masking_token = model_args.sam2_masking_token

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(
            model_args, "mm_projector_type", "linear"
        )
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, "mm_projector", None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if "unpad" in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(
                    torch.tensor(self.config.hidden_size, dtype=self.dtype)
                )
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(
                pretrain_mm_mlp_adapter, map_location="cpu"
            )

            def get_w(weights, keyword):
                return {
                    k.split(keyword + ".")[1]: v
                    for k, v in weights.items()
                    if keyword in k
                }

            self.mm_projector.load_state_dict(
                get_w(mm_projector_weights, "mm_projector")
            )
        if pretrain_mm_bom_mask_token is not None:
            mm_bom_mask_token_weights = torch.load(
                pretrain_mm_bom_mask_token, map_location="cpu"
            )
            projector_input_features = self.mm_projector[0].in_features

            # Initialize the helper module
            self.mm_bom_mask_token = BOMMaskToken(
                projector_input_features, self.dtype, device="cuda"
            )

            def get_w(weights, keyword):
                return {k.split("model.")[1]: v for k, v in weights.items()}

            self.mm_bom_mask_token.load_state_dict(
                get_w(mm_bom_mask_token_weights, "mm_bom_mask_token")
            )

        else:
            print("Radomly Initializing mm_bom_mask_token")
            projector_input_features = self.mm_projector[0].in_features
            self.mm_bom_mask_token = nn.Parameter(
                torch.randn(
                    (1, projector_input_features), dtype=self.dtype, device="cuda"
                )
            )


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def downsample_mask_to_1d_counts(
        self, mask: torch.Tensor, output_size: int, threshold_count: float = 0.5
    ) -> torch.Tensor:
        """
        Downsamples a 1D or 2D boolean mask to a 1D boolean tensor of length 'output_size'
        while preserving the information of True pixels in a less aggressive way.

        Instead of using a quantile-based threshold (which can be overly aggressive for sparse masks),
        this function uses adaptive average pooling to compute the average (i.e. the fraction of True
        values) in each bin, multiplies by the approximate bin size to get a count, and then marks a bin
        as True if that count is above a threshold.

        Parameters:
        mask (torch.Tensor): Input mask (1D or 2D) of booleans or 0/1 values.
        output_size (int): The desired length of the final 1D mask.
        threshold_count (float): A threshold on the count of True pixels per bin.
                                Default 0.5 means that if a bin receives at least 1 True pixel
                                (on average) it will be marked True.

        Returns:
        torch.Tensor: A 1D boolean tensor of length 'output_size'.
        """
        # Convert mask to float and flatten
        mask_flat = mask.float().flatten().unsqueeze(0).unsqueeze(0)  # shape: (1,1,N)

        # Compute approximate number of pixels per bin.
        total_pixels = mask.numel()
        bin_size = (
            total_pixels / output_size
        )  # average number of original pixels per output bin

        # Adaptive average pooling: each bin now contains the fraction of True pixels over ~bin_size pixels.
        pooled = torch.nn.functional.adaptive_avg_pool1d(
            mask_flat, output_size
        ).squeeze()  # shape: (output_size,)

        # Convert the fraction to an estimated count per bin.
        counts = pooled * bin_size

        # Binarize: mark a bin as True if the estimated count is at least threshold_count.
        downsampled_mask = counts >= threshold_count
        return downsampled_mask

    def apply_masks_with_tokens(
        self, images_batch, masks_batch, global_view=False, averaging=False
    ):
        """
        Applies masks to the image features and appends special tokens signaling the masked feature start/end.

        Parameters:
            images_batch (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
            masks_batch (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
                                        the inner Tensors are of size (n. of masks, (image_h, image_w)).

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
        """

        batch_size = images_batch.shape[0]

        batched_features = []
        for i, (image_features, image_masks) in enumerate(
            zip(images_batch, masks_batch)
        ):

            # Get image features shape (batch_size, first_dim, feature_dim)
            first_dim, feature_dim = image_features.shape

            # import pdb; pdb.set_trace()
            resized_image_masks = [
                self.downsample_mask_to_1d_counts(mask, first_dim)
                for mask in image_masks
            ]

            masked_features = [image_features[mask] for mask in resized_image_masks]

            ##appending whole image feature at the beginning of the masks features
            if global_view:
                masked_features.insert(0, image_features)

            if averaging:
                masked_features = [
                    torch.mean(masked_feat, dim=0).reshape(1, feature_dim)
                    for mask_feat in mask_features
                ]  # averaging across 576 --> (bs, feature_dim)

            # import pdb; pdb.set_trace()
            masked_with_tokens = [
                torch.cat(
                    [
                        self.get_model().mm_bom_mask_token.to(image_features.device),
                        mask_feat,
                    ],
                    dim=0,
                )
                for mask_feat in masked_features
            ]

            # Step 4: Concatenate all features
            single_image_features = torch.cat(
                masked_with_tokens, dim=0
            )  # Shape (bs, (1 + n) * 576 + 2n, 1024)
            batched_features.append(single_image_features.to(image_features.device))

        padding = True
        if padding:
            # Find max dimension size
            max_dim_size = max(features.shape[0] for features in batched_features)

            # # Pad each tensor to match max_dim_size
            padded_features = []
            for features in batched_features:
                pad_size = max_dim_size - features.shape[0]
                padded = torch.cat(
                    [
                        features,
                        torch.zeros(
                            (pad_size, features.shape[1]), device=image_features.device
                        ),
                    ],
                    dim=0,
                )
                padded_features.append(padded)

            batched_features = padded_features

        final_output = torch.stack(batched_features).to(images_batch.device)

        return final_output

    # def apply_masks_with_tokens_batched(
    #     self, images_batch, masks_batch, global_view=False, averaging=False
    # ):
    #     """
    #     Applies masks to the image features and appends special tokens signaling the masked feature start/end.

    #     Parameters:
    #         images_batch (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
    #         masks_batch (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
    #                                     the inner Tensors are of size (n. of masks, (image_h, image_w)).

    #     Returns:
    #         torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
    #     """
    #     batch_size, vision_encoder_dim, feature_dim = images_batch.shape
    #     for i, image_masks in enumerate(masks_batch):
    #         masks_batch[i] = torch.stack(
    #             [
    #                 self.downsample_mask_to_1d_counts(mask, vision_encoder_dim)
    #                 for mask in image_masks
    #             ]
    #         )

    #     ## calculate maximum number of masks across the batch and pad with it for standard tensor operations for it, masks_batch is now a tensor
    #     masks_batch = pad_sequence(
    #         masks_batch, batch_first=True, padding_value=False
    #     ).to(images_batch.device)

    #     ## expand the images_batch adding the n.masks and add feature dim to masks_batch
    #     images_batch = images_batch.unsqueeze(1)
    #     masks_batch = masks_batch.unsqueeze(-1)

    #     masked_features = torch.where(
    #         masks_batch, images_batch, torch.tensor(0.0)
    #     )  # --> output is of shape (batch_size, max_masks, vision_encoder_dim, feature_dim)

    #     ##TODO: Do we aim to have this shape (batch_size, max_masks, vision_encoder_dim, feature_dim)?
    #     ###     Or, for inference purpose, we want to have something like this (batch_size, max_masks, x, feature_dim), where x is the maximum number of non-zero values across the masks?
    #     ###     We kinda have to do this to avoid giving to many visual tokens to the llm.
    #     ###     However, the problem with the latter is we kill any locality we might have in the image, because we are essentially removing all the spatial information by taking out the zeros.
    #     ##      If the latter, we need to reshape the tensor to have the desired shape and padd all the other masks.
    #     # padding = True
    #     # if padding:
    #     #     valid_mask = (masked_features != 0).any(dim=3)

    #     #     # Compute the number of valid (nonzero) rows for each mask
    #     #     # This gives a tensor of shape [B, M]
    #     #     valid_counts = valid_mask.sum(dim=2)

    #     #     # Find the maximum valid rows among all masks. In your example, you mentioned
    #     #     # that one mask has 213 and the other 588 valid rows. You can choose the maximum
    #     #     # (588) or a custom target (like 558) if needed. Here, we'll use the maximum.
    #     #     max_valid = valid_counts.max().item()  # This would be 588 in your example

    #     #     # Now, for each batch (B) and each mask (M), extract valid rows and pad to max_valid.
    #     #     padded_res = []  # Will collect results per batch
    #     #     batch_size, max_n_masks, _, _ = (
    #     #         masked_features.shape
    #     #     )  # e.g., (1, 2, 729, 1125)

    #     #     for b in range(batch_size):
    #     #         batch_masks = []
    #     #         for m in range(max_n_masks):
    #     #             # Get the indices (in dim 2) where the row has any nonzero element.
    #     #             valid_indices = valid_mask[b, m].nonzero(as_tuple=True)[0]
    #     #             valid_rows = masked_features[
    #     #                 b, m, valid_indices, :
    #     #             ]  # shape: [num_valid, E]
    #     #             num_valid = valid_rows.shape[0]
    #     #             # Determine how many rows to pad to reach max_valid
    #     #             pad_rows = max_valid - num_valid
    #     #             # Use F.pad to pad at the bottom (i.e. after the valid rows) along dimension 0.
    #     #             # F.pad expects a tuple (pad_left, pad_right, pad_top, pad_bottom)
    #     #             valid_rows_padded = torch.nn.functional.pad(
    #     #                 valid_rows, (0, 0, 0, pad_rows)
    #     #             )
    #     #             # valid_rows_padded now has shape [max_valid, E]
    #     #             batch_masks.append(valid_rows_padded)
    #     #         # Stack the masks for this batch along the mask dimension (dim 1)
    #     #         batch_tensor = torch.stack(
    #     #             batch_masks, dim=0
    #     #         )  # shape: [M, max_valid, E]
    #     #         padded_res.append(batch_tensor)
    #     #     # Finally, stack across batches
    #     #     padded_res = torch.stack(padded_res, dim=0)  # shape: [B, M, max_valid, E]
    #     #     masked_features = padded_res

    #     final_output = masked_features.flatten(start_dim=1, end_dim=2)

    #     return final_output

    def encode_images(self, images, masks, masking=False):
        """
        Encodes images in the vision encoder and then in the multimodal projector

        Parameters:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, vision_encoder_input, feature_dim).
            masks (torch.Tensor): List of Tensor. The Outer list if of batch_size length, the inner list is of lenght number of masks in each image.

        Returns:
            torch.Tensor: Image features outputted from multimodal projector.
        """

        ##passing images to vision encoder
        image_features = self.get_model().get_vision_tower()(images)
        # image_features = self.get_model().vision_resampler(image_features, images=images)
        ## B.O. MASKING PART
        if masking and any(torch.any(mask != 0).item() for mask in masks):
            # Inserting Masking information within Image embeddings
            image_features = self.apply_masks_with_tokens(image_features, masks).to(
                images.dtype
            )
        ## E.O. MASKING PART
        image_features = self.get_model().mm_projector(image_features)

        return image_features

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        masks,
        image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)

            image_features = self.encode_images(
                concat_images, masks, masking=self.get_model().sam2_masking_token
            )
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, "mm_patch_merge_type", "flat")
            image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "square")
            if mm_patch_merge_type == "flat":
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith("spatial"):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == "anyres":
                            num_patch_width, num_patch_height = (
                                get_anyres_image_grid_shape(
                                    image_sizes[image_idx],
                                    self.config.image_grid_pinpoints,
                                    self.get_vision_tower().config.image_size,
                                )
                            )
                            image_feature = image_feature.view(
                                num_patch_height, num_patch_width, height, width, -1
                            )
                        else:
                            raise NotImplementedError
                        if "unpad" in mm_patch_merge_type:
                            image_feature = image_feature.permute(
                                4, 0, 2, 1, 3
                            ).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(
                                image_feature, image_sizes[image_idx]
                            )
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[:, None, None]
                                    .expand(*image_feature.shape[:-1], 1)
                                    .to(image_feature.device),
                                ),
                                dim=-1,
                            )
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(
                                0, 2, 1, 3, 4
                            ).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat(
                            (base_image_feature, image_feature), dim=0
                        )
                    else:
                        image_feature = image_feature[0]
                        if "unpad" in mm_patch_merge_type:
                            image_feature = torch.cat(
                                (
                                    image_feature,
                                    self.model.image_newline[None].to(
                                        image_feature.device
                                    ),
                                ),
                                dim=0,
                            )
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(
                    f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}"
                )
        else:
            image_features = self.encode_images(
                images, masks, masking=self.get_model().sam2_masking_token
            )

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, "tune_mm_mlp_adapter", False) and getattr(
            self.config, "mm_use_im_start_end", False
        ):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for cur_labels, cur_attention_mask in zip(labels, attention_mask)
        ]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat(
                    [cur_input_embeds_1, cur_image_features[0:0]], dim=0
                )
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            ##splitting input_ids based on where the IMAGE_TOKEN_INDEX is
            image_token_indices = (
                [-1]
                + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                + [cur_input_ids.shape[0]]
            )
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(
                    cur_input_ids[
                        image_token_indices[i] + 1 : image_token_indices[i + 1]
                    ]
                )
                cur_labels_noim.append(
                    cur_labels[image_token_indices[i] + 1 : image_token_indices[i + 1]]
                )

            ## keep acting on split version of token
            ## creating two split types of embeddings based on before/after IMAGE_TOKEN_INDEX
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(
                torch.cat(cur_input_ids_noim)
            )
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            # print('cur_input_embeds', cur_input_embeds)
            # print('cur_input_embeds_no_im', cur_input_embeds_no_im)
            cur_new_input_embeds = []
            cur_new_labels = []

            ## adding for each image (in our case 1) the image-embedding between the text embedding from before the IMAGE_TOKEN_INDEX and the text embedding after.
            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(
                        torch.full(
                            (cur_image_features.shape[0],),
                            IGNORE_INDEX,
                            device=cur_labels.device,
                            dtype=cur_labels.dtype,
                        )
                    )

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
            ## new_input_embeds and new_labels are the input to the model which include both image and text embeddings. 'text..text <image> text..text' --> [text_embedding] + [image_embedding] + [text_embedding]

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(
            self.config, "tokenizer_model_max_length", None
        )
        if tokenizer_model_max_length is not None:
            new_input_embeds = [
                x[:tokenizer_model_max_length] for x in new_input_embeds
            ]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )

        for i, (cur_new_embed, cur_new_labels) in enumerate(
            zip(new_input_embeds, new_labels)
        ):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, "tokenizer_padding_side", "right") == "left":
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                            cur_new_embed,
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )
            else:
                new_input_embeds_padded.append(
                    torch.cat(
                        (
                            cur_new_embed,
                            torch.zeros(
                                (max_len - cur_len, cur_new_embed.shape[1]),
                                dtype=cur_new_embed.dtype,
                                device=cur_new_embed.device,
                            ),
                        ),
                        dim=0,
                    )
                )
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(
                        0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                    )

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return (
            None,
            position_ids,
            attention_mask,
            past_key_values,
            new_input_embeds,
            new_labels,
        )

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            ## dynamically adding the DEFAULT_IMAGE_PATCH_TOKEN to the special tokens
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            ## dynamically adding the DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN to the special tokens
            num_new_tokens = tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True
                )

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.pretrain_mm_bom_mask_token:
                mm_bom_mask_token_weights = torch.load(
                    model_args.pretrain_mm_bom_mask_token, map_location="cpu"
                )
                mm_bom_mask_token_weight = mm_bom_mask_token_weights[
                    "model.mm_bom_mask_token"
                ]
                # assert num_new_tokens == 2
                # output_embeddings[-num_new_tokens:] = mm_bom_mask_token_weight

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(
                    model_args.pretrain_mm_mlp_adapter, map_location="cpu"
                )
                embed_tokens_weight = mm_projector_weights["model.embed_tokens.weight"]
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[
                        -num_new_tokens:
                    ]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}."
                    )

        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
