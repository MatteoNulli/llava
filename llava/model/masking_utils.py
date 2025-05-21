import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.ops as ops
import itertools
import math


from typing import List, Callable, Optional, Tuple, Union, Sequence
from torch import nn
from dataclasses import dataclass, field


def check_device_type_shape(resized_image_masks, image_features):
    for m in resized_image_masks:
        assert (
            m.device == image_features.device
        ), f"mask device {m.device} doesn't match features {image_features.device}"
        assert (
            m.dtype == torch.bool
        ), f"mask dtype {m.dtype} doesn't match features {torch.bool}"
        assert (
            m.shape[0] == image_features.shape[0]
        ), f"mask shape {m.shape} doesn't match features {image_features.shape}"


def check_resised_image_masks(resized_image_masks, image_features, device="cuda"):
    """
    Cleaning resized_image_masks.
    Either by replacing replace the only element with an all-True mask
    Or removing all tensors which have no True values in them, if we have more than 1 element.
    """
    if len(resized_image_masks) == 1:
        mask = resized_image_masks[0]
        # if the only mask has no True’s, replace it with an all-True mask
        if not mask.any().item():
            resized_image_masks[0] = torch.ones_like(mask)
    else:
        # if there’s more than one mask, drop any that are all-False
        resized_image_masks = [m for m in resized_image_masks if m.any().item()]

    resized_image_masks = [m.to(device) for m in resized_image_masks]
    check_device_type_shape(resized_image_masks, image_features)

    return resized_image_masks


# Helper function: pad a tensor to the target shape
def adjust_tensor_size(tensor, target_channels, target_height, target_width):
    c, h, w = tensor.shape

    # --- Pad or trim the channel dimension (if necessary) ---
    # For this example, if the tensor has more channels than target,
    # you might choose to select a subset.
    if c < target_channels:
        pad_channels = target_channels - c
        pad_front = pad_channels // 2
        pad_back = pad_channels - pad_front
        front_pad = torch.zeros(
            (pad_front, h, w), dtype=tensor.dtype, device=tensor.device
        )
        back_pad = torch.zeros(
            (pad_back, h, w), dtype=tensor.dtype, device=tensor.device
        )
        tensor = torch.cat([front_pad, tensor, back_pad], dim=0)
    elif c > target_channels:
        tensor = tensor[:target_channels, :, :]

    # --- Adjust Spatial Dimensions ---
    # Downsample if larger than target
    if h > target_height or w > target_width:
        tensor = tensor.to(torch.float32)
        tensor = tensor.unsqueeze(0)  # add batch dimension
        tensor = F.interpolate(
            tensor, size=(target_height, target_width), mode="nearest"
        )
        tensor = tensor.squeeze(0)
    # Pad if smaller than target
    elif h < target_height or w < target_width:
        pad_h_total = target_height - h
        pad_w_total = target_width - w
        pad_top = pad_h_total // 2
        pad_bottom = pad_h_total - pad_top
        pad_left = pad_w_total // 2
        pad_right = pad_w_total - pad_left
        tensor = F.pad(tensor, (pad_left, pad_right, pad_top, pad_bottom))

    return tensor


def downsample_mask_to_1d_counts(
    mask: torch.Tensor,
    output_size: int,
    threshold_count: float = 0.5,
    device: torch.device = "cpu",
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
    mask_flat = (
        mask.float().flatten().contiguous().unsqueeze(0).unsqueeze(0)
    )  # shape: (1,1,N)

    # Compute approximate number of pixels per bin.
    total_pixels = mask.numel()
    assert mask.numel() > 0, "Input mask is empty"
    bin_size = (
        total_pixels / output_size
    )  # average number of original pixels per output bin

    # Adaptive average pooling: each bin now contains the fraction of True pixels over ~bin_size pixels.
    # print("mask_flat shape device", mask_flat.shape, mask_flat.device)
    # print("output_size", output_size)
    pooled = torch.nn.functional.adaptive_avg_pool1d(
        mask_flat, output_size
    ).squeeze()  # shape: (output_size,)

    # Convert the fraction to an estimated count per bin.
    counts = pooled * bin_size

    # Binarize: mark a bin as True if the estimated count is at least threshold_count.
    downsampled_mask = counts >= threshold_count
    return downsampled_mask


def create_deterministic_dummy_masks(
    shape,
    num_masks: int = 10,
    seed: int = 42,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bool,
) -> list[torch.Tensor]:
    """
    Generate `num_masks` binary (0/1 or bool) **torch Tensor** masks with a
    deterministic pattern and return them as a *list* (no numpy arrays).

    Parameters
    ----------
    shape : int | tuple[int, ...]
        Shape of each individual mask.  A single int is treated as a 1‑D length.
    num_masks : int, default 10
        How many masks to create.
    seed : int, default 42
        Base seed for reproducibility.  Each mask gets `seed + i` so they differ
        while remaining deterministic across runs.
    device : torch.device | str | None, optional
        Where to place the tensors (e.g. `"cuda"`).  `None` → current default device.
    dtype : torch.dtype, default torch.bool
        Data type of the mask elements.  Use `torch.uint8` or `torch.int8`
        if you prefer 0/1 integers.

    Returns
    -------
    list[torch.Tensor]
        A Python list containing `num_masks` tensors of identical shape.
    """
    # Normalise shape
    if isinstance(shape, int):
        shape = (shape,)

    masks: list[torch.Tensor] = []

    for i in range(num_masks):
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)  # deterministic but unique
        # Draw random 0/1 integers and cast to the requested dtype
        mask = torch.randint(0, 2, shape, generator=gen, device=device).to(dtype)
        masks.append(mask)

    return masks


def create_sliding_masks(ve_dim, dtype=torch.bool, device="cuda", num_masks=10):
    masks = []
    patch_size = ve_dim // num_masks
    for i in range(num_masks):
        # For the last mask, include any remainder
        mask = torch.zeros(ve_dim, dtype=dtype, device=device)

        # compute slice (include any remainder on the final mask)
        start = i * patch_size
        end = (i + 1) * patch_size if i < num_masks - 1 else ve_dim

        mask[start:end] = True  # set active window
        masks.append(mask)

    return masks


class MaskEmbedder(torch.nn.Module):

    def __init__(
        self,
        model,
        get_model,
        config,
        vision_tower,
    ):
        super(MaskEmbedder, self).__init__()

        # objects
        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower

        # Init absolute positional encoding
        if self.model.absolute_positional_encoding:
            ve_dim = (
                int(
                    self.vision_encoder.vision_tower.vision_model.embeddings.position_embedding.num_embeddings
                )
                - 1
            )
            feature_dim = self.config.mm_hidden_size
            pe = torch.zeros(ve_dim, feature_dim)
            position = torch.arange(ve_dim, dtype=torch.float).unsqueeze(1)  # [576,1]
            div_term = torch.exp(
                torch.arange(0, feature_dim, 2, dtype=torch.float)
                * -(math.log(10000.0) / feature_dim)
            )  # [512]
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            # register as buffer so it moves with the module, but isn't a learnable param
            self.register_buffer("abs_pos_embed", pe)  # shape [576,1024]
        if self.model.learnable_positional_encoding:
            ve_dim = (
                int(
                    self.vision_encoder.vision_tower.vision_model.embeddings.position_embedding.num_embeddings
                )
                - 1
            )
            feature_dim = self.config.mm_hidden_size
            self.learn_pos_embed = nn.Embedding(ve_dim, feature_dim)
            nn.init.trunc_normal_(self.learn_pos_embed.weight, std=0.02)

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    def _compute_batch_ranges(
        self,
        masked_features: List[torch.Tensor],
        image_features: torch.Tensor,
        mm_bom_mask_token: torch.Tensor,
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the same grouping you used in your .cat(...) logic,
        then compute start/end offsets along dim=0.

        Returns e.g. [(0,100), (100,200), …].
        """

        groups = []

        # 1) the “global view” chunk (if any)
        if self.model.global_view and not self.model.no_masktoken:
            if not self.model.averaging_global_view:
                # whole image_features as one chunk
                groups.append([image_features])
            else:
                # mean-pooled image_features as one chunk
                avg = image_features.mean(dim=0, keepdim=True)  # shape (1, D)
                groups.append([avg])

        # 2) each mask → possibly prefixed by its mask_token
        for feat in masked_features:
            if not self.model.no_masktoken:
                # [token, feat] is one chunk
                token = mm_bom_mask_token.to(feat.device).unsqueeze(
                    0
                )  # ensure shape (1,D)
                groups.append([token, feat])
            else:
                # just the feat
                groups.append([feat])

        # 3) now sum up lengths per chunk
        lengths = [sum(t.size(0) for t in chunk) for chunk in groups]
        ends = list(itertools.accumulate(lengths))
        starts = [0] + ends[:-1]

        return list(zip(starts, ends))

    def _apply_bom_tokens(self, image_features, masked_features, ve_dim, feature_dim):
        """
        Applying begnining of mask tokens to masked features.
        """
        ##appending whole image feature at the beginning of the masks features

        if (
            self.model.global_view
            and not self.model.averaging_global_view
            and not self.model.no_masktoken
        ):
            if hasattr(self.get_model, "mm_bom_mask_token"):  # during training
                masked_with_tokens = torch.cat(
                    [image_features]
                    + [
                        item
                        for mask_feat in masked_features
                        for item in [
                            self.get_model.mm_bom_mask_token.mm_bom_mask_token.to(
                                image_features.device
                            ),
                            mask_feat,
                        ]
                    ],
                    dim=0,
                )
            else:  # during inference
                masked_with_tokens = torch.cat(
                    [image_features]
                    + [
                        item
                        for mask_feat in masked_features
                        for item in [
                            self.model.mm_bom_mask_token.mm_bom_mask_token.to(
                                image_features.device
                            ),
                            mask_feat,
                        ]
                    ],
                    dim=0,
                )
        elif (
            self.model.global_view
            and self.model.averaging_global_view
            and not self.model.no_masktoken
        ):
            if hasattr(self.get_model, "mm_bom_mask_token"):  # during training
                masked_with_tokens = torch.cat(
                    [torch.mean(image_features, dim=0).reshape(1, feature_dim)]
                    + [
                        item
                        for mask_feat in masked_features
                        for item in [
                            self.get_model.mm_bom_mask_token.mm_bom_mask_token.to(
                                image_features.device
                            ),
                            mask_feat,
                        ]
                    ],
                    dim=0,
                )

            else:  # during inference
                masked_with_tokens = torch.cat(
                    [torch.mean(image_features, dim=0).reshape(1, feature_dim)]
                    + [
                        item
                        for mask_feat in masked_features
                        for item in [
                            self.model.mm_bom_mask_token.mm_bom_mask_token.to(
                                image_features.device
                            ),
                            mask_feat,
                        ]
                    ],
                    dim=0,
                )

        elif (
            self.model.global_view
            and not self.model.averaging_global_view
            and self.model.no_masktoken
        ):
            masked_with_tokens = torch.cat(
                [image_features] + [mask_feat for mask_feat in masked_features],
                dim=0,
            )
        elif (
            not self.model.global_view
            and not self.model.averaging_global_view
            and self.model.no_masktoken
        ):
            masked_with_tokens = torch.cat(
                [mask_feat for mask_feat in masked_features], dim=0
            )
        elif (
            self.model.global_view
            and self.model.averaging_global_view
            and self.model.no_masktoken
        ):
            masked_with_tokens = torch.cat(
                [torch.mean(image_features, dim=0).reshape(1, feature_dim)]
                + [mask_feat for mask_feat in masked_features],
                dim=0,
            )

        else:
            if hasattr(self.get_model, "mm_bom_mask_token"):
                masked_with_tokens = torch.cat(
                    [
                        torch.cat(
                            [
                                self.get_model.mm_bom_mask_token.mm_bom_mask_token.to(
                                    image_features.device
                                ),
                                mask_feat,
                            ],
                            dim=0,
                        )
                        for mask_feat in masked_features
                    ],
                    dim=0,
                )
            else:
                masked_with_tokens = torch.cat(
                    [
                        torch.cat(
                            [
                                self.model.mm_bom_mask_token.mm_bom_mask_token.to(
                                    image_features.device
                                ),
                                mask_feat,
                            ],
                            dim=0,
                        )
                        for mask_feat in masked_features
                    ],
                    dim=0,
                )

        return masked_with_tokens

    def _pad_batched_feat(self, image_features, batched_features, padding=True):
        """
        Applying padding to masked torch sequence.
        """
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
        else:
            padded_features = batched_features

        return padded_features

    def _apply_masks_with_tokens(self, images_batch, masks_batch):
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
        # print("images_batch", images_batch)
        batched_features = []
        batch_ranges = []
        for i, (image_features, image_masks) in enumerate(
            zip(images_batch, masks_batch)
        ):

            # Get image features shape (batch_size, ve_dim, feature_dim)
            ve_dim, feature_dim = image_features.shape

            ## for indexing masks operation pass them to cpu
            passing_device = "cpu"
            image_masks = image_masks.to(passing_device)
            if self.model.use_sliding_window:
                resized_image_masks = create_sliding_masks(
                    ve_dim, device=passing_device, num_masks=self.model.number_of_masks
                )
            elif self.model.use_dummy_masks:
                resized_image_masks = create_deterministic_dummy_masks(
                    ve_dim, device=passing_device, num_masks=self.model.number_of_masks
                )
            else:
                resized_image_masks = [
                    downsample_mask_to_1d_counts(mask, ve_dim, device=passing_device)
                    for mask in image_masks
                ]

            if (
                self.model.mask_removing
                and len(resized_image_masks) > self.model.mask_limit
            ):
                raise NotImplementedError("Not yet implemented feature")

            elif (
                self.model.mask_limiting
                and len(resized_image_masks) > self.model.mask_limit
            ):
                # checking consistency between masks and image_features and that all resized masks are valid (i.e. all have at least one true value) and passing to cuda
                resized_image_masks = check_resised_image_masks(
                    resized_image_masks, image_features, device=image_features.device
                )
                masked_features = [
                    image_features[mask]
                    for mask in resized_image_masks[: self.mask_limit]
                ]
            elif self.model.image_filling:

                # 1) check for areas of the feature space which are not covered by the masks
                covered = torch.stack(resized_image_masks, dim=0).any(dim=0).to("cpu")

                # 2) invert to get the “uncovered” positions
                uncovered = ~covered

                # 3) if there really is anything uncovered, append it
                if uncovered.any().item() > 0:
                    resized_image_masks.append(uncovered)

                # 4) checking consistency between masks and image_features and that all resized masks are valid (i.e. all have at least one true value) and passing to cuda
                resized_image_masks = check_resised_image_masks(
                    resized_image_masks, image_features, device=image_features.device
                )

                # 5) now apply all masks (including the dummy one) to extract features
                masked_features = [image_features[mask] for mask in resized_image_masks]
            else:
                # checking consistency between masks and image_features and that all resized masks are valid (i.e. all have at least one true value) and passing to cuda
                resized_image_masks = check_resised_image_masks(
                    resized_image_masks, image_features, device=image_features.device
                )
                if getattr(self.model, "absolute_positional_encoding", False):
                    self.abs_pos_embed = self.abs_pos_embed.to(image_features.device)
                    masked_features = [
                        image_features[mask] + self.abs_pos_embed[mask]
                        for mask in resized_image_masks
                    ]

                elif getattr(self.model, "learnable_positional_encoding", False):
                    self.learn_pos_embed = self.learn_pos_embed.to(
                        image_features.device
                    )

                    masked_features = [
                        image_features[mask]
                        + self.learn_pos_embed(mask.nonzero(as_tuple=True)[0])
                        for mask in resized_image_masks
                    ]
                else:
                    masked_features = [
                        image_features[mask] for mask in resized_image_masks
                    ]

            if self.model.averaging:
                masked_features = [
                    torch.mean(mask_feat, dim=0).reshape(1, feature_dim)
                    for mask_feat in masked_features
                ]  # averaging across 576 --> (bs, feature_dim)

            masked_with_tokens = self._apply_bom_tokens(
                image_features, masked_features, ve_dim, feature_dim
            )

            if getattr(self.get_model, "custom_rotary_embedding", False):
                batch_ranges_single_image = self._compute_batch_ranges(
                    masked_features=masked_features,
                    image_features=image_features,
                    mm_bom_mask_token=(
                        self.get_model.mm_bom_mask_token
                        if hasattr(self.get_model, "mm_bom_mask_token")
                        else self.model.mm_bom_mask_token
                    ).mm_bom_mask_token,
                )
                batch_ranges.append(batch_ranges_single_image)
            else:
                batch_ranges = []
            # Shape (bs, (1 + n) * 576 + 2n, 1024)
            batched_features.append(masked_with_tokens.to(image_features.device))

        padding = True
        if padding:
            batched_features = self._pad_batched_feat(
                image_features, batched_features, padding
            )

        final_output = torch.stack(batched_features).to(images_batch.device)

        return final_output, batch_ranges

    def forward(self, image_features, masks):
        """
        Passes images through masks application process to the image features.

        Parameters:
            image_features (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) outputted from Vision Encoder.
            masks (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
                                        the inner Tensors are of size (n. of masks, (image_h, image_w)).

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
        """
        ## apply masks with bom tokens
        output_features, batch_ranges = self._apply_masks_with_tokens(
            image_features, masks
        )

        return output_features, batch_ranges


# class BatchedMaskEmbedder(torch.nn.Module):

#     def __init__(
#         self,
#         model,
#         get_model,
#         config,
#         vision_tower,
#         global_view=False,
#         averaging=False,
#         mask_removing=False,
#         mask_limiting=False,
#         mask_limit=20,
#         averaging_global_view=False,
#         no_masktoken=False,
#         use_sliding_window=False,
#         use_dummy_masks=False,
#         number_of_masks=10,
#     ):
#         super(BatchedMaskEmbedder, self).__init__()

#         # objects
#         self.model = model
#         self.get_model = get_model
#         self.config = config
#         self.vision_encoder = vision_tower

#         # bool values
#         self.global_view = global_view
#         self.mask_removing = mask_removing
#         self.averaging = averaging
#         self.mask_limiting = mask_limiting
#         self.averaging_global_view = averaging_global_view
#         self.no_masktoken = no_masktoken

#         # bool values for ablations
#         self.use_sliding_window = use_sliding_window
#         self.use_dummy_masks = use_dummy_masks
#         self.number_of_masks = number_of_masks

#         # other
#         self.mask_limit = mask_limit

#     @property
#     def dtype(self):
#         return self.vision_encoder.dtype

#     @property
#     def device(self):
#         return self.vision_encoder.device

#     def downsample_masks(
#         self,
#         images_features,
#         masks_batch,
#     ):
#         """
#         Applies masks to the image features and appends special tokens signaling the masked feature start/end.

#         Parameters:
#             images_features (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
#             masks_batch (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
#                                         the inner Tensors are of size (n. of masks, (image_h, image_w)).

#         Returns:
#             torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
#         """

#         batch_size, ve_dim, feature_dim = images_features.shape
#         # Create a tensor by padding masks_batch
#         # Step 1: Determine the target shape for each dimension
#         target_nummasks = max(mask.shape[0] for mask in masks_batch)
#         target_height, target_width = self.image_h, self.image_w
#         # Step 2: Pad each mask to the target shape
#         padded_masks = [
#             adjust_tensor_size(mask, target_nummasks, target_height, target_width)
#             for mask in masks_batch
#         ]
#         masks_batch = torch.stack(padded_masks)

#         if self.use_sliding_window:
#             resized_image_masks = torch.stack(
#                 [
#                     torch.tensor(
#                         create_sliding_masks(
#                             ve_dim,
#                             device=images_features.device,
#                             num_masks=self.number_of_masks,
#                         )
#                     )
#                     for i in range(batch_size)
#                 ]
#             ).to(images_features.device)
#         elif self.use_dummy_masks:
#             resized_image_masks = torch.stack(
#                 [
#                     torch.tensor(
#                         create_deterministic_dummy_masks(
#                             ve_dim,
#                             device=images_features.device,
#                             num_masks=self.number_of_masks,
#                         )
#                     )
#                     for i in range(batch_size)
#                 ]
#             ).to(images_features.device)
#         else:
#             resized_image_masks = torch.stack(
#                 [
#                     downsample_mask_to_1d_counts(
#                         mask, ve_dim, device=images_features.device
#                     ).to(images_features.device)
#                     for image_masks in masks_batch
#                     for mask in image_masks
#                 ]
#             )
#             resized_image_masks = resized_image_masks.view(
#                 batch_size, target_nummasks, -1
#             ).to(images_features.device)

#         return resized_image_masks

#     def apply_masks(self, images_features, resized_image_masks):
#         """
#         Applies masks to the image features.

#         Parameters:
#             images_features (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
#             resized_image_masks (torch.Tensor): Tensor of shape (batch_size, M, vision_encoder_dim). M is the maximum number of masks across the batch.

#         Returns:
#             torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, M, y, feature_dim), where y varies based on the method / number of masks for each image and number of True values per mask.
#         """
#         if self.global_view:
#             ## include a mask which is f
#             whole_image_mask = torch.ones(
#                 (batch_size, 1, ve_dim),
#                 dtype=resized_image_masks.dtype,
#                 device=resized_image_masks.device,
#             )
#             resized_image_masks = torch.cat(
#                 [whole_image_mask, resized_image_masks], dim=1
#             )

#         batch_size, M, ve_dim = tuple(resized_image_masks.shape)
#         F = images_features.shape[-1]  # aka feature_dim

#         # 1. Count the number of True entries per [B, M] and find the maximum count.
#         counts = resized_image_masks.sum(dim=-1)  # shape: [B, M]
#         self.max_count = int(
#             counts.max().item()
#         )  # scalar (largest number of True values)

#         # 2. Expand images_features so that each mask sees [ve_dim, F]:
#         #    From [B, ve_dim, F] --> [B, 1, ve_dim, F] and then expand to [B, M, ve_dim, F]
#         image_features_expanded = images_features.unsqueeze(1).expand(
#             batch_size, M, ve_dim, F
#         )

#         # 3. Compute an ordering index for the D-dimension that preserves original order.
#         #    Create an index tensor for positions 0...D-1.
#         order = (
#             torch.arange(ve_dim, device=images_features.device)
#             .view(1, 1, ve_dim)
#             .expand(batch_size, M, ve_dim)
#         )
#         # For positions where the mask is False, substitute a large value (here, D).
#         order_masked = torch.where(
#             resized_image_masks, order, torch.full_like(order, ve_dim)
#         )
#         # When you sort order_masked along the D dimension (ascending), the True indices (with
#         # their original order) will appear first and the False ones will be pushed to the end.
#         _, sort_indices = torch.sort(order_masked, dim=-1)

#         # 4. Reorder the image features using the same sort order from the masks.
#         #    sort_indices has shape [B, M, D]. We use it to gather along the D dimension.
#         sorted_features = torch.gather(
#             image_features_expanded,
#             2,
#             sort_indices.unsqueeze(-1).expand(batch_size, M, ve_dim, F),
#         )
#         # Now, for each [B, M] pair, the first counts[b, m] rows are from the True positions in order.

#         # 5. Slice the image_features to keep only up to max_count rows along the D (now position) dimension.
#         selected_features = sorted_features[
#             :, :, : self.max_count, :
#         ]  # shape: [B, M, max_count, F]

#         # 6. Create a mask to zero out any “padding” positions:
#         #    For each [B, M] pair, we know how many valid rows there are (counts[b, m]).
#         #    Create a positions index for the new (padded) dimension.
#         positions = (
#             torch.arange(self.max_count, device=images_features.device)
#             .view(1, 1, self.max_count)
#             .expand(batch_size, M, self.max_count)
#         )
#         valid_mask = positions < counts.unsqueeze(-1)  # shape: [B, M, max_count]
#         # Finally Apply the valid_mask along the feature dimension.
#         no_tokens_features = selected_features * valid_mask.unsqueeze(-1).float()
#         return no_tokens_features

#     def add_visual_tokens(self, no_tokens_features):
#         batch_size, M, y, F = no_tokens_features.shape

#         if hasattr(self.get_model, "mm_bom_mask_token"):
#             bom_token_expanded = (
#                 self.get_model.mm_bom_mask_token.mm_bom_mask_token.unsqueeze(
#                     0
#                 ).unsqueeze(0)
#             )  # shape: [1, 1, 1, F]
#             bom_tokens = bom_token_expanded.expand(
#                 batch_size, M, 1, F
#             )  # shape: [B, M, 1, F]
#         else:
#             bom_token_expanded = (
#                 self.model.mm_bom_mask_token.mm_bom_mask_token.unsqueeze(0).unsqueeze(0)
#             )  # shape: [1, 1, 1, F]
#             bom_tokens = bom_token_expanded.expand(
#                 batch_size, M, 1, F
#             )  # shape: [B, M, 1, F]

#         # Now prepend the BOM tokens along the sequence (3rd) dimension.
#         final_features_with_bom = torch.cat([bom_tokens, no_tokens_features], dim=2)
#         # Reshape to combine the mask and sequence dimensions:
#         final_features = final_features_with_bom.reshape(
#             batch_size, M * (1 + self.max_count), F
#         )
#         # result now has shape [B, M * (1 + max_count), F]
#         return final_features

#     def forward(self, images, masks):
#         """
#         Passes images through Vision Encoder and applies masks to the image features along with special tokens signaling the masked feature start/end.

#         Parameters:
#             images (torch.Tensor): Tensor of shape (batch_size, channels, image_height, image_width) preprocessed and ready for Vision Encoder.
#             masks (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
#                                         the inner Tensors are of size (n. of masks, (image_h, image_w)).

#         Returns:
#             torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
#         """

#         # Get shape information from images
#         self.image_h, self.image_w = int(images.shape[2]), int(images.shape[3])

#         ## Pass images through vision_encoder
#         images_features = self.vision_encoder(images)

#         ## Downsample masks
#         downsampled_masks = self.downsample_masks(images_features, masks)

#         ## Apply Downsampled masks to image features
#         no_tokens_features = self.apply_masks(images_features, downsampled_masks)

#         ## Add bom tokens
#         if self.no_masktoken:
#             return no_tokens_features.view(
#                 no_tokens_features.size(0), -1, no_tokens_features.size(3)
#             )
#         else:
#             final_features = self.add_visual_tokens(no_tokens_features)
#             return final_features
