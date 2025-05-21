import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops
from typing import List, Callable, Optional, Tuple, Union, Sequence


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


def boxes_xyxy_to_xywh(boxes):
    # boxes: (N, M, 4) - x1, y1, x2, y2
    # convert to center x, center y, width, height
    xywh = boxes.clone()
    xywh[:, :, 2] = boxes[:, :, 2] - boxes[:, :, 0]  # width
    xywh[:, :, 3] = boxes[:, :, 3] - boxes[:, :, 1]  # height
    xywh[:, :, 0] = boxes[:, :, 0] + 0.5 * xywh[:, :, 2]  # center x
    xywh[:, :, 1] = boxes[:, :, 1] + 0.5 * xywh[:, :, 3]  # center y
    return xywh


class VisualTokenEmbedding(torch.nn.Module):

    def __init__(self, model, get_model, config, vision_tower, no_masks: bool = False):
        super(VisualTokenEmbedding, self).__init__()

        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower
        self.no_masks = no_masks

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    def forward(self, images, batch_masks):
        """
        Forward pass of the visual token embedding model.
        Args:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, channels, H, W).
            batch_masks (List[torch.Tensor]): A list of length batch_size of tensors of shape (number_of_masks, H, W) containing binary masks.

        Returns:
            roi_boxes  (torch.Tensor): A tensor of shape (batch_size, number_of_masks, 4) containing the bounding boxes of each mask.
            roi_masks  (torch.Tensor): A tensor of shape (batch_size, number_of_masks, token_roi_resolution, token_roi_resolution) containing the cropped masks.
            embeddings (torch.Tensor): A tensor of shape (batch_size, number_of_masks, channels * token_roi_resolution * token_roi_resolution) containing the visual token embeddings.
        """
        self.output_resolution = int(images.shape[2])
        batch_features = self.vision_encoder(
            images
        )  # output shape is (batch_size, VE_output, feature_dim)
        # Determine spatial size for upsamplign
        if self.no_masks:
            return batch_features, [], []
        spatial_size = int(batch_features.shape[1] ** 0.5)  # from 576 to 24x24.

        # Reshape the tensor to [batch_size, H, W, feature_dim]
        batch_features = batch_features.reshape(
            int(images.shape[0]), spatial_size, spatial_size, self.config.mm_hidden_size
        )

        # Permute to channels-first format: [B, feature_dim, spatial_size, spatial_size]
        batch_features = batch_features.permute(0, 3, 1, 2)

        # upsample batch_features to output_resolution:
        # N, C, spatial_size, spatial_size -> N, C, output_resolution, output_resolution
        # output_resolution is the original image H and W after vision encoder preprocessing (for clip is 336)
        batch_features = F.interpolate(
            batch_features,
            size=(self.output_resolution, self.output_resolution),
            mode="bilinear",
        )
        # Step 1: Determine the target shape for each dimension
        target_nummasks = max(mask.shape[0] for mask in batch_masks)
        target_height, target_width = self.output_resolution, self.output_resolution
        # Step 2: Pad each mask to the target shape
        padded_masks = [
            adjust_tensor_size(mask, target_nummasks, target_height, target_width)
            for mask in batch_masks
        ]

        batch_masks = torch.stack(padded_masks)

        batch_masks = batch_masks.to(batch_features.device).to(batch_features.dtype)

        roi_boxes, roi_masks, embeddings = self.mask_roi_pooling(
            batch_features, batch_masks
        )
        return roi_boxes, roi_masks, embeddings

    def mask_roi_pooling(self, batch_features, batch_masks):

        N, C, _resolution, _resolution = batch_features.shape
        _N, M, resolution, _resolution = batch_masks.shape
        dtype = batch_features.dtype

        # Get ROI boxes for each mask
        roi_boxes = self.get_roi_boxes_from_masks(batch_masks)

        # Perform ROIAlign for features
        roi_features = ops.roi_align(
            batch_features.float(),
            roi_boxes,
            output_size=(
                self.config.token_roi_resolution,
                self.config.token_roi_resolution,
            ),
            sampling_ratio=1,
        ).view(
            N, M, C, self.config.token_roi_resolution, self.config.token_roi_resolution
        )
        # breakpoint()

        # Perform ROIAlign for masks
        roi_masks = self.crop_roi_masks(
            batch_masks, roi_boxes, self.config.token_roi_resolution
        ).to(roi_features.device, dtype=roi_features.dtype)
        # roi_masks Shape: (N, M, 1, token_roi_resolution, token_roi_resolution)

        embeddings = self.average_pool(roi_features, roi_masks)

        return (
            torch.stack(roi_boxes) / resolution,
            roi_masks[:, :, 0],
            embeddings.to(dtype),
        )

    def average_pool(self, roi_features, roi_masks):
        # Apply mask to the features, and average pool
        roi_features = roi_features * roi_masks
        mask_sum = roi_masks.sum(dim=(-2, -1)).clamp(min=1e-6)

        feature_sum = roi_features.sum(dim=(-2, -1))
        embeddings = feature_sum / mask_sum

        return embeddings

    def get_roi_boxes_from_masks(self, batch_masks):
        N, M, H, W = batch_masks.shape

        y_coords = (
            torch.arange(H, device=batch_masks.device)
            .view(1, 1, H, 1)
            .expand(N, M, H, W)
        )
        x_coords = (
            torch.arange(W, device=batch_masks.device)
            .view(1, 1, 1, W)
            .expand(N, M, H, W)
        )

        mask = batch_masks > 0

        max_int = torch.iinfo(torch.int64).max
        min_int = torch.iinfo(torch.int64).min

        y_min = (
            torch.where(mask, y_coords, torch.full_like(y_coords, max_int))
            .view(N, M, -1)
            .min(dim=-1)
            .values
        )
        y_max = (
            torch.where(mask, y_coords, torch.full_like(y_coords, min_int))
            .view(N, M, -1)
            .max(dim=-1)
            .values
        )
        x_min = (
            torch.where(mask, x_coords, torch.full_like(x_coords, max_int))
            .view(N, M, -1)
            .min(dim=-1)
            .values
        )
        x_max = (
            torch.where(mask, x_coords, torch.full_like(x_coords, min_int))
            .view(N, M, -1)
            .max(dim=-1)
            .values
        )

        # Handle empty masks
        mask_sums = batch_masks.view(N, M, -1).sum(dim=-1)
        empty_masks = mask_sums == 0

        # Expand bounding boxes by 1 pixel and clip to image boundaries
        x_min = torch.clamp(x_min, min=0)
        y_min = torch.clamp(y_min, min=0)
        x_max = torch.clamp(x_max + 1, max=W - 1)
        y_max = torch.clamp(y_max + 1, max=H - 1)

        # Combine into bounding boxes
        roi_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        # Set empty mask boxes to [0, 0, 0, 0]
        roi_boxes[empty_masks] = 0

        return [box.float() for box in roi_boxes]

    def crop_roi_masks(self, batch_masks, roi_boxes, token_roi_resolution):
        N, M, H, W = batch_masks.shape
        device = batch_masks.device
        dtype = batch_masks.dtype

        # Flatten the batch and mask dimensions
        batch_masks_flat = batch_masks.reshape(N * M, H, W).unsqueeze(
            1
        )  # Shape: (N*M, 1, H, W)

        # Prepare the boxes tensor with correct batch indices
        # roi_boxes is a list of length N, each with shape (M, 4)
        # Stack roi_boxes into a single tensor of shape (N*M, 4)
        roi_boxes_tensor = torch.cat(roi_boxes, dim=0).to(
            device=device, dtype=torch.long
        )  # Shape: (N*M, 4)
        batch_indices = torch.arange(N * M, device=device).unsqueeze(1).type(dtype)
        boxes = torch.cat([batch_indices, roi_boxes_tensor], dim=1)  # Shape: (N*M, 5)

        # Perform roi_align on the masks
        cropped_masks = ops.roi_align(
            batch_masks_flat.float(),  # Ensure the masks are in float
            boxes.float(),
            output_size=token_roi_resolution,
            spatial_scale=1.0,  # Masks are in the same scale
            sampling_ratio=0,
            aligned=True,
        )  # Output shape: (N*M, C, token_roi_resolution, token_roi_resolution)
        cropped_masks = cropped_masks.reshape(
            N, M, 1, token_roi_resolution, token_roi_resolution
        )

        return cropped_masks > 0

    def prepare_visual_embeds(self, images, masks):
        """
        Added function taken from https://github.com/ChenDelong1999/subobjects-VLM/blob/main/model/modeling.py
        Used to pass the visual embedding into the MLP projector.

        Args:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, channels, H, W).
            masks (List[torch.Tensor]): A list of length batch_size of tensors of shape (number_of_masks, H, W) containing binary masks.

        Returns:
            visual_token_embends (torch.Tensor): Output features to be passed to the Language Model. A tensor of shape (batch_size, number_of_masks, projector_output_features).
            features (torch.Tensor): Output features from the embedding. A tensor of shape (batch_size, number_of_masks, projector_input_features).
            n_visual_tokens (torch.Tensor): A tensor of shape (batch_size, ) containing the number of images in the batch.
        """
        ## The pass to the vision encoder happens within the forward pass
        if self.no_masks:
            visual_token_embeds, _, _ = self.forward(images, masks)
        else:
            boxes, masks, features = self.forward(images, masks)
            # boxes:    (N, M, 4)
            # masks:    (N, M, token_roi_resolution, token_roi_resolution)
            # features: (N, M, C * token_roi_resolution * token_roi_resolution)

            boxes = boxes.to(self.dtype).to(self.device)
            masks = masks.to(self.dtype).to(self.device)
            features = features.to(self.dtype).to(self.device)

            not_padding = (boxes.sum(dim=-1) != 0).unsqueeze(-1)

            # box_embeds = self.box_embed(self.boxes_xyxy_to_xywh(boxes)) * not_padding
            # mask_embeds = self.mask_embed(masks.view(masks.shape[0], masks.shape[1], -1)) * not_padding
            # feature_embeds = self.feature_embed(features) * not_padding

            # concant and project
            box_embeds = boxes_xyxy_to_xywh(boxes)
            mask_embeds = masks.view(masks.shape[0], masks.shape[1], -1)

            # concat
            visual_token_embeds = torch.cat((box_embeds, mask_embeds, features), dim=-1)
        # project
        if hasattr(self.get_model, "mm_subobject_projector"):
            visual_token_embeds = (
                self.get_model.mm_subobject_projector(visual_token_embeds) * not_padding
            )
        else:
            visual_token_embeds = (
                self.get_model.mm_projector(visual_token_embeds) * not_padding
            )

        return (
            visual_token_embeds,
            features,
            not_padding.squeeze(-1).sum(dim=-1).cpu().numpy(),
        )


class MaskingVisualTokenEmbedding(torch.nn.Module):

    def __init__(self, model, get_model, config, vision_tower, no_masks: bool = False):
        super(VisualTokenEmbedding, self).__init__()

        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower
        self.no_masks = no_masks

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

                masked_features = [image_features[mask] for mask in resized_image_masks]

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

    def forward(self, images, batch_masks):
        """
        Forward pass of the visual token embedding model.
        Args:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, channels, H, W).
            batch_masks (List[torch.Tensor]): A list of length batch_size of tensors of shape (number_of_masks, H, W) containing binary masks.

        Returns:
            roi_boxes  (torch.Tensor): A tensor of shape (batch_size, number_of_masks, 4) containing the bounding boxes of each mask.
            roi_masks  (torch.Tensor): A tensor of shape (batch_size, number_of_masks, token_roi_resolution, token_roi_resolution) containing the cropped masks.
            embeddings (torch.Tensor): A tensor of shape (batch_size, number_of_masks, channels * token_roi_resolution * token_roi_resolution) containing the visual token embeddings.
        """
        self.output_resolution = int(images.shape[2])
        batch_features = self.vision_encoder(
            images
        )  # output shape is (batch_size, VE_output, feature_dim)
        # Determine spatial size for upsamplign
        batch_features = self._apply_masks_with_tokens(batch_features, batch_masks)

        roi_boxes, roi_masks, embeddings = self.mask_roi_pooling(
            batch_features, batch_masks
        )
        return roi_boxes, roi_masks, embeddings

    def mask_roi_pooling(self, batch_features, batch_masks):

        N, C, _resolution, _resolution = batch_features.shape
        _N, M, resolution, _resolution = batch_masks.shape
        dtype = batch_features.dtype

        # Get ROI boxes for each mask
        roi_boxes = self.get_roi_boxes_from_masks(batch_masks)

        # Perform ROIAlign for features
        roi_features = ops.roi_align(
            batch_features.float(),
            roi_boxes,
            output_size=(
                self.config.token_roi_resolution,
                self.config.token_roi_resolution,
            ),
            sampling_ratio=1,
        ).view(
            N, M, C, self.config.token_roi_resolution, self.config.token_roi_resolution
        )

        # Perform ROIAlign for masks
        roi_masks = self.crop_roi_masks(
            batch_masks, roi_boxes, self.config.token_roi_resolution
        ).to(roi_features.device, dtype=roi_features.dtype)
        # roi_masks Shape: (N, M, 1, token_roi_resolution, token_roi_resolution)

        embeddings = self.average_pool(roi_features, roi_masks)

        return (
            torch.stack(roi_boxes) / resolution,
            roi_masks[:, :, 0],
            embeddings.to(dtype),
        )

    def average_pool(self, roi_features, roi_masks):
        # Apply mask to the features, and average pool
        roi_features = roi_features * roi_masks
        mask_sum = roi_masks.sum(dim=(-2, -1)).clamp(min=1e-6)

        feature_sum = roi_features.sum(dim=(-2, -1))
        embeddings = feature_sum / mask_sum

        return embeddings

    def get_roi_boxes_from_masks(self, batch_masks):
        N, M, H, W = batch_masks.shape

        y_coords = (
            torch.arange(H, device=batch_masks.device)
            .view(1, 1, H, 1)
            .expand(N, M, H, W)
        )
        x_coords = (
            torch.arange(W, device=batch_masks.device)
            .view(1, 1, 1, W)
            .expand(N, M, H, W)
        )

        mask = batch_masks > 0

        max_int = torch.iinfo(torch.int64).max
        min_int = torch.iinfo(torch.int64).min

        y_min = (
            torch.where(mask, y_coords, torch.full_like(y_coords, max_int))
            .view(N, M, -1)
            .min(dim=-1)
            .values
        )
        y_max = (
            torch.where(mask, y_coords, torch.full_like(y_coords, min_int))
            .view(N, M, -1)
            .max(dim=-1)
            .values
        )
        x_min = (
            torch.where(mask, x_coords, torch.full_like(x_coords, max_int))
            .view(N, M, -1)
            .min(dim=-1)
            .values
        )
        x_max = (
            torch.where(mask, x_coords, torch.full_like(x_coords, min_int))
            .view(N, M, -1)
            .max(dim=-1)
            .values
        )

        # Handle empty masks
        mask_sums = batch_masks.view(N, M, -1).sum(dim=-1)
        empty_masks = mask_sums == 0

        # Expand bounding boxes by 1 pixel and clip to image boundaries
        x_min = torch.clamp(x_min, min=0)
        y_min = torch.clamp(y_min, min=0)
        x_max = torch.clamp(x_max + 1, max=W - 1)
        y_max = torch.clamp(y_max + 1, max=H - 1)

        # Combine into bounding boxes
        roi_boxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)

        # Set empty mask boxes to [0, 0, 0, 0]
        roi_boxes[empty_masks] = 0

        return [box.float() for box in roi_boxes]

    def crop_roi_masks(self, batch_masks, roi_boxes, token_roi_resolution):
        N, M, H, W = batch_masks.shape
        device = batch_masks.device
        dtype = batch_masks.dtype

        # Flatten the batch and mask dimensions
        batch_masks_flat = batch_masks.reshape(N * M, H, W).unsqueeze(
            1
        )  # Shape: (N*M, 1, H, W)

        # Prepare the boxes tensor with correct batch indices
        # roi_boxes is a list of length N, each with shape (M, 4)
        # Stack roi_boxes into a single tensor of shape (N*M, 4)
        roi_boxes_tensor = torch.cat(roi_boxes, dim=0).to(
            device=device, dtype=torch.long
        )  # Shape: (N*M, 4)
        batch_indices = torch.arange(N * M, device=device).unsqueeze(1).type(dtype)
        boxes = torch.cat([batch_indices, roi_boxes_tensor], dim=1)  # Shape: (N*M, 5)

        # Perform roi_align on the masks
        cropped_masks = ops.roi_align(
            batch_masks_flat.float(),  # Ensure the masks are in float
            boxes.float(),
            output_size=token_roi_resolution,
            spatial_scale=1.0,  # Masks are in the same scale
            sampling_ratio=0,
            aligned=True,
        )  # Output shape: (N*M, C, token_roi_resolution, token_roi_resolution)
        cropped_masks = cropped_masks.reshape(
            N, M, 1, token_roi_resolution, token_roi_resolution
        )

        return cropped_masks > 0

    def prepare_visual_embeds(self, images, masks):
        """
        Added function taken from https://github.com/ChenDelong1999/subobjects-VLM/blob/main/model/modeling.py
        Used to pass the visual embedding into the MLP projector.

        Args:
            images (torch.Tensor): Tensor of input images to the vision encoder shape (batch_size, channels, H, W).
            masks (List[torch.Tensor]): A list of length batch_size of tensors of shape (number_of_masks, H, W) containing binary masks.

        Returns:
            visual_token_embends (torch.Tensor): Output features to be passed to the Language Model. A tensor of shape (batch_size, number_of_masks, projector_output_features).
            features (torch.Tensor): Output features from the embedding. A tensor of shape (batch_size, number_of_masks, projector_input_features).
            n_visual_tokens (torch.Tensor): A tensor of shape (batch_size, ) containing the number of images in the batch.
        """
        ## The pass to the vision encoder happens within the forward pass
        if self.no_masks:
            visual_token_embeds, _, _ = self.forward(images, masks)
        else:
            boxes, masks, features = self.forward(images, masks)
            # boxes:    (N, M, 4)
            # masks:    (N, M, token_roi_resolution, token_roi_resolution)
            # features: (N, M, C * token_roi_resolution * token_roi_resolution)

            boxes = boxes.to(self.dtype).to(self.device)
            masks = masks.to(self.dtype).to(self.device)
            features = features.to(self.dtype).to(self.device)

            not_padding = (boxes.sum(dim=-1) != 0).unsqueeze(-1)

            # box_embeds = self.box_embed(self.boxes_xyxy_to_xywh(boxes)) * not_padding
            # mask_embeds = self.mask_embed(masks.view(masks.shape[0], masks.shape[1], -1)) * not_padding
            # feature_embeds = self.feature_embed(features) * not_padding

            # concant and project
            box_embeds = boxes_xyxy_to_xywh(boxes)
            mask_embeds = masks.view(masks.shape[0], masks.shape[1], -1)
            breakpoint()

            # concat
            visual_token_embeds = torch.cat((box_embeds, mask_embeds, features), dim=-1)
        # project
        if hasattr(self.get_model, "mm_subobject_projector"):
            visual_token_embeds = (
                self.get_model.mm_subobject_projector(visual_token_embeds) * not_padding
            )
        else:
            visual_token_embeds = (
                self.get_model.mm_projector(visual_token_embeds) * not_padding
            )

        return (
            visual_token_embeds,
            features,
            not_padding.squeeze(-1).sum(dim=-1).cpu().numpy(),
        )
