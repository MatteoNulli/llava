import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.ops as ops


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

    def __init__(self, model, get_model, config, vision_tower):
        super(VisualTokenEmbedding, self).__init__()

        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower

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
        visual_token_embeds = (
            self.get_model.mm_projector(visual_token_embeds) * not_padding
        )

        return (
            visual_token_embeds,
            features,
            not_padding.squeeze(-1).sum(dim=-1).cpu().numpy(),
        )
