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


def downsample_mask_to_1d_counts(
    mask: torch.Tensor, output_size: int, threshold_count: float = 0.5
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
    # print("pooled device", pooled.device, pooled.shape)

    # Convert the fraction to an estimated count per bin.
    counts = pooled * bin_size

    # Binarize: mark a bin as True if the estimated count is at least threshold_count.
    downsampled_mask = counts >= threshold_count
    return downsampled_mask


def create_deterministic_dummy_masks(shape, num_masks=10, seed=42):
    """
    Creates a binary tensor (array of 1s and 0s) with a deterministic random pattern.

    Parameters:
    - shape: Integer or tuple of integers defining the shape of the tensor
            (e.g., 576 for a 1D array or (24, 24) for a 2D array)
    - seed: Random seed to ensure reproducibility (default: 42)

    Returns:
    - A numpy array with the specified shape filled with random 1s and 0s
    """
    # Convert shape to tuple if it's a single integer
    if isinstance(shape, int):
        shape = (shape,)

    # Set the random seed to ensure deterministic results
    masks = []
    for i in range(num_masks):
        # Set the random seed to ensure deterministic results
        np.random.seed(seed)
        # Generate a binary tensor with random 1s and 0s
        # Generate a boolean tensor with random True and False values
        boolean_tensor = np.random.choice([True, False], size=shape)
        masks.append(boolean_tensor)

    return np.array(masks)


def create_sliding_masks(ve_dim, num_masks=10):
    masks = []
    patch_size = ve_dim // num_masks
    for i in range(num_masks):
        mask = np.zeros(ve_dim, dtype=bool)
        # For the last mask, include any remainder
        start = i * patch_size
        end = (i + 1) * patch_size if i < num_masks - 1 else ve_dim
        mask[start:end] = True
        masks.append(mask)

    return np.array(masks)


class BatchedMaskEmbedder(torch.nn.Module):

    def __init__(
        self,
        model,
        get_model,
        config,
        vision_tower,
        global_view=False,
        averaging=False,
        mask_removing=False,
        mask_limiting=False,
        mask_limit=20,
        averaging_global_view=False,
        no_masktoken=False,
        use_sliding_window=False,
        use_dummy_masks=False,
    ):
        super(BatchedMaskEmbedder, self).__init__()

        # objects
        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower

        # bool values
        self.global_view = global_view
        self.mask_removing = mask_removing
        self.averaging = averaging
        self.mask_limiting = mask_limiting
        self.averaging_global_view = averaging_global_view
        self.no_masktoken = no_masktoken

        # bool values for ablations
        self.use_sliding_window = use_sliding_window
        self.use_dummy_masks = use_dummy_masks

        # other
        self.mask_limit = mask_limit

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    def downsample_masks(
        self,
        images_features,
        masks_batch,
    ):
        """
        Applies masks to the image features and appends special tokens signaling the masked feature start/end.

        Parameters:
            images_features (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
            masks_batch (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
                                        the inner Tensors are of size (n. of masks, (image_h, image_w)).

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
        """

        batch_size, ve_dim, feature_dim = images_features.shape
        # Create a tensor by padding masks_batch
        # Step 1: Determine the target shape for each dimension
        target_nummasks = max(mask.shape[0] for mask in masks_batch)
        target_height, target_width = self.image_h, self.image_w
        # Step 2: Pad each mask to the target shape
        padded_masks = [
            adjust_tensor_size(mask, target_nummasks, target_height, target_width)
            for mask in masks_batch
        ]
        masks_batch = torch.stack(padded_masks)

        if self.use_sliding_window:
            resized_image_masks = torch.stack(
                [
                    torch.tensor(create_sliding_masks(ve_dim, num_masks=10))
                    for i in range(batch_size)
                ]
            ).to(images_features.device)
        elif self.use_dummy_masks:
            resized_image_masks = torch.stack(
                [
                    torch.tensor(create_deterministic_dummy_masks(ve_dim, num_masks=10))
                    for i in range(batch_size)
                ]
            ).to(images_features.device)
        else:
            resized_image_masks = torch.stack(
                [
                    downsample_mask_to_1d_counts(mask, ve_dim).to(
                        images_features.device
                    )
                    for image_masks in masks_batch
                    for mask in image_masks
                ]
            )
            resized_image_masks = resized_image_masks.view(
                batch_size, target_nummasks, -1
            ).to(images_features.device)

        return resized_image_masks

    def apply_masks(self, images_features, resized_image_masks):
        """
        Applies masks to the image features.

        Parameters:
            images_features (torch.Tensor): Tensor of shape (batch_size, vision_encoder_dim, feature_dim) from Vision Encoder embeddings.
            resized_image_masks (torch.Tensor): Tensor of shape (batch_size, M, vision_encoder_dim). M is the maximum number of masks across the batch.

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, M, y, feature_dim), where y varies based on the method / number of masks for each image and number of True values per mask.
        """
        batch_size, M, ve_dim = tuple(resized_image_masks.shape)
        F = images_features.shape[-1]  # aka feature_dim

        # 1. Count the number of True entries per [B, M] and find the maximum count.
        counts = resized_image_masks.sum(dim=-1)  # shape: [B, M]
        self.max_count = int(
            counts.max().item()
        )  # scalar (largest number of True values)

        # 2. Expand images_features so that each mask sees [ve_dim, F]:
        #    From [B, ve_dim, F] → [B, 1, ve_dim, F] and then expand to [B, M, ve_dim, F]
        image_features_expanded = images_features.unsqueeze(1).expand(
            batch_size, M, ve_dim, F
        )

        # 3. Compute an ordering index for the D-dimension that preserves original order.
        #    Create an index tensor for positions 0...D-1.
        order = (
            torch.arange(ve_dim, device=images_features.device)
            .view(1, 1, ve_dim)
            .expand(batch_size, M, ve_dim)
        )
        # For positions where the mask is False, substitute a large value (here, D).
        order_masked = torch.where(
            resized_image_masks, order, torch.full_like(order, ve_dim)
        )
        # When you sort order_masked along the D dimension (ascending), the True indices (with
        # their original order) will appear first and the False ones will be pushed to the end.
        _, sort_indices = torch.sort(order_masked, dim=-1)

        # 4. Reorder the image features using the same sort order from the masks.
        #    sort_indices has shape [B, M, D]. We use it to gather along the D dimension.
        sorted_features = torch.gather(
            image_features_expanded,
            2,
            sort_indices.unsqueeze(-1).expand(batch_size, M, ve_dim, F),
        )
        # Now, for each [B, M] pair, the first counts[b, m] rows are from the True positions in order.

        # 5. Slice the image_features to keep only up to max_count rows along the D (now position) dimension.
        selected_features = sorted_features[
            :, :, : self.max_count, :
        ]  # shape: [B, M, max_count, F]

        # 6. Create a mask to zero out any “padding” positions:
        #    For each [B, M] pair, we know how many valid rows there are (counts[b, m]).
        #    Create a positions index for the new (padded) dimension.
        positions = (
            torch.arange(self.max_count, device=images_features.device)
            .view(1, 1, self.max_count)
            .expand(batch_size, M, self.max_count)
        )
        valid_mask = positions < counts.unsqueeze(-1)  # shape: [B, M, max_count]
        # Finally Apply the valid_mask along the feature dimension.
        no_tokens_features = selected_features * valid_mask.unsqueeze(-1).float()

        return no_tokens_features

    def add_visual_tokens(self, no_tokens_features):
        batch_size, M, y, F = no_tokens_features.shape

        if hasattr(self.get_model, "mm_bom_mask_token"):
            bom_token_expanded = (
                self.get_model.mm_bom_mask_token.mm_bom_mask_token.unsqueeze(
                    0
                ).unsqueeze(0)
            )  # shape: [1, 1, 1, F]
            bom_tokens = bom_token_expanded.expand(
                batch_size, M, 1, F
            )  # shape: [B, M, 1, F]
        else:
            bom_token_expanded = (
                self.model.mm_bom_mask_token.mm_bom_mask_token.unsqueeze(0).unsqueeze(0)
            )  # shape: [1, 1, 1, F]
            bom_tokens = bom_token_expanded.expand(
                batch_size, M, 1, F
            )  # shape: [B, M, 1, F]

        # Now prepend the BOM tokens along the sequence (3rd) dimension.
        final_features_with_bom = torch.cat([bom_tokens, no_tokens_features], dim=2)
        # Reshape to combine the mask and sequence dimensions:
        final_features = final_features_with_bom.reshape(
            batch_size, M * (1 + self.max_count), F
        )
        # result now has shape [B, M * (1 + max_count), F]
        return final_features

    def forward(self, images, masks):
        """
        Passes images through Vision Encoder and applies masks to the image features along with special tokens signaling the masked feature start/end.

        Parameters:
            images (torch.Tensor): Tensor of shape (batch_size, channels, image_height, image_width) preprocessed and ready for Vision Encoder.
            masks (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
                                        the inner Tensors are of size (n. of masks, (image_h, image_w)).

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
        """

        # Get shape information from images
        self.image_h, self.image_w = int(images.shape[2]), int(images.shape[3])

        ## Pass images through vision_encoder
        images_features = self.vision_encoder(images)

        ## Downsample masks
        downsampled_masks = self.downsample_masks(images_features, masks)

        ## Apply Downsampled masks to image features
        no_tokens_features = self.apply_masks(images_features, downsampled_masks)

        ## Add bom tokens
        final_features = self.add_visual_tokens(no_tokens_features)

        return final_features


class MaskEmbedder(torch.nn.Module):

    def __init__(
        self,
        model,
        get_model,
        config,
        vision_tower,
        global_view=False,
        averaging=False,
        mask_removing=False,
        mask_limiting=False,
        mask_limit=20,
        averaging_global_view=False,
        no_masktoken=False,
        use_sliding_window=False,
        use_dummy_masks=False,
    ):
        super(MaskEmbedder, self).__init__()

        # objects
        self.model = model
        self.get_model = get_model
        self.config = config
        self.vision_encoder = vision_tower

        # bool values
        self.global_view = global_view
        self.mask_removing = mask_removing
        self.averaging = averaging
        self.mask_limiting = mask_limiting
        self.averaging_global_view = averaging_global_view
        self.no_masktoken = no_masktoken

        # bool values for ablations
        self.use_sliding_window = use_sliding_window
        self.use_dummy_masks = use_dummy_masks

        # other
        self.mask_limit = mask_limit

    @property
    def dtype(self):
        return self.vision_encoder.dtype

    @property
    def device(self):
        return self.vision_encoder.device

    def apply_masks_with_tokens(self, images_batch, masks_batch):
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
        for i, (image_features, image_masks) in enumerate(
            zip(images_batch, masks_batch)
        ):

            # Get image features shape (batch_size, ve_dim, feature_dim)
            ve_dim, feature_dim = image_features.shape
            torch.set_printoptions(threshold=image_masks[0].numel())

            # print("image_masks[0].shape", image_masks[0].shape)

            # print("image_masks[0]", image_masks[0])
            # print("image_masks[1]", image_masks[1])
            # print("image_masks[2]", image_masks[2])
            if self.use_sliding_window:
                resized_image_masks = create_sliding_masks(ve_dim, num_masks=10)
            elif self.use_dummy_masks:
                resized_image_masks = create_deterministic_dummy_masks(
                    ve_dim, num_masks=10
                )
            else:
                resized_image_masks = [
                    downsample_mask_to_1d_counts(mask, ve_dim).to(image_features.device)
                    for mask in image_masks
                ]

            # print("resized_image_masks[0]", resized_image_masks[1])
            # print("resized_image_masks[0]", resized_image_masks[2])

            image_features = image_features.to(images_batch.device)

            if self.mask_removing and len(resized_image_masks) > self.mask_limit:
                masked_features = []
            elif self.mask_limiting and len(resized_image_masks) > self.mask_limit:
                masked_features = [
                    image_features[mask]
                    for mask in resized_image_masks[: self.mask_limit]
                ]
            else:
                masked_features = [image_features[mask] for mask in resized_image_masks]

            if self.averaging:
                masked_features = [
                    torch.mean(mask_feat, dim=0).reshape(1, feature_dim)
                    for mask_feat in masked_features
                ]  # averaging across 576 --> (bs, feature_dim)
            ##appending whole image feature at the beginning of the masks features
            if (
                self.global_view
                and not self.averaging_global_view
                and not self.no_masktoken
            ):
                if hasattr(self.get_model, "mm_bom_mask_token"):
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
                else:
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
                self.global_view
                and self.averaging_global_view
                and not self.no_masktoken
            ):
                if hasattr(self.get_model, "mm_bom_mask_token"):
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

                else:
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
                self.global_view
                and not self.averaging_global_view
                and self.no_masktoken
            ):
                masked_with_tokens = torch.cat(
                    [image_features] + [mask_feat for mask_feat in masked_features],
                    dim=0,
                )
            elif (
                not self.global_view
                and not self.averaging_global_view
                and self.no_masktoken
            ):
                masked_with_tokens = torch.cat(
                    [mask_feat for mask_feat in masked_features], dim=0
                )
            elif self.global_view and self.averaging_global_view and self.no_masktoken:
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
            # Shape (bs, (1 + n) * 576 + 2n, 1024)
            batched_features.append(masked_with_tokens.to(image_features.device))

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

    def forward(self, images, masks):
        """
        Passes images through Vision Encoder and applies masks to the image features along with special tokens signaling the masked feature start/end.

        Parameters:
            images (torch.Tensor): Tensor of shape (batch_size, channels, image_height, image_width) preprocessed and ready for Vision Encoder.
            masks (List[torch.LongTensor]): List of Tensors. The Outer list if of length batch_size length,
                                        the inner Tensors are of size (n. of masks, (image_h, image_w)).

        Returns:
            torch.Tensor: Tensor with the masked features across the batch. Shape should be (batch_size, x, feature_dim), where x varies based on the method / number of masks for each image.
        """
        ## pass images through vision encoder
        images_features = self.vision_encoder(images)

        ## apply masks with bom tokens
        output_features = self.apply_masks_with_tokens(images_features, masks)

        return output_features
