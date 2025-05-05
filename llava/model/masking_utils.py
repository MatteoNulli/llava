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
        number_of_masks=10,
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
        self.number_of_masks = number_of_masks

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
                    torch.tensor(
                        create_sliding_masks(
                            ve_dim,
                            device=images_features.device,
                            num_masks=self.number_of_masks,
                        )
                    )
                    for i in range(batch_size)
                ]
            ).to(images_features.device)
        elif self.use_dummy_masks:
            resized_image_masks = torch.stack(
                [
                    torch.tensor(
                        create_deterministic_dummy_masks(
                            ve_dim,
                            device=images_features.device,
                            num_masks=self.number_of_masks,
                        )
                    )
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
        if self.global_view:
            ## include a mask which is f
            whole_image_mask = torch.ones(
                (batch_size, 1, ve_dim),
                dtype=resized_image_masks.dtype,
                device=resized_image_masks.device,
            )
            resized_image_masks = torch.cat(
                [whole_image_mask, resized_image_masks], dim=1
            )

        batch_size, M, ve_dim = tuple(resized_image_masks.shape)
        F = images_features.shape[-1]  # aka feature_dim

        # 1. Count the number of True entries per [B, M] and find the maximum count.
        counts = resized_image_masks.sum(dim=-1)  # shape: [B, M]
        self.max_count = int(
            counts.max().item()
        )  # scalar (largest number of True values)

        # 2. Expand images_features so that each mask sees [ve_dim, F]:
        #    From [B, ve_dim, F] --> [B, 1, ve_dim, F] and then expand to [B, M, ve_dim, F]
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
        if self.no_masktoken:
            return no_tokens_features.view(
                no_tokens_features.size(0), -1, no_tokens_features.size(3)
            )
        else:
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
        number_of_masks=10,
        image_filling=False,
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
        self.number_of_masks = number_of_masks

        # other
        self.mask_limit = mask_limit
        self.image_filling = image_filling

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
        zero_indices = []
        for i, (image_features, image_masks) in enumerate(
            zip(images_batch, masks_batch)
        ):

            # Get image features shape (batch_size, ve_dim, feature_dim)
            ve_dim, feature_dim = image_features.shape
            # torch.set_printoptions(threshold=image_masks[0].numel())

            if self.use_sliding_window:
                resized_image_masks = create_sliding_masks(
                    ve_dim, device=image_features.device, num_masks=self.number_of_masks
                )
            elif self.use_dummy_masks:
                resized_image_masks = create_deterministic_dummy_masks(
                    ve_dim, device=image_features.device, num_masks=self.number_of_masks
                )
            else:
                resized_image_masks = [
                    downsample_mask_to_1d_counts(mask, ve_dim).to(image_features.device)
                    for mask in image_masks
                ]

            # for j, t in enumerate(resized_image_masks):
            #     if t.count_nonzero().item() == 0:
            #         zero_indices.append(j)
            # zero_indices = []

            # print("resized_image_masks[0]", resized_image_masks[1])
            # print("resized_image_masks[0]", resized_image_masks[2])

            image_features = image_features.to(image_features.device)
            resized_image_masks = [
                m.to(image_features.device).contiguous() for m in resized_image_masks
            ]
            try:
                if self.mask_removing and len(resized_image_masks) > self.mask_limit:
                    masked_features = []
                elif self.mask_limiting and len(resized_image_masks) > self.mask_limit:
                    masked_features = [
                        image_features[mask]
                        for mask in resized_image_masks[: self.mask_limit]
                    ]
                elif self.image_filling:
                    for m in resized_image_masks:
                        assert m.dtype == torch.bool
                        assert (
                            m.shape[0] == image_features.shape[0]
                        ), f"mask shape {m.shape} doesn't match features {image_features.shape}"

                    covered = (
                        torch.stack(resized_image_masks, dim=0)
                        .any(dim=0)
                        .to(image_features.device)
                    )

                    # 2) invert to get the “uncovered” positions
                    uncovered = ~covered
                    uncovered = uncovered.to(image_features.device)

                    # 3) if there really is anything uncovered, append it
                    if torch.count_nonzero(uncovered).item() > 0:
                        resized_image_masks.append(uncovered)

                    # 4) now apply all masks (including the dummy one) to extract features
                    masked_features = [
                        image_features[mask] for mask in resized_image_masks
                    ]

                else:
                    masked_features = [
                        image_features[mask] for mask in resized_image_masks
                    ]
            except:
                print(
                    f"Raised Exception within masking application process, using only image_features for image number {i}"
                )
                masked_features = image_features

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
                self.global_view
                and self.averaging_global_view
                and not self.no_masktoken
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

        return final_output, zero_indices

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
        output_features, zero_indices = self.apply_masks_with_tokens(
            images_features, masks
        )

        return output_features, zero_indices
