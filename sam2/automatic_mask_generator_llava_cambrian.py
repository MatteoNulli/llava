import ray
import torch
import numpy as np
import os
import fire
import time
import argparse
import errno

from PIL import Image
from itertools import cycle
from tqdm import tqdm


from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from automatic_mask_generator_llava import PartitionedSegmentationManager, data_instance


@ray.remote(num_gpus=1)
class SAM2MaskActor:
    def __init__(self, model_cfg, arrays_dir, checkpoint, partition_id, data_manager):
        # Load model on the assigned GPU
        self.device = torch.device("cuda")
        self.model = build_sam2(
            model_cfg, checkpoint, device=self.device, apply_postprocessing=False
        )
        self.mask_generator = SAM2AutomaticMaskGenerator(
            model=self.model,
            points_per_batch=6,
            pred_iou_thresh=0.9,
        )

        self.arrays_dir = arrays_dir
        self.partiton_id = partition_id
        self.data_manager = data_manager
        print(f"Actor initialized on GPU {ray.get_gpu_ids()[0]}")

    def generate_mask(self, image_path, image_key, metadata):
        image = Image.open(image_path)
        image = np.array(image.convert("RGB"))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            masks = self.mask_generator.generate(image)

        # pull out only the segmentation arrays
        segmentations = [m["segmentation"] for m in masks]

        # prepare new save directory for segmentations
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        # image_key == "coco/train2017/000000033471.jpg"
        # # join to new root → ".../arrays/coco/train2017/000000033471.jpg"
        save_dir_without_key = os.path.join(
            self.arrays_dir, f"partition_{self.partiton_id}"
        )
        save_dir = os.path.join(save_dir_without_key, image_key)
        # try primary directory, fallback to _v2 on ENOSPC
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            if e.errno == errno.ENOSPC:
                save_dir_base_v2 = save_dir_base + "_v2"
                save_dir_v2 = os.path.join(save_dir_base_v2, image_key)
                os.makedirs(save_dir_v2, exist_ok=True)
                save_dir = save_dir_v2
                print(
                    f"[WARN] No space on device for {save_dir_base}, using {save_dir_base_v2} instead."
                )
            else:
                raise

        # final .npy path
        save_path = os.path.join(save_dir, f"{base_name}_masks.npy")
        try:
            np.save(save_path, segmentations)

            # Save metadata
            # build metadata: keep all fields but point 'segmentation' to the .npy path
            masks_meta = []
            for mask in masks:
                m = mask.copy()
                m["segmentation"] = save_path
                masks_meta.append(m)

            # update and write metadata
            metadata[image_key] = masks_meta
            self.data_manager._safe_json_write(
                metadata, self.data_manager.metadata_file
            )

        except:
            print(f"mask {image_key} not saved at {save_path}. Skipping...")

        return save_path


def distributed_sam2_inference(args):
    # Initialize data to retrieve image_paths
    data = data_instance(args)

    # Initiate Data Manager to handle saving
    data_manager = PartitionedSegmentationManager(
        args=args,
        arrays_directory=args.arrays_directory,
        metadata_directory=args.metadata_directory,
        partition_id=args.partition_id,
        total_partitions=args.total_partitions,
        captioning=args.captioning,
        cambrian=args.cambrian,
    )

    start_idx, end_idx = data_manager.get_partition_indices(len(data))
    partition_data = data[start_idx:end_idx]

    metadata, image_keys = data_manager.metadata_and_list_image_keys()

    num_gpus = torch.cuda.device_count()
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(num_gpus=num_gpus)
    print("Ray Available Resources:", ray.available_resources())

    # Create actors
    actors = [
        SAM2MaskActor.remote(
            args.model_cfg,
            args.arrays_directory,
            args.sam2_checkpoint,
            args.partition_id,
            data_manager,
        )
        for _ in range(num_gpus)
    ]
    actor_pool = cycle(actors)

    # Distribute tasks
    start = time.time()

    input_root = "/mnt/nushare2/data/mnulli/thesis/data/training_data/nyu-visionx--Cambrian-10M--extracted"

    futures = []
    for item in tqdm(
        partition_data,
        desc=f"Processing partition {args.partition_id} of length {len(partition_data)}",
    ):

        image_key = data_manager._compute_image_key(item)  # your existing logic

        # Skip if already processed
        if image_key in image_keys:
            continue

        image_path = item["image"]
        image_path_check = image_path.split("extracted")[-1]
        if "nan" in image_path_check:
            continue

        actor = next(actor_pool)
        futures.append(actor.generate_mask.remote(image_path, image_key, metadata))

    elapsed = time.time() - start
    print("elapsed for actor", elapsed)
    # Gather results
    print("calling ray.get")
    start = time.time()
    results = ray.get(futures)
    elapsed = time.time() - start
    print("elapsed for ray.get", elapsed)
    print(f"Finished processing {len(results)} images.")


if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data/checkpoints/sam2.1_hiera_large.pt",
        help="checkpoint of sam2 model",
    )
    parser.add_argument(
        "--model_cfg",
        type=str,
        default="/data/chatgpt/notebooks/mnulli/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        help="checkpoint of model configuration",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--metadata_directory",
        type=str,
        default="/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data",
        help="Where to store the metadata file with pointers to .npy masks files",
    )
    parser.add_argument(
        "--arrays_directory",
        type=str,
        default="/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data",
        help="Where to store the actual .npy arrays of segmentation data",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json",
        help="Path to data file",
    )
    parser.add_argument(
        "--benchmark_images_dir",
        type=str,
        default="/mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/images",
        help="Directory containing benchmark images",
    )
    parser.add_argument(
        "--partition-id",
        type=int,
        required=True,
        help="ID of the partition to process (0-9)",
    )
    parser.add_argument(
        "--total-partitions", type=int, default=10, help="Total number of partitions"
    )

    parser.add_argument(
        "--captioning",
        type=str2bool,
        nargs="?",
        const=False,  # if you write just “--captioning” → True
        default=False,  # if you omit it altogether → False
        help="Enable or disable captioning. If True we are performing masking for captioning data.",
    )
    parser.add_argument(
        "--cambrian",
        type=str2bool,
        nargs="?",
        const=False,  # if you write just “--captioning” → True
        default=False,  # if you omit it altogether → False
        help="If True we are performing masking for cambrian data.",
    )

    args = parser.parse_args()

    fire.Fire(distributed_sam2_inference(args))
