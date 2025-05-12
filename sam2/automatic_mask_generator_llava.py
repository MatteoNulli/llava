# if using Apple MPS, fall back to CPU for unsupported ops
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# import torch
import os
import io
import argparse
import json
import shutil
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import base64
import ray
import time
import torch
import fire
import ast

from typing import Dict, Any, List, Union
from tqdm import tqdm
from typing import Dict, List, Any, Union, Tuple
from PIL import Image
from tqdm import tqdm
from itertools import cycle

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


np.random.seed(3)


def sam2_instance(args):

    sam2 = build_sam2(
        args.model_cfg,
        args.sam2_checkpoint,
        device=args.device,
        apply_postprocessing=False,
    )

    mask_generator = SAM2AutomaticMaskGenerator(
        model=sam2,
        points_per_batch=6,
        pred_iou_thresh=0.9,
        # stability_score_thresh=0.97,
    )

    return mask_generator


def data_instance(args):
    # '/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json'
    if not args.cambrian:
        data = json.load(open(args.data_path))
    else:
        data = []
        with open(args.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                data.append(record)

    return data


class PartitionedSegmentationManager:
    def __init__(
        self,
        args,
        arrays_directory: str,
        metadata_directory: str,
        partition_id: int,
        total_partitions: int,
        captioning: bool = False,
        cambrian: bool = False,
    ):
        """
        Initialize the manager with partition information.

        Args:
            arrays_directory (str): Directory for storing array data.
            metadata_directory (str): Directory for storing metadata.
            partition_id (int): ID of current partition (0 to total_partitions-1).
            total_partitions (int): Total number of partitions.
        """
        self.args = args
        self.partition_id = partition_id
        self.total_partitions = total_partitions

        # Create partition-specific directories
        self.arrays_dir = os.path.join(arrays_directory, f"partition_{partition_id}")
        self.metadata_file = os.path.join(
            metadata_directory, f"metadata_partition_{partition_id}.json"
        )

        # Create directories if they don't exist
        os.makedirs(self.arrays_dir, exist_ok=True)
        os.makedirs(metadata_directory, exist_ok=True)

        # Initialize metadata
        self._initialize_or_recover_metadata()

        # Preprocessing arguments
        self.captioning = captioning
        self.cambrian = cambrian

    def _safe_json_read(self, filepath: str) -> Dict:
        """
        Safely read JSON file with error handling and recovery.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Try to recover the JSON by reading until the last valid line
            with open(filepath, "r") as f:
                content = f.read()

            # Find the last complete object (ending with })
            last_brace_index = content.rfind("}")
            if last_brace_index != -1:
                valid_content = content[: last_brace_index + 1]
                try:
                    return json.loads(valid_content)
                except json.JSONDecodeError:
                    pass

            # If recovery failed, return empty dict
            return {}

    def _safe_json_write(self, data: Dict, filepath: str) -> None:
        """
        Safely write JSON file using a temporary file.
        """
        # Create a temporary file in the same directory
        temp_dir = os.path.dirname(filepath)
        with tempfile.NamedTemporaryFile(mode="w", dir=temp_dir, delete=False) as tf:
            json.dump(data, tf, indent=2)
            temp_filepath = tf.name

        # Rename temporary file to target file (atomic operation)
        shutil.move(temp_filepath, filepath)

    def _get_array_path(self, image_key: str, mask_index: int) -> str:
        """
        Generate the path for storing a segmentation array.
        """
        # Create a subdirectory for each image to organize the arrays
        image_array_dir = os.path.join(self.arrays_dir, image_key)
        os.makedirs(image_array_dir, exist_ok=True)
        return os.path.join(image_array_dir, f"mask_{mask_index}.npy")

    def _save_array(self, array: np.ndarray, filepath: str) -> None:
        """
        Safely save a numpy array to a file.
        """
        # Save to temporary file first
        temp_dir = os.path.dirname(filepath)
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=temp_dir, delete=False, suffix=".npy"
        ) as tf:
            np.save(tf, array)
            temp_filepath = tf.name

        # Move temporary file to final location
        shutil.move(temp_filepath, filepath)

    def _initialize_or_recover_metadata(self) -> None:
        """
        Initialize metadata file or recover from backup if main file is corrupted.
        """
        metadata = {}

        # Try to read main metadata file
        if os.path.exists(self.metadata_file):
            metadata = self._safe_json_read(self.metadata_file)
            print(f"Recovered Metadata from checkpoint {self.metadata_file}")
        else:
            print("No existing metadata found. Initializing new metadata.")

        # Save recovered or empty metadata
        self._safe_json_write(metadata, self.metadata_file)

    def metadata_and_list_image_keys(self):
        """
        Check if an image has already been processed and saved.
        """
        metadata = self._safe_json_read(self.metadata_file)
        return metadata, list(metadata.keys())

    def save_data(self, image_key: str, segmentations: List[Dict[str, Any]]) -> None:
        """
        Save multiple segmentation data for a single image, storing arrays as .npy files.
        """
        # Load existing metadata
        metadata = self._safe_json_read(self.metadata_file)

        # Process and save each segmentation
        processed_segmentations = []
        for idx, seg in enumerate(segmentations):
            # Create a copy of the segmentation data without the array
            seg_metadata = seg.copy()

            # Save the segmentation array as a .npy file
            array_path = self._get_array_path(image_key, idx)
            self._save_array(seg["segmentation"], array_path)

            # Replace the array in metadata with the file path
            seg_metadata["segmentation"] = os.path.basename(array_path)
            processed_segmentations.append(seg_metadata)

        # Save metadata
        metadata[image_key] = processed_segmentations

        # Save updated metadata and backup
        self._safe_json_write(metadata, self.metadata_file)

    def get_partition_indices(self, data_size: int) -> Tuple[int, int]:
        """
        Calculate start and end indices for this partition.

        Args:
            data_size (int): Total size of the dataset

        Returns:
            Tuple[int, int]: Start and end indices for this partition
        """
        items_per_partition = data_size // self.total_partitions
        remainder = data_size % self.total_partitions

        start_idx = self.partition_id * items_per_partition + min(
            self.partition_id, remainder
        )
        end_idx = (
            start_idx
            + items_per_partition
            + (1 if self.partition_id < remainder else 0)
        )

        return start_idx, end_idx

    def _compute_image_key(self, item):
        # Get image key
        ## The image key is going to be used as name to store the image masks inside the partition directory.

        ## TIP: change "images/","nyu-visionx--Cambrian-10M--extracted/" or "/mnt/nushare2/data/baliao/multimodal/data/"
        ##      with the last element of the pth before the dataset informations (eg. 'coco/train2017/' or 'llava/llava_pretrain/images')

        if self.captioning:
            ## Your file system will look like /disklocation/segmentation_data/arrays/partition_i/image_key/mask_j.npy
            image_key = item["image"].split("images/", 1)[-1]
        elif self.cambrian:
            ## post-process for cambrian
            ## Your file system will look like /disklocation/segmentation_data/arrays/partition_i/image_key/000000012805_masks.npy containing all boolean segmentations
            image_key = item["image"].split("nyu-visionx--Cambrian-10M--extracted/", 1)[
                -1
            ]
        else:
            image_key = item["image"].strip(
                "/mnt/nushare2/data/baliao/multimodal/data/"
            )

        return image_key

    def process_partition(
        self, data: List[Dict[str, str]], mask_generator: Any
    ) -> None:
        """
        Process only this partition's portion of the dataset.
        """
        start_idx, end_idx = self.get_partition_indices(len(data))
        partition_data = data[start_idx:end_idx]

        metadata, image_keys = self.metadata_and_list_image_keys()

        for item in tqdm(
            partition_data, desc=f"Processing partition {self.partition_id}"
        ):

            image_key = self._compute_image_key(item)  # your existing logic

            # Skip if already processed
            if image_key in image_keys:
                continue

            try:
                # Process image and generate masks
                image_path = item["image"]
                image = Image.open(image_path)
                image = np.array(image.convert("RGB"))
                masks = mask_generator.generate(image)

                # Save masks and metadata
                # Process and save each segmentation
                processed_segmentations = []
                for idx, seg in enumerate(masks):
                    # Create a copy of the segmentation data without the array
                    seg_metadata = seg.copy()

                    # Save the segmentation array as a .npy file
                    array_path = self._get_array_path(image_key, idx)
                    self._save_array(seg["segmentation"], array_path)

                    # Replace the array in metadata with the file path
                    seg_metadata["segmentation"] = os.path.basename(array_path)
                    processed_segmentations.append(seg_metadata)

                # Save metadata
                metadata[image_key] = processed_segmentations

                # Save updated metadata and backup
                self._safe_json_write(metadata, self.metadata_file)

            except Exception as e:
                print(
                    f"Error processing image {image_key} in partition {self.partition_id}: {str(e)}"
                )
                continue

    def load_data(
        self, image_key: str, indices: Union[int, List[int]] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Load segmentation data for a specific image.
        """
        metadata = self._safe_json_read(self.metadata_file)

        if image_key not in metadata:
            raise KeyError(f"No data found for image: {image_key}")

        image_metadata = metadata[image_key]

        # Handle single index
        if isinstance(indices, int):
            result = image_metadata[indices].copy()
            array_path = os.path.join(
                self.arrays_dir, image_key, result["segmentation"]
            )
            result["segmentation"] = np.load(array_path)
            return result

        # Handle multiple indices or None (all segmentations)
        load_indices = indices if indices is not None else range(len(image_metadata))
        result = []

        for idx in load_indices:
            entry = image_metadata[idx].copy()
            array_path = os.path.join(self.arrays_dir, image_key, entry["segmentation"])
            entry["segmentation"] = np.load(array_path)
            result.append(entry)

        return result

    def save_data_benchmark(
        self, image_key: str, segmentations: List[Dict[str, Any]]
    ) -> None:
        """
        Save segmentation arrays for a single image without updating metadata.

        Args:
            image_key (str): Key identifying the image.
            segmentations (List[Dict[str, Any]]): List of segmentation dictionaries. Each dictionary should contain
                                                  a 'segmentation' key with a numpy array as value.
        """
        for idx, seg in enumerate(segmentations):
            array_path = self._get_array_path(image_key, idx)

            self._save_array(seg["segmentation"], array_path)

    def process_benchmark(
        self,
        benchmark_images_dir: str,
        mask_generator: Any,
        _bytes=False,
    ) -> None:
        """
        Process benchmark data. This method iterates over all image files in the given benchmark
        directory (which is expected to be a small dataset) and creates masks for each image.
        Instead of updating metadata, it uses save_data_benchmark to only store the segmentation arrays.

        Args:
            benchmark_images_dir (str): Directory containing benchmark images.
            mask_generator (Any): An instance with a generate(image) method that returns segmentation masks.
        """

        if _bytes:
            if "mme" in benchmark_images_dir:
                # Path to your combined Arrow file
                file_path = "/data/chatgpt/notebooks/mnulli/combined.arrow"

                # Open the Arrow file and read the table
                with open(file_path, "rb") as f:
                    try:
                        # Try reading using the file reader (Arrow file format)
                        reader = ipc.RecordBatchFileReader(f)
                    except pa.ArrowInvalid:
                        # If that fails, try the stream reader (Arrow stream format)
                        f.seek(0)
                        reader = ipc.RecordBatchStreamReader(f)
                    table = reader.read_all()

                # Convert to a pandas DataFrame for easy viewing
                benchmark_df = table.to_pandas()
            elif "aro" in benchmark_images_dir:
                benchmark_df = pd.read_csv(benchmark_images_dir + "/combined.csv")

            elif "mmbench" in benchmark_images_dir:
                benchmark_df = pd.read_csv(benchmark_images_dir + "/test.csv")

            elif "mmstar" in benchmark_images_dir:
                benchmark_df = pd.read_parquet(benchmark_images_dir + "/test.parquet")
            else:
                # Load benchmark from test.parquet
                benchmark_df = pd.read_parquet(benchmark_images_dir + "/test.parquet")

            for i, row in tqdm(
                benchmark_df.iterrows(),
                desc="Processing Benchmark Images",
                total=len(benchmark_df),
            ):

                # if "filename" in row.keys():
                #     image_key = row["filename"]
                #     image = Image.open(io.BytesIO(row["image"]["bytes"]))
                if "mme" in benchmark_images_dir:
                    image_key = row["question_id"]

                    image = Image.open(io.BytesIO(row["image"]["bytes"]))

                elif "mmbench" in benchmark_images_dir:
                    image_key = row["category"]
                    image = base64.b64decode(row["image"])
                    image = Image.open(io.BytesIO(image)).convert("RGB")

                elif "mmstar" in benchmark_images_dir:
                    image_key = row["category"]
                    image = Image.open(io.BytesIO(row["image"])).convert("RGB")

                elif "gowitheflow___aro-coco-order" in row["benchmark"]:
                    image_key = f"gowitheflow___aro-coco-order_picture_{i}"
                    image = Image.open(
                        io.BytesIO(ast.literal_eval(row["images"])["bytes"])
                    ).convert("RGB")
                elif "gowitheflow___aro-flickr-order" in row["benchmark"]:
                    image_key = f"gowitheflow___aro-flickr-order_picture_{i}"
                    image = Image.open(
                        io.BytesIO(ast.literal_eval(row["images"])["bytes"])
                    ).convert("RGB")

                elif "gowitheflow___aro-visual-relation" in row["benchmark"]:
                    image_key = f"gowitheflow___aro-visual-relation_picture_{i}"
                    image = Image.open(
                        io.BytesIO(ast.literal_eval(row["images"])["bytes"])
                    ).convert("RGB")

                elif "gowitheflow___aro-visual-attribution" in row["benchmark"]:
                    image_key = f"gowitheflow___aro-visual-attribution_picture__{i}"
                    image = Image.open(
                        io.BytesIO(ast.literal_eval(row["images"])["bytes"])
                    ).convert("RGB")

                image = np.array(image.convert("RGB"))
                masks = mask_generator.generate(image)

                try:
                    # Save only the arrays, without updating metadata
                    self.save_data_benchmark(image_key, masks)
                except Exception as e:
                    print(f"Error processing benchmark image {image_key}: {str(e)}")
                    continue

        else:
            # For benchmarks we are not tracking metadata.
            for image_file in tqdm(
                os.listdir(benchmark_images_dir), desc="Processing Benchmark Images"
            ):
                image_path = os.path.join(benchmark_images_dir, image_file)
                if not os.path.isfile(image_path):
                    continue

                # Use the image file name as the key
                image_key = image_file

                image = Image.open(image_path)
                image = np.array(image.convert("RGB"))
                masks = mask_generator.generate(image)
                try:
                    # print("image_key", image_key)
                    # Save only the arrays, without updating metadata
                    self.save_data_benchmark(image_key, masks)
                except Exception as e:

                    print(f"Error processing benchmark image {image_key}: {str(e)}")
                    continue


if __name__ == "__main__":

    def str2bool(v):
        return v.lower() in ("yes", "true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sam2_checkpoint",
        type=str,
        default="/data/chatgpt/notebooks/mnulli/sam2/checkpoints/sam2.1_hiera_large.pt",
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
        nargs="?",  # 0 or 1 args
        const="/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json",  # if flag is given but no value
        default="/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json",  # if flag is omitted entirely
        help=(
            "Path to data file. "
            "If you omit the flag or give it no value, "
            "it will use the built-in default."
        ),
    )
    parser.add_argument(
        "--benchmark_images_dir",
        type=str,
        nargs="?",  # 0 or 1 args
        const="/mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/images",  # if flag is given but no value
        default="/mnt/nushare2/data/mnulli/thesis/data/benchmarks/conme/images",  # if flag is omitted entirely
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

    data = data_instance(args)

    manager = PartitionedSegmentationManager(
        args=args,
        arrays_directory=args.arrays_directory,
        metadata_directory=args.metadata_directory,
        partition_id=args.partition_id,
        total_partitions=args.total_partitions,
        captioning=args.captioning,
        cambrian=args.cambrian,
    )

    mask_generator = sam2_instance(args)
    # manager.process_partition(data, mask_generator)
    manager.process_benchmark(args.benchmark_images_dir, mask_generator, _bytes=True)
