# import argparse
# import os
# # if using Apple MPS, fall back to CPU for unsupported ops
# # os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# import numpy as np
# # import torch
# import matplotlib.pyplot as plt
# from PIL import Image
# import json
# import os
# from typing import Dict, Any, List, Union
# from tqdm import tqdm
# import shutil
# import tempfile

# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


# def _set_device(args):
#     if args.device.type == "cuda":
#         # use bfloat16 for the entire notebook
#         torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
#         # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#         if torch.cuda.get_device_properties(0).major >= 8:
#             torch.backends.cuda.matmul.allow_tf32 = True
#             torch.backends.cudnn.allow_tf32 = True
#     elif args.device.type == "mps":
#         print(
#             "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
#             "give numerically different outputs and sometimes degraded performance on MPS. "
#             "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
#         )


np.random.seed(3)

def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)


def sam2_instance(args):

    sam2 = build_sam2(args.model_cfg, args.sam2_checkpoint, device=args.device, apply_postprocessing=False)

    mask_generator = SAM2AutomaticMaskGenerator(
        model = sam2,
        points_per_batch=6,
        pred_iou_thresh=0.9,
        # stability_score_thresh=0.97,
    )

    return mask_generator

def data_instance(args):
    # '/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json'
    data = json.load(open(args.data_path))

    return data

# class SegmentationDataManager:
#     def __init__(self, base_directory: str):
#         """
#         Initialize the manager with a base directory for storing all data.
        
#         Args:
#             base_directory (str): Directory where all data will be stored
#         """
#         self.base_directory = base_directory
#         self.arrays_dir = os.path.join(base_directory, "arrays")
#         self.metadata_file = os.path.join(base_directory, "metadata.json")
#         self.backup_file = os.path.join(base_directory, "metadata_backup.json")
        
#         # Create directories if they don't exist
#         os.makedirs(self.arrays_dir, exist_ok=True)
        
#         # Initialize or recover metadata file
#         self._initialize_or_recover_metadata()

#     def _safe_json_read(self, filepath: str) -> Dict:
#         """
#         Safely read JSON file with error handling and recovery.
#         """
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             # Try to recover the JSON by reading until the last valid line
#             with open(filepath, 'r') as f:
#                 content = f.read()
            
#             # Find the last complete object (ending with })
#             last_brace_index = content.rfind('}')
#             if last_brace_index != -1:
#                 valid_content = content[:last_brace_index + 1]
#                 try:
#                     return json.loads(valid_content)
#                 except json.JSONDecodeError:
#                     pass
            
#             # If recovery failed, return empty dict
#             return {}

#     def _safe_json_write(self, data: Dict, filepath: str) -> None:
#         """
#         Safely write JSON file using a temporary file.
#         """
#         # Create a temporary file in the same directory
#         temp_dir = os.path.dirname(filepath)
#         with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as tf:
#             # Write to temporary file
#             json.dump(data, tf, indent=2)
#             temp_filepath = tf.name
        
#         # Rename temporary file to target file (atomic operation)
#         shutil.move(temp_filepath, filepath)

#     def _initialize_or_recover_metadata(self) -> None:
#         """
#         Initialize metadata file or recover from backup if main file is corrupted.
#         """
#         metadata = {}
        
#         # Try to read main metadata file
#         if os.path.exists(self.metadata_file):
#             metadata = self._safe_json_read(self.metadata_file)
        
#         # If main file is empty or corrupted, try backup
#         if not metadata and os.path.exists(self.backup_file):
#             metadata = self._safe_json_read(self.backup_file)
        
#         # Save recovered or empty metadata
#         self._safe_json_write(metadata, self.metadata_file)
#         # Create backup
#         self._safe_json_write(metadata, self.backup_file)

#     def list_image_keys(self):
#         """
#         Check if an image has already been processed and saved.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
#         image_keys_list = []  
#         for image_key, val in metadata.items():
#             image_keys_list.append(image_key)

#         return image_keys_list
    
    
#     def save_data(self, image_key: str, segmentations: List[Dict[str, Any]]) -> None:
#         """
#         Save multiple segmentation data for a single image.
#         """
#         # Create image directory
#         image_arrays_dir = os.path.join(self.arrays_dir, image_key)
#         os.makedirs(image_arrays_dir, exist_ok=True)
        
#         # Load existing metadata
#         metadata = self._safe_json_read(self.metadata_file)
        
#         # Initialize or get existing image metadata
#         metadata[image_key] = []
        
#         # Save each segmentation
#         for idx, seg_dict in enumerate(segmentations):
#             # Save the numpy array
#             array_filename = f"segmentation_{idx}.npy"
#             array_path = os.path.join(image_arrays_dir, array_filename)
#             np.save(array_path, seg_dict['segmentation'])
            
#             # Prepare metadata
#             meta_entry = seg_dict.copy()
#             meta_entry['segmentation'] = array_filename
            
#             # Add metadata for this segmentation
#             metadata[image_key].append(meta_entry)
        
#         # Save updated metadata and backup
#         self._safe_json_write(metadata, self.metadata_file)
#         self._safe_json_write(metadata, self.backup_file)
        
#     def process_dataset(self, data: List[Dict[str, str]], mask_generator: Any) -> None:
#         """
#         Process a dataset of images, generating and saving masks only for unprocessed images.
#         """
#         image_keys = self.list_image_keys()

#         for i in tqdm(range(len(data))):
#             # Get image key
#             image_key = data[i]['image'].strip('/mnt/nushare2/data/baliao/multimodal/data/')
            
#             # Check if image has already been processed
#             if image_key in image_keys:
#                 continue
            
#             try:
#                 # Get image path
#                 image_path = data[i]['image']
                    
#                 # Process image and generate masks
#                 image = Image.open(image_path)
#                 image = np.array(image.convert("RGB"))
#                 masks = mask_generator.generate(image)
                
#                 # Save masks
#                 self.save_data(image_key, masks)
                
#             except Exception as e:
#                 print(f"Error processing image {image_key}: {str(e)}")
#                 continue

    
#     def load_data(self, image_key: str, indices: Union[int, List[int]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
#         """
#         Load segmentation data for a specific image.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
            
#         if image_key not in metadata:
#             raise KeyError(f"No data found for image: {image_key}")
        
#         image_metadata = metadata[image_key]
#         image_arrays_dir = os.path.join(self.arrays_dir, image_key)
        
#         # Handle single index
#         if isinstance(indices, int):
#             metadata = image_metadata[indices].copy()
#             array_path = os.path.join(image_arrays_dir, metadata['segmentation'])
#             metadata['segmentation'] = np.load(array_path)
#             return metadata
        
#         # Handle multiple indices or None (all segmentations)
#         load_indices = indices if indices is not None else range(len(image_metadata))
#         result = []
        
#         for idx in load_indices:
#             metadata = image_metadata[idx].copy()
#             array_path = os.path.join(image_arrays_dir, metadata['segmentation'])
#             metadata['segmentation'] = np.load(array_path)
#             result.append(metadata)
        
#         return result

# import numpy as np
# import json
# import os
# from typing import Dict, Any, List, Union
# from tqdm import tqdm
# from PIL import Image
# import shutil
# import tempfile


# class SegmentationDataManager_metadataonly:
#     def __init__(self, base_directory: str):
#         """
#         Initialize the manager with a base directory for storing metadata.
        
#         Args:
#             base_directory (str): Directory where metadata will be stored
#         """
#         self.base_directory = base_directory
#         self.arrays_dir = os.path.join(base_directory, "arrays")  # Keep arrays_dir for backward compatibility
#         self.metadata_file = os.path.join(base_directory, "metadata.json")
#         self.backup_file = os.path.join(base_directory, "metadata_backup.json")
        
#         # Create directories if they don't exist
#         os.makedirs(base_directory, exist_ok=True)
#         os.makedirs(self.arrays_dir, exist_ok=True)
        
#         # Initialize or recover metadata file
#         # self._initialize_or_recover_metadata()

#     def _safe_json_read(self, filepath: str) -> Dict:
#         """
#         Safely read JSON file with error handling and recovery.
#         """
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             # Try to recover the JSON by reading until the last valid line
#             with open(filepath, 'r') as f:
#                 content = f.read()
            
#             # Find the last complete object (ending with })
#             last_brace_index = content.rfind('}')
#             if last_brace_index != -1:
#                 valid_content = content[:last_brace_index + 1]
#                 try:
#                     return json.loads(valid_content)
#                 except json.JSONDecodeError:
#                     pass
            
#             # If recovery failed, return empty dict
#             return {}

#     def _safe_json_write(self, data: Dict, filepath: str) -> None:
#         """
#         Safely write JSON file using a temporary file.
#         """
#         # Create a temporary file in the same directory
#         temp_dir = os.path.dirname(filepath)
#         with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as tf:
#             # Convert numpy arrays to lists for JSON serialization
#             serializable_data = self._convert_arrays_to_lists(data)
#             # Write to temporary file
#             json.dump(serializable_data, tf, indent=2)
#             temp_filepath = tf.name
        
#         # Rename temporary file to target file (atomic operation)
#         shutil.move(temp_filepath, filepath)

#     def _convert_arrays_to_lists(self, data: Any) -> Any:
#         """
#         Convert numpy arrays to lists for JSON serialization.
#         """
#         if isinstance(data, dict):
#             return {k: self._convert_arrays_to_lists(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             return [self._convert_arrays_to_lists(item) for item in data]
#         elif isinstance(data, np.ndarray):
#             return data.tolist()
#         else:
#             return data

#     def _convert_lists_to_arrays(self, data: Any) -> Any:
#         """
#         Convert lists back to numpy arrays when loading data.
#         """
#         if isinstance(data, dict):
#             return {k: self._convert_lists_to_arrays(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             # Check if this list represents a segmentation array
#             if isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
#                 return np.array(data)
#             return [self._convert_lists_to_arrays(item) for item in data]
#         else:
#             return data

#     def _initialize_or_recover_metadata(self) -> None:
#         """
#         Initialize metadata file or recover from backup if main file is corrupted.
#         """
#         metadata = {}
        
#         # Try to read main metadata file
#         if os.path.exists(self.metadata_file):
#             metadata = self._safe_json_read(self.metadata_file)
        
#         # If main file is empty or corrupted, try backup
#         if not metadata and os.path.exists(self.backup_file):
#             metadata = self._safe_json_read(self.backup_file)
        
#         # Save recovered or empty metadata
#         self._safe_json_write(metadata, self.metadata_file)
#         # Create backup
#         self._safe_json_write(metadata, self.backup_file)

#     def metadata_and_list_image_keys(self):
#         """
#         Check if an image has already been processed and saved.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
#         return metadata, list(metadata.keys())
    
#     def save_data(self, image_key: str, segmentations: List[Dict[str, Any]]) -> None:
#         """
#         Save multiple segmentation data for a single image directly in JSON.
#         Only to use for single image.
#         """
#         # Load existing metadata
#         metadata = self._safe_json_read(self.metadata_file)
        
#         # Save segmentations directly in metadata
#         metadata[image_key] = segmentations
        
#         # Save updated metadata and backup
#         self._safe_json_write(metadata, self.metadata_file)
#         self._safe_json_write(metadata, self.backup_file)
        
#     def process_dataset(self, data: List[Dict[str, str]], mask_generator: Any) -> None:
#         """
#         Process a dataset of images, generating and saving masks only for unprocessed images.
#         """
#         metadata, image_keys = self.metadata_and_list_image_keys()

#         for i in tqdm(range(len(data))):
#             # Get image key
#             image_key = data[i]['image'].strip('/mnt/nushare2/data/baliao/multimodal/data/')
            
#             # Check if image has already been processed
#             if image_key in image_keys:
#                 continue
            
#             try:
#                 # Get image path
#                 image_path = '/mnt/nushare2/data/baliao/multimodal/data/' + data[i]['image']
                    
#                 # Process image and generate masks
#                 image = Image.open(image_path)
#                 image = np.array(image.convert("RGB"))
#                 masks = mask_generator.generate(image)
                
#                 # Save masks
#                 # Save segmentations directly in metadata
#                 metadata[image_key] = masks
                
#                 # self.save_data(image_key, masks)
                
#             except Exception as e:
#                 print(f"Error processing image {image_key}: {str(e)}")
#                 continue
            
#             if i % 5 == 0:
#                 # Save updated metadata every five steps and backup
#                 self._safe_json_write(metadata, self.metadata_file)
#                 self._safe_json_write(metadata, self.backup_file)
                

#     def _load_segmentation(self, image_key: str, metadata: Dict[str, Any]) -> np.ndarray:
#         """
#         Load segmentation data, handling both .npy files and direct JSON storage.
        
#         Args:
#             image_key: Key of the image
#             metadata: Metadata dictionary containing segmentation data
            
#         Returns:
#             np.ndarray: The loaded segmentation data
#         """
#         # If segmentation is a string, assume it's a .npy file path
#         if isinstance(metadata['segmentation'], str) and metadata['segmentation'].endswith('.npy'):
#             array_path = os.path.join(self.arrays_dir, image_key, metadata['segmentation'])
#             if os.path.exists(array_path):
#                 return np.load(array_path)
#             else:
#                 raise FileNotFoundError(f"Segmentation file not found: {array_path}")
#         # Otherwise, assume it's direct JSON data
#         else:
#             return np.array(metadata['segmentation'])
    
#     def load_data(self, image_key: str, indices: Union[int, List[int]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
#         """
#         Load segmentation data for a specific image, supporting both .npy files and direct JSON storage.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
            
#         if image_key not in metadata:
#             raise KeyError(f"No data found for image: {image_key}")
        
#         image_metadata = metadata[image_key]
        
#         # Handle single index
#         if isinstance(indices, int):
#             result = image_metadata[indices].copy()
#             result['segmentation'] = self._load_segmentation(image_key, result)
#             return result
        
#         # Handle multiple indices or None (all segmentations)
#         load_indices = indices if indices is not None else range(len(image_metadata))
#         result = []
        
#         for idx in load_indices:
#             entry = image_metadata[idx].copy()
#             entry['segmentation'] = self._load_segmentation(image_key, entry)
#             result.append(entry)
        
#         return result
    

# import os
# import json
# import numpy as np
# import tempfile
# import shutil
# from typing import Dict, List, Any, Union
# from PIL import Image
# from tqdm import tqdm

# class SegmentationDataManager:
#     def __init__(self, base_directory: str):
#         """
#         Initialize the manager with a base directory for storing metadata and arrays.
        
#         Args:
#             base_directory (str): Directory where metadata and arrays will be stored
#         """
#         self.base_directory = base_directory
#         self.arrays_dir = os.path.join(base_directory, "arrays")
#         self.metadata_file = os.path.join(base_directory, "metadata.json")
#         self.backup_file = os.path.join(base_directory, "metadata_backup.json")
#         # self.partition = args.partition
        
#         # Create directories if they don't exist
#         os.makedirs(base_directory, exist_ok=True)
#         os.makedirs(self.arrays_dir, exist_ok=True)
        
#         # Initialize or recover metadata file
#         self._initialize_or_recover_metadata()

#     def _safe_json_read(self, filepath: str) -> Dict:
#         """
#         Safely read JSON file with error handling and recovery.
#         """
#         try:
#             with open(filepath, 'r') as f:
#                 return json.load(f)
#         except json.JSONDecodeError:
#             # Try to recover the JSON by reading until the last valid line
#             with open(filepath, 'r') as f:
#                 content = f.read()
            
#             # Find the last complete object (ending with })
#             last_brace_index = content.rfind('}')
#             if last_brace_index != -1:
#                 valid_content = content[:last_brace_index + 1]
#                 try:
#                     return json.loads(valid_content)
#                 except json.JSONDecodeError:
#                     pass
            
#             # If recovery failed, return empty dict
#             return {}

#     def _safe_json_write(self, data: Dict, filepath: str) -> None:
#         """
#         Safely write JSON file using a temporary file.
#         """
#         # Create a temporary file in the same directory
#         temp_dir = os.path.dirname(filepath)
#         with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as tf:
#             json.dump(data, tf, indent=2)
#             temp_filepath = tf.name
        
#         # Rename temporary file to target file (atomic operation)
#         shutil.move(temp_filepath, filepath)

#     def _get_array_path(self, image_key: str, mask_index: int) -> str:
#         """
#         Generate the path for storing a segmentation array.
#         """
#         # Create a subdirectory for each image to organize the arrays
#         image_array_dir = os.path.join(self.arrays_dir, image_key)
#         os.makedirs(image_array_dir, exist_ok=True)
#         return os.path.join(image_array_dir, f"mask_{mask_index}.npy")

#     def _save_array(self, array: np.ndarray, filepath: str) -> None:
#         """
#         Safely save a numpy array to a file.
#         """
#         # Save to temporary file first
#         temp_dir = os.path.dirname(filepath)
#         os.makedirs(temp_dir, exist_ok=True)
#         with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.npy') as tf:
#             np.save(tf, array)
#             temp_filepath = tf.name
        
#         # Move temporary file to final location
#         shutil.move(temp_filepath, filepath)

#     def _initialize_or_recover_metadata(self) -> None:
#         """
#         Initialize metadata file or recover from backup if main file is corrupted.
#         """
#         metadata = {}
        
#         # Try to read main metadata file
#         if os.path.exists(self.metadata_file):
#             metadata = self._safe_json_read(self.metadata_file)
        
#         # If main file is empty or corrupted, try backup
#         if not metadata and os.path.exists(self.backup_file):
#             metadata = self._safe_json_read(self.backup_file)
        
#         # Save recovered or empty metadata
#         self._safe_json_write(metadata, self.metadata_file)
#         # Create backup
#         self._safe_json_write(metadata, self.backup_file)

#     def metadata_and_list_image_keys(self):
#         """
#         Check if an image has already been processed and saved.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
#         return metadata, list(metadata.keys())
    
#     def save_data(self, image_key: str, segmentations: List[Dict[str, Any]]) -> None:
#         """
#         Save multiple segmentation data for a single image, storing arrays as .npy files.
#         """
#         # Load existing metadata
#         metadata = self._safe_json_read(self.metadata_file)
        
#         # Process and save each segmentation
#         processed_segmentations = []
#         for idx, seg in enumerate(segmentations):
#             # Create a copy of the segmentation data without the array
#             seg_metadata = seg.copy()
            
#             # Save the segmentation array as a .npy file
#             array_path = self._get_array_path(image_key, idx)
#             self._save_array(seg['segmentation'], array_path)
            
#             # Replace the array in metadata with the file path
#             seg_metadata['segmentation'] = os.path.basename(array_path)
#             processed_segmentations.append(seg_metadata)
        
#         # Save metadata
#         metadata[image_key] = processed_segmentations
        
#         # Save updated metadata and backup
#         self._safe_json_write(metadata, self.metadata_file)
#         self._safe_json_write(metadata, self.backup_file)
        
#     def process_dataset(self, data: List[Dict[str, str]], mask_generator: Any) -> None:
#         """
#         Process a dataset of images, generating and saving masks only for unprocessed images.
#         """
#         metadata, image_keys = self.metadata_and_list_image_keys()

#         for i in tqdm(range(279065, 418597)):
#             # Get image key
#             image_key = data[i]['image'].strip('/mnt/nushare2/data/baliao/multimodal/data/')
            
#             # Check if image has already been processed
#             if image_key in image_keys:
#                 continue
            
#             try:
#                 # Get image path
#                 image_path = '/home/mnulli1/data/LLaVA-Pretrain/' + data[i]['image']
                    
#                 # Process image and generate masks
#                 image = Image.open(image_path)
#                 image = np.array(image.convert("RGB"))
#                 masks = mask_generator.generate(image)
                
#                 # Save masks and metadata
#                 # self.save_data(image_key, masks)
#                 # Process and save each segmentation
#                 processed_segmentations = []
#                 for idx, seg in enumerate(masks):
#                     # Create a copy of the segmentation data without the array
#                     seg_metadata = seg.copy()
                    
#                     # Save the segmentation array as a .npy file
#                     array_path = self._get_array_path(image_key, idx)
#                     self._save_array(seg['segmentation'], array_path)
                    
#                     # Replace the array in metadata with the file path
#                     seg_metadata['segmentation'] = os.path.basename(array_path)
#                     processed_segmentations.append(seg_metadata)
                
#                 # Save metadata
#                 metadata[image_key] = processed_segmentations
                
#                 # Save updated metadata and backup
#                 self._safe_json_write(metadata, self.metadata_file)
#                 self._safe_json_write(metadata, self.backup_file)
                
                
#             except Exception as e:
#                 print(f"Error processing image {image_key}: {str(e)}")
#                 continue

#     def load_data(self, image_key: str, indices: Union[int, List[int]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
#         """
#         Load segmentation data for a specific image.
#         """
#         metadata = self._safe_json_read(self.metadata_file)
            
#         if image_key not in metadata:
#             raise KeyError(f"No data found for image: {image_key}")
        
#         image_metadata = metadata[image_key]
        
#         # Handle single index
#         if isinstance(indices, int):
#             result = image_metadata[indices].copy()
#             array_path = os.path.join(self.arrays_dir, image_key, result['segmentation'])
#             result['segmentation'] = np.load(array_path)
#             return result
        
#         # Handle multiple indices or None (all segmentations)
#         load_indices = indices if indices is not None else range(len(image_metadata))
#         result = []
        
#         for idx in load_indices:
#             entry = image_metadata[idx].copy()
#             array_path = os.path.join(self.arrays_dir, image_key, entry['segmentation'])
#             entry['segmentation'] = np.load(array_path)
#             result.append(entry)
        
#         return result
    

import os
import json
import numpy as np
import tempfile
import shutil
from typing import Dict, List, Any, Union, Tuple
from PIL import Image
from tqdm import tqdm

# class PartitionedSegmentationManager:
#     def __init__(self, base_directory: str, partition_id: int, total_partitions: int):
#         """
#         Initialize the manager with partition information.
        
#         Args:
#             base_directory (str): Base directory for storing all data
#             partition_id (int): ID of current partition (0 to total_partitions-1)
#             total_partitions (int): Total number of partitions
#         """
#         self.partition_id = partition_id
#         self.total_partitions = total_partitions
        
#         # Create partition-specific directory
#         self.partition_dir = os.path.join(base_directory, f"partition_{partition_id}")
#         self.arrays_dir = os.path.join(self.partition_dir, "arrays")
#         self.metadata_file = os.path.join(self.partition_dir, "metadata.json")
#         self.backup_file = os.path.join(self.partition_dir, "metadata_backup.json")
        
#         # Create directories
#         os.makedirs(self.partition_dir, exist_ok=True)
#         os.makedirs(self.arrays_dir, exist_ok=True)
        
#         # Initialize metadata
#         self._initialize_or_recover_metadata()
    
class PartitionedSegmentationManager:
    def __init__(self, arrays_directory: str, metadata_directory: str, partition_id: int, total_partitions: int):
        """
        Initialize the manager with partition information.
        
        Args:
            arrays_directory (str): Directory for storing array data.
            metadata_directory (str): Directory for storing metadata.
            partition_id (int): ID of current partition (0 to total_partitions-1).
            total_partitions (int): Total number of partitions.
        """
        self.partition_id = partition_id
        self.total_partitions = total_partitions
        
        # Create partition-specific directories
        self.arrays_dir = os.path.join(arrays_directory, f"partition_{partition_id}")
        self.metadata_file = os.path.join(metadata_directory, f"metadata_partition_{partition_id}.json")
        
        # Create directories if they don't exist
        os.makedirs(self.arrays_dir, exist_ok=True)
        os.makedirs(metadata_directory, exist_ok=True)
        
        # Initialize metadata
        self._initialize_or_recover_metadata()

    def _safe_json_read(self, filepath: str) -> Dict:
        """
        Safely read JSON file with error handling and recovery.
        """
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Try to recover the JSON by reading until the last valid line
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Find the last complete object (ending with })
            last_brace_index = content.rfind('}')
            if last_brace_index != -1:
                valid_content = content[:last_brace_index + 1]
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
        with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False) as tf:
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
        with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False, suffix='.npy') as tf:
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
            self._save_array(seg['segmentation'], array_path)
            
            # Replace the array in metadata with the file path
            seg_metadata['segmentation'] = os.path.basename(array_path)
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
        
        start_idx = self.partition_id * items_per_partition + min(self.partition_id, remainder)
        end_idx = start_idx + items_per_partition + (1 if self.partition_id < remainder else 0)
        
        return start_idx, end_idx

    def process_partition(self, data: List[Dict[str, str]], mask_generator: Any) -> None:
        """
        Process only this partition's portion of the dataset.
        """
        start_idx, end_idx = self.get_partition_indices(len(data))
        partition_data = data[start_idx:end_idx]
        
        metadata, image_keys = self.metadata_and_list_image_keys()
        
        for item in tqdm(partition_data, desc=f"Processing partition {self.partition_id}"):
            # Get image key
            image_key = item['image'].strip('/mnt/nushare2/data/baliao/multimodal/data/')
            
            # Skip if already processed
            if image_key in image_keys:
                continue
            
            try:
                # Process image
                image_path = '/home/mnulli1/data/LLaVA-Pretrain/' + item['image']
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
                    self._save_array(seg['segmentation'], array_path)
                    
                    # Replace the array in metadata with the file path
                    seg_metadata['segmentation'] = os.path.basename(array_path)
                    processed_segmentations.append(seg_metadata)
                
                # Save metadata
                metadata[image_key] = processed_segmentations
                
                # Save updated metadata and backup
                self._safe_json_write(metadata, self.metadata_file)

                
            except Exception as e:
                print(f"Error processing image {image_key} in partition {self.partition_id}: {str(e)}")
                continue
    
    def load_data(self, image_key: str, indices: Union[int, List[int]] = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
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
            array_path = os.path.join(self.arrays_dir, image_key, result['segmentation'])
            result['segmentation'] = np.load(array_path)
            return result
        
        # Handle multiple indices or None (all segmentations)
        load_indices = indices if indices is not None else range(len(image_metadata))
        result = []
        
        for idx in load_indices:
            entry = image_metadata[idx].copy()
            array_path = os.path.join(self.arrays_dir, image_key, entry['segmentation'])
            entry['segmentation'] = np.load(array_path)
            result.append(entry)
        
        return result
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sam2_checkpoint", type=str, 
                        default= "/data/chatgpt/notebooks/mnulli/sam2/checkpoints/sam2.1_hiera_large.pt",
                        help='checkpoint of sam2 model')
    parser.add_argument("--model_cfg", type=str,
                        default="/data/chatgpt/notebooks/mnulli/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
                        help='checkpoint of model configuration')
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("--metadata_directory", type=str, 
                        default='/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data',
                        help='Where to store the metadata file with pointers to .npy masks files')
    parser.add_argument("--arrays_directory", type=str, 
                        default='/mnt/nushare2/data/mnulli/thesis/data/sam2/segmentation_data',
                        help='Where to store the actual .npy arrays of segmentation data')
    parser.add_argument("--data_path", type=str, 
                        default='/mnt/nushare2/data/mnulli/pretrainingdata/blip_laion_cc_sbu_558k.json',
                        help='Path to data file')
    parser.add_argument('--partition-id', type=int, required=True,
                      help='ID of the partition to process (0-9)')
    parser.add_argument('--total-partitions', type=int, default=10,
                      help='Total number of partitions')


    args = parser.parse_args()

    mask_generator = sam2_instance(args)

    data = data_instance(args)

    manager = PartitionedSegmentationManager(
        arrays_directory = args.arrays_directory,
        metadata_directory = args.metadata_directory,
        partition_id=args.partition_id,
        total_partitions=args.total_partitions)

    manager.process_partition(data, mask_generator)

