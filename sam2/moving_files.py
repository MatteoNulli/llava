# import os
# import numpy as np
# import json

# def group_masks_to_json(image_dir, output_dir):
#     """
#     Groups all mask_*.npy files in the specified directory and saves them as a JSON file.
    
#     Parameters:
#     image_dir (str): Path to the directory containing mask_*.npy files.
#     """
#     mask_files = sorted([f for f in os.listdir(image_dir) if f.startswith("mask_") and f.endswith(".npy")])
    
#     if not mask_files:
#         print(f"No mask files found in {image_dir}")
#         return
    
#     # Load masks and convert to lists for JSON serialization
#     mask_data = {f: np.load(os.path.join(image_dir, f)).tolist() for f in mask_files}
    
#     # Extract the base name from the directory (assuming the last folder name is the image ID)
#     image_id = os.path.basename(os.path.normpath(image_dir))
#     output_file = os.path.join(output_dir, f"{image_id}.json")
    
#     # Save to JSON
#     with open(output_file, "w") as f:
#         json.dump(mask_data, f, indent=4)
    
#     print(f"Saved masks to {output_file}")

# if __name__ == "__main__":
#     image_directory = "/scratch-shared/mnulli1/segmentation_data_0/partition_0/arrays/00000/000001052.jpg"
#     output_directory = '/home/mnulli1/'
#     group_masks_to_json(image_directory, output_directory)


import os
import numpy as np
import json

# def group_masks_to_json(base_directory, output_directory):
#     """
#     Groups all mask_*.npy files in all subdirectories of the '00000' directory within base_directory
#     and saves them as JSON files in the specified output directory.
    
#     Parameters:
#     base_directory (str): Path to the base directory containing the '00000' subdirectory.
#     output_directory (str): Path to the directory where JSON files will be saved.
#     """
#     target_directory = os.path.join(base_directory, "00000")
#     os.makedirs(output_directory, exist_ok=True)
    
#     for root, _, files in os.walk(target_directory):
#         mask_files = sorted([f for f in files if f.startswith("mask_") and f.endswith(".npy")])
        
#         if not mask_files:
#             continue
        
#         # Load masks and convert to lists for JSON serialization
#         mask_data = {f: np.load(os.path.join(root, f)).tolist() for f in mask_files}
        
#         # Extract the base name from the directory (assuming the last folder name is the image ID)
#         image_id = os.path.basename(os.path.normpath(root))
#         output_file = os.path.join(output_directory, f"{image_id}.json")
        
#         # Save to JSON
#         with open(output_file, "w") as f:
#             json.dump(mask_data, f, indent=4)
        
#         print(f"Saved masks to {output_file}")

# if __name__ == "__main__":
#     base_directory = "/scratch-shared/mnulli1/segmentation_data_0/partition_0/arrays"
#     output_directory = "/home/mnulli1/output_jsons"
#     group_masks_to_json(base_directory, output_directory)
    
import os
import numpy as np
import json
import shutil

def process_and_move_masks(base_directory, temp_output_directory, final_output_directory, folder_id):
    """
    Processes mask_*.npy files in a given folder, converts them to JSON, moves the JSON files,
    and deletes the original .npy files.
    
    Parameters:
    base_directory (str): Path to the base directory containing subdirectories with .npy files.
    temp_output_directory (str): Temporary output directory for JSON files.
    final_output_directory (str): Final destination directory for JSON files.
    folder_id (str): Folder ID to process (e.g., "00000").
    """
    source_directory = os.path.join(base_directory, folder_id)
    temp_dir = os.path.join(temp_output_directory, folder_id)
    final_dir = os.path.join(final_output_directory, folder_id)
    
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    
    for root, _, files in os.walk(source_directory):
        mask_files = sorted([f for f in files if f.startswith("mask_") and f.endswith(".npy")])
        
        if not mask_files:
            continue
        
        mask_data = {f: np.load(os.path.join(root, f)).tolist() for f in mask_files}
        image_id = os.path.basename(os.path.normpath(root))
        json_file = os.path.join(temp_dir, f"{image_id}.json")
        
        with open(json_file, "w") as f:
            json.dump(mask_data, f, indent=4)
        
        print(f"Saved JSON to {json_file}")
        
        # Delete original .npy files
        # print("Deleting original .npy files")
        # for mask_file in mask_files:
        #     os.remove(os.path.join(root, mask_file))
        
    # Move JSON files to the final destination
    for json_file in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, json_file), os.path.join(final_dir, json_file))
    
    print(f"Moved JSON files to {final_dir}")
    
if __name__ == "__main__":
    base_directory = "/scratch-shared/mnulli1/segmentation_data_0/arrays/partition_0"
    temp_output_directory = "/home/mnulli1/temp_outs/partition_0/arrays"
    final_output_directory = "/scratch-shared/mnulli1/segmentation_data_0/arrays/partition_0"
    folder_id = "00000"  # For testing first on 00000
    
    process_and_move_masks(base_directory, temp_output_directory, final_output_directory, folder_id)
