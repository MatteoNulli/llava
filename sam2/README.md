# Running Segmentation masks
Be sure to have cloned https://github.com/MatteoNulli/llava/tree/main and 
```
cd sam2/
```

## Masks for LLaVA dataset
to be written


## Masks for Cambrian dataset
You will need to modify the following two scripts:
```
automatic_mask_generator_llava.py
automatic_masks_cambrian.sh 
```

In the first you will need to change the `_compute_image_key()` function to adjust the self.cambrian path to be the name of the folder in which you have all the pictures stored. 
Say the pictures are all stored into a folder like this: `cambrian/data/images`, then you need to put `images/` there instead of `nyu-visionx--Cambrian-10M--extracted/`

In the second you would need to change the following:
- DATA_PATH
- ARRAYS_DIR
- METADATA_DIR
- SAM2_CHECKPOINT
- PARTITION_ID (to be changed for every run)
If your environment has all the relevant packages also remove the `pip install...` 

If everthing works right your file system should look like this
`segmentation_data/arrays/partition_{i}/coco/train2017/0000001829.jpg/0000001829.npy`

`segmentation_data/metadata/metadata_partition_{i}.json`

Final note, the model used is [sam2.1_hiera_large.pt](https://huggingface.co/facebook/sam2.1-hiera-large/blob/main/sam2.1_hiera_large.pt).


```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}
```
