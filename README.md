# An End-To-End Framework For Low-Resolution Remote Sensing Semantic Segmentation
An end-to-end framework that unites super-resolution and semantic segmentation for low-resolution remote sensing images.

------------

### Training
Modify and run train.sh file according to the required configuration or 

```
python3 main.py [parameters]
```

**Parameters**: please refer to the definition in main.py code

------------

### Testing
Modify and run test.sh file according to the required configuration or 

```
python3 eval.py [parameters]
```

**Parameters**: please refer to the definition in eval.py code

------------

### Dataset Folder configuration

- dataset_name (main dataset folder -> data_dir in the parameters)
	- train (can be changed by the parameter train_dir)
		- {training HR images} (the images are automatically downsampled during runtime)
	- train_msk (can be changed by the parameter train_dir)
  		- {semantic segmentation ground truth of training HR images} (the masks should have the exact same name as the corresponding image in the train folder)   
	- test (can be changed by the parameter test_dir)
  		- {testing LR images} (the images are automatically downsampled during runtime)
	- test_msk (can be changed by the parameter test_dir)
  		- {semantic segmentation ground truth of testing HR images} (the masks should have the exact same name as the corresponding image in the test folder)
	- (OPTIONAL) val (can be changed by the parameter val_dir) (if not set, a portion of the training set will be separated for validation)
  		- {validation HR images}
	- (OPTIONAL) val_msk (can be changed by the parameter val_dir)
  		- {semantic segmentation ground truth of validation HR images} (the masks should have the exact same name as the corresponding image in the validation folder)
   
### Observation:
  - The masks (ground truth images) should be converted beforehand to a 1-channel (grayscale) image, where each pixel represents the class of the image (pixel value 0 for the first class, pixel value 1 for second class, ..., pixel value N-1 for the Nth class).

### Dataset Folder Organization Example:
  - data_root_path/
    - train/
      - top_mosaic_09cm_area1_crop1.png
      - top_mosaic_09cm_area1_crop2.png
    - train_msk/ (contains 1-channel masks images with pixels values between 0 and N-1, where N is the number of classes of the dataset)
      - top_mosaic_09cm_area1_crop1.png
      - top_mosaic_09cm_area1_crop2.png 
    - test/
      - top_mosaic_09cm_area11_crop0.png 
    - test_msk/ (contains 1-channel masks images with pixels values between 0 and N-1, where N is the number of classes of the dataset)
      - top_mosaic_09cm_area11_crop0.png


------------

## Citations
If you find this work useful, please consider citing it:

M. B. Pereira and J. A. d. Santos, "An End-To-End Framework For Low-Resolution Remote Sensing Semantic Segmentation," 2020 IEEE Latin American GRSS & ISPRS Remote Sensing Conference (LAGIRS), Santiago, Chile, 2020, pp. 6-11, doi: 10.1109/LAGIRS48042.2020.9165642.
