from os.path import join
from torchvision.transforms import Compose, ToTensor
from dataset import DatasetFromFolderEval, DatasetFromFolder, DatasetFromFolderValidation

def transform():
    return Compose([
        ToTensor(),
    ])

def get_training_set(data_dir, train_dir, patch_size, sr_patch_size, upscale_factor, num_classes, data_augmentation):
    return DatasetFromFolder(data_dir, train_dir, patch_size, sr_patch_size, upscale_factor, num_classes, data_augmentation, transform=transform())

def get_eval_set(data_dir, test_dir, upscale_factor, num_classes):
    return DatasetFromFolderEval(data_dir, test_dir, upscale_factor, num_classes, transform=transform())    

def get_validation_set(image_dir, upscale_factor, num_classes):
    return DatasetFromFolderValidation(image_dir, upscale_factor, num_classes, transform=transform())
