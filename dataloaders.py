import os


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as nd

import torch
import torch.utils as utils

from torchio import transforms

import paths
import utils

task_1_data_path = paths.task_1_data_path
task_2_data_path = paths.task_2_data_path
task_3_data_path = paths.task_3_data_path
task_4_data_path = paths.task_4_data_path

test_task_1_data_path = paths.test_task_1_data_path
test_task_2_data_path = paths.test_task_2_data_path
test_task_3_data_path = paths.test_task_3_data_path
test_task_4_data_path = paths.test_task_4_data_path

class Task1Loader(torch.utils.data.Dataset):
    def __init__(self, mode='training', validation_ids=None, load_landmarks=False):
        self.data_path = os.path.join(task_1_data_path)
        self.mode = mode
        self.validation_ids = [item[0] for item in validation_ids]
        self.load_landmarks = load_landmarks

        if self.mode == "training":
            self.all_ids = self.get_all_ids()
            print("Training pairs: ", self.all_ids)
        elif self.mode == "validation":
            self.all_ids = self.validation_ids
            print("Validation pairs: ", self.validation_ids)

    def get_all_ids(self):
        all_ids = [item for item in os.listdir(self.data_path) if "Case" in item]
        all_ids = [int(item.split("Case")[1]) for item in all_ids]
        all_ids = [item for item in all_ids if item not in self.validation_ids]
        return all_ids

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        current_id = self.all_ids[idx]
        fixed_path, moving_path = utils.build_task_1_path(current_id, current_id)
        fixed_image, moving_image = utils.load_volume(fixed_path), utils.load_volume(moving_path)
        fixed_image, moving_image = utils.normalize(fixed_image), utils.normalize(moving_image)
        if self.load_landmarks:
            fixed_label_path = os.path.join(os.path.split(self.data_path)[0], "landmarks", "Voxels", "Case" + str(current_id) + "-MRI-seg.nii.gz")
            moving_label_path = os.path.join(os.path.split(self.data_path)[0], "landmarks", "Voxels", "Case" + str(current_id) + "-US-seg.nii.gz")
            fixed_label, moving_label = utils.load_mask(fixed_label_path), utils.load_mask(moving_label_path)
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_label_tensor, moving_label_tensor = torch.from_numpy(fixed_label.astype(np.float32)), torch.from_numpy(moving_label.astype(np.float32))

            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)  
            fixed_label_tensor.unsqueeze_(0)
            moving_label_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor, fixed_label_tensor, moving_label_tensor
        else:
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor

class Task2Loader(torch.utils.data.Dataset):
    def __init__(self, mode='training', validation_pairs=None, load_labels=False):
        self.data_path = os.path.join(task_2_data_path, "scans")
        self.mode = mode
        self.validation_pairs = validation_pairs
        self.load_labels = load_labels

        self.tr = transforms.RandomElasticDeformation(num_control_points=5, max_displacement=4, p = 0.7)

        if self.mode == "training":
            self.all_pairs = self.create_pairs()
            print("Training pairs: ", self.all_pairs)
        elif self.mode == "validation":
            self.all_pairs = self.validation_pairs
            print("Validation pairs: ", self.validation_pairs)

    def create_pairs(self):
        all_files = os.listdir(self.data_path)
        all_ids = [int(item.split('case_')[1].split(".nii.gz")[0].split("_")[0]) for item in all_files]
        all_pairs = []
        for i in all_ids:
            if (i, i) not in self.validation_pairs:
                all_pairs.append((i, i))
        all_pairs = list(set(all_pairs))
        return all_pairs

    def __len__(self):
        return len(self.all_pairs)

    def __getitem__(self, idx):
        moving_id = self.all_pairs[idx][1]
        fixed_id = self.all_pairs[idx][0]
        fixed_path, moving_path = utils.build_task_2_path(fixed_id, moving_id)
        fixed_image, moving_image = utils.load_volume(fixed_path), utils.load_volume(moving_path)
        fixed_image, moving_image = utils.normalize_to_range(fixed_image, -1000, 2000), utils.normalize_to_range(moving_image, -1000, 2000)
        if self.load_labels:
            fixed_label_path = fixed_path.replace("scans", "lungMasks")
            moving_label_path = moving_path.replace("scans", "lungMasks")
            fixed_label, moving_label = utils.load_mask(fixed_label_path), utils.load_mask(moving_label_path)
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_label_tensor, moving_label_tensor = torch.from_numpy(fixed_label.astype(np.float32)), torch.from_numpy(moving_label.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)  
            fixed_label_tensor.unsqueeze_(0)
            moving_label_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor, fixed_label_tensor, moving_label_tensor
        else:
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor

class Task3Loader(torch.utils.data.Dataset):
    def __init__(self, mode='training', validation_pairs=None, load_labels=False, exclude_all=False):
        if mode == "test":
            self.data_path = os.path.join(test_task_3_data_path, "img")
        else:
            self.data_path = os.path.join(task_3_data_path, "img")
        self.mode = mode
        self.validation_pairs = validation_pairs
        self.load_labels = load_labels
        self.exclude_all = exclude_all # There was an error that did not exlude all validation cases from training. exclude_all = True fixes that.

        if self.mode == "training":
            self.all_pairs = self.create_pairs()
            print("Training pairs: ", self.all_pairs)
        elif self.mode == "validation":
            self.all_pairs = self.validation_pairs
            print("Validation pairs: ", self.validation_pairs)
        else:
            self.all_pairs = self.validation_pairs
            print("Test pairs: ", self.validation_pairs)

        self.tr = transforms.RandomElasticDeformation(num_control_points=5, max_displacement=4, p = 0.7)

    def create_pairs(self):
        all_files = os.listdir(self.data_path)
        all_ids = [int(item.split('img')[1].split(".nii.gz")[0]) for item in all_files]
        all_pairs = []
        all_validation_cases = set([item for tup in self.validation_pairs for item in tup])
        for i in all_ids:
            for j in all_ids:
                if i != j and (i, j) not in self.validation_pairs and (j, i) not in self.validation_pairs:
                    if self.exclude_all:
                        if i in all_validation_cases or j in all_validation_cases:
                            continue
                        else:
                            all_pairs.append((i, j))
                    else:
                        all_pairs.append((i, j))
        all_pairs = list(set(all_pairs))
        return all_pairs

    def __len__(self):
        if self.mode == "training":
            return len(self.all_pairs)
        elif self.mode == "validation":
            return len(self.all_pairs)
        else:
            return len(self.all_pairs)

    def __getitem__(self, idx):
        moving_id = self.all_pairs[idx][1]
        fixed_id = self.all_pairs[idx][0]
        if self.mode == "test":
            fixed_path, moving_path = utils.build_test_task_3_path(fixed_id, moving_id)
        else:
            fixed_path, moving_path = utils.build_task_3_path(fixed_id, moving_id)
        fixed_image, moving_image = utils.load_volume(fixed_path), utils.load_volume(moving_path)
        fixed_image, moving_image = utils.normalize_to_range(fixed_image, -1000, 2000), utils.normalize_to_range(moving_image, -1000, 2000)
        if self.load_labels:
            fixed_label_path = fixed_path.replace("img", "label")
            moving_label_path = moving_path.replace("img", "label")
            fixed_label, moving_label = utils.load_mask(fixed_label_path), utils.load_mask(moving_label_path)
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_label_tensor, moving_label_tensor = torch.from_numpy(fixed_label.astype(np.float32)), torch.from_numpy(moving_label.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)  
            fixed_label_tensor.unsqueeze_(0)
            moving_label_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor, fixed_label_tensor, moving_label_tensor
        else:
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor


class Task4Loader(torch.utils.data.Dataset):
    def __init__(self, mode='training', validation_pairs=None, load_labels=False, exclude_all=False):
        if mode == "test":
            self.data_path = os.path.join(test_task_4_data_path, "img")
        else:
            self.data_path = os.path.join(task_4_data_path, "img")
        self.mode = mode
        self.validation_pairs = validation_pairs
        self.load_labels = load_labels
        self.exclude_all = exclude_all # There was an error that did not exlude all validation cases from training. exclude_all = True fixes that.

        if self.mode == "training":
            self.all_pairs = self.create_pairs()
        elif self.mode == "validation":
            self.all_pairs = self.validation_pairs
            print("Validation pairs: ", self.validation_pairs)
        elif self.mode == "test":
            self.all_pairs = self.validation_pairs
            print("Test pairs: ", self.validation_pairs)

        self.tr = transforms.RandomElasticDeformation(num_control_points=5, max_displacement=4, p = 0.7)

    def create_pairs(self):
        all_files = os.listdir(self.data_path)
        all_ids = [int(item.split('hippocampus_')[1].split(".nii.gz")[0]) for item in all_files]
        all_pairs = []
        all_validation_cases = set([item for tup in self.validation_pairs for item in tup])
        for i in all_ids:
            for j in all_ids:
                if i != j and (i, j) not in self.validation_pairs and (j, i) not in self.validation_pairs:
                    if self.exclude_all:
                        if i in all_validation_cases or j in all_validation_cases:
                            continue
                        else:
                            all_pairs.append((i, j))
                    else:
                        all_pairs.append((i, j))
        all_pairs = list(set(all_pairs))
        return all_pairs

    def __len__(self):
        if self.mode == "training":
            return len(self.all_pairs)
        elif self.mode == "validation":
            return len(self.all_pairs)
        else:
            return len(self.all_pairs)

    def __getitem__(self, idx):
        moving_id = self.all_pairs[idx][1]
        fixed_id = self.all_pairs[idx][0]
        if self.mode == "test":
            fixed_path, moving_path = utils.build_test_task_4_path(fixed_id, moving_id)
        else:
            fixed_path, moving_path = utils.build_task_4_path(fixed_id, moving_id)
        fixed_image, moving_image = utils.load_volume(fixed_path), utils.load_volume(moving_path)
        fixed_image, moving_image = utils.normalize(fixed_image), utils.normalize(moving_image)

        if self.load_labels:
            fixed_label_path = fixed_path.replace("img", "label")
            moving_label_path = moving_path.replace("img", "label")
            fixed_label, moving_label = utils.load_mask(fixed_label_path), utils.load_mask(moving_label_path)
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_label_tensor, moving_label_tensor = torch.from_numpy(fixed_label.astype(np.float32)), torch.from_numpy(moving_label.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)  
            fixed_label_tensor.unsqueeze_(0)
            moving_label_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor, fixed_label_tensor, moving_label_tensor
        else:
            fixed_tensor, moving_tensor = torch.from_numpy(fixed_image.astype(np.float32)), torch.from_numpy(moving_image.astype(np.float32))
            fixed_tensor.unsqueeze_(0)
            moving_tensor.unsqueeze_(0)
            return fixed_tensor, moving_tensor
