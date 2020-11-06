import os

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import scipy.ndimage
import scipy.ndimage as nd

import torch
import torch.nn.functional as F
import torch.utils
import torch.nn as nn
import torch.optim as optim

import paths

data_path = paths.data_path
task_1_data_path = paths.task_1_data_path
task_2_data_path = paths.task_2_data_path
task_3_data_path = paths.task_3_data_path
task_4_data_path = paths.task_4_data_path

test_task_1_data_path = paths.test_task_1_data_path
test_task_2_data_path = paths.test_task_2_data_path
test_task_3_data_path = paths.test_task_3_data_path
test_task_4_data_path = paths.test_task_4_data_path


def load_volume(path):
    volume = sitk.GetArrayFromImage(sitk.ReadImage(path)).swapaxes(0, 2)
    return volume

def load_mask(path):
    mask = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(mask).swapaxes(0, 2)
    return volume

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

def normalize_to_range(array, min_value, max_value):
    return (array - min_value) / (max_value - min_value)

def build_df_path(df, id):
    fixed_id = df.loc[id, :]['fixed']
    moving_id =  df.loc[id, :]['moving']
    fixed_str = "%04d" % fixed_id
    moving_str = "%04d" % moving_id
    output_path = "disp_" + fixed_str + "_" + moving_str + ".npy"
    return output_path

def build_task_1_path(fixed_id, moving_id):
    fixed_path = os.path.join(task_1_data_path, "Case" + str(fixed_id), "Case" + str(fixed_id) + "-T1-resize.nii")
    moving_path = os.path.join(task_1_data_path, "Case" + str(moving_id), "Case" + str(moving_id) + "-US-before-resize.nii")
    return fixed_path, moving_path

def build_task_2_path(fixed_id, moving_id):
    fixed_str = "%03d" % fixed_id
    moving_str = "%03d" % moving_id
    fixed_path = os.path.join(task_2_data_path, "scans", "case_" + fixed_str + "_exp.nii.gz")
    moving_path = os.path.join(task_2_data_path, "scans", "case_" + moving_str + "_insp.nii.gz") 
    return fixed_path, moving_path

def build_task_3_path(fixed_id, moving_id):
    fixed_str = "%04d" % fixed_id
    moving_str = "%04d" % moving_id
    fixed_path = os.path.join(task_3_data_path, "img", "img" + fixed_str + ".nii.gz")
    moving_path = os.path.join(task_3_data_path, "img", "img" + moving_str + ".nii.gz") 
    return fixed_path, moving_path

def build_test_task_3_path(fixed_id, moving_id):
    fixed_str = "%04d" % fixed_id
    moving_str = "%04d" % moving_id
    fixed_path = os.path.join(test_task_3_data_path, "img", "img" + fixed_str + ".nii.gz")
    moving_path = os.path.join(test_task_3_data_path, "img", "img" + moving_str + ".nii.gz") 
    return fixed_path, moving_path

def build_task_4_path(fixed_id, moving_id):
    fixed_str = "%03d" % fixed_id
    moving_str = "%03d" % moving_id
    fixed_path = os.path.join(task_4_data_path, "img", "hippocampus_" + fixed_str + ".nii.gz")
    moving_path = os.path.join(task_4_data_path, "img", "hippocampus_" + moving_str + ".nii.gz") 
    return fixed_path, moving_path

def build_test_task_4_path(fixed_id, moving_id):
    fixed_str = "%03d" % fixed_id
    moving_str = "%03d" % moving_id
    fixed_path = os.path.join(test_task_4_data_path, "img", "hippocampus_" + fixed_str + ".nii.gz")
    moving_path = os.path.join(test_task_4_data_path, "img", "hippocampus_" + moving_str + ".nii.gz") 
    return fixed_path, moving_path

def get_task_1_pairs():
    df_path = os.path.join(data_path, "Task1", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_task_2_pairs():
    df_path = os.path.join(data_path, "Task2", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_task_3_pairs():
    df_path = os.path.join(data_path, "Task3", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_test_task_3_pairs():
    df_path = os.path.join(data_path, "Test", "Task3", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_task_4_pairs():
    df_path = os.path.join(data_path, "Task4", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_test_task_4_pairs():
    df_path = os.path.join(data_path, "Test", "Task4", "pairs_val.csv")
    df, len_df = load_dataframe(df_path)
    pairs = [(int(df.loc[i, :]['fixed']), int(df.loc[i, :]['moving'])) for i in range(len_df)]
    return pairs

def get_ids(df, id):
    fixed_id = df.loc[id, :]['fixed']
    moving_id =  df.loc[id, :][' moving']   
    return fixed_id, moving_id

def load_task_1_pair(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_1_path(fixed_id, moving_id)
    fixed_image = load_volume(fixed_path)
    moving_image = load_volume(moving_path)
    return fixed_image, moving_image

def load_task_2_pair(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_2_path(fixed_id, moving_id)
    fixed_image = load_volume(fixed_path)
    moving_image = load_volume(moving_path)
    return fixed_image, moving_image

def load_task_3_pair(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_3_path(fixed_id, moving_id)
    fixed_image = load_volume(fixed_path)
    moving_image = load_volume(moving_path)
    return fixed_image, moving_image

def load_task_4_pair(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_4_path(fixed_id, moving_id)
    fixed_image = load_volume(fixed_path)
    moving_image = load_volume(moving_path)
    return fixed_image, moving_image

def load_test_task_4_pair(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_test_task_4_path(fixed_id, moving_id)
    fixed_image = load_volume(fixed_path)
    moving_image = load_volume(moving_path)
    return fixed_image, moving_image

def load_task_1_labels(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_label_path = os.path.join(os.path.split(task_1_data_path)[0], "landmarks", "Voxels", "Case" + str(fixed_id) + "-MRI-seg.nii.gz")
    moving_label_path = os.path.join(os.path.split(task_1_data_path)[0], "landmarks", "Voxels", "Case" + str(moving_id) + "-US-seg.nii.gz")  
    fixed_label, moving_label = load_mask(fixed_label_path), load_mask(moving_label_path)
    return fixed_label, moving_label

def load_task_2_labels(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_2_path(fixed_id, moving_id)    
    fixed_label_path = fixed_path.replace("scans", "lungMasks")
    moving_label_path = moving_path.replace("scans", "lungMasks")
    fixed_label, moving_label = load_mask(fixed_label_path), load_mask(moving_label_path)
    return fixed_label, moving_label

def load_task_3_labels(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_3_path(fixed_id, moving_id)    
    fixed_label_path = fixed_path.replace("img", "label")
    moving_label_path = moving_path.replace("img", "label")
    fixed_label, moving_label = load_mask(fixed_label_path), load_mask(moving_label_path)
    return fixed_label, moving_label

def load_task_4_labels(df, id):
    fixed_id, moving_id = get_ids(df, id)
    fixed_path, moving_path = build_task_4_path(fixed_id, moving_id)    
    fixed_label_path = fixed_path.replace("img", "label")
    moving_label_path = moving_path.replace("img", "label")
    fixed_label, moving_label = load_mask(fixed_label_path), load_mask(moving_label_path)
    return fixed_label, moving_label

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    num_samples = len(df)
    return df, num_samples

def create_empty_df(shape):
    return np.zeros(shape).astype(np.float32)

def save_df(df, output_path):
    np.save(output_path, df)

def load_df(input_path):
    return np.load(input_path)

def tensor_laplacian(tensor, device="cpu"):
    laplacian_filter = torch.Tensor([
        [
        [0, 0, 0], 
        [0, -1, 0], 
        [0, 0, 0]
        ],
        [
        [0, -1, 0], 
        [-1, 6, -1], 
        [0, -1, 0]
        ],
        [
        [0, 0, 0], 
        [0, -1, 0], 
        [0, 0, 0]
        ]
    ]).type(tensor.type()).to(device)
    laplacian = F.conv3d(tensor, laplacian_filter.view(1, 1, 3, 3, 3), padding=1)
    return laplacian

def warp_tensors(tensors, displacement_fields, device="cpu", interpolation_mode='bilinear'):
    size = tensors.size()
    no_samples = size[0]
    y_size = size[2]
    x_size = size[3]
    z_size = size[4]
    gz, gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size), torch.arange(z_size))
    gx = gx.type(torch.FloatTensor).to(device)
    gy = gy.type(torch.FloatTensor).to(device)
    gz = gz.type(torch.FloatTensor).to(device)
    grid_x = (gx / (z_size - 1) - 0.5)*2
    grid_y = (gy / (x_size - 1) - 0.5)*2
    grid_z = (gz / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1), grid_x.size(2))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1), grid_y.size(2))
    n_grid_z = grid_z.view(1, -1).repeat(no_samples, 1).view(-1, grid_z.size(0), grid_z.size(1), grid_z.size(2))
    n_grid = torch.stack((n_grid_x, n_grid_y, n_grid_z), dim=4)
    displacement_fields = displacement_fields.permute(0, 2, 3, 4, 1)
    u_x = displacement_fields[:, :, :, :, 0]
    u_y = displacement_fields[:, :, :, :, 1]
    u_z = displacement_fields[:, :, :, :, 2]
    u_x = u_x / (x_size - 1) * 2
    u_y = u_y / (y_size - 1) * 2
    u_z = u_z / (z_size - 1) * 2
    n_grid[:, :, :, :, 0] = n_grid[:, :, :, :, 0] + u_z
    n_grid[:, :, :, :, 1] = n_grid[:, :, :, :, 1] + u_x
    n_grid[:, :, :, :, 2] = n_grid[:, :, :, :, 2] + u_y
    transformed_tensors = F.grid_sample(tensors, n_grid, mode=interpolation_mode, padding_mode='zeros')
    return transformed_tensors

def resample_tensors(tensors, new_size, device="cpu", interpolation_mode='bilinear'):
    current_size = tensors.size()
    no_samples = new_size[0]
    y_size = new_size[2]
    x_size = new_size[3]
    z_size = new_size[4]
    gz, gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size), torch.arange(z_size))
    gx = gx.type(torch.FloatTensor).to(device)
    gy = gy.type(torch.FloatTensor).to(device)
    gz = gz.type(torch.FloatTensor).to(device)
    grid_x = (gx / (z_size - 1) - 0.5)*2
    grid_y = (gy / (x_size - 1) - 0.5)*2
    grid_z = (gz / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1), grid_x.size(2))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1), grid_y.size(2))
    n_grid_z = grid_z.view(1, -1).repeat(no_samples, 1).view(-1, grid_z.size(0), grid_z.size(1), grid_z.size(2))
    n_grid = torch.stack((n_grid_x, n_grid_y, n_grid_z), dim=4)
    resampled_tensors = F.grid_sample(tensors, n_grid, mode=interpolation_mode, padding_mode='zeros')
    return resampled_tensors

def df_to_df(df, device="cpu"):
    size = df.size()
    no_samples = size[0]
    y_size = size[1]
    x_size = size[2]
    z_size = size[3]
    gz, gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size), torch.arange(z_size))
    gx = gx.type(torch.FloatTensor).to(device)
    gy = gy.type(torch.FloatTensor).to(device)
    gz = gz.type(torch.FloatTensor).to(device)
    grid_x = (gx / (z_size - 1) - 0.5)*2
    grid_y = (gy / (x_size - 1) - 0.5)*2
    grid_z = (gz / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1), grid_x.size(2))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1), grid_y.size(2))
    n_grid_z = grid_z.view(1, -1).repeat(no_samples, 1).view(-1, grid_z.size(0), grid_z.size(1), grid_z.size(2))
    n_grid = torch.stack((n_grid_x, n_grid_y, n_grid_z), dim=4)

    u_z = df[:, :, :, :, 0] - n_grid[:, :, :, :, 0]
    u_x = df[:, :, :, :, 1] - n_grid[:, :, :, :, 1]
    u_y = df[:, :, :, :, 2] - n_grid[:, :, :, :, 2]
    u_x = u_x / 2 * (x_size - 1)
    u_y = u_y / 2 * (y_size - 1)
    u_z = u_z / 2 * (z_size - 1)
    new_df = torch.stack((u_x, u_y, u_z), dim=4)
    new_df = new_df.permute(0, 4, 1, 2, 3).to(device)
    return new_df

def warp_masks(tensors, displacement_fields, device="cpu"):
    return warp_tensors(tensors, displacement_fields, device, "nearest")

def resample_masks(tensors, new_size, device="cpu"):
    return resample_tensors(tensors, new_size, device, "nearest")

def unfold(tensor, patch_size, stride, device="cpu"):
    pad_x = math.ceil(tensor.size(3) / patch_size[1])*patch_size[1] - tensor.size(3)
    pad_y = math.ceil(tensor.size(2) / patch_size[0])*patch_size[0] - tensor.size(2)
    pad_z = math.ceil(tensor.size(4) / patch_size[2])*patch_size[2] - tensor.size(4)
    b_x, e_x = math.floor(pad_x / 2) + patch_size[0], math.ceil(pad_x / 2) + patch_size[0]
    b_y, e_y = math.floor(pad_y / 2) + patch_size[1], math.ceil(pad_y / 2) + patch_size[1]
    b_z, e_z = math.floor(pad_z / 2) + patch_size[2], math.ceil(pad_z / 2) + patch_size[2]
    new_tensor = F.pad(tensor, (b_z, e_z, b_x, e_x, b_y, e_y))
    padding_tuple = (b_y, b_x, b_z, e_y, e_x, e_z)
    padded_output_size = (new_tensor.size(2), new_tensor.size(3), new_tensor.size(4))
    new_tensor = new_tensor.unfold(2, patch_size[0], stride).unfold(3, patch_size[1], stride).unfold(4, patch_size[2], stride)
    new_tensor = new_tensor.reshape(new_tensor.size(0), tensor.size(1), new_tensor.size(2)*new_tensor.size(3)*new_tensor.size(4), patch_size[0], patch_size[1], patch_size[2])
    new_tensor = new_tensor.permute(0, 1, 3, 4, 5, 2)
    new_tensor = new_tensor[0].permute(4, 0, 1, 2, 3)
    return new_tensor, padded_output_size, padding_tuple

def fold(unfolded_tensor, padded_output_size, padding_tuple, patch_size, stride, device="cpu"):
    new_tensor = torch.zeros((1, unfolded_tensor.size(1),) + padded_output_size).to(device)
    col_y, col_x, col_z = int(padded_output_size[0] / stride - 1), int(padded_output_size[1] / stride - 1), int(padded_output_size[2] / stride - 1)
    for j in range(col_y):
        for i in range(col_x):
            for k in range(col_z):
                current_patch = unfolded_tensor[j*col_x*col_z + i*col_z + k, :, int(stride/2):-int(stride/2), int(stride/2):-int(stride/2), int(stride/2):-int(stride/2)]
                b_x = i*stride + int(stride/2)
                e_x = (i+1)*stride + int(stride/2)
                b_y = j*stride + int(stride/2)
                e_y = (j+1)*stride + int(stride/2)
                b_z = k*stride + int(stride/2)
                e_z = (k+1)*stride + int(stride/2)
                new_tensor[0, :, b_y:e_y, b_x:e_x, b_z:e_z] = current_patch
    if padding_tuple[3] == 0:
        new_tensor = new_tensor[:, :, padding_tuple[0]:, :, :]
    else:
        new_tensor = new_tensor[:, :, padding_tuple[0]:-padding_tuple[3], :, :]
    if padding_tuple[4] == 0:
        new_tensor = new_tensor[:, :, :, padding_tuple[1]:, :]
    else:
        new_tensor = new_tensor[:, :, :, padding_tuple[1]:-padding_tuple[4], :]
    if padding_tuple[5] == 0:
        new_tensor = new_tensor[:, :, :, :, padding_tuple[2]:]
    else:
        new_tensor = new_tensor[:, :, :, :, padding_tuple[2]:-padding_tuple[5]]
    return new_tensor

def build_pyramid(tensor, num_levels, device="cpu"):
    pyramid = []
    for i in range(num_levels):
        if i == num_levels - 1:
            pyramid.append(tensor)
        else:
            current_size = tensor.size()
            new_size = torch.Size((current_size[0], current_size[1], int(current_size[2]/(2**(num_levels-i-1))), int(current_size[3]/(2**(num_levels-i-1))), int(current_size[4]/(2**(num_levels-i-1)))))
            new_tensor = resample_tensors(tensor, new_size, device=device)
            pyramid.append(new_tensor)
    return pyramid

def build_mask_pyramid(tensor, num_levels, device="cpu"):
    pyramid = []
    for i in range(num_levels):
        if i == num_levels - 1:
            pyramid.append(tensor)
        else:
            current_size = tensor.size()
            new_size = torch.Size((current_size[0], current_size[1], int(current_size[2]/(2**(num_levels-i-1))), int(current_size[3]/(2**(num_levels-i-1))), int(current_size[4]/(2**(num_levels-i-1)))))
            new_tensor = resample_masks(tensor, new_size, device=device)
            pyramid.append(new_tensor)
    return pyramid

def upsample_displacement_fields(displacement_fields, new_size, device="cpu"):
    no_samples = new_size[0]
    old_x_size = displacement_fields.size(3)
    old_y_size = displacement_fields.size(2)
    old_z_size = displacement_fields.size(4)
    x_size = new_size[3]
    y_size = new_size[2]
    z_size = new_size[4]
    gz, gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size), torch.arange(z_size))
    gx = gx.type(torch.FloatTensor).to(device)
    gy = gy.type(torch.FloatTensor).to(device)
    gz = gz.type(torch.FloatTensor).to(device)
    grid_x = (gx / (z_size - 1) - 0.5)*2
    grid_y = (gy / (x_size - 1) - 0.5)*2
    grid_z = (gz / (y_size - 1) - 0.5)*2
    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1), grid_x.size(2))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1), grid_y.size(2))
    n_grid_z = grid_z.view(1, -1).repeat(no_samples, 1).view(-1, grid_z.size(0), grid_z.size(1), grid_z.size(2))
    n_grid = torch.stack((n_grid_x, n_grid_y, n_grid_z), dim=4)
    resampled_displacement_fields = F.grid_sample(displacement_fields, n_grid, mode='bilinear', padding_mode='zeros')
    resampled_displacement_fields[:, 0, :, :, :] *= x_size / old_x_size
    resampled_displacement_fields[:, 1, :, :, :] *= y_size / old_y_size
    resampled_displacement_fields[:, 2, :, :, :] *= z_size / old_z_size
    return resampled_displacement_fields

def compose_displacement_fields(u, v, device="cpu"):
    size = u.size()
    no_samples = size[0]
    y_size = size[2]
    x_size = size[3]
    z_size = size[4]
    gz, gy, gx = torch.meshgrid(torch.arange(y_size), torch.arange(x_size), torch.arange(z_size))
    gx = gx.type(torch.FloatTensor).to(device)
    gy = gy.type(torch.FloatTensor).to(device)
    gz = gz.type(torch.FloatTensor).to(device)
    grid_x = (gx / (z_size - 1) - 0.5)*2
    grid_y = (gy / (x_size - 1) - 0.5)*2
    grid_z = (gz / (y_size - 1) - 0.5)*2

    u_x_1 = u[:, 0, :, :].view(u.size(0), 1, u.size(2), u.size(3), u.size(4))
    u_y_1 = u[:, 1, :, :].view(u.size(0), 1, u.size(2), u.size(3), u.size(4))
    u_z_1 = u[:, 2, :, :].view(u.size(0), 1, u.size(2), u.size(3), u.size(4))
    u_x_2 = v[:, 0, :, :].view(v.size(0), 1, v.size(2), v.size(3), v.size(4))
    u_y_2 = v[:, 1, :, :].view(v.size(0), 1, v.size(2), v.size(3), v.size(4))
    u_z_2 = v[:, 2, :, :].view(v.size(0), 1, v.size(2), v.size(3), v.size(4))

    u_x_1 = u_x_1 / (x_size - 1) * 2
    u_y_1 = u_y_1 / (y_size - 1) * 2
    u_z_1 = u_z_1 / (z_size - 1) * 2
    u_x_2 = u_x_2 / (x_size - 1) * 2
    u_y_2 = u_y_2 / (y_size - 1) * 2
    u_z_2 = u_z_2 / (z_size - 1) * 2

    n_grid_x = grid_x.view(1, -1).repeat(no_samples, 1).view(-1, grid_x.size(0), grid_x.size(1), grid_x.size(2))
    n_grid_y = grid_y.view(1, -1).repeat(no_samples, 1).view(-1, grid_y.size(0), grid_y.size(1), grid_y.size(2))
    n_grid_z = grid_z.view(1, -1).repeat(no_samples, 1).view(-1, grid_z.size(0), grid_z.size(1), grid_z.size(2))
    n_grid = torch.stack((n_grid_x, n_grid_y, n_grid_z), dim=4)

    nv = torch.stack((u_x_2.view(u_x_2.size(0), u_x_2.size(2), u_x_2.size(3), u_x_2.size(4)), u_y_2.view(u_y_2.size(0), u_y_2.size(2), u_y_2.size(3), u_y_2.size(4)), u_z_2.view(u_z_2.size(0), u_z_2.size(2), u_z_2.size(3), u_z_2.size(4))), dim=4)
    t_x = n_grid_x.view(n_grid_x.size(0), 1, n_grid_x.size(1), n_grid_x.size(2), n_grid_x.size(3))
    t_y = n_grid_y.view(n_grid_y.size(0), 1, n_grid_y.size(1), n_grid_y.size(2), n_grid_y.size(3))
    t_z = n_grid_z.view(n_grid_z.size(0), 1, n_grid_z.size(1), n_grid_z.size(2), n_grid_z.size(3))
    added_x = u_x_1 + t_x
    added_y = u_y_1 + t_y
    added_z = u_z_1 + t_z

    added_grid = n_grid + nv
    i_u_x = F.grid_sample(added_x, added_grid, padding_mode='border')
    i_u_y = F.grid_sample(added_y, added_grid, padding_mode='border')
    i_u_z = F.grid_sample(added_z, added_grid, padding_mode='border')
    indexes = (added_grid[:, :, :, :, 0] >= 1.0) | (added_grid[:, :, :, :, 0] <= -1.0) | (added_grid[:, :, :, :, 1] >= 1.0) | (added_grid[:, :, :, :, 1] <= -1.0) | (added_grid[:, :, :, :, 2] >= 1.0) | (added_grid[:, :, :, :, 2] <= -1.0)
    indexes = indexes.view(indexes.size(0), 1, indexes.size(1), indexes.size(2), indexes.size(3))
    n_x = i_u_x - grid_x
    n_y = i_u_y - grid_y
    n_z = i_u_z - grid_z
    n_x[indexes] = 0.0
    n_y[indexes] = 0.0
    n_z[indexes] = 0.0
    n_x = n_x / 2 * (x_size - 1)
    n_y = n_y / 2 * (y_size - 1)
    n_z = n_z / 2 * (z_size - 1)
    return torch.cat((n_x, n_y, n_z), dim=1)


def load_networks_multiter(models_path, models_list, initial_model_name, num_levels, num_iters, learning_rate, scheduler_rates, mode="training", device="cpu"):
    models = list()
    if mode == "training":
        parameters = list()
        optimizers = list()
        schedulers = list()
    for i in range(num_levels):
        current_models = list()
        current_parameters = list()
        current_optimizers = list()
        current_schedulers = list()
        for j in range(num_iters):
            if initial_model_name is not None:
                current_models.append(models_list[i].load_network(device, path=os.path.join(models_path, initial_model_name + "_level_" + str(i+1) + "_iter_" + str(j+1))))
            else:
                current_models.append(models_list[i].load_network(device))
            if mode == "training":
                current_parameters.append(current_models[j].parameters())
                current_optimizers.append(optim.Adam(current_parameters[j], learning_rate))
                current_schedulers.append(optim.lr_scheduler.LambdaLR(current_optimizers[j], lambda epoch: scheduler_rates[i][j]**epoch))
        models.append(current_models)
        if mode == "training":
            parameters.append(current_parameters)
            optimizers.append(current_optimizers)
            schedulers.append(current_schedulers)
    if mode == "training":
        return models, parameters, optimizers, schedulers
    else:
        return models

def save_multilevel_network_multiter(models, model_name, models_path, num_levels, num_iters):
    for i in range(num_levels):
        for j in range(num_iters):
            torch.save(models[i][j].state_dict(), os.path.join(models_path, model_name + "_level_" + str(i+1) + "_iter_" + str(j+1)))   

def save_multilevel_network(models, model_name, models_path, num_levels):
    for i in range(num_levels):
        torch.save(models[i].state_dict(), os.path.join(models_path, model_name + "_level_" + str(i+1)))

def load_networks(models_path, models_list, initial_model_name, num_levels, learning_rate, scheduler_rates, mode="training", device="cpu"):
    models = list()
    if mode == "training":
        parameters = list()
        optimizers = list()
        schedulers = list()
    for i in range(num_levels):
        if initial_model_name is not None:
            models.append(models_list[i].load_network(device, path=os.path.join(models_path, initial_model_name + "_level_" + str(i+1))))
        else:
            models.append(models_list[i].load_network(device))
        if mode == "training":
            parameters.append(models[i].parameters())
            optimizers.append(optim.Adam(parameters[i], learning_rate))
            schedulers.append(optim.lr_scheduler.LambdaLR(optimizers[i], lambda epoch: scheduler_rates[i]**epoch))
    if mode == "training":
        return models, parameters, optimizers, schedulers
    else:
        return models

def jacobian_determinant(disp):
    # Code from the L2R challenge website
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
    
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)
    
    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)

    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
        
    return jacdet

def mask_tre(source_mask, target_mask):
    unique_values = np.unique(source_mask)
    tre = list()
    for i in range(1, len(unique_values)):
        cs, ct = (source_mask == i).reshape(source_mask.shape), (target_mask == i).reshape(target_mask.shape)
        xs, ys, zs = nd.center_of_mass(cs)
        xt, yt, zt = nd.center_of_mass(ct)
        c_tre = np.sqrt((xs-xt)**2 + (ys-yt)**2 + (zs-zt)**2)
        tre.append(c_tre)
    print("TRE: ", np.mean(tre))
    return np.mean(tre)