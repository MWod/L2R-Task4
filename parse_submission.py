import os

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as plt

import utils
import paths

from networks import task_4 as task4
import task4 as t4_3



data_path = paths.data_path
models_path = paths.models_path
# output_path = os.path.join(data_path, "Submissions", "TEST", "submission")
output_path = os.path.join(data_path, "Submissions", "FINAL", "submission")

task_1_val = os.path.join(data_path, "Task1", "pairs_val.csv")
task_2_val = os.path.join(data_path, "Task2", "pairs_val.csv")
task_3_val = os.path.join(data_path, "Task3", "pairs_val.csv")
task_4_val = os.path.join(data_path, "Task4", "pairs_val.csv")

test_task_1_val = os.path.join(data_path, "Test", "Task1", "pairs_val.csv")
test_task_2_val = os.path.join(data_path, "Test", "Task2", "pairs_val.csv")
test_task_3_val = os.path.join(data_path, "Test", "Task3", "pairs_val.csv")
test_task_4_val = os.path.join(data_path, "Test", "Task4", "pairs_val.csv")

task_1_expected_shape = (3, 256, 256, 288)
task_2_expected_shape = (3, 192, 192, 208)
task_3_expected_shape = (3, 192, 160, 256)
task_4_expected_shape = (3, 64, 64, 64)


def create_empty_submission():
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
        os.makedirs(os.path.join(output_path, "task_01"))
        os.makedirs(os.path.join(output_path, "task_02"))
        os.makedirs(os.path.join(output_path, "task_03"))
        os.makedirs(os.path.join(output_path, "task_04"))

def fill_with_zeros():
    task_1_df, task_1_samples = utils.load_dataframe(task_1_val)
    task_2_df, task_2_samples = utils.load_dataframe(task_2_val)
    task_3_df, task_3_samples = utils.load_dataframe(task_3_val)
    task_4_df, task_4_samples = utils.load_dataframe(task_4_val)

    print("Task 1.")
    for i in range(task_1_samples):
        df_path = utils.build_df_path(task_1_df, i)
        df = utils.create_empty_df(task_1_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_01", df_path))

    print("Task 2.")
    for i in range(task_2_samples):
        df_path = utils.build_df_path(task_2_df, i)
        df = utils.create_empty_df(task_2_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_02", df_path))

    print("Task 3.")
    for i in range(task_3_samples):
        df_path = utils.build_df_path(task_3_df, i)
        df = utils.create_empty_df(task_3_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_03", df_path))

    print("Task 4.")
    for i in range(task_4_samples):
        df_path = utils.build_df_path(task_4_df, i)
        df = utils.create_empty_df(task_4_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_04", df_path))

def fill_with_zeros_test():
    task_1_df, task_1_samples = utils.load_dataframe(test_task_1_val)
    task_2_df, task_2_samples = utils.load_dataframe(test_task_2_val)
    task_3_df, task_3_samples = utils.load_dataframe(test_task_3_val)
    task_4_df, task_4_samples = utils.load_dataframe(test_task_4_val)

    print("Task 1.")
    for i in range(task_1_samples):
        df_path = utils.build_df_path(task_1_df, i)
        df = utils.create_empty_df(task_1_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_01", df_path))

    print("Task 2.")
    for i in range(task_2_samples):
        df_path = utils.build_df_path(task_2_df, i)
        df = utils.create_empty_df(task_2_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_02", df_path))

    print("Task 3.")
    for i in range(task_3_samples):
        df_path = utils.build_df_path(task_3_df, i)
        df = utils.create_empty_df(task_3_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_03", df_path))

    print("Task 4.")
    for i in range(task_4_samples):
        df_path = utils.build_df_path(task_4_df, i)
        df = utils.create_empty_df(task_4_expected_shape)
        utils.save_df(df, os.path.join(output_path, "task_04", df_path))

def create_task_4_submission():
    device = "cuda:0"
    num_levels = 1
    task_4_df, task_4_samples = utils.load_dataframe(task_4_val)

    models_list = [task4, task4, task4]
    model_name = None # To specify
    models = utils.load_networks(models_path, models_list, model_name, len(models_list), None, None, mode="validation", device=device)
    params = dict()
    params['device'] = device
    params['models_list'] = models_list
    params['num_levels'] = num_levels
    params['models'] = models
    params['num_iters'] = 3
    print("Task 4 Start.")
    for i in range(task_4_samples):
        target, source = utils.load_task_4_pair(task_4_df, i)
        df = t4_3.nonrigid_registration(source, target, params, device=device)
        copy_df = df.copy()
        df[0, :, :, :], df[1, :, :, :] = copy_df[1, :, :, :], copy_df[0, :, :, :]
        df_path = utils.build_df_path(task_4_df, i)
        utils.save_df(df, os.path.join(output_path, "task_04", df_path))
    print("Task 4 End.")

def create_task_4_test_submission():
    device = "cuda:0"
    num_levels = 1
    task_4_df, task_4_samples = utils.load_dataframe(test_task_4_val)

    models_list = [task4, task4, task4]
    model_name = None # To specify
    models = utils.load_networks(models_path, models_list, model_name, len(models_list), None, None, mode="validation", device=device)
    params = dict()
    params['device'] = device
    params['models_list'] = models_list
    params['num_levels'] = num_levels
    params['models'] = models
    params['num_iters'] = 3
    print("Task 4 Start.")
    for i in range(task_4_samples):
        target, source = utils.load_test_task_4_pair(task_4_df, i)
        df = t4_3.nonrigid_registration(source, target, params, device=device)
        copy_df = df.copy()
        df[0, :, :, :], df[1, :, :, :] = copy_df[1, :, :, :], copy_df[0, :, :, :]
        df_path = utils.build_df_path(task_4_df, i)
        utils.save_df(df, os.path.join(output_path, "task_04", df_path))
    print("Task 4 End.")

def warp_image(mask, df):
    D, H, W = mask.shape
    identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    warped_mask = nd.map_coordinates(mask, identity + df, order=1)
    return warped_mask


def run():
    # create_empty_submission()
    # fill_with_zeros_test()
    # create_task_4_test_submission()
    pass


if __name__ == "__main__":
    run()