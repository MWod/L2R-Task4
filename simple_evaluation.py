import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from scipy import spatial
from scipy.ndimage.interpolation import map_coordinates, zoom

import paths
import utils

from surface_distance import *

data_path = paths.data_path
submission_path = os.path.join(data_path, "Submissions", "TEST", "submission")

task_1_val = os.path.join(data_path, "Task1", "pairs_val.csv")
task_2_val = os.path.join(data_path, "Task2", "pairs_val.csv")
task_3_val = os.path.join(data_path, "Task3", "pairs_val.csv")
task_4_val = os.path.join(data_path, "Task4", "pairs_val.csv")


def jacobian_determinant(disp):
    # Code from L2R challenge website
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

def warp_mask(mask, df):
    D, H, W = mask.shape
    identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    warped_mask = scipy.ndimage.map_coordinates(mask, identity + df, order=0)
    return warped_mask

def lowest_30(input):
    sorted_input = sorted(input)
    total_len = len(sorted_input)
    ratio = 0.3
    worst_values = sorted_input[:int(ratio*total_len)]
    return np.array(worst_values) 

def dice(m1, m2):
    loss = 0
    unique_values = np.unique(m1)
    for i in range(1, len(unique_values)):
        c_m1 = m1 == unique_values[i]
        c_m2 = m2 == unique_values[i]
        c_dice = compute_dice_coefficient(c_m1, c_m2)
        loss += c_dice
    loss = loss / (len(unique_values) - 1)
    return loss

def hausdorff(m1, m2):
    loss = 0
    unique_values = np.unique(m1)
    for i in range(1, len(unique_values)):
        c_m1 = m1 == unique_values[i]
        c_m2 = m2 == unique_values[i]
        c_loss = compute_robust_hausdorff(compute_surface_distances(c_m1, c_m2, np.ones(3)), 95.)
        if np.isinf(c_loss):
            loss += 20
        else:
            loss += c_loss
    loss = loss / (len(unique_values) - 1)
    return loss

def evaluate_task_4():
    print("Task 4 Evaluation.")

    initial_dices = []
    warped_dices = []
    initial_hausdorff = []
    warped_hausdorff = []
    jacobians = []

    task_4_df, task_4_samples = utils.load_dataframe(task_4_val)
    for i in range(task_4_samples):
        df_path = utils.build_df_path(task_4_df, i)
        df_path = os.path.join(submission_path, "task_04", df_path)
        df = utils.load_df(df_path)

        target_mask, source_mask = utils.load_task_4_labels(task_4_df, i)
        warped_source_mask = warp_mask(source_mask, df)

        c_dice = dice(source_mask, target_mask)
        w_dice = dice(warped_source_mask, target_mask)
        initial_dices.append(c_dice)
        warped_dices.append(w_dice)

        c_hausdorff = hausdorff(source_mask, target_mask)
        w_hausdorff = hausdorff(warped_source_mask, target_mask)
        initial_hausdorff.append(c_hausdorff)
        warped_hausdorff.append(w_hausdorff)

        jac_det = (jacobian_determinant(df[np.newaxis, :, :, :, :]) + 3).clip(0.000000001, 1000000000)
        log_jac_det = np.log(jac_det)
        jacobians.append(log_jac_det)

    print("Mean log jac det: ", np.mean(jacobians))
    print("Std log jac det: ", np.std(jacobians))

    print("Mean initial dice: ", np.mean(initial_dices))
    print("Std initial dice ", np.std(initial_dices))

    print("Mean warped dice: ", np.mean(warped_dices))
    print("Std warped dice ", np.std(warped_dices))

    print("Initial DSC30: ", np.mean(lowest_30(initial_dices)))
    print("Warped DSC30: ", np.mean(lowest_30(warped_dices)))

    print("Initial HD: ", np.mean(initial_hausdorff), "+-", np.std(initial_hausdorff))
    print("Warped HD: ", np.mean(warped_hausdorff), "+-", np.std(warped_hausdorff))




def run():
    evaluate_task_4()







if __name__ == "__main__":
    run()