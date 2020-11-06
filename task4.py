import os
import time
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import math

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import dataloaders as dl
import cost_functions as cf
import utils
import paths

from networks import task_4 as task4


current_path = os.path.abspath(os.path.dirname(__file__))
data_path = paths.task_4_data_path
test_data_path = paths.test_task_4_data_path
models_path = os.path.join(current_path, "models")
figures_path = os.path.join(current_path, "figures")
device = "cuda:0"

def training(training_params):
    models_list = training_params['models_list']
    model_name = training_params['model_name']
    initial_model_name = training_params['initial_model_name']
    learning_rate = training_params['learning_rate'] 
    num_epochs = training_params['epochs']
    scheduler_rates = training_params['scheduler_rates']
    num_levels = training_params['num_levels']
    alphas = training_params['alphas']
    batch_size = training_params['batch_size']
    num_iters = training_params['num_iters']

    models, parameters, optimizers, schedulers = utils.load_networks(models_path, models_list, initial_model_name, num_iters, learning_rate,
        scheduler_rates, mode="training", device=device)

    training_dataset = dl.Task4Loader("training", validation_pairs=utils.get_task_4_pairs(), load_labels=True, exclude_all=True)
    training_dataloader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    validation_dataset = dl.Task4Loader("validation", validation_pairs=utils.get_task_4_pairs(), load_labels=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    training_size = len(training_dataloader.dataset)
    validation_size = len(validation_dataloader.dataset)
    print("Training size: ", training_size)
    print("Validation size: ", validation_size)

    # cost_function = cf.ncc_loss
    # cost_function_params = dict()
    cost_function = cf.mind_loss
    cost_function_params = dict()
    penalty_function = cf.dice_loss

    training_params['models'] = models
    training_params['parameters'] = parameters
    training_params['optimizers'] = optimizers
    training_params['schedulers'] = schedulers
    training_params['cost_function'] = cost_function
    training_params['cost_function_params'] = cost_function_params
    training_params['penalty_function'] = penalty_function

    cost_training_history = list()
    reg_training_history = list()
    penalty_training_history = list()
    cost_validation_history = list()
    reg_validation_history = list()
    penalty_validation_history = list()

    print_step = 10*batch_size
    for current_epoch in range(num_epochs):
        b_ce = time.time()
        print("Current epoch: ", str(current_epoch + 1) + "/" + str(num_epochs))
        # Training
        current_image = 0
        c_cost = 0.0
        c_reg = 0.0
        c_penalty = 0.0
        for targets, sources, targets_masks, sources_masks in training_dataloader:
            if not current_image % print_step:
                print("Training images: ", current_image + 1, "/", training_size)
                print_current_cost(c_cost, c_reg, c_penalty, current_image)
            current_image += batch_size
            source = sources.to(device)
            target = targets.to(device)
            source_mask = sources_masks.to(device)
            target_mask = targets_masks.to(device)
            displacement_field = register_training(source, target, source_mask, target_mask, training_params, device=device)
            t_cost, t_reg, t_penalty = calculate_metrics(source, target, displacement_field, training_params, source_mask=source_mask, target_mask=target_mask, device=device)
            c_cost = c_cost + (t_cost.item() * batch_size) ; c_reg = c_reg + (t_reg.item() * batch_size); c_penalty = c_penalty + (t_penalty.item() * batch_size)
        cost_training_history.append(c_cost / training_size); reg_training_history.append(c_reg / training_size); penalty_training_history.append(c_penalty / training_size)
        # Validation
        current_image = 0
        c_cost = 0.0
        c_reg = 0.0
        c_penalty = 0.0
        for targets, sources, targets_masks, sources_masks in validation_dataloader:
            if not current_image % print_step:
                print("Validation images: ", current_image + 1, "/", validation_size)
                print_current_cost(c_cost, c_reg, c_penalty, current_image)
            current_image += batch_size
            source = sources.to(device)
            target = targets.to(device)
            source_mask = sources_masks.to(device)
            target_mask = targets_masks.to(device)
            displacement_field = register(source, target, training_params, device=device)
            t_cost, t_reg, t_penalty = calculate_metrics(source, target, displacement_field, training_params, source_mask=source_mask, target_mask=target_mask, device=device)
            c_cost = c_cost + (t_cost.item() * batch_size) ; c_reg = c_reg + (t_reg.item() * batch_size); c_penalty = c_penalty + (t_penalty.item() * batch_size)
        cost_validation_history.append(c_cost / validation_size); reg_validation_history.append(c_reg / validation_size); penalty_validation_history.append(c_penalty / validation_size)

        for i in range(num_iters):
            schedulers[i].step()

        print("Current training cost: ", cost_training_history[current_epoch])
        print("Current training reg: ", reg_training_history[current_epoch])
        print("Current training penalty: ", penalty_training_history[current_epoch])
        
        print("Current validation cost: ", cost_validation_history[current_epoch])
        print("Current validation reg: ", reg_validation_history[current_epoch])
        print("Current validation penalty: ", penalty_validation_history[current_epoch])

        e_ce = time.time()
        print("Epoch time: ", e_ce - b_ce, "seconds.")
        print("Estimated time to end epochs: ", (e_ce - b_ce)*(num_epochs - current_epoch - 1), "seconds.")

    utils.save_multilevel_network(training_params['models'], model_name, models_path, num_iters)

    plt.figure()
    plt.plot(cost_training_history, color='red', linestyle="-")
    plt.plot(cost_validation_history, color='blue', linestyle="-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend(['Training', 'Validation'])
    plt.savefig(os.path.join(figures_path, model_name + "_cost.png"), bbox_inches = 'tight', pad_inches = 0)
    
    plt.figure()
    plt.plot(reg_training_history, color='red', linestyle="-")
    plt.plot(reg_validation_history, color='blue', linestyle="-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Reg")
    plt.legend(['Training', 'Validation'])
    plt.savefig(os.path.join(figures_path, model_name + "_reg.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.figure()
    plt.plot(penalty_training_history, color='red', linestyle="-")
    plt.plot(penalty_validation_history, color='blue', linestyle="-")
    plt.grid(True)
    plt.xlabel("Epoch")
    plt.ylabel("Penalty")
    plt.legend(['Training', 'Validation'])
    plt.savefig(os.path.join(figures_path, model_name + "_penalty.png"), bbox_inches = 'tight', pad_inches = 0)

    plt.show()

def visualization(params):
    models_list = params['models_list']
    model_name = params['model_name']
    num_iters = params['num_iters']

    models = utils.load_networks(models_path, models_list, model_name, num_iters, None, None, mode="validation", device=device)
    validation_dataset = dl.Task4Loader("validation", validation_pairs=utils.get_task_4_pairs(), load_labels=True)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = True, num_workers = 4)

    params['models'] = models
    for targets, sources, targets_masks, sources_masks in validation_dataloader:
        source = sources.to(device)
        target = targets.to(device)
        source_mask = sources_masks.to(device)
        target_mask = targets_masks.to(device)
        displacement_field = register(source, target, params, device=device)

        warped_source = utils.warp_tensors(source, displacement_field, device=device)
        warped_source_mask = utils.warp_masks(source_mask, displacement_field, device=device)

        print("Initial NCC: ", cf.ncc_loss(source, target, device=device))
        print("Initial MIND-SSC: ", cf.mind_loss(source, target, device=device))
        print("Registered NCC: ", cf.ncc_loss(warped_source, target, device=device))
        print("Registered MIND-SSC: ", cf.mind_loss(warped_source, target, device=device))
        print("Initial average Dice: ", cf.dice_loss(source_mask, target_mask, device=device))
        print("Registered average Dice: ", cf.dice_loss(warped_source_mask, target_mask, device=device))
        print("Curvature: ", cf.curvature_regularization(displacement_field, device=device))

        show_all(source, target, warped_source, source_mask, target_mask, warped_source_mask, displacement_field)

        plt.show()

def show_all(source, target, warped_source, source_mask, target_mask, warped_source_mask, displacement_field):
    source = source.detach().cpu().numpy()[0, 0, :, :, :]
    target = target.detach().cpu().numpy()[0, 0, :, :, :]
    warped_source = warped_source.detach().cpu().numpy()[0, 0, :, :, :]
    source_mask = source_mask.detach().cpu().numpy()[0, 0, :, :, :]
    target_mask = target_mask.detach().cpu().numpy()[0, 0, :, :, :]
    warped_source_mask = warped_source_mask.detach().cpu().numpy()[0, 0, :, :, :]
    u_x = displacement_field.detach().cpu()[0, 0, :, :, :]
    u_y = displacement_field.detach().cpu()[0, 0, :, :, :]
    u_z = displacement_field.detach().cpu()[0, 0, :, :, :]
    
    y_size, x_size, z_size = source.shape
    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source[int(y_size / 2), :, :], cmap='gray')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target[int(y_size / 2), :, :], cmap='gray')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source[int(y_size / 2), :, :], cmap='gray')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "im_y.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source[:, int(x_size / 2), :], cmap='gray')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target[:, int(x_size / 2), :], cmap='gray')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source[:, int(x_size / 2), :], cmap='gray')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "im_x.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source[:, :, int(z_size / 2)], cmap='gray')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target[:, :, int(z_size / 2)], cmap='gray')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source[:, :, int(z_size / 2)], cmap='gray')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "im_z.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source_mask[int(y_size / 2), :, :], cmap='jet')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target_mask[int(y_size / 2), :, :], cmap='jet')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source_mask[int(y_size / 2), :, :], cmap='jet')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "m_y.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source_mask[:, int(x_size / 2), :], cmap='jet')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target_mask[:, int(x_size / 2), :], cmap='jet')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source_mask[:, int(x_size / 2), :], cmap='jet')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "m_x.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

    plt.figure(dpi=300)
    plt.subplot(1, 3, 1)
    plt.imshow(source_mask[:, :, int(z_size / 2)], cmap='jet')
    plt.axis('off')
    plt.title("Source")
    plt.subplot(1, 3, 2)
    plt.imshow(target_mask[:, :, int(z_size / 2)], cmap='jet')
    plt.axis('off')
    plt.title("Target")
    plt.subplot(1, 3, 3)
    plt.imshow(warped_source_mask[:, :, int(z_size / 2)], cmap='jet')
    plt.axis('off')
    plt.title("Warped Source")
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(os.path.join(figures_path, "m_z.png"), transparent = True, bbox_inches = 'tight', pad_inches = 0)

def visualization_test(params):
    models_list = params['models_list']
    model_name = params['model_name']
    num_iters = params['num_iters']

    models = utils.load_networks(models_path, models_list, model_name, num_iters, None, None, mode="validation", device=device)
    validation_dataset = dl.Task4Loader("test", validation_pairs=utils.get_test_task_4_pairs(), load_labels=False)
    validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size = 1, shuffle = True, num_workers = 4)

    params['models'] = models
    for targets, sources in validation_dataloader:
        source = sources.to(device)
        target = targets.to(device)
        displacement_field = register(source, target, params, device=device)

        warped_source = utils.warp_tensors(source, displacement_field, device=device)

        print("Initial NCC: ", cf.ncc_loss(source, target, device=device))
        print("Initial MIND-SSC: ", cf.mind_loss(source, target, device=device))
        print("Registered NCC: ", cf.ncc_loss(warped_source, target, device=device))
        print("Registered MIND-SSC: ", cf.mind_loss(warped_source, target, device=device))
        print("Curvature: ", cf.curvature_regularization(displacement_field, device=device))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(source.detach().cpu().numpy()[0, 0, :, :, int(source.size(4)/2)], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Source")
        plt.subplot(1, 3, 2)
        plt.imshow(target.detach().cpu().numpy()[0, 0, :, :, int(target.size(4)/2)], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Target")
        plt.subplot(1, 3, 3)
        plt.imshow(warped_source.detach().cpu().numpy()[0, 0, :, :, int(warped_source.size(4)/2)], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title("Warped Source")

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(displacement_field.detach().cpu().numpy()[0, 0, :, :, int(source.size(4)/2)], cmap='gray')
        plt.axis('off')
        plt.title("Ux")
        plt.subplot(1, 3, 2)
        plt.imshow(displacement_field.detach().cpu().numpy()[0, 1, :, :, int(source.size(4)/2)], cmap='gray')
        plt.axis('off')
        plt.title("Uy")
        plt.subplot(1, 3, 3)
        plt.imshow(displacement_field.detach().cpu().numpy()[0, 2, :, :, int(source.size(4)/2)], cmap='gray')
        plt.axis('off')
        plt.title("Uz")

        plt.show()

def print_current_cost(cost, reg, penalty, num_images):
    try:
        print("Current cost: ", cost / num_images)
        print("Current reg: ", reg / num_images)
        print("Current penalty: ", penalty / num_images)
    except:
        pass

def register_training(source, target, source_mask, target_mask, params, device="cpu"):
    num_iters = params['num_iters']
    with torch.set_grad_enabled(False):
        num_levels = params['num_levels']
        source_pyramid, target_pyramid, source_mask_pyramid, target_mask_pyramid = create_pyramids(source, target, source_mask, target_mask, num_levels, device=device)
    for i in range(num_levels):
        with torch.set_grad_enabled(False):
            displacement_field = displacement_field if i != 0 else None
            displacement_field, current_source, current_target, current_source_mask, current_target_mask = prepare_next_level(source_pyramid, target_pyramid, displacement_field, i, params,
                source_mask_pyramid=source_mask_pyramid, target_mask_pyramid=target_mask_pyramid, device=device)
        for j in range(num_iters):
            if j == 0:
                velocity_field = network_pass_training(current_source, current_target, params['models'][j], j, params, source_mask=current_source_mask, target_mask=current_target_mask, device=device)
            else:
                temp_source = utils.warp_tensors(current_source, displacement_field, device=device)
                temp_source_mask = utils.warp_masks(current_source_mask, displacement_field, device=device)
                velocity_field = network_pass_training(temp_source, current_target, params['models'][j], j, params, source_mask=temp_source_mask, target_mask=current_target_mask, device=device)
            with torch.set_grad_enabled(False):
                displacement_field = utils.compose_displacement_fields(displacement_field, velocity_field, device=device)
    return displacement_field

def register(source, target, params, device="cpu"):
    num_iters = params['num_iters']
    with torch.set_grad_enabled(False):
        num_levels = params['num_levels']
        source_pyramid, target_pyramid = create_image_pyramids(source, target, num_levels, device=device)
        for i in range(num_levels):
            displacement_field = displacement_field if i != 0 else None
            displacement_field, current_source, current_target = prepare_next_level(source_pyramid, target_pyramid, displacement_field, i, params,
                source_mask_pyramid=None, target_mask_pyramid=None, device=device)
            for j in range(num_iters):
                if j == 0:
                    velocity_field = network_pass(current_source, current_target, params['models'][j], device=device)
                else:
                    temp_source = utils.warp_tensors(current_source, displacement_field, device=device)
                    velocity_field = network_pass(temp_source, current_target, params['models'][j], device=device)
                displacement_field = utils.compose_displacement_fields(displacement_field, velocity_field, device=device)
    return displacement_field

def network_pass_training(source, target, model, current_level, params, source_mask=None, target_mask=None, device="cpu"):
    optimizers = params['optimizers']
    cost_function = params['cost_function']
    cost_function_params = params['cost_function_params']
    regularization_function = params['regularization_function']
    penalty_function = cf.mse_loss
    alpha = params['alphas'][current_level]
    beta = params['betas'][current_level]
    optimizers[current_level].zero_grad()
    model.train()
    with torch.set_grad_enabled(True):
        displacement_field = model(source, target)
        warped_source = utils.warp_tensors(source, displacement_field, device=device)
        cost = cost_function(warped_source, target, device=device, **cost_function_params)
        loss = cost
        if regularization_function is not None:
            reg = alpha*regularization_function(displacement_field, device=device)
            loss = loss + reg
        if penalty_function is not None and source_mask is not None and target_mask is not None:
            warped_mask = utils.warp_tensors(source_mask, displacement_field, device=device)
            penalty = beta*penalty_function(warped_mask, target_mask, device=device)
            loss = loss + penalty
        loss.backward()
        optimizers[current_level].step()
    return displacement_field

def network_pass(source, target, model, device="cpu"):
    model.eval()
    with torch.set_grad_enabled(False):
        displacement_field = model(source, target)
    return displacement_field

def prepare_next_level(source_pyramid, target_pyramid, displacement_field, current_level, params, source_mask_pyramid = None, target_mask_pyramid = None, device="cpu"):
    i = current_level
    with torch.set_grad_enabled(False):
        if i == 0:
            displacement_field = torch.zeros(source_pyramid[i].size(0), 3, target_pyramid[i].size(2), target_pyramid[i].size(3), target_pyramid[i].size(4)).to(device)
            current_source = source_pyramid[i]
            current_target = target_pyramid[i]
            if source_mask_pyramid is not None and target_mask_pyramid is not None:
                current_source_mask = source_mask_pyramid[i]
                current_target_mask = target_mask_pyramid[i]
        else:
            displacement_field = utils.upsample_displacement_fields(displacement_field, target_pyramid[i].size(), device=device)
            current_source = utils.warp_tensors(source_pyramid[i], displacement_field, device=device)
            current_target = target_pyramid[i]
            if source_mask_pyramid is not None and target_mask_pyramid is not None:
                current_source_mask = utils.warp_masks(source_mask_pyramid[i], displacement_field, device=device)  
                current_target_mask = target_mask_pyramid[i]
    if source_mask_pyramid is not None and target_mask_pyramid is not None:           
        return displacement_field, current_source, current_target, current_source_mask, current_target_mask
    else:
        return displacement_field, current_source, current_target

def create_pyramids(source, target, source_mask, target_mask, num_levels, device="cpu"):
    source_pyramid = utils.build_pyramid(source, num_levels, device)
    target_pyramid = utils.build_pyramid(target, num_levels, device)
    source_mask_pyramid = utils.build_mask_pyramid(source_mask, num_levels, device)
    target_mask_pyramid = utils.build_mask_pyramid(target_mask, num_levels, device)
    return source_pyramid, target_pyramid, source_mask_pyramid, target_mask_pyramid

def create_image_pyramids(source, target, num_levels, device="cpu"):
    source_pyramid = utils.build_pyramid(source, num_levels, device)
    target_pyramid = utils.build_pyramid(target, num_levels, device)   
    return source_pyramid, target_pyramid

def calculate_metrics(source, target, displacement_field, params, source_mask=None, target_mask=None, device="cpu"):
    with torch.set_grad_enabled(False):
        warped_source = utils.warp_tensors(source, displacement_field, device=device)
        cost = params['cost_function'](warped_source, target, device=device, **params['cost_function_params'])
        reg = params['regularization_function'](displacement_field, device=device)
        if source_mask is not None and target_mask is not None:
            warped_source_mask = utils.warp_masks(source_mask, displacement_field, device=device)
            penalty = params['penalty_function'](warped_source_mask, target_mask, device=device)
    if source_mask is not None and target_mask is not None:
        return cost, reg, penalty
    else:
        return cost, reg

def mask_df(source):
    x_filter = source - nd.uniform_filter(source, size=(1, 5, 1))
    y_filter = source - nd.uniform_filter(source, size=(5, 1, 1))
    mask = np.logical_or(x_filter == 0, y_filter == 0)
    mask = np.logical_not(nd.binary_opening(mask))
    return mask

def nonrigid_registration(source, target, params, device="cpu"):
    source, target = utils.normalize(source), utils.normalize(target)
    _, source_masked = mask_df(target), mask_df(source)
    source, target = torch.from_numpy(source.astype(np.float32)), torch.from_numpy(target.astype(np.float32))
    source = source.to(device).view(-1, 1, source.size(0), source.size(1), source.size(2))
    target = target.to(device).view(-1, 1, target.size(0), target.size(1), target.size(2))
    displacement_field = register(source, target, params, device)
    displacement_field = displacement_field[0, :, :, :, :].detach().cpu().numpy()
    displacement_field[:, source_masked == 0] = 0
    return displacement_field

def run():
    training_params = dict()
    training_params['models_list'] = [task4, task4, task4]
    training_params['epochs'] = 30
    training_params['batch_size'] = 4
    training_params['scheduler_rates'] = [0.92, 0.92, 0.92]
    training_params['num_levels'] = 1
    training_params['alphas'] = [2.0, 2.6, 3.4]
    training_params['betas'] = [0.8, 0.8, 0.8]
    training_params['learning_rate'] = 0.002
    training_params['num_iters'] = 3
    training_params['regularization_function'] = cf.simple_regularization
    training_params['initial_model_name'] = None
    training_params['model_name'] = None # To specify
    training(training_params)

    visualization_params = dict()
    visualization_params['models_list'] = [task4, task4, task4]
    visualization_params['model_name'] = None # To specify
    visualization_params['num_levels'] = 1
    visualization_params['num_iters'] = 3
    visualization(visualization_params)



if __name__ == "__main__":
    run()