###########################
# Neural Laplace: Learning diverse classes of differential equations in the Laplace domain
# Author: Samuel Holt
###########################
import shelve
from functools import partial
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch
# from ddeint import ddeint
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchlaplace.data_utils import basic_collate_fn

local_path = Path(__file__).parent

def generate_data_set(
    seed,
    name,
    device,
    double=False,
    batch_size=256,
    extrap=0,
    trajectories_to_sample=256,
    percent_missing_at_random=0.0,
    normalize=True,
    test_set_out_of_distribution=False,
    noise_std=None,
    t_nsamples=200,
    observe_step=1,
    predict_step=1,
    noise_level = None
):
    if name == "state_dependent":
        t = torch.Tensor(np.load(f"../data/" + str(name) + "/ts.npy")).double().to(device)
        trajectories = torch.Tensor(np.load(f"../data/" + str(name) + "/ys.npy")).double().to(device)
        train_idx = np.load(f"../data/" + str(name) + f"/train_indices_{seed}.npy")
        test_idx = np.load(f"../data/" + str(name) + f"/test_indices_{seed}.npy")

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]

    if name == "time_dependent":
        if noise_level is None :
            raise ValueError("Noise level must be specified (0,2,5 or 10)")
        t = torch.Tensor(np.load(f"../data/" + str(name) + "" + "/ts_" + str(int(noise_level)) + "_noise_level.npy")).double().to(device)
        trajectories = torch.Tensor(np.load(f"../data/" + str(name) + "/ys_" + str(int(noise_level)) + "_noise_level.npy")).double().to(device)
        train_idx = np.load(f"../data/" + str(name) + f"/train_indices_{seed}.npy")
        test_idx = np.load(f"../data/" + str(name) + f"/test_indices_{seed}.npy")

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]

    if name == "diffusion_delay":
        t = torch.Tensor(np.load(f"../data/" + str(name) + "/ts.npy")).double().to(device)
        trajectories = torch.Tensor(np.load(f"../data/" + str(name) + "/ys.npy")).double().to(device)
        train_idx = np.load(f"../data/" + str(name) + f"/train_indices_{seed}.npy")
        test_idx = np.load(f"../data/" + str(name) + f"/test_indices_{seed}.npy")

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]

    if not extrap:
        bool_mask = torch.FloatTensor(*trajectories.shape).uniform_() < (
            1.0 - percent_missing_at_random
        )
        if double:
            float_mask = (bool_mask).float().double().to(device)
        else:
            float_mask = (bool_mask).float().to(device)
        trajectories = float_mask * trajectories

    print("originial t", t.shape)
    print("before normalize", trajectories.shape)
    if normalize:
        samples = trajectories.shape[0]
        dim = trajectories.shape[2]
        traj = (
            torch.reshape(trajectories, (-1, dim))
            - torch.reshape(trajectories, (-1, dim)).mean(0)
        ) / torch.reshape(trajectories, (-1, dim)).std(0)
        trajectories = torch.reshape(traj, (samples, -1, dim))
    print("normalize", trajectories.shape)
    if noise_std:
        trajectories += torch.randn(trajectories.shape).to(device) * noise_std

   
    # name_dataset = "lk"
    # t = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+ str(name_dataset)+ "/run_seed_{seed}/ts.npy")).float().to(device)
    # train_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+ str(name_dataset)+ "/run_seed_{seed}/ys.npy")).float().to(device)
    # val_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+ str(name_dataset)+ "/run_seed_{seed}/ys_test.npy")).float().to(device)
    # test_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+ str(name_dataset)+ "/run_seed_{seed}/ys_test.npy")).float().to(device)

    print("initial size trajectory", train_trajectories.shape, t.shape)    

    # import matplotlib.pyplot as plt
    test_plot_traj = test_trajectories[0, :, :]
    # plt.plot(t.cpu(), test_plot_traj.cpu())
    # plt.show()
    input_dim = train_trajectories.shape[2]
    output_dim = input_dim

    dltrain = DataLoader(
        train_trajectories,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="train",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    dlval = DataLoader(
        val_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    dltest = DataLoader(
        test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrap,
            observe_step=observe_step,
            predict_step=predict_step,
        ),
    )
    return (
        input_dim,
        output_dim,
        dltrain,
        dlval,
        dltest,
        test_plot_traj,
        t,
        test_trajectories,
    )
