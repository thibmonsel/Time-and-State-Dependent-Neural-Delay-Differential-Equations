import functools
from copy import deepcopy
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
from torchlaplace.core import laplace_reconstruct
from torchlaplace.data_utils import basic_collate_fn

extrapolate = True
latent_dim = 2
hidden_units = 64
encode_obs_time = True
s_recon_terms = 33
patience = None
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model (encoder and Laplace representation func)
class ReverseGRUEncoder(nn.Module):
    # Encodes observed trajectory into latent vector
    def __init__(self, dimension_in, latent_dim, hidden_units, encode_obs_time=True):
        super(ReverseGRUEncoder, self).__init__()
        self.encode_obs_time = encode_obs_time
        if self.encode_obs_time:
            dimension_in += 1
        self.gru = nn.GRU(dimension_in, hidden_units, 2, batch_first=True)
        self.linear_out = nn.Linear(hidden_units, latent_dim)
        nn.init.xavier_uniform_(self.linear_out.weight)

    def forward(self, observed_data, observed_tp):
        trajs_to_encode = observed_data  # (batch_size, t_observed_dim, observed_dim)
        if self.encode_obs_time:
            trajs_to_encode = torch.cat(
                (
                    observed_data,
                    observed_tp.view(1, -1, 1).repeat(observed_data.shape[0], 1, 1),
                ),
                dim=2,
            )
        reversed_trajs_to_encode = torch.flip(trajs_to_encode, (1,))
        out, _ = self.gru(reversed_trajs_to_encode)
        return self.linear_out(out[:, -1, :])


class LaplaceRepresentationFunc(nn.Module):
    # SphereSurfaceModel : C^{b+k} -> C^{bxd} - In Riemann Sphere Co ords : b dim s reconstruction terms, k is latent encoding dimension, d is output dimension
    def __init__(self, s_dim, output_dim, latent_dim, hidden_units=64):
        super(LaplaceRepresentationFunc, self).__init__()
        self.s_dim = s_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.linear_tanh_stack = nn.Sequential(
            nn.Linear(s_dim * 2 + latent_dim, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, hidden_units),
            nn.Tanh(),
            nn.Linear(hidden_units, (s_dim) * 2 * output_dim),
        )

        for m in self.linear_tanh_stack.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

        phi_max = torch.pi / 2.0
        self.phi_scale = phi_max - -torch.pi / 2.0

    def forward(self, i):
        out = self.linear_tanh_stack(i.view(-1, self.s_dim * 2 + self.latent_dim)).view(
            -1, 2 * self.output_dim, self.s_dim
        )
        theta = nn.Tanh()(out[:, : self.output_dim, :]) * torch.pi  # From - pi to + pi
        phi = (
            nn.Tanh()(out[:, self.output_dim :, :]) * self.phi_scale / 2.0
            - torch.pi / 2.0
            + self.phi_scale / 2.0
        )  # Form -pi / 2 to + pi / 2
        return theta, phi


def create_history_array(a_samples, t_history, exp_amplitude=0.01):
    t_history = t_history.cpu().numpy()
    xs = np.linspace(0, 1.0, 100)
    fn_u0_x = lambda x: np.sin(np.pi * x)
    fn_u0_t = lambda u0_x, a_sample, t: a_sample * np.exp(-exp_amplitude*t) * u0_x
    u0 = fn_u0_x(xs)
    ys_history = np.empty((a_samples.shape[0],  t_history.shape[0], xs.shape[0]))
    for i, a_sample in enumerate(a_samples.cpu().numpy()) :
        fn_u0_partial = functools.partial(fn_u0_t, u0, a_sample)
        a_sample_ys_history = np.empty((t_history.shape[0], xs.shape[0]))
        for j, t in enumerate(t_history) :
            a_sample_ys_history[j] = fn_u0_partial(t) 
        ys_history[i] = a_sample_ys_history
    return ys_history        



def get_dataset(train_test_split_idx, name, device, noise_level=None):
    if name == "state_dependent":
        trajectories = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/ys.npy"))
            .float()
            .to(device)
        )
        train_idx = np.load(
            f"../data/" + str(name) + f"/train_indices_{train_test_split_idx}.npy"
        )
        test_idx = np.load(
            f"../data/" + str(name) + f"/test_indices_{train_test_split_idx}.npy"
        )

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]
        return train_trajectories, val_trajectories, test_trajectories

    if name == "time_dependent":
        if noise_level is None:
            raise ValueError("Noise level must be specified (0,2,5 or 10)")
        trajectories = (
            torch.Tensor(
                np.load(
                    f"../data/"
                    + str(name)
                    + "/ys_"
                    + str(int(noise_level))
                    + "_noise_level.npy"
                )
            )
            .float()
            .to(device)
        )
        train_idx = np.load(
            f"../data/" + str(name) + f"/train_indices_{train_test_split_idx}.npy"
        )
        test_idx = np.load(
            f"../data/" + str(name) + f"/test_indices_{train_test_split_idx}.npy"
        )

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]
        return train_trajectories, val_trajectories, test_trajectories

    if name == "diffusion_delay":
        trajectories = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/ys.npy"))
            .float()
            .to(device)
        )
        a_sample = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/a_sample.npy"))
            .float()
            .to(device)
        )
        train_idx = np.load(
            f"../data/" + str(name) + f"/train_indices_{train_test_split_idx}.npy"
        )
        test_idx = np.load(
            f"../data/" + str(name) + f"/test_indices_{train_test_split_idx}.npy"
        )

        train_trajectories = trajectories[train_idx]
        val_trajectories = trajectories[test_idx]
        test_trajectories = trajectories[test_idx]
        train_a_sample = a_sample[train_idx]
        val_a_sample = a_sample[test_idx]
        test_a_sample = a_sample[test_idx]
        return (
            train_trajectories,
            val_trajectories,
            test_trajectories,
            train_a_sample,
            val_a_sample,
            test_a_sample,
        )


def get_extrapolate_dataset(name, device):
    if name == "diffusion_delay":
        trajectories = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/ys_extrapolate.npy"))
            .float()
            .to(device)
        )
        a_sample = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/a_sample_extrapolate.npy"))
            .float()
            .to(device)
        )
        return (trajectories, a_sample)

    else:
        trajectories = (
            torch.Tensor(np.load(f"../data/" + str(name) + "/ys_extrapolate.npy"))
            .float()
            .to(device)
        )

        return trajectories


def get_other_history_fn_dataset(name, device):
    other_history_trajectories = (
        torch.Tensor(np.load("../data/" + str(name) + "/ys_other_history_test.npy"))
        .float()
        .to(device)
    )
    amplitude_other_history = (
        torch.Tensor(np.load("../data/" + str(name) + "/y0_other_history.npy"))
        .float()
        .to(device)
    )
    time_other_history = (
        torch.Tensor(np.load("../data/" + str(name) + "/ts_history.npy"))
        .float()
        .to(device)
    )
    return other_history_trajectories, amplitude_other_history, time_other_history


fifty_percent = False
noise_level = 0
name_dataset = "diffusion_delay"
max_delays = {"time_dependent": 3, "state_dependent": 1 / 2, "diffusion_delay": 1}
epochs_dict = {"time_dependent": 2000, "state_dependent": 1000, "diffusion_delay": 500}
lr_dict = {"time_dependent": 1e-3, "state_dependent": 1e-3, "diffusion_delay": 0.01}
batch_size_dic =  {"time_dependent": 256, "state_dependent": 256, "diffusion_delay": 128}
lr, epochs, batch_size = lr_dict[name_dataset], epochs_dict[name_dataset], batch_size_dic[name_dataset]
tse_loss, noisyless_tse_mse = [], []


for train_test_split_idx in range(1, 6):
    os.makedirs("../results/" + str(name_dataset) + f"/split_index_{train_test_split_idx}/", exist_ok=True )

    if name_dataset == "time_dependent" :
        if noise_level is None:
            raise ValueError("Noise level must be specified (0,2,5 or 10)")
        t = (
            torch.Tensor(
                np.load(
                    f"../data/"
                    + str(name_dataset)
                    + "/ts_"
                    + str(int(noise_level))
                    + "_noise_level.npy"
                )
            )
            .float()
            .to(device)
        )

    else:
        t = (
            torch.Tensor(np.load(f"../data/" + str(name_dataset) + "/ts.npy"))
            .float()
            .to(device)
        )

    t_history = torch.arange(-max_delays[name_dataset], 0.0, (t[1] - t[0]))
    t_history = t_history.to(device)
    t = torch.cat([t_history, t]) + max_delays[name_dataset]

    ###### CREATING TRAIN TEST DATASET ######
    if name_dataset == "diffusion_delay":
        (
            train_trajectories,
            val_trajectories,
            test_trajectories,
            train_a_sample,
            val_a_sample,
            test_a_sample,
        ) = get_dataset(train_test_split_idx, name_dataset, device, noise_level)
    else:
        train_trajectories, val_trajectories, test_trajectories = get_dataset(
            train_test_split_idx, name_dataset, device, noise_level
        )

    if name_dataset == "diffusion_delay":
        train_history = create_history_array(train_a_sample, t_history)
        train_history = torch.Tensor(train_history).float().to(device)
        train_trajectories = torch.cat([train_history, train_trajectories], dim=1)

        val_history = create_history_array(val_a_sample, t_history)
        val_history = torch.Tensor(val_history).float().to(device)
        val_trajectories = torch.cat([val_history, val_trajectories], dim=1)

        test_history = create_history_array(test_a_sample, t_history)
        test_history = torch.Tensor(test_history).float().to(device)
        test_trajectories = torch.cat([test_history, test_trajectories], dim=1)

    else:
        train_history = torch.ones(
            train_trajectories.shape[0],
            t_history.shape[0],
            train_trajectories.shape[-1],
        ).to(device)
        train_history = torch.einsum(
            "ijl, il -> ijl", train_history, train_trajectories[:, 0]
        ).to(device)
        train_trajectories = torch.cat([train_history, train_trajectories], dim=1)

        val_history = torch.ones(
            val_trajectories.shape[0], t_history.shape[0], val_trajectories.shape[-1]
        ).to(device)
        val_history = torch.einsum(
            "ijl, il -> ijl", val_history, val_trajectories[:, 0]
        )
        val_trajectories = torch.cat([val_history, val_trajectories], dim=1)

        test_history = torch.ones(
            test_trajectories.shape[0], t_history.shape[0], test_trajectories.shape[-1]
        ).to(device)
        test_history = torch.einsum(
            "ijl, il -> ijl", test_history, test_trajectories[:, 0]
        ).to(device)
        test_trajectories = torch.cat([test_history, test_trajectories], dim=1)

    ###### CREATING EXTRAPOLATE DATASET ######
    print("Loading extrapolate dataset")
    if name_dataset == "diffusion_delay":
        extrapolate_test_trajectories, extrapolate_a_sample = get_extrapolate_dataset(
            name_dataset, device
        )
        extrapolate_test_history = create_history_array(extrapolate_a_sample, t_history)
        extrapolate_test_history = torch.Tensor(test_history).float().to(device)
        extrapolate_test_trajectories = torch.cat(
            [extrapolate_test_history, extrapolate_test_trajectories], dim=1
        )
    else:
        extrapolate_test_trajectories = get_extrapolate_dataset(name_dataset, device)

        extrapolate_test_history = torch.ones(
            extrapolate_test_trajectories.shape[0],
            t_history.shape[0],
            extrapolate_test_trajectories.shape[-1],
        ).to(device)
        extrapolate_test_history = torch.einsum(
            "ijl, il -> ijl",
            extrapolate_test_history,
            extrapolate_test_trajectories[:, 0],
        )
        extrapolate_test_trajectories = torch.cat(
            [extrapolate_test_history, extrapolate_test_trajectories], dim=1
        )

    ###### CREATING OTHER HISTORY STEP FN DATASET ######
    ###### ONLY for Time and State Dependent ######
    if not (name_dataset == "diffusion_delay"):
        (
            other_history_trajectories,
            amplitude_other_history,
            time_other_history,
        ) = get_other_history_fn_dataset(name_dataset, device)

        def h(t, time_other_history, amplitude_other_history):
            return jnp.array(
                jax.lax.cond(
                    bool(t > time_other_history[0]),
                    lambda: amplitude_other_history[0],
                    lambda: amplitude_other_history[1],
                )
            )

        batch_t_history = torch.unsqueeze(t_history, 0).expand(
            other_history_trajectories.shape[0], -1
        )
        masked_t_history = batch_t_history > time_other_history

        first_part_mask_repeat = torch.repeat_interleave(
            torch.unsqueeze(masked_t_history, -1),
            extrapolate_test_trajectories.shape[-1],
            dim=2,
        )
        second_part_mask_repeat = torch.repeat_interleave(
            torch.unsqueeze(~masked_t_history, -1),
            extrapolate_test_trajectories.shape[-1],
            dim=2,
        )
        if len(amplitude_other_history.size()) > 2:
            first_part_step = torch.einsum(
                "btf, bf -> btf", first_part_mask_repeat, amplitude_other_history[:, 0]
            )
            second_part_step = torch.einsum(
                "btf, bf -> btf", second_part_mask_repeat, amplitude_other_history[:, 1]
            )
        else:
            first_part_step = torch.einsum(
                "btf, bf -> btf",
                first_part_mask_repeat,
                torch.unsqueeze(amplitude_other_history[:, 0], 1),
            )
            second_part_step = torch.einsum(
                "btf, bf -> btf",
                second_part_mask_repeat,
                torch.unsqueeze(amplitude_other_history[:, 1], 1),
            )

        step_function = first_part_step + second_part_step

        other_history_trajectories = torch.cat(
            [step_function, other_history_trajectories], dim=1
        )

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
            extrap=extrapolate,
            history_nb_time_step= int(t.shape[0]//2) if fifty_percent else t_history.shape[0] ,
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
            extrap=extrapolate,
            history_nb_time_step=int(t.shape[0]//2) if fifty_percent else t_history.shape[0] 
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
            extrap=extrapolate,
            history_nb_time_step= int(t.shape[0]//2) if fifty_percent else t_history.shape[0] 
        ),
    )

    dltest2 = DataLoader(
        extrapolate_test_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrapolate,
            history_nb_time_step=int(t.shape[0]//2) if name_dataset== "time_dependent_50_percent" else t_history.shape[0] 
        ),
    )
    if name_dataset == "diffusion_delay":
        pass
    else:
        dltest3 = DataLoader(
            other_history_trajectories,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: basic_collate_fn(
                batch,
                t,
                data_type="test",
                extrap=extrapolate,
                history_nb_time_step=t_history.shape[0],
            ),
        )

    # for data in dltrain :
    #     print(data["tp_to_predict"].shape,data["observed_tp"].shape, data["data_to_predict"].shape )
    #     plt.plot(data["observed_tp"].flatten().cpu(), data["observed_data"][1].cpu())
    #     plt.plot(data["tp_to_predict"].flatten().cpu(), data["data_to_predict"][1].cpu(), '--' )
    #     plt.show()

    encoder = ReverseGRUEncoder(
        input_dim,
        latent_dim,
        hidden_units // 2,
        encode_obs_time=encode_obs_time,
    ).to(device)
    laplace_rep_func = LaplaceRepresentationFunc(
        s_recon_terms, output_dim, latent_dim
    ).to(device)

    if not patience:
        patience = epochs

    params = list(laplace_rep_func.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    waiting = 0

    loss_per_step = []
    for epoch in range(epochs):
        iteration = 0
        epoch_train_loss_it_cum = 0
        start_time = time()
        laplace_rep_func.train(), encoder.train()
        for batch in dltrain:
            optimizer.zero_grad()
            trajs_to_encode = batch[
                "observed_data"
            ]  # (batch_size, t_observed_dim, observed_dim)
            observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
            p = encoder(
                trajs_to_encode, observed_tp
            )  # p is the latent tensor encoding the initial states
            tp_to_predict = batch["tp_to_predict"]
            predictions = laplace_reconstruct(
                laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
            )
            loss = loss_fn(
                torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
            )
            loss_per_step.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1)
            optimizer.step()
            epoch_train_loss_it_cum += loss.item()
            iteration += 1
        epoch_train_loss = epoch_train_loss_it_cum / iteration
        epoch_duration = time() - start_time

        # Validation step
        laplace_rep_func.eval(), encoder.eval()
        cum_val_loss = 0
        cum_val_batches = 0
        for batch in dlval:
            trajs_to_encode = batch[
                "observed_data"
            ]  # (batch_size, t_observed_dim, observed_dim)
            observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
            p = encoder(
                trajs_to_encode, observed_tp
            )  # p is the latent tensor encoding the initial states
            tp_to_predict = batch["tp_to_predict"]
            predictions = laplace_reconstruct(
                laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
            )
            cum_val_loss += loss_fn(
                torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
            ).item()
            cum_val_batches += 1
        val_mse = cum_val_loss / cum_val_batches
        if epoch % 100 == 0:
            print(
                "[epoch={}] epoch_duration={:.2f} | train_loss={}\t| val_mse={}\t|".format(
                    epoch, epoch_duration, epoch_train_loss, val_mse
                )
            )
            if name_dataset == "diffusion_delay":
                plt.subplot(1,2,1)
                plt.imshow(batch["data_to_predict"].detach().cpu()[0])
                plt.colorbar()
                plt.subplot(1,2,2)
                plt.imshow(predictions.detach().cpu()[0])
                plt.colorbar()
                plt.title("Neural Laplace Training Phase")
                plt.savefig(
                    f"../results/"
                    + str(name_dataset)
                    + f"/split_index_{train_test_split_idx}/training_phase_epoch_{epoch}.png"
                )
                plt.close()
            else : 
                plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
                plt.plot(
                    tp_to_predict.detach().cpu()[0],
                    batch["data_to_predict"].detach().cpu()[0],
                    "--",
                )
                plt.plot(
                    tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0], c="r"
                )
                plt.title("Neural Laplace Training Phase")
                plt.savefig(
                    f"../results/"
                    + str(name_dataset)
                    + f"/split_index_{train_test_split_idx}/training_phase_epoch_{epoch}.png"
                )
                plt.close()

        # Early stopping procedure
        if val_mse < best_loss:
            best_loss = val_mse
            best_laplace_rep_func = deepcopy(laplace_rep_func.state_dict())
            best_encoder = deepcopy(encoder.state_dict())
            waiting = 0
        elif waiting > patience:
            break
        else:
            waiting += 1
    np.save(
        f"../results/" + str(name_dataset) + f"/split_index_{train_test_split_idx}/loss_array.npy",
        np.array(loss_per_step),
    )

    # Test step
    laplace_rep_func.eval(), encoder.eval()
    cum_test_loss = 0
    cum_test_batches = 0
    for batch in dltest:
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(
            laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
        )

        cum_test_loss += loss_fn(
            torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
        ).item()
        if name_dataset == "diffusion_delay":
            plt.subplot(1,2,1)
            plt.imshow(batch["data_to_predict"].detach().cpu()[0])
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(predictions.detach().cpu()[0])
            plt.colorbar()
            plt.title("Neural Laplace Testing")
            plt.savefig(
                f"../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/testing_phase.png"
            )
            plt.close()
        else : 
            plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
            plt.plot(
                tp_to_predict.detach().cpu()[0],
                batch["data_to_predict"].detach().cpu()[0],
                "--",
            )
            plt.plot(tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0])
            plt.savefig(
                f"../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/testing_phase.png"
            )
            plt.close()
        
        if name_dataset == "time_dependent":
            np.save(
                f"../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/ys_" + str(int(noise_level)) + "_noise_level.npy",
                predictions.cpu().detach().numpy(),
            )
        else : 
            np.save(
                f"../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/ys_test_pred.npy",
                predictions.cpu().detach().numpy(),
            )
        cum_test_batches += 1
    test_mse = cum_test_loss / cum_test_batches
    print(f"test_mse= {test_mse}")
    tse_loss.append(test_mse)

    for batch in dltest2:
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(
            laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
        )
        cum_test_loss += loss_fn(
            torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
        ).item()
        
        if name_dataset == "diffusion_delay":
            plt.subplot(1,2,1)
            plt.imshow(batch["data_to_predict"].detach().cpu()[0])
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(predictions.detach().cpu()[0])
            plt.colorbar()
            plt.title("Neural Laplace Testing")
            plt.savefig(
                "../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/extrapolate_testing_phase.png"
            )
            plt.close()
        else : 
            plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
            plt.plot(
                tp_to_predict.detach().cpu()[0],
                batch["data_to_predict"].detach().cpu()[0],
                "--",
            )
            plt.plot(tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0])
            plt.savefig(
                "../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/extrapolate_testing_phase.png"
            )
            plt.close()
        # predictions = torch.cat([trajs_to_encode, predictions], axis=1)
        np.save(
            "../results/"
            + str(name_dataset)
            + f"/split_index_{train_test_split_idx}/ys_extrapolate_pred.npy",
            predictions.cpu().detach().numpy(),
        )

    if name_dataset == "diffusion_delay":
        pass
    else:
        for batch in dltest3:
            trajs_to_encode = batch[
                "observed_data"
            ]  # (batch_size, t_observed_dim, observed_dim)
            observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
            p = encoder(
                trajs_to_encode, observed_tp
            )  # p is the latent tensor encoding the initial states
            tp_to_predict = batch["tp_to_predict"]
            predictions = laplace_reconstruct(
                laplace_rep_func, p, tp_to_predict, recon_dim=output_dim
            )

            cum_test_loss += loss_fn(
                torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
            ).item()
            plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
            plt.plot(
                tp_to_predict.detach().cpu()[0],
                batch["data_to_predict"].detach().cpu()[0],
                "--",
            )
            plt.plot(tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0])
            plt.savefig(
                "../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/new_history_testing_phase.png"
            )
            plt.close()
            
            np.save(
                f"../results/"
                + str(name_dataset)
                + f"/split_index_{train_test_split_idx}/ys_new_history_pred.npy",
                predictions.cpu().detach().numpy(),
            )

print(f"MSE LOSS TESS LAPLACE : {np.mean(tse_loss)} +/-  {np.std(tse_loss)}")