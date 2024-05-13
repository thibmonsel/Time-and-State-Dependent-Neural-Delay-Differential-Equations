import functools
import os
from copy import deepcopy
from time import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from torchlaplace.core import laplace_reconstruct
from torchlaplace.data_utils import basic_collate_fn

normalize_dataset = True
batch_size = 128 
extrapolate = True
latent_dim = 2
hidden_units = 64
encode_obs_time = True
s_recon_terms = 33
learning_rate = 1e-3
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


def create_history_array(a_samples, t_history):
    t_history = t_history.cpu().numpy()
    xs = np.linspace(0, 1.0, 100)
    fn_u0_x = lambda x: np.sin(np.pi * x)
    fn_u0_t = lambda u0_x, a_sample, t: a_sample * np.exp(-0.01*t) * u0_x
    u0 = fn_u0_x(xs)
    ys_history = np.empty((a_samples.shape[0],  t_history.shape[0], xs.shape[0]))
    print("ys_history", ys_history.shape)
    for i, a_sample in enumerate(a_samples) :
        fn_u0_partial = functools.partial(fn_u0_t, u0, a_sample)
        a_sample_ys_history = np.empty((t_history.shape[0], xs.shape[0]))
        for j, t in enumerate(t_history) :
            a_sample_ys_history[j] = fn_u0_partial(t)
        ys_history[i] = a_sample_ys_history
    return ys_history        

name_dataset = "time_dependent" # "mg", "time", "state"
max_delays = {"time_dependent" : 3, "state_dependent": 1/2, "diffusion_delay":1 }
epochs = 1000
tse_loss, noisyless_tse_mse = [], []


for seed in range(1,6) :
    t = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ts.npy")).float().to(device)
    # t_history = torch.arange(- max_delays[name_dataset], 0.0, (t[1]- t[0])/5)
    # t_history = t_history.to(device)
    # t = torch.cat([t_history, t])+ max_delays[name_dataset]

    train_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys.npy")).float().to(device)
    # train_history = torch.ones(train_trajectories.shape[0], t_history.shape[0], train_trajectories.shape[-1]).to(device)
    # train_history = torch.einsum("ijl, il -> ijl", train_history, train_trajectories[:, 0 ]).to(device)
    # train_a_sample = np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/a_sample.npy")
    # train_history = create_history_array(train_a_sample, t_history)
    # train_history = torch.Tensor(train_history).float().to(device)
    # train_trajectories = torch.cat([train_history, train_trajectories], dim=1)
    
    val_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_test.npy")).float().to(device)
    # val_history = torch.ones(val_trajectories.shape[0], t_history.shape[0], val_trajectories.shape[-1]).to(device)
    # val_history = torch.einsum("ijl, il -> ijl", val_history, val_trajectories[:, 0 ])
    # val_a_sample = np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/a_sample_test.npy")
    # val_history = create_history_array(val_a_sample, t_history)
    # val_history = torch.Tensor(val_history).float().to(device)
    # val_trajectories = torch.cat([val_history, val_trajectories], dim=1)
    
    test_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_test.npy")).float().to(device)
    # test_history = torch.ones(test_trajectories.shape[0], t_history.shape[0], test_trajectories.shape[-1]).to(device)
    # test_history = torch.einsum("ijl, il -> ijl", test_history, test_trajectories[:, 0 ]).to(device)
    # test_a_sample = np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/a_sample_test.npy")
    # test_history = create_history_array(test_a_sample, t_history)
    # test_history = torch.Tensor(test_history).float().to(device)
    # test_trajectories = torch.cat([test_history, test_trajectories], dim=1)

    extrapolate_test_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_extrapolate.npy")).float().to(device)
    # extrapolate_test_history = torch.ones(extrapolate_test_trajectories.shape[0], t_history.shape[0], extrapolate_test_trajectories.shape[-1]).to(device)
    # extrapolate_test_history = torch.einsum("ijl, il -> ijl", extrapolate_test_history, extrapolate_test_trajectories[:, 0 ])
    # extrapolate_a_sample = np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/a_sample_extrapolate.npy")
    # extrapolate_test_history = create_history_array(extrapolate_a_sample, t_history)
    # extrapolate_test_history = torch.Tensor(test_history).float().to(device)
    # extrapolate_test_trajectories = torch.cat([extrapolate_test_history, extrapolate_test_trajectories], dim=1)
    
    other_history_trajectories =  torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_new_history.npy")).float().to(device)
    # amplitude_other_history = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/y0_other_history.npy")).float().to(device)
    # time_other_history = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ts_history.npy")).float().to(device)
    
    # def h(t, time_other_history, amplitude_other_history):
    #     return jnp.array(jax.lax.cond(
    #                 bool(t > time_other_history[0]),
    #                 lambda: amplitude_other_history[0],
    #                 lambda: amplitude_other_history[1],
    #             ))

    # batch_t_history = torch.unsqueeze(t_history, 0).expand(other_history_trajectories.shape[0], -1)
    # masked_t_history = batch_t_history > time_other_history
    
    # first_part_mask_repeat = torch.repeat_interleave(torch.unsqueeze( masked_t_history, -1), extrapolate_test_trajectories.shape[-1], dim=2)
    # second_part_mask_repeat = torch.repeat_interleave(torch.unsqueeze( ~masked_t_history, -1), extrapolate_test_trajectories.shape[-1], dim=2)
    # if len(amplitude_other_history.size()) > 2 : 
    #     first_part_step =  torch.einsum("btf, bf -> btf", first_part_mask_repeat, amplitude_other_history[:, 0])  
    #     second_part_step = torch.einsum("btf, bf -> btf", second_part_mask_repeat  , amplitude_other_history[:, 1])  
    # else :
    #     first_part_step =  torch.einsum("btf, bf -> btf", first_part_mask_repeat , torch.unsqueeze(amplitude_other_history[:, 0], 1))  
    #     second_part_step = torch.einsum("btf, bf -> btf",  second_part_mask_repeat ,  torch.unsqueeze(amplitude_other_history[:, 1], 1))  
    
    # step_function = first_part_step + second_part_step
    
    # for i in range(first_part_step.shape[0]):
    #     jax_t_history = jnp.arange(- max_delays[name_dataset], 0.0, float(t[1]- t[0])/5)
    #     print(time_other_history[i],amplitude_other_history[i])
    #     plt.plot([t for t in jax_t_history], [h(t, time_other_history[i].cpu().numpy(), amplitude_other_history[i].cpu().numpy()) for t in jax_t_history], '--')
    #     plt.plot([t for t in t_history.cpu()], step_function[i].cpu())
    #     plt.show()
        
  
    # other_history_trajectories = torch.cat([step_function, other_history_trajectories], dim=1)
    # print(train_trajectories.shape, other_history_trajectories.shape, other_history_trajectories.shape)
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
        history_nb_time_step = int(t.shape[0]/2)
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
            history_nb_time_step = int(t.shape[0]/2)

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
            history_nb_time_step = int(t.shape[0]/2)

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
            history_nb_time_step = int(t.shape[0]/2)

        ),
    )

    dltest3 = DataLoader(
        other_history_trajectories,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: basic_collate_fn(
            batch,
            t,
            data_type="test",
            extrap=extrapolate,
        history_nb_time_step = int(t.shape[0]/2)
        ),
    )
    
    # for data in dltrain : 
    #     plt.plot(torch.cat([data["observed_tp"].flatten().cpu(), data["tp_to_predict"].flatten().cpu()]), torch.cat([data["observed_data"][1].cpu(),  data["data_to_predict"][1].cpu()]))
    #     plt.show()
        
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
    optimizer = torch.optim.Adam(params, lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    best_loss = float("inf")
    waiting = 0

    name_dataset = "time_50"

    if not os.path.exists(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/"):
        os.makedirs(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/")
        
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
            plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
            plt.plot(
                    tp_to_predict.detach().cpu()[0],
                    batch["data_to_predict"].detach().cpu()[0],
                    "--",
                )
            plt.plot(tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0], c="r")
            plt.title("Neural Laplace Training Phase")
            plt.savefig(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/training_phase_epoch_{epoch}.png")
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
    np.save(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/loss_array.npy", np.array(loss_per_step)) 
    
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
        predictions = laplace_reconstruct(laplace_rep_func, p, tp_to_predict,  recon_dim=output_dim)

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
        plt.savefig(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/testing_phase.png")
        plt.close()
        # predictions = torch.cat([trajs_to_encode, predictions], axis=1)
        # for i in range(10):
        #     plt.plot(predictions.detach().cpu()[i, :, 0])
        #     plt.plot(predictions.detach().cpu()[i, :, 1])
        #     plt.show()
            
        #     plt.plot(predictions.detach().cpu()[i, :, 0], predictions.detach().cpu()[i, :, 1])
        #     plt.show()
        np.save( f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_test_pred.npy",predictions.cpu().detach().numpy())
        cum_test_batches += 1
    test_mse = cum_test_loss / cum_test_batches
    print(f"test_mse= {test_mse}")
    tse_loss.append(test_mse)

    ### FOR NOISELESS DATA FOR TIME DEPENDENT
    # ys_test_noise = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_test_noise.npy")).float().to(device)
    # test_trajectories = torch.Tensor(np.load(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_test.npy")).float().to(device)
    # test_trajectories = test_trajectories - ys_test_noise
    # test_history = torch.ones(test_trajectories.shape[0], t_history.shape[0], test_trajectories.shape[-1]).to(device)
    # test_history = torch.einsum("ijl, il -> ijl", test_history, test_trajectories[:, 0 ]).to(device)
    # test_trajectories = torch.cat([test_history, test_trajectories], dim=1)
    # noiseless_test_trajectories = torch.cat([test_history, test_trajectories], dim=1)
   
    # noiseless_dltest = DataLoader(
    #     test_trajectories,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     collate_fn=lambda batch: basic_collate_fn(
    #         batch,
    #         t,
    #         data_type="test",
    #         extrap=extrapolate,
    #         history_nb_time_step = t_history.shape[0]

    #     ),
    # )

    # laplace_rep_func.eval(), encoder.eval()
    # cum_test_loss = 0
    # cum_test_batches = 0
    # for batch in noiseless_dltest:
    #     trajs_to_encode = batch[
    #         "observed_data"
    #     ]  # (batch_size, t_observed_dim, observed_dim)
    #     observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
    #     p = encoder(
    #         trajs_to_encode, observed_tp
    #     )  # p is the latent tensor encoding the initial states
    #     tp_to_predict = batch["tp_to_predict"]
    #     predictions = laplace_reconstruct(laplace_rep_func, p, tp_to_predict,  recon_dim=output_dim)

    #     cum_test_loss += loss_fn(
    #         torch.flatten(predictions), torch.flatten(batch["data_to_predict"])
    #     ).item()
    #     plt.plot(observed_tp.detach().cpu()[0], trajs_to_encode.detach().cpu()[0])
    #     plt.plot(
    #         tp_to_predict.detach().cpu()[0],
    #         batch["data_to_predict"].detach().cpu()[0],
    #         "--",
    #     )
    #     plt.plot(tp_to_predict.detach().cpu()[0], predictions.detach().cpu()[0])
    #     plt.savefig(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/noiseless_testing_phase.png")
    #     plt.close()
    
    #     np.save( f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/noiseless_ys_test_pred.npy",predictions.cpu().detach().numpy())
    #     cum_test_batches += 1
        
    # noiseless_test_mse = cum_test_loss / cum_test_batches
    # print(f"noiseless test_mse= {noiseless_test_mse}")
    # noisyless_tse_mse.append(noiseless_test_mse)

    for batch in dltest2:
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(laplace_rep_func, p, tp_to_predict,  recon_dim=output_dim)

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
        plt.savefig(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/extrapolate_testing_phase.png")
        plt.close()
        # predictions = torch.cat([trajs_to_encode, predictions], axis=1)
        np.save( f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_extrapolate_pred.npy",predictions.cpu().detach().numpy())
        
        
    for batch in dltest3:
        trajs_to_encode = batch[
            "observed_data"
        ]  # (batch_size, t_observed_dim, observed_dim)
        observed_tp = batch["observed_tp"]  # (1, t_observed_dim)
        p = encoder(
            trajs_to_encode, observed_tp
        )  # p is the latent tensor encoding the initial states
        tp_to_predict = batch["tp_to_predict"]
        predictions = laplace_reconstruct(laplace_rep_func, p, tp_to_predict,  recon_dim=output_dim)

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
        plt.savefig(f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/new_history_testing_phase.png")
        plt.close()
        
        # for i in range(10):
        #     plt.plot(predictions.detach().cpu()[i, :, 0])
        #     plt.plot(predictions.detach().cpu()[i, :, 1])
        #     plt.show()
            
        #     plt.plot(predictions.detach().cpu()[i, :, 0], predictions.detach().cpu()[i, :, 1])
        #     plt.show()
        # predictions = torch.cat([trajs_to_encode, predictions], axis=1)
        np.save( f"../../NeuralLaplace/data/"+str(name_dataset) + f"/run_seed_{seed}/ys_new_history_pred.npy",predictions.cpu().detach().numpy())
        
print(f"MSE LOSS TESS LAPLACE : {np.mean(tse_loss)} +/-  {np.std(tse_loss)}")
# print(f"NOISELESS MSE LOSS TESS LAPLACE : {np.mean(noisyless_tse_mse)} +/-  {np.std(noisyless_tse_mse)}")
