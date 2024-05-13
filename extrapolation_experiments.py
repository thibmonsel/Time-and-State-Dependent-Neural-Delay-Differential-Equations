import argparse
from diffrax import Delays
import json
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jaxlib
import os
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
from models import NeuralDDEWithTime, NeuralDDE, ANODE, NeuralODE, LatentODE
from utils import plot_subplots, plot_phase_space_subplots, predict


dic_act = {
    "relu": jnn.relu,
    "tanh": jnn.tanh,
    "softplus": jnn.softplus,
    "elu": jnn.elu,
    "sigmoid": jnn.sigmoid,
    "silu": jnn.silu,
    "gelu": jnn.gelu,
    "id": lambda x: x,
    "swish": jnn.swish,
    "hard_tanh": jnn.hard_tanh,
    "lecun_tanh": lambda x: 1.7159 * jnn.tanh(2 / 3 * x),
    "mish": lambda x: x * jnn.tanh(jnp.log(1 + jnn.sigmoid(x))),
}

#### CHANGE ACCORDING TO WHERE MODEL WAS SAVED ######
dde_path = "titanic/meta_data/time_dep/dde"
anode_path = "titanic/meta_data/time_dep/anode"
ode_path = "titanic/meta_data/time_dep/ode"
latent_ode_path = "titanic/meta_data/time_dep/latent_ode"

dde_runs = [filename for filename in os.listdir(dde_path)]
anode_runs = [filename for filename in os.listdir(anode_path)]
latent_ode_runs = [filename for filename in os.listdir(latent_ode_path)]
ode_runs = [filename for filename in os.listdir(ode_path)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating dataset")
    parser.add_argument("--noise_level", type=float, default=-1)
    parser.add_argument(
        "--dataset", choices=["time_dependent", "diffusion_delay", "state_dependent"]
    )
    parser.add_argument("--dataset_saving_path", default="data/")
    args = parser.parse_args()

    key = jrandom.PRNGKey(0)
    if args.dataset == "state_dependent":
        nb_plots = 2
        delays = Delays(
            delays=[lambda t, y, args: 1 / 2 * jnp.cos(y[0])],
            max_discontinuities=100,
            recurrent_checking=False,
            rtol=10e-3,
            atol=10e-6,
        )
        ####### LOADING TEST DATASET #######
        seed_train_test_split = 1
        ys_extrapolate, ts = jnp.load("data/state_dependent/ys_extrapolate.npy"), jnp.load(
            "data/state_dependent/ts.npy"
        )
        data_size = ys_extrapolate.shape[-1]
        #### LOADING TRAINED DDE MODEL ####
        with open(dde_path + "/" +  dde_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_dde = NeuralDDE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            delays,
            key=key,
        )

        model_dde_loaded = eqx.tree_deserialise_leaves(
            dde_path + "/" +  dde_runs[0] + "/dde/last.eqx", model_dde
        )

        #### LOADING TRAINED ANODE MODEL ####
        with open(anode_path + "/"+ anode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        anode = ANODE(
            data_size,
            hyperparams["augmented_dim"],
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )
        anode_loaded = eqx.tree_deserialise_leaves(
            anode_path + "/"+ anode_runs[0] + "/anode/last.eqx", anode
        )

        #### LOADING TRAINED NODE MODEL ####
        with open(ode_path + "/" + ode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_ode = NeuralODE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )

        model_ode_loaded = eqx.tree_deserialise_leaves(
            ode_path + "/" +  ode_runs[0] + "/ode/last.eqx", model_ode
        )

        #### LOADING TRAINED LATENT ODE MODEL ####
        with open(latent_ode_path + "/" +  latent_ode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        latent_ode = LatentODE(
            data_size,
            hyperparams["width"],
            hyperparams["width"],
            hyperparams["width"],
            hyperparams["depth"],
            key=key,
        )

        latent_ode_loaded = eqx.tree_deserialise_leaves(
            latent_ode_path + "/" +  latent_ode_runs[0] + "/latent_ode/last.eqx", latent_ode
        )

        ###### PREDICTION TEST DATASET ##########
        ypred_ode, anode_ypred, ypred_dde, ypred_latent = predict(
            model_ode_loaded,
            model_dde_loaded,
            anode_loaded,
            latent_ode_loaded,
            key,
            ts,
            ys_extrapolate,
        )

        ypred_laplace = jnp.load("results/" + str(args.dataset) + "/split_index_5/ys_test_pred.npy")
        
        plot_subplots(
            nb_plots,
            ts,
            ypred_ode,
            anode_ypred,
            ypred_dde,
            ypred_latent,
            ypred_laplace,
            ys_extrapolate,
        )

    if args.dataset == "time_dependent":
        nb_plots = 2
        delays = Delays(
            delays=[lambda t, y, args: 2 + jnp.sin(t)],
            initial_discontinuities=jnp.array([0.0]),
            max_discontinuities=2,
        )
        ####### LOADING TEST DATASET #######
        seed_train_test_split = 1
        ys_extrapolate, ts = jnp.load("data/time_dependent/ys_extrapolate.npy"), jnp.load(
            "data/time_dependent/ts_0_noise_level.npy"
        )
        data_size = ys_extrapolate.shape[-1]
        #### LOADING TRAINED DDE MODEL ####
        with open(dde_path + "/" +  dde_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_dde = NeuralDDE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            delays,
            key=key,
        )

        model_dde_loaded = eqx.tree_deserialise_leaves(
            dde_path + "/" +  dde_runs[0] + "/dde/last.eqx", model_dde
        )

        #### LOADING TRAINED ANODE MODEL ####
        with open(anode_path + "/"+ anode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        anode = ANODE(
            data_size,
            hyperparams["augmented_dim"],
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )
        anode_loaded = eqx.tree_deserialise_leaves(
            anode_path + "/"+ anode_runs[0] + "/anode/last.eqx", anode
        )

        #### LOADING TRAINED NODE MODEL ####
        with open(ode_path + "/" + ode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_ode = NeuralODE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )

        model_ode_loaded = eqx.tree_deserialise_leaves(
            ode_path + "/" +  ode_runs[0] + "/ode/last.eqx", model_ode
        )

        #### LOADING TRAINED LATENT ODE MODEL ####
        with open(latent_ode_path + "/" +  latent_ode_runs[0] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        latent_ode = LatentODE(
            data_size,
            hyperparams["width"],
            hyperparams["width"],
            hyperparams["width"],
            hyperparams["depth"],
            key=key,
        )

        latent_ode_loaded = eqx.tree_deserialise_leaves(
            latent_ode_path + "/" +  latent_ode_runs[0] + "/latent_ode/last.eqx", latent_ode
        )

        ###### PREDICTION TEST DATASET ##########
        ypred_ode, anode_ypred, ypred_dde, ypred_latent = predict(
            model_ode_loaded,
            model_dde_loaded,
            anode_loaded,
            latent_ode_loaded,
            key,
            ts,
            ys_extrapolate,
        )

        ypred_laplace = jnp.load("results/" + str(args.dataset) + "/split_index_5/ys_test_pred.npy")
        
        for i in range(10):
            plot_subplots(
                nb_plots,
                ts,
                ypred_ode,
                anode_ypred,
                ypred_dde,
                ypred_latent,
                ypred_laplace,
                ys_extrapolate,
            )
