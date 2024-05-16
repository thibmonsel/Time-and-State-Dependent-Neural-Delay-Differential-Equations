import argparse
from diffrax import Delays
import json
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import os
import numpy as np
from models import NeuralDDEWithTime, PDENeuralDDE, PDENeuralODE, PDEANODE, NeuralDDE, ANODE, NeuralODE, LatentODE
from utils import plot_subplots, plot_subplots_1d, plot_subplots_diff_1d,predict, dic_act


#### CHANGE ACCORDING TO WHERE MODEL WAS SAVED ######
dde_path = "cluster_jobs/diffusion_delay_mlp_128/dde"
anode_path = "cluster_jobs/diffusion_delay_mlp_128/anode"
ode_path = "cluster_jobs/diffusion_delay_mlp_128/ode"
latent_ode_path = "cluster_jobs/diffusion_delay_mlp_128/latent_ode"

dde_runs = [filename for filename in os.listdir(dde_path)]
anode_runs = [filename for filename in os.listdir(anode_path)]
latent_ode_runs = [filename for filename in os.listdir(latent_ode_path)]
ode_runs = [filename for filename in os.listdir(ode_path)]

seed_train_test_split = np.random.randint(1,5)
print(f"seed_train_test_split{seed_train_test_split}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating dataset")
    parser.add_argument("--noise_level", type=float, default=-1)
    parser.add_argument(
        "--dataset", choices=["time_dependent", "diffusion_delay", "state_dependent"]
    )
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
        ys_raw, ts = jnp.load("data/state_dependent/ys.npy"), jnp.load("data/state_dependent/ts.npy")
        print("ys.shape, ts.shape", ys_raw.shape, ts.shape)

        test_idx = jnp.load(f"data/state_dependent/test_indices_{seed_train_test_split}.npy")

        ystest = ys_raw[test_idx]
        data_size = ys_raw.shape[-1]
        #### LOADING TRAINED DDE MODEL ####
        with open(dde_path + "/" +  dde_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_dde = NeuralDDEWithTime(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            delays,
            key=key,
        )

        model_dde_loaded = eqx.tree_deserialise_leaves(
            dde_path + "/" +  dde_runs[seed_train_test_split] + "/dde/last.eqx", model_dde
        )

        #### LOADING TRAINED ANODE MODEL ####
        with open(anode_path + "/"+ anode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            anode_path + "/"+ anode_runs[seed_train_test_split] + "/anode/last.eqx", anode
        )

        #### LOADING TRAINED NODE MODEL ####
        with open(ode_path + "/" + ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_ode = NeuralODE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )

        model_ode_loaded = eqx.tree_deserialise_leaves(
            ode_path + "/" +  ode_runs[seed_train_test_split] + "/ode/last.eqx", model_ode
        )

        #### LOADING TRAINED LATENT ODE MODEL ####
        with open(latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/latent_ode/last.eqx", latent_ode
        )

        ###### PREDICTION TEST DATASET ##########
        ypred_ode, anode_ypred, ypred_dde, ypred_latent = predict(
            model_ode_loaded,
            model_dde_loaded,
            anode_loaded,
            latent_ode_loaded,
            key,
            ts,
            ystest,
        )

        ypred_laplace = jnp.load("results/" + str(args.dataset) + f"/split_index_{seed_train_test_split}/ys_test_pred.npy")
        
        for i in range(4):
            plot_subplots(
                nb_plots,
                ts,
                ypred_ode,
                anode_ypred,
                ypred_dde,
                ypred_latent,
                ypred_laplace,
                ystest,
            )

    if args.dataset == "time_dependent":
        nb_plots = 2
        delays = Delays(
            delays=[lambda t, y, args: 2 + jnp.sin(t)],
            initial_discontinuities=jnp.array([0.0]),
            max_discontinuities=2,
        )
        ####### LOADING TEST DATASET #######
        ys_raw, ts = jnp.load(
            "data/time_dependent/ys_" + str(int(args.noise_level)) + "_noise_level.npy"
        ), jnp.load(
            "data/time_dependent/ts_" + str(int(args.noise_level)) + "_noise_level.npy"
        )
        print("ys.shape, ts.shape", ys_raw.shape, ts.shape)

        test_idx = jnp.load(f"data/time_dependent/test_indices_{seed_train_test_split}.npy")
        ystest = ys_raw[test_idx]

        data_size = ystest.shape[-1]
        #### LOADING TRAINED DDE MODEL ####
        with open(dde_path + "/" +  dde_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            dde_path + "/" +  dde_runs[seed_train_test_split] + "/dde/last.eqx", model_dde
        )

        #### LOADING TRAINED ANODE MODEL ####
        with open(anode_path + "/"+ anode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            anode_path + "/"+ anode_runs[seed_train_test_split] + "/anode/last.eqx", anode
        )

        #### LOADING TRAINED NODE MODEL ####
        with open(ode_path + "/" + ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_ode = NeuralODE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )

        model_ode_loaded = eqx.tree_deserialise_leaves(
            ode_path + "/" +  ode_runs[seed_train_test_split] + "/ode/last.eqx", model_ode
        )

        #### LOADING TRAINED LATENT ODE MODEL ####
        with open(latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/latent_ode/last.eqx", latent_ode
        )

        ###### PREDICTION TEST DATASET ##########
        ypred_ode, anode_ypred, ypred_dde, ypred_latent = predict(
            model_ode_loaded,
            model_dde_loaded,
            anode_loaded,
            latent_ode_loaded,
            key,
            ts,
            ystest,
        )

        ypred_laplace = jnp.load("results/" + str(args.dataset) + f"/split_index_{seed_train_test_split}/ys_" + str(int(args.noise_level)) + "_noise_level.npy")
        
        for i in range(4):
            plot_subplots(
                nb_plots,
                ts,
                ypred_ode,
                anode_ypred,
                ypred_dde,
                ypred_latent,
                ypred_laplace,
                ystest,
            )

    if args.dataset == "diffusion_delay":
        nb_plots = 4
        delays = Delays(
            delays=[lambda t, y, args: 2.0],
            initial_discontinuities=jnp.array([0.0]),
            max_discontinuities=2,
        )
        ####### LOADING TEST DATASET #######
        ys_raw, ts = jnp.load("data/diffusion_delay/ys.npy"), jnp.load(
            "data/diffusion_delay/ts.npy"
        )
        a_sample_raw = jnp.load("data/diffusion_delay/a_sample.npy")
        print("ys.shape, ts.shape", ys_raw.shape, ts.shape)

        test_idx = jnp.load(f"data/diffusion_delay/test_indices_{seed_train_test_split}.npy")

        ystest = ys_raw[test_idx]
        a_sample_test = a_sample_raw[test_idx]

        data_size = ystest.shape[-1]
        #### LOADING TRAINED DDE MODEL ####
        with open(dde_path + "/" +  dde_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_dde = PDENeuralDDE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            delays,
            key=key,
        )

        model_dde_loaded = eqx.tree_deserialise_leaves(
            dde_path + "/" +  dde_runs[seed_train_test_split] + "/dde/last.eqx", model_dde
        )

        #### LOADING TRAINED ANODE MODEL ####
        with open(anode_path + "/"+ anode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        anode = PDEANODE(
            data_size,
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )
        anode_loaded = eqx.tree_deserialise_leaves(
            anode_path + "/"+ anode_runs[seed_train_test_split] + "/anode/last.eqx", anode
        )

        #### LOADING TRAINED NODE MODEL ####
        with open(ode_path + "/" + ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
            hyperparams = json.load(json_file)[0]["metadata"]

        model_ode = PDENeuralODE(
            data_size,
            hyperparams["width"],
            hyperparams["depth"],
            dic_act[hyperparams["activation"]],
            key=key,
        )

        model_ode_loaded = eqx.tree_deserialise_leaves(
            ode_path + "/" +  ode_runs[seed_train_test_split] + "/ode/last.eqx", model_ode
        )

        #### LOADING TRAINED LATENT ODE MODEL ####
        with open(latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/hyper_parameters.json") as json_file:
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
            latent_ode_path + "/" +  latent_ode_runs[seed_train_test_split] + "/latent_ode/last.eqx", latent_ode
        )

        ###### PREDICTION TEST DATASET ##########
        ypred_ode, anode_ypred, ypred_dde, ypred_latent = predict(
            model_ode_loaded,
            model_dde_loaded,
            anode_loaded,
            latent_ode_loaded,
            key,
            ts,
            ystest,
            a_sample_test
        )

        ypred_laplace = jnp.load("results/" + str(args.dataset) + "/split_index_1/ys_test_pred.npy")
        
        for i in range(4):
            plot_subplots_1d(
                nb_plots,
                ts,
                ypred_ode,
                anode_ypred,
                ypred_latent,
                ypred_dde,
                ypred_laplace,
                ystest,
            )
            plot_subplots_diff_1d(
                nb_plots,
                ts,
                ypred_ode,
                anode_ypred,
                ypred_latent,
                ypred_dde,
                ypred_laplace,
                ystest,
            )