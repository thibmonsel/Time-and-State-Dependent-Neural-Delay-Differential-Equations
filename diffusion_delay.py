import argparse
import datetime
import json
import os

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from diffrax import Delays
from jax.lib import xla_bridge
from models import (
    PDEANODE,
    fit_latent,
    LatentODE,
    PDENeuralODE,
    pde_fit,
    PDENeuralDDE,
)
import numpy as np
from utils import dic_act


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    print("Hardware :", xla_bridge.get_backend().platform)
    parser = argparse.ArgumentParser(description="Running node experiments")
    parser.add_argument(
        "--model",
        help="which model to use",
        choices=["anode", "ode", "dde", "latent_ode"],
    )
    parser.add_argument("--seed", type=int, default=np.random.randint(0,10000))
    parser.add_argument("--exp_path", default="")
    parser.add_argument("--augmented_dim", type=int, default=10)
    args = parser.parse_args()

    augmented_dim = args.augmented_dim

    if args.exp_path == "":
        default_save_dir = "meta_data"
    else:
        default_save_dir = "meta_data/" + args.exp_path

    if not os.path.exists(default_save_dir):
        os.makedirs(default_save_dir)

    datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    default_dir = default_save_dir + "/" + datestring
    os.makedirs(default_dir)

    key = jrandom.PRNGKey(args.seed)
    model_key, view_key = jax.random.split(key, 2)
    seed_train_test_split = os.environ["seed_train_test_split"]

    ys_raw, ts = jnp.load("data/diffusion_delay/ys.npy"), jnp.load(
        "data/diffusion_delay/ts.npy"
    )
    a_sample_raw = jnp.load("data/diffusion_delay/a_sample.npy")
    print("ys.shape, ts.shape", ys_raw.shape, ts.shape)

    train_idx, test_idx = jnp.load(
        f"data/diffusion_delay/train_indices_{seed_train_test_split}.npy"
    ), jnp.load(f"data/diffusion_delay/test_indices_{seed_train_test_split}.npy")

    ys = ys_raw[train_idx]
    ystest = ys_raw[test_idx]
    a_sample = a_sample_raw[train_idx]
    a_sample_test = a_sample_raw[test_idx]

    print("ys.shape, ystest.shape", ys.shape, ystest.shape)
    print("a_sample.shape, a_sample_test.shape", a_sample.shape, a_sample_test.shape)

    xs = jnp.linspace(0, 1.0, 100)
    ts = jnp.linspace(0, 10.0, 100)

    batch_size = 256
    width, depth, activation = 64, 3, "relu"
    _, length_size, data_size = ys.shape

    plt.imshow(ys[30])
    plt.tight_layout()
    plt.xlabel("x(t)")
    plt.ylabel("t")
    plt.savefig(default_dir + "/data_phase_space.png")
    plt.close()

    length_strategy = (1.0,)
    epoch_strategy = (500,)
    lr_strategy = (0.01,)

    json_filename = "hyper_parameters.json"
    dic_data = {
        "id": datestring,
        "metadata": {
            "seed": args.seed,
            "seed_train_test_split" : seed_train_test_split,
            "augmented_dim": augmented_dim,
            "batch_size": batch_size,
            "epoch_strategy": epoch_strategy,
            "length_strategy": length_strategy,
            "tf": 1.0,
            "nb_steps": 500,
            "dataset_size": ys.shape[0],
            "width": width,
            "depth": depth,
            "lr_strategy": lr_strategy,
            "activation": activation,
            "hardware": xla_bridge.get_backend().platform,
        },
    }

    with open(default_dir + "/" + json_filename, "w") as file:
        json.dump([dic_data], file)

    ##### Neural ODE #######
    if args.model == "ode":
        print("TRAINING ODE")
        os.makedirs(default_dir + "/ode")
        os.makedirs(default_dir + "/ode/training")
        model_ode = PDENeuralODE(
            data_size, width, depth, dic_act[activation], key=model_key
        )

        loss_per_step =  pde_fit(
            ts,
            xs,
            ys,
            ystest,
            a_sample,
            a_sample_test,
            model_ode,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "dde":
        print("TRAINING DDE")
        ##### Neural DDE #######
        os.makedirs(default_dir + "/dde")
        os.makedirs(default_dir + "/dde/training")
        delays = Delays(
            delays=[lambda t, y, args: 2.0],
            initial_discontinuities=jnp.array([0.0]),
            max_discontinuities=2,
        )
        model_dde = PDENeuralDDE(
            data_size, width, depth, dic_act[activation], delays, key=model_key
        )
        loss_per_step2 = pde_fit(
            ts,
            xs,
            ys,
            ystest,
            a_sample,
            a_sample_test,
            model_dde,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "anode":
        print("TRAINING ANODE")
        ##### Neural ANODE #######
        os.makedirs(default_dir + "/anode")
        os.makedirs(default_dir + "/anode/training")
        model_anode = PDEANODE(
            data_size,
            data_size,
            width,
            depth,
            dic_act[activation],
            key=model_key,
        )
        loss_per_step3 = pde_fit(
            ts,
            xs,
            ys,
            ystest,
            a_sample,
            a_sample_test,
            model_anode,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "latent_ode":
        print("TRAINING Latent ODE")
        ##### Latent ODE #######
        os.makedirs(default_dir + "/latent_ode")
        os.makedirs(default_dir + "/latent_ode/training")
        latent_ode = LatentODE(data_size, width, width, width, depth, model_key)

        loss_per_step4 = fit_latent(
            ts,
            ys,
            ystest,
            latent_ode,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )
