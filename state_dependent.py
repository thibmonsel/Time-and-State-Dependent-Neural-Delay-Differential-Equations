import argparse
import datetime
import json
import os

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
from diffrax import Delays
from jax.lib import xla_bridge
import numpy as np
from models import ANODE, fit, NeuralDDEWithTime, NeuralODE, LatentODE, fit_latent
from utils import dic_act

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == "__main__":
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
    try : 
        seed_train_test_split = os.environ["seed_train_test_split"]
    except:
        seed_train_test_split = np.random.randint(1, 5)
    
    print("using seed_train_test_split", seed_train_test_split)

    ys_raw, ts = jnp.load("data/state_dependent/ys.npy"), jnp.load(
        "data/state_dependent/ts.npy"
    )
    print("ys.shape, ts.shape", ys_raw.shape, ts.shape)

    train_idx, test_idx = jnp.load(
        f"data/state_dependent/train_indices_{seed_train_test_split}.npy"
    ), jnp.load(f"data/state_dependent/test_indices_{seed_train_test_split}.npy")

    ys = ys_raw[train_idx]
    ystest = ys_raw[test_idx]
    print("ys.shape, ystest.shape", ys.shape, ystest.shape)

    batch_size = 256
    width, depth, activation = 64, 3, "relu"
    _, length_size, data_size = ys.shape
    
    for i in range(ys.shape[0]):
        plt.plot(jnp.gradient(ys[i, :, 0]), ys[i])
    plt.tight_layout()
    plt.xlabel("x(t)")
    plt.ylabel("x'(t)")
    plt.savefig(default_dir + "/data_phase_space.png")
    plt.close()

    length_strategy = (1.0,)
    epoch_strategy = (1000,) 
    lr_strategy = (1e-3,)

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
            "tf": float(ts[-1]),
            "nb_steps": ts.shape[0],
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
        os.makedirs(default_dir + "/ode")
        os.makedirs(default_dir + "/ode/training")
        model_ode = NeuralODE(
            data_size, width, depth, dic_act[activation], key=model_key
        )

        loss_per_step = fit(
            ts,
            ys,
            ystest,
            model_ode,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "dde":
        ##### Neural DDE #######
        os.makedirs(default_dir + "/dde")
        os.makedirs(default_dir + "/dde/training")

        delays = Delays(
            delays=[lambda t, y, args: 1 / 2 * jnp.cos(y[0])],
            initial_discontinuities=jnp.array([0.0]),
            max_discontinuities=2,
            recurrent_checking=False,
            rtol=10e-3,
            atol=10e-6,
        )

        model_dde = NeuralDDEWithTime(
            data_size, width, depth, dic_act[activation], delays, key=model_key
        )
        loss_per_step2 = fit(
            ts,
            ys,
            ystest,
            model_dde,
            batch_size//2,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "anode":
        ##### Neural ANODE #######
        os.makedirs(default_dir + "/anode")
        os.makedirs(default_dir + "/anode/training")
        model_anode = ANODE(
            data_size,
            augmented_dim,
            width,
            depth,
            dic_act[activation],
            key=model_key,
        )
        loss_per_step3 = fit(
            ts,
            ys,
            ystest,
            model_anode,
            batch_size,
            default_dir,
            key,
            lr_strategy,
            epoch_strategy,
            length_strategy,
        )

    if args.model == "latent_ode":
        ####### Latent ODE #######
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
