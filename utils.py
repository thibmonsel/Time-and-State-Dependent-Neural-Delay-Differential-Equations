import functools
from math import prod

import jax
import jaxlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import jaxlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as random
import matplotlib.colors as colors

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


def get_number_params(model):
    leaves = jax.tree_util.tree_leaves(model)
    weight_dim = [
        x.shape for x in leaves if isinstance(x, jaxlib.xla_extension.DeviceArray)
    ]
    weight_dim = functools.reduce(
        lambda x, y: x + y, map(lambda x: prod(x), weight_dim)
    )
    return weight_dim


def plot_subplots(
    nb_subplots, ts, ode_ys, anode_ys, dde_ys, latent_ys, laplace_ys, ys_truth
):
    nb_datapoint, pred_length, _ = dde_ys.shape
    random_idx = random.randint(0, nb_datapoint - 1, size=(nb_subplots,), dtype=int)
    font = {"size": 25}

    plt.rc("font", **font)
    print("random_idx", random_idx)
    labels = ["ode", "anode", "laplace", "sddde", "truth"]
    fig, axes = plt.subplots(figsize=(16, 6), nrows=1, ncols=nb_subplots)
    for rdn_idx, i in zip(random_idx, range(nb_subplots)):
        axes[i].plot(ts, ode_ys[rdn_idx], label=labels[0], linewidth=5.0)
        axes[i].plot(ts, anode_ys[rdn_idx], label=labels[1], linewidth=5.0)
        axes[i].plot(ts, laplace_ys[rdn_idx], label=labels[2], linewidth=5.0)
        axes[i].plot(ts, dde_ys[rdn_idx], label=labels[3], linewidth=5.0)
        axes[i].plot(ts, ys_truth[rdn_idx], "-.", label=labels[4], linewidth=5.0)
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("y(t)")

    plt.subplots_adjust(hspace=-3.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), prop={"size": 22})
    plt.tight_layout()
    plt.show()


def plot_phase_space_subplots(
    nb_subplots, ts, ode_ys, anode_ys, dde_ys, latent_ys, laplace_ys, ys_truth
):
    nb_datapoint, pred_length, _ = dde_ys.shape
    random_idx = random.randint(0, nb_datapoint - 1, size=(nb_subplots,), dtype=int)

    font = {"size": 25}

    plt.rc("font", **font)

    print("random_idx", random_idx)
    labels = ["ode", "anode", "laplace", "sddde", "truth"]

    fig, axes = plt.subplots(figsize=(12, 6), nrows=1, ncols=nb_subplots)
    for rdn_idx, i in zip(random_idx, range(nb_subplots)):
        axes[i].plot(
            ode_ys[rdn_idx, :, 0],
            ode_ys[rdn_idx, :, 1],
            label=labels[0],
            linewidth=5.0,
        )
        axes[i].plot(
            anode_ys[rdn_idx, :, 0],
            anode_ys[rdn_idx, :, 1],
            label=labels[1],
            linewidth=5.0,
        )
        axes[i].plot(
            laplace_ys[rdn_idx, :, 0],
            laplace_ys[rdn_idx, :, 1],
            label=labels[2],
            linewidth=5.0,
        )
        axes[i].plot(
            dde_ys[rdn_idx, :, 0],
            dde_ys[rdn_idx, :, 1],
            label=labels[3],
            linewidth=5.0,
        )

        axes[i].plot(
            ys_truth[rdn_idx, :, 0],
            ys_truth[rdn_idx, :, 1],
            "-.",
            label=labels[4],
            linewidth=5.0,
        )

        axes[i].set_xlabel("y(t)")
        axes[i].set_ylabel("x(t)")
        # axes.set_aspect("equal", adjustable="box")
    plt.subplots_adjust(hspace=-3.0)
    handles, labels = axes[i].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), prop={"size": 22})
    plt.tight_layout()
    plt.show()


def predict(node, dde, anode, latent_ode, key, ts, ystest, a_sample=None):
    model_keys = jrandom.split(key, ystest.shape[0])
    if a_sample is None:
        ypred_ode, _ = jax.vmap(node, (None, 0))(ts, ystest[:, 0])
        ypred_dde, _ = jax.vmap(dde, (None, 0))(ts, ystest[:, 0])
        ypred_latent, _ = jax.vmap(latent_ode.sample_deterministic, (None, 0, 0))(
            ts, ystest, model_keys
        )
        y0_anode = jnp.hstack(
            [
                ystest[:, 0],
                jnp.zeros((ystest.shape[0], anode.augmented_dim)),
            ]
        )
        anode_ypred, _ = jax.vmap(anode, (None, 0))(ts, y0_anode)
        return ypred_ode, anode_ypred, ypred_dde, ypred_latent
    else:
        xs = jnp.linspace(0, 1.0, 100)
        ypred_ode = jax.vmap(node, (None, None, 0))(ts, xs, a_sample)
        ypred_dde = jax.vmap(node, (None, None, 0))(ts, xs, a_sample)
        ypred_latent, _ = jax.vmap(latent_ode.sample_deterministic, (None, 0, 0))(
            ts, ystest, model_keys
        )
        anode_ypred = jax.vmap(anode, (None, None, 0))(ts, xs, a_sample)
        return ypred_ode, anode_ypred, ypred_dde, ypred_latent


def predict_other_history(
    node,
    dde,
    anode,
    latent_ode,
    key,
    ts,
    ts_history,
    y0_other_history,
    ys_other_history,
):
    model_keys = jrandom.split(key, ys_other_history.shape[0])
    ypred_ode, _ = jax.vmap(node, (None, 0))(ts, ys_other_history[:, 0])
    ypred_dde, _ = jax.vmap(dde, (None, 0, 0))(ts, y0_other_history, ts_history)
    ypred_latent, _ = jax.vmap(latent_ode.sample_deterministic, (None, 0, 0))(
        ts, ys_other_history, model_keys
    )
    y0_anode = jnp.hstack(
        [
            ys_other_history[:, 0],
            jnp.zeros((ys_other_history.shape[0], anode.augmented_dim)),
        ]
    )

    anode_ypred, _ = jax.vmap(anode, (None, 0))(ts, y0_anode)
    return ypred_ode, anode_ypred, ypred_dde, ypred_latent


def plot_subplots_1d(
    nb_subplots, ts, ode_ys, anode_ys, latent_ode_ys, dde_ys, laplace_ys, ys_truth
):
    nb_datapoint, pred_length, _ = dde_ys.shape
    random_idx = random.randint(0, nb_datapoint - 1, size=1, dtype=int)
    font = {"size": 25}

    plt.rc("font", **font)
    maxi, mini = 0, 0
    labels = ["truth", "ode", "anode", "laplace", "latent_ode", "sddde"]
    for ys_pred, i in zip(
        (ys_truth, ode_ys, anode_ys, laplace_ys, latent_ode_ys, dde_ys),
        [i for i in range(len(labels))],
    ):
        maxi = max(maxi, np.max(ys_pred[random_idx]).all())
        mini = min(mini, np.max(ys_pred[random_idx]).all())
    fig, axes = plt.subplots(figsize=(4.2 * 4.2, 4.2), nrows=1, ncols=len(labels))
    for ys_pred, i in zip(
        (ys_truth, ode_ys, anode_ys, laplace_ys, latent_ode_ys, dde_ys),
        [i for i in range(len(labels))],
    ):
        im = axes[i].imshow(
            np.squeeze(ys_pred[random_idx]),
            vmin=mini,
            vmax=maxi,
            extent=[0, 1, 4, 0],
            aspect=0.25,
            cmap="ocean_r",
        )
        axes[i].invert_yaxis()
        axes[i].set_xlabel("u(x,t)")
        if i == 0:
            axes[i].set_ylabel("t")
        axes[i].set_title("{}".format(labels[i]))

        fig.colorbar(
            im, ax=axes[i], orientation="horizontal"
        )  # ax=axes.ravel().tolist()
    plt.tight_layout()
    plt.show()


def plot_subplots_diff_1d(
    nb_subplots, ts, ode_ys, anode_ys, latent_ode_ys, dde_ys, laplace_ys, ys_truth
):
    nb_datapoint, pred_length, _ = dde_ys.shape
    random_idx = random.randint(0, nb_datapoint - 1, size=1, dtype=int)
    font = {"size": 25}

    plt.rc("font", **font)
    maxi = 0
    labels = ["truth", "ode", "anode", "laplace", "latent_ode", "sddde"]
    fig, axes = plt.subplots(
        figsize=(4.2 * 4.2, 4.2), nrows=1, ncols=len(labels)
    )  # figsize=(4*4,4),extent=[0,4,1,0],
    for ys_pred, i in zip(
        (ys_truth, ode_ys, anode_ys, laplace_ys, latent_ode_ys, dde_ys),
        [i for i in range(len(labels))],
    ):
        maxi = max(
            maxi, jnp.max(jnp.abs((ys_pred[random_idx] - ys_truth[random_idx]))).all()
        )
    for ys_pred, i in zip(
        (ys_truth, ode_ys, anode_ys, laplace_ys, latent_ode_ys, dde_ys),
        [i for i in range(len(labels))],
    ):
        im = axes[i].imshow(
            np.squeeze(jnp.abs(ys_pred[random_idx] - ys_truth[random_idx])),
            norm=colors.LogNorm(vmin=1e-4, vmax=maxi),
            extent=[0, 1, 4, 0],
            aspect=0.25,
            cmap="ocean_r",
        )  # ,
        axes[i].invert_yaxis()
        axes[i].set_xlabel("u(x,t)")
        if i == 0:
            axes[i].set_ylabel("t")
        axes[i].set_title("{}".format(labels[i]))
        fig.colorbar(im, ax=axes[i], orientation="horizontal")

    plt.tight_layout()
    plt.show()
