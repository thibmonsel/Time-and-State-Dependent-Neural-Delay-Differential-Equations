import argparse
import json
import os
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import Delays
import numpy as np
import matplotlib.pyplot as plt

#### REGULAR DATASETS #####

dataset = "time_dependent"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts_5_noise_level.npy")
ys = jnp.load(data_pth + "/ys_5_noise_level.npy")

for i in range(200):
    plt.plot(ts, ys[i])
plt.show()

dataset = "state_dependent"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts.npy")
ys = jnp.load(data_pth + "/ys.npy")

for i in range(200):
    plt.plot(ts, ys[i])
plt.show()

dataset = "diffusion_delay"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts.npy")
ys = jnp.load(data_pth + "/ys.npy")

for i in range(5):
    plt.imshow(ys[i])
    plt.show()


print("EXTRAPOLATE DATASETS")

dataset = "time_dependent"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts_5_noise_level.npy")
ys = jnp.load(data_pth + "/ys_extrapolate.npy")

for i in range(200):
    plt.plot(ts, ys[i])
plt.show()

dataset = "state_dependent"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts.npy")
ys = jnp.load(data_pth + "/ys_extrapolate.npy")

for i in range(200):
    plt.plot(ts, ys[i])
plt.show()

dataset = "diffusion_delay"
data_pth = "data/" + dataset

ts = jnp.load(data_pth + "/ts.npy")
ys = jnp.load(data_pth + "/ys_extrapolate.npy")

for i in range(5):
    plt.imshow(ys[i])
    plt.show()
