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
from models import NeuralDDEWithTime, PDENeuralDDE, PDENeuralODE, PDEANODE, NeuralDDE, ANODE, NeuralODE, LatentODE


#### CHANGE ACCORDING TO WHERE MODEL WAS SAVED ######
dde_path = "cluster_jobs/time_dep/dde"
anode_path = "cluster_jobs/time_dep/anode"
ode_path = "cluster_jobs/time_dep/ode"
latent_ode_path = "cluster_jobs/time_dep/latent_ode"


dde_runs = [filename for filename in os.listdir(dde_path)]
anode_runs = [filename for filename in os.listdir(anode_path)]
latent_ode_runs = [filename for filename in os.listdir(latent_ode_path)]
ode_runs = [filename for filename in os.listdir(ode_path)]

dde_losses = []
for run in dde_runs:
    test_loss = jnp.load(dde_path + "/" + run + "/dde/test_loss_array.npy")
    dde_losses.append(test_loss[-1])

anode_losses = []
for run in anode_runs:
    test_loss = jnp.load(anode_path + "/" + run + "/anode/test_loss_array.npy")
    anode_losses.append(test_loss[-1])

latent_ode_losses = []
for run in latent_ode_runs:
    test_loss = jnp.load(latent_ode_path + "/" + run + "/latent_ode/test_loss_array.npy")
    latent_ode_losses.append(test_loss[-1])

ode_losses = []
for run in ode_runs:
    test_loss = jnp.load(ode_path + "/" + run + "/ode/test_loss_array.npy")
    ode_losses.append(test_loss[-1])

print("ODE: ", np.mean(ode_losses), np.std(ode_losses))
print("ANODE: ", np.mean(anode_losses), np.std(anode_losses))
print("DDE: ", np.mean(dde_losses), np.std(dde_losses))
print("Latent ODE: ", np.mean(latent_ode_losses), np.std(latent_ode_losses))
