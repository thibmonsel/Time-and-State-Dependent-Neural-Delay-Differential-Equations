import argparse
import json
import os
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from diffrax import Delays
import numpy as np
from dataset import (
    time_dependent_dataset,
    time_dependent_dataset_different,
    state_dependent_dataset,
    state_dependent_dataset_different,
    diffusion_delay,
    check_split_indices,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generating dataset")
    parser.add_argument("--data_seed", type=int, default=np.random.randint(0, 1000))
    parser.add_argument("--nb_split_indices", type=int, default=5)
    parser.add_argument("--noise_level", type=float, default=-1)
    parser.add_argument(
        "--dataset", choices=["time_dependent", "diffusion_delay", "state_dependent"]
    )
    parser.add_argument("--dataset_saving_path", default="data/")
    args = parser.parse_args()

    saving_path = args.dataset_saving_path + args.dataset
    os.makedirs(saving_path, exist_ok=True)

    if args.dataset == "time_dependent":
        ####### GENERATING TIME DEPENDENT TRAIN/TEST DATASET #######
        if any(
            file.endswith("ys_" + str(args.noise_level) + "_noise_level.npy")
            for file in os.listdir(saving_path)
        ):
            pass
            print("Already have time dependent dataset")
        else:
            if args.noise_level == -1:
                raise ValueError("Noise level must be specified")
            if args.noise_level < 0 or args.noise_level > 100:
                raise ValueError("Noise level must be between 0 and 100")
            print(
                f"Generating time dependent dataset with noise level {args.noise_level}"
            )
            data_key = jrandom.PRNGKey(np.random.randint(0, 1000))
            tf, nb_steps, dataset_size, test_size = (
                20.0,
                200,
                256,
                32,
            )
            ts = jnp.linspace(0.0, tf, nb_steps)
            delays = Delays(
                delays=[lambda t, y, args: 2 + jnp.sin(t)],
                initial_discontinuities=jnp.array([0.0]),
                max_discontinuities=2,
                recurrent_checking=False,
                rtol=10e-3,
                atol=10e-6,
            )
            _, data_key = jrandom.split(data_key, 2)

            y0_sample = jrandom.uniform(
                data_key, (dataset_size + test_size, 1), minval=0.1, maxval=2.0
            )
            ys = jax.vmap(time_dependent_dataset, (None, None, 0))(
                ts, delays, y0_sample
            )
            ys = (
                ys
                + jrandom.normal(data_key, ys.shape)
                * jnp.expand_dims(
                    jnp.repeat(jnp.std(ys, axis=1), axis=1, repeats=ys.shape[1]), -1
                )
                * args.noise_level
                / 100
            )

            dic_data = {
                "data_seed": args.data_seed,
                "input_shape": ys.shape[1:],
                "dataset_size": dataset_size + test_size,
                "t0": float(ts[0]),
                "t1": float(ts[-1]),
                "ts_shape": ts.shape,
            }

            jnp.save(
                saving_path + "/ys_" + str(int(args.noise_level)) + "_noise_level.npy",
                ys,
            )
            jnp.save(
                saving_path + "/ts_" + str(int(args.noise_level)) + "_noise_level.npy",
                ts,
            )

            with open(
                saving_path + "/data_generation_hyper_parameters.json", "w"
            ) as file:
                json.dump([dic_data], file)

        if any(file.startswith("test_indices") for file in os.listdir(saving_path)):
            print(
                "Not creating additional split indices for other noise levels since they already exist"
            )
            pass
        else:
            print("Creating train/test split indices")
            _, split_idx_key = jrandom.split(data_key, 2)
            ys = jnp.load(
                saving_path + "/ys_" + str(int(args.noise_level)) + "_noise_level.npy"
            )
            check_split_indices(
                ys, dataset_size, saving_path, args.nb_split_indices, split_idx_key
            )

        ####### GENERATING TIME DEPENDENT Extrapolate DATASET #######
        if any(file.endswith("extrapolate.npy") for file in os.listdir(saving_path)):
            print("Already have extrapolation and step history datasets")
        else:
            print("Creating Extrapolation and other history datasets")
            data_key = jrandom.PRNGKey(np.random.randint(0, 1000))
            tf, nb_steps, dataset_size, test_size = (
                20.0,
                200,
                256,
                32,
            )
            ts = jnp.linspace(0.0, tf, nb_steps)

            delays = Delays(
                delays=[lambda t, y, args: 2 + jnp.sin(t)],
                initial_discontinuities=jnp.array([0.0]),
                max_discontinuities=2,
                recurrent_checking=False,
                rtol=10e-3,
                atol=10e-6,
            )

            _, extrapolate_data_key = jrandom.split(data_key, 2)
            y0_extrapolate = jrandom.uniform(
                extrapolate_data_key, (test_size, 1), minval=2.0, maxval=3.0
            )
            ys_extrapolate = jax.vmap(time_dependent_dataset, (None, None, 0))(
                ts, delays, y0_extrapolate
            )
            jnp.save(saving_path + "/ys_extrapolate.npy", ys_extrapolate)

            _, other_history_key = jrandom.split(extrapolate_data_key)
            y0_other_history = jrandom.uniform(
                other_history_key, (test_size, 2), minval=0.1, maxval=3.0
            )
            _, other_ts_history_key = jrandom.split(other_history_key)
            ts_history = jrandom.uniform(
                other_ts_history_key, (test_size, 1), minval=-3.0, maxval=0.0
            )

            ys_other_history_test = jax.vmap(
                time_dependent_dataset_different, (None, None, 0, 0)
            )(ts, delays, y0_other_history, ts_history)

            # This corresponds to the 2 values of the step fn
            jnp.save(saving_path + "/y0_other_history.npy", y0_other_history)
            # This corresponds to the time t where the step fn changes
            jnp.save(saving_path + "/ts_history.npy", ts_history)
            # This is the next dynamics with this new history function
            jnp.save(saving_path + "/ys_other_history_test.npy", ys_other_history_test)

    if args.dataset == "diffusion_delay":
        if any(file.startswith("ys.npy") for file in os.listdir(saving_path)):
            print("Already have diffusion delay dataset")
            pass
        else:
            print("Generating diffusion delay dataset")
            data_key = jrandom.PRNGKey(np.random.randint(0, 1000))
            dataset_size, test_size = 256, 32

            xs = jnp.linspace(0, 1.0, 100)
            ts = jnp.linspace(0, 4.0, 100)

            delays = Delays(
                delays=[lambda t, y, args: 1.0],
                initial_discontinuities=jnp.array([0.0]),
                max_discontinuities=2,
            )
            a_sample = jrandom.uniform(
                data_key, (dataset_size + test_size,), minval=0.1, maxval=4.0
            )
            ys = jax.vmap(diffusion_delay, (None, None, None, 0))(
                ts, xs, delays, a_sample
            )
            print("Creating Extrapolation dataset")
            _, data_key = jrandom.split(data_key, 2)
            a_sample_extrapolate = jrandom.uniform(
                data_key, (test_size,), minval=4.0, maxval=7.0
            )
            ys_extrapolate = jax.vmap(diffusion_delay, (None, None, None, 0))(
                ts, xs, delays, a_sample_extrapolate
            )
            jnp.save(saving_path + "/ys.npy", ys)
            jnp.save(saving_path + "/ts.npy", ts)
            jnp.save(saving_path + "/a_sample.npy", a_sample)

            # Saving extrapolation dataset
            jnp.save(saving_path + "/ys_extrapolate.npy", ys_extrapolate)
            jnp.save(saving_path + "/a_sample_extrapolate.npy", a_sample_extrapolate)

        if any(file.startswith("test_indices") for file in os.listdir(saving_path)):
            print(
                "Not creating additional split indices for other noise levels since they already exist"
            )
            pass
        else:
            print("Creating train/test split indices")
            _, split_idx_key = jrandom.split(data_key, 2)
            ys = jnp.load(saving_path + "/ys.npy")
            check_split_indices(
                ys, dataset_size, saving_path, args.nb_split_indices, split_idx_key
            )

    if args.dataset == "state_dependent":
        ####### GENERATING STATE DEPENDENT TRAIN/TEST DATASET #######
        if any(file.endswith("ys.npy") for file in os.listdir(saving_path)):
            pass
            print("Already have state dependent dataset")
        else:
            print("Generating state dependent dataset ")
            data_key = jrandom.PRNGKey(np.random.randint(0, 1000))
            tf, nb_steps, dataset_size, batch_size, test_size = (
                10.0,
                150,
                256,
                256,
                32,
            )
            ts = jnp.linspace(0.0, tf, nb_steps)
            delays = Delays(
                delays=[lambda t, y, args: 1 / 2 * jnp.cos(y[0])],
                initial_discontinuities=jnp.array([0.0]),
                max_discontinuities=2,
                recurrent_checking=False,
                rtol=10e-3,
                atol=10e-6,
            )
            _, data_key = jrandom.split(data_key, 2)

            y0_sample = jrandom.uniform(
                data_key, (dataset_size + test_size, 1), minval=0.1, maxval=1.0
            )
            ys = jax.vmap(state_dependent_dataset, (None, None, 0))(
                ts, delays, y0_sample
            )

            dic_data = {
                "data_seed": args.data_seed,
                "input_shape": ys.shape[1:],
                "dataset_size": dataset_size + test_size,
                "t0": float(ts[0]),
                "t1": float(ts[-1]),
                "ts_shape": ts.shape,
            }

            jnp.save(
                saving_path + "/ys.npy",
                ys,
            )
            jnp.save(
                saving_path + "/ts.npy",
                ts,
            )

            with open(
                saving_path + "/data_generation_hyper_parameters.json", "w"
            ) as file:
                json.dump([dic_data], file)

        if any(file.startswith("test_indices") for file in os.listdir(saving_path)):
            print(
                "Not creating additional split indices for other noise levels since they already exist"
            )
            pass
        else:
            print("Creating train/test split indices")
            _, split_idx_key = jrandom.split(data_key, 2)
            ys = jnp.load(saving_path + "/ys.npy")
            check_split_indices(
                ys, dataset_size, saving_path, args.nb_split_indices, split_idx_key
            )

        ####### GENERATING STATE DEPENDENT Extrapolate DATASET AND OTHER HISTORY #######
        if any(file.endswith("extrapolate.npy") for file in os.listdir(saving_path)):
            print("Already have extrapolation and step history datasets")
        else:
            print("Creating Extrapolation and other history datasets")
            data_key = jrandom.PRNGKey(np.random.randint(0, 1000))
            tf, nb_steps, dataset_size, test_size = (
                10.0,
                150,
                256,
                32,
            )
            ts = jnp.linspace(0.0, tf, nb_steps)
            delays = Delays(
                delays=[lambda t, y, args: 1 / 2 * jnp.cos(y[0])],
                initial_discontinuities=jnp.array([0.0]),
                max_discontinuities=2,
                recurrent_checking=False,
                rtol=10e-3,
                atol=10e-6,
            )

            _, extrapolate_data_key = jrandom.split(data_key, 2)
            y0_extrapolate = jrandom.uniform(
                extrapolate_data_key, (test_size, 1), minval=-1.0, maxval=0.1
            )
            ys_extrapolate = jax.vmap(state_dependent_dataset, (None, None, 0))(
                ts, delays, y0_extrapolate
            )
            jnp.save(saving_path + "/ys_extrapolate.npy", ys_extrapolate)

            _, other_history_key = jrandom.split(extrapolate_data_key)
            y0_other_history = jrandom.uniform(
                other_history_key, (test_size, 2), minval=-1.0, maxval=1.0
            )
            _, other_ts_history_key = jrandom.split(other_history_key)
            ts_history = jrandom.uniform(
                other_ts_history_key, (test_size, 1), minval=-1 / 2, maxval=0.0
            )

            ys_other_history_test = jax.vmap(
                state_dependent_dataset_different, (None, None, 0, 0)
            )(ts, delays, y0_other_history, ts_history)

            # This corresponds to the 2 values of the step fn
            jnp.save(saving_path + "/y0_other_history.npy", y0_other_history)
            # This corresponds to the time t where the step fn changes
            jnp.save(saving_path + "/ts_history.npy", ts_history)
            # This is the next dynamics with this new history function
            jnp.save(saving_path + "/ys_other_history_test.npy", ys_other_history_test)
