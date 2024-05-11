import functools
import jax
import os
import diffrax
import jax.numpy as jnp
import jax.random as jrandom


def time_dependent_dataset(ts, delays, y0_sample):
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=lambda t: y0_sample,
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    return sol.ys


def state_dependent_dataset(ts, delays, y0_sample):
    def vector_field(t, y, args, *, history):
        alpha = (
            4
            + jnp.array([jnp.sin(t)])
            + jnp.array([jnp.sin(jnp.sqrt(2) * t)])
            + 1 / (1 + t**2)
        )
        beta = alpha - 4
        return -alpha * y + beta * history[0] ** 2 / (1 + history[0] ** 2) + beta

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=lambda t: y0_sample,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    return sol.ys


def diffusion_delay(ts, xs, delays, a_sample):
    def vf_diffusion_delay_in_time(t, y, args, *, history):
        D, r = args
        output = D * (jnp.roll(y, -1) - 2 * y + jnp.roll(y, 1)) / (
            0.01
        ) ** 2 + r * y * (1 - history[0])
        return output

    D, r = 0.01, 0.9
    fn_u0_x = lambda x: jnp.sin(jnp.pi * x)
    fn_u0_t = lambda u0_x, t: a_sample * jnp.exp(-0.01 * t) * u0_x
    u0 = fn_u0_x(xs)
    fn_u0_partial = functools.partial(fn_u0_t, u0)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf_diffusion_delay_in_time),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=lambda t: fn_u0_partial(t),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
        args=(D, r),
    )
    return sol.ys


def state_dependent_dataset_different(ts, delays, y0_other_history, ts_history):
    def vector_field(t, y, args, *, history):
        alpha = (
            4
            + jnp.array([jnp.sin(t)])
            + jnp.array([jnp.sin(jnp.sqrt(2) * t)])
            + 1 / (1 + t**2)
        )
        beta = alpha - 4
        return (
            -alpha * y + beta * history[0] ** 2 / (1 + history[0] ** 2) + beta
        )

    def history(t):
        return jnp.array(
            [
                jax.lax.cond(
                    t > ts_history[0],
                    lambda: y0_other_history[0],
                    lambda: y0_other_history[1],
                )
            ]
        )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=lambda t: history(t),
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    return sol.ys

def time_dependent_dataset_different(ts, delays, y0_other_history, ts_history):
    def vector_field(t, y, args, *, history):
        return y * (1 - history[0])

    def history(t):
        return jnp.array(
            [
                jax.lax.cond(
                    t > ts_history[0],
                    lambda: y0_other_history[0],
                    lambda: y0_other_history[1],
                )
            ]
        )

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Dopri5(),
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=lambda t: history(t),
        stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
        saveat=diffrax.SaveAt(ts=ts, dense=True),
        delays=delays,
    )

    return sol.ys



def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays)
    indices = jnp.arange(dataset_size)
    perm = jrandom.permutation(key, indices)
    for batch_nb in range(int(dataset_size // batch_size)):
        batch_perm = perm[batch_nb * batch_size : (batch_nb + 1) * batch_size]
        yield tuple(array[batch_perm] for array in arrays)



def check_split_indices(ys, trainset_size, saving_path, nb_split_indices, key):
    assert ys.shape[0] >= trainset_size
    for i in range(1, nb_split_indices + 1):
        _, key = jrandom.split(key, 2)
        if not os.path.exists(saving_path + f"/train_indices_{i}.pt"):
            indices = jnp.arange(ys.shape[0])
            indices = jax.random.permutation(key, indices)
            train_indices, test_indices = (
                indices[:trainset_size],
                indices[trainset_size:],
            )
            jnp.save(saving_path + f"/train_indices_{i}.npy", train_indices)
            jnp.save(saving_path + f"/test_indices_{i}.npy", test_indices)
        else:
            print(f"Already exist file : saving_path" + f"/train_indices_{i}.pt")
