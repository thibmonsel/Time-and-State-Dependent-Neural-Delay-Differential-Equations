import functools
import time

import diffrax
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from dataset import dataloader
from diffrax import Delays
from matplotlib import pyplot as plt

name_dic = {
    "NeuralDDE": "dde",
    "ANODE": "anode",
    "NeuralODE": "ode",
    "LatentODE": "latent_ode",
    "Warm_Up_ANODE": "warmup_anode",
}

##### NODE ######
class Func3(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, activation, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )
    
    def __call__(self, t,y,args):
        return self.mlp(y)
    
class NeuralODE(eqx.Module):
    func: Func3

    def __init__(self, data_size, width_size, depth, activation, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func3(data_size, width_size, depth, activation, key=key)

    def __name__(self):
        return "NeuralODE"

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
        )
        return solution.ys, solution.stats["num_steps"]


####### ANODE ###### 

class LinearLayer(eqx.Module):
    mlp: eqx.nn.Linear

    def __init__(self, in_features, out_features, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.Linear(
            in_features=in_features, out_features=out_features, key=key
        )

    def __call__(self, y):
        return self.mlp(jnp.hstack([y]))
    
class ANODE(eqx.Module):
    func: Func3
    linear: LinearLayer
    augmented_dim: int

    def __init__(
        self,
        data_size,
        augmented_dim,
        width_size,
        depth,
        activation,
        *,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.func = Func3(
            data_size + augmented_dim, width_size, depth, activation, key=key
        )
        self.linear = LinearLayer(data_size + augmented_dim, data_size, key=key)
        self.augmented_dim = augmented_dim

    def __name__(self):
        return "ANODE"

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-3),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
        )

        return (
            solution.ys[:, : -self.augmented_dim],
            solution.stats["num_steps"],
        )



#### LATENT ODE ##### 

class FuncLatent(eqx.Module):
    scale: jnp.ndarray
    mlp: eqx.nn.MLP

    def __call__(self, t, y, args):
        return self.scale * self.mlp(y)


class LatentODE(eqx.Module):
    func: FuncLatent
    rnn_cell: eqx.nn.GRUCell

    hidden_to_latent: eqx.nn.Linear
    latent_to_hidden: eqx.nn.MLP
    hidden_to_data: eqx.nn.Linear

    hidden_size: int
    latent_size: int

    def __init__(
        self,
        data_size,
        hidden_size,
        latent_size,
        width_size,
        depth,
        key,
        **kwargs,
    ):
        super().__init__(**kwargs)

        mkey, gkey, hlkey, lhkey, hdkey = jrandom.split(key, 5)

        scale = jnp.ones(())
        mlp = eqx.nn.MLP(
            in_size=hidden_size,
            out_size=hidden_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jnn.tanh,
            key=mkey,
        )
        # This is f_theta
        self.func = FuncLatent(scale, mlp)
        self.rnn_cell = eqx.nn.GRUCell(data_size + 1, hidden_size, key=gkey)

        self.hidden_to_latent = eqx.nn.Linear(hidden_size, 2 * latent_size, key=hlkey)
        # outputNN
        self.latent_to_hidden = eqx.nn.MLP(
            latent_size,
            hidden_size,
            width_size=width_size,
            depth=depth,
            key=lhkey,
        )
        # With its linear layer
        self.hidden_to_data = eqx.nn.Linear(hidden_size, data_size, key=hdkey)

        self.hidden_size = hidden_size
        self.latent_size = latent_size

    def __name__(self):
        return "LatentODE"

    # Encoder of the VAE
    def _latent(self, ts, ys, key):
        data = jnp.concatenate([ts[:, None], ys], axis=1)
        # initializes h0
        hidden = jnp.zeros((self.hidden_size,))
        for data_i in reversed(data):
            hidden = self.rnn_cell(data_i, hidden)
        context = self.hidden_to_latent(hidden)
        mean, logstd = context[: self.latent_size], context[self.latent_size :]
        std = jnp.exp(logstd)
        latent = mean + jrandom.normal(key, (self.latent_size,)) * std
        return latent, mean, std

    # Decoder of the VAE
    def _sample(self, ts, latent):
        y0 = self.latent_to_hidden(latent)
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            ts[0],
            ts[-1],
            ts[1] - ts[0],
            y0,
            saveat=diffrax.SaveAt(ts=ts),
        )
        return jax.vmap(self.hidden_to_data)(sol.ys), sol.stats["num_steps"]

    @staticmethod
    def _loss(ys, pred_ys, mean, std):
        # -log p_θ with Gaussian p_θ
        reconstruction_loss = 0.5 * jnp.sum((ys - pred_ys) ** 2)
        # KL(N(mean, std^2) || N(0, 1))
        variational_loss = 0.5 * jnp.sum(mean**2 + std**2 - 2 * jnp.log(std) - 1)
        return reconstruction_loss + variational_loss

    @staticmethod
    def _reconstruction_loss(ys, pred_ys):
        return jnp.mean((pred_ys - ys) ** 2)

    # Run both encoder and decoder during training.
    def train(self, ts, ys, key):
        # z0 is latent and mean, std a used for variational loss
        latent, mean, std = self._latent(ts, ys, key)
        pred_ys, stats = self._sample(ts, latent)
        return self._loss(ys, pred_ys, mean, std), stats

    def sample_deterministic(self, ts, ys, key):
        latent, _, _ = self._latent(ts, ys, key)
        pred_ys, stats = self._sample(ts, latent)
        return pred_ys, stats

    # Run just the decoder during inference.
    def sample(self, ts, key):
        latent, stats = jrandom.normal(key, (self.latent_size,))
        return self._sample(ts, latent), stats



######## STATE DEPENDENT MODEL ########

class Func_Time(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, nb_delays, data_size, width_size, depth, activation, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size=(nb_delays + 1) * data_size + 1,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, t, y, args, *, history):
        return self.mlp(jnp.hstack([y, *history, t]))


class NeuralDDEWithTime(eqx.Module):
    func: Func_Time
    delays: Delays

    def __init__(
        self, data_size, width_size, depth, activation, delays, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func_Time(len(delays.delays), data_size, width_size, depth, activation, key=key)
        self.delays = delays

    def __name__(self):
        return "NeuralDDE"

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=lambda t: y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            delays=self.delays,
            made_jump=True,
        )
        return solution.ys, solution.stats["num_steps"]


######## TIME DEPENDENT MODEL ########

class Func(eqx.Module):
    mlp: eqx.nn.MLP

    def __init__(self, nb_delays, data_size, width_size, depth, activation, *, key, **kwargs):
        super().__init__(**kwargs)
        self.mlp = eqx.nn.MLP(
            in_size= (nb_delays+1) * data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=activation,
            key=key,
        )

    def __call__(self, t, y, args, *, history):
        return self.mlp(jnp.hstack([y, *history]))

class NeuralDDE(eqx.Module):
    func: Func
    delays: Delays

    def __init__(
        self, data_size, width_size, depth, activation, delays, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func(len(delays.delays), data_size, width_size, depth, activation, key=key)
        self.delays = delays

    def __name__(self):
        return "NeuralDDE"

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Bosh3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=lambda t: y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            max_steps=500,
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            delays=self.delays,
            made_jump=True,
        )
        return solution.ys, solution.stats["num_steps"]


######## DIFFUSION MODEL MODEL ########

class PDENeuralDDE(eqx.Module):
    func: Func
    delays: Delays
    def __init__(
        self, data_size, width_size, depth, activation, delays, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func(len(delays.delays), data_size, width_size, depth, activation, key=key)
        self.delays = delays

    def __name__(self):
        return "NeuralDDE"

    def __call__(self, ts, xs, a_sample):

        fn_u0_x = lambda x: jnp.sin(jnp.pi * x)
        fn_u0_t = lambda u0_x, t: a_sample * jnp.exp(-0.1*t) * u0_x
        u0 = fn_u0_x(xs)
        fn_u0_partial = functools.partial(fn_u0_t, u0)

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=lambda t: fn_u0_partial(t),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-6),
            max_steps=200,
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            delays=self.delays,
            made_jump=True,
        )
        return solution.ys

class PDEANODE(eqx.Module):
    augmented_dim: int
    func: Func3

    def __init__(self, augmented_dim, data_size, width_size, depth, activation,*,key,**kwargs):
        super().__init__(**kwargs)
        self.func = Func3(augmented_dim + data_size, width_size, depth, activation, key=key)
        self.augmented_dim = augmented_dim
        
    def __name__(self):
        return "ANODE"

    def __call__(self, ts, xs, a_sample):
        fn_u0_x = lambda x: jnp.sin(jnp.pi * x)
        fn_u0_t = lambda u0_x, t: a_sample * jnp.exp(-0.1*t) * u0_x
        u0 = fn_u0_x(xs)
        fn_u0_partial = functools.partial(fn_u0_t, u0)
        y0 = jnp.concatenate(
            (fn_u0_partial(ts[0]), jnp.zeros((self.augmented_dim,))),
            axis=-1,
        )
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Bosh3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-3),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
        )
        return solution.ys[:, : -self.augmented_dim] 


class PDENeuralODE(eqx.Module):
    func: Func3

    def __init__(
        self,data_size, width_size, depth, activation, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func3(data_size, width_size, depth, activation, key=key)

    def __name__(self):
        return "NeuralODE"

    def __call__(self, ts, xs, a_sample):
        fn_u0_x = lambda x: jnp.sin(jnp.pi * x)
        fn_u0_t = lambda u0_x, t: a_sample * jnp.exp(-0.1*t) * u0_x
        u0 = fn_u0_x(xs)
        fn_u0_partial = functools.partial(fn_u0_t, u0)
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Bosh3(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=fn_u0_partial(ts[0]),
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
        )
        return solution.ys



#### OTHER STEP HISTORY FN STATE DEPENDENT ####

class NeuralDDEWithTimeModif(eqx.Module):
    func: Func_Time
    delays: Delays

    def __init__(
        self, data_size, width_size, depth, activation, delays, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func_Time(len(delays.delays), data_size, width_size, depth, activation, key=key)
        self.delays = delays

    def __name__(self):
        return "NeuralDDE"

    def __call__(self, ts, y0_other_history, ts_history):
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

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=lambda t: history(t),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            delays=self.delays,
            made_jump=True,
        )
        return solution.ys, solution.stats["num_steps"]

#### OTHER STEP HISTORY FN TIME DEPENDENT ####

class NeuralDDE_Modif(eqx.Module):
    func: Func
    delays: Delays

    def __init__(
        self, data_size, width_size, depth, activation, delays, *, key, **kwargs
    ):
        super().__init__(**kwargs)
        self.func = Func(len(delays.delays), data_size, width_size, depth, activation, key=key)
        self.delays = delays

    def __name__(self):
        return "NeuralDDE"

    def __call__(self, ts, y0_other_history, ts_history):
        def history(t):
            if y0_other_history[0].size > 1:
                return jax.lax.cond(
                    t > ts_history[0],
                    lambda: y0_other_history[0],
                    lambda: y0_other_history[1],
                )
            else:
                return jnp.array(
                    [
                        jax.lax.cond(
                            t > ts_history[0],
                            lambda: y0_other_history[0],
                            lambda: y0_other_history[1],
                        )
                    ]
                )

        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Dopri5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=lambda t: history(t),
            stepsize_controller=diffrax.PIDController(rtol=1e-6, atol=1e-9),
            max_steps=2**10,
            saveat=diffrax.SaveAt(ts=ts, dense=True),
            delays=self.delays,
            made_jump=True,
        )
        return solution.ys, solution.stats["num_steps"]
    
@eqx.filter_value_and_grad(has_aux=True)
def grad_loss(model, ti, yi):
    if model.__name__() == "ANODE":
        y0 = jnp.concatenate(
            (yi[:, 0], jnp.zeros((yi[:, 0].shape[0], model.augmented_dim))),
            axis=-1,
        )
        y_pred, stats = jax.vmap(model, (None, 0))(ti, y0)
        return jnp.mean((yi - y_pred) ** 2), stats
    else:
        y_pred, stats = jax.vmap(model, (None, 0))(ti, yi[:, 0])
        return jnp.mean((yi - y_pred) ** 2), stats


@eqx.filter_jit
def make_step(ti, yi, model, opt_state, optim):
    loss, grads = grad_loss(model, ti, yi)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state


@eqx.filter_value_and_grad
def pde_grad_loss(model, ts, xs, a_sample, ys):
    y_pred = jax.vmap(model, (None, None, 0))(ts, xs, a_sample)
    return jnp.mean((ys - y_pred) ** 2)


@eqx.filter_jit
def pde_make_step(ts, xs, ys, a_sample, model, opt_state, optim):
    loss, grads = pde_grad_loss(model, ts, xs, a_sample, ys)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    return loss, model, opt_state



def fit(
    ts,
    ys,
    ys_test,
    model,
    batch_size,
    default_dir,
    key,
    lr_strategy,
    epoch_strategy,
    length_strategy,
):
    loss_per_step, test_loss, nfe, model_name = [], [], [], model.__name__()
    dataset_size, length_size, _ = ys.shape
    for lr, max_epoch, length in zip(lr_strategy, epoch_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length_size * length)]
        _ys = ys[:, : int(length_size * length)]
        _ys_test = ys_test[:, : int(length_size * length)]
        for epoch in range(max_epoch):
            start = time.time()
            nfe_epoch = []
            for (yi,) in dataloader((_ys,), batch_size, key=key):
                start = time.time()
                value, model, opt_state = make_step(_ts, yi, model, opt_state, optim)
                loss_value, num_acc = value
                nfe_epoch.append(jnp.mean(num_acc))
                if model_name == "ANODE":
                    y0_test = jnp.concatenate(
                        (
                            _ys_test[:, 0],
                            jnp.zeros(
                                (
                                    _ys_test.shape[0],
                                    model.augmented_dim,
                                ),
                            ),
                        ),
                        axis=1,
                    )
                    ytest_pred, _ = jax.lax.stop_gradient(
                        jax.vmap(model, (None, 0))(_ts, y0_test)
                    )
                else:
                    ytest_pred, _ = jax.lax.stop_gradient(
                        jax.vmap(model, (None, 0))(_ts, _ys_test[:, 0])
                    )
                test_value_loss = jax.lax.stop_gradient(
                    jnp.mean((_ys_test - ytest_pred) ** 2)
                )
                test_loss.append(test_value_loss)
                loss_per_step.append(loss_value)
                end = time.time()
                print(
                    f"Epoch: {epoch}, Loss: {loss_value} | Test Loss: {test_value_loss}, Computation time: {end - start}"
                )
                end = time.time()

                if epoch % (max_epoch / 10) == 0 or epoch == 0:
                    view_key, _ = jax.random.split(key, 2)
                    rdx_idx = jax.random.randint(view_key, (), 0, dataset_size)

                    plt.plot(loss_per_step)
                    plt.yscale("log")
                    plt.title("Loss curve")
                    plt.xlabel("Steps")
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/"
                        + r"loss_curve.png"
                    )
                    plt.close()

                    plt.plot(test_loss)
                    plt.yscale("log")
                    plt.title("Test Loss curve")
                    plt.xlabel("Steps")
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/"
                        + r"test_loss_curve.png"
                    )
                    plt.close()

                    plt.plot(_ts, _ys[rdx_idx], label="Real subset")
                    if model_name == "ANODE":
                        y0 = jnp.concatenate(
                            (
                                ys[rdx_idx, 0],
                                jnp.zeros(
                                    model.augmented_dim,
                                ),
                            )
                        )
                        model_y, _ = model(_ts, y0)
                    else:
                        model_y, _ = model(_ts, ys[rdx_idx, 0])

                    plt.plot(_ts, model_y, "--", label="Model")
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/training/{}_length_{}_step.png".format(length, epoch)
                    )
                    plt.close()
            nfe.append(sum(nfe_epoch) / len(nfe_epoch))

    if model_name == "ANODE":
        y0 = jnp.concatenate(
            (
                ys[-3, 0],
                jnp.zeros(
                    model.augmented_dim,
                ),
            )
        )
        model_y, _ = model(ts, y0)
    else:
        model_y, _ = model(ts, ys[-3, 0])

    plt.plot(ts, ys[-3, :], label="Real")
    plt.plot(ts, model_y, "--", label="Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(default_dir + "/" + name_dic[model_name] + "/testing.png")
    plt.close()

    eqx.tree_serialise_leaves(
        default_dir + "/" + name_dic[model_name] + "/last.eqx", model
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/loss_array.npy",
        jnp.asarray(loss_per_step),
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/test_loss_array.npy",
        jnp.asarray(test_loss),
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/nfe_fwd.npy",
        jnp.asarray(nfe),
    )

    plt.scatter(jnp.arange(len(nfe)), nfe, c=nfe)
    plt.title("NFEs")
    plt.xlabel("Epochs")
    plt.savefig(default_dir + "/" + name_dic[model_name] + "/" + r"nfes.png")
    plt.close()
    return loss_per_step, nfe, model


def pde_fit(
    ts,
    xs,
    ys,
    ys_test,
    a_sample,
    a_sample_test,
    model,
    batch_size,
    default_dir,
    key,
    lr_strategy,
    epoch_strategy,
    length_strategy,
):
    loss_per_step, test_loss, model_name = [], [], model.__name__()
    dataset_size, length_size, _ = ys.shape
    for lr, max_epoch, length in zip(lr_strategy, epoch_strategy, length_strategy):
        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
        _ys = ys[:, : int(length_size * length)]
        _ys_test = ys_test[:, : int(length_size * length)]
        for epoch in range(max_epoch):
            start = time.time()
            for (yi, a_sample_batch) in dataloader(
                (_ys, a_sample), batch_size, key=key
            ):
                start = time.time()
                value, model, opt_state = pde_make_step(
                    ts, xs, yi, a_sample_batch, model, opt_state, optim
                )

                if model_name == "ANODE":
                    y0_test = jnp.concatenate(
                        (
                            _ys_test[:, 0],
                            jnp.zeros(
                                (
                                    _ys_test.shape[0],
                                    model.augmented_dim,
                                ),
                            ),
                        ),
                        axis=1,
                    )
                    ytest_pred = jax.lax.stop_gradient(
                        jax.vmap(model, (None, None, 0))(ts, xs, a_sample_test)
                    )
                else:
                    ytest_pred = jax.lax.stop_gradient(
                        jax.vmap(model, (None, None, 0))(ts, xs, a_sample_test)
                    )
                test_value_loss = jax.lax.stop_gradient(
                    jnp.mean((_ys_test - ytest_pred) ** 2)
                )
                test_loss.append(test_value_loss)
                loss_per_step.append(value)
                end = time.time()
                print(
                    f"Epoch: {epoch}, Loss: {value} | Test Loss: {test_value_loss}, Computation time: {end - start}"
                )
                end = time.time()

                if epoch % (max_epoch / 20) == 0 or epoch == 0:
                    view_key, _ = jax.random.split(key, 2)
                    rdx_idx = jax.random.randint(view_key, (), 0, dataset_size)

                    plt.plot(loss_per_step)
                    plt.yscale("log")
                    plt.title("Loss curve")
                    plt.xlabel("Steps")
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/"
                        + r"loss_curve.png"
                    )
                    plt.close()

                    plt.plot(test_loss)
                    plt.yscale("log")
                    plt.title("Test Loss curve")
                    plt.xlabel("Steps")
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/"
                        + r"test_loss_curve.png"
                    )
                    plt.close()

                    plt.imshow(yi[rdx_idx], aspect="auto", origin="lower")
                    plt.colorbar()
                    plt.xlabel("x")
                    plt.ylabel("t")
                    plt.tight_layout()
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/training/gt_{}_length_{}_step.png".format(length, epoch)
                    )
                    plt.close()
                    if model_name == "ANODE":
                        y0 = jnp.concatenate(
                            (
                                ys[rdx_idx, 0],
                                jnp.zeros(
                                    model.augmented_dim,
                                ),
                            )
                        )
                        model_y = model(ts, xs, a_sample_batch[rdx_idx])
                    else:
                        model_y = model(ts, xs, a_sample_batch[rdx_idx])

                    plt.imshow(model_y, aspect="auto", origin="lower")
                    plt.tight_layout()
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/training/pred_{}_length_{}_step.png".format(length, epoch)
                    )
                    plt.close()

                    plt.imshow(
                        jnp.abs(model_y - yi[rdx_idx]), aspect="auto", origin="lower"
                    )
                    plt.tight_layout()
                    plt.savefig(
                        default_dir
                        + "/"
                        + name_dic[model_name]
                        + "/training/diff_{}_length_{}_step.png".format(length, epoch)
                    )
                    plt.close()

    eqx.tree_serialise_leaves(
        default_dir + "/" + name_dic[model_name] + "/last.eqx", model
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/loss_array.npy",
        jnp.asarray(loss_per_step),
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/test_loss_array.npy",
        jnp.asarray(test_loss),
    )

    return loss_per_step, model


def fit_latent(
    ts,
    ys,
    ys_test,
    model,
    batch_size,
    default_dir,
    key,
    lr_strategy,
    epoch_strategy,
    length_strategy,
):
    @eqx.filter_value_and_grad(has_aux=True)
    def loss(model, ts_i, ys_i, key_i):
        batch_size = ys_i.shape[0]
        key_i = jrandom.split(key_i, batch_size)
        loss, stats = jax.vmap(model.train, (None, 0, 0))(ts_i, ys_i, key_i)
        return jnp.mean(loss), stats

    @eqx.filter_jit
    def make_step(model, opt_state, ts_i, ys_i, key_i):
        value, grads = loss(model, ts_i, ys_i, key_i)
        key_i = jrandom.split(key_i, 1)[0]
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return value, model, opt_state, key_i

    train_key, sample_key = jrandom.split(key, 2)
    loss_per_step, test_loss, model_name, nfe, step = (
        [],
        [],
        model.__name__(),
        [],
        0,
    )
    dataset_size, _, _ = ys.shape
    lr, tot_epochs = lr_strategy[0], sum(epoch_strategy)
    optim = optax.adabelief(lr)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))
    for epoch in range(tot_epochs):
        start = time.time()
        nfe_epoch = []
        for (yi,) in dataloader((ys,), batch_size, key=key):
            step += 1
            start = time.time()
            value, model, opt_state, train_key = make_step(
                model, opt_state, ts, yi, train_key
            )
            loss_value, num_acc = value
            nfe_epoch.append(jnp.mean(num_acc))
            end = time.time()
            ytest_pred, _ = jax.lax.stop_gradient(
                jax.vmap(model.sample_deterministic, (None, 0, None))(
                    ts, ys_test, sample_key
                )
            )
            test_value_loss = jax.lax.stop_gradient(
                model._reconstruction_loss(ytest_pred, ys_test)
            )
            test_loss.append(test_value_loss)
            loss_per_step.append(loss_value)
            end = time.time()
            print(
                f"Step: {step}, Epoch : {epoch}, Loss: {loss_value} | Test Loss: {test_value_loss}, Computation time: {end - start}"
            )
            end = time.time()
            if epoch % (tot_epochs / 10) == 0 or epoch == 0:
                view_key, _ = jax.random.split(key, 2)
                rdx_idx = jax.random.randint(view_key, (), 0, dataset_size)

                plt.plot(loss_per_step)
                plt.title("Loss curve")
                plt.xlabel("Steps")
                plt.savefig(
                    default_dir + "/" + name_dic[model_name] + "/" + r"loss_curve.png"
                )
                plt.close()

                plt.plot(test_loss)
                plt.title("Test Loss curve")
                plt.xlabel("Steps")
                plt.savefig(
                    default_dir
                    + "/"
                    + name_dic[model_name]
                    + "/"
                    + r"test_loss_curve.png"
                )
                plt.close()

                plt.plot(ts, ys[rdx_idx], label="Real subset")
                model_y, _ = model.sample_deterministic(ts, ys[rdx_idx], sample_key)

                plt.plot(ts, model_y, "--", label="Model")
                plt.legend()
                plt.tight_layout()
                plt.savefig(
                    default_dir
                    + "/"
                    + name_dic[model_name]
                    + "/training/{}_epoch.png".format(epoch)
                )
                plt.close()
        nfe.append(sum(nfe_epoch) / len(nfe_epoch))
    eqx.tree_serialise_leaves(
        default_dir + "/" + name_dic[model_name] + "/last.eqx", model
    )

    model_y, _ = model.sample_deterministic(ts, ys[-3], sample_key)
    plt.plot(ts, ys[-3, :], label="Real")
    plt.plot(ts, model_y, "--", label="Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(default_dir + "/" + name_dic[model_name] + "/testing.png")
    plt.close()

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/loss_array.npy",
        jnp.asarray(loss_per_step),
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/test_loss_array.npy",
        jnp.asarray(test_loss),
    )

    jnp.save(
        default_dir + "/" + name_dic[model_name] + "/nfe_fwd.npy",
        jnp.asarray(nfe),
    )

    plt.scatter(jnp.arange(len(nfe)), nfe, c=nfe)
    plt.title("NFEs")
    plt.xlabel("Epochs")
    plt.savefig(default_dir + "/" + name_dic[model_name] + "/" + r"nfes.png")
    plt.close()

    return loss_per_step
