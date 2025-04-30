import jax 
import jax.numpy as jnp 
import jax.random as random 
from tqdm import tqdm 
from jax import vmap 
from mcmc.hmc import hmc_step
import optax 

def dual_averaging_step(log_e_bar, H_bar, alpha, m, h_0, delta = 0.651, kappa = 0.75, t_0 = 10, gamma = 0.05):
    mu = jnp.log(10*h_0)
    eta = 1/(m + t_0)
    H_bar = (1 - eta)*H_bar + eta*(delta - alpha)
    log_e = mu - jnp.sqrt(m)/gamma * H_bar
    eta = m**(-kappa)
    log_e_bar = (1 - eta)*log_e_bar + eta*log_e
    return H_bar, log_e, log_e_bar

def chees_criterion(x_new, x, v, alpha, traj_length): 
    x_new_centered = x_new - jnp.mean(x_new, axis = 0)
    x_centered = x - jnp.mean(x, axis = 0)
    chees_crit = (jnp.sum(x_new_centered**2, axis = 1) - jnp.sum(x_centered**2, axis = 1)) * jnp.sum(x_new_centered * v)
    traj_grads = traj_length * chees_crit
    weighted_grads = jnp.sum(alpha * traj_grads) / jnp.sum(alpha)
    return weighted_grads

def chees_hmc(N, D, K, h, p, key, x_0, lr = 0.1, decay_rate = 0.5):
    
    # initialise the samples 
    if x_0 != None: 
        x = x_0
    else: 
        x = random.multivariate_normal(key, jnp.zeros(D), jnp.eye(D), shape=(N,))
    
    # Store the samples 
    samples = []
    
    # Dual averaging constants 
    log_e = jnp.log(h)
    log_e_bar = 0.
    log_step_size_ma = 0.
    H_bar = 0.
    h_0 = h
    log_step_size_ma = 0.
    
    # Stuff needed for chees criterion
    trajectory_length = h_0 
    log_trajectory_length = jnp.log(trajectory_length)
    optimiser = optax.adam(learning_rate = lr)
    optim_state = optimiser.init(log_trajectory_length)
    log_trajectory_length_ma = 0
        
    # Initialise mean and variance arrays 
    for i in tqdm(range(1, K+1)):
        
        # split the key 
        key, subkey = random.split(key)
        subkeys = random.split(subkey, N)
        
        num_steps = int(jnp.ceil(h/trajectory_length))
        
        if i %50 == 0: 
            print(f'\nWe are currently on iteration {i}, our step size is {h} and our number of integration steps is {num_steps}')
        
        # run the hmc iteration 
        x_new, accept, energy, acceptance_prob, v = vmap(hmc_step, in_axes = (0, None, None, None, 0))(x, h, num_steps, p, subkeys)
        
        # calculate the harmonic mean for the dual averaging 
        acceptance_prob = jnp.clip(acceptance_prob, a_max = 1.0)
        harmonic_mean_alpha = 1 / jnp.mean(1 / acceptance_prob)

        # dual averaging step 
        H_bar_, log_e_, log_e_bar_ = dual_averaging_step(log_e_bar, H_bar, harmonic_mean_alpha, i, h_0)
        
        h_ = jnp.exp(log_e_)
        
        h, H_bar, log_e, log_e_bar = jax.lax.cond(
            jnp.isfinite(h_),
            lambda _: (h_, H_bar_, log_e_, log_e_bar_),
            lambda _: (h, H_bar, log_e, log_e_bar),
            None,
        )

        update_weight = i ** (-decay_rate)
        new_log_step_size_ma = (
            1.0 - update_weight
        ) * log_step_size_ma + update_weight * log_e
        
        # Update the log trajectory length             
        trajectory_gradient = chees_criterion(x_new, x, v, acceptance_prob, trajectory_length)

        log_trajectory_length = jnp.log(trajectory_length)
        updates, optim_state_ = optimiser.update(
            trajectory_gradient, optim_state, log_trajectory_length
        )
        
        log_trajectory_length_ = optax.apply_updates(log_trajectory_length, updates)
        new_log_trajectory_length, new_optim_state = jax.lax.cond(
            jnp.isfinite(
                jax.flatten_util.ravel_pytree(log_trajectory_length_)[0]
            ).all(),
            lambda _: (log_trajectory_length_, optim_state_),
            lambda _: (log_trajectory_length, optim_state),
            None,
        )
        
        new_log_trajectory_length_ma = (
            1.0 - update_weight
        ) * log_trajectory_length_ma + update_weight * new_log_trajectory_length
        new_trajectory_length = jnp.exp(new_log_trajectory_length)
        
        trajectory_length = new_trajectory_length
        
        
        # calculate the mean and variance 
        samples.append(x_new)
        x = x_new
        
    return samples

