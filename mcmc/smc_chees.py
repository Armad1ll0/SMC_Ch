import sys 
sys.path.append('../')
import jax 
from jax import vmap, random 
import jax.numpy as jnp 
import optax 
from tqdm import tqdm

from mcmc.smc_hmc import normalise_weights, estimate_values, multinomial_resampling, update_weight
from mcmc.hmc import leapfrog, leapfrog_chees
from mcmc.chees_hmc import dual_averaging_step, chees_criterion

def calc_alphas(x_new, x, v_new, v, p): 
    # MH acceptance criteria 
    proposed_energy = - p.log_prob(x_new) + 0.5 * jnp.dot(v_new, v_new)
    current_energy = - p.log_prob(x) + 0.5 * jnp.dot(v, v)
    
    # calculate alpha 
    energy_diff = jnp.abs(current_energy - proposed_energy)
    acceptance_prob = jnp.exp(current_energy - proposed_energy)
    
    return acceptance_prob, energy_diff

# @jax.jit
def smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, x_0 = None, lr = 0.1, decay_rate = 0.5, 
                  DA = True, adapt_length = True, max_steps = 500, verbose = False, burn_in = 0):
    # Initialisation 
    if x_0 != None: 
        x = x_0
    else: 
        x = random.multivariate_normal(key, jnp.zeros(D), jnp.eye(D), shape=(N,))
        
    # initialise mean and variance estimate arrays 
    mean_est = []
    var_est = []
    samples = []
    Neffs = []
    num_grad_evals = []
    
    # Dual averaging constants 
    log_e = jnp.log(h)
    log_e_bar = 0.
    log_step_size_ma = 0.
    H_bar = 0.
    h_0 = h
    log_step_size_ma = 0.
    # h = jnp.ones(N)*h
    
    # Stuff needed for chees criterion
    trajectory_length = h_0 
    log_trajectory_length = jnp.log(trajectory_length)
    optimiser = optax.adam(learning_rate = lr)
    optim_state = optimiser.init(log_trajectory_length)
    log_trajectory_length_ma = 0

    print('We are using the following random sequence', rn_seq)
    
    p_logpdf_x = vmap(p.log_prob, in_axes=(0))(x)
    q_logpdf_x = vmap(q_0.log_prob, in_axes=(0))(x)
    logw = p_logpdf_x - q_logpdf_x
    
    # main sampling loop 
    for i in tqdm(range(1, K+1)):

        # Keys needed 
        key, subkey = random.split(key)
        
        # normalise weights 
        wn = normalise_weights(logw)
        
        # estimate quantities needed
        mean, var = estimate_values(x, wn, D, N)
        mean_est.append(mean)
        var_est.append(var)
        adj_trajectory_length = trajectory_length*rn_seq[:, i-1]
        num_steps = jnp.ceil(h/adj_trajectory_length).astype(int).reshape(-1, 1)
        num_steps = jnp.clip(num_steps, a_min = 1, a_max = max_steps)
        max_steps_this_iter = int(jnp.max(num_steps))
        

        if i % 50 == 0 and verbose == True: 
            print(f'\nWe are currently on iteration {i}')
            print(f'Our current mean and variance respectively estimate is {mean}, {var}')
            # print(f'Our current step size and integration steps is respectively estimate is {h}, {num_steps}')
            print(f'Our moving average trajectory Length is now {int(jnp.ceil(h/jnp.exp(log_trajectory_length_ma)))}')

    
        
        # calculate effective sample size 
        Neff = 1/jnp.sum(jnp.square(wn))
        Neffs.append(Neff)
        
        # resample if needed 
        if Neff < N / 2:
            print('We have had to resample')
            print(Neff)
            indices = multinomial_resampling(wn, N)
            indices = indices.detach().numpy()
            x = x[indices]
            wn = jnp.ones(N)/N
            logw = jnp.log(wn)
        
        # calculate gradients needed for first leapfrog step 
        grad_x = vmap(p.log_prob_grad, in_axes = (0))(x)
        
        # sample the momentum we need
        v = random.normal(subkey, shape=x.shape)
        
        # propogate samples
        if rn_seq is not None: 
            x_new, v_new = vmap(leapfrog_chees, in_axes=(0, 0, None, 0, 0, None, None))(x, v, h, num_steps, grad_x, p.log_prob_grad, max_steps_this_iter)
        else:
            num_steps = 10
            x_new, v_new = vmap(leapfrog, in_axes=(0, 0, None, None, 0, None))(x, v, h, num_steps, grad_x, p)

        # print("x_new_all", x_new)

        # Initial gradient calculation (*N) plus one per leapfrog step from sum of num_steps 
        num_grad_evals.append(int(jnp.sum(num_steps)+x.shape[0]))
        
        # update weights 
        logw_new = update_weight(logw, x_new, x, v_new, v, p, q)
        
        # We have to simulate the 
        acceptance_prob, energy_diff = vmap(calc_alphas, in_axes=(0, 0, 0, 0, None))(x_new, x, v_new, v, p)
        is_divergent = jnp.isnan(energy_diff) | (energy_diff > 10000)
        
        # calculate the harmonic mean for the dual averaging 
        acceptance_prob = jnp.clip(acceptance_prob, a_max = 1.0)
        harmonic_mean_alpha = 1 / jnp.mean(1 / acceptance_prob, where=~is_divergent)
        
        weighting = i ** (-decay_rate)
        
        # TRY DOING STEP SIZE FOR EACH ONE AND SEE HOW IT WORKS. 
        # for i in range(N): 
        #     H_bar_, log_e_, log_e_bar_ = dual_averaging_step(log_e_bar, H_bar, harmonic_mean_alpha, i, h_0)
        
        if DA == True: 
            # dual averaging step 
            H_bar_, log_e_, log_e_bar_ = dual_averaging_step(log_e_bar, H_bar, harmonic_mean_alpha, i, h_0)
            
            h_ = jnp.exp(log_e_)
            
            h, H_bar, log_e, log_e_bar = jax.lax.cond(
                jnp.isfinite(h_),
                lambda _: (h_, H_bar_, log_e_, log_e_bar_),
                lambda _: (h, H_bar, log_e, log_e_bar),
                None,
            )
    
            new_log_step_size_ma = (
                1.0 - weighting
            ) * log_step_size_ma + weighting * log_e
            log_step_size_ma = new_log_step_size_ma
            h = jnp.exp(log_step_size_ma)
        
        if adapt_length == True:# and burn_in >= i:
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
            
            new_log_trajectory_length_ma = (1.0 - weighting) * log_trajectory_length_ma + \
                                            weighting * new_log_trajectory_length
            new_trajectory_length = jnp.exp(new_log_trajectory_length)
            
            trajectory_length = new_trajectory_length
            log_trajectory_length_ma = new_log_trajectory_length_ma
        
        x = x_new
        logw = logw_new
        samples.append(x)        
        
        
    return mean_est, var_est, samples, Neffs, num_grad_evals