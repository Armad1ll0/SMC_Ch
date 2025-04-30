import jax
import jax.numpy as jnp
import numpy as np
from jax import random, grad, vmap 
from tqdm import tqdm 
from jax import jit
from functools import partial


# Leapfrog integrator
def leapfrog_step(x, v, grad_x, step_size, p):
    v = v + 0.5 * step_size * grad_x
    x = x + step_size * v
    grad_x = p.log_prob_grad(x)
    v = v + 0.5 * step_size * grad_x
    return x, v, grad_x

# @partial(jit, static_argnums=(3, 4))
def leapfrog(x, v, h, num_steps, grad_x, p):
    for i in range(num_steps):
        x, v, grad_x = leapfrog_step(x, v, grad_x, h, p)
    return x, v

def leapfrog_chees(x, v, h, num_steps, grad_x, p_grad, max_steps):
    def step_fn(carry, i):
        x, v, grad_x = carry
        # Only update if i < num_steps (using mask)
        mask = i < num_steps
        v = jnp.where(mask, v + 0.5 * h * grad_x, v)
        x = jnp.where(mask, x + h * v, x)
        grad_x = p_grad(x)
        v = jnp.where(mask, v + 0.5 * h * grad_x, v)
        return (x, v, grad_x), None

    carry = (x, v, grad_x)
    for i in range(max_steps):
        carry, _ = step_fn(carry, i)
    x_final, v_final, _ = carry
    return x_final, v_final

def hmc_step(x, h, num_steps, p, key):
    # get the keys for the rng 
    key, subkey = random.split(key)
    
    # Sample momentum 
    v = random.normal(subkey, shape=x.shape)
    
    # calculate the gradient needed for first leapfrog step 
    grad_x = p.log_prob_grad(x)
    
    # run leapfrog 
    x_new, v_new = leapfrog(x, v, h, num_steps, grad_x, p)
    
    # MH acceptance criteria 
    proposed_energy = - p.log_prob(x_new) + 0.5 * jnp.dot(v_new, v_new)
    current_energy = - p.log_prob(x) + 0.5 * jnp.dot(v, v)
    
    # calculate alpha 
    energy_diff = jnp.abs(current_energy - proposed_energy)
    acceptance_prob = jnp.exp(current_energy - proposed_energy)
    u = random.uniform(key)
    accept = u < acceptance_prob
    
    # accept where we need to 
    return jnp.where(accept, x_new, x), accept, energy_diff, acceptance_prob, v

def hmc_base(N, D, K, h, p, num_steps, key, x_0):
    # initialise the samples 
    if x_0 != None: 
        x = x_0
    else: 
        x = random.multivariate_normal(key, jnp.zeros(D), jnp.eye(D), shape=(N,))
    
    # Store the samples 
    samples = []
        
    # Initialise mean and variance arrays 
    for i in tqdm(range(K)):
        
        # split the key 
        key, subkey = random.split(key)
        subkeys = random.split(subkey, N)
        
        # run the hmc iteration 
        x, accept, energy, accept_prob, v = vmap(hmc_step, in_axes = (0, None, None, None, 0))(x, h, num_steps, p, subkeys)
        
        # calculate the mean and variance 
        samples.append(x)
        
    return samples




