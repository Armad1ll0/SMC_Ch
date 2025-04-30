import sys
sys.path.append('../')
import jax.numpy as jnp 
import numpy as np
from jax import random, vmap 
from tqdm import tqdm 
from mcmc.hmc import leapfrog 
import torch 

def update_weight(logw, x_new, x, v_new, v, p, q):
    p_logprob_x_new = vmap(p.log_prob, in_axes = (0))(x_new)
    p_logprob_x = vmap(p.log_prob, in_axes = (0))(x)
    q_logprob_v_new = vmap(q.log_prob, in_axes = (0))(-v_new)
    q_logprob_v = vmap(q.log_prob, in_axes = (0))(v)
    logw_new = (logw + 
                p_logprob_x_new - 
                p_logprob_x + 
                q_logprob_v_new - 
                q_logprob_v)
    return logw_new

def normalise_weights(logw):
    # w = jnp.exp(logw)
    # wn = w / jnp.sum(w)
    # return wn
    max_weight = jnp.max(logw)
    exp_weights = jnp.exp(logw - max_weight)
    return exp_weights / jnp.sum(exp_weights)

def multinomial_resampling(wn, N):
    # Need to use a jax function instead of this when doing production code 
    # indices = jax.scipy.stats.multinomial.pmf(n=N, p=wn, shape=(N,))
    wn = np.asarray(wn)
    wn = torch.tensor(wn)
    idx = torch.multinomial(torch.t(wn), N, replacement=True)
    return idx

def estimate_values(x, wn, D, N):
    mean = jnp.transpose(wn) @ x
    x_shift = x - mean
    var = jnp.transpose(wn) @ jnp.square(x_shift)
    return mean, var

def smc_hmc_base(N, D, K, h, p, q_0, q, num_steps, key, x_0 = None, verbose = False):
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
        
        if i % 50 == 0 and verbose == True: 
            print(f'\nWe are currently on iteration {i}')
            print(f'Our current mean and variance respectively estimate is {mean}, {var}')
        
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
        x_new, v_new = vmap(leapfrog, in_axes=(0, 0, None, None, 0, None))(x, v, h, num_steps, grad_x, p)
        # Initial gradient calculation plus one per leapfrog step (*N)
        num_grad_evals.append(x.shape[0]*(num_steps+1))
        
        # update weights 
        logw_new = update_weight(logw, x_new, x, v_new, v, p, q)
        
        x = x_new
        logw = logw_new
        samples.append(x)
        
    return mean_est, var_est, samples, Neffs, num_grad_evals