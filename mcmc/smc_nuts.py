import sys 
sys.path.append('../')
from jax import vmap, random 
from tqdm import tqdm
import jax.numpy as jnp 
from mcmc.smc_hmc import normalise_weights, estimate_values, multinomial_resampling, update_weight
from mcmc.nuts_functions import generate_nuts_samples
from mcmc.nuts_vectorised import NUTS as NUTS_proposal

def smc_nuts(N, D, K, h, p, q_0, q, key, x_0 = None, verbose = False):
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
    NUTS = NUTS_proposal(N, D, p, h)
    
    
    # main sampling loop 
    for i in tqdm(range(1, K+1)):
        
        # Keys needed 
        key, subkey = random.split(key)
        subkeys = random.split(subkey, N)
        
        # normalise weights 
        wn = normalise_weights(logw)
        
        # estimate quantities needed
        mean, var = estimate_values(x, wn, D, N)
        mean_est.append(mean)
        var_est.append(var)
        
        if i % 10 == 0 and verbose == True: 
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
        
        # sample the momentum we need
        v = random.normal(key, shape=x.shape)
        
        # propogate samples 
        x_new = jnp.zeros_like(x)
        v_new = jnp.zeros_like(v)
        iter_grad_evals = 0
        
        # Running the NUTS proposal 
        x_new, v_new , num_evals = NUTS.rvs(x, v, subkey)
        num_grad_evals.append(jnp.sum(num_evals))
        
        # update weights 
        logw_new = update_weight(logw, x_new, x, v_new, v, p, q)
        
        x = x_new
        logw = logw_new
        samples.append(x)
        
        
        
    return mean_est, var_est, samples, Neffs, num_grad_evals