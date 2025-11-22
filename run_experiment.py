import jax.numpy as jnp 
from jax import random
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats.qmc import Halton
from scipy.stats.qmc import Sobol
from mcmc.smc_chees import smc_hmc_chees
from mcmc.smc_hmc import smc_hmc_base
from mcmc.smc_nuts import smc_nuts
from utils.distributions import MultivariateNormal
from utils.save_load_funcs import save_data
from utils.targets import BananaDistribution, random_orthogonal_matrix, create_gaussian, LogRegTarget
from quasi_random_generators import golden_ratio, primes_seq, equidistant

import time 

#%%
def run_experiment(num_runs, N, D, K, h, num_steps, p, q_0, q, target_name, burn_in = 0): 
    for i in range(0, num_runs):
        key = random.PRNGKey(i)
        methods = ['NUTS', 'no_jitter', 
                   'Nd-ChEES_hal', 'Nd-ChEES-inv-hal', '1d-ChEES-hal', 
                   '1d-ChEES-gr', '1d-ChEES-uni', 'Nd-ChEES-primes', 
                   'Nd-ChEES-inv-primes', 'Nd-ChEES-equi', 'Nd-ChEES-ofset-equi', 
                   'Nd-Sobol', 'Nd-Sobol_inv', '1D-Sobol']

        for method in methods:
            start = time.time()
            print(f'\nWe are currently on run {i+1}/{num_runs}, Method: {method}, Target: {target_name}')

            
            # Haltons sequence
            if method == 'Nd-ChEES_hal':
                halton = Halton(K, scramble=False)
                halton.fast_forward(1)
                rn_seq = halton.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                        
            elif method == 'Nd-Sobol':
                sobol = Sobol(K, scramble=False)
                sobol.fast_forward(1)
                rn_seq = sobol.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq,
                                                                            DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == 'Nd-Sobol_inv':
                sobol = Sobol(K, scramble=False)
                sobol.fast_forward(1)
                rn_seq = sobol.random(N)
                rn_seq = rn_seq[:, ::-1]
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq,
                                                                            DA = False, adapt_length = True, burn_in=burn_in)

            elif method == '1D-Sobol':
                sobol = Sobol(1, scramble=False)
                sobol.fast_forward(1)
                rn_seq = sobol.random(N*K).reshape(N, K)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq,
                                                                            DA = False, adapt_length = True, burn_in=burn_in)    
              
            elif method == 'no_jitter':
                rn_seq = np.ones((N, K))
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)    
            elif method == 'Nd-ChEES-inv-hal':
                halton = Halton(K, scramble=False)
                halton.fast_forward(1)
                rn_seq = halton.random(N)
                rn_seq = rn_seq[:, ::-1]
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == '1d-ChEES-hal':
                halton = Halton(1, scramble=False)
                halton.fast_forward(1)
                rn_seq = halton.random(N*K).reshape(N, K)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == '1d-ChEES-gr':
                gr = golden_ratio(K)
                rn_seq = gr.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == '1d-ChEES-uni':
                rn_seq = np.random.uniform(size=(N, K))
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == 'Nd-ChEES-primes':
                primes = primes_seq(K)
                rn_seq = primes.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == 'Nd-ChEES-inv-primes':
                primes = primes_seq(K)
                rn_seq = primes.random(N)
                rn_seq = rn_seq[:, ::-1]
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == 'Nd-ChEES-equi':
                equi = equidistant(K)
                rn_seq = equi.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                
            elif method == 'Nd-ChEES-ofset-equi':
                equi = equidistant(K, offset=True)
                rn_seq = equi.random(N)
                print("rn_seq shape:", rn_seq.shape)
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_chees(N, D, K, h, p, q_0, q, key, rn_seq, 
                                                                          DA = False, adapt_length = True, burn_in=burn_in)
                                
            elif method == 'NUTS':
                mean, var, samples, Neffs, num_grad_evals = smc_nuts(N, D, K, h, p, q_0, q, key)
            elif method == 'HMC':
                mean, var, samples, Neffs, num_grad_evals = smc_hmc_base(N, D, K, h, p, q_0, q, num_steps, key)

            end = time.time()

            print(f'Run {i+1}/{num_runs}, Method: {method}, Target: {target_name}, Time taken: {end-start:.2f}s')

            # Remove non-burn in samples
            mean, var, samples, Neffs, num_grad_evals = mean[burn_in:], var[burn_in:], samples[burn_in:], Neffs[burn_in:], num_grad_evals[burn_in:]
            
            mean = jnp.array(mean)
            var = jnp.array(var)
            samples = jnp.array(samples)

            print("parameters:", "N", N, "D", D, "K", K, "h", h, "num_steps", num_steps, "target_name", target_name, "burn_in", burn_in)
            folder_path = f'results/{target_name}/{method}'
            print("Saving data to", folder_path)
            file_names = [f'mean_estimates_run_{i+1}', f'variance_estimates_run_{i+1}', f'samples_run_{i+1}', f'Neffs_run_{i+1}', f'num_grad_evals_run_{i+1}']
            save_data(folder_path, file_names, mean, var, samples, Neffs, num_grad_evals)


#%%
num_runs = 10
N = 1000
K = 200
burn_in = 100
num_steps = 5

#%%
# Banana distribution experiments 
def run_banana(num_runs, N, K, h, num_steps):
    p = BananaDistribution()
    D = 2
    q_0 = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    q = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    
    run_experiment(num_runs, N, D, K, h, num_steps, p, q_0, q, target_name = f'new_banana_h_{h}_N_{N}', burn_in=burn_in)

#%%
def run_gauss(num_runs, N, K, h, num_steps):
    true_mean = jnp.array([-4., -2., 0., 2., 4.])  # Mean of the multivariate Gaussian
    true_var = jnp.array([1, 1.5, 2., 2.5, 3.])  # Covariance matrix
    D = len(true_mean)
    p = MultivariateNormal(true_mean, true_var)
    q_0 = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    q = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    
    run_experiment(num_runs, N, D, K, h, num_steps, p, q_0, q, target_name = f'gauss_h_{h}_N_{N}', burn_in=burn_in)


#%%
def run_ill_conditioned_gauss(num_runs, N, K, h, num_steps):
    D = 100
    key = random.PRNGKey(0)
    true_var = create_gaussian(key, D=100)
    true_mean = jnp.zeros(100)
    p = MultivariateNormal(true_mean, 0, true_var)
    q_0 = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    q = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    
    run_experiment(num_runs, N, D, K, h, num_steps, p, q_0, q, target_name = f'ill_conditioned_gauss_h_{h}_N_{N}', burn_in=burn_in)

#%%
def run_german_credit(num_runs, N, K, h, num_steps):
    # Load the data 
    # German credit dataset 
    path = 'german.data-numeric'
    data = pd.read_csv(path, delim_whitespace = True, header = None)

    #%%
    # Separate features and target
    X = data.iloc[:, :-1].values  # All columns except the last as features
    y = data.iloc[:, -1].values   # Last column as the target

    # The last column is the target, 1 is good and 2 is bad, we change that so 0 is good and 1 is bad 
    y = jnp.where(y == 1, 0, jnp.where(y == 2, 1, y))

    # Optionally convert them to JAX arrays
    X = jnp.array(X)
    y = jnp.array(y)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to JAX arrays
    X_train = jnp.array(X_train)
    X_test = jnp.array(X_test)
    y_train = jnp.array(y_train)
    y_test = jnp.array(y_test)
    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
    # type
    print(f'X_train type: {type(X_train)}, y_train type: {type(y_train)}')
    print(f'X_test type: {type(X_test)}, y_test type: {type(y_test)}')
    # if any nans print location
    print(f'X_train nan locations: {jnp.argwhere(jnp.isnan(X_train))}')
    print(f'y_train nan locations: {jnp.argwhere(jnp.isnan(y_train))}')
    print(f'X_test nan locations: {jnp.argwhere(jnp.isnan(X_test))}')
    print(f'y_test nan locations: {jnp.argwhere(jnp.isnan(y_test))}')
    # mma16816 data type not supported
    print(f'X_train dtype: {X_train.dtype}, y_train dtype: {y_train.dtype}')
    print(f'X_test dtype: {X_test.dtype}, y_test dtype: {y_test.dtype}')
    # Create the target 
    p = LogRegTarget(X_train, y_train)
    D = X_train.shape[1] + 1
    q_0 = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    q = MultivariateNormal(jnp.zeros(D), jnp.ones(D))
    
    run_experiment(num_runs, N, D, K, h, num_steps, p, q_0, q, target_name = 'nbi_german_credit_n100_0.001', burn_in=burn_in)
    

#%%
# h=0.1
# run_gauss(num_runs, N, K, h, num_steps)
# h=0.01
# run_banana(num_runs, N, K, h, num_steps)
# h=0.001
# run_ill_conditioned_gauss(num_runs, N, K, h, num_steps)
h=0.001
run_german_credit(num_runs, N, K, h, num_steps)
