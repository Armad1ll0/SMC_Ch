import jax.numpy as jnp 
from jax import random
import jax 

# converted from https://github.com/UoL-SignalProcessingGroup/SMC-NUTS/blob/main/smcnuts/proposal/nuts.py
def generate_nuts_samples(x0, r0, target, step_size, key, max_tree_depth = 10):

    """
    Description
    -----------
    Generates samples using the NUTS proposal, Based off Alg. 3 in [1]
    """
    
    sample_grad_evals = 0

    logp = target.log_prob(x0)    
    H0 = logp - 0.5 * jnp.dot(r0, r0.T) 

    logu = H0 - jnp.exp(1.)
    
    # Precalculate the gradient for efficiency
    grad_x = target.log_prob_grad(x0)
    sample_grad_evals += 1
    
    # initialize the NUTS tree 
    x = x0
    xminus = x0
    xplus = x0
    rminus = r0
    rplus = r0
    r = r0
    gradminus = grad_x
    gradplus = grad_x
 

    depth = 0  
    n = 1  
    stop = 0  

    while (stop == 0):
        # Using a Bernoulli trial choose a direction. -1 (backwards) or +1 (forwards)
        direction = int(2 * (random.uniform(key) < 0.5) - 1)
        key, subkey = random.split(key)
        
        if (direction == -1):
            xminus, rminus, gradminus, _, _, _, xprime, rprime, nprime, stopprime, sample_grad_evals = build_tree(xminus, rminus, gradminus, logu, direction, depth, target, step_size, key, sample_grad_evals)
        else:
            _, _, _, xplus, rplus, gradplus, xprime, rprime, nprime, stopprime, sample_grad_evals  = build_tree(xplus, rplus, gradplus, logu, direction, depth, target, step_size, key, sample_grad_evals)


        if (stopprime == 0 and random.uniform(key) < min(1., float(nprime) / float(n))):
            x = xprime
            r = rprime
            key, subkey = random.split(key)

        n += nprime

        stop = stopprime or stop_criterion(xminus, xplus, rminus, rplus)           
        
        depth += 1
        
        if(depth > max_tree_depth):
            print('Max tree depth reached')
            break
    
    return x, r, sample_grad_evals

def build_tree(x, r, grad_x, logu, direction, depth, target, step_size, key, sample_grad_evals):
    """
    Description
    -----------
    Generates samples using the recursive NUTS tree-building procedure [1]
    """
    if (depth == 0):
        xprime, rprime, gradprime, sample_grad_evals = NUTSLeapfrog(x, r, grad_x, direction, step_size, target, sample_grad_evals)
        logpprime = target.log_prob(xprime)
        joint = logpprime - 0.5 * jnp.dot(rprime, rprime.T)
        nprime = int(logu < joint)
        stopprime = int((logu - 100.) >= joint)
        xminus = xprime
        xplus = xprime
        rminus = rprime
        rplus = rprime
        gradminus = gradprime
        gradplus = gradprime
    else:
                                                                                     
        xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime, sample_grad_evals = build_tree(x, r, grad_x, logu, direction, depth - 1,  target, step_size, key, sample_grad_evals)
        
        if (stopprime == 0):
            if (direction == -1):
                xminus, rminus, gradminus, _, _, _, xprime2, rprime2, nprime2, stopprime2, sample_grad_evals  = build_tree(xminus, rminus, gradminus, logu, direction, depth - 1,  target, step_size, key, sample_grad_evals)
            else:
                _, _, _, xplus, rplus, gradplus, xprime2, rprime2, nprime2, stopprime2, sample_grad_evals   = build_tree(xplus, rplus, gradplus, logu, direction, depth - 1, target, step_size, key, sample_grad_evals)           
           
            if (random.uniform(key) < (float(nprime2) / max(float(int(nprime) + int(nprime2)), 1.))):
                xprime = xprime2
                rprime = rprime2
                key, subkey = random.split(key)

            nprime = int(nprime) + int(nprime2)

            stopprime = int(stopprime or stopprime2 or stop_criterion(xminus, xplus, rminus, rplus))

    return xminus, rminus, gradminus, xplus, rplus, gradplus, xprime, rprime,  nprime, stopprime, sample_grad_evals

def stop_criterion(xminus, xplus, rminus, rplus):
    """
    Description
    -----------
    Checks if a U-turn is present in the furthest nodes in the NUTS
    tree
    """
    dx = xplus - xminus
    return (jnp.dot(dx, rminus.T) < 0) or (jnp.dot(dx, rplus.T) < 0)

def NUTSLeapfrog(x, r, grad_x, direction, step_size, target, sample_grad_evals):

    """
    Description
    -----------
    Performs a single Leapfrog step returning the final position, momentum and gradient.
    """
    r = jnp.add(r, (direction*step_size/2)*grad_x)
    x = jnp.add(x, direction*step_size*r)
    grad_x = target.log_prob_grad(x)
    sample_grad_evals += 1
    r = jnp.add(r, (direction*step_size/2)*grad_x)
    
    return x, r, grad_x, sample_grad_evals