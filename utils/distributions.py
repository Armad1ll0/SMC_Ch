import jax 
import jax.numpy as jnp 
import jax.random as random 

class MultivariateNormal:
    def __init__(self, mean, var, cov = None):
        self.mean = mean
        if cov == None: 
            self.cov = jnp.diag(var)
        else: 
            self.cov = cov 
        self.dim = mean.shape[0]

        # Precompute the Cholesky decomposition for sampling
        self.L = jnp.linalg.cholesky(self.cov)

    def sample(self, key, num_samples=1):
        z = random.normal(key, (num_samples, self.dim))  # Standard normal samples
        return self.mean + jnp.dot(z, self.L.T)

    def log_prob(self, x):
        diff = x - self.mean
        log_det_cov = 2 * jnp.sum(jnp.log(jnp.diag(self.L)))  # log determinant of the covariance
        exponent = -0.5 * jnp.dot(diff, jnp.linalg.solve(self.cov, diff))
        return -0.5 * self.dim * jnp.log(2 * jnp.pi) - 0.5 * log_det_cov + exponent
    
    def log_prob_grad(self, x):
        grad_f = jax.grad(self.log_prob)
        return grad_f(x)