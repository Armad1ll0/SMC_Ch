import numpy as np 
import jax 
import jax.numpy as jnp 
from jax import random
from jax.scipy.stats import norm
from scipy.stats import special_ortho_group

#%%
# Ill conditioned gaussian example
# Function to generate a random orthogonal matrix
def random_orthogonal_matrix(D, key):
    # Use scipy's special_ortho_group to generate an orthogonal matrix
    return jnp.array(special_ortho_group.rvs(D, random_state=42))

# Function to create the ill-conditioned Gaussian
def create_gaussian(key, D=100):
    # Split key for randomness
    key_eigvals, key_orth = random.split(key)

    # 1. Generate eigenvalues from a gamma distribution with shape 0.5 and scale 1
    eigenvalues = random.gamma(key_eigvals, 0.5, shape=(D,)) * 1.0

    # 2. Create a random orthogonal matrix for the eigenvectors
    orthogonal_matrix = random_orthogonal_matrix(D, key_orth)

    # 3. Construct the covariance matrix Σ using eigenvalues and orthogonal matrix
    diag_eigenvalues = jnp.diag(eigenvalues)
    covariance_matrix = orthogonal_matrix @ diag_eigenvalues @ orthogonal_matrix.T

    return covariance_matrix

#%%
# Banana distribution 
# class BananaDistribution:
#     def __init__(self, a = 10, b = 0.03, c = 100, d = 1):
#         self.a = a
#         self.b = b
#         self.c = c
#         self.d = d

#     def log_prob(self, theta):
#         theta1, theta2 = theta
#         # Log-probability of theta1 ~ N(0, 10)
#         log_p_theta1 = norm.logpdf(theta1, loc=0.0, scale=jnp.sqrt(self.a))
        
#         # Log-probability of theta2 ~ N(0.03(theta1^2 - 100), 1)
#         theta2_mean = self.b * (theta1**2 - self.c)
#         log_p_theta2 = norm.logpdf(theta2, loc=theta2_mean, scale=jnp.sqrt(self.d))
        
#         return log_p_theta1 + log_p_theta2

#     def log_prob_grad(self, theta):
#         # Use JAX's autodiff to compute the gradient of the log-probability
#         return jax.grad(self.log_prob)(theta)
    
#     def prob(self, theta):
#         return jnp.exp(self.log_prob(theta))
    
class BananaDistribution:
    def __init__(self, dim=2, a = 0, b = 100):
        self.dim = dim
        self.a = a
        self.b = b
 
    def log_prob(self, x):
        x1 = x[..., 0]
        x2 = x[..., 1]
        term1 = (self.a - x1) ** 2
        term2 = self.b * (x2 - x1 ** 2) ** 2
        log_prob = -(term1 + term2)
        return log_prob
 
    def log_prob_grad(self, x):
        return jax.grad(self.log_prob)(x)
    
class LogRegTarget():
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def logits(self, params):
        return jnp.dot(self.X, params[:-1]) + params[-1]
        
    def log_prob(self, params):
        logits = self.logits(params)
        preds = jax.nn.sigmoid(logits)
        log_probs = jnp.sum(self.y * jnp.log(preds + 1e-8) + (1 - self.y) * jnp.log(1 - preds + 1e-8))
        return log_probs 
    
    def log_prob_grad(self, params):
        grad_f = jax.grad(self.log_prob)
        grads = grad_f(params)
        return grads