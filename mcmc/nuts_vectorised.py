import jax 
from jax import random 
import jax.numpy as jnp 
from jax import vmap 

class NUTS:

    def __init__(self, num_samples, state_dim, target, h):
        self.N = num_samples
        self.D = state_dim
        self.target = target
        massMatrix = jnp.eye(self.D)

        if jnp.isscalar(h):
            self.h = jnp.repeat(h, self.N)
        else:
            self.h = h
        self.MM = jnp.tile(massMatrix, (self.N, 1, 1))
        self.inv_MM = jnp.tile(jnp.linalg.inv(massMatrix), (self.N, 1, 1))
        self.max_tree_depth = 10
        self.delta_max = 100


    def rvs(self, x, v, keys):
        grad_x = self.get_grad(x)
        x_new, v_new, acceptance, num_nodes = self.generate_NUTS_samples(x, v, grad_x, keys)
        return x_new, v_new, num_nodes# , v, acceptance
    
    # Using vmap in order to generate the log probabilities 
    def get_grad(self, x):
        log_prob_grads = vmap(self.target.log_prob_grad, in_axes=(0, ))(x)
        return log_prob_grads
    
    # The same with the gradients 
    def get_log_prob(self, x):
        log_probs = vmap(self.target.log_prob, in_axes=(0, ))(x)
        return log_probs
    
    # Leapfrog integrator 
    def Integrate_LF_vec(self, x, v, grad_x, direction, h):
        h = h.reshape(self.N, 1)
        direction_reshaped = direction[:, jnp.newaxis]
        v = v + direction_reshaped * (h / 2) * grad_x
        x = x + direction_reshaped * h * jnp.einsum('bij,bj->bi', self.inv_MM, v)
        grad_x = self.get_grad(x)
        v = v + direction_reshaped * (h / 2) * grad_x
        return x, v, grad_x

    # Return True for particles we want to stop (NB opposite way round to s in Hoffman and Gelman paper)
    def stop_criterion_vec(self, xminus, xplus, rminus, rplus):
        dx = xplus - xminus
        left = (jnp.sum(dx * jnp.einsum('bij,bj->bi', self.inv_MM, rminus), axis=1) < 0)
        right = (jnp.sum(dx * jnp.einsum('bij,bj->bi', self.inv_MM, rplus), axis=1) < 0)
        return jnp.logical_or(left, right)

    # Get Hamiltonian energy of system given log target weight logp
    def get_hamiltonian(self, x, v, logp):
        return logp - 0.5 * jnp.sum(v * jnp.einsum('bij,bj->bi', self.inv_MM, v), axis=1)

    # Return xmerge = vectors of xminus where direction < 0 and xplus where direction > 0, and similarly for v
    # and grad_x
    def merge_states_dir(self, xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction):
        mask = direction < 0
        mask = mask.reshape(len(mask), 1).astype(int)
        xmerge = mask * xminus + (1 - mask) * xplus
        vmerge = mask * vminus + (1 - mask) * vplus
        grad_xmerge = mask * grad_xminus + (1 - mask) * grad_xplus
        return xmerge, vmerge, grad_xmerge


    def generate_NUTS_samples(self, x0, v0, grad_x0, key):

        # Sample energy: note that log(U(0,1)) has same distribution as -exponential(1)
        nsamples = len(x0)
        logp0 = self.get_log_prob(x0)
        joint = self.get_hamiltonian(x0, v0, logp0)
        rngs = jax.random.uniform(key, (self.N, ))
        key, subkey = random.split(key)
        logu = joint + jnp.log(rngs)

        xminus = x0
        xplus = x0
        vminus = v0
        vplus = v0
        xprime = x0
        vprime = v0
        grad_xplus = grad_x0
        grad_xminus = grad_x0

        depth = 0;

        stopped = jnp.zeros(nsamples).astype(bool) # 0 if still running (opposite to MATLAB)
        numnodes = jnp.ones(nsamples).astype(int)

        # Used to compute acceptance rate
        alpha = jnp.zeros(nsamples)
        nalpha = jnp.zeros(nsamples).astype(int)

        while jnp.any(stopped == 0):

            # Generate random direction in {-1, +1}
            rngs = jax.random.uniform(key, (self.N, ))
            key, subkey = random.split(key)
            direction = 2 * (rngs < 0.5).astype(int) - 1

            # Get new states from minus and plus depending on direction and build tree
            x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction)
            xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, numnodes2, stopped2, alpha2, nalpha2 = \
                self.build_tree(x_pm, v_pm, grad_x_pm, joint, logu, direction, stopped, depth, key)

            # Split the output back based on direction - keep the stopped samples the same
            idxminus = jnp.logical_and(jnp.logical_not(stopped), direction < 0).reshape(nsamples,1).astype(int)
            xminus = idxminus * xminus2 + (1 - idxminus) * xminus
            vminus = idxminus * vminus2 + (1 - idxminus) * vminus
            grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
            idxplus = jnp.logical_and(jnp.logical_not(stopped), direction > 0).reshape(nsamples,1).astype(int)
            xplus = idxplus * xplus2 + (1 - idxplus) * xplus
            vplus = idxplus * vplus2 + (1 - idxplus) * vplus
            grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

            # Update acceptance rate
            alpha = jnp.logical_not(stopped) * alpha2 + stopped * alpha
            nalpha = jnp.logical_not(stopped) * nalpha2 + stopped * nalpha

            # If no U-turn, choose new state
            rngs = jax.random.uniform(key, (self.N, ))
            key, subkey = random.split(key)
            u = numnodes * rngs < numnodes2
            selectnew = jnp.logical_and(jnp.logical_not(stopped2), u).reshape(nsamples,1).astype(int)
            xprime = selectnew * xprime2 + (1 - selectnew) * xprime
            vprime = selectnew * vprime2 + (1 - selectnew) * vprime

            # Update number of nodes and tree height
            numnodes = numnodes + numnodes2;
            depth = depth + 1;
            if depth > self.max_tree_depth:
                print("Max tree size in NUTS reached")
                break

            # Do U-turn test
            stopped = jnp.logical_or(stopped, stopped2)
            stopped = jnp.logical_or(stopped, self.stop_criterion_vec(xminus, xplus, vminus, vplus))

        acceptance = alpha / nalpha

        return xprime, vprime, acceptance, numnodes 


    def build_tree(self, x, v, grad_x, joint, logu, direction, stopped, depth, key):

        nsamples = len(x)

        if depth == 0:

            # Base case
            # ---------

            not_stopped = jnp.logical_not(stopped)

            # Do leapfrog
            xprime2, vprime2, grad_xprime2 = self.Integrate_LF_vec(x, v, grad_x, direction, self.h)

            idx_notstopped = not_stopped.reshape(nsamples,1).astype(int)
            xprime = idx_notstopped * xprime2 + (1- idx_notstopped) * x
            vprime = idx_notstopped * vprime2 + (1- idx_notstopped) * v
            grad_xprime = idx_notstopped * grad_xprime2 + (1- idx_notstopped) * grad_x

            # Get number of nodes
            logpprime = self.get_log_prob(xprime)
            jointprime = self.get_hamiltonian(xprime, vprime, logpprime)
            numnodes = (logu <= jointprime).astype(int)

            # Update acceptance rate
            logalphaprime = jnp.where(jointprime > joint, 0.0, jointprime - joint)
            alphaprime = jnp.zeros(nsamples)
            alphaprime = jnp.where(not_stopped, jnp.exp(logalphaprime), alphaprime)
            alphaprime = jnp.where(jnp.isnan(alphaprime), 0.0, alphaprime)
            nalphaprime = jnp.ones_like(alphaprime, dtype=int)

            # Stop bad samples
            stopped = jnp.logical_or(stopped, logu - self.delta_max >= jointprime)

            return xprime, vprime, grad_xprime, xprime, vprime, grad_xprime, xprime, vprime, numnodes, stopped, alphaprime, nalphaprime

        else:

            # Recursive case
            # --------------

            # Build one subtree
            xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, vprime, numnodes, stopped, alpha, nalpha = self.build_tree(
                x, v, grad_x, joint, logu, direction, stopped, depth-1, key)

            if jnp.any(stopped == 0):

                # Get new states from minus and plus depending on direction and build tree
                x_pm, v_pm, grad_x_pm = self.merge_states_dir(xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, direction)
                xminus2, vminus2, grad_xminus2, xplus2, vplus2, grad_xplus2, xprime2, vprime2, numnodes2, stopped2, alpha2, nalpha2 = self.build_tree(
                    x_pm, v_pm, grad_x_pm, joint, logu, direction, stopped, depth-1, key)

                # Split the output back based on direction - keep the stopped samples the same
                idxminus = jnp.logical_and(jnp.logical_not(stopped), direction < 0).reshape(nsamples, 1).astype(int)
                xminus = idxminus * xminus2 + (1 - idxminus) * xminus
                vminus = idxminus * vminus2 + (1 - idxminus) * vminus
                grad_xminus = idxminus * grad_xminus2 + (1 - idxminus) * grad_xminus
                idxplus = jnp.logical_and(jnp.logical_not(stopped), direction > 0).reshape(nsamples, 1).astype(int)
                xplus = idxplus * xplus2 + (1 - idxplus) * xplus
                vplus = idxplus * vplus2 + (1 - idxplus) * vplus
                grad_xplus = idxplus * grad_xplus2 + (1 - idxplus) * grad_xplus

                # Do new sampling
                rngs = jax.random.uniform(key, (self.N, ))
                key, subkey = random.split(key)
                u = numnodes * rngs < numnodes2
                selectnew = jnp.logical_and(jnp.logical_not(stopped2), u).reshape(nsamples, 1).astype(int)
                xprime = selectnew * xprime2 + (1 - selectnew) * xprime
                vprime = selectnew * vprime2 + (1 - selectnew) * vprime

                # Do U-turn test
                stopped = jnp.logical_or(stopped, stopped2)
                stopped = jnp.logical_or(stopped, self.stop_criterion_vec(xminus, xplus, vminus, vplus))

                # Update number of nodes
                not_stopped = jnp.logical_not(stopped)
                numnodes = numnodes + numnodes2;

                # Update acceptance rate
                alpha += not_stopped * alpha2
                nalpha += not_stopped * nalpha2

            return xminus, vminus, grad_xminus, xplus, vplus, grad_xplus, xprime, vprime, numnodes, stopped, alpha, nalpha