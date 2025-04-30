import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np 

from scipy.stats.qmc import Halton
from scipy.stats.qmc import Sobol

# use agg
plt.switch_backend('agg')

class golden_ratio:
    def __init__(self, K):
        self.K=K

    def random(self, N):
        print((5**0.5-1)/2)
        rn_seq = jnp.arange(1, N*self.K+1)*(5**0.5-1)/2 % 1
        return rn_seq.reshape(N, self.K)#.reshape(self.K, N).T # have to do this or at each K it is constrained to a region o 1/N, same or 1 D halton
    
class primes_seq:
    def __init__(self, K):
        self.K=K
        preloaded_primes = np.array([
                                        2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
                                        31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
                                        73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
                                        127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
                                        179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
                                        233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
                                        283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
                                        353, 359, 367, 373, 379, 383, 389, 397, 401, 409,
                                        419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                                        467, 479, 487, 491, 499, 503, 509, 521, 523, 541
                                    ])
        self.primes = preloaded_primes[:K]

    def random(self, N):
        prime_coeffs = (self.primes**0.5-1)/2
        rn_seq = jnp.arange(1, N*self.K+1).reshape(N, self.K)*prime_coeffs% 1
        return rn_seq
    
class equidistant:
    def __init__(self, K, offset=False):
        self.K=K
        self.offset = offset


    def random(self, N):
        equi_seq = np.arange(1, N+1)/N 
        # randomly shuffle the sequence K times and fill N x K matrix
        rn_seq = np.ones((N, self.K))
        for k in range(self.K):
            rn_seq[:, k] = np.random.permutation(equi_seq)

        if self.offset:
            rn_seq = rn_seq + np.random.uniform(size=(N, self.K))*0.1
            # clip to [0.001, 1]
            rn_seq = np.clip(rn_seq, 0.001, 1)


        print("rn_seq", rn_seq)
        return rn_seq
    
class no_jitter:
    def __init__(self, K):
        self.K=K

    def random(self, N):
        rn_seq = np.ones((N, self.K))
        return rn_seq    
    
    
if __name__ == '__main__':
    K = 5
    N = 10
    gr = golden_ratio(K)
    gr_samples = gr.random(N)
    print("GR______")
    print(gr_samples)

    for k in range(K):
        plt.plot(gr_samples[:, k], jnp.repeat(0, N), 'o')
        plt.savefig(f'golden_ratio_iter_{k}.png')
        plt.cla()

    # halton = Halton(K, scramble=True)
    # halton_samples = halton.random(N)
    # print("Halton_____")
    # print(halton_samples)

    # for k in range(K):
    #     plt.plot(halton_samples[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'halton_iter_{k}.png')
    #     plt.cla()


    # unscrambled_halton = Halton(K, scramble=False)
    # unscrambled_halton.fast_forward(1)
    # unscrambled_halton_samples = unscrambled_halton.random(N)
    # print("unscrambled_Halton_____")
    # print(unscrambled_halton_samples)
    # print(unscrambled_halton_samples[:, 0])

    # for k in range(K):
    #     plt.plot(unscrambled_halton_samples[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'unscrambled_halton_iter_{k}.png')
    #     plt.cla()

    # # reverse order of K
    # unscrambled_halton_samples = unscrambled_halton_samples[:, ::-1]
    # print("unscrambled_Halton_____rev")
    # print(unscrambled_halton_samples)

    # 1D halton
    halton1D = Halton(1, scramble=False)
    halton1D.fast_forward(1)
    halton1D_samples = halton1D.random(N*K)
    halton1D_samples = halton1D_samples.reshape(N, K)

    print("Halton1D_____")
    print(halton1D_samples)
    print(halton1D_samples[:, 0])

    for k in range(K):
        plt.plot(halton1D_samples[:, k], jnp.repeat(0, N), 'o')
        plt.savefig(f'halton1D_iter_{k}.png')
        plt.cla()

    # 1D sobol
    sobol1D = Sobol(1, scramble=False)
    sobol1D.fast_forward(1)
    sobol1D_samples = sobol1D.random(N*K)
    sobol1D_samples = sobol1D_samples.reshape(N, K)

    print("Sobol1D_____")
    print(sobol1D_samples)
    print(sobol1D_samples[:, 0])

    # sobol
    sobol = Sobol(K, scramble=True)
    sobol_samples = sobol.random(N)
    print("Sobol_____")
    print(sobol_samples)
    print(sobol_samples[:, 0])

    # # uniform 
    # uniform = np.random.uniform(size=(N, K))
    # print("uniform_____")
    # print(uniform)

    # for k in range(K):
    #     plt.plot(uniform[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'uniform_iter_{k}.png')
    #     plt.cla()

    # # primes
    # primes = primes_seq(K)
    # primes_samples = primes.random(N)
    # print("Primes_____")
    # print(primes_samples)

    # for k in range(K):
    #     plt.plot(primes_samples[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'primes_iter_{k}.png')
    #     plt.cla()

    # # equidistant
    # equidistant_class = equidistant(K, offset=True)
    # equidistant_samples = equidistant_class.random(N)
    # print("equidistant_____")
    # print(equidistant_samples)
    
    # for k in range(K):
    #     plt.plot(equidistant_samples[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'equidistant_iter_{k}.png')
    #     plt.cla()

