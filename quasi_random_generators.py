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
        return rn_seq.reshape(N, self.K)
    
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
            467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
            547, 557, 563, 569, 571, 577, 587, 593, 599, 601,
            607, 613, 617, 619, 631, 641, 643, 647, 653, 659,
            661, 673, 677, 683, 691, 701, 709, 719, 727, 733,
            739, 743, 751, 757, 761, 769, 773, 787, 797, 809,
            811, 821, 823, 827, 829, 839, 853, 857, 859, 863,
            877, 881, 883, 887, 907, 911, 919, 929, 937, 941,
            947, 953, 967, 971, 977, 983, 991, 997, 1009, 1013,
            1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
            1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151,
            1153, 1163, 1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223,
            1229, 1231, 1237, 1249, 1259, 1277, 1279, 1283, 1289, 1291,
            1297, 1301, 1303, 1307, 1319, 1321, 1327, 1361, 1367, 1373,
            1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439, 1447, 1451,
            1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
            1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583,
            1597, 1601, 1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657,
            1663, 1667, 1669, 1693, 1697, 1699, 1709, 1721, 1723, 1733,
            1741, 1747, 1753, 1759, 1777, 1783, 1787, 1789, 1801, 1811,
            1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877, 1879, 1889,
            1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987
        ])
        self.primes = preloaded_primes[:K]

    def random(self, N):
        # prime_coeffs = (self.primes**0.5-1)/2
        # rn_seq = jnp.arange(1, N*self.K+1).reshape(N, self.K)*prime_coeffs% 1
        prime_coeffs = self.primes**0.5
        print("prime_coeffs:", prime_coeffs.shape)
        coeffs = jnp.tile(prime_coeffs, (N,1))
        print("coeffs shape", coeffs.shape)
        rn_seq = jnp.arange(1, N*self.K+1).reshape(N, self.K)*coeffs% 1
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

    print("Primes_____")
    primes = primes_seq(K)
    primes_samples = primes.random(N)
    print(primes_samples)

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

    # # 1D halton
    # halton1D = Halton(1, scramble=False)
    # halton1D.fast_forward(1)
    # halton1D_samples = halton1D.random(N*K)
    # halton1D_samples = halton1D_samples.reshape(N, K)

    # print("Halton1D_____")
    # print(halton1D_samples)
    # print(halton1D_samples[:, 0])

    # for k in range(K):
    #     plt.plot(halton1D_samples[:, k], jnp.repeat(0, N), 'o')
    #     plt.savefig(f'halton1D_iter_{k}.png')
    #     plt.cla()

    # # 1D sobol
    # sobol1D = Sobol(1, scramble=False)
    # sobol1D.fast_forward(1)
    # sobol1D_samples = sobol1D.random(N*K)
    # sobol1D_samples = sobol1D_samples.reshape(N, K)

    # print("Sobol1D_____")
    # print(sobol1D_samples)
    # print(sobol1D_samples[:, 0])

    # # sobol
    # sobol = Sobol(K, scramble=True)
    # sobol_samples = sobol.random(N)
    # print("Sobol_____")
    # print(sobol_samples)
    # print(sobol_samples[:, 0])

    # # # uniform 
    # # uniform = np.random.uniform(size=(N, K))
    # # print("uniform_____")
    # # print(uniform)

    # # for k in range(K):
    # #     plt.plot(uniform[:, k], jnp.repeat(0, N), 'o')
    # #     plt.savefig(f'uniform_iter_{k}.png')
    # #     plt.cla()

    # # # primes
    # # primes = primes_seq(K)
    # # primes_samples = primes.random(N)
    # # print("Primes_____")
    # # print(primes_samples)

    # # for k in range(K):
    # #     plt.plot(primes_samples[:, k], jnp.repeat(0, N), 'o')
    # #     plt.savefig(f'primes_iter_{k}.png')
    # #     plt.cla()

    # # # equidistant
    # # equidistant_class = equidistant(K, offset=True)
    # # equidistant_samples = equidistant_class.random(N)
    # # print("equidistant_____")
    # # print(equidistant_samples)
    
    # # for k in range(K):
    # #     plt.plot(equidistant_samples[:, k], jnp.repeat(0, N), 'o')
    # #     plt.savefig(f'equidistant_iter_{k}.png')
    # #     plt.cla()

