import matplotlib.pyplot as plt
import numpy as np

def plot_MSE(mean, var, true_mean, true_var, method = 'HMC', target=''): 
    means_np = np.array(mean)
    vars_np = np.array(var)
    fig, ax = plt.subplots(1, means_np.shape[1], figsize=(15, 5))
    for i in range(means_np.shape[1]):
        ax[i].plot(means_np[:, i], c = 'b')
        ax[i].axhline(true_mean[i], c = 'r')
    plt.title('Means')
    plt.show()
    plt.savefig(f'plots/{target}/{method}_means.png')
    plt.clf()
    
    fig, ax = plt.subplots(1, means_np.shape[1], figsize=(15, 5))
    for i in range(means_np.shape[1]):
        ax[i].plot(vars_np[:, i], c = 'b')
        ax[i].axhline(true_var[i], c = 'r')
    plt.title('Variances')
    plt.show()
    plt.savefig(f'plots/{target}/{method}_vars.png')
    plt.clf()
    
    mse_means = []
    mse_vars = []
    for i in range(len(means_np)):
        mse_means.append(np.mean((means_np[i] - true_mean)**2))
        mse_vars.append(np.mean((vars_np[i] - true_var)**2))
    
    plt.plot(mse_means)
    plt.yscale('log')
    plt.title(f'MSE for Means: Method {method}')
    plt.show()
    plt.savefig(f'plots/{target}/{method}_MSE_means.png')
    plt.clf()
    
    plt.plot(mse_vars)
    plt.yscale('log')
    plt.title(f'MSE for Vars: Method {method}')
    plt.show() 
    plt.savefig(f'plots/{target}/{method}_MSE_vars.png')
    
    return mse_means, mse_vars


def plot_samples(samples, dim_1, dim_2, method, target):
    
    plt.close()
    plt.clf()
    plt.scatter(samples[:, :, dim_1], samples[:, :, dim_2], c = 'b', s = 1)
    plt.title(f'Samples: Method {method}, Distribution {target}')
    plt.xlabel(f'Dimension 1 ({dim_1})')
    plt.ylabel(f'Dimension 2 ({dim_2})')
    plt.show()
    plt.savefig(f'plots/{target}/{method}_samples.png')
    plt.close()


def plot_only_MSE(mean, var, true_mean, true_var, method = 'HMC', target=''): 
    means_np = np.array(mean)
    vars_np = np.array(var)
    mse_means = []
    mse_vars = []
    for i in range(len(means_np)):
        mse_means.append(np.mean((means_np[i] - true_mean)**2))
        mse_vars.append(np.mean((vars_np[i] - true_var)**2))
    
    plt.clf()
    plt.plot(mse_means)
    plt.yscale('log')
    plt.title(f'MSE for Means: Method {method}')
    plt.show()
    plt.savefig(f'plots/{target}/{method}_MSE_means.png')
    plt.clf()
    
    plt.plot(mse_vars)
    plt.yscale('log')
    plt.title(f'MSE for Vars: Method {method}')
    plt.show() 
    plt.savefig(f'plots/{target}/{method}_MSE_vars.png')
    
    return mse_means, mse_vars