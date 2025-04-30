import os
import jax
from jax import random, vmap
jax.config.update("jax_platform_name", "cpu")
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split

from utils.targets import BananaDistribution, random_orthogonal_matrix, create_gaussian, LogRegTarget

mpl.use('Agg')

# use grid everywhere
mpl.rcParams['axes.grid'] = True
# grid parameters
mpl.rcParams['grid.color'] = 'gray'
mpl.rcParams['grid.linestyle'] = '--'
mpl.rcParams['grid.alpha'] = 0.6
# set legend fontsize
mpl.rcParams['legend.fontsize'] = 16
# set suptitle fontsize
mpl.rcParams['figure.titlesize'] = 40
# set title fontsize
mpl.rcParams['axes.titlesize'] = 35
# set label fontsize
mpl.rcParams['axes.labelsize'] = 30
# set tick fontsize
mpl.rcParams['xtick.labelsize'] = 20
mpl.rcParams['ytick.labelsize'] = 20

def load_numpy_files(folder_path, file_type = "mean", run=None):
    numpy_files = []
    for file_name in os.listdir(folder_path):
        if run is not None:
            if file_name.startswith(file_type) and file_name.endswith(f"_{run}.npy"):
                file_path = os.path.join(folder_path, file_name)
                data = np.load(file_path)
                return np.array(data)
        else:
            if file_name.startswith(file_type) and file_name.endswith(".npy"):  
                file_path = os.path.join(folder_path, file_name)
                data = np.load(file_path)
                numpy_files.append(data)
    
    return numpy_files

# need to load and avg mean, variance, num_grad_evals, neffs
def get_metrics(folder_path):
    mean_files = load_numpy_files(folder_path, file_type = 'mean')
    var_files = load_numpy_files(folder_path, file_type = 'variance')
    num_grad_evals = load_numpy_files(folder_path, file_type = 'num_grad_evals')
    neffs = load_numpy_files(folder_path, file_type = 'Neffs')
    return mean_files, var_files, num_grad_evals, neffs

def avg_metrics(mean_files, var_files, num_grad_evals, neffs):
    avg_mean = np.mean(np.array(mean_files), axis=0)
    avg_var = np.mean(np.array(var_files), axis=0)
    avg_num_grad_evals = np.mean(np.array(num_grad_evals), axis=0)
    avg_neffs = np.mean(np.array(neffs), axis=0)
    return avg_mean, avg_var, avg_num_grad_evals, avg_neffs

def plot_neff_per_grad_evals(experiments_results, experiment, method_fps):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'$J^{\text{eff}}/\nabla\text{eval}$')
    for method, method_fp in zip(experiments_results[experiment].keys(), method_fps):
        neffs = experiments_results[experiment][method]['neffs']
        num_grad_evals = experiments_results[experiment][method]['num_grad_evals']
        ax.plot(neffs/num_grad_evals, label=method_fp)    
    
    ax.legend()
    ax.set_yscale('log')
    fig.suptitle(f'{experiment}')
    plt.savefig(f'plots/paper/{experiment}_neff_per_grad_evals.pdf', bbox_inches='tight', dpi=300)

def plot_mse(experiments_results, experiment, true_values, method_fps, metric='mean'):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('MSE')

    for method, method_fp in zip(experiments_results[experiment].keys(), method_fps):
        metric_values = experiments_results[experiment][method][metric]
        mse = np.mean((metric_values - true_values)**2, axis=1)
        ax.plot(mse, label=method_fp)

    ax.legend()
    ax.set_yscale('log')
    # fig.suptitle(f'{experiment}')
    plt.savefig(f'plots/paper/{experiment}_mse_{metric}.pdf', bbox_inches='tight', dpi=300)

def plot_roc_curve(experiments_results, experiment, X_test, y_test):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    for method in experiments_results[experiment].keys():
       
        # params = np.mean(experiments_results[experiment][method]['mean'], axis=0)
        params = experiments_results[experiment][method]['mean'][-1, :]
        preds = np.round(jax.nn.sigmoid(np.dot(X_test, params[:-1]) + params[-1]))
        # calculate accuracy
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        print(method, "tn", tn, "fp", fp, "fn", fn, "tp", tp)
        # acc
        acc = (tp + tn) / (tn + fp + fn + tp)
        print(method, "acc", acc)
        fpr, tpr, _ = roc_curve(y_test, preds)
        print(method, "roc_auc", roc_auc_score(y_test, preds))
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f'{method} (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    
    ax.legend()
    # fig.suptitle(f'{experiment}')
    plt.savefig(f'plots/paper/{experiment}_roc_curve.pdf', bbox_inches='tight', dpi=300)

def plot_banana_samples(folder_path, method, target, run=1):
    samples = load_numpy_files(folder_path, file_type = 'samples', run=run)
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.linspace(-2.5, 2.5, 300)
    y = np.linspace(-2.0, 5.5, 300)
    X, Y = np.meshgrid(x, y)
    Z = vmap(target.log_prob, in_axes = (0))(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20)
    ax.scatter(samples[:, :, 0], samples[:, :, 1], color='red', alpha=0.5)
    ax.set_xlabel(r'$\theta_{(1)}$')
    ax.set_ylabel(r'$\theta_{(2)}$')
    fig.suptitle(f'{method}')
    plt.savefig(f'plots/paper/banana_samples_{method}.pdf', bbox_inches='tight', dpi=300)

def create_latex_table(experiments_results, method_fps):
    experiment1 = list(experiments_results.keys())[0]
    for method, method_fp in zip(experiments_results[experiment1].keys(), method_fps):
        if method == 'ChEES_hal_scram':
            method = 'sChEES'
        print(f"\\textbf{{{method_fp}}}", end=' ')
        for experiment in experiments_results.keys():
            # add mean num grad evals and neffs per num grad evals
            avg_num_grad_evals = np.mean(experiments_results[experiment][method]['num_grad_evals'])
            avg_neffs = np.mean(experiments_results[experiment][method]['neffs'])
            # display avg neffs/avg num grad evals as exponent
            test = f"{avg_neffs/avg_num_grad_evals:.2f}"
            print(f"& {avg_num_grad_evals/100:.2f} & {avg_neffs/avg_num_grad_evals:.2e}", end=' ')

        print("\\\\")

# create classification table german credit
def create_classification_table(experiments_results, method_fps):
    for method, method_fp in zip(experiments_results['German Credit'].keys(), method_fps):
        if method == 'ChEES_hal_scram':
            method = 'sChEES'
        print(f"\\textbf{{{method_fp}}}", end=' ')
        params = experiments_results["German Credit"][method]['mean'][-1, :]
        preds = np.round(jax.nn.sigmoid(np.dot(X_test, params[:-1]) + params[-1]))
        # accuracy, precision, recall, f1, specificty, auroc
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        acc = (tp + tn) / (tn + fp + fn + tp)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * (precision * recall) / (precision + recall)
        specificity = tn / (tn + fp)
        auroc = roc_auc_score(y_test, preds)
        print(f"& {acc:.2f} & {precision:.2f} & {recall:.2f} & {f1:.2f} & {specificity:.2f} & {auroc:.2f}", end=' ')

        print("\\\\")
        

methods = ['NUTS', 'no_jitter', 'Nd-ChEES_hal',  'Nd-ChEES-inv-hal', 
           '1d-ChEES-hal', '1d-ChEES-gr','1d-ChEES-uni', 'Nd-ChEES-primes', 
           'Nd-ChEES-inv-primes', 'Nd-ChEES-equi', 'Nd-ChEES-ofset-equi',  
           "Nd-Sobol", 'Nd-Sobol_inv', '1D-Sobol']
method_plot_names = ['NUTS', 'No Jitter', 'N-d Halton', 'N-d Inverse Halton', 
                    '1-d Halton', '1-d Golden Ratio', '1-d Uniform', 'N-d Primes', 
                    'N-d Inverse Primes', 'N-d Equidistant', 'N-d Offset Equidistant',
                    "N-d Sobol", 'N-d Inverse Sobol', '1-D Sobol']
experiments = ['long_nbi_ill_conditioned_gauss_h_0.001_N_100', 'nbi_new_banana',  'nbi_gauss', 'nbi_german_credit_n100_0.001',] 

experiments_results = {}
for experiment in experiments:
    print("__________________________")
    if 'banana' in experiment:
        experiment_key = 'Banana'
    elif 'gauss' in experiment:
        if 'ill_conditioned' in experiment:
            experiment_key = 'Ill-conditioned Gaussian'
        else:
            experiment_key = 'Gaussian'
    elif 'german_credit' in experiment:
        experiment_key = 'German Credit'
    print("experiment", experiment)
    experiments_results[experiment_key] = {}
    for method in methods:
        folder_path = f'results/{experiment}/{method}'
        mean_files, var_files, num_grad_evals, neffs = get_metrics(folder_path)
        avg_mean, avg_var, avg_num_grad_evals, avg_neffs = avg_metrics(mean_files, var_files, num_grad_evals, neffs)
        if method == 'ChEES_hal_scram':
            method = 'sChEES'
        elif method == 'ChEES_gr':
            method = 'grChEES'
        experiments_results[experiment_key][method] = {'mean': avg_mean, 'var': avg_var, 'num_grad_evals': avg_num_grad_evals, 'neffs': avg_neffs}

create_latex_table(experiments_results, method_plot_names)

path = 'examples/data/german.data-numeric'
data = pd.read_csv(path, delim_whitespace = True, header = None)
X = data.iloc[:, :-1].values  # All columns except the last as features
y = data.iloc[:, -1].values   # Last column as the target

# The last column is the target, 1 is good and 2 is bad, we change that so 0 is good and 1 is bad 
y = np.where(y == 1, 0, np.where(y == 2, 1, y))

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

# make plots paper
plots_path = 'plots/paper'
if not os.path.exists(plots_path):
    os.makedirs(plots_path)

for experiment in experiments_results.keys():
    
    print("__________________________Experiment", experiment)
    # plot mse of mean and variance for gauss and ill_conditioned_gauss (2x2 plots)
    if experiment == 'Gaussian':
        true_mean = np.array([-4., -2., 0., 2., 4.])  # Mean of the multivariate Gaussian
        true_var = np.array([1, 1.5, 2., 2.5, 3.])  # Covariance matrix
        plot_mse(experiments_results, experiment, true_mean, metric='mean', method_fps=method_plot_names)
        plot_mse(experiments_results, experiment, true_var, metric='var', method_fps=method_plot_names)
    elif experiment == 'Ill-conditioned Gaussian':
        true_mean = np.zeros(100)
        key = random.PRNGKey(0)
        true_covar = create_gaussian(key, D=100)
        true_var = np.diag(true_covar)
        plot_mse(experiments_results, experiment, true_mean, metric='mean', method_fps=method_plot_names)
        plot_mse(experiments_results, experiment, true_var, metric='var', method_fps=method_plot_names)
    elif experiment == 'German Credit':
        # plot german credit auroc (1 plot)
        plot_roc_curve(experiments_results, experiment, X_test, y_test)
        create_classification_table(experiments_results, method_plot_names)

    # plot neffs/grad_eval_per iter for all, (4 plots)
    plot_neff_per_grad_evals(experiments_results, experiment, method_plot_names) 
        
## plt samples for banana (4 plots)
banana = BananaDistribution()

def plot_all_banana_samples(folder_path, methods, target):
    fig, ax = plt.subplots(figsize=(10, 10))
    x = np.linspace(-2.75, 2.75, 300)
    y = np.linspace(-2.0, 6.25, 300)
    X, Y = np.meshgrid(x, y)
    Z = vmap(target.log_prob, in_axes = (0))(np.array([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    ax.contourf(X, Y, Z, levels=20)
    for method, method_fp in zip(methods, methods):
        samples = load_numpy_files(f'{folder_path}/{method_fp}', file_type = 'samples', run=2)
        ax.scatter(samples[:, :, 0], samples[:, :, 1], alpha=0.5, label=method)
    ax.set_xlabel(r'$\theta_{(1)}$')
    ax.set_ylabel(r'$\theta_{(2)}$')
    ax.legend()
    plt.savefig(f'plots/paper/banana_samples_all.pdf', bbox_inches='tight', dpi=300)

    
# plot all bananas with offset
def plot_all_samples_offset(folder_path, methods, method_fps, target):
    fig, ax = plt.subplots(figsize=(20, 10))
    y = np.linspace(-2.75, 2.75, 300)
    x = np.linspace(-2.0, 16.25, 300)
    for method, method_fp in zip(methods, method_fps):
        # calculate offset for number in methods list
        offset = methods.index(method)
        samples = load_numpy_files(f'{folder_path}/{method}', file_type = 'samples', run=3)
        # set size proportional to target logpdf
        samples_eval = np.array([samples[:, :, 0].flatten(), samples[:, :, 1].flatten()]).T 
        sizes = np.exp(vmap(target.log_prob, in_axes = (0))(samples_eval))
        print("sizes", sizes)
        ax.scatter(samples[:, :, 1]+offset, samples[:, :, 0], alpha=0.5, label=method_fp, s=sizes)
    ax.set_xlabel(r'$\theta_{(1)}$')
    ax.set_ylabel(r'$\theta_{(2)}$')
    ax.legend()
    plt.savefig(f'plots/paper/banana_samples_all_offset.pdf', bbox_inches='tight', dpi=300)

plot_all_samples_offset('results/nbi_new_banana', methods, method_plot_names, banana)