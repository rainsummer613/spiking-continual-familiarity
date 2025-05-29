import numpy as np
from utils import exp_convolve
from sklearn.metrics import confusion_matrix
from bct import modularity, centrality, clustering

def measure_rsync(firings):
    firings = np.apply_along_axis(exp_convolve, 1, firings)
    num = np.var(np.mean(firings, axis=0))  # spatial mean across cells, at each time
    den = np.mean(np.var(firings, axis=1))  # variance over time of each cell
    return num / (den + 1e-100)

def measure_mean_sc(firings):
    sums = firings.sum(axis=1)
    return sums.mean()

def measure_max_sc(firings):
    sums = firings.sum(axis=1)
    return sums.max()
def measure_gini(network):
    x = np.array(network.flatten())
    x = np.sort(x)
    n = len(x)
    return (2 / n) * np.sum((np.arange(1, n + 1) - (n + 1) / 2) * x) / np.sum(x)

def measure_transitivity(network):
    return round(clustering.transitivity_wd(network), 4)

def measure_centrality(network):
    return round(centrality.betweenness_wei(network).mean(), 4)

def measure_modularity(network):
    ci, modul = modularity.modularity_dir(network)
    return round(modul, 4)

def measure_participation(network):
    ci, modul = modularity.modularity_dir(network)
    return round(centrality.participation_coef(network, ci, degree='in').mean(), 4)

def accuracy_abbott_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tp_rate = tp / y_true.sum()
    fp_rate = fp / (len(y_true) - y_true.sum())

    f = 1 - y_true.sum() / len(y_true) # 2/3
    return (1 - f) * tp_rate + f * (1 - fp_rate)
