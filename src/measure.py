import numpy as np
from utils import exp_convolve, neo_spike_transform
from sklearn.metrics import confusion_matrix

def measure_rsync(firings):
    firings = np.apply_along_axis(exp_convolve, 1, firings)
    num = np.var(np.mean(firings, axis=0))  # spatial mean across cells, at each time
    den = np.mean(np.var(firings, axis=1))  # variance over time of each cell
    return num / (den + 1e-100)

def measure_mean_sc(firings):
    sums = firings.sum(axis=1)
    return sums.mean() / 1000  # firings.shape[-1] * delta_t * 1000

def measure_max_sc(firings):
    sums = firings.sum(axis=1)
    return sums.max() / 1000  # firings.shape[-1] * delta_t * 1000

def accuracy_abbott_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tp_rate = tp / y_true.sum()
    fp_rate = fp / (len(y_true) - y_true.sum())

    f = 1 - y_true.sum() / len(y_true) # 2/3
    return (1 - f) * tp_rate + f * (1 - fp_rate)
