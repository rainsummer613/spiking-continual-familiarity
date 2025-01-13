import itertools
import os

from measure import measure_rsync, measure_mean_sc

data_dir = "data"
log_dir = os.path.join(data_dir, "logs")
plot_dir = os.path.join(data_dir, "plots")
param_dir = os.path.join(data_dir, "params")

params_fixed = {
                'exc_neurons': 100,
                'inh_neurons': 0,
                'delta_t': 1.0,
                'threshold': 30,
                'voltage_noise': 0.6,
                't_refr': 4,
                'external_input_scale': 21.0,
                'lateral_input_scale': 1.0,
                }

params_change = {
                 'plasticity_scale': {'min': 0.05, 'max': 1.0, 'step': 0.05},
                 'total_synaptic_weight': {'min': 20.0, 'max': 200.0, 'step': 4.0},
                 'weight_norm_freq': {'min': 0.0, 'max': 100.0, 'step': 1.0},
                 'tau_stdp': {'min': 5, 'max': 20, 'step': 1},
                 'trace_scale': {'min': 0.01, 'max': 1.0, 'step': 0.02}
                }

repeat_interval_list = [3, 6, 10]
pattern_size_list = [5, 10, 20]
combinations = list(itertools.product(repeat_interval_list, pattern_size_list))

metrics = {'sc': {'func': measure_mean_sc, 'kwargs': {}},
           'rsync': {'func': measure_rsync, 'kwargs': {}},
           }
metric_names_test = ['sc', 'rsync'] 

optimization_iter_list = list(range(10))
combinations_check = list(itertools.product(repeat_interval_list, pattern_size_list, metric_names_test, optimization_iter_list))
