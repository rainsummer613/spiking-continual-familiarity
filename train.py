import argparse
import numpy as np
import os
import re
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import combinations_check, params_fixed, params_change, metrics, metric_names_test
from src.utils import generate_recog_data, create_logger
from src.model import Izhikevich
from src.experiment import ContinualFamiliarityPlastic
from src.genetic import GeneticOptimizer

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--combination', type=int, default=0,
                        help="combination of a repeat interval, pattern size and optimization iteration from config.py")
    parser.add_argument('-lo', '--overwrite', type=int, default=0,
                        help="overwrite optimization log file if it exists")
    args = parser.parse_args()

    overwrite = bool(args.overwrite)
    logging = bool(args.logging)
    combination_idx = int(args.combination)
    repeat_interval, pattern_size, metric, file_idx = combinations_check[combination_idx]
    metric_idx = metric_names_test.index(metric)

    spatiotemporal = False
    n_jitter = 0

    n_gens = 300
    start_gen = 1
    n_samples = 250
    generation_size = 12
    target_acc = 1.0
    n_early_stop = 10

    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000

    """
    # Initial parameters sets for the first generation
    init_sets = [
        {
         'plasticity_scale': 0.5,
         'total_synaptic_weight': 70.0,
         'weight_norm_freq': 20,
         'tau_stdp': 5,
         'trace_scale': 0.3
        },

        {
         'plasticity_scale': 0.7,
         'total_synaptic_weight': 40.0,
         'weight_norm_freq': 140,
         'tau_stdp': 3,
         'trace_scale': 0.6
        },

        {
         'plasticity_scale': 0.3,
         'total_synaptic_weight': 30.0,
         'weight_norm_freq': 100,
         'tau_stdp': 10,
         'trace_scale': 0.05
        },

        {
         'plasticity_scale': 0.4,
         'total_synaptic_weight': 60.0,
         'weight_norm_freq': 60,
         'tau_stdp': 6,
         'trace_scale': 1.0
        },
    ]
    """
    init_sets = None

    # params for continual familiarity data generation
    data_params = {"stimulus_size": params_fixed['exc_neurons'],
                   "repeat_interval": repeat_interval,
                   "p_repeat": 0.5,
                   "pattern_size": pattern_size,
                   "n_samples": n_samples,
                   }
    # generate simple familiarity data
    train_data = list(generate_recog_data(**data_params))

    metric_weights = {m: 0 for m in metric_names_test}
    metric_weights[metric] = 1
    metrics_valid = [m for m in metric_weights if metric_weights[m] > 0]
    metric = '_'.join(metrics_valid)

    data_dir = f"data/{input_rate}_{simulation_length}_{simulation_length_test}/{repeat_interval}_{pattern_size}"
    log_dir = os.path.join(data_dir, "logs")
    logger_dir = os.path.join(log_dir, metric, 'train')
    os.makedirs(logger_dir, exist_ok=True)

    init_fits = []
    to_train = True

    # Create logger
    logs_file_path = os.path.join(logger_dir, f'gen_opt_{file_idx}_{n_samples}.log')
    if logging is False:
        print('Train without logging')
        logger = None

    # if log file does not exist
    if not os.path.isfile(logs_file_path):
        logger = create_logger(logs_file_path)
        print(f'Logger {logs_file_path} created')

    else:
        # if log file exists
        print(f'Logger {logs_file_path} exists')
        if overwrite is True:
            os.remove(logs_file_path)
            logger = create_logger(logs_file_path)
            print(f'Logger {logs_file_path} overwritten')
        else:
            # continue logging into existing log file
            logger = create_logger(logs_file_path)
            rows = open(logs_file_path).readlines()
            if 0 < len(rows) and len(rows[0]) > 1:
                gens = [int(re.findall('GEN (\d+)', r)[0]) for r in rows]

                if max(gens) < n_gens:
                    param_rows_split = [row.split(' PARAMS ')[1].split(' ') for row in rows[-3:]]
                    init_sets = [{r_split[i][:-1]: float(r_split[i + 1]) for i in range(0, len(r_split), 2)}
                                        for r_split in param_rows_split]
                    start_gen = max(gens) + 1

                    fitness_rows_split = [re.findall(r"FITNESS (.+) THRESHOLD", row)[0].split(' ') 
                                                for row in rows[-min(len(rows), n_early_stop*3):]]
                    fitness_rows_max = [max([float(el.strip(";")) for i, el in enumerate(row) if i % 2 == 1])
                                              for row in fitness_rows_split[-3:]]
                    if max(fitness_rows_max) >= target_acc:
                        to_train = False
                        print(f'NO OPTIMIZATION REQUIRED. Already achieved max fitness {max(fitness_rows_max)} with target fitness {target_acc}')
                    else:
                        for gen_i_start in range(0, len(fitness_rows_split), 3):
                            gen_i_end = gen_i_start + 3
                            fitness_gen_max = max([max([float(el.strip(";")) for i, el in enumerate(row) if i % 2 == 1])
                                for row in fitness_rows_split[gen_i_start:gen_i_end]])
                            init_fits.append(fitness_gen_max)
                        if len(init_fits) >= n_early_stop and all(i >= j
                                for i, j in zip(init_fits[-n_early_stop:], init_fits[-n_early_stop + 1:])):
                            print(f'NO OPTIMIZATION REQUIRED. No improvements for {n_early_stop} generations')
                            to_train = False
                else:
                    print(f'NO OPTIMIZATION REQUIRED. Already optimized for {n_gens} generations')
                    to_train = False

    if to_train:
        # start optimization
        cf_experiment = ContinualFamiliarityPlastic(metrics=metrics, data=None, data_params=data_params,
                                             input_rate=input_rate,
                                             simulation_length=simulation_length, 
                                             simulation_length_test=simulation_length_test,
                                             )

        # TO DEL
        opt = GeneticOptimizer(metric_weights=metric_weights,
                           generation_size=generation_size,
                           experiment=cf_experiment, model_class=Izhikevich, 
                           params_change=params_change, params_fixed=params_fixed,
                           logger=logger)
        if len(init_fits) > n_early_stop:
            init_fits = init_fits[-n_early_stop:]
        opt.fit(n_gens, start_gen, target_acc, n_early_stop, init_sets, init_fits)

