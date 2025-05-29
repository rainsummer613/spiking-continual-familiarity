import argparse
import numpy as np
import os
import re
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import params_fixed, params_change, metrics_func, metric_list, data_dir, plasticity_sym
from src.utils import generate_recog_data, create_logger
from src.model import Izhikevich
from src.experiment import ContinualFamiliarityPlastic
from src.genetic import GeneticOptimizer

np.set_printoptions(suppress=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--combination', type=int, default=4,
                        help="combination of a repeat interval, pattern size and optimization iteration from config.py")
    parser.add_argument('-lo', '--log_overwrite', type=int, default=1,
                        help="overwrite optimization log file if it exists")
    args = parser.parse_args()

    log_overwrite = bool(args.log_overwrite)
    combination_idx = int(args.combination)
    file_idx, repeat_interval, pattern_size, p_match, metric, plasticity_type = 0, 6, 20, 0.0, "rsync", "anti_hebb"  # combinations_check[combination_idx]
    metric_idx = metric_list.index(metric)

    n_gens = 500
    start_gen = 1
    n_samples = 250
    generation_size = 12
    target_acc = 1.0
    n_early_stop = 10

    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000

    metric_weights = {m: 0 for m in metric_list}
    metric_weights[metric] = 1
    metrics_valid = [m for m in metric_weights if metric_weights[m] > 0]
    metric = '_'.join(metrics_valid)

    exp_data_dir = f"{data_dir}/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}/{repeat_interval}_{pattern_size}/{p_match}"
    log_dir = os.path.join(exp_data_dir, "logs")
    logger_dir = os.path.join(log_dir, metric, 'train')
    os.makedirs(logger_dir, exist_ok=True)

    # Initial parameters sets for the first generation
    init_sets = [
        {
         'learning_rate': -0.25,
         'total_incoming_weight': 200.0,
         'normalization_interval': 20,
         'trace_memory': 15,
         'trace_increase': 1.0,
         'minimal_weight': 0.0025,
         #'weight_growth': 0.005
        },

        {
         'learning_rate': -0.7,
         'total_incoming_weight': 70.0,
         'normalization_interval': 140,
         'trace_memory': 13,
         'trace_increase': 0.6,
         'minimal_weight': 0.01,
         #'weight_growth': 0.01
        },

        {
         'learning_rate': -0.3,
         'total_incoming_weight': 130.0,
         'normalization_interval': 100,
         'trace_memory': 10,
         'trace_increase': 0.05,
         'minimal_weight': 0.02,
         #'weight_growth': 0.001
        },

        {
         'learning_rate': -0.4,
         'total_incoming_weight': 100.0,
         'normalization_interval': 60,
         'trace_memory': 6,
         'trace_increase': 1.0,
         'minimal_weight': 0.5,
         #'weight_growth': 0.1
        },
    ]
    #init_sets = None

    # params for continual familiarity data generation
    data_params = {"stimulus_size": params_fixed[plasticity_type]['exc_neurons'],
                   "repeat_interval": repeat_interval,
                   "p_repeat": 0.5,
                   "pattern_size": pattern_size,
                   "n_samples": n_samples,
                   "p_match": p_match
                   }
    # generate simple familiarity data
    train_data = list(generate_recog_data(**data_params))

    init_fits = []
    to_train = True

    # Create logger
    logs_file_path = os.path.join(logger_dir, f'gen_opt_{file_idx}_{n_samples}.log')

    # if log file does not exist
    if not os.path.isfile(logs_file_path):
        logger = create_logger(logs_file_path)
        print(f'Logger {logs_file_path} created')
    else:
        # if log file exists
        print(f'Logger {logs_file_path} exists')
        if log_overwrite is True:
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
        cf_experiment = ContinualFamiliarityPlastic(metrics=metrics_func, data=None, data_params=data_params,
                                             input_rate=input_rate,
                                             simulation_length=simulation_length, 
                                             simulation_length_test=simulation_length_test,
                                             )

        opt = GeneticOptimizer(metric_weights=metric_weights,
                           generation_size=generation_size,
                           experiment=cf_experiment, model_class=Izhikevich,
                           plasticity_type=plasticity_type,
                           plasticity_sym=plasticity_sym,
                           params_change=params_change[plasticity_type], params_fixed=params_fixed[plasticity_type],
                           logger=logger)
        if len(init_fits) > n_early_stop:
            init_fits = init_fits[-n_early_stop:]
        opt.fit(n_gens, start_gen, target_acc, n_early_stop, init_sets, init_fits)

