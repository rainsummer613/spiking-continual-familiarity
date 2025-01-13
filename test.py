import argparse
import json
import multiprocessing as mp
import numpy as np
import os
import re
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.utils import create_logger
from src.config import metrics, metric_names_test, combinations_check
from src.model import Izhikevich
from src.experiment import ContinualFamiliarityPlastic

np.set_printoptions(suppress=True)


def get_score(test_file_path, param_file_path, metric, repeat_intervals_list, n_samples):
    """
    Evaluate models on different repeat intervals.
    Each model was optimized for a specific interval, but we evaluate how well it extrapolates to other repeat intervals
    Writes results to test log file

    Args:
        test_file_path (str): path to a file for test data
        param_file_path (str): path to a file with optimal parameters
        metric (str): metric name (e.g. rate or rsync)
        repeat_intervals_list (list)): list of repeat intervals to test extrapolation on
    """
    logger = create_logger(test_file_path)
    print(f'Logger {test_file_path} created')

    params = {}
    thresholds = {}
    print("param_file", param_file_path, os.path.isfile(param_file_path))
    if os.path.isfile(param_file_path):
        print("param_file FOUND", param_file_path)
        with open(param_file_path, 'r') as file:
            all_params = json.load(file)
            if "model_params" in all_params and "thresholds" in all_params:
                params = all_params["model_params"]
                thresholds = all_params["thresholds"]

    print(len(thresholds), len(params))
    if len(thresholds) > 0 and len(params) > 0:
        # check each repeat interval
        for interval in repeat_intervals_list:
            # params for continual familiarity data generation
            data_params = {"stimulus_size": params['exc_neurons'],
                           "repeat_interval": interval,
                           "p_repeat": 0.5,
                           "pattern_size": pattern_size,
                           "n_samples": n_samples,
                           }
            rsync_list = []
            sc_list = []

            for it in range(15):
                cf_experiment = ContinualFamiliarityPlastic(metrics=metrics, data=None, data_params=data_params,
                                                            input_rate=input_rate,
                                                            simulation_length=simulation_length,
                                                            simulation_length_test=simulation_length_test,
                                                            )
                print(f'START experiment: METRIC={metric} FILE={test_file_path} REP={interval}')
                model, res = cf_experiment.run(model_class=Izhikevich, model_params=params, thresholds=thresholds, optimize=False)
                print('score', res['rsync']['score'])

                m_rsync, m_sc = round(res['rsync']['score'], 4), round(res['sc']['score'], 4)
                rsync_list.append(m_rsync)
                sc_list.append(m_sc)


            rsync_median, sc_median = np.median(rsync_list), np.median(sc_list),
            rsync_mean, sc_mean = np.mean(rsync_list), np.mean(sc_list)
            rsync_std, sc_std = np.std(rsync_list), np.std(sc_list)
            rsync_25, sc_25 = np.percentile(rsync_list, 25), np.percentile(sc_list, 25)
            rsync_75, sc_75 = np.percentile(rsync_list, 75), np.percentile(sc_list, 75)

            log_text = f'REP={interval} ' \
                       f'MEAN_rsync={rsync_mean} MEAN_sc={sc_mean} ' \
                       f'MEDIAN_rsync={rsync_median} MEDIAN_sc={sc_median} ' \
                       f'STD_rsync={rsync_std} STD_sc={sc_std} ' \
                       f'25_rsync={rsync_25} 25_sc={sc_25} ' \
                       f'75_rsync={rsync_75} 75_sc={sc_75}'
            logger.info(log_text)
            print(f'METRIC={metric} FILE={test_file_path} ' + log_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--combination', type=int, default=0,
                        help="combination of a repeat interval, pattern size, metric and optimization iteration from config.py")
    parser.add_argument('-lo', '--log_overwrite', type=int, default=0,
                        help="overwrite existing log file")
    parser.add_argument('-po', '--param_overwrite', type=int, default=0,
                        help="overwrite file with optimal parameters")
    args = parser.parse_args()

    combination_idx = args.combination
    repeat_interval, pattern_size, metric, test_file_idx = combinations_check[combination_idx]
    metric_idx = metric_names_test.index(metric)

    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000

    n_samples = 250

    log_dir = f"data/{input_rate}_{simulation_length}_{simulation_length_test}/{repeat_interval}_{pattern_size}/logs"
    param_dir = f"data/{input_rate}_{simulation_length}_{simulation_length_test}/{repeat_interval}_{pattern_size}/params"

    multiproc = bool(args.multiproc)
    param_overwrite = bool(args.param_overwrite)
    log_overwrite = bool(args.log_overwrite)

    train_dir = os.path.join(log_dir, metric, 'train')
    cur_param_dir = os.path.join(param_dir, metric)
    os.makedirs(cur_param_dir, exist_ok=True)

    logger_dir = os.path.join(log_dir, metric, 'test')
    os.makedirs(logger_dir, exist_ok=True)

    print("LOG DIR", logger_dir)

    # test on all repeat intervals, multiprocessing recommended
    rep_max = 30
    processes = []
    if os.path.isdir(cur_param_dir) and len(os.listdir(cur_param_dir)) > 0:
        list_dir = cur_param_dir

    for f in os.listdir(list_dir):
        rep_min = 1
        test_path = os.path.join(logger_dir, f.split('.')[0] + '.log')
        is_ready = False

        if os.path.isfile(test_path):
            f_test = open(test_path, 'r').readlines()
            if len(f_test) >= rep_max:
                print("All repeat intervals are done", test_path, len(f_test))
                is_ready = True
            else:
                is_ready = False
                rep_list_file = [int(re.findall(r'REP=(\d+)', r)[0]) for r in f_test]
                if len(rep_list_file) > 0:
                    rep_min = max(rep_list_file) + 1
                    print("Next repeat interval", rep_min)

        if is_ready is False:
            print(metric, f, 'REP', rep_min, rep_max)
            rep_list = np.arange(rep_min, rep_max + 1)
            p = mp.Process(target=get_score, args=(f, logger_dir, train_dir, cur_param_dir,
                                                       param_overwrite, metric, rep_list,
                                                       n_samples
                                                       ),
                            name=f'Process_{f[:-4]}')
            processes.append(p)
            p.start()
            print(f'{p.name} started')

    for p in processes:
        p.join()
        print(f'{p.name} finished')

    for p in processes:
        p.terminate()

