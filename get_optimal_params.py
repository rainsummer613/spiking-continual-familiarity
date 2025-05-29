import argparse
import numpy as np
import os
import re
import json
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import combinations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-po', '--param_overwrite', type=int, default=1,
                        help="overwrite file with optimal parameters")
    args = parser.parse_args()
    param_overwrite = bool(args.param_overwrite)

    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000

    n_samples = 250

    for repeat_interval, pattern_size, p_match, plasticity_type in combinations:
        log_dir = f"data/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}/{repeat_interval}_{pattern_size}/{p_match}/logs"
        param_dir = f"data/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}/{repeat_interval}_{pattern_size}/{p_match}/params"

        if not os.path.isdir(log_dir):
            print(log_dir, "DOES NOT EXIST")
        else:
            for metric in os.listdir(log_dir):
                train_dir = os.path.join(log_dir, metric, "train")

                if os.path.isdir(train_dir):
                    cur_param_dir = os.path.join(param_dir, metric)
                    os.makedirs(cur_param_dir, exist_ok=True)

                    for log_file in [f for f in os.listdir(train_dir) if f[-3:] == "log"]:

                        log_file_path = os.path.join(train_dir, log_file)
                        with open(log_file_path, 'r') as file:
                            rows = file.readlines()

                            if len(rows) > 0:
                                # use a single metric (rsync or rate)
                                if "_" not in metric:
                                    vals = [float(re.findall(metric + r': ([10]\.\d+)', r)[0]) for r in rows]
                                    vals = [el if 0 < el < 1.0 else 0 for el in vals]

                                # use two metrics (rsync and rate)
                                else:
                                    metric1, metric2 = metric.split('_')
                                    vals1 = [float(re.findall(metric1 + r': ([10]\.\d+)', r)[0]) for r in rows]
                                    vals1 = [el if 0 < el < 1.0 else 0 for el in vals1]
                                    vals2 = [float(re.findall(metric2 + r': ([10]\.\d+)', r)[0]) for r in rows]
                                    vals2 = [el if 0 < el < 1.0 else 0 for el in vals2]

                                    vals = []
                                    for i in range(len(vals1)):
                                        if vals1[i] > 0 and vals2[i] > 0:
                                            vals.append(np.mean([vals1[i], vals2[i]]))
                                        else:
                                            vals.append(0)

                                # find parameter set with the best fitness, use it for testing
                                json_content = {}
                                best_id = np.array(vals).argmax()
                                r_params = rows[best_id].split(' PARAMS ')[1].split(' ')
                                params = {r_params[i][:-1]: float(r_params[i + 1]) for i in range(0, len(r_params), 2)}
                                json_content["model_params"] = params

                                r_thresholds = re.findall(r'THRESHOLD (.+); PARAMS', rows[best_id])[0].split(' ')
                                thresholds = {r_thresholds[i][:-1]: float(r_thresholds[i + 1]) for i in
                                              range(0, len(r_thresholds), 2)}
                                json_content["thresholds"] = thresholds

                                json_param_path = os.path.join(cur_param_dir, log_file.split(".")[0]) + ".json"
                                print(f"{json_param_path} OPTIMAL PARAMS SAVED TO JSON")

                                if param_overwrite or not os.path.isfile(json_param_path):
                                    with open(json_param_path, "w") as param_file:
                                        json.dump(json_content, param_file)
