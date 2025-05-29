import os
import pandas as pd
import seaborn as sns
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import multiprocessing as mp

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.utils import generate_recog_data, get_continuous_cmap, generate_poisson_input, generate_corr_data
from src.config import network_stats_func
from src.model import Izhikevich
np.set_printoptions(suppress=True)

rsync_pattern = f"MEAN_rsync=([\d\.]+)"
sc_pattern = f"MEAN_sc=([\d\.]+)"

key_mapping = {
    "total_synaptic_weight": "total_incoming_weight",
    "t_refr": "refractory_period",
    "plasticity_scale": "learning_rate",
    "weight_norm_freq": "normalization_interval",
    "tau_stdp": "trace_memory",
    "trace_scale": "trace_increase"
}

def compute_gini(x):
    # Ensure x is a 1D numpy array
    x = np.array(x.flatten())
    # Sort the array
    x = np.sort(x)
    # Calculate the Gini coefficient
    n = len(x)
    cumulative_values = np.cumsum(x)
    return (2 / n) * np.sum((np.arange(1, n + 1) - (n + 1) / 2) * x) / np.sum(x)

def calculate_network_stats(dir_all, exp_params_list, model_params_list, metrics_list, network_stats_list,
                            input_rate, simulation_length, plasticity_type, plasticity_sym,
                            sparseness, repeat_interval, p_match, opt, n_samples):
    process_name = mp.current_process().name

    param_dir = f"{dir_all}/{repeat_interval}_{sparseness}/{p_match}/params/{opt}"
    test_log_dir = f"{dir_all}/{repeat_interval}_{sparseness}/{p_match}/logs/{opt}/test"

    plot_dir = f"{dir_all}/{repeat_interval}_{sparseness}/{p_match}/plots/test/{opt}"
    os.makedirs(plot_dir, exist_ok=True)

    out_exp_params = {k: [] for k in exp_params_list}
    out_model_params = {k: [] for k in model_params_list}
    out_metrics = {k: [] for k in metrics_list}
    out_network_stats = {k: [] for k in network_stats_list}

    param_files = os.listdir(param_dir)
    log_files = os.listdir(test_log_dir)

    i = 0
    for f_param, f_log in list(zip(param_files, log_files)):

        print(process_name, param_dir, f_param)
        param_file_path = os.path.join(param_dir, f_param)
        log_file_path = os.path.join(test_log_dir, f_log)

        # upload model params from the file
        with open(param_file_path, 'r') as f:
            params = json.load(f)

        #if plasticity_type == "hebb":
        #    params["model_params"] = {
        #        key_mapping.get(k, k): v for k, v in params["model_params"].items()
        #    }

        with open(log_file_path, 'r') as f:
            lines = f.readlines()

        if params and lines:
            log_rsync = []
            log_sc = []
            for line in lines:
                rsync = re.findall(rsync_pattern, line)[0]
                sc = re.findall(sc_pattern, line)[0]
                log_rsync.append(rsync)
                log_sc.append(sc)
            rsync_gen = round(np.mean([float(metric_val) for metric_val in log_rsync]), 4)
            sc_gen = round(np.mean([float(metric_val) for metric_val in log_sc]), 4)

            model_params = params["model_params"]
            if "plasticity_type" not in model_params:
                model_params["plasticity_type"] = plasticity_type
            if "plasticity_sym" not in model_params:
                model_params["plasticity_sym"] = plasticity_sym

            # generate experimental data: stimuli + labels
            # params for continual familiarity data generation
            data_params = {"stimulus_size": int(model_params['exc_neurons']),
                           "repeat_interval": 6,
                           "p_repeat": 0.5,
                           "pattern_size": sparseness,
                           "n_samples": n_samples,
                           "p_match": p_match}

            # generate simple familiarity data
            data = list(generate_corr_data(**data_params))
            model = Izhikevich(**model_params)
            all_input = []

            for t, d in enumerate(data):
                x, y = d

                poisson_input = generate_poisson_input(x, input_rate, model.delta_t, simulation_length)
                all_input.append(poisson_input)

                voltage, firings = model.simulate(length=simulation_length,
                                                  external_input=poisson_input,
                                                  plastic=True, verbose=True, verbose_freq=3000)

            connectivity_matrix = model.connectivity_matrix

            try:
                for network_stat in network_stats_list:
                    out_network_stats[network_stat].append(network_stats_func[network_stat](connectivity_matrix))

                out_metrics["rsync"].append(rsync_gen)
                out_metrics["sc"].append(sc_gen)

                out_exp_params["repeat_interval"].append(repeat_interval)
                out_exp_params["sparseness"].append(sparseness)
                out_exp_params["readout_method"].append(opt)
                out_exp_params["input_correlation"].append(p_match)

                for param in out_model_params:
                    out_model_params[param].append(model_params[param])

                #print(process_name, i, f"{repeat_interval}_{sparseness}_{p_match} {opt}", ":",
                #      rsync_gen, sc_gen, "::",
                #      gini, modul, trans, part, centr_btw)
            except Exception as e:
                print("ERROR", str(e), "::", param_dir, f_param)

            i += 1

        plot_matrix = model.connectivity_matrix / model.connectivity_matrix.sum(axis=1, keepdims=1)
        print("CONNECTIVITY MAX", sparseness, repeat_interval, p_match, opt, "::", round(model.connectivity_matrix.max(), 4), round(plot_matrix.max(), 4))
        hex_codes_matrix = ['#000000', '#6ccdc0', '#eb7341', '#ffffff']
        cmap_matrix = get_continuous_cmap(hex_codes_matrix)
        fig_con, ax = plt.subplots(figsize=(20, 20))
        ax.imshow(plot_matrix, cmap=cmap_matrix, vmin=0, vmax=0.25)
        ax.axis('off')
        ax.invert_yaxis()

        for ext in ("png", "svg"):
            fig_con.savefig(os.path.join(plot_dir, f"conn_{i}.{ext}"),
                            bbox_inches="tight")

        #if repeat_interval == 3 and sparseness == 5:
        #    cbar = ax.figure.colorbar(img_con, ax=ax)
        #    cbar.ax.tick_params(labelsize=70)

        #img_con.set_clim(0, 0.18)
        ax.invert_yaxis()
        ax.xaxis.set_tick_params(labelsize=75)
        ax.yaxis.set_tick_params(labelsize=75)

    def clean_array(arr):
        arr = np.array(arr)
        return np.nan_to_num(arr, nan=0.0)

    for stat in out_network_stats:
        out_network_stats[stat] = clean_array(out_network_stats[stat])

    return out_exp_params, out_model_params, out_metrics, out_network_stats

if __name__ == "__main__":
    n_samples = 100
    simulation_length = 1000
    simulation_length_test = 1000
    input_rate = 100

    plasticity_sym = False
    data_dir = "data"
    for plasticity_type in ("anti_hebb", "hebb"):

        dir_all = f"{data_dir}/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}"

        exp_params_list = ["sparseness", "repeat_interval", "input_correlation", "readout_method"]
        exp_params_plot = [k.replace("_", " ").capitalize() for k in exp_params_list]

        if plasticity_type == "anti_hebb":
            model_params_list = ["trace_memory", "learning_rate", "total_incoming_weight",
                                 "minimal_weight", "normalization_interval"]
        else:
            model_params_list = ["trace_memory", "learning_rate", "total_incoming_weight",
                                 "normalization_interval"]
        model_params_plot = [k.replace("_", " ").capitalize() for k in model_params_list]

        metrics_list = ["rsync", "sc"]
        metrics_plot = ["Rsync gen.", "Spike count gen."]

        network_stats_list = ["gini_index", "transitivity", "betwenness_centrality"]
        network_stats_plot = [k.replace("_", " ").capitalize() for k in network_stats_list]

        exp_params_dict = {k: [] for k in exp_params_list}
        model_params_dict = {k: [] for k in model_params_list}
        network_stats_dict = {k: [] for k in network_stats_list}
        metrics_dict = {k: [] for k in metrics_list}

        tasks = [(dir_all, exp_params_list, model_params_list, metrics_list, network_stats_list,
                  input_rate, simulation_length, plasticity_type, plasticity_sym,
                  sparseness, repeat_interval, p_match, opt, n_samples)
                 for sparseness in (5, 10, 20)
                 for repeat_interval in (3, 6, 10)
                 for p_match in (0.0,)
                 for opt in ("rsync", "sc")]

        stats_dir = f"{dir_all}/stats"
        os.makedirs(stats_dir, exist_ok=True)

        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.starmap(calculate_network_stats, tasks)

        for out_exp_params, out_model_params, out_metrics, out_network_stats in results:
            for k, v in out_exp_params.items():
                exp_params_dict[k].extend(v)

            for k, v in out_model_params.items():
                model_params_dict[k].extend(v)

            for k, v in out_metrics.items():
                metrics_dict[k].extend(v)

            for k, v in out_network_stats.items():
                network_stats_dict[k].extend(v)

        df = pd.DataFrame()
        for k, v in exp_params_dict.items():
            df[k.replace("_", " ").capitalize()] = v

        for k, v in model_params_dict.items():
            df[k.replace("_", " ").capitalize()] = v

        for k, v in network_stats_dict.items():
            df[k.replace("_", " ").capitalize()] = v

        df["Rsync gen."] = metrics_dict["rsync"]
        df["Spike count gen."] = metrics_dict["sc"]
        df["Sparseness"] = df["Sparseness"].replace({5: 0.9, 10: 0.8, 20: 0.6})

        df.to_excel(f"{stats_dir}/network_stats.xlsx", index=False)

        # Plot connectivity statistics and their correlations
        corr = pd.DataFrame()
        for a in model_params_plot + metrics_plot:
            for b in network_stats_plot:
                corr.loc[a, b] = df.corr(numeric_only=True).loc[a, b]

        fig, ax = plt.subplots()
        ax = sns.heatmap(corr.T, cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-.87, vmax=.95, annot=True)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{stats_dir}/param_conn_corr.{ext}")

        color_dict = {3: "#e79573", 6: "#91e3d8", 10: "#159fa1"}

        for stat in network_stats_plot:
            fig, ax = plt.subplots()
            ax = sns.boxplot(data=df, x="Sparseness", y=stat, hue="Repeat interval",
                        flierprops={"marker": "o", "markerfacecolor": "None"},
                        palette=color_dict)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylabel("", fontsize=23, labelpad=15)
            ax.set_xlabel("", fontsize=23, labelpad=15)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.set_title(stat, fontsize=23)

            leg = ax.legend(title="Repeat interval R", fontsize=20, title_fontsize=22)

            fig.tight_layout()

            plot_dir = f"{stats_dir}/sc/boxplots"
            os.makedirs(plot_dir, exist_ok=True)
            for ext in ("png", "svg"):
                fig.savefig(f"{plot_dir}/{stat}.{ext}")

        # Pedict generalizability from connectivity

        # predict Rsync using Linear regression
        X = df[df['Readout method'] == 'rsync']
        y = X["Rsync gen."]
        X = X[network_stats_plot]

        model = ElasticNet()
        cv = KFold(n_splits=10, shuffle=True)

        rmse_all = []
        r2_all = []

        for n in range(50):
            r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
            rmse = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

            rmse = -round(np.mean(rmse), 4)
            r2 = round(np.mean(r2), 4)
            rmse_all.append(rmse)
            r2_all.append(r2)

        print("rmse", np.mean(rmse), "r2", np.mean(r2))
        print()

        model = ElasticNet()
        model.fit(X, y)

        coefs = {}
        for i, p in enumerate(X.columns):
            coefs[p] = model.coef_[i]  #model.feature_importances_[i]

        coefs_abs = {k: abs(v) for k, v in coefs.items()}
        print(f"COEFFICIENTS:")
        for k in sorted(coefs_abs, key=coefs_abs.get, reverse=True):
            print(k, round(coefs[k], 4))

        # predict Rsync using Decision tree
        X = df[df['Readout method'] == 'rsync']
        y = X["Rsync gen."]
        X = X[network_stats_plot]

        model = ElasticNet()
        cv = KFold(n_splits=10, shuffle=True)

        rmse_all = []
        r2_all = []

        for n in range(50):
            r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
            rmse = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

            rmse = -round(np.mean(rmse), 4)
            r2 = round(np.mean(r2), 4)
            rmse_all.append(rmse)
            r2_all.append(r2)

        print("rmse", np.mean(rmse), "r2", np.mean(r2))
        print()

        model = ElasticNet()
        model.fit(X, y)

        coefs = {}
        for i, p in enumerate(X.columns):
            coefs[p] = model.coef_[i]  #model.feature_importances_[i]

        coefs_abs = {k: abs(v) for k, v in coefs.items()}
        print("COEFFICIENTS:")
        for k in sorted(coefs_abs, key=coefs_abs.get, reverse=True):
            print(k, round(coefs[k], 4))

        # Predict Spike count using Linear regression
        X = df[df['Readout method'] == 'sc']
        y = X["Spike count gen."]
        X = X[network_stats_plot]

        model = ElasticNet()
        cv = KFold(n_splits=10, shuffle=True)

        rmse_all = []
        r2_all = []

        for n in range(50):
            r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
            rmse = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

            rmse = -round(np.mean(rmse), 4)
            r2 = round(np.mean(r2), 4)
            rmse_all.append(rmse)
            r2_all.append(r2)

        print("rmse", np.mean(rmse), "r2", np.mean(r2))
        print()

        model = ElasticNet()
        model.fit(X, y)

        coefs = {}
        for i, p in enumerate(X.columns):
            coefs[p] = model.coef_[i]  #model.feature_importances_[i]

        coefs_abs = {k: abs(v) for k, v in coefs.items()}
        print("COEFFICIENTS:")
        for k in sorted(coefs_abs, key=coefs_abs.get, reverse=True):
            print(k, round(coefs[k], 4))

        # Predict Spike count using Decision tree
        X = df[df['Readout method'] == 'sc']
        y = X["Spike count gen."]
        X = X[network_stats_plot]

        max_depth = 3
        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=10, min_samples_leaf=5)

        rmse_all = []
        r2_all = []

        for n in range(50):
            r2 = cross_val_score(model, X, y, cv=cv, scoring='r2')
            rmse = cross_val_score(model, X, y, cv=cv, scoring='neg_root_mean_squared_error')

            rmse = -round(np.mean(rmse), 4)
            r2 = round(np.mean(r2), 4)
            rmse_all.append(rmse)
            r2_all.append(r2)

        print("rmse", np.mean(rmse), "r2", np.mean(r2))
        print()

        model = DecisionTreeRegressor(max_depth=max_depth, min_samples_split=10, min_samples_leaf=5)
        model.fit(X, y)

        coefs = {}
        for i, p in enumerate(X.columns):
            coefs[p] = model.feature_importances_[i]

        coefs_abs = {k: abs(v) for k, v in coefs.items()}
        print("COEFFICIENTS:")
        for k in sorted(coefs_abs, key=coefs_abs.get, reverse=True):
            print(k, round(coefs[k], 4))
            
    