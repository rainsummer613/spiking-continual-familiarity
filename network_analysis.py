import os
import pandas as pd
import seaborn as sns
import sys
import re
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import multiprocessing as mp

from bctpy import modularity, centrality, clustering

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.utils import generate_recog_data, get_continuous_cmap, generate_poisson_input, jitter_spikes
from src.model import Izhikevich
np.set_printoptions(suppress=True)

rsync_pattern = f"MEAN_rsync=([\d\.]+)"
sc_pattern = f"MEAN_sc=([\d\.]+)"

n_samples = 500
simulation_length = 1000
simulation_length_test = 1000
input_rate = 100

dir_all = f"data/{input_rate}_{simulation_length}_{simulation_length_test}"
save_dir = f"stats/{input_rate}_{simulation_length}_{simulation_length_test}"
os.makedirs(save_dir, exist_ok=True)

def compute_gini(x):
    # Ensure x is a 1D numpy array
    x = np.array(x.flatten())
    # Sort the array
    x = np.sort(x)
    # Calculate the Gini coefficient
    n = len(x)
    cumulative_values = np.cumsum(x)
    return (2 / n) * np.sum((np.arange(1, n + 1) - (n + 1) / 2) * x) / np.sum(x)

def calculate_network_stats(sparseness, repeat_interval, opt):
    process_name = mp.current_process().name

    param_dir = f"{dir_all}/{repeat_interval}_{sparseness}/params/{opt}"
    test_log_dir = f"{dir_all}/{repeat_interval}_{sparseness}/logs/{opt}/test"

    out_gini = []
    out_modul = []
    out_trans = []
    out_part = []
    out_centr_btw = []

    out_acc_rsync = []
    out_acc_sc = []

    out_repeat_interval = []
    out_sparseness = []
    out_opt = []

    out_trace_memory = []
    out_trace_increase = []
    out_stdp_update = []
    out_total_weight = []
    out_norm_interval = []

    plot_dir = f"{save_dir}/data/{opt}"
    os.makedirs(plot_dir, exist_ok=True)

    param_files = os.listdir(param_dir)
    log_files = os.listdir(test_log_dir)

    i = 0
    for f_param, f_log in list(zip(param_files, log_files)):
        param_file_path = os.path.join(param_dir, f_param)
        log_file_path = os.path.join(test_log_dir, f_log)

        # upload model params from the file
        with open(param_file_path, 'r') as f:
            params = json.load(f)

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

            if "input_rate" in model_params:
                model_params.pop("input_rate")

            # generate experimental data: stimuli + labels
            # params for continual familiarity data generation
            data_params = {"stimulus_size": model_params['exc_neurons'],
                           "repeat_interval": repeat_interval,
                           "p_repeat": 0.5,
                           "pattern_size": sparseness,
                           "n_samples": n_samples}

            # generate simple familiarity data
            data = list(generate_recog_data(**data_params))
            model = Izhikevich(**model_params)
            all_input = []

            for t, d in enumerate(data):
                x, y = d

                if y == 1 and spatiotemp_exp is True:
                    poisson_input = np.apply_along_axis(jitter_spikes, 1, all_input[t - repeat_interval])
                else:
                    poisson_input = generate_poisson_input(x, input_rate, model.delta_t, simulation_length)
                all_input.append(poisson_input)

                voltage, firings = model.simulate(length=simulation_length,
                                                  external_input=poisson_input,
                                                  plastic=True, verbose=True, verbose_freq=3000)

            connectivity_matrix = model.connectivity_matrix
            gini = round(compute_gini(connectivity_matrix), 4)
            ci, modul = modularity.modularity_dir(connectivity_matrix)
            modul = round(modul, 4)
            trans = round(clustering.transitivity_wd(connectivity_matrix), 4)
            part = round(centrality.participation_coef(connectivity_matrix, ci, degree='in').mean(), 4)
            centr_btw = round(centrality.betweenness_wei(connectivity_matrix).mean(), 4)

            out_gini.append(gini)
            out_modul.append(modul)
            out_trans.append(trans)
            out_part.append(part)
            out_centr_btw.append(centr_btw)

            out_acc_rsync.append(rsync_gen)
            out_acc_sc.append(sc_gen)

            out_repeat_interval.append(repeat_interval)
            out_sparseness.append(sparseness)
            out_opt.append(opt)

            out_trace_memory.append(model_params["tau_stdp"])
            out_trace_increase.append(model_params["trace_scale"])
            out_stdp_update.append(model_params["plasticity_scale"])
            out_total_weight.append(model_params["total_synaptic_weight"])
            out_norm_interval.append(model_params["weight_norm_freq"])

            print(process_name, i, f"{repeat_interval}_{sparseness}_{opt}", ":",
                  rsync_gen, sc_gen, "::",
                  gini, modul, trans, part, centr_btw)
            i += 1

    plot_matrix = model.connectivity_matrix / model_params["total_synaptic_weight"]
    hex_codes_matrix = ['#000000', '#6ccdc0', '#eb7341', '#ffffff']
    cmap_matrix = get_continuous_cmap(hex_codes_matrix)
    fig_con, ax = plt.subplots(figsize=(20, 20))
    img_con = ax.imshow(plot_matrix, cmap=cmap_matrix)

    #if repeat_interval == 3 and sparseness == 5:
    #    cbar = ax.figure.colorbar(img_con, ax=ax)
    #    cbar.ax.tick_params(labelsize=70)

    img_con.set_clim(0, 0.18)
    ax.invert_yaxis()
    ax.xaxis.set_tick_params(labelsize=75)
    ax.yaxis.set_tick_params(labelsize=75)

    text_gini = f"G {round(np.mean(out_gini),2)}±{round(np.std(out_gini),2)}"
    text_modul = f"M {round(np.mean(out_modul),2)}±{round(np.std(out_modul),2)}"
    text_trans = f"T {round(np.mean(out_trans),2)}±{round(np.std(out_trans),2)}"
    text_part = f"P {round(np.mean(out_part),2)}±{round(np.std(out_part),2)}"
    text_centr = f"C {int(np.mean(out_centr_btw))}±{int(np.std(out_centr_btw))}"

    patch = patches.Rectangle((44, 46), 57, 61, facecolor="white", edgecolor="white")
    ax.add_patch(patch)

    y_text = 0.93
    for text in (text_gini, text_modul, text_trans, text_part, text_centr):
        ax.text(0.48, y_text, text,
            horizontalalignment='left',
            verticalalignment='center', transform=ax.transAxes,
            fontsize=84, weight="regular")
        y_text -= 0.1

    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)

    for ext in ("png", "svg"):
        fig_con.savefig(os.path.join(plot_dir, f"{repeat_interval}_{sparseness}.{ext}"),
                        bbox_inches="tight")
    return out_sparseness, out_repeat_interval, out_opt, \
            out_gini, out_modul, out_trans, out_part, out_centr_btw, \
            out_acc_rsync, out_acc_sc, \
            out_trace_memory, out_trace_increase, out_stdp_update, out_total_weight, out_norm_interval

if __name__ == "__main__":
    sparseness_list = []
    repeat_interval_list = []
    opt_list = []

    gini_list = []
    modul_list = []
    trans_list = []
    part_list = []
    centr_btw_list = []

    rsync_acc_list = []
    sc_acc_list = []

    trace_memory_list = []
    trace_increase_list = []
    stdp_update_list = []
    total_weight_list = []
    norm_interval_list = []

    tasks = [(sparseness, repeat_interval, opt)
             for repeat_interval in (3, 6, 10)
             for sparseness in (5, 10, 20)
             for opt in ("rsync", "sc")]

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(calculate_network_stats, tasks)

    for sparseness, repeat_interval, opt, \
        gini, modul, trans, part, centr_btw, \
        rsync_acc, sc_acc, \
        trace_memory, trace_increase, stdp_update, total_weight, norm_interval \
            in results:
        # Store the results in the appropriate dictionary
        sparseness_list.extend(sparseness)
        repeat_interval_list.extend(repeat_interval)
        opt_list.extend(opt)
        gini_list.extend(gini)
        modul_list.extend(modul)
        trans_list.extend(trans)
        part_list.extend(part)
        centr_btw_list.extend(centr_btw)
        rsync_acc_list.extend(rsync_acc)
        sc_acc_list.extend(sc_acc)
        trace_memory_list.extend(trace_memory)
        trace_increase_list.extend(trace_increase)
        stdp_update_list.extend(stdp_update)
        total_weight_list.extend(total_weight)
        norm_interval_list.extend(norm_interval)

    network_stats = pd.DataFrame()
    network_stats["Sparseness"] = sparseness_list
    network_stats["Repeat interval"] = repeat_interval_list
    network_stats["Opt"] = opt_list

    network_stats["Gini index"] = gini_list
    network_stats["Modularity"] = modul_list
    network_stats["Transitivity"] = trans_list
    network_stats["Participation coefficient"] = part_list
    network_stats["Betwenness centrality"] = centr_btw_list

    network_stats["Rsync gen."] = rsync_acc_list
    network_stats["Spike count gen."] = sc_acc_list

    network_stats["Trace memory"] = trace_memory_list
    network_stats["Trace increase"] = trace_increase_list
    network_stats["STDP update scaling"] = stdp_update_list
    network_stats["Total incoming weight"] = total_weight_list
    network_stats["Normalization interval"] = norm_interval_list

    network_stats.to_excel(f"{save_dir}/network_stats.xlsx", index=False)
    
    # Plot connectivity statistics and their correlations
    
    col_params = ["Trace memory", "Trace increase", "STDP update scaling", "Total incoming weight", "Normalization interval", "Rsync gen.", "Spike count gen."]
    col_network = ["Gini index", "Modularity", "Transitivity", "Participation coefficient", "Betwenness centrality"]

    corr = pd.DataFrame()
    for a in col_params:
        for b in col_network:
            corr.loc[a, b] = network_stats.corr(numeric_only=True).loc[a, b]
            
    fig, ax = plt.subplots()
    ax = sns.heatmap(corr.T, cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-.87, vmax=.95, annot=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{save_dir}/param_conn_corr.{ext}")  
    fig.close()

    color_dict = {3: "#e79573", 6: "#91e3d8", 10: "#159fa1"}

    for stat in ("Gini index", "Modularity", "Transitivity", "Participation coefficient", "Betwenness centrality"):
        fig, ax = plt.subplots()
        ax = sns.boxplot(data=network_stats, x="Sparseness", y=stat, hue="Repeat interval",
                    flierprops={"marker": "o", "markerfacecolor": "None"}, 
                    palette=color_dict)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel(stat, fontsize=23, labelpad=15)
        ax.set_xlabel('Sparseness', fontsize=23, labelpad=15)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_title(stat, fontsize=23)

        leg = ax.legend(title="Repeat interval R", fontsize=20, title_fontsize=22)

        fig.tight_layout()
        plt.show()
        
        plot_dir = f"{save_dir}/sc/boxplots"
        os.makedirs(plot_dir, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(f"{plot_dir}/{stat}.{ext}") 
            
    # Pedict generalizability from connectivity
    
    # predict Rsync using Linear regression
    X = network_stats[network_stats['Readout method'] == 'rsync']
    y = X["Rsync gen."]
    X = X[["Gini index", "Betwenness centrality", "Transitivity"]]

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
        
    # predict Rsync using Decision tree
    X = network_stats[network_stats['Readout method'] == 'rsync']
    y = X["Rsync gen."]
    X = X[["Gini index", "Betwenness centrality", "Transitivity"]]

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
    X = network_stats[network_stats['Readout method'] == 'sc']
    y = X["Spike count gen."]
    X = X[["Gini index", "Betwenness centrality", "Transitivity"]]

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
    X = network_stats[network_stats['Readout method'] == 'sc']
    y = X["Spike count gen."]
    X = X[["Gini index", "Betwenness centrality", "Transitivity"]]

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
            
    