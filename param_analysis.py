import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd

from scipy.spatial.distance import cosine
from scipy.stats import mannwhitneyu
import multiprocessing as mp

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

param_rename = {
        "plasticity_scale": "STDP update scaling",
        "trace_scale": "Trace increase",
        "tau_stdp": "Trace memory",
        "total_synaptic_weight": "Total incoming weight",
        "weight_norm_freq": "Normalization interval"
    }
param_rename_inv = {v:k for k,v in param_rename.items()}
params_change = list(param_rename.keys())

input_rate = 100
simulation_length = 1000
simulation_length_tes = 1000
    
dir_all = f"data/{input_rate}_{simulation_length}_{simulation_length_test}"
save_dir = f"stats/{input_rate}_{simulation_length}_{simulation_length_test}"
os.makedirs(save_dir, exist_ok=True)
    
rsync_pattern = f"MEAN_rsync=([\d\.]+)"
sc_pattern = f"MEAN_sc=([\d\.]+)"

def mann_whitney_test(group1, group2):
    """Perform the Mann-Whitney U Test."""
    stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    return stat, p_value


def permutation_test(data1, data2, n_permutations=10000):
    """Perform a permutation test to compare the means of two groups."""
    observed_diff = np.mean(data1) - np.mean(data2)
    combined = np.concatenate([data1, data2])
    count = 0

    for _ in range(n_permutations):
        np.random.shuffle(combined)
        new_data1 = combined[:len(data1)]
        new_data2 = combined[len(data1):]
        permuted_diff = np.mean(new_data1) - np.mean(new_data2)
        if permuted_diff >= observed_diff:
            count += 1

    p_value = count / n_permutations
    return observed_diff, p_value


def bonferroni_correction(p_values, alpha=0.05):
    """Apply Bonferroni correction to p-values."""
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    return [p < corrected_alpha for p in p_values], corrected_alpha


def analyze_parameter(param_name, data):
    # Gather all values from both groups across all subgroups
    rsync_values_all = [val for subgroup in data['rsync'].values() for val in subgroup[param_name]]
    sc_values_all = [val for subgroup in data['sc'].values() for val in subgroup[param_name]]

    # Perform permutation test for the overall comparison
    overall_diff, overall_p_value = permutation_test(np.array(rsync_values_all), np.array(sc_values_all))

    # Now analyze each subgroup separately
    subgroup_results = []
    subgroup_diffs = []
    for subgroup_key in data['rsync'].keys():
        rsync_values_subgroup = data['rsync'][subgroup_key][param_name]
        sc_values_subgroup = data['sc'][subgroup_key][param_name]

        if len(rsync_values_subgroup) == 0 or len(sc_values_subgroup) == 0:
            subgroup_results.append((np.nan, 0))  # Handle empty subgroups
            subgroup_diffs.append(np.nan)
            continue

        subgroup_diff, subgroup_p_value = permutation_test(np.array(rsync_values_subgroup),
                                                           np.array(sc_values_subgroup))
        subgroup_results.append(
            (subgroup_p_value, subgroup_p_value < (0.05 / len(data['rsync']))))  # Check significance
        subgroup_diffs.append(subgroup_diff)  # Store the difference

    # Apply Bonferroni correction to the subgroup p-values
    subgroup_p_values = [result[0] for result in subgroup_results]
    significant_flags, _ = bonferroni_correction(subgroup_p_values)

    return {
        'parameter': param_name,
        'overall_diff': overall_diff,
        **{f'{key}_diff': subgroup_diffs[i] for i, key in enumerate(data['rsync'].keys())},
        'overall_p': overall_p_value,
        **{key: subgroup_p_values[i] for i, key in enumerate(data['rsync'].keys())},
        **{f'{key}_sig': int(significant_flags[i]) for i, key in enumerate(data['rsync'].keys())},
    }


def analyze_all_parameters(data):
    parameters = ("STDP update scaling", "Trace increase", "Trace memory", "Total incoming weight",
                   "Normalization interval")  # Assuming all subgroups have the same parameters
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use multiprocessing to analyze parameters in parallel
        results = pool.starmap(analyze_parameter, [(param, data) for param in parameters])

    return results
def permutation_batch(all_vectors, n1, batch_index, batch_size, results):
    """Calculate mean cosine distances for a batch of permutations and print progress."""
    np.random.seed(batch_index)  # Set seed for reproducibility per batch
    batch_distances = []

    for perm_index in range(batch_size):
        # Shuffle and split vectors
        np.random.shuffle(all_vectors)
        perm_group1 = all_vectors[:n1]
        perm_group2 = all_vectors[n1:]

        # Calculate the mean cosine distance for the current permutation
        perm_distances = [cosine(v1, v2) for v1 in perm_group1 for v2 in perm_group2]
        mean_distance = np.mean(perm_distances)
        batch_distances.append(mean_distance)

        # Print progress
        if perm_index % 10 == 0:
            print(f"Batch {batch_index}, Permutation {perm_index}: Mean Distance = {mean_distance}")

    # Store the result in the shared list
    results.extend(batch_distances)

def permute_param_vectors(vectors1, vectors2):
    # Calculate the observed mean cosine distance
    distances = [cosine(v1, v2) for v1 in vectors1 for v2 in vectors2]
    observed_mean_distance = np.mean(distances)

    # Combine vectors from both groups into one array for easy shuffling
    all_vectors = np.vstack([vectors1, vectors2])
    n1 = len(vectors1)

    # Parameters for the permutation test
    n_permutations = 1000  # Total permutations needed
    batch_size = 100  # Number of permutations per batch
    n_batches = n_permutations // batch_size  # Total number of batches

    # Create a list to hold the results of each batch
    manager = mp.Manager()
    results = manager.list()  # Shared list for storing results across processes

    # Create and start processes for each batch
    processes = []
    for i in range(n_batches):
        p = mp.Process(target=permutation_batch, args=(all_vectors, n1, i, batch_size, results))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()

    # Convert results to numpy array and calculate p-value
    permuted_distances = np.array(results)
    p_value = np.mean(permuted_distances >= observed_mean_distance)

    return observed_mean_distance, p_value


def fisher_z(r):
    """Apply Fisher's Z transformation."""
    return 0.5 * np.log((1 + r) / (1 - r))


def permutation_test_corr_diff(group_A, group_B, n_permutations=1000):
    # Calculate initial correlation matrices and apply Fisher Z-transformation
    corr_A = np.corrcoef(group_A, rowvar=False)
    corr_B = np.corrcoef(group_B, rowvar=False)
    Z_A = fisher_z(corr_A)
    Z_B = fisher_z(corr_B)
    observed_diffs = Z_A - Z_B

    # Prepare for permutation testing
    combined_data = np.vstack([group_A, group_B])
    perm_diffs = np.zeros((n_permutations, observed_diffs.shape[0], observed_diffs.shape[1]))

    for i in range(n_permutations):
        # Shuffle and split into two new groups
        np.random.shuffle(combined_data)
        perm_group_A = combined_data[:len(group_A)]
        perm_group_B = combined_data[len(group_A):]

        # Calculate permuted correlation matrices and Z-scores
        perm_corr_A = np.corrcoef(perm_group_A, rowvar=False)
        perm_corr_B = np.corrcoef(perm_group_B, rowvar=False)
        perm_Z_A = fisher_z(perm_corr_A)
        perm_Z_B = fisher_z(perm_corr_B)

        # Store absolute differences
        perm_diffs[i] = perm_Z_A - perm_Z_B

    # Determine significance (two-tailed test)
    p_values = np.mean(np.abs(perm_diffs) >= np.abs(observed_diffs), axis=0)
    return observed_diffs, p_values

if __name__ == "__main__":
    # PARSE PARAMS
    params_d = {}
    params_all = {"STDP update scaling": [], "Trace increase": [], "Trace memory": [], "Total incoming weight": [],
                  "Normalization interval": []}
    params_repeat = {3: {}, 6: {}, 10: {}}
    params_sparseness = {5: {}, 10: {}, 20: {}}
    params_opt = {"rsync": {}, "sc": {}}

    params_opt_sparseness = {"rsync": {5: {}, 10: {}, 20: {}}, "sc": {5: {}, 10: {}, 20: {}}}
    for p in params_all.keys():
        for opt in params_opt_sparseness:
            for sparseness in params_opt_sparseness[opt]:
                params_opt_sparseness[opt][sparseness][p] = []

    params_opt_repeat = {"rsync": {3: {}, 6: {}, 10: {}}, "sc": {3: {}, 6: {}, 10: {}}}
    for p in params_all.keys():
        for opt in params_opt_repeat:
            for repeat in params_opt_repeat[opt]:
                params_opt_repeat[opt][repeat][p] = []

    list_repeat = []
    list_sparseness = []
    list_opt = []
    vectors = []

    vectors_repeat = {3: [], 6: [], 10: []}
    vectors_sparseness = {5: [], 10: [], 20: []}
    vectors_opt = {"rsync": [], "sc": []}

    for d in os.listdir(dir_all):
        repeat_interval, sparseness = d.split("_")
        repeat_interval, sparseness = int(repeat_interval), int(sparseness)
        print(d)

        if sparseness < 30:

            params_one = {"STDP update scaling": [], "Trace increase": [], "Trace memory": [],
                          "Total incoming weight": [],
                          "Normalization interval": []}

            for opt in ("rsync", "sc"):
                param_dir = os.path.join(dir_all, d, "params", opt)

                for f in os.listdir(param_dir):
                    param_file_path = os.path.join(param_dir, f)

                    with open(param_file_path) as param_file:
                        params = json.load(param_file)

                        vector = []
                        for p in params_change:
                            params_one[param_rename[p]].append(params["model_params"][p])
                            vector.append(params["model_params"][p])

                    vectors.append(np.array(vector))
                    vectors_repeat[repeat_interval].append(vector)
                    vectors_sparseness[sparseness].append(vector)
                    vectors_opt[opt].append(vector)

                    list_repeat.append(repeat_interval)
                    list_sparseness.append(sparseness)
                    list_opt.append(opt)

                params_d[d] = params_one.copy()

                for p in params_one:
                    params_all[p].extend(params_one[p].copy())
                    params_opt_sparseness[opt][sparseness][p].extend(params_one[p].copy())
                    params_opt_repeat[opt][repeat_interval][p].extend(params_one[p].copy())

                for p in params_one:
                    if p not in params_repeat[repeat_interval]:
                        params_repeat[repeat_interval][p] = params_one[p].copy()
                    else:
                        params_repeat[repeat_interval][p].extend(params_one[p].copy())

                    if p not in params_sparseness[sparseness]:
                        params_sparseness[sparseness][p] = params_one[p].copy()
                    else:
                        params_sparseness[sparseness][p].extend(params_one[p].copy())

                    if p not in params_opt[opt]:
                        params_opt[opt][p] = params_one[p].copy()
                    else:
                        params_opt[opt][p].extend(params_one[p].copy())

    vectors1 = vectors_opt["rsync"]
    vectors2 = vectors_opt["sc"]

    processes = []
    results = []

    # Analyze all individual parameters in parallel
    results_sparseness = analyze_all_parameters(params_opt_sparseness)
    results_sparseness_df = pd.DataFrame(results_sparseness)
    results_sparseness_df = results_sparseness_df.round(3)
    results_sparseness_df.to_excel(f'{save_dir}/opt_diff_sparseness.xlsx', index=False)

    results_repeat = analyze_all_parameters(params_opt_repeat)
    results_repeat_df = pd.DataFrame(results_repeat)
    results_repeat_df = results_repeat_df.round(3)
    results_repeat_df.to_excel(f'{save_dir}/opt_diff_repeat.xlsx', index=False)

    # Analyze differences in parameter correlations
    group_rsync = pd.DataFrame(params_opt["rsync"]).to_numpy()
    group_sc = pd.DataFrame(params_opt["sc"]).to_numpy()
    observed_diffs, p_values = permutation_test_corr_diff(group_rsync, group_sc)

    parameter_names = ("STDP update scaling", "Trace increase", "Trace memory", "Total incoming weight",
                       "Normalization interval")
    observed_diffs_df = pd.DataFrame(observed_diffs, index=parameter_names, columns=parameter_names)
    p_values_df = pd.DataFrame(p_values, index=parameter_names, columns=parameter_names)

    observed_diffs_df = observed_diffs_df.replace([np.inf, -np.inf], 0).fillna(0)
    observed_diffs_df = observed_diffs_df.round(3)
    observed_diffs_df.to_excel(f"{save_dir}/corr_diff.xlsx")
    p_values_df = p_values_df.round(3)
    p_values_df.to_excel(f"{save_dir}/corr_diff_p.xlsx")

    fig, ax = plt.subplots()
    ax_img = sns.heatmap(observed_diffs_df, cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-0.32, vmax=0.32,
                     annot=True)
    ax.set_title("Correlation differences", fontsize=16)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{save_dir}/param_corr_diff.{ext}")

    # Correlations for Rsync and SC separately
    for opt in ("rsync", "sc"):
        params_opt_df = pd.DataFrame(params_opt[opt])

        fig, ax = plt.subplots()
        ax_img = sns.heatmap(params_opt_df.corr(), cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-1.0, annot=True)
        fig.tight_layout()

        plot_dir = f"{save_dir }/{opt}"
        os.makedirs(plot_dir, exist_ok=True)

        if opt == "rsync":
            ax.set_title("Rsync", fontsize=16)
        else:
            ax.set_title("Spike count", fontsize=16)

        for ext in ("png", "svg"):
            fig.savefig(f"{plot_dir}/param_corr.{ext}")

    # Plot individual parameter distributions for Repeat intervals
    color_dict = {3: "#eb7341",
                  6: "#6ccdc0",
                  10: "#3d85c6"}

    df_param = pd.read_excel(f"{save_dir}/opt_diff_repeat.xlsx")
    for p in parameter_names:
        fig, ax = plt.subplots()
        y_text = 0.9

        for repeat_interval in params_repeat:
            ax = sns.kdeplot(params_opt_repeat["sc"][repeat_interval][p], label=f"Spike count {repeat_interval}",
                             color=color_dict[repeat_interval])
            ax_img = sns.kdeplot(params_opt_repeat["rsync"][repeat_interval][p], label=f"Rsync {repeat_interval}",
                             linestyle="--", color=color_dict[repeat_interval])
            text = df_param.loc[df_param['parameter'] == p, repeat_interval].values[0]
            sign = df_param.loc[df_param['parameter'] == p, f"{str(repeat_interval)}_sig"].values[0]

            text_weight = "regular"
            if sign == 1:
                text_weight = "demibold"

            ax.text(0.8, y_text, text,
                    horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    fontsize=21, weight=text_weight,
                    color=color_dict[repeat_interval])
            y_text -= 0.1

        ax.set_ylabel("Density", fontsize=22, labelpad=15)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if p == "Normalization interval":
            ax.legend(fontsize=20, title_fontsize=22)
        ax.set_title(p, fontsize=24)

        plot_dir = f"{save_dir}/compare/repeat_interval"
        os.makedirs(plot_dir, exist_ok=True)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{plot_dir}/{p}.{ext}")

    # Plot individual parameter distributions for Sparseness
    df_param = pd.read_excel(f"{save_dir}/opt_diff_sparseness.xlsx")
    sparseness_labels = {5: 0.9, 10: 0.8, 20: 0.6}
    color_dict = {5: "#eb7341",
                  10: "#6ccdc0",
                  20: "#3d85c6"}

    for p in params_one:
        fig, ax = plt.subplots()
        y_text = 0.9

        for sparseness in params_sparseness:
            ax = sns.kdeplot(params_opt_sparseness["sc"][sparseness][p],
                             label=f"Spike count {sparseness_labels[sparseness]}",
                             color=color_dict[sparseness])
            ax_img = sns.kdeplot(params_opt_sparseness["rsync"][sparseness][p],
                             label=f"Rsync {sparseness_labels[sparseness]}",
                             linestyle="--", color=color_dict[sparseness])
            text = df_param.loc[df_param['parameter'] == p, sparseness].values[0]
            sign = df_param.loc[df_param['parameter'] == p, f"{str(sparseness)}_sig"].values[0]

            text_weight = "regular"
            if sign == 1:
                text_weight = "demibold"

            ax.text(0.8, y_text, text,
                    horizontalalignment='left',
                    verticalalignment='center', transform=ax.transAxes,
                    fontsize=21, weight=text_weight,
                    color=color_dict[sparseness])
            y_text -= 0.1

        ax.set_ylabel("Density", fontsize=22, labelpad=15)
        ax.xaxis.set_tick_params(labelsize=20)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if p == "Normalization interval":
            ax.legend(fontsize=20, title_fontsize=22)

        ax.set_title(p, fontsize=24)

        plot_dir = f"{save_dir}/compare/sparseness"
        os.makedirs(plot_dir, exist_ok=True)

        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{plot_dir}/{p}.{ext}")

    """
    params_avg = {}
    for p in params_all:
        params_avg[param_rename_inv[p]] = np.mean(params_all[p])

    model_params = {**params_avg, **params_fixed}
    model = Izhikevich(**model_params)
    simulation_length_exp = 1000
    conn = model.connectivity_matrix

    gini_list = []
    sd_list = []
    mean_list = []
    sd_mean_list = []

    for n in np.arange(1, 21):
        model_params["tau_stdp"] = n
        model = Izhikevich(**model_params)

        gini = []
        sd = []
        mean = []
        sd_mean = []

        for i in range(50):

            x = np.zeros(int(100), dtype=int)
            idx = np.random.choice(range(len(x)), 10, replace=False)
            x[idx] = 1
            poisson_input = generate_poisson_input(x, 100, model.delta_t, simulation_length_exp)

            voltage, firings = model.simulate(length=simulation_length_exp,
                                              external_input=poisson_input,
                                              plastic=True, verbose=True, verbose_freq=3000)
            gini.append(compute_gini(model.connectivity_matrix[idx]))
            sd.append(np.std(model.connectivity_matrix[idx]))
            mean.append(model.connectivity_matrix[idx].mean())
            sd_mean.append(np.std(model.connectivity_matrix[idx]) / model.connectivity_matrix[idx].mean())
            conn = model.connectivity_matrix

        gini, sd, mean, sd_mean = np.mean(gini), np.mean(sd), np.mean(mean), np.mean(sd_mean)

        print(n, gini, sd, mean, sd_mean)
        gini_list.append(gini)
        mean_list.append(mean)
        sd_list.append(sd)
        sd_mean_list.append(sd_mean)
    """