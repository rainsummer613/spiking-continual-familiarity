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

from src.config import params_change
    
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


def analyze_all_parameters(data, parameters):
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
    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000

    data_dir = "data"
    for plasticity_type in ("anti_hebb", "hebb"):
        dir_all = f"{data_dir}/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}"
        stats_dir = f"{dir_all}/stats"
        os.makedirs(stats_dir, exist_ok=True)

        # PARSE PARAMS
        params_d = {}
        if plasticity_type == "anti_hebb":
            model_params_list = ["trace_memory", "learning_rate", "total_incoming_weight",
                                 "minimal_weight", "normalization_interval"]
        else:
            model_params_list = ["trace_memory", "learning_rate", "total_incoming_weight",
                                 "normalization_interval"]
        model_params_plot = [k.replace("_", " ").capitalize() for k in model_params_list]

        params_all = {p: [] for p in model_params_plot}
        params_repeat = {3: {}, 6: {}, 10: {}}
        params_sparseness = {5: {}, 10: {}, 20: {}}
        params_p_match = {0.0: {}, 0.2: {}, 0.4: {}, 0.6: {}, 0.8: {}}
        params_opt = {"rsync": {}, "sc": {}}

        params_opt_sparseness = {"rsync": {5: {}, 10: {}, 20: {}}, "sc": {5: {}, 10: {}, 20: {}}}
        for p in params_all:
            for opt in params_opt_sparseness:
                for sparseness in params_opt_sparseness[opt]:
                    params_opt_sparseness[opt][sparseness][p] = []

        params_opt_repeat = {"rsync": {3: {}, 6: {}, 10: {}}, "sc": {3: {}, 6: {}, 10: {}}}
        for p in params_all:
            for opt in params_opt_repeat:
                for repeat in params_opt_repeat[opt]:
                    params_opt_repeat[opt][repeat][p] = []

        params_opt_p_match = {"rsync": {0.0: {}, 0.2: {}, 0.4: {}, 0.6: {}, 0.8: {}},
                              "sc": {0.0: {}, 0.2: {}, 0.4: {}, 0.6: {}, 0.8: {}},
                              }
        for p in params_all:
            for opt in params_opt_p_match:
                for p_match in params_opt_p_match[opt]:
                    params_opt_p_match[opt][p_match][p] = []

        list_repeat = []
        list_sparseness = []
        list_opt = []
        list_p_match = []
        vectors = []

        vectors_repeat = {3: [], 6: [], 10: []}
        vectors_sparseness = {5: [], 10: [], 20: []}
        vectors_p_match = {0.0: [], 0.2: [], 0.4: [], 0.6: [], 0.8: []}
        vectors_opt = {"rsync": [], "sc": []}

        for d in os.listdir(dir_all):
            if not d[0].isalpha():
                repeat_interval, sparseness = d.split("_")
                repeat_interval, sparseness = int(repeat_interval), int(sparseness)
                print(d)

                for p_match_str in os.listdir(os.path.join(dir_all, d)):
                    p_match = float(p_match_str)

                    params_one = {p: [] for p in params_all}

                    for opt in ("rsync", "sc"):
                        param_dir = os.path.join(dir_all, d, p_match_str, "params", opt)

                        for f in os.listdir(param_dir):
                            param_file_path = os.path.join(param_dir, f)

                            with open(param_file_path) as param_file:
                                params = json.load(param_file)

                                if plasticity_type == "hebb":
                                    params["model_params"] = {
                                        key_mapping.get(k, k): v for k, v in params["model_params"].items()
                                    }

                                vector = []
                                for p in model_params_list:
                                    p_orig = p.replace("_", " ").capitalize()
                                    params_one[p_orig].append(params["model_params"][p])
                                    vector.append(params["model_params"][p])

                            vectors.append(np.array(vector))
                            vectors_repeat[repeat_interval].append(vector)
                            vectors_sparseness[sparseness].append(vector)
                            vectors_opt[opt].append(vector)
                            vectors_p_match[p_match].append(vector)

                            list_repeat.append(repeat_interval)
                            list_sparseness.append(sparseness)
                            list_opt.append(opt)
                            list_p_match.append(p_match)

                        params_d[d] = params_one.copy()

                        for p in params_one:
                            params_all[p].extend(params_one[p].copy())
                            params_opt_sparseness[opt][sparseness][p].extend(params_one[p].copy())
                            params_opt_repeat[opt][repeat_interval][p].extend(params_one[p].copy())
                            params_opt_p_match[opt][p_match][p].extend(params_one[p].copy())

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

                            if p not in params_p_match[p_match]:
                                params_p_match[p_match][p] = params_one[p].copy()
                            else:
                                params_p_match[p_match][p].extend(params_one[p].copy())

        vectors1 = vectors_opt["rsync"]
        vectors2 = vectors_opt["sc"]

        processes = []
        results = []

        # Analyze all individual parameters in parallel
        results_sparseness = analyze_all_parameters(params_opt_sparseness, model_params_plot)
        results_sparseness_df = pd.DataFrame(results_sparseness)
        results_sparseness_df = results_sparseness_df.round(3)
        results_sparseness_df.to_excel(f'{stats_dir}/opt_diff_sparseness.xlsx', index=False)

        results_repeat = analyze_all_parameters(params_opt_repeat, model_params_plot)
        results_repeat_df = pd.DataFrame(results_repeat)
        results_repeat_df = results_repeat_df.round(3)
        results_repeat_df.to_excel(f'{stats_dir}/opt_diff_repeat.xlsx', index=False)

        results_p_match = analyze_all_parameters(params_opt_p_match,model_params_plot)
        results_p_match_df = pd.DataFrame(results_p_match)
        results_p_match_df = results_p_match_df.round(3)
        results_p_match_df.to_excel(f'{stats_dir}/opt_diff_p_match.xlsx', index=False)

        # Analyze differences in parameter correlations
        group_rsync = pd.DataFrame(params_opt["rsync"]).to_numpy()
        group_sc = pd.DataFrame(params_opt["sc"]).to_numpy()
        observed_diffs, p_values = permutation_test_corr_diff(group_rsync, group_sc)

        observed_diffs_df = pd.DataFrame(observed_diffs, index=model_params_plot, columns=model_params_plot)
        p_values_df = pd.DataFrame(p_values, index=model_params_plot, columns=model_params_plot)

        observed_diffs_df = observed_diffs_df.replace([np.inf, -np.inf], 0).fillna(0)
        observed_diffs_df = observed_diffs_df.round(3)
        observed_diffs_df.to_excel(f"{stats_dir}/corr_diff.xlsx")
        p_values_df = p_values_df.round(3)
        p_values_df.to_excel(f"{stats_dir}/corr_diff_p.xlsx")

        fig, ax = plt.subplots()
        ax_img = sns.heatmap(observed_diffs_df, cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-0.32, vmax=0.32,
                             annot=True)
        ax.set_title("Correlation differences", fontsize=16)
        fig.tight_layout()
        for ext in ("png", "svg"):
            fig.savefig(f"{stats_dir}/param_corr_diff.{ext}")

        # Correlations for Rsync and SC separately
        for opt in ("rsync", "sc"):
            params_opt_df = pd.DataFrame(params_opt[opt])

            fig, ax = plt.subplots()
            ax_img = sns.heatmap(params_opt_df.corr(), cmap=sns.diverging_palette(190, 22, as_cmap=True), vmin=-1.0, annot=True)
            fig.tight_layout()

            plot_dir = f"{stats_dir}/{opt}"
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

        df_param = pd.read_excel(f"{stats_dir}/opt_diff_repeat.xlsx")
        for p in params_all:
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

            ax.set_ylabel("", fontsize=22, labelpad=15)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if p == "Normalization interval":
                ax.legend(fontsize=20, title_fontsize=22)
            #ax.set_title(p, fontsize=24)

            plot_dir = f"{stats_dir}/compare/repeat_interval"
            os.makedirs(plot_dir, exist_ok=True)
            fig.tight_layout()
            for ext in ("png", "svg"):
                fig.savefig(f"{plot_dir}/{p}.{ext}")

        # Plot individual parameter distributions for Sparseness
        df_param = pd.read_excel(f"{stats_dir}/opt_diff_sparseness.xlsx")
        sparseness_labels = {5: 0.9, 10: 0.8, 20: 0.6}
        color_dict = {5: "#eb7341",
                      10: "#6ccdc0",
                      20: "#3d85c6"}

        for p in params_all:
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

            ax.set_ylabel("", fontsize=22, labelpad=15)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if p == "Normalization interval":
                ax.legend(fontsize=20, title_fontsize=22)

            #ax.set_title(p, fontsize=24)

            plot_dir = f"{stats_dir}/compare/sparseness"
            os.makedirs(plot_dir, exist_ok=True)

            fig.tight_layout()
            for ext in ("png", "svg"):
                fig.savefig(f"{plot_dir}/{p}.{ext}")

        # Plot individual parameter distributions for p_match
        df_param = pd.read_excel(f"{stats_dir}/opt_diff_p_match.xlsx")
        color_dict = {0.0: "#eb7341",
                      0.2: "#6ccdc0",
                      0.4: "#3d85c6",
                      0.6: "firebrick",
                      0.8: "rebeccapurple"}

        for p in params_all:
            fig, ax = plt.subplots()
            y_text = 0.9

            for p_match in params_p_match:
                ax = sns.kdeplot(params_opt_p_match["sc"][p_match][p], label=f"Spike count {p_match}",
                                     color=color_dict[p_match])
                ax_img = sns.kdeplot(params_opt_p_match["rsync"][p_match][p], label=f"Rsync {p_match}",
                                         linestyle="--", color=color_dict[p_match])
                text = df_param.loc[df_param['parameter'] == p, p_match].values[0]
                sign = df_param.loc[df_param['parameter'] == p, f"{str(p_match)}_sig"].values[0]

                text_weight = "regular"
                if sign == 1:
                    text_weight = "demibold"

                ax.text(0.8, y_text, text,
                            horizontalalignment='left',
                            verticalalignment='center', transform=ax.transAxes,
                            fontsize=21, weight=text_weight,
                            color=color_dict[p_match])
                y_text -= 0.1

            ax.set_ylabel("", fontsize=22, labelpad=15)
            ax.xaxis.set_tick_params(labelsize=20)
            ax.yaxis.set_tick_params(labelsize=20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            if p == "Normalization interval":
                ax.legend(fontsize=20, title_fontsize=22)
            #ax.set_title(p, fontsize=24)

            plot_dir = f"{stats_dir}/compare/p_match"
            os.makedirs(plot_dir, exist_ok=True)
            fig.tight_layout()
            for ext in ("png", "svg"):
                fig.savefig(f"{plot_dir}/{p}.{ext}")