import matplotlib.pyplot as plt
import numpy as np
import os
import re
import sys

paths = [os.path.dirname(os.path.abspath(__file__)),
         os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]
sys.path.extend(paths)

from src.config import combinations

if __name__ == "__main__":
    input_rate = 100
    simulation_length = 1000
    simulation_length_test = 1000
    n_samples = 250

    n_rep = 30
    data_dir = "data"
    for repeat_interval, pattern_size, p_match, plasticity_type in combinations:
        exp_data_dir = f"{data_dir}/{input_rate}_{simulation_length}_{simulation_length_test}_{plasticity_type}/{repeat_interval}_{pattern_size}"
        log_dir = os.path.join(exp_data_dir, "logs")
        plot_dir = os.path.join(exp_data_dir, "plots")

        test_plot_dir = os.path.join(plot_dir, 'test')
        os.makedirs(test_plot_dir, exist_ok=True)

        test_log_dir_0 = os.path.join(log_dir, "sc", 'test')
        test_log_dir_1 = os.path.join(log_dir, "rsync", 'test')

        line_metrics = ["mean", "median", "std", "25", "75"]
        rsync_dict_1 = {line_metric: [[] for i in range(n_rep)] for line_metric in line_metrics}
        sc_dict_0    = {line_metric: [[] for i in range(n_rep)] for line_metric in line_metrics}

        for log_file in os.listdir(test_log_dir_0):

            with open(os.path.join(test_log_dir_0, log_file), 'r') as f:
                lines_0 = f.readlines()

            with open(os.path.join(test_log_dir_1, log_file), 'r') as f:
                lines_1 = f.readlines()

            rep_list = [int(re.findall("REP=(\d+)", line)[0]) for line in lines_0]

            for line_metric in line_metrics:
                rsync_pattern = f"{line_metric.upper()}_rsync=([\d\.]+)"
                sc_pattern = f"{line_metric.upper()}_sc=([\d\.]+)"

                for i, line in enumerate(lines_1):
                    rsync_dict_1[line_metric][i].append(float(re.findall(rsync_pattern, line)[0]))

                for i, line in enumerate(lines_0):
                    sc_dict_0[line_metric][i].append(float(re.findall(sc_pattern, line)[0]))

        sc_dict, rsync_dict = sc_dict_0, rsync_dict_1

        for metric in sc_dict:
            print("SC", metric, sc_dict[metric][10][:3], np.array(sc_dict[metric][10]).mean())
            sc_dict[metric] = [np.array(el).mean() for el in sc_dict[metric]]
        for metric in rsync_dict:
            print("RSYNC", metric, rsync_dict[metric][10][:3], np.array(rsync_dict[metric][10]).mean())
            rsync_dict[metric] = [np.array(el).mean() for el in rsync_dict[metric]]

        fig, ax = plt.subplots()
        ax.set_ylim(0.65, 1.0)
        ax.plot(rep_list, sc_dict["median"], linewidth=2.0, label='spike count', color='#eb7341')
        ax.fill_between(rep_list, sc_dict["25"], sc_dict["75"], alpha=0.2, color='#eb7341', lw=0)

        ax.plot(rep_list, rsync_dict["median"], linewidth=2.0, label='Rsync', color='#6ccdc0')
        ax.fill_between(rep_list, rsync_dict["25"], rsync_dict["75"], alpha=0.2, color='#6ccdc0', lw=0)

        ax.axhline(y=0.68, linestyle='dotted', color='grey', label='baseline')
        ax.axvline(x=int(repeat_interval), linestyle='dashed', color='black', label="optimization R")

        leg = ax.legend(title="Prediction method", fontsize=22, title_fontsize=24)
        leg._legend_box.align = "left"

        ax.set_xticks(np.arange(min(rep_list), max(rep_list)+1, 1))
        xlabels = ax.get_xticklabels()
        ylabels = ax.get_yticklabels()

        for i, l in enumerate(xlabels):
            val = int(l.get_text())
            if val % 5 != 0:
                xlabels[i] = ''
            plt.gca().set_xticklabels(xlabels, fontsize=24)

        for i, l in enumerate(ylabels):
            val = int(float(l.get_text()) * 100)
            if val % 10 != 0:
                ylabels[i] = ''
            else:
                ylabels[i] = str(round(float(l.get_text()), 1))
            plt.gca().set_yticklabels(ylabels, fontsize=24)

        # ax.set_ylabel('Accuracy', fontsize=23, labelpad=15)
        # ax.set_xlabel('Repeat interval R', fontsize=23, labelpad=15)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        text_rsync = round(np.array(rsync_dict["median"]).mean(), 3)
        text_sc = round(np.array(sc_dict["median"]).mean(), 3)

        text_rsync_weight, text_sc_weight = "regular", "regular"
        if text_rsync > text_sc:
            text_rsync_weight = "demibold"
        elif text_rsync < text_sc:
            text_sc_weight = "demibold"
        elif text_rsync == text_sc:
            text_rsync_weight = text_sc_weight = "demibold"

        ax.text(1.0, 1.0, text_sc,
                    horizontalalignment='right',
                    verticalalignment='top', transform=ax.transAxes,
                    fontsize=26, weight=text_sc_weight,
                    color='#eb7341')

        ax.text(1.0, 0.89, text_rsync,
                    horizontalalignment='right',
                    verticalalignment='top', transform=ax.transAxes,
                    fontsize=26, weight=text_rsync_weight,
                    color='#6ccdc0')

        fig_path = os.path.join(test_plot_dir, "f1")
        fig.savefig(fig_path + '.png', bbox_inches="tight")
        fig.savefig(fig_path + '.svg', bbox_inches="tight")
        plt.close(fig)
        print('file SAVED', fig_path)