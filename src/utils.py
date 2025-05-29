import logging
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import seaborn as sns

import neo
import quantities as pq

from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV

def create_logger(logger_name):
    """
    Gets or creates a logger
    """
    logger = logging.getLogger(__name__)
    # set log level
    logger.setLevel(logging.INFO)

    # define file handler and set formatter
    file_handler = logging.FileHandler(logger_name)
    formatter    = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # add file handler to logger
    logger.addHandler(file_handler)
    return logger

def generate_poisson_input(stimulus, input_rate, delta_t, length):
    time_steps = len(np.arange(0, length, delta_t))
    poisson_input = np.zeros((stimulus.size, time_steps), dtype=np.float32)
    for t in range(1, time_steps):
        random_numbers = np.random.random(stimulus.size)
        thalamic_input_thresholds = stimulus * (input_rate / 1000) * delta_t
        poisson_input[random_numbers < thalamic_input_thresholds, t] = 1
    return poisson_input

def exp_convolve(spike_train):
    tau = 3.0  # ms
    exp_kernel_time_steps = np.arange(0, tau * 10, 1)
    decay = np.exp(-exp_kernel_time_steps / tau)
    exp_kernel = decay
    return np.convolve(spike_train, exp_kernel, 'same')  # 'valid'

def neo_spike_transform(spike_train):
    return neo.SpikeTrain(spike_train.nonzero()[0], units=pq.ms, t_start=1, t_stop=len(spike_train))

def downsample_spikes(data, delta_t):
    """
    Downsamples output of spiking model to have result in ms

    Args:
        data (numpy array): spiking or voltahe data to downsample
        delta_t (float): delta_t parameter of the spiking model, defines time step
    """
    length = data.shape[1]  # + transient_steps
    downsample = int(1 / delta_t)
    length_downsampled = int(length / downsample)

    # downsample spike trains
    spike_indices, spike_times = np.nonzero(data)
    spikes_downsampled = np.zeros((data.shape[0], length_downsampled), dtype='bool')
    spikes_downsampled[spike_indices, (spike_times / downsample).astype(int)] = True
    return spikes_downsampled
    
def find_transient(spikes):
    """
    Finds a time step from which it makes sense to interpret spiking results.
    Before this time step there is a transient period:
        1) neurons don't spike at all, then all spike at the same time,
        2) then spiking becomes meaningful, activity stabilizes. The end of transient period

    Args:
        spikes (numpy array): spike trains
    """
    zero_after_transient = 0
    transient_start = -1
    transient_end = 0

    for i, s in enumerate(spikes.T):
        if s.sum() > len(s) / 5 and transient_start == -1:
            transient_start = i

        if transient_start > -1 and s.sum() == 0:
            zero_after_transient += 1

        elif transient_start > -1 and s.sum() > 0:
            zero_after_transient = 0

        if zero_after_transient > 5:
            transient_end = i
            break
    return transient_start, transient_end

def generate_recog_data(stimulus_size=100, repeat_interval=3, p_repeat=0.5, pattern_size=20, n_samples=20, **kwargs):
    """
    Generates data for familiarity task: stimuli + labels
    Args:
        stimulus_size (int): length of a binary stimulus vector
        repeat_interval (int): interval at which a stimulus can be repeated
        p_repeat (float): probability of a stimulus repetition
        pattern_size (int): how many ones will be in a binary stimulus vector
        n_samples: the amount of samples to generate
    """
    data_x = []
    data_y = []
    for n in range(n_samples):
        to_repeat = np.random.random(1)[0] < p_repeat
        
        if to_repeat and n >= repeat_interval and data_y[n-repeat_interval] == 0:
            x = data_x[n-repeat_interval]
            y = 1
        else:
            x = np.zeros(int(stimulus_size), dtype=int)
            idx = np.random.choice(range(len(x)), pattern_size, replace=False)
            x[idx] = 1
            y = 0
        data_x.append(x)
        data_y.append(y)
    return zip(data_x, data_y)

def generate_corr_data(stimulus_size=100, repeat_interval=3, p_repeat=0.5, pattern_size=20, n_samples=20, p_match=0.0, **kwargs):
    """
    Generates data for familiarity task: stimuli + labels
    Args:
        stimulus_size (int): length of a binary stimulus vector
        repeat_interval (int): interval at which a stimulus can be repeated
        p_repeat (float): probability of a stimulus repetition
        pattern_size (int): how many ones will be in a binary stimulus vector
        n_samples (int): the amount of samples to generate
        p_match (float): the input correlation 
    """
    data_x = []
    data_y = []

    template = np.zeros(stimulus_size, dtype=int)
    template_active = np.random.choice(stimulus_size, pattern_size, replace=False)
    template[template_active] = 1
    #print("TEMPLATE", len(template_active))
    
    for n in range(n_samples):
        to_repeat = np.random.random(1)[0] < p_repeat
        
        if to_repeat and n >= repeat_interval and data_y[n-repeat_interval] == 0:
            x = data_x[n-repeat_interval]
            y = 1
        else:
            x = np.zeros(int(stimulus_size), dtype=int)

            if p_match == 0:
                idx = np.random.choice(stimulus_size, pattern_size, replace=False)
            else:
                n_match = int(p_match * pattern_size)
                n_nonmatch = pattern_size - n_match
                match_indices = np.random.choice(template_active, n_match, replace=False)
                available = np.setdiff1d(np.arange(stimulus_size), template_active)
                nonmatch_indices = np.random.choice(available, n_nonmatch, replace=False)
                idx = np.concatenate([match_indices, nonmatch_indices])
            
            x[idx] = 1
            y = 0
        data_x.append(x)
        data_y.append(y)
    return zip(data_x, data_y)

def plot_classes(y_true, scores, threshold, scores_label, plot_dir, color_new, color_fam):
    col = np.where(np.array(y_true) < 1, color_new, color_fam)
    fig, ax = plt.subplots()
    ax.scatter(range(len(scores)), scores, c=col)
    ax.set_xlabel('Data samples', fontsize=24, labelpad=15)
    ax.set_ylabel(scores_label, fontsize=24, labelpad=15)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color_new, markersize=8),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=color_fam, markersize=8),
                    ax.axhline(y=threshold, linestyle='dotted', color='black', label="class threshold")
                    ]
    ax.legend(custom_lines, ['novel', 'familiar', 'class threshold'],
              title="Stimulus", fontsize=20, title_fontsize=22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.savefig(f"{plot_dir}/{scores_label}.svg", bbox_inches="tight")
    fig.savefig(f"{plot_dir}/{scores_label}.png", bbox_inches="tight")

def plot_class_distributions(arrays, threshold, limits, labels, colors, endings, scores_label, plot_dir):
    fig, ax = plt.subplots()
    for i in range(len(arrays)):
        sns.kdeplot(arrays[i], color=colors[i], label=labels[i], fill=True)
    ax.axvline(x=threshold, linestyle='dotted', color='black', label="class threshold")
    #ax.legend(title="Stimulus", fontsize=20, title_fontsize=22)
    ax.xaxis.set_tick_params(labelsize=20)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_xlim(limits[0], limits[1])
    ax.set_xlabel(scores_label.split("_")[0], fontsize=24, labelpad=15)
    ax.set_ylabel("Density", fontsize=24, labelpad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for ending in endings:
        fig.savefig(f"{plot_dir}/{scores_label}.{ending}", bbox_inches="tight")

def hex_to_rgb(value):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

def get_kde(values, x, bandwidth = 0.2, kernel = 'gaussian'):
    model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
    model.fit(values[:, np.newaxis])
    log_density = model.score_samples(x[:, np.newaxis])
    return np.exp(log_density)

def get_extreme_points(data, typeOfInflexion=None, maxPoints=None):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    typeOfInflexion = None returns all inflexion points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange == 1)[0]

    if typeOfInflexion == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]

    elif typeOfInflexion == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif typeOfInflexion is not None:
        idx = idx[::2]

    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data) - 1) in idx:
        idx = np.delete(idx, len(data) - 1)
    idx = idx[np.argsort(data[idx])]
    # If we have maxpoints we want to make sure the timeseries has a cutpoint
    # in each segment, not all on a small interval
    if maxPoints is not None:
        idx = idx[:maxPoints]
        if len(idx) < maxPoints:
            return (np.arange(maxPoints) + 1) * (len(data) // (maxPoints + 1))

    return idx
