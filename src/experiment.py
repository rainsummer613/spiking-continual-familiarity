import math
import numpy as np

from measure import accuracy_abbott_score, measure_mean_sc, measure_max_sc
from utils import find_transient, downsample_spikes, generate_recog_data, generate_poisson_input

class Experiment:
    def __init__(self, metrics, data, data_params, simulation_length=None):
        """
        Base class for running a simulation experiment
        Args:
            metrics (dict):  keys are metric names, values are metric functions
            simulation_length (int): length of simulation in ms
            data (array of tuples OR None): if array of tuples, then stimuli + labels
            data_params (dict or None): if dict, then parameters for continuous data generation with values
        """
        self.metrics = metrics
        self.simulation_length = simulation_length
        self.data = data
        self.data_params = data_params
        
    def run(self, model_class, model_params):
        """
        A model simulates spike trains with a given set of the parameters.
        Returns statistics calculated on simulated spike trains

        Args:
            model_class (class): a spiking model class
            model_params (dict): parameters for model initialization
        """
        if self.data is not None:
            model = model_class(**model_params)
            voltage, firings, _ = model.simulate(self.data, self.simulation_length)
            return {metric: self.metrics[metric]['func'](firings, **self.metrics[metric]['kwargs'])
                                               for metric in self.metrics}

    @staticmethod
    def evaluate_mult_metric(res_dict, threshold_dict, y_true):
        y_pred = {}
        for metric in res_dict:
            y_pred[metric] = [1 if r > threshold_dict[metric] else 0 for r in res_dict[metric]]

        best_score = {}
        for metric in y_pred:
            best_score[metric] = accuracy_abbott_score(np.array(y_true), np.array(y_pred[metric]))
        return best_score

    @staticmethod
    def evaluate(res, y_true, threshold=None, hebbian=True):
        """
        Returns accuracy of a binary classification task.
        Classification procedure:
            1) Find an optimal threshold metric (e.g. rate or synchrony) value
            2) Values over the threshold would correspond to one class, values below - to another one

        Args:
            res (numpy array): array of metric values (e.g. rate or synchrony) for every data sample
            y_true (list or numpy array): ground truth metrics for each data sample (1 if familiar, 0 otherwise)
        """
        if threshold is not None:
            best_threshold = threshold
            if hebbian:
                y_pred = [1 if r > threshold else 0 for r in res]
            else:
                y_pred = [1 if r < threshold else 0 for r in res]
            best_score = accuracy_abbott_score(y_true, y_pred)
        else:
            res_mean, res_std = np.mean(res), np.std(res)
            best_threshold, best_score = 0, 0

            if 2 > 1:
                threshold_min, threshold_max = max(res_mean - res_std*4, np.min(res)), min(res_mean + res_std*4, np.max(res))
                if 0 < res_mean < 1:
                    threshold_step = (threshold_max - threshold_min) / 40
                else:
                    threshold_step = 1.0

                for threshold in np.arange(threshold_min, threshold_max, threshold_step):
                    threshold = round(threshold, 4)
                    if hebbian:
                        y_pred = [1 if r > threshold else 0 for r in res]
                    else:
                        y_pred = [1 if r < threshold else 0 for r in res]

                    acc = accuracy_abbott_score(y_true, y_pred)
                    if acc > best_score:
                        best_score, best_threshold = acc, threshold

        return round(best_score, 4), best_threshold
        
class ContinualFamiliarityPlastic(Experiment):
    
    def __init__(self, metrics, data, data_params, input_rate,
                 simulation_length, simulation_length_test,
                 ):
        """
        Class for continual familiarity experiments with plasticity

        Args:
            metrics (dict):  keys are metric names, values are metric functions
            simulation length_test (int): length of test simulation part in ms
            simulation length (int): total length of the simulation in ms
            data (array of tuples OR None): if array of tuples, then stimuli + labels
            data_params (dict or None): if dict, then parameters for continuous data generation with values
        """
        super().__init__(metrics, data=data, data_params=data_params)
        self.length          = simulation_length
        self.length_test     = simulation_length_test
        self.input_rate      = input_rate
        self.repeat_interval = self.data_params.get('repeat_interval', 3)

        if self.data is None:
            self.data = list(generate_recog_data(repeat_interval=self.repeat_interval,
                                                 stimulus_size=self.data_params.get("exc_neurons", 100),
                                                 pattern_size=self.data_params.get('pattern_size', 30),
                                                 n_samples=self.data_params.get('n_samples', 150),
                                                 p_repeat=0.5)
                             )

    def run(self, model_class, model_params, thresholds, optimize=True, **kwargs):
        """
        An experiment pipeline includes three stages:
            1) Model produces spiking output in response to input stimuli, no plasticity.
                Stimulus familiarity is decoded from spike trains
            2) Model remembers stimuli: it produces spiking output with plasticity on
            3) Spike trains from step 1 are used to classify each stimulus' familiarity

        Returns accuracy of familiarity classification for a given model on given input data

        Args:
            model_class (class): a spiking model class
            model_params (dict): parameters for model initialization
        """
        model = model_class(**model_params)
        res = {metric: [] for metric in self.metrics}
        if "rsync" in res:
            res = dict({"rsync_num": [], "rsync_den": []}, **res)
        y_true = []
        rate_high = []
        rate_low = []

        all_input = []
        for i, d in enumerate(self.data):
            verbose = False
            if i % 10 == 0:
                verbose = True

            x, y = d
            poisson_input = generate_poisson_input(x, self.input_rate, model.delta_t, self.length)
            all_input.append(poisson_input)

            model_kwargs = {'sample': str(i), 'proc_name': ''}
            if 'proc_name' in kwargs:
                model_kwargs['proc_name'] = kwargs['proc_name']

            # decide stimulus familiarity + memorize the stimulus
            if verbose:
                print(f'{model_kwargs["proc_name"]} {i}/{len(self.data)} PREDICT FAMILIARITY')
            voltage, spikes = model.simulate(length=self.length, external_input=poisson_input, plastic=True,
                                              verbose=verbose, verbose_freq=5000, **model_kwargs)

            if i > 0:
                neurons = np.nonzero(x)[0]
                transient_start, transient_end = find_transient(spikes)
                # transient_end = 0
                # check if the output firing rate is appropriate wrt input firing rate
                n_seconds = spikes.shape[1] / 1000
                max_rate = measure_max_sc(spikes[neurons, transient_end:]) / n_seconds
                mean_rate = measure_mean_sc(spikes[neurons, transient_end:]) / n_seconds

                # firing rate too high (suspect unstable activity) or too low wrt input firing rate
                if max_rate >= self.input_rate * 1.5:
                    rate_high.append(i-1)
                elif mean_rate < self.input_rate * 0.2:
                    rate_low.append(i-1)

                for metric in self.metrics:
                    metric_func = self.metrics[metric]['func']
                    metric_func_kwargs = self.metrics[metric]['kwargs']

                    spikes_measure = spikes[neurons, transient_end:]
                    score = metric_func(spikes_measure, **metric_func_kwargs)
                    res[metric].append(score)
                y_true.append(y)

        out_score = {}

        # check if hebbian or anti-Hebbian plasticity
        if model_params["plasticity_scale"] >= 0:
            hebbian = True
        else:
            hebbian = False

        for metric in res:
            out_score[metric] = {}
            threshold = None
            if thresholds is not None:
                threshold = thresholds.get(metric, None)
            score, res_threshold = Experiment.evaluate(np.array(res[metric]), np.array(y_true),
                                                       threshold, hebbian)
            out_score[metric]['score'] = score
            out_score[metric]['threshold'] = res_threshold

        if optimize is False:
            thresholds = {metric: out_score[metric]['threshold'] for metric in out_score}
            mult_metric_scores = Experiment.evaluate_mult_metric(res, thresholds, np.array(y_true))

            for metric in mult_metric_scores:
                out_score[metric] = {"score": mult_metric_scores[metric]}

        return model, out_score
