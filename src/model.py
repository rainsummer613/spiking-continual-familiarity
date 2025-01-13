import numpy as np
import time

class Izhikevich:

    def __init__(self, exc_neurons=100, inh_neurons=0, delta_t=0.5, voltage_noise=0.6,
                 external_input_scale=1., lateral_input_scale=1.,
                 plasticity_scale=0.001,
                 hebb_window=10, weight_norm_freq=1,
                 threshold=30, total_synaptic_weight=None,
                 t_refr=3, tau_stdp=20, trace_scale=1.0,
                 connectivity_matrix=None,
                 random_seed=None):
        
        np.random.seed() if not random_seed else np.random.seed(random_seed)

        self.n_neurons_exc = int(exc_neurons)
        self.n_neurons_inh = int(inh_neurons)
        self.n_neurons = self.n_neurons_exc + self.n_neurons_inh

        self.total_synaptic_weight = total_synaptic_weight
        self.external_input_scale = external_input_scale
        self.lateral_input_scale = lateral_input_scale
        self.plasticity_scale = plasticity_scale
        self.hebb_window = hebb_window / delta_t
        self.weight_norm_freq = weight_norm_freq
        
        if not isinstance(connectivity_matrix, np.ndarray) or len(connectivity_matrix) != self.n_neurons:
            connectivity_exc = np.concatenate([np.random.rand(self.n_neurons_exc, self.n_neurons_exc),
                                                -np.random.rand(self.n_neurons_exc, self.n_neurons_inh)], 1)

            connectivity_inh = np.concatenate([np.random.rand(self.n_neurons_inh, self.n_neurons_exc),
                                                np.zeros((self.n_neurons_inh, self.n_neurons_inh))], 1)
            self.connectivity_matrix = np.concatenate([connectivity_exc, connectivity_inh], 0)
            self.connectivity_matrix = self.normalize_weights(self.connectivity_matrix)
            
        else:
            self.connectivity_matrix = self.normalize_weights(connectivity_matrix)
        np.fill_diagonal(self.connectivity_matrix, 0)
        self.init_connectivity_matrix = self.connectivity_matrix.copy()

        self.voltage_noise = voltage_noise

        self.voltage = 30.0
        self.recov = 30.0

        # Izhikevich parameters
        self.delta_t = delta_t  # ms
        self.threshold = threshold
        self.izhi_a = np.concatenate([0.02 * np.ones(self.n_neurons_exc), 0.1 * np.ones(self.n_neurons_inh)])
        self.izhi_b = np.concatenate([0.2 * np.ones(self.n_neurons_exc), 0.25 * np.ones(self.n_neurons_inh)])
        self.izhi_voltage_reset = -65.0 
        self.izhi_recov_update = 2.0
        
        self.starting_voltage = self.voltage * (np.random.random(self.n_neurons) - 0.5)  
        self.starting_recov = self.recov * (np.random.random(self.n_neurons) - 0.5)

        self.t_refr = t_refr
        self.tau_stdp = tau_stdp
        self.trace_scale = trace_scale
        
    def normalize_weights(self, connectivity_matrix):
        return connectivity_matrix / connectivity_matrix.sum(axis=1, keepdims=1) * self.total_synaptic_weight

    def stdp_update(self, trace, firing, connectivity_matrix):

        dweights = np.outer(firing, trace)
        #connectivity_matrix += self.plasticity_scale * (dweights + dweights.T)
        connectivity_matrix += self.plasticity_scale * dweights

        np.fill_diagonal(connectivity_matrix, 0)

        return connectivity_matrix
        
    def _time_step(self, voltage, recov, input_lat, input_ext, refr):
        
        substeps = 2  # for numerical stability
        
        voltage += (np.random.random(len(voltage)) - 0.5) * self.voltage_noise * self.delta_t
        
        for i in range(substeps):
            voltage += (1.0 / substeps) * (self.delta_t * (
                        0.04 * (voltage ** 2) + (5 * voltage) + 140 - recov + input_lat + input_ext))
            recov += (1.0 / substeps) * self.delta_t * self.izhi_a * (self.izhi_b * voltage - recov)

        voltage[refr > 0] = self.izhi_voltage_reset
        refr[refr > 0] -= 1
            
        fired = voltage > self.threshold  # array of indices of spikes
        voltage[fired] = self.izhi_voltage_reset  # reset the voltage of every neuron that fired
        recov[fired] += self.izhi_recov_update  # update the recovery variable of every fired neuron
        refr[fired] = self.t_refr / self.delta_t
        
        return voltage, recov, fired, refr
    
    def simulate(self, length, external_input=None, plastic=True,
                 verbose=True, verbose_freq=5000, **kwargs):
        time_steps = len(np.arange(0, length, self.delta_t))
        
        # Initialize voltage variable
        voltage_out = np.zeros((self.n_neurons, time_steps), dtype=np.float32)
        voltage_out[:, 0] = self.starting_voltage
        
        # Initialize recovery variable
        recov_out = np.zeros((self.n_neurons, time_steps), dtype=np.float32)
        recov_out[:, 0] = self.starting_recov
        
        # Initialize firings
        firings_out = np.zeros((self.n_neurons, time_steps), dtype=np.float32)
        
        # Initialize lateral input
        input_lat_out = np.zeros((self.n_neurons, time_steps), dtype=np.float32)

        # Initialize lateral spike arrival times
        spike_arrivals_lat = np.full((self.n_neurons,), -1, dtype=np.float32)

        # Initialize eligibility traces
        trace = np.zeros((self.n_neurons,), dtype=np.float32)

        # Initialize refractory period counters
        refr = np.zeros((self.n_neurons,), dtype=np.float32)

        t0 = time.perf_counter()

        hebb_updates = 0
        for t in range(1, time_steps):
            # recieve lateral input only from fired neurons
            fired = firings_out[:, t-1].nonzero()[0]
            input_lat_out[:, t] = self.connectivity_matrix[:, fired].sum(1) * self.lateral_input_scale
            spike_arrivals_lat[fired] = t

            if plastic:
                if len(fired) > 0:
                    trace[fired] += 1.0 # self.trace_scale * 1.0
                    # Hebbian update
                    post_firing = firings_out[:, t - 1] * trace
                    self.connectivity_matrix = self.stdp_update(connectivity_matrix=self.connectivity_matrix.copy(),
                                                                trace=trace,
                                                                firing=post_firing,
                                                                )
                    hebb_updates += 1
                # update pre_trace
                trace -= (self.delta_t / self.tau_stdp) * trace

                # normalize weights
                if self.weight_norm_freq > 0 and self.weight_norm_freq > 0 and hebb_updates % self.weight_norm_freq == 0:
                    self.connectivity_matrix = self.normalize_weights(self.connectivity_matrix)

            voltage_out[:, t], recov_out[:, t], firings_out[:, t], refr = self._time_step(
                voltage_out[:, t-1].copy(), recov_out[:, t-1].copy(), input_lat_out[:, t],
                external_input[:, t] * self.external_input_scale,
                refr.copy())
            
            self.starting_voltage = voltage_out[:, t]
            self.starting_recov = recov_out[:, t]
            
            row_start = ""
            if verbose and t % verbose_freq == 0:
                row_main = f"Simulated {str(t)} timesteps in {str(round(time.perf_counter() - t0, 4))} s"
                
                if "proc_name" in kwargs:
                    row_start += kwargs["proc_name"] + " "
                    
                if "sample" in kwargs:
                    row_start += kwargs["sample"] + " "
                    
                print(row_start + row_main)
         
        return voltage_out, firings_out
