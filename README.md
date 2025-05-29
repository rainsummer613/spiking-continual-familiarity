[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14639677.svg)](https://zenodo.org/doi/10.5281/zenodo.14639677)

## Continual familiarity decoding from recurrent connections in spiking networks
This is a repository for the paper **Continual familiarity decoding from recurrent connections in spiking networks**. 
This repository contains code to run experiments and analyze a spiking neural network model that encodes familiarity memory through local spike-timing-dependent plasticity (STDP). 
The model decodes familiarity using spike count and synchrony, outperforming LSTM models in time-invariant familiarity tasks, with superior generalization across temporal scales and enhanced performance under sparse input conditions.

### Our model
- One-layer Izhikevich spiking network with 100 excitatory and 0 inhibitory neurons.
- All-to-all lateral/recurrent connectivity + one-to-one feedforward connectivity to Poisson input neurons.
- No read-out neuron, the classification is performed solely based on firing statistics.

### Classification procedure
1) Model receives new input sample and runs for 1000 ms to encode it in lateral connectivity according to a [symmetric STDP rule](src/model.py#L65). Then firing data from the smulation run is used to predict whether the stimulus was encountered before (binary classification).
3) Familiarity classification performance is evaluated with [Abbott accuracy](src/measure.py#L19).

### Decoding familiarity
1) Familiarity is decoded from spike trains for each stimulus characteristics: either their synchrony (Rsync) or spike count. If the metric value exceeds a certain threshold, the stimulus is classified as familiar.
2) The threshold is calculated the [following way](src/experiment.py#L49): every time different thresholds are tested, and the one which leads to highest classification accurycy is selected. 

### Optimization
Parameters for best familiarity classification accuracy are found via the genetic algorithm. 
The following parameters are optimized:
- Learning rate (_learning_rate_),
- Trace memory (_trace_memory_),
- Total incoming weight (_total_incoming_weight_)
- Weight normalization interval (_normalization_interval_).

Parameter ranges can be found in `src/config.py`

### Installation
Clone the repository and all libraries from `requirements.txt`.

### The code base
Brief description of files. 

The **src** folder contains the source code for the model:
- `model.py` contains class for Izhikevich spiking model.
- `genetic.py` contains optimization genetic algoithm.
- `experiment.py` defines the logic for running a model on specific data within the particular experimental setup for continual familiarity.
- `measure.py` defines metrics for spike trains (currently firing rate and rsync).
- `utils.py` contains additional helper functions for the simulation.
- `config.py` defines some useful simulation parameters.

The **main** folder has scripts for running all simulations and analyses:
- `train.py` runs the entire optimization pipeline: from generating artificial stimuli to finding optimal parameters via genetic algorithm. Data is saved to _data_ folder.
- `test.py` is used to evaluate the model with optimal parameters on other repeat intervals. Data is saved to _data_ folder.
- `plot_performance.py` plots accuracy for each model trained on one repeat interval, tested on all other repeat intervals.
- `get_optimal_params.py` finds optimal parameters in the training log files and saves them in json format.
- `network_analysis.py` computes connectivity statistics across the conditions and saves them as plots and Excel files.
- `param_analysis.py` analyses parameters for synchrony- and rate-optimized models across the conditions and saves them as plots and Excel files.

Files for training and testing with HPC. Each runs a separate optimization process for each repeat interval & sparseness.
- `train.sh` runs optimization with multiprocessing on every core.
- `test.sh` evaluates each model with multiprocessing on every core. 

The **data** folder contains subfolder(s) of the structure: 
- `stats` folder contains the results of connectivity and parameter analyses.
_{input\_firing\_rate}\_{simulation\_length}\_{simulation\_length\_predict}_{plasticity\_type}/{optimization\_repeat\_interval}/{optimization\_sparseness}_. Currently all similation data for a single stimulus is used to predict familiarity. 
Subfolders are organized follows: 
- `logs` directory with log files for train (optimization) and test experiments. Subirectories are named after metrics used for optimization (sc, rsync and rsync_sc - in this case both metrics were used).
- `plots` directory with the same structure as `data/logs`. 
- `params` directory with best parameters found after optimization for model with each combination of repeat interval and pattern size.
- Name of each log and parameter file is formed as follows: _gen\_opt\_{iteration}\_{n\_data\_samples}_.
