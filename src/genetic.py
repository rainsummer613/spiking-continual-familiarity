import multiprocessing as mp
import numpy as np

np.set_printoptions(suppress=True)

class GeneticOptimizer:
    
    def __init__(self, experiment, model_class,
                 metric_weights, generation_size,
                 params_change, params_fixed,
                 plasticity_type, plasticity_sym,
                 logger):
        """
        Class for the genetic optimizer

        Args:
            experiment (class): a class for data simulation
            model_class (class): a class for running a model
            metrics (dict): a dictionary with metric names as keys and metric functions as values
            metric_weights (dict): a dictionary with metric names as keys and metric weights as values
                (how important is every metric)
            generation_size (int): the amount of the descendants in one generation
            n_trials (int): how many times run a simulation with a single set of parameters
            simulation_length_train (int): duration of a single simulation
            params_change (list): parameters to change with minimum and maximum value for each
            params_fixed (list): parameters with fixed values
        """
        self.metric_weights = metric_weights
        self.index_to_metric = dict(enumerate(list(self.metric_weights.keys())))
        self.metric_to_index = {v: k for k, v in self.index_to_metric.items()}
            
        params_intersection = list(set(params_change) & set(params_fixed))
        if len(params_intersection) > 0:
            raise Exception(f'Some parameters are in both lists params_change and params_fixed: {params_intersection}')
        self.params_fixed = params_fixed
        self.params_change = params_change

        if generation_size % 4 != 0:
            raise Exception('Generation size should be divisible by 4')
        self.generation_size = generation_size
        
        all_params = list(params_fixed.keys()) + list(params_change.keys())
        self.n_params = len(all_params)
        self.index_to_param = dict(enumerate(all_params))
        self.param_to_index = {v: k for k, v in self.index_to_param.items()}
        self.param_change_ids = [self.param_to_index[p] for p in params_change]
        self.param_fixed_ids = [self.param_to_index[p] for p in params_fixed]
            
        self.group_desc = generation_size // 4
        self.experiment = experiment
        self.model_class = model_class
        self.plasticity_type = plasticity_type
        self.plasticity_sym = plasticity_sym
        self.logger = logger
        
    def random_gen(self, n_descendants):
        """
        Creates a random generation

        Args:
            n_descendants (int): number of param_sets (descendants) to generate
        """
        generation = np.zeros((n_descendants, self.n_params), dtype=np.float32)
        
        for p in self.param_change_ids:
            low_val = self.params_change[self.index_to_param[p]]['min']
            high_val = self.params_change[self.index_to_param[p]]['max']
            step = self.params_change[self.index_to_param[p]]['step']
            
            if low_val % 1 == 0 and high_val % 1 == 0 and step % 1 == 0:
                param_across_gens = np.random.choice((low_val, high_val), n_descendants).astype(np.float32).round(1)
            else:
                param_across_gens = np.random.uniform(low_val, high_val, n_descendants).round(4)
            generation[:, p] = param_across_gens
        
        generation[:, self.param_fixed_ids] = [self.params_fixed[self.index_to_param[p]]
                                                  for p in self.param_fixed_ids]
        return generation

    def remove_duplicates(self, param_sets, max_iter=10):
        """
        Checks for duplicated parameter sets in one generation, removes them by mutation.

        Args:
            param_sets (numpy array): all descendants of one generation
            max_iter (int): maximum number of mutation iterations to resolve duplicates

        Returns:
            numpy array: deduplicated and possibly mutated parameter sets
        """
        new_param_sets = param_sets.copy()
        iteration = 0

        while iteration < max_iter:
            unq, count = np.unique(new_param_sets, axis=0, return_counts=True)
            repeated_groups = unq[count > 1]

            if len(repeated_groups) == 0:
                break  # No duplicates left

            print(f'Iteration {iteration + 1}: duplicates = {count[count > 1]}')

            for repeated_group in repeated_groups:
                repeated_idx = np.argwhere(np.all(new_param_sets == repeated_group, axis=1)).ravel()[1:]
                new_param_sets[repeated_idx] = np.apply_along_axis(self.mutate, 1, new_param_sets[repeated_idx])

            iteration += 1

        if iteration == max_iter:
            print("Warning: Maximum deduplication iterations reached; some duplicates may remain.")

        return new_param_sets
        
    def first_gen(self, init_sets=None):
        """
        Creates a first generation

        Args:
            init_sets (numpy array): if None, create a random first generation. else
                parameter sets that have to be present in the generation
        """
        random_sets = self.random_gen(self.generation_size)
                                 
        if init_sets is not None:
            for n in range(min(len(init_sets), self.generation_size)):
                init_set = np.zeros(self.n_params)
                init_set[self.param_fixed_ids] = [self.params_fixed[self.index_to_param[p]] 
                                                  for p in self.param_fixed_ids]
                init_set[self.param_change_ids] = [init_sets[n][self.index_to_param[p]] for p in self.param_change_ids]
                random_sets[n] = init_set
        random_sets = self.remove_duplicates(random_sets).round(4)
        return random_sets
    
    def mutate(self, set_params, n_mutated=1):
        """
        Perform a mutation operation: randomly change one or more parameters

        Args:
            set_params (numpy array or list): one set of the parameters
            n_mutated (int): how many parameters are subject to mutation
        """
        params_mutated = np.random.choice(self.param_change_ids, n_mutated, False)
        new_params = set_params.copy()
        
        for p_num in params_mutated:
            p_name = self.index_to_param[p_num]
            p_change = np.random.choice([-self.params_change[p_name]['step'], self.params_change[p_name]['step']])
            new_params[p_num] += p_change
            new_params[p_num] = np.clip(a=new_params[p_num],
                                        a_min=self.params_change[p_name]['min'],
                                        a_max=self.params_change[p_name]['max'])
            new_params[p_num] = new_params[p_num].round(4)
        return new_params
    
    def crossover(self, parent0, parent1, n_crossed=1):
        """
        Perform a crossover operation: replace values of random parameters in set 1 by parameters from set 2

        Args:
            parent0 (numpy array or list): set where parameter values will be changed
            parent1 (n_descendants): set which provides parameter values for change in set 1
            n_crossed (int): how many parameters to change
        """
        cross_genes = [g for g in self.param_change_ids if parent0[g] != parent1[g]]
        if len(cross_genes) > 0:
            params_crossed = np.random.choice(cross_genes, n_crossed, False)
            child = parent0.copy()
            child[params_crossed] = parent1[params_crossed]
        else:
            # if parents happen to have the same parameters, mutate instead
            parents = [parent0, parent1]
            child = self.mutate(parents[np.random.choice(2, 1)[0]])
        return child

    def get_best_ids_weighted(self, fits):
        """
        Calculates best parameter sets (with the highest fitness) with weights for different metrics

        Args:
            fits (numpy array): list of accuracies for each parameter set
        """
        weights = np.array([self.metric_weights[self.index_to_metric[i]] for i in range(fits.shape[1])])
        weighted_scores = fits @ weights
        return np.argsort(weighted_scores)[::-1]

    def get_best_metric_value(self, metric_values):
        weights = [self.metric_weights[self.index_to_metric[i]] for i in range(len(metric_values))]
        return metric_values[np.argmax(weights)]

    def calculate_fitness(self, param_set, output, idx):
        """
        Measures fitness (e.g. accuracy) of one parameter set

        Args:
            param_set (list or numpy array): parameter set
            output: a variable for multiprocessing that stores output from different processes
            idx (int): index of the parameter set in multiprocessing
        """
        print(f'{mp.current_process().name} STARTED params_set {idx}')
        
        model_params = {self.index_to_param[i]: v for i, v in enumerate(param_set)}
        model_params["plasticity_type"] = self.plasticity_type
        model_params["plasticity_sym"] = self.plasticity_sym
        model, res = self.experiment.run(model_class=self.model_class, model_params=model_params,
                                  thresholds=None,
                                  proc_name=mp.current_process().name)
        score_indexed = [res[self.index_to_metric[i]]['score'] for i in self.index_to_metric]
        threshold_indexed = [res[self.index_to_metric[i]]['threshold'] for i in self.index_to_metric]
        
        print(f'{mp.current_process().name} FINISHED params_set {idx}: {res}')
        cur_output = {'fitness': score_indexed, 'threshold': threshold_indexed, 'params': param_set}
        output.put(cur_output)
        return cur_output
    
    def select_best(self, param_sets, generation):
        """
        Performs selection operation: selects the best descendants of the generation.
        They will be used for further mutation and crossover
        Return best parameter sets per generation + best fitness

        Args:
            param_sets (numpy array or list): all parameter sets in current generation
            generation (int): generation index
        """

        # Initialize multiprocessing across parameter sets in one generation
        results = []
        output = mp.Queue()
        processes = []
        for i in range(len(param_sets)):
            p = mp.Process(target=self.calculate_fitness, args=(param_sets[i], output, i),
                           name=f'Process_{i}')
            processes.append(p)
            p.start()
        for p in processes:
            p.join()
            print(f'{p.name} joined')
            
        while not output.empty():
            o = output.get()
            results.append(o)

        for p in processes:
            p.terminate()

        fits = np.array([el['fitness'] for el in results])
        
        # find best descendants
        print("metric_weights", self.metric_weights)
        ids_sorted = self.get_best_ids_weighted(fits)
        best_ids = ids_sorted[:self.group_desc]
        best_fit = self.get_best_metric_value(fits[ids_sorted[0]])

        print("best_fit", best_fit)
        for i in best_ids:
            print('best', i, results[int(i)])

        for best_i in best_ids:
            res_fitness = {}
            res_params = {}
            res_threshold = {}
            
            logger_row = f'GEN {generation} FITNESS '
            for i, v in enumerate(results[best_i]['fitness']):
                k = self.index_to_metric[i]
                res_fitness[k] = v
                logger_row += f'{k}: {str(round(v, 4))} '
                
            logger_row = logger_row[:-1]
            logger_row += '; THRESHOLD '
            for i, v in enumerate(results[best_i]['threshold']):
                k = self.index_to_metric[i]
                res_threshold[k] = v
                logger_row += f'{k}: {str(round(v,4))} '

            logger_row = logger_row[:-1]
            logger_row += '; PARAMS '
            for i, v in enumerate(results[best_i]['params']):
                k = self.index_to_param[i]
                res_params[k] = v
                logger_row += f'{k}: {str(round(v,4))} '

            if self.logger is not None:
                self.logger.info(logger_row[:-1])
            print(res_fitness, res_threshold, res_params)

        return np.array([results[i]['params'] for i in best_ids]), best_fit
    
    def next_gen(self, param_sets, generation):
        """
        Performs selection, crossover and mutation operations, to create next generation

        Args:
            param_sets (numpy array): all parameter sets in one generation
            generation (int): generation index
        """
        print('REMOVING DUPLICATES')
        param_sets = self.remove_duplicates(param_sets)

        print('SELECTION')
        gens_best, best_fit = self.select_best(param_sets, generation)
        print(f'best shape {gens_best.shape}')
        
        print('MUTATION')
        gens_mutated = np.apply_along_axis(self.mutate, 1, gens_best)
        print(f'mutated shape {gens_mutated.shape}')
        
        print('CROSSOVER')
        parents0 = gens_best.copy()
        parents1_ids = [np.random.choice(np.delete(np.arange(len(gens_best)), [i])) for i in range(len(gens_best))]
        parents1 = gens_best[parents1_ids]
        gens_crossover = np.array([self.crossover(parents0[i], parents1[i]) for i in range(len(parents0))])
        print(f'crossover shape {gens_crossover.shape}')
        
        print('RANDOM')
        gens_random = self.random_gen(self.group_desc)
        print(f'random shape {gens_random.shape}')
        
        gens = np.concatenate([gens_best, gens_mutated, gens_crossover, gens_random])
        print(f'all shape {gens.shape}')
        return gens, best_fit
    
    def fit(self, n_generations=100, start_generation=1, target_acc=1.0, n_early_stop=10, init_params=None, init_fits=None):
        """
        Performs optimization with a genetic algorithm

        Args:
            n_generations (int): the number of generations to optimize
            start_generation (int): which generation to start optimization from
                (for correct logging, e.g. when the optimization continues from the middle)
            target_acc (float): stop if accuracy reaches this value
            n_early_stop (int): stop if fitness is not increasing for this number of generations
            init_params (numpy array): initial parameter sets
            init_fits: initial fitness values. useful if optimization was performed previously and starts over
        """
        # initial parameter sets
        params = np.zeros(shape=(n_generations-start_generation+2, self.generation_size, self.n_params))
        params[0] = self.first_gen(init_params)

        best_fits = []
        if init_fits is not None:
            best_fits.extend(init_fits)

        early_stop = False
        for g in range(start_generation, n_generations+1):
            print(f'GENERATION {g}/{n_generations} STARTED')
            params[g-start_generation+1], best_fit = self.next_gen(params[g-start_generation], g)
            print(f'GENERATION {g}/{n_generations} FINISHED')

            print('best fits BEFORE CHECK', best_fit, best_fits)
            best_fits.append(best_fit)
            if best_fit >= target_acc:
                early_stop = True
                print(f'OPTIMIZATION FINISHED at generation {g}. Achieved target_acc {target_acc}')
                break
            elif len(best_fits) >= n_early_stop and all(i >= j for i, j in zip(best_fits[-n_early_stop:], best_fits[-n_early_stop+1:])):
                early_stop = True
                print(f'OPTIMIZATION FINISHED at generation {g}. No improvements for {n_early_stop} generations')
                break

        if early_stop is False:
            print(f'OPTIMIZATION FINISHED. Ran for {n_generations} generations')