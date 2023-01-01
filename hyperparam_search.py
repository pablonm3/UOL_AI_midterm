# If you on a Windows machine with any Python version
# or an M1 mac with any Python version
# or an Intel Mac with Python > 3.7
# this multi-threaded version does not work
# please use test_ga_single_thread.py on those setups

import unittest
import population
import simulation
import genome
import creature
import numpy as np
import pandas as pd
import time
import datetime;



DEFAULT_PARAMS = {
    "pop_size": 10,
    "init_gene_count": 3,
    "iterations": 1000,
    "mutation_rate": 0.1,
    "pool_size": 8
}

EXPERIMENTS = [
    {"name": "test", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [20]}},
    {"name": "test2", "log_interval": "NORMAL", "Y": "MEAN_FITNESS_DELTA", "X":{"pop_size": [10, 50,]}, "Z": {"iterations": [10, 30]}},
    {"name": "test3", "log_interval": "EVERY_ITERATION", "Y": "EVOLUTION_SPEED", "X":{"iterations": [10, 20, 30]}},
    {"name": "no_iterations", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [2000]}},
    {"name": "no_iterations2", "log_interval": "NORMAL", "Y": "EVOLUTION_SPEED", "X":{"iterations": [100, 300, 500, 1000]}},
    {"name": "pop_size", "log_interval": "NORMAL", "Y": "MEAN_FITNESS_DELTA", "X":{"pop_size": [10, 15, 20,30, 50, 60, 100]}, "Z": {"iterations": [100, 300, 500, 1000]}},
    {"name": "pop_size2", "log_interval": "NORMAL", "Y": "EVOLUTION_SPEED", "X":{"pop_size": [10, 20, 30, 40, 50, 60, 70, 100]}},
    {"name": "pool_size", "log_interval": "NORMAL", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [100, 150, 300, 400, 500, 1000]}, "Z": {"pool_size": [1,4,8]}},
    {"name": "pool_size2", "log_interval": "NORMAL", "Y": "EVOLUTION_SPEED", "X":{"pool_size": [1, 2 , 4, 8]}},
    {"name": "init_gene_count", "log_interval": "NORMAL", "Y": "MEAN_FITNESS_DELTA", "X":{"init_gene_count": [1, 3, 5,6, 8, 10]}},
    {"name": "init_gene_count2", "log_interval": "NORMAL", "Y": "EVOLUTION_SPEED", "X":{"init_gene_count": [1, 3, 5,6, 8, 10]}},
    {"name": "mutation_rate", "log_interval": "NORMAL", "Y": "MEAN_FITNESS_DELTA", "X":{"mutation_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8]}},
    {"name": "mutation_rate2", "log_interval": "NORMAL", "Y": "EVOLUTION_SPEED", "X":{"mutation_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8]}}
]

EXPERIMENTS_GENOME = [
    {"name": "baseline", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]} },
    {"name": "link-length-longer1", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-length", 4)  },
    {"name": "link-length-longer2", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-length", 6)  },
    {"name": "link-length-shorter1", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-length", 1)  },
    {"name": "link-length-shorter2", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-length", 0.5)  },
    {"name": "link-mass-lighter1", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-mass", 0.5)  },
    {"name": "link-mass-lighter2", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-mass", 0.25)  },
    {"name": "link-mass-heavier1", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-mass", 2)  },
    {"name": "link-mass-heavier2", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-mass", 4)  },
    {"name": "link-recurrence-higher1", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-recurrence", 4)  },
    {"name": "link-recurrence-higher2", "log_interval": "EVERY_ITERATION", "Y": "MEAN_FITNESS_DELTA", "X":{"iterations": [1000]}, "GEN_CHANGE": ("link-recurrence", 5)  },

]

TEST_EXPERIMENTS_TO_RUN = [
                      "test",
                      "test2",
                      "test3"]
EXPERIMENTS_TO_RUN=  [
                      #"no_iterations",
                      #"no_iterations2",
                      #"pop_size",
                      #"pop_size2",
                      #"pool_size",
                      #"pool_size2",
                      #"init_gene_count",
                      #"init_gene_count2",
                      #"mutation_rate",
                      "mutation_rate2"
                    ]

EXPERIMENTS_GENOME_TO_RUN = [
    #"link-length-longer1",
    #"link-length-longer2",
    #"link-length-shorter1",
    #"link-length-shorter2",
    #"link-mass-lighter1",
    #"link-mass-lighter2",
    #"link-mass-heavier1",
   # "link-mass-heavier2",
    #"baseline",
    #"link-recurrence-higher1",
    "link-recurrence-higher2"
]


class HyperparamSearcher():

    def get_hyperparam(self, type):
        param = DEFAULT_PARAMS[type]
        return param

    def get_mean_fitness(self, creatures):
        sum = 0
        for cr in creatures:
            sum += cr.get_distance_travelled()
        mean_fitness = sum / len(creatures)
        return mean_fitness

    def get_metric(self, hyperparams, experiment_params, creatures, initial_fitness):
        metric_name = experiment_params["Y"]
        mean_fitness = self.get_mean_fitness(creatures)
        if(metric_name == "MEAN_FITNESS_DELTA"):
            return mean_fitness - initial_fitness
        if(metric_name == "EVOLUTION_SPEED"):
            return (mean_fitness - initial_fitness) / hyperparams["iterations"]

        raise Exception("Invalid metric")

    def find_exp_w_name(self, name):
        for e in EXPERIMENTS_GENOME:
            if(e["name"] == name):
                return e
        return None

    def run_wrapper(self, hyperparams, experiment_params, y, x , x_name, z=None, z_name=None, log_interval=None):
        trial = 0
        while(True):
            try:
                return self.run(hyperparams, experiment_params, y=y, x=x, x_name=x_name, z=z, z_name=z_name,  log_interval=log_interval)
            except Exception as ex:
                print("Exception Raised: ", ex)
                if(trial > 3):
                    print("skipping experiment for params: ", experiment_params)
                    print("hyperparams: ", hyperparams)
                    return []
                else:
                    print("Retriying in 5 seconds...")
                    time.sleep(5)
                    trial += 1


    def run_experiment(self, experiment_name):
        experiment_params = self.find_exp_w_name(experiment_name)
        if("GEN_CHANGE" in experiment_params):
            genome.Genome.change_scale(experiment_params["GEN_CHANGE"][0], experiment_params["GEN_CHANGE"][1])
        params = DEFAULT_PARAMS.copy()
        X_dict = experiment_params["X"]
        X_prop = list(X_dict.keys())[0] if X_dict else None
        X_values = X_dict.get(X_prop, [])
        Y_metric = experiment_params["Y"]
        Z_dict = experiment_params.get("Z", {})
        Z_prop = list(Z_dict.keys())[0] if Z_dict else None
        Z_values = Z_dict.get(Z_prop, [])
        log_interval = experiment_params["log_interval"]
        r = []
        for X_value in X_values:
            params[X_prop] = X_value
            if(Z_prop):
                for Z_value in Z_values:
                    params[Z_prop] = Z_value
                    records = self.run_wrapper(params, experiment_params, y=Y_metric, x=X_value, x_name=X_prop, z=Z_value, z_name=Z_prop,  log_interval=log_interval)
                    r = r + records
            else:
                records = self.run_wrapper(params, experiment_params, y=Y_metric, x=X_value, x_name=X_prop, log_interval= log_interval)
                r = r + records
            print("storing intermediate values")
            store_exp_name(pd.DataFrame(r), experiment_name, is_final=False)
        df = pd.DataFrame(r)
        return df

    def run(self, hyperparams, experiment_params, y, x , x_name, z=None, z_name=None, log_interval=None):
        print("starting experiment with experiment_params: ", experiment_params, " and hyperparams: ", hyperparams)
        logs = []
        no_iterations = hyperparams.get("iterations")
        pop_size = hyperparams.get("pop_size")
        init_gene_count = hyperparams.get("init_gene_count")
        mutation_rate = hyperparams.get("mutation_rate")
        pool_size = hyperparams.get("pool_size")

        pop = population.Population(pop_size=pop_size,
                                    gene_count=init_gene_count)
        sim = simulation.ThreadedSim(pool_size=pool_size)
        #sim = simulation.Simulation()
        init_fitness = None
        max_fitness =0
        best_cr = None
        iteration_best_cr = 0
        for iteration in range(no_iterations):
            sim.eval_population(pop, 2400)
            fits = [cr.get_distance_travelled()
                    for cr in pop.creatures]
            links = [len(cr.get_expanded_links())
                     for cr in pop.creatures]
            if(iteration % 50 == 0):
                print("timestamp: ", datetime.datetime.now(), iteration, "fittest:", np.round(np.max(fits), 3),
                      "mean:", np.round(np.mean(fits), 3), "mean links", np.round(np.mean(links)), "max links", np.round(np.max(links)))
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                # now we have the parents!
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=mutation_rate, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            # elitism
            max_fit = np.max(fits)
            for cr in pop.creatures:
                distance_travelled = cr.get_distance_travelled()
                if distance_travelled == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    if distance_travelled > max_fitness:
                        max_fitness = distance_travelled
                        best_cr = cr
                        iteration_best_cr = iteration
                    break
            if(iteration == 0):
                init_fitness = self.get_mean_fitness(pop.creatures)
            if(experiment_params["log_interval"] == "EVERY_ITERATION"):
                metric = self.get_metric(hyperparams, experiment_params, pop.creatures, initial_fitness=init_fitness)
                record = {}
                record[y] = metric
                if(x_name == "iterations"):
                    record[x_name] = iteration +1
                else:
                    record[x_name] = x
                if(z_name):
                    record[z_name] = z
                logs.append(record)
            last_evaluated_creatures = pop.creatures
            pop.creatures = new_creatures
        filename = f"best_creatures/{experiment_params['name']}_{round(max_fitness,2)}_"+str(iteration_best_cr)+".csv"
        print("saving best creature: ", filename)
        genome.Genome.to_csv(best_cr.dna, filename)
        if(experiment_params["log_interval"] == "NORMAL"):
            metric = self.get_metric(hyperparams, experiment_params, last_evaluated_creatures, initial_fitness=init_fitness)
            record = {}
            record[y] = metric
            if(x_name == "iterations"):
                record[x_name] = iteration +1
            else:
                record[x_name] = x
            if(z_name):
                record[z_name] = z
            logs.append(record)
        return logs

def store_exp_name(df, experiment_name, is_final=True):
    dir_name = None
    if(is_final):
        print("experiment finished: ", experiment_name)
        dir_name = "hyperparam_search_gene_spec"
    else:
        dir_name = "hyperparam_search_gene_spec_WIP"
    print("saving to CSV...")
    path = f"{dir_name}/{experiment_name}.csv"
    df.to_csv(path, index=False)

if __name__ == '__main__':
    hs = HyperparamSearcher()
    for experiment_name in EXPERIMENTS_GENOME_TO_RUN:
        print("starting experiment: ", experiment_name)
        results_dataframe = hs.run_experiment(experiment_name)
        store_exp_name(results_dataframe, experiment_name)