import retro
from typing import Tuple, List, Optional
import random
import math
import numpy as np
import os
from argparse import Namespace

from utils import SMB
from config import Config
from mario import Mario, save_mario, save_stats, load_mario

from genetic_algorithm.individual import Individual
from genetic_algorithm.population import Population
from genetic_algorithm.selection import (
    elitism_selection,
    tournament_selection,
    roulette_wheel_selection,
)
from genetic_algorithm.crossover import simulated_binary_crossover as SBX
from genetic_algorithm.mutation import gaussian_mutation


class Console(object):
    def __init__(self, args: Namespace, config: Optional[Config] = None):
        self.args = args
        self.config = config

        self.current_generation = 0
        # This is the generation that is actual 0. If you load individuals then you might end up starting at gen 12, in which case
        # gen 12 would be the true 0
        self._true_zero_gen = 0

        # Keys correspond with B, NULL, SELECT, START, U, D, L, R, A
        # index                0  1     2       3      4  5  6  7  8
        self.keys = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)

        # I only allow U, D, L, R, A, B and those are the indices in which the output will be generated
        # We need a mapping from the output to the keys above
        self.ouput_to_keys_map = {
            0: 4,  # U
            1: 5,  # D
            2: 6,  # L
            3: 7,  # R
            4: 8,  # A
            5: 0,  # B
        }

        # Initialize the starting population
        individuals: List[Individual] = []

        # Load any individuals listed in the args.load_inds
        num_loaded = 0
        if args.load_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.load_file, "settings.config"))
                except:
                    raise Exception(f"settings.config not found under {args.load_file}")

            set_of_inds = set(args.load_inds)

            for ind_name in os.listdir(args.load_file):
                if ind_name.startswith("best_ind_gen"):
                    ind_number = int(ind_name[len("best_ind_gen") :])
                    if ind_number in set_of_inds:
                        individual = load_mario(args.load_file, ind_name, self.config)
                        # Set debug stuff if needed
                        if args.debug:
                            individual.name = f"m{num_loaded}_loaded"
                            individual.debug = True
                        individuals.append(individual)
                        num_loaded += 1

            # Set the generation
            self.current_generation = max(set_of_inds) + 1  # +1 becauase it's the next generation
            self._true_zero_gen = self.current_generation

        # Load any individuals listed in args.replay_inds
        if args.replay_inds:
            # Overwrite the config file IF one is not specified
            if not self.config:
                try:
                    self.config = Config(os.path.join(args.replay_file, "settings.config"))
                except:
                    raise Exception(f"settings.config not found under {args.replay_file}")

            for ind_gen in args.replay_inds:
                ind_name = f"best_ind_gen{ind_gen}"
                fname = os.path.join(args.replay_file, ind_name)
                if os.path.exists(fname):
                    individual = load_mario(args.replay_file, ind_name, self.config)
                    # Set debug stuff if needed
                    if args.debug:
                        individual.name = f"m_gen{ind_gen}_replay"
                        individual.debug = True
                    individuals.append(individual)
                else:
                    raise Exception(f"No individual named {ind_name} under {args.replay_file}")
        # If it's not a replay then we need to continue creating individuals
        else:
            num_parents = max(self.config.Selection.num_parents - num_loaded, 0)
            for _ in range(num_parents):
                individual = Mario(self.config)
                # Set debug stuff if needed
                if args.debug:
                    individual.name = f"m{num_loaded}"
                    individual.debug = True
                individuals.append(individual)
                num_loaded += 1

        self.best_fitness = 0.0
        self._current_individual = 0
        self.population = Population(individuals)

        self.mario = self.population.individuals[self._current_individual]

        self.max_distance = 0  # Track farthest traveled in level
        self.max_fitness = 0.0
        self.env = retro.make(game="SuperMarioBros-Nes", state=f"Level{self.config.Misc.level}")

        # Determine the size of the next generation based off selection type
        self._next_gen_size = None
        if self.config.Selection.selection_type == "plus":
            self._next_gen_size = self.config.Selection.num_parents + self.config.Selection.num_offspring
        elif self.config.Selection.selection_type == "comma":
            self._next_gen_size = self.config.Selection.num_offspring

        self.env.reset()

    def run(self) -> None:
        # TODO: put all individuals of this generation into queue
        # start a process pool of N=CPU
        # each process will process an individual
        # this process will wait until all individuals are processed then start the new generation
        while True:
            self._update()

    def next_generation(self) -> None:
        self._increment_generation()
        self._current_individual = 0

        # Calculate fitness
        # print(', '.join(['{:.2f}'.format(i.fitness) for i in self.population.individuals]))

        if self.args.debug:
            print(f"----Current Gen: {self.current_generation}, True Zero: {self._true_zero_gen}")
            fittest = self.population.fittest_individual
            print(f"Best fitness of gen: {fittest.fitness}, Max dist of gen: {fittest.farthest_x}")
            num_wins = sum(individual.did_win for individual in self.population.individuals)
            pop_size = len(self.population.individuals)
            print(f"Wins: {num_wins}/{pop_size} (~{(float(num_wins)/pop_size*100):.2f}%)")

        if self.config.Statistics.save_best_individual_from_generation:
            folder = self.config.Statistics.save_best_individual_from_generation
            best_ind_name = "best_ind_gen{}".format(self.current_generation - 1)
            best_ind = self.population.fittest_individual
            save_mario(folder, best_ind_name, best_ind)

        if self.config.Statistics.save_population_stats:
            fname = self.config.Statistics.save_population_stats
            save_stats(self.population, fname)

        self.population.individuals = elitism_selection(self.population, self.config.Selection.num_parents)

        random.shuffle(self.population.individuals)
        next_pop = []

        # Parents + offspring
        if self.config.Selection.selection_type == "plus":
            # Decrement lifespan
            for individual in self.population.individuals:
                individual.lifespan -= 1

            for individual in self.population.individuals:
                config = individual.config
                chromosome = individual.network.params
                hidden_layer_architecture = individual.hidden_layer_architecture
                hidden_activation = individual.hidden_activation
                output_activation = individual.output_activation
                lifespan = individual.lifespan
                name = individual.name

                # If the indivdual would be alve, add it to the next pop
                if lifespan > 0:
                    m = Mario(
                        config,
                        chromosome,
                        hidden_layer_architecture,
                        hidden_activation,
                        output_activation,
                        lifespan,
                    )
                    # Set debug if needed
                    if self.args.debug:
                        m.name = f"{name}_life{lifespan}"
                        m.debug = True
                    next_pop.append(m)

        num_loaded = 0

        while len(next_pop) < self._next_gen_size:
            selection = self.config.Crossover.crossover_selection
            if selection == "tournament":
                p1, p2 = tournament_selection(self.population, 2, self.config.Crossover.tournament_size)
            elif selection == "roulette":
                p1, p2 = roulette_wheel_selection(self.population, 2)
            else:
                raise Exception('crossover_selection "{}" is not supported'.format(selection))

            L = len(p1.network.layer_nodes)
            c1_params = {}
            c2_params = {}

            # Each W_l and b_l are treated as their own chromosome.
            # Because of this I need to perform crossover/mutation on each chromosome between parents
            for l in range(1, L):
                p1_W_l = p1.network.params["W" + str(l)]
                p2_W_l = p2.network.params["W" + str(l)]
                p1_b_l = p1.network.params["b" + str(l)]
                p2_b_l = p2.network.params["b" + str(l)]

                # Crossover
                # @NOTE: I am choosing to perform the same type of crossover on the weights and the bias.
                c1_W_l, c2_W_l, c1_b_l, c2_b_l = self._crossover(p1_W_l, p2_W_l, p1_b_l, p2_b_l)

                # Mutation
                # @NOTE: I am choosing to perform the same type of mutation on the weights and the bias.
                self._mutation(c1_W_l, c2_W_l, c1_b_l, c2_b_l)

                # Assign children from crossover/mutation
                c1_params["W" + str(l)] = c1_W_l
                c2_params["W" + str(l)] = c2_W_l
                c1_params["b" + str(l)] = c1_b_l
                c2_params["b" + str(l)] = c2_b_l

                #  Clip to [-1, 1]
                np.clip(c1_params["W" + str(l)], -1, 1, out=c1_params["W" + str(l)])
                np.clip(c2_params["W" + str(l)], -1, 1, out=c2_params["W" + str(l)])
                np.clip(c1_params["b" + str(l)], -1, 1, out=c1_params["b" + str(l)])
                np.clip(c2_params["b" + str(l)], -1, 1, out=c2_params["b" + str(l)])

            c1 = Mario(
                self.config,
                c1_params,
                p1.hidden_layer_architecture,
                p1.hidden_activation,
                p1.output_activation,
                p1.lifespan,
            )
            c2 = Mario(
                self.config,
                c2_params,
                p2.hidden_layer_architecture,
                p2.hidden_activation,
                p2.output_activation,
                p2.lifespan,
            )

            # Set debug if needed
            if self.args.debug:
                c1_name = f"m{num_loaded}_new"
                c1.name = c1_name
                c1.debug = True
                num_loaded += 1

                c2_name = f"m{num_loaded}_new"
                c2.name = c2_name
                c2.debug = True
                num_loaded += 1

            next_pop.extend([c1, c2])

        # Set next generation
        random.shuffle(next_pop)
        self.population.individuals = next_pop

    def _crossover(
        self,
        parent1_weights: np.ndarray,
        parent2_weights: np.ndarray,
        parent1_bias: np.ndarray,
        parent2_bias: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        eta = self.config.Crossover.sbx_eta

        # SBX weights and bias
        child1_weights, child2_weights = SBX(parent1_weights, parent2_weights, eta)
        child1_bias, child2_bias = SBX(parent1_bias, parent2_bias, eta)

        return child1_weights, child2_weights, child1_bias, child2_bias

    def _mutation(
        self,
        child1_weights: np.ndarray,
        child2_weights: np.ndarray,
        child1_bias: np.ndarray,
        child2_bias: np.ndarray,
    ) -> None:
        mutation_rate = self.config.Mutation.mutation_rate
        scale = self.config.Mutation.gaussian_mutation_scale

        if self.config.Mutation.mutation_rate_type == "dynamic":
            mutation_rate = mutation_rate / math.sqrt(self.current_generation + 1)

        # Mutate weights
        gaussian_mutation(child1_weights, mutation_rate, scale=scale)
        gaussian_mutation(child2_weights, mutation_rate, scale=scale)

        # Mutate bias
        gaussian_mutation(child1_bias, mutation_rate, scale=scale)
        gaussian_mutation(child2_bias, mutation_rate, scale=scale)

    def _increment_generation(self) -> None:
        self.current_generation += 1

    def _update(self) -> None:
        """
        This is the main update method which is called based on the FPS timer.
        Genetic Algorithm updates, window updates, etc. are performed here.
        """
        ret = self.env.step(self.mario.buttons_to_press)

        ram = self.env.get_ram()
        tiles = SMB.get_tiles(ram)  # Grab tiles on the screen
        enemies = SMB.get_enemy_locations(ram)

        # self.mario.set_input_as_array(ram, tiles)
        self.mario.update(ram, tiles, self.keys, self.ouput_to_keys_map)

        if self.mario.is_alive:
            # New farthest distance?
            if self.mario.farthest_x > self.max_distance:
                if self.args.debug:
                    print("New farthest distance:", self.mario.farthest_x)
                self.max_distance = self.mario.farthest_x
        else:
            self.mario.calculate_fitness()
            fitness = self.mario.fitness

            if fitness > self.max_fitness:
                self.max_fitness = fitness
                max_fitness = "{:.2f}".format(self.max_fitness)
            # Next individual
            self._current_individual += 1

            # Is it the next generation?
            if (self.current_generation > self._true_zero_gen and self._current_individual == self._next_gen_size) or (
                self.current_generation == self._true_zero_gen and self._current_individual == self.config.Selection.num_parents
            ):
                self.next_generation()
            else:
                if self.current_generation == self._true_zero_gen:
                    current_pop = self.config.Selection.num_parents
                else:
                    current_pop = self._next_gen_size

            self.env.reset()
            self.mario = self.population.individuals[self._current_individual]
