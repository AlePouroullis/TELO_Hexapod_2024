from mailer.hexapod.simulator import Simulator
from .DecoupledSUPGController18Motors import DecoupledSUPGController18Motors
import neat
import neat.nn
import numpy as np
import pickle
import visualize as vz
import os
import sys


### GLOBALS ### 

config_file_path = os.path.join(os.path.dirname(
    os.path.realpath(__file__)), "config_DecoupledSUPG")

# configure neat for the SUPG CPPN
# chnage to config_BSUPG or coupled config_SUPG
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file_path)

# radius, offset, step_height, phase, duty_factor
tripod_gait = [0.15, 0, 0.05, 0.5, 0.5,  # leg 1
               0.15, 0, 0.05, 0.0, 0.5,  # leg 2
               0.15, 0, 0.05, 0.5, 0.5,  # leg 3
               0.15, 0, 0.05, 0.0, 0.5,  # leg 4
               0.15, 0, 0.05, 0.5, 0.5,  # leg 5
               0.15, 0, 0.05, 0.0, 0.5]  # leg 6

VISUALIZE = False

### END OF GLOBALS ###


def fitness_function(genome, config, duration=5):
    '''
    The fitness function is the Cartesian distance from the starting point 
    to the endpoint at the end of the 15-second trial.
    The duration was chosen based on the SUPG paper (Morse et al., 2013)
    '''
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)

    try:
        # Initialize Decoupled SUPG controller
        # Add any additional parameters if needed
        controller = DecoupledSUPGController18Motors(cppn, [])
    except:
        return 0

    # Initialise Simulator
    simulator = Simulator(controller=controller,
                          visualiser=VISUALIZE, collision_fatal=True)

    # Step in simulator
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            return 0

    fitness = simulator.base_pos()[0]   # distance travelled along x axis
    # Terminate Simulator
    simulator.terminate()

    return fitness


def run(gens: int):
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    stats = neat.statistics.StatisticsReporter()
    p.add_reporter(stats)
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(False))

    # running in parallel
    pe = neat.ParallelEvaluator(
        6, fitness_function)

    # Run until a solution is found

    winner = p.run(pe.evaluate, gens)
    return winner, stats


if __name__ == "__main__":
    """
    This script is designed to run an evolutionary algorithm using the NEAT (NeuroEvolution of Augmenting Topologies) framework. It sets up the necessary directory structure for storing output files, executes the evolutionary process, and saves the results including fitness statistics, graphical representations of species evolution, and the network structure of the best performing genome.

    Key functionalities of the script include:
    - Creation of directories for storing outputs such as fitness data, graphs, best genomes, and CPPNs (Compositional Pattern Producing Networks).
    - Parsing command-line arguments for controlling the number of evolutionary runs and specifying unique identifiers for output files.
    - Running the NEAT algorithm for a specified number of generations and storing the resulting statistics.
    - Generating and saving visualizations of the evolutionary process and species distribution.
    - Saving the best performing network (winner) as a serialized object for future use or analysis.

    Usage:
        The script is run from the command line, requiring two arguments:
        1. num_runs: An integer specifying the number of generations to run the NEAT algorithm.
        2. file_number: A string or number used to uniquely identify output files.

    Example:
        python evolve.py 100 1
 
    Note:
        - The script assumes the presence of a properly configured NEAT configuration file.
        - The NEAT framework and required visualization modules should be installed and imported as `neat` and `vz` respectively.
    """
    base_path = os.path.join("DecoupledSUPG", "evolution_output")
    genome_fitness_path = os.path.join(base_path, "genome_fitness")
    graphs_path = os.path.join(base_path, "graphs")
    best_genomes_path = os.path.join(base_path, "best_genomes")
    stats_path = os.path.join(base_path, "stats")
    CPPNS_path = os.path.join(base_path, "CPPNS")
    pickles_path = os.path.join(base_path, "pickles")
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    if not os.path.exists(genome_fitness_path):
        os.mkdir(genome_fitness_path)
    if not os.path.exists(graphs_path):
        os.mkdir(graphs_path)
    if not os.path.exists(best_genomes_path):
        os.mkdir(best_genomes_path)
    if not os.path.exists(stats_path):
        os.mkdir(stats_path)
    if not os.path.exists(CPPNS_path):
        os.mkdir(CPPNS_path)
    if not os.path.exists(pickles_path):
        os.mkdir(pickles_path)

    num_runs = int(sys.argv[1])
    file_number = (sys.argv[2])
    winner, stats = run(num_runs)

    fitness_history_file = os.path.join(
        genome_fitness_path, "fitness_history" + file_number + ".csv")
    average_fitness_file = os.path.join(
        genome_fitness_path, "average_fitness" + file_number + ".svg")
    speciation_file = os.path.join(
        graphs_path, "speciation" + file_number + ".svg")
    stats.save_genome_fitness(delimiter=',', filename=fitness_history_file)
    vz.plot_stats(stats, ylog=False, view=True, filename=average_fitness_file)
    vz.plot_species(stats, view=True, filename=speciation_file)

    # create network with winning genome
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    # save CPPN
    with open(os.path.join(pickles_path, "decoupled_supg_18_motors" + file_number + ".pkl"), 'wb') as output:
        pickle.dump(winner_net, output, pickle.HIGHEST_PROTOCOL)

    # save cppn structure as graph
    vz.draw_net(config, winner,
                filename="Output/graphs/NEATWINNER" + file_number)
