from mailer.hexapod.simulator import Simulator
from DecoupledSUPGController18Motors import DecoupledSUPGController18Motors
import neat
import neat.nn
import numpy as np

#configure neat for the SUPG CPPN
#chnage to config_BSUPG or coupled config_SUPG 
supgConfig = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'config_DecoupledSUPG') 


def evaluate_gait(x, duration=5, visualizer=False,collision_fatal=False,failed_legs=[]):
    cppn = neat.nn.FeedForwardNetwork.create(x, supgConfig)

    try:
        controller = DecoupledSUPGController18Motors(cppn, [] )
    except:
        return 0, np.zeros(6)
    # Initialise Simulator
    simulator = Simulator(controller=controller, visualiser=visualizer, collision_fatal=collision_fatal, failed_legs=failed_legs)
    # Step in simulator
    contact_sequence = np.full((6, 0), False)
    for t in np.arange(0, duration, step=simulator.dt):
        try:
            simulator.step()
        except RuntimeError as collision:
            fitness = 0, np.zeros(6)
        contact_sequence = np.append(contact_sequence, simulator.get_supporting_legs().reshape(-1, 1), axis=1)
    fitness = simulator.base_pos()[0]  # distance travelled along x axis
    descriptor = np.nan_to_num(np.sum(contact_sequence, axis=1) / np.size(contact_sequence, axis=1), nan=0.0,
                               posinf=0.0, neginf=0.0)
    # Terminate Simulator
    simulator.terminate()
    # print(difference)
    # fitness = difference
    # Assign fitness to genome
    x.fitness = fitness
    return fitness, descriptor

