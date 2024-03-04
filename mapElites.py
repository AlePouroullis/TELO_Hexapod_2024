import pickle
import neat
import neat.nn
import os
import sys
import pymap_elites.map_elites.common as cm
import pymap_elites.map_elites.cvt as cvt_map_elites
from neat.reporting import ReporterSet
import glob
import re
from controller_tools import evaluate_gait

"""
A script to produce the HyperNEAT maps
The script takes two command line arguments:
1) The size of the map to be tested
2) The run/map number
"""
#configure neat for the SUPG CPPN
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     r'config_DecoupledSUPG')  


# Method to load in initial high performing genomes
def load_genomes(num=200):
    reporters = ReporterSet()
    stagnation = config.stagnation_type(config.stagnation_config, reporters)
    reproduction = config.reproduction_type(config.reproduction_config, reporters, stagnation)

    genomes = reproduction.create_new(config.genome_type, config.genome_config, num)
    return list(genomes.values())

def get_latest_checkpoint(runNum, mapSize):
    # List all genome files for the given runNum
    files = glob.glob(os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', 'archive_genome*.pkl'))
    
   # Extract the evaluation numbers from these filenames
    matches = [re.search(r'genome(\d+).pkl$', file) for file in files]
    evals = [int(m.group(1)) for m in matches if m]

    if not evals:
        raise ValueError(f"No evaluations found for run {runNum} and map size {mapSize}")
    
    # Determine the file with the highest evaluation number
    latest_eval = max(evals)
    
    return os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', f'archive_genome{latest_eval}.pkl'), \
        os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', f'archive{latest_eval}.dat'), latest_eval

def map_elites_evaluate_gait(x, collision_fatal=False):
    return evaluate_gait(x, collision_fatal=collision_fatal)

if __name__ == '__main__':
    mapSize = int(sys.argv[1])
    runNum = (sys.argv[2])

    # Map Elites paramters
    params = \
        {
            # more of this -> higher-quality CVT
            "cvt_samples": 100000,
            # we evaluate in batches to parallelise
            "batch_size": 2390,
            # proportion of niches to be filled before starting
            "random_init": 0.01,
            # batch for random initialization
            "random_init_batch": 2390,
            # when to write results (one generation = one batch)
            "dump_period": 1e6,   # Change that
            # do we use several cores?
            "parallel": True,
            # do we cache the result of CVT and reuse?
            "cvt_use_cache": True,
            # min/max of parameters
            "min": 0,
            "max": 1,
        }
    
    max_evals = 1e7


     # Create necessary directory if it doesn't exist
    os.makedirs(os.path.join("mapElitesOutput", f"{mapSize}", f"{runNum}"), exist_ok=True)

    archive_file = os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', 'archive')

    # Check if there are checkpoints for the given run number and map size
    if os.path.exists(os.path.join("mapElitesOutput", str(mapSize), str(runNum))):
        # Use the latest checkpoint
        genome_filename, archive_checkpoint, start_index = get_latest_checkpoint(runNum, mapSize)
        
        with open(genome_filename, 'rb') as f:
            genomes = pickle.load(f)

        log_file = open(os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', 'log.dat'), 'a')

        # Used when loading in checkpointed values
        archive = cvt_map_elites.compute(6, genomes, map_elites_evaluate_gait, n_niches=mapSize, max_evals=max_evals,
                                        log_file=log_file, archive_file=archive_file,
                                        archive_load_file=archive_checkpoint, params=params, start_index=start_index,
                                        variation_operator=cm.neatMutation)
    else:
        # Load in high performing genomes
        genomes = load_genomes()

        log_file = open(os.path.join('mapElitesOutput', f'{mapSize}', f'{runNum}', 'log.dat'), 'w')

        archive = cvt_map_elites.compute(6, genomes, map_elites_evaluate_gait, n_niches = mapSize, max_evals=max_evals,
                                log_file=log_file, archive_file=archive_file,
                                params=params, variation_operator=cm.neatMutation)

