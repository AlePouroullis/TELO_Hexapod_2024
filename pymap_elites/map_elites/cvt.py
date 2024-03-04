#! /usr/bin/env python
#| This file is a part of the pymap_elites framework.
#| Copyright 2019, INRIA
#| Main contributor(s):
#| Jean-Baptiste Mouret, jean-baptiste.mouret@inria.fr
#| Eloise Dalin , eloise.dalin@inria.fr
#| Pierre Desreumaux , pierre.desreumaux@inria.fr
#|
#|
#| **Main paper**: Mouret JB, Clune J. Illuminating search spaces by
#| mapping elites. arXiv preprint arXiv:1504.04909. 2015 Apr 20.
#|
#| This software is governed by the CeCILL license under French law
#| and abiding by the rules of distribution of free software.  You
#| can use, modify and/ or redistribute the software under the terms
#| of the CeCILL license as circulated by CEA, CNRS and INRIA at the
#| following URL "http://www.cecill.info".
#|
#| As a counterpart to the access to the source code and rights to
#| copy, modify and redistribute granted by the license, users are
#| provided only with a limited warranty and the software's author,
#| the holder of the economic rights, and the successive licensors
#| have only limited liability.
#|
#| In this respect, the user's attention is drawn to the risks
#| associated with loading, using, modifying and/or developing or
#| reproducing the software by the user in light of its specific
#| status of free software, that may mean that it is complicated to
#| manipulate, and that also therefore means that it is reserved for
#| developers and experienced professionals having in-depth computer
#| knowledge. Users are therefore encouraged to load and test the
#| software's suitability as regards their requirements in conditions
#| enabling the security of their systems and/or data to be ensured
#| and, more generally, to use and operate it in the same conditions
#| as regards security.
#|
#| The fact that you are presently reading this means that you have
#| had knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import time
import multiprocessing    # Comment
import copy
#from mpi4py.futures import MPIPoolExecutor    # Uncomment
# from scipy.spatial import cKDTree : TODO -- faster?
from sklearn.neighbors import KDTree

from pymap_elites.map_elites import common as cm



def __add_to_archive(s, centroid, archive, kdt):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    s.centroid = n
    if n in archive:
        if s.fitness > archive[n].fitness:
            archive[n] = s
            return 1
        return 0
    else:
        archive[n] = s
        return 1


# evaluate a single vector (x) with a function f and return a species
# t = vector, function
def __evaluate(t):
    z, f = t  # evaluate z with function f
    fit, desc = f(z, collision_fatal=True)
    return cm.Species(z, desc, fit)

# map-elites algorithm (CVT variant)
def compute(dim_map, genomes, f,
            n_niches=1000,
            max_evals=1e5,
            params=cm.default_params,
            log_file=None,
            archive_file = None,
            archive_load_file = None,
            start_index = 0,
            variation_operator=cm.variation):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    # setup the parallel processing pool
    num_cores = multiprocessing.cpu_count()     # Comment
    pool = multiprocessing.Pool(num_cores)     # Comment
    #pool = MPIPoolExecutor()    # Uncomment

    # create the CVT
    start_time = time.time()
    c = cm.cvt(n_niches, dim_map,
              params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    cm.__write_centroids(c)
    print("Time Taken for centroid: ", time.time() - start_time)

    archive = {} # init archive (empty)
    n_evals = start_index # number of evaluations since the beginning
    b_evals = 0 # number evaluation since the last dump

    # main loop
    ## Read in archive file and archive genomes if they exist
    if (archive_load_file is not None):
        with open(archive_load_file) as file:
            line_count = 0
            for line in file:
                archive_val = line.split(' ')
                fitness = float(archive_val[0])
                #centroid = archive_val[1:7]
                description = archive_val[7:13]
                description = [float(x) for x in description]
                x = genomes[line_count]
                s = cm.Species(x = x, desc=description, fitness=fitness)
                __add_to_archive(s, s.desc, archive, kdt)
                line_count += 1

    while (n_evals < max_evals):

        print(n_evals)
        start_time = time.time()
        to_evaluate = []
        # random initialization
        #if len(archive) <= params['random_init'] * n_niches:
        if (n_evals == 0):
            for i in range(0, len(genomes)):

                # x = np.random.uniform(low=params['min'], high=params['max'], size=dim_x)     ## Where the map is initialised, this is where I would pass in the high performing gaits
                x = genomes[i]
                to_evaluate += [(x, f)]
        else:  # variation/selection loop
            keys = list(archive.keys())
            # we select all the parents at the same time because randint is slow
            rand1 = np.random.randint(len(keys), size=params['batch_size'])
            #rand2 = np.random.randint(len(keys), size=params['batch_size'])
            for n in range(0, params['batch_size']):
                # parent selection
                x = archive[keys[rand1[n]]]
                #y = archive[keys[rand2[n]]]
                # copy & add variation
                z = copy.deepcopy(x)
                z = variation_operator(z.x)
                to_evaluate += [(z, f)]
        # evaluation of the fitness for to_evaluate
        s_list = cm.parallel_eval(__evaluate, to_evaluate, pool, params)    ## Problem here
        # natural selection
        for s in s_list:
            __add_to_archive(s, s.desc, archive, kdt)
        # count evals
        n_evals += len(to_evaluate)
        b_evals += len(to_evaluate)

        # write archive
        if b_evals >= params['dump_period'] and params['dump_period'] != -1:
            print("[{}/{}]".format(n_evals, int(max_evals)), end=" ", flush=True)
            cm.__save_archive(archive, n_evals, archive_file)
            b_evals = 0
        # write log
        if log_file != None:
            fit_list = np.array([x.fitness for x in archive.values()])
            log_file.write("{} {} {} {} {} {} {}\n".format(n_evals, len(archive.keys()),
                    fit_list.max(), np.mean(fit_list), np.median(fit_list),
                    np.percentile(fit_list, 5), np.percentile(fit_list, 95)))
            log_file.flush()
        print("Time taken for one batch: ", time.time() - start_time)
    cm.__save_archive(archive, n_evals, archive_file)
    return archive