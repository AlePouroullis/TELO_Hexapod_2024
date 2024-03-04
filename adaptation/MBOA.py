import sys
import os
sys.path.append(os.path.abspath("."))
from copy import copy
import numpy as np
import GPy
import pickle
from controller_tools import evaluate_gait

def mboa_gait_evaluation(x, failed_legs, show_visual=False):
	"""Takes parammeters from MBOA and returns fitness value for that controller/gait"""
	fitness, descriptor = evaluate_gait(x, duration=5, visualizer=show_visual,collision_fatal=False,failed_legs=failed_legs)
	return fitness

def load_centroids(filename):
	"""Loads in the CVT voronoi centroids from input archive"""
	points = np.loadtxt(filename)
	return points

def load_map(archive_filename, genome_filename, dim=6):
	"""Loads the generated map

	Args:
		foldername: filename of map archive
		genome_file: filename of the pickled genome file
		dim: number of dimensions of map

	Returns:
		Fitness, descriptor, and genome values for every controller in the map
	"""
	fit = []
	desc = []
	genomes = []
	with open(archive_filename, 'r') as f:
		for line in f:
			tokens = line.strip().split(" ")
			# [0 = fitness, 1 to dim = centroids, (1+dim) to (dim+6) = descriptors]
			fit.append(float(tokens[0]))
			desc.append([float(x) for x in tokens[1+dim:1+dim+6]])
	with open(genome_filename, 'rb') as f:
		genomes = pickle.load(f)
	fit = np.array(fit)
	desc = np.array(desc)
	return fit, desc, genomes

def UCB(mu_map, kappa, sigma_map):
	"""Upper confidence bound aquisition function for the bayesian optimization"""
	GP = []
	for i in range(0, len(mu_map)):
		GP.append(mu_map[i] + kappa*sigma_map[i])
	return np.argmax(GP)


def MBOA(map_genome_path, map_data_path, centroids_filename, failed_legs, max_iter, rho=0.4, print_output=True, show_visual=False):
	"""Implementation of Map-Based Bayseian Optimization Algorithm
	
	Returns:
		num_it: number of iterations taken to find a replacement controller
		best_index: index of that controller in the map
		best_perf: the best performance acheived
		new_map: the new, adjusted map, reflecting the new expected fitness values given the damage/real world experience
	"""

	alpha = 0.90
	kappa = 0.05
	variance_noise_square = 0.001

	dim_x = 6

	num_it = 0
	real_perfs, tested_indexes = [-1],[]
	X, Y = np.empty((0, dim_x)), np.empty((0,1))

	# load map and centroids
	centroids = load_centroids(centroids_filename)
	fits, descs, ctrls = load_map(map_data_path, map_genome_path, centroids.shape[1])

	n_fits, n_descs, n_ctrls = np.array(fits), np.array(descs), np.array(ctrls)

	n_fits_real = copy(np.array(n_fits))
	fits_saved = copy(n_fits)

	started = False

	# used to track performance over iterations
	performance_data = {'iteration': [], 'performance': []}

	while((max(real_perfs) < alpha*max(n_fits_real)) and (num_it < max_iter)):

		if started:
			#define GP kernel
			kernel = GPy.kern.Matern52(dim_x, lengthscale=rho, ARD=False) + GPy.kern.White(dim_x, np.sqrt(variance_noise_square))
			#define Gp which is here the difference between map perf and real perf
			m = GPy.models.GPRegression(X, Y, kernel)
			#predict means and variances for the difference btwn map perf and real perf
			means, variances = m.predict(n_descs)
			#Add the predicted difference to the map found in simulation
			for j in range(0, len(n_fits_real)):
				n_fits_real[j] = means[j] + fits_saved[j]

			#apply acquisition function to get next index to test
			index_to_test = UCB(n_fits_real, kappa, variances)
		else:
			index_to_test = np.argmax(n_fits)
			started = True
			real_perfs = []
		if print_output: print("Expected perf:", n_fits_real[index_to_test])
		#if the behavior to test has already been tested, don't test it again
		if(index_to_test in tested_indexes):
			if print_output: print("Behaviour already tested")
			break
		else:
			ctrl_to_test = n_ctrls[index_to_test]
			tested_indexes.append(index_to_test)
			
			# eval the performance
			real_perf = mboa_gait_evaluation(ctrl_to_test, failed_legs, show_visual)

			if print_output: print("Real perf:", real_perf)
		
		num_it += 1

		# add descriptor and real performance
		X = np.append(X, n_descs[[index_to_test],:], axis=0)
		Y = np.append(Y, (np.array(real_perf)-fits_saved[index_to_test]).reshape((1,1)), axis=0)

		#store
		real_perfs.append(real_perf)

		#store performance data
		performance_data['iteration'].append(num_it)
		performance_data['performance'].append(real_perf)

		# combine updated fitness values and reconstruct map
		new_map = np.loadtxt(map_data_path)
		new_map[:,0] = n_fits_real

		
	o = np.argmax(real_perfs)
	best_index = tested_indexes[o]
	best_perf = real_perfs[o]

	return num_it, best_index, best_perf, new_map, performance_data



if __name__ == "__main__":
	map_size = int(sys.argv[1])
	run_num = sys.argv[2]
	path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "mapElitesOutput", str(map_size), run_num)
	centroid_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "centroids", f"centroids_{map_size}_6.dat")

	num_it, best_index, best_perf, new_map = MBOA(path, centroid_path, [], max_iter=40, print_output=False, show_visual=True)
	print(num_it, best_index, best_perf)


