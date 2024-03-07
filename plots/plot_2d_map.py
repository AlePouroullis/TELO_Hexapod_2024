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
"""Plots a single MAP-Elites map and saves it to figures folder

Takes in the following command line arguments:
    Flag    Flag (long)             Description
    _____   _____________________   ____________________________________________
    -n      --niches                : Number of niches (20 or 40)k
    -m      --map                   : Which map number to plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.spatial import Voronoi, voronoi_plot_2d
import sys
from matplotlib.ticker import FuncFormatter
from sklearn.neighbors import KDTree
import matplotlib.cm as cm
import os
import argparse

my_cmap = cm.viridis # viridis jet

def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite voronoi regions in a 2D diagram to finite regions.
    
    Source: https://stackoverflow.com/questions/20515554/colorize-voronoi-diagram/20678647#20678647

    Args:
        vor : (Voronoi) Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

    Returns:
        regions : (list of tuples) Indices of vertices in each revised Voronoi regions.
        vertices : (list of tuples) Coordinates for revised Voronoi vertices. Same as coordinates of input vertices, with 'points at infinity' appended to the end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def load_data(filename, dim,dim_x):
    """Reads in map/archive data

    Args:
        filename: map file
        dim: number of dimensions (usually 6)
        dim_x: number of parameters used for MAP-Elites

    Returns:
        The fitness, descriptor and x value co-ordinates (not parameters!) for every controller in the map
        NOTE: x is not the parameters of the controller
    """
    print("Loading ",filename)
    data = np.loadtxt(filename)
    fit = data[:, 0:1]
    desc = data[:,1: dim+1]
    x = data[:,dim+1:dim+1+dim_x]

    return fit, desc, x

def load_centroids(filename):
    """Reads in centroids from file
    
    Args:
        filename: file of centroids

    Returns:
        Points for centroids
    """
    points = np.loadtxt(filename)
    return points

def plot_cvt(ax, centroids, fit, desc, min_fit, max_fit):
    """Compute Voronoi tesselation

    Args:
        ax: axis to use for plot
        centroids: centroid points
        fit: fitness values
        desc: descriptor values
        min_fit: minimun fitness value (affects colouring)
        max_fit: maximum fitness value (affects colouring)
    
    Returns:
        Scatter plot object. 
        NOTE:The scatter is already plotted by this function
    """
    print("Voronoi...")
    vor = Voronoi(centroids[:,0:2])
    regions, vertices = voronoi_finite_polygons_2d(vor)
    print("fit:", min_fit, max_fit)
    norm = matplotlib.colors.Normalize(vmin=min_fit, vmax=max_fit)
    print("KD-Tree...")
    kdt = KDTree(centroids, leaf_size = 30, metric = 'euclidean')

    print("plotting contour...")
    #ax.scatter(centroids[:, 0], centroids[:,1], c=fit)
    # contours
    for i, region in enumerate(regions):
        polygon = vertices[region]
        ax.fill(*zip(*polygon), alpha=0.05, edgecolor='black', facecolor='white', lw=1)

    print("plotting data...")
    k = 0
    for i in range(0, len(desc)):
        q = kdt.query([desc[i]], k = 1)
        index = q[1][0][0]
        region = regions[index]
        polygon = vertices[region]
        color_map = my_cmap(norm(fit[i]))
        ax.fill(*zip(*polygon), alpha=0.9, color=color_map[0])
        k += 1
        if k % 100 == 0:
            print(k, end=" ", flush=True)
    fit_reshaped = fit.reshape((len(fit),))
    sc = ax.scatter(desc[:,0], desc[:,1], c=fit_reshaped, cmap=my_cmap, s=10, zorder=0)
    return sc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plots a MAP-Elites map and saves it to figures folder.')
    parser.add_argument('-n','--niches', required=True, type=int,  default=20, help='Number of niches (20 or 40)k')
    parser.add_argument('-mi','--map-index', required=True, type=int,  default=1, help='Map index to plot')
    args = parser.parse_args() 

    if args.niches not in [5,10,20,40]:
        raise Exception(f"Only niches sizes of 5, 10, 20 and 40 are allowed (5k, 10k, 20k and 40k).\nYou put {args.niches}")

    # load in centroids
    centroids = load_centroids(os.path.join(os.path.dirname(__file__), "..",  "centroids", f"centroids_{args.niches}000_6.dat")) 
    fit, beh, x = load_data(os.path.join(os.path.dirname(__file__), "..", "maps", f"{args.niches}000", f"map_{args.map_index}.dat"), centroids.shape[1], 24)

    index = np.argmax(fit)


    min_fit = min(fit)
    max_fit = max(fit)
    # print map details 
    print(f"""
Fitness max:        {max(fit)}
Average fit:        {fit.sum() / fit.shape[0]}
Associated desc:    {beh[index]}
Associated ctrl:    
{x[index]}

Index:              {index}
total len:          {len(fit)}
Min = {min_fit} Max={max_fit}
"""
    )

    # Plot
    fig, axes = plt.subplots(1, 1, figsize=(10, 10), facecolor='white', edgecolor='white')
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    sc = plot_cvt(axes, centroids, fit, beh, min_fit, max_fit)

    # Adjusting the color scale normalization directly in plot_cvt function
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)  # Set min and max values for the color scale
    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=my_cmap, norm=norm)
    sm._A = []  # This is needed for ScalarMappable
    colorbar = plt.colorbar(sm, ax=axes)  # Associate with the axes
    colorbar.ax.get_yaxis().labelpad = 50
    colorbar.ax.set_ylabel('Controller performance', rotation=270, fontsize=20) # ($ms^{-1}$)'
    # colorbar.ax.set_ylabel('Gait speed ($m/s$)', rotation=270, fontsize=20) # ($ms^{-1}$)'
    axes.set_title(f"SUPG Controller map - {args.niches}k niches", fontsize=20, pad=30)

    # Save figures to figures folder
    base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures","maps")
    # create folder if it doesn't exist
    if not os.path.exists(base_path):
        os.makedirs(base_path)

    path = os.path.join(base_path,f"SUPG Controller map - {args.niches}k niches.pdf")
    path_png = os.path.join(base_path, f"SUPG Controller map - {args.niches}k niches.png")
    fig.savefig(path)
    fig.savefig(path_png)
    
    plt.show()