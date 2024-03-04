"""Used for plotting adaptation graphs and for running Welche's t-tests.
"""
import sys
import os
sys.path.append(os.path.abspath("."))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats

X_WIDTH_GAP=10

def plot_adaption_graph(perform_performance_test=True, perform_t_test=True, save_figure=True, filename_suffix=''):
    """Plots adaptation graph and runs t-tests.
    """

    def set_box_color(bp, color):
        """Sets colours of box + whisker plot things"""
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        # plt.setp(bp['medians'], color='black')

    # cpg
    sim_data_5k = []
    sim_data_10k = []
    sim_data_20k = []
    sim_data_40k = []
    sim_data_80k = []
    # ref
    sim_data_ref_5k = []
    sim_data_ref_10k = []
    sim_data_ref_20k = []
    sim_data_ref_40k = []
    sim_data_ref_80k = []
    # neat + tripod
    # sim_data_neat = []
    # sim_data_ref_tripod = []

    # read in data
    for scenario in range(5):
        # cpg
        data_5k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",          "experiments", "output", "normalized", "CPG", "5k", f"perfs_{scenario}.dat")).flatten())
        data_10k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",          "experiments", "output", "normalized", "CPG", "10k", f"perfs_{scenario}.dat")).flatten())
        data_20k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",          "experiments", "output", "normalized", "CPG", "20k", f"perfs_{scenario}.dat")).flatten())
        data_40k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",          "experiments", "output", "normalized", "CPG", "40k", f"perfs_{scenario}.dat")).flatten())
        data_80k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",          "experiments", "output", "normalized", "CPG", "80k", f"perfs_{scenario}.dat")).flatten())
        # ref
        data_ref_5k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",      "experiments", "output", "normalized", "REF", "5k", f"perfs_{scenario}.dat")).flatten())
        data_ref_10k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",      "experiments", "output", "normalized", "REF", "10k", f"perfs_{scenario}.dat")).flatten())
        data_ref_20k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",      "experiments", "output", "normalized", "REF", "20k", f"perfs_{scenario}.dat")).flatten())
        data_ref_40k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",      "experiments", "output", "normalized", "REF", "40k", f"perfs_{scenario}.dat")).flatten())
        data_ref_80k = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",      "experiments", "output", "normalized", "REF", "80k", f"perfs_{scenario}.dat")).flatten())
        # neat + tripod
        # data_neat = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",         "experiments", "output", "normalized", "CPG", "neat-no-adpatation", f"NEAT-CPG-{scenario}.dat")).flatten()/5)
        # data_ref_tripod = list(np.loadtxt(os.path.join(os.path.dirname(__file__), "..", "..",   "experiments", "output", "normalized", "REF", "tripod-no-adapatation", f"tripod_{scenario}.dat")).flatten()/5)
        # add to array
        # cpg
        sim_data_5k.append(data_5k)
        sim_data_10k.append(data_10k)
        sim_data_20k.append(data_20k)
        sim_data_40k.append(data_40k)
        sim_data_80k.append(data_80k)
        # ref
        sim_data_ref_5k.append(data_ref_5k)
        sim_data_ref_10k.append(data_ref_10k)
        sim_data_ref_20k.append(data_ref_20k)
        sim_data_ref_40k.append(data_ref_40k)
        sim_data_ref_80k.append(data_ref_80k)
        # neat + tripod
        # sim_data_neat.append(data_neat)
        # sim_data_ref_tripod.append(data_ref_tripod)

    #################### BEGIN PERFORMANCE DROP TESTS ####################
    if(perform_performance_test):
        # performance stats. Not required for graph. Should have a reference perhaps
        print("performance stats - % drops")
        sim_data_20k_summary        = [np.average(subarray) for subarray in sim_data_20k]
        sim_data_40k_summary        = [np.average(subarray) for subarray in sim_data_40k]
        sim_data_neat_summary       = [np.average(subarray) for subarray in sim_data_neat]
        sim_data_ref_20k_summary    = [np.average(subarray) for subarray in sim_data_ref_20k]
        sim_data_ref_40k_summary    = [np.average(subarray) for subarray in sim_data_ref_40k]
        sim_data_ref_tripod_summmary = [np.average(subarray) for subarray in sim_data_ref_tripod]
        
        print(f'SX:\t\tNEAT\t20k\t40k\t\t20kR\t40kr\tref')
        for scenario in range(0, 5):
            perf_drop_20k_summary           = 1 - (sim_data_20k_summary[scenario] / sim_data_20k_summary[0])
            perf_drop_40k_summary           = 1 - (sim_data_40k_summary[scenario] / sim_data_40k_summary[0])
            perf_drop_neat_summary          = 1 - (sim_data_neat_summary[scenario] / sim_data_neat_summary[0])
            perf_drop_ref_20k_summary       = 1 - (sim_data_ref_20k_summary[scenario] / sim_data_ref_20k_summary[0])
            perf_drop_ref_40k_summary       = 1 - (sim_data_ref_40k_summary[scenario] / sim_data_ref_40k_summary[0])
            perf_drop_ref_tripod_summmary   = 1 - (sim_data_ref_tripod_summmary[scenario] / sim_data_ref_tripod_summmary[0])
            print(f'S{scenario}:\t\t{perf_drop_20k_summary:.3f}\t{perf_drop_40k_summary:.3f}\t{perf_drop_neat_summary:.3f}\t\t{perf_drop_ref_20k_summary:.3f}\t{perf_drop_ref_40k_summary:.3f}\t{perf_drop_ref_tripod_summmary:.3f}')
    #################### END PERFORMANCE DROP TESTS ####################

    #################### BEGIN T-TESTS ####################
    if(perform_t_test):
        print("Performance stats - p values")
        print(f'SX:\t\t CPG Adapt v. Ref adapt \t CPG Adapt v. CPG No adapt \t Ref adapt v. Ref no adapt')
        for scenario in range(1, 5):
            t_statistic, p_value            = stats.ttest_ind(sim_data_20k[scenario] + sim_data_40k[scenario]       , sim_data_ref_20k[scenario]+sim_data_ref_40k[scenario] , equal_var=False)
            t_statistic_neat, p_value_neat  = stats.ttest_ind(sim_data_20k[scenario]       , sim_data_neat[scenario]                                         , equal_var=False)
            t_statistic, p_value_ref        = stats.ttest_ind(sim_data_ref_20k[scenario]+sim_data_ref_40k[scenario] , sim_data_ref_tripod[scenario]                                  , equal_var=False)
            # t_statistic, p_value = stats.ttest_ind(sim_data_ref_20k[scenario] + sim_data_ref_40k[scenario], sim_data_ref_20k[scenario]+sim_data_ref_40k[scenario], equal_var=False)
            print(f'S{scenario}:\t\t {p_value:.3f} \t\t\t\t {p_value_neat:.3f} \t\t\t\t {p_value_ref:.8f}')

        # map size stats. Also not really needed for graphs
        print('\n\nmap size statistics - p values')
        print(f'SX:\t\t CPG 20k vs 40k \t Ref 20k vs 40k')
        for scenario in range(5):
            t_statistic, p_value = stats.ttest_ind(sim_data_20k[scenario], sim_data_40k[scenario])
            t_statistic, p_value_ref = stats.ttest_ind(sim_data_ref_20k[scenario], sim_data_ref_40k[scenario])
            print(f'S{scenario}:\t\t {p_value:.3f} \t\t\t {p_value_ref:.3f}')
    #################### END T-TESTS ####################

    ticks = ['None (S0)', 'S1', 'S2', 'S3', 'S4']
    # https://matplotlib.org/stable/tutorials/colors/colormaps.html
    # https://matplotlib.org/stable/tutorials/colors/colors.html
    # cpg
    color_5k = 'plum'
    color_10k = 'violet'
    color_20k = 'blue'
    color_40k = 'indigo'
    color_80k = 'violet'
    # ref
    color_ref_5k = 'crimson'
    color_ref_10k = 'orange'
    color_ref_20k = 'salmon'
    color_ref_40k = 'red'
    color_ref_80k = 'brown'

    fig, ax = plt.subplots()
    # fig.set_size_inches(w=3.3, h=2.0)
    fig.set_size_inches(w=30, h=20) # TODO: play around
    ax.set_axisbelow(True)
    ax.yaxis.grid(True)
    ax.set_title('Adapted Fitness', fontsize = 40)
    ax.set_xlabel('Failure scenario', fontsize = 40)
    ax.set_ylabel('Fitness', fontsize = 40)  #$ms^{-1}$
    # ax.set_ylabel('Fitness ($m/s$)', fontsize = 40)  #$ms^{-1}$

    # PROPS for decorating plot objects
    flierprops = dict(marker='o', markersize=5, linestyle='none', markeredgecolor='darkgray')
    # meanline = dict(linestyle='-', color='white')
    # meanpoint = dict(marker='D', markeredgecolor='black', markerfacecolor='red')
    medianprops = dict(linestyle='-', color='silver')
    medianprops_ref = dict(linestyle='-', color='black')
    positions = np.array(range(len(sim_data_40k)))       ## Maybe change to 4 # TODO: play around
    positions = positions * X_WIDTH_GAP      ## Maybe change to 4 # TODO: play around

    # Plot actual Data
    # TODO: play around
    # cpg
    box_width = 0.5 # TODO: play around
    bp5k = plt.boxplot(sim_data_5k, positions=positions   - 3.75,        showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops, patch_artist=True, )
    bp10k = plt.boxplot(sim_data_10k, positions=positions - 3.20,        showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops, patch_artist=True, )
    bp20k = plt.boxplot(sim_data_20k, positions=positions - 2.65,        showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops, patch_artist=True, )
    bp40k = plt.boxplot(sim_data_40k, positions=positions - 2.10,        showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops, patch_artist=True, )
    bp80k = plt.boxplot(sim_data_80k, positions=positions - 1.55,        showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops, patch_artist=True, )
    # ref
    bp5k_ref = plt.boxplot(sim_data_ref_5k, positions=positions   - 1.00,   showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops_ref, patch_artist=True, )
    bp10k_ref = plt.boxplot(sim_data_ref_10k, positions=positions - 0.45,   showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops_ref, patch_artist=True, )
    bp20k_ref = plt.boxplot(sim_data_ref_20k, positions=positions + 0.10,   showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops_ref, patch_artist=True, )
    bp40k_ref = plt.boxplot(sim_data_ref_40k, positions=positions + 0.65,   showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops_ref, patch_artist=True, )
    bp80k_ref = plt.boxplot(sim_data_ref_80k, positions=positions + 1.20,   showfliers=False,showmeans=False,flierprops=flierprops, zorder=-1, medianprops=medianprops_ref, patch_artist=True, )

    # sim_data_neat_summmary = [np.average(subarray) for subarray in sim_data_neat]
    # plt.scatter(positions-0.5, sim_data_neat_summmary, marker='*', color='yellowgreen',zorder=1, s=100)   # Might use with ref gaits of some sort # TODO: play around
    # sim_data_ref_tripod  = [np.average(subarray) for subarray in sim_data_ref_tripod]
    # plt.scatter(positions+1, sim_data_ref_tripod, marker='*', color='teal',zorder=1, s=100)   # Might use with ref gaits of some sort # TODO: play around

    # Colourize
    # cpg
    set_box_color(bp5k, color_5k)
    set_box_color(bp10k, color_10k)
    set_box_color(bp20k, color_20k)
    set_box_color(bp40k, color_40k)
    set_box_color(bp80k, color_80k)
    # ref
    set_box_color(bp5k_ref, color_ref_5k)
    set_box_color(bp10k_ref, color_ref_10k)
    set_box_color(bp20k_ref, color_ref_20k)
    set_box_color(bp40k_ref, color_ref_40k)
    set_box_color(bp80k_ref, color_ref_80k)

    custom_lines = [
        mpatches.Patch(color=color_5k), 
        mpatches.Patch(color=color_10k), 
        mpatches.Patch(color=color_20k), 
        mpatches.Patch(color=color_40k), 
        mpatches.Patch(color=color_80k), 
        # mlines.Line2D([], [], color='yellowgreen', marker='*', linestyle='None',markersize=10), 
        mpatches.Patch(color=color_ref_5k), 
        mpatches.Patch(color=color_ref_10k), 
        mpatches.Patch(color=color_ref_20k), 
        mpatches.Patch(color=color_ref_40k),
        mpatches.Patch(color=color_ref_80k),
        # mlines.Line2D([], [], color='teal', marker='*', linestyle='None',markersize=10), 
    ]
    plt.legend(custom_lines, [
        'CPG 5k',
        'CPG 10k',
        'CPG 20k',
        'CPG 40k',
        'CPG 80k',
        # 'CPG - No adapatation (avg)',
        'Reference 5k',
        'Reference 10k',
        'Reference 20k',
        'Reference 40k',
        'Reference 80k',
        # 'Reference - No adapatation (avg)',
    ], prop={'size': 20})

    # Set ticks
    plt.xticks(range(0, len(ticks) * X_WIDTH_GAP, X_WIDTH_GAP), ticks)  # TODO: play around
    ax.tick_params(axis='both', labelsize=25)
    plt.xlim(-4, len(ticks) * X_WIDTH_GAP) # TODO: play around
    plt.ylim(0,1)
    #plt.tight_layout(pad=0.1)
    
    # Save figure
    if (save_figure):
        plt.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "performance_adaption", "Adaption Fitness {filename_suffix}.pdf".format(filename_suffix=filename_suffix)))
        plt.savefig(os.path.join(os.path.dirname(__file__), "..", "..", "figures", "performance_adaption", "Adaption Fitness {filename_suffix}.png".format(filename_suffix=filename_suffix)))
    
    # Display figure
    plt.show()


plot_adaption_graph(perform_performance_test=False, perform_t_test=False, filename_suffix="normalized")