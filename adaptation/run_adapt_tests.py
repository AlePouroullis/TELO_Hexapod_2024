import os
import numpy as np
from .MBOA import MBOA
import csv
import json

# Parameters
map_count = 1
niches = 5  # k
SHOW_VISUAL = False

# Failure scenarios
scenarios = [
    [[]],
    [[1], [2], [3], [4], [5], [6]],
    [[1, 4], [2, 5], [3, 6]],
    [[1, 3], [2, 4], [3, 5], [4, 6], [5, 1], [6, 2]],
    [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 1]]
]

niches = [5000, 10000, 20000, 40000]

# Assuming other parts of the script remain the same

def load_checkpoint(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    return None

def save_checkpoint(data, filepath):
    with open(filepath, 'w') as file:
        json.dump(data, file)

def get_centroids_path(niches):
	return os.path.join(os.path.dirname(os.path.dirname(__file__)), "centroids", f"centroids_{niches}_6.dat")

def get_map_path(niches, map_id):
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "maps", str(niches))
    map_genome_path = os.path.join(base_dir, f"map_{map_id}_genome.pkl")
    map_data_path = os.path.join(base_dir, f"map_{map_id}.dat")
    return map_genome_path, map_data_path

def run_experiment_for_niche(niche, map_ids, max_iter=40, show_visual=False):
    checkpoint_path = os.path.join(os.path.dirname(__file__), f"checkpoint_niche_{niche}.json")
    checkpoint = load_checkpoint(checkpoint_path)
    
    if checkpoint:
        start_scenario_index = checkpoint['scenario_index']
        start_combination_index = checkpoint.get('combination_index', 0)  # Get combination index if available
        start_map_index = checkpoint['map_index'] + 1 if start_combination_index == 0 else checkpoint['map_index']
        results = checkpoint['results']
        print(f"Resuming from Scenario {start_scenario_index + 1}, Combination {start_combination_index + 1}, Map ID {map_ids[start_map_index]}")
    else:
        start_scenario_index = 0
        start_combination_index = 0  # Start from the first combination
        start_map_index = 0
        results = []

    # Assuming the number of scenarios and combinations are fixed, use the lengths of these structures
    total_scenarios = len(scenarios)
    total_combinations = max(len(scenario) for scenario in scenarios)  # Assumes max combinations in any scenario
    total_maps = len(map_ids)
    
    # Adjust the condition to check for completion
    # Note: Ensure indices and counts are correctly aligned with how you're iterating
    if start_scenario_index + 1 >= total_scenarios and start_combination_index + 1 >= total_combinations and start_map_index + 1 >= total_maps:
        print("All scenarios, combinations, and maps have been completed. Exiting.")
        return

    centroid_path = get_centroids_path(niche)

    for scenario_index, scenario in enumerate(scenarios[start_scenario_index:], start=start_scenario_index):
        for combination_index, combination in enumerate(scenario[start_combination_index:], start=start_combination_index):
            for map_id_index, map_id in enumerate(map_ids[start_map_index:], start=start_map_index):
                map_genome_path, map_data_path = get_map_path(niche, map_id)
                print(map_genome_path)
                print(f"Niche: {round(niche/1000)}k, Map ID: {map_id}, Scenario: {scenario_index + 1}, Combination: {combination_index + 1}")

                _, _, best_perf, _, performance_data = MBOA(map_genome_path, map_data_path, centroid_path, combination, max_iter=max_iter, print_output=False, show_visual=show_visual)
                
                results.append({
                    "niche": niche,
                    "map_id": map_id,
                    "scenario": scenario_index + 1,
                    "combination": combination_index + 1,
                    "best_performance": best_perf
                })
                
                # Save checkpoint after each map, scenario, and combination
                checkpoint_data = {
                    "scenario_index": scenario_index,
                    "combination_index": combination_index,  # Save the combination index
                    "map_index": map_id_index,
                    "results": results
                }
                save_checkpoint(checkpoint_data, checkpoint_path)

            # Reset start_map_index after completing a combination for a map
            start_map_index = 0
        # Reset start_combination_index after completing all combinations for a scenario
        start_combination_index = 0

    # Save results to CSV for box plot
    save_results_for_boxplot(results, niche)
    # Optionally, delete the checkpoint file after successful completion
    os.remove(checkpoint_path)

def save_results_for_boxplot(data, niche):
    filename = f"boxplot_data_niche_{niche}.csv"
    filepath = os.path.join(os.path.dirname(__file__), filename)
    with open(filepath, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["niche", "map_id", "scenario", "best_performance", "combination"])
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    print(f"Saved box plot data to {filepath}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_visual", action="store_true", help="Show simulator visualization")
    parser.add_argument("--niche", type=int, default=None, help="Specific niche to run experiments for. If not specified, will run for all niches (5k, 10k, 20k, 40k)")
    parser.add_argument("--map_ids", type=int, nargs="+", default=None, help="Specific map IDs to run experiments for. If not specified, will run for all maps (1-20)")
    args = parser.parse_args()

    SHOW_VISUAL = args.show_visual
    selected_niche = args.niche
    # list from 1 to 20
    map_ids = args.map_ids if args.map_ids else list(range(1, 21))

    # Logic to iterate over maps for each niche
    if selected_niche:
        run_experiment_for_niche(selected_niche, map_ids,  show_visual=SHOW_VISUAL)
    else:
        for niche in niches:
            run_experiment_for_niche(niche, map_ids, show_visual=SHOW_VISUAL)

	
