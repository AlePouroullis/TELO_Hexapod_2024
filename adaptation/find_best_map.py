import csv
import os
from typing import Dict, List

def read_csv(filepath: str) -> List[Dict[str, float]]:
    """
    Read CSV file and return a list of dictionaries with performance data.
    """
    data = []
    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append({
                "niche": int(row["niche"]),
                "map_id": int(row["map_id"]),
                "scenario": int(row["scenario"]),
                "best_performance": float(row["best_performance"])
            })
    return data

def calculate_average_performance(data: List[Dict[str, float]]) -> float:
    """
    Calculate the average performance from the performance data.
    """
    total_performance = sum(item["best_performance"] for item in data)
    return total_performance / len(data)

def find_best_performing_niche(niches: List[int], base_dir: str, highest_average_performance) -> int:
    """
    Find the niche with the highest average performance.
    """
    best_niche = None
    
    for niche in niches:
        filepath = os.path.join(base_dir, f"boxplot_data_niche_{niche}.csv")
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} does not exist. Skipping.")
            continue

        data = read_csv(filepath)
        average_performance = calculate_average_performance(data)
        print(f"Niche {niche} has an average performance of {average_performance:.2f}")


        if average_performance > highest_average_performance:
            highest_average_performance = average_performance
            best_niche = niche

    return best_niche, highest_average_performance

def find_best_performing_index(niche):
    """
    Find the best performing index for a given niche.
    """
    filepath = os.path.join(base_dir, f"boxplot_data_niche_{niche}.csv")
    if not os.path.exists(filepath):
        print(f"Warning: File {filepath} does not exist. Skipping.")
        return None
    
    data = read_csv(filepath)
    best_performance = -float('inf')
    best_index = None
    for item in data:
        if item["best_performance"] > best_performance:
            best_performance = item["best_performance"]
            best_index = item["map_id"]
    return best_index



if __name__ == "__main__":
    niches = [5000, 10000, 20000, 40000]
    base_dir = "./adaptation"  # Adjust this to the directory where your CSV files are stored
    highest_average_performance = -float('inf')
    best_niche, highest_average_performance = find_best_performing_niche(niches, base_dir, highest_average_performance)
    
    if best_niche is not None:
        best_index = find_best_performing_index(best_niche)

        print(f"The best performing niche on average is: {best_niche} with an average performance of {highest_average_performance:.2f}. The best performing index is: {best_index}.")
    else:
        print("No valid data found for comparison.")





