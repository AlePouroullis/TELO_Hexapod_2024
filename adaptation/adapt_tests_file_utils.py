import csv
import os

def save_results_to_csv(data, niches):
   '''saves the results to a csv file'''
   path = f'adapt_output/results_{niches}.csv'
   print(f"Saving results to {path}")
   # check path exists; if not, create it
   if not os.path.exists(os.path.dirname(path)):
      os.makedirs(os.path.dirname(path))

  # Check if file already exists to decide whether to write headers
   write_header = not os.path.exists(path)

   with open(path, 'a') as csvfile:
      fieldnames = ['run_num', 'leg_damage_scenario_index', 'num_iterations', 'best_index', 'best_performance']
      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
      
      if write_header: 
         writer.writeheader()
      for row in data:
         writer.writerow(row)

def save_average_performance_to_csv(average_performance_data, niches, output_dir='adapt_output'):
    '''
    Saves the average performance data to a CSV file.

    :param average_performance_data: A dictionary with keys as scenario indices and values as average performance arrays.
    :param niches: Number of niches in the map.
    :param output_dir: The directory where the output CSV files will be saved.
    '''
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for scenario_index, performance_array in average_performance_data.items():
        file_path = os.path.join(output_dir, f'average_performance_niches_{niches}_scenario_{scenario_index }.csv')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(['iteration', 'average_performance'])

            # Write data
            for iteration, performance in enumerate(performance_array):
                writer.writerow([iteration + 1, performance])

        print(f"Average performance for scenario {scenario_index + 1} saved to {file_path}")
