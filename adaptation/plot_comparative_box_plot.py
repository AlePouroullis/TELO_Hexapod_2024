import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Define RGB to Matplotlib color conversion function
def rgb_to_mpl_color(rgb):
    return tuple([c / 255. for c in rgb])

# Set color for each controller using the provided RGB values
color_dict = {
    'supg': rgb_to_mpl_color((181, 176, 85)),
    'cpg': rgb_to_mpl_color((53, 90, 229)),
    'neat': rgb_to_mpl_color((255, 151, 0)),
    'hyperneat': rgb_to_mpl_color((0, 107, 0)),
    'reference': rgb_to_mpl_color((111, 111, 111))
}

# Mapping of controller codes to descriptive labels
controller_to_label_map = {
    "supg": "ME-10k SUPG",
    "cpg": "ME-40k CPG",
    "neat": "ME-10k NEAT",
    "hyperneat": "ME-5k HyperNEAT",
    "reference": "ME-10k Reference",
}

# Load the dataset
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'box_plot_data.csv'))

# Prepare figure for plotting
plt.figure(figsize=(12, 8))

# Sort controllers by the specified order for plotting
controller_order = ['supg', 'cpg', 'neat', 'hyperneat', 'reference']

# Initialize position for box plots
position = 0
positions = []  # To keep track of the positions for setting x-ticks
labels = []

# Iterate through scenarios and then through controllers in specified order
for scenario in sorted(df['scenario'].unique()):
    for controller in controller_order:
        # Filter data for the current scenario and controller
        controller_label = controller_to_label_map[controller]
        scenario_df = df[(df['scenario'] == scenario) & (df['controller'] == controller)]
        
        if not scenario_df.empty:
            # Extract summary statistics for the box plot
            data_to_plot = [
                scenario_df['minimum'].values[0],
                scenario_df['lower_quartile'].values[0],
                scenario_df['median'].values[0],
                scenario_df['upper_quartile'].values[0],
                scenario_df['maximum'].values[0],
            ]
            # Extract outliers if they exist
            outlier_str = scenario_df["outliers"].values[0].replace("[", "").replace("]", "")
            if outlier_str:  # Check if the outliers string is not empty
                outliers = [float(outlier) for outlier in outlier_str.split(",") if outlier.strip()]
                data_to_plot.extend(outliers)  # Append outliers to the data_to_plot list
            
            positions.append(position)
            plt.boxplot([data_to_plot], positions=[position], widths=0.6,
                        patch_artist=True, boxprops=dict(facecolor=color_dict[controller]),
                        medianprops=dict(color='red', linewidth=2))  # Increase median line thickness
            labels.append(f'S{scenario}')
            position += 1

# Adjust to make sure labels match scenarios without duplication
adjusted_labels = []
adjusted_positions = []
for i, label in enumerate(labels):
    if i % len(controller_order) == 0:
        adjusted_labels.append(label.split('-')[0])  # Keep only the scenario part
    else:
        adjusted_labels.append('')
    adjusted_positions.append(positions[i])

# Set x-ticks and labels with a gap between scenarios
plt.xticks(adjusted_positions, adjusted_labels)
plt.xlim(adjusted_positions[0] - 1, adjusted_positions[-1] + 1)  # Adjust gaps at edges

# make font size for labels larger
plt.ylabel('Task Performance', fontsize=14)
plt.title('Controller Performance', fontsize=16)
plt.xlabel('Scenarios', fontsize=14)

plt.grid(axis='y')

# Create custom legend
legend_elements = [plt.Line2D([0], [0], color=color_dict[ctrl], lw=4, label=controller_to_label_map[ctrl]) for ctrl in controller_order]
legend = plt.legend(handles=legend_elements, title='Controllers', bbox_to_anchor=(1, 1), loc='upper right', prop={'size': 16})  # Increase the font size to 16
plt.setp(legend.get_title(), fontsize=18)  # Increase the title font size to 18

plt.tight_layout()
plt.figure(figsize=(16, 8))  # Increase the width of the plot
plt.show()
