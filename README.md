# SUPG Controller for 18-Motor Hexapod


This repository contains all the code necessary to replicate the adaptation experiments for the SUPG controller. This involves both generating maps for the SUPG controller as well as running the actual adaptation experiments using Map-Based Bayesian Optimization (MBOA).

1. [Setup](#setup)
2. [Project Structure Overview](#project-structure-overview)
    - [Root Directory](#root-directory)
    - [Subdirectories](#subdirectories)
3. [Generating Maps](#generating-maps)
    - [Running MAP-Elites with HyperNEAT](#running-map-elites-with-hyperneat)
    - [Understanding the Parameters](#understanding-the-parameters)
    - [Output](#output)
    - [Customization](#customization)
    - [Existing Maps](#existing-maps)
4. [Adaptation Experiments with MAP-Elites and MBOA](#adaptation-experiments-with-map-elites-and-mboa)
    - [Experiment Setup](#experiment-setup)
    - [Running the Experiments](#running-the-experiments)
    - [Understanding the Output](#understanding-the-output)
    - [Analyzing the Results](#analyzing-the-results)
5. [Visualization](#visualization)
    - [QD-Map](#qd-map)
        - [How to Use the Script](#how-to-use-the-script)
        - [Customization](#customization-1)

## Setup
To get started, you must first create a virtual environment and install all the dependencies. This project uses an anaconda environment, which allows the environment to easily be replicated from an `environment.yml` file.

To create the environment from the `environment.yml` file:
```
conda env create --name <env_name> -f environment.yml
```

Then, to enter the environment:
```
conda activate <env_name>
```

## Project Structure Overview

This project is organized into various directories, each serving a specific purpose. Here's a breakdown of the main components:

### Root Directory

- **DecoupledSUPGController18Motors.py**: A script defining the Decoupled SUPG Controller for 18 motors, focusing on modular control strategies for robotic motion.

- **evolve.py**: Evolves the SUPG using NEAT.

- **mapElites.py**: Implements the MAP-Elites algorithm for the SUPG controller.

- **controller_tools.py**, **environment.yml**, **runSim.py**, **sNeuron.py**, and **viz.py**: Utility scripts and configurations for setting up the simulation environment, visualizing results, and providing additional computational tools.

### Subdirectories

- **GPy**: A comprehensive Python library for Gaussian Process (GP) models, including modules for core GP operations, examples, inference methods, kernels, likelihoods, mappings, and models. Used for Map-Based Bayesian Optimization

- **NEATHex**: Contains configuration files specific to the NEAT algorithm applied to hexapod robots.

- **adaptation**: Focuses on the adaptation mechanisms for the robotic controllers, with scripts like `MBOA.py` for model-based optimization and adaptation.

- **centroids**: Stores data files (.dat) containing centroids for various configurations, used in the MAP-Elites algorithm for defining the feature space.

- **config_DecoupledSUPG**: Configuration files for the Decoupled SUPG controller.

- **evolution_output**: Captures the output from evolutionary runs, including fitness histories, genome data, and speciation visualizations.

- **figures/maps**: Contains visual maps representing the performance or feature space illuminated by the MAP-Elites algorithm.

- **mailer/hexapod**: A directory with an encapsulated module for simulating hexapod robots, including a simulator, controllers, URDF models, and utilities for gait visualization.

- **maps**: Organizes the evolved maps produced by MAP-Elites into subdirectories by the number of niches, each containing data and pickle files representing the elite individuals.

- **neat**: The NEAT (NeuroEvolution of Augmenting Topologies) Python implementation for evolving neural networks, including modules for neural network operations, genetic encoding, and evolution strategies.

- **plots**: Contains scripts for generating plots and visualizations of the evolutionary process, MAP-Elites feature space, and other analysis tools, such as the QD-map.

- **pymap_elites/map_elites**: A Python implementation of the MAP-Elites algorithm, with modules for common operations and Centroidal Voronoi Tessellation (CVT) support.

This project structure facilitates a modular approach to robotic control and evolution, leveraging advanced algorithms like NEAT and MAP-Elites for optimizing and understanding robotic behaviors.

## Generating Maps
### Running MAP-Elites with HyperNEAT

The `mapElites.py` script is designed to produce HyperNEAT maps by evolving solutions over a specified feature space. The script takes two command-line arguments to control the execution:

1. **Map Size**: The size of the map, representing the granularity of the feature space.
2. **Run/Map Number**: A unique identifier for the run, allowing for multiple experiments to be conducted in parallel or sequentially without overwriting results.

#### Pre-requisites

Before running the script, ensure that:

- The NEAT Python library is installed and correctly configured.
- The `pymap_elites` module is available and functioning.
- The necessary configuration files for NEAT are located in the `config_DecoupledSUPG` directory.

#### Step-by-step Instructions

1. **Open a Terminal or Command Prompt**.

2. **Navigate** to the directory containing `mapElites.py`.

3. **Execute the script** by providing the required command-line arguments. For example, to run MAP-Elites with a map size of 10000 and for run number 1, enter:

   ```bash
   python mapElites.py 10000 1
   ```

   Replace `10000` and `1` with the desired map size and run/map number, respectively.

#### Understanding the Parameters:

- **cvt_samples**: Controls the quality of the Centroidal Voronoi Tessellation (CVT). A higher number results in a higher-quality CVT but requires more computational resources.
- **batch_size**: The number of evaluations performed in parallel. Adjust according to your computational resources for optimal performance.
- **random_init**: The proportion of the map filled with random individuals before the evolutionary process begins. Useful for exploring the feature space initially.
- **dump_period**: Determines how frequently the results are written to disk. Adjust based on your preferences for tracking the evolutionary process.
- **min** and **max**: Define the range of parameters for the evolutionary algorithm.

#### Output:

- The script outputs the evolving feature map and genomes to the `mapElitesOutput` directory, organized by map size and run number. This includes log files, archive files, and serialized genomes at different stages of evolution.

- Checkpoints are created automatically, allowing the evolutionary process to be resumed from the last saved state in case of interruption.

#### Customization:

- Modify the parameters in the script or pass different command-line arguments to explore various configurations and optimizations according to your research or project needs.

By following these instructions, you can successfully run the `mapElites.py` script to explore and optimize the feature space of your robotic controllers or other evolutionary algorithms using the MAP-Elites method.

## Existing Maps
The 20 independent runs/maps for each niche size (5k, 10k, 20k, 40k) resulted in a total of 80 maps. This comprises a considerable amount of storage. for this reason, the maps were uploaded to cloud storage. They can be downloaded from this [link](https://uctcloud-my.sharepoint.com/:f:/g/personal/prlale004_myuct_ac_za/Eo9RnLFYD29EttGfDUKgbigB0vBsa3nqei6g-b7k7R_hXA?e=A5eTBv) (OneDrive link).

The drive will contain folders with numbers as names. Install all of them. Once downloaded, place them as is in a directory called `/maps` at the top level of this project.

## Adaptation Experiments with MAP-Elites and MBOA

The adaptation experiments in this project are designed to assess the robustness and adaptability of evolved controllers under various failure scenarios. Utilizing the Multi-Behavioral Optimization Algorithm (MBOA) in conjunction with the MAP-Elites algorithm, these experiments simulate different types of leg failures in a hexapod robot to evaluate the controller's performance in adapting to new conditions.

#### Experiment Setup

- **Scenarios**: Defined failure conditions ranging from no failure to specific legs being non-functional. These scenarios simulate real-world issues that could arise during the robot's operation.
  
- **Niches**: Variations in the feature space size (5k, 10k, 20k, 40k), representing different levels of granularity in the evolutionary search.

- **Map Count**: The number of maps tested against each scenario, providing a comprehensive evaluation across the feature space.

- **Visualization**: Optional visualization of the simulation can be enabled to observe the robot's behavior during the adaptation process.

#### Running the Experiments

1. **Dependencies**: Ensure all dependencies are installed, including the necessary Python libraries and the MBOA module.

2. **Data Preparation**: The scripts automatically load centroids from the specified directories and organize MAP-Elites maps for each niche.

3. **Command Line Arguments**:
   - `--show_visual`: Enables the simulation's graphical visualization (default is off).
   - `--niche`: Specifies a single niche to run experiments for. If not specified, experiments will run for all niches.

4. **Execution**:
   Navigate to the directory containing the adaptation experiment scripts and run:
   ```bash
   python -m adaptation.adaptation_experiment --show_visual --niche 10000
   ```
   Replace `10000` with your chosen niche size or omit the `--niche` argument to run for all niches.

#### Understanding the Output

- **Checkpointing**: The scripts automatically save progress checkpoints. This feature allows experiments to resume from the last saved point in case of interruption.

- **Results**: Performance data is saved to CSV files, one per niche. These files are intended for generating comparative box plots to analyze the adaptation performance across different scenarios and niches.

- **Visualization**: If enabled, the simulation will display the robot attempting to walk, highlighting the effectiveness of the adaptation in real-time.

#### Analyzing the Results

- The saved CSV files contain detailed performance metrics for each run, scenario, and map. These metrics can be used to generate visualizations or statistical analyses to understand the adaptation strategies' effectiveness.

- Comparative box plots generated from the CSV data provide insights into the robustness of the evolved controllers under various failure scenarios, enabling researchers to identify areas of strength and potential improvement.

## Visualization

### QD-Map
The script provided is designed to visualize the maps produced by the MAP-Elites algorithm. These maps represent the diversity of solutions (or "elites") across different feature dimensions or "niches". The script uses Voronoi tessellation to visualize each niche and the performance of the elite solutions within those niches. Here’s how the script operates and how to use it for visualizing maps:

#### Script Operation:

1. **Command Line Arguments**: The script accepts command-line arguments to specify the controller type ("CPG" or "REF"), the number of niches (5k, 10k, 20k, 40k), and which map number to plot. These parameters determine the data source for the visualization.

2. **Data Loading**: It loads the centroids data (from the "centroids" directory) and the MAP-Elites map data (from the "maps" directory), which include fitness scores, behavior descriptors, and other relevant information for each elite solution.

3. **Voronoi Tessellation**: The script uses Voronoi tessellation to divide the space based on the centroids of the niches, creating a visual representation of the division of the solution space.

4. **Plotting**: For each niche, the script plots a polygon with a color indicating the fitness of the elite solution in that niche. This creates a comprehensive view of the performance landscape, highlighting areas of high fitness and diversity.

5. **Color Scale**: The script uses a color map (by default, "viridis") to represent the fitness scores, with options to manually adjust the normalization of the color scale to emphasize differences in fitness.

6. **Output**: The visualization is plotted using matplotlib and saved as both a PDF and a PNG file in the "figures/maps" directory. An optional step shows the plot interactively.

#### How to Use the Script:

1. **Prepare Your Data**: Ensure your centroids and MAP-Elites map data are correctly placed in the "centroids" and "maps" directories, respectively.

2. **Run the Script**: Use the command line to run the script with appropriate arguments. For example, to plot a map for a controller with 20k niches and map number 1:
   ```sh
   python plot_map.py -n 20 -m 1
   ```
   Replace `20` with your desired number of niches and `1` with the specific map number you wish to visualize.

3. **View and Analyze**: Open the generated PDF or PNG file in the "figures/maps" directory to view the visualized map. This visualization helps in understanding the distribution of solutions across the feature space and identifying regions of high performance.

4. **Customization**: Modify the script if necessary to adjust the color scale, plotting parameters, or to integrate additional data into the visualization.

This visualization tool is vital for analyzing the outcomes of the MAP-Elites algorithm, offering insights into the diversity and quality of solutions discovered during the evolutionary process.


