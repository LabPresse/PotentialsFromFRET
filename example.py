
# Import libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
from fret_analyzer import FRETAnalyzer
from fret_analyzer_HMM import FRETAnalyzerHMM

# Set parameters
print("Setting parameters")
num_data = 1000  # Number of data points
parameters = {
    'dt': 1e6,    # Time step in nanoseconds
    'kT': 4.114,  # Temperature in kT
    'R0': 5,      # Characteristic FRET distance in nanometers
}

# Load data
print("Loading data")
file = 'data/exampledata.h5'
h5 = h5py.File(file, 'r')
data = h5['data'][()]
h5.close()
data = data[:, :num_data]

# Run analysis using Skipper-FRET
print("Running analysis using Skipper-FRET")
MAP = FRETAnalyzer.learn_potential(
    data, parameters=parameters, num_iter=100,
)

# Plot variables
print("Plotting variables")
FRETAnalyzer.plot_variables(data, MAP)


# # Uncomment this code block to run the HMM analysis
# # Run analysis using HMM
# print("Running analysis using HMM")
# MAP_HMM = FRETAnalyzerHMM.learn_potential(
#     data, parameters=parameters, num_iter=100,
# )
# energies = FRETAnalyzerHMM.calculate_energy(MAP_HMM)  # Get energies
# print("Energies:", energies)
# FRETAnalyzerHMM.plot_variables(data, MAP_HMM)  # Plot variables

# Done
print("Done")
