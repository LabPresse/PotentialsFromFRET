
import h5py
import numpy as np
import matplotlib.pyplot as plt
from fret_analyzer import FRETAnalyzer
from fret_analyzer_HMM import FRETAnalyzerHMM

# Set parameters
file = 'exampledata.h5'
num_data = 1000
parameters = {
    'dt': 1e6,    # Time step in nanoseconds
    'kT': 4.114,  # Temperature in kT
    'R0': 5,      # Characteristic FRET distance in nanometers
}

# Load data
h5 = h5py.File(file, 'r')
data = h5['data'][()]
h5.close()
data = data[:, :num_data]

# Run analysis
MAP = FRETAnalyzer.learn_potential(data, parameters=parameters)
FRETAnalyzer.plot_variables(data, MAP)

# Run analysis HMM
MAP_HMM = FRETAnalyzerHMM.learn_potential(data, parameters=parameters)
FRETAnalyzerHMM.plot_variables(data, MAP_HMM)
energies = FRETAnalyzerHMM.calculate_energy(MAP_HMM)

print("Completed")
