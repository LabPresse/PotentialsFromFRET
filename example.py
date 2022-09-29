
import h5py
import numpy as np
import matplotlib.pyplot as plt
from fret_analyzer import FRETAnalyzer
from fret_analyzer_HMM import FRETAnalyzerHMM

# Set parameters
file = 'exampledata.h5'
savename = 'exampleresults.h5'
num_data = 1000
num_iter = 10

# Load data
h5 = h5py.File(file, 'r')
data = h5['data'][()]
parameters = {key: h5[key][()] for key in h5.keys() if key != 'data'}
h5.close()

# Crop data
data = data[:, :num_data]
if 'traj_mask' in parameters:
    parameters['traj_mask'] = parameters['traj_mask'][:num_data]

# Run analysis
MAP, history = FRETAnalyzer.learn_potential(
    data, parameters=parameters, num_iter=num_iter, saveas=savename,
)

# Run analysis HMM
MAP_HMM, history_HMM = FRETAnalyzerHMM.learn_potential(
    data, parameters=parameters, num_iter=num_iter, saveas=savename+'_HMM',
)

# Plot results
FRETAnalyzer.plot_variables(data, MAP)
FRETAnalyzerHMM.plot_variables(data, MAP_HMM)


