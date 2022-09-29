# PotentialsFromFRET
Code for inferring continuous potentials from FRET data

# Using the code

The code is organized into a class called "FRETAnalyzer". FRETAnalyzer contains a function, "learn_potential", which is used to infer potential energy landscapes from FRET data. FRET analyzer takes in two channel FRET data of shape (N,2) where each row is the number of photons collected in each channel and each column is the photons collected in a time level. FRET analyzer also optionally takes in a dictionary, "parameters", which contains information relevent to the experiment such as the time step (dt), temperature (kT), and characteristic FRET pair distance (R0).

To analyze a data set simply import FRETAnalyzer and run learn_potential on your dataset and parameters. The output is the maximum a posteriori set of variables, which can be visualized using FRETAnalyzer.plot_variables.

See example.py for a demonstration.

# Comparing to HMM

We additionally created a class "FRETAnalyzerHMM" which performs the same analysis with HMM methods. Running the FRETAnalyzerHMM is identical to running FRETAnalyzer.

# Questions and further explanation

FRETAnalyzer is a work in progress. Further documentation will be provided as it is created. If you require assistance or would like more details, please do not hesitate to contact us at jsbryan4@asu.edu

