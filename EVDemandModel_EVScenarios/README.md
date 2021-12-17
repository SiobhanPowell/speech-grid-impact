# EV Charging Demand Model Elements

This code accompanies the paper: 

S. Powell, G. V. Cezar, L. Min, I. Azevedo and R. Rajagopal, "Charging Infrastructure Access and Operation to Reduce the Grid Impacts of Deep Electric Vehicle Adoption", Submitted.

### SPEECh

This model is an extension of the SPEECh model (Scalable Probabilistic Estimates of EV Charging) available here: https://github.com/SiobhanPowell/speech. 

Siobhan Powell, Gustavo Vianna Cezar, & Ram Rajagopal. (2021). SiobhanPowell/speech: First release (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.5593509

Copyright (c) 2021, SiobhanPowell
All rights reserved.

Paper: 

S. Powell, G. V. Cezar, R. Rajagopal, "Scalable Probabilistic Estimates of Electric Vehicle Charging Given Observed Driver Behavior", Applied Energy. Accepted.

### SCRIPT Control Model

The methodology used to model charging control was first proposed as part of the project SCRIPT, available here: https://github.com/slacgismo/SCRIPT-tool.

Paper: 

Powell, S., Cezar, G. V., Apostolaki-Iosifidou, E., & Rajagopal, R. (2021). "Large Scale Scenarios of Electric Vehicle Charging with a Data-Driven Model of Control." Under review.

### Folder Organization
This folder is divided into four sub-folders: 

1. PreProcessing
    - These files show how the model inputs were prepared
    - The raw charging data used in these files cannot be made public
    - We cannot post the census data. It was accessed through: simplyanalytics.com
2. RunningModel
    - The main model of EV charging demand
    - `speech_classes.py` contains the model
    - `main_scenarios.ipynb` runs the main scenarios of Universal Home, High Home, Low Home High Work, and Low Home Low Work Access
    - `supplement_scenarios.ipynb` runs scenarios included in the supplementary information including the sensitivity to using more fast charging or larger vehicle batteries.
    - Subfolder `Outputs` contains the model outputs; subfolder `Data` contains a folder with the model inputs
    - `PreparePostedData.ipynb` shows how the group demand profiles were created that are in the posted data set.
3. Plotting
    - Code for plotting the scenario load profiles, used to create figures 1, 2a, and a figure in the supplement.
4. Controls
    - Learning the control model for each workplace control rule
    - Applying the learned control models to the profiles output above by `main_scenarios.ipynb`
 
 Finally, we include a sample notebook to show how you can use the code and model to run your own scenarios with the data objects we were able to share. This is in `RunningModel/Example_RunYourOwnScenarios.ipynb`. 
 
 ### Data
 
 From the posted data, https://data.mendeley.com/datasets/y872vhtfrc/1, download the contents of `ChargingModelData` and save it in the following folder structure: `RunningModel/Data/CP136/`. Please refer to the sample notebook `RunningModel/Example_RunYourOwnScenarios.ipynb` for how to run your own scenarios. Please contact siobhan.powell@stanford.edu with questions. 

 
  