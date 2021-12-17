# Grid Model 

Siobhan Powell, 2021.

This code accompanies the paper: 

S. Powell, G. V. Cezar, L. Min, I. Azevedo and R. Rajagopal, "Charging Infrastructure Access and Operation to Reduce the Grid Impacts of Deep Electric Vehicle Adoption", Submitted.

### Fossil Fuel Generator Dispatch

The dispatch of fossil fuel generators is an extension of the simple dispatch model available here: https://github.com/tdeetjen/simple_dispatch. 

That code was developed by Thomas Deetjen. The model accompagnies the paper:

Deetjen, Thomas A., and Inês L. Azevedo. "Reduced-order dispatch model for simulating marginal emissions factors for the united states power sector." Environmental science & technology 53.17 (2019): 10506-10513.

### Extension

This extension of the model can be broken down into several steps: 
1. Updated model of fossil fuel generators
2. Model of future net demand
    1. New loads, including EV charging demand under different scenarios
    2. Increased renewable generation
3. Model of grid storage


### Folder Organization

The code behind the main analysis and plots in the paper is in `MainAnalysis` and `MainPlotting`. 

This folder is divided into several sub folders: 
1. Main model in the main folder, `simple_dispatch.py` and `future_grid.py`
1. Analysis in `MainAnalysis` and `SupplementAnalysis`
2. Plotting in `MainPlotting` and `SupplementPlotting`
3. PreProcessing which contains the files used to update the grid model
4. Data
    1. `InputData` which includes the raw input data needed to prepare the grid model. Note this is NOT needed to run the model - you can use the `generator_data_short_WECC_2019.obj` object instead. This folder contains updated input data for 2019 from the same sources as listed in https://github.com/tdeetjen/simple_dispatch. Most cannot be rehosted, so it is described here in detail: 
        1. `EIA923_Schedules_2_3_4_5_M_12_2019_Final_Revision.xslx` available from here: https://www.eia.gov/electricity/data/eia923/
        3. `egrid2019_data.xlsx` available here: https://www.epa.gov/egrid/download-data
        4. A subfolder `CEMS` with data available from here, organized into further subfolders by state code: ftp://newftp.epa.gov/DmDnLoad/emissions/hourly/monthly/
        5. `Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv` and `Respondent IDs.csv` from here: https://www.ferc.gov/industries-data/electric/general-information/electric-industry-forms/form-no-714-annual-electric/data
        6. `fuel_default_prices.xlsx` available from: https://github.com/tdeetjen/simple_dispatch and updated with more recent data in the code below.
    2. `IntermediateData` which includes processed data needed to run the model. 
        1. `region_eia_generation_data_2019.csv` which was collected from the EIA using the code in `PreProcessing/4_NonFossilFuelGeneration2019.ipynb`.
        2. `generator_additions.csv` and `scheduled_retirements_2019.csv` which are posted in the online data set and which were created in `PreProcessing/5_Generator_Retirements.ipynb` and `PreProcessing/6_Generator_Additions.ipynb`. 
        3. `generator_data_short_WECC_2019.obj` which is posted in the online data set and which was created in `PreProcessing/2_UpdateGeneratorModel.ipynb`
5. Results in `Results`

### Data

From the posted data, https://data.mendeley.com/datasets/y872vhtfrc/1, please download the contents of `GridModel_Objects` and save them in a folder called `IntermediateData`.

### Run the Model Yourself
To run the model yourself, you will need to download the IntermediateData folder objects in the posted data set, and run `PreProcessing/4_NonFossilFuelGeneration2019.ipynb` to collect the non fossil fuel generation data.

Then, running the model involves the following steps. First, import all the classes you may need:
```
from simple_dispatch import generatorData
from simple_dispatch import bidStack
from simple_dispatch import dispatch
from simple_dispatch import generatorDataShort
from future_grid import FutureDemand
from future_grid import FutureGrid
from simple_dispatch import StorageModel
```
Then, load the generator model (available with the posted data):
```
gd_short = pickle.load(open('IntermediateOutputs/generator_data_short_WECC_2019.obj', 'rb'))
```
Then, set up the grid model. For this example, we will multiply solar and wind 2x 2019 levels. If you have run a new scenario of charging demand using the ev demand model code, you can use that here:
```
grid = FutureGrid(gd_short)
grid.set_up_scenario(year=2030, solar=2, wind=2, fuel=1, ev_scenario='CustomScenario', 
                     ev_timers='_midnighttimer', ev_pen=1.0, ev_workplace_bool=False, evs_bool=True, 
                     ev_scenario_date='20211217', weekend_date='20211217',  weekend_timers='_midnighttimer')
grid.check_overgeneration(save_str='testing')
grid.run_dispatch(0.5, save_str='testing', result_date='20211217')
```
You could also adjust the demand yourself, not using our model of EVs. For example, if you had a dataframe `custom_demand` with hourly EV demand from your own modeling:
```
grid = FutureGrid(gd_short)
grid.set_up_scenario(year=2030, solar=2, wind=2, fuel=1, evs_bool=False)
grid.check_overgeneration(save_str='testing')
grid.future.demand['demand'] = grid.future.demand['demand'] + custom_demand['demand'].values
grid.run_dispatch(0.5, save_str='testing', result_date='20211217') 
```
Finally, you could run storage before the dispatch as we do in the paper. E.g. with 5 GW 4 hour storage: 
```
grid = FutureGrid(gd_short)
grid.set_up_scenario(year=2030, solar=2, wind=2, fuel=1, ev_scenario='CustomScenario', 
                     ev_timers='_midnighttimer', ev_pen=1.0, ev_workplace_bool=False, evs_bool=True, 
                     ev_scenario_date='20211217', weekend_date='20211217',  weekend_timers='_midnighttimer')
grid.check_overgeneration(save_str='testing')
# Run storage before dispatch: 
grid.run_storage_before_capacitydispatch(cap=int(4*5000), max_rate=5000)
grid.storage.df.to_csv('testing'+'_storagebeforedf_'+'20211217'+'.csv')
grid.future.demand['demand'] = np.copy(grid.storage.df.comb_demand_after_storage.values)
grid.run_dispatch(0.5, 'testing', result_date='20211217')
```

