# This code was developed by Siobhan Powell, 2021. siobhan.powell@stanford.edu

import pandas as pd
import numpy as np
import copy


class FutureGrid(object):
    """By Siobhan Powell. This class manages the model of the future grid and implements dispatch / capacity calculations.

    :param gd_short: The generator model
    :type gd_short: An object of class `generatorDataShort` from `simple_dispatch.py`

    :param unit_drops: Information about which generators are retired in each year
    :type unit_drops: Dataframe
    :param additions_df: Information about which generators are added each year
    :type additions_df: Dataframe

    :param year: Year for the future grid
    :type year: int

    :param future: Future grid demand, including EV demand
    :type future: An object of class `FutureDemand` from later in this file

    :param stor_df: Demand that needs to be met by storage; passed to storage model object
    :type stor_df: Dataframe
    :param storage: Storage model
    :type storage: An object of the class `StorageModel` from `simple_dispatch.py`

    :param bs: Bidstack
    :type bs: An object of the class `bidStack` by Thomas Deetjen from `simple_dispatch.py`
    :param dp: Dispatch
    :type dp: An object of the class `dispatch` by Thomas Deetjen from `simple_dispatch.py`
    """
    
    def __init__(self, gd_short):
        
        self.gd_short = gd_short
        self.gd_short_original = copy.deepcopy(gd_short)
        self.unit_drops = pd.read_csv('IntermediateOutputs/scheduled_retirements_2019.csv', index_col=0)
        self.additions_df = pd.read_csv('IntermediateOutputs/generator_additions.csv', index_col=0)
        self.year = None
        self.future = None
        self.stor_df = None
        self.storage = None
        self.bs = None
        self.dp = None

    def add_generators(self, future_year):
        """Duplicate generators to simulate new additions in the future WECC grid."""
        
        gd_short_final = copy.deepcopy(self.gd_short)
        added_units = self.additions_df[self.additions_df['Year']<future_year]['orispl_unit'].values
        for i, val in enumerate(added_units):
            idx = len(gd_short_final.df)
            loc1 = gd_short_final.df[gd_short_final.df['orispl_unit']==val].index
            gd_short_final.df = pd.concat((gd_short_final.df, gd_short_final.df.loc[loc1]), ignore_index=True)
            gd_short_final.df.loc[idx, 'orispl_unit'] = 'added_'+str(i)
            
        self.gd_short = copy.deepcopy(gd_short_final)
        
    def drop_generators(self, future_year):
        """Drop generators to match announced retirements in the WECC grid."""
        
        gd_short_final = copy.deepcopy(self.gd_short)

        dropped_units = self.unit_drops[self.unit_drops['retirement_year']<future_year]['orispl_unit'].values
        gd_short_final.df = gd_short_final.df[~gd_short_final.df['orispl_unit'].isin(dropped_units)].copy(deep=True).reset_index(drop=True)
        
        self.gd_short = copy.deepcopy(gd_short_final)
        
    def change_gas_prices(self, fuel):
        """Change fuel prices for gas generators to test sensitivity."""
        
        gd_short_final = copy.deepcopy(self.gd_short)
        
        inds = gd_short_final.df[gd_short_final.df['fuel'].isin(['ng', 'og'])].index
        gd_short_final.df.loc[inds, ['fuel_price'+str(i) for i in np.arange(1, 53)]] = fuel*gd_short_final.df.loc[inds, ['fuel_price'+str(i) for i in np.arange(1, 53)]]
        
        self.gd_short = copy.deepcopy(gd_short_final)
        
    def set_up_scenario(self, year=2030, solar=2.5, wind=2.5, fuel=1.0, ev_pen=1.0,
                        ev_scenario='High Home', ev_timers='', ev_workplace_control='', 
                        ev_workplace_bool=False, evs_bool=True, ev_scenario_date='20211119', 
                        weekend_timers='', weekend_date='20211119'):
        """Set up scenario of future demand."""

        # drop and add generators
        self.year = year
        if year != 2019:
            self.add_generators(year)
            self.drop_generators(year)
        # change fuel prices
        if fuel != 1.0:
            self.change_gas_prices(fuel)
        # model future demand
        self.future = FutureDemand(self.gd_short, year=year)
        if year != 2019:
            self.future.electrification(scale_vs_given=True)  # electrification in other sectors
        # adjust renewables levels
        self.future.solar_multiplier[year] = solar
        self.future.wind_multiplier[year] = wind
        self.future.solar()
        self.future.wind()
        # add EVs
        if evs_bool:
            if ev_workplace_bool:
                self.future.evs(pen_level=ev_pen, scenario_name=ev_scenario, timers_extra_info=ev_timers, wp_control=ev_workplace_control, scenario_date=ev_scenario_date, timers_extra_info_weekends=weekend_timers, weekend_date=weekend_date)
            else:
                self.future.evs(pen_level=ev_pen, scenario_name=ev_scenario, timers_extra_info=ev_timers, scenario_date=ev_scenario_date, timers_extra_info_weekends=weekend_timers, weekend_date=weekend_date)
        # update
        self.future.update_total()
        
    def check_overgeneration(self, save_str=None):
        """Check for negative demand. Clip and save overgeneration amount."""

        if self.future.demand['demand'].min() < 0:
            if save_str is not None:
                self.future.demand.loc[self.future.demand['demand'] < 0].to_csv(save_str+'_overgeneration.csv', index=None)
            self.future.demand['demand'] = self.future.demand['demand'].clip(0, 1e10)

    def run_storage_before_capacitydispatch(self, cap, max_rate):
        """If running storage on net demand before dispatch, do that here."""
    
        self.stor_df = pd.DataFrame({'datetime': pd.to_datetime(self.future.demand['datetime'].values),
                                     'total_demand': self.future.demand['demand'].values})
        self.storage = StorageModel(self.stor_df)
        self.storage.calculate_operation_beforecapacity(cap, max_rate)

    def run_dispatch(self, max_penlevel, save_str, result_date='20211119', return_generator_limits=False):
        """Run the dispatch. max_penlevel indicates whether storage will be needed or whether the model will break
        without it, but the try except clause will ensure the simulation is run if that is incorrect."""

        self.bs = bidStack(self.gd_short, co2_dol_per_kg=0, time=1, dropNucHydroGeo=True, include_min_output=False, mdt_weight=0.5, include_easiur=False) 
        self.dp = dispatch(self.bs, self.future.demand, time_array=scipy.arange(52)+1, return_generator_limits=return_generator_limits)
    
        if self.future.ev_pen_level <= max_penlevel:
            try:
                self.dp.calcDispatchAll()
                if save_str is not None:
                    self.dp.df.to_csv(save_str+'_dpdf_'+result_date+'.csv', index=False)
            except:
                print('Error!')
                pd.DataFrame({'Error':['Needed storage in dispatch'], 'Case':[save_str]}, index=[0]).to_csv(save_str+'_error_record.csv', index=False)
                print('----Capacity too low----')
                print('Try with storage:')
                self.dp = dispatch(self.bs, self.future.demand, time_array=scipy.arange(52)+1, include_storage=True)
                self.dp.calcDispatchAll()
                if save_str is not None:
                    self.dp.df.to_csv(save_str+'_withstorage'+'_dpdf_'+result_date+'.csv', index=False)
                self.storage = StorageModel(self.dp.storage_df)
                self.storage.calculate_minbatt_forcapacity()
                print('Storage Rate Result:', int(self.storage.min_maxrate))
                print('Storage Capacity: ', int(self.storage.min_capacity))
                if save_str is not None:
                    self.storage.df.to_csv(save_str+'_storage_operations_'+result_date+'.csv', index=False)
                self.storage_stats = pd.DataFrame({'Storage Rate Result':int(self.storage.min_maxrate),'Storage Capacity':int(self.storage.min_capacity)}, index=[0])
                if save_str is not None:
                    self.storage_stats.to_csv(save_str+'_storage_stats_'+result_date+'.csv', index=False)
        else:
            print('----Capacity too low----')
            print('Try with storage:')
            self.dp = dispatch(self.bs, self.future.demand, time_array=scipy.arange(52)+1, include_storage=True)
            self.dp.calcDispatchAll()
            if save_str is not None:
                self.dp.df.to_csv(save_str+'_withstorage'+'_dpdf_'+result_date+'.csv', index=False)
            self.storage = StorageModel(self.dp.storage_df)
            self.storage.calculate_minbatt_forcapacity()
            print('Storage Rate Result:', int(self.storage.min_maxrate))
            print('Storage Capacity: ', int(self.storage.min_capacity))
            if save_str is not None:
                self.storage.df.to_csv(save_str+'_storage_operations_'+result_date+'.csv', index=False)
            self.storage_stats = pd.DataFrame({'Storage Rate Result':int(self.storage.min_maxrate),'Storage Capacity':int(self.storage.min_capacity)}, index=[0])
            if save_str is not None:
                self.storage_stats.to_csv(save_str+'_storage_stats_'+result_date+'.csv', index=False)

    def find_capacity_limit_10_binarysearch(self, bs_limits=None, lims_8760=None, year=2030, solar=2.5, wind=2.5,
                                            fuel=1.0, ev_scenario='BaseCase_NoL1', ev_timers='',
                                            ev_workplace_control='', ev_workplace_bool=False, evs_bool=True,
                                            ev_scenario_date='20211119', with_storage_before=False, cap=None,
                                            max_rate=None, minpen=0.01, weekend_timers=None, weekend_date=None):
        """Find capacity limits. To avoid starting the search from 1% adoption each time, this method does a short
        search to find which quadrant to start looking in. It returns both the 1-hour and 10-hour breaking points."""
    
        if weekend_timers is None:
            weekend_timers = ev_timers
        if weekend_date is None:
            weekend_date = ev_scenario_date

        violated1 = False
        violated2 = False
        limit1 = 0
        limit2 = 0
        if lims_8760 is None:
            lims_8760 = np.concatenate((np.repeat(bs_limits['Max Capacity'], (24*7)), np.repeat(np.array(bs_limits.loc[51, 'Max Capacity']), 24)))
        print('Short Binary Search: ')
        penlevel = np.round((minpen+1)/2, 2)
        self.set_up_scenario(year=year, solar=solar, wind=wind, fuel=fuel, ev_scenario=ev_scenario,
                             ev_timers=ev_timers, ev_pen=penlevel, ev_workplace_control=ev_workplace_control,
                             ev_workplace_bool=ev_workplace_bool, evs_bool=evs_bool,
                             ev_scenario_date=ev_scenario_date, weekend_timers=weekend_timers, weekend_date=weekend_date)
        self.check_overgeneration()
        if with_storage_before:
            self.run_storage_before_capacitydispatch(cap, max_rate)
            total_overs = np.shape(np.where(self.storage.df.comb_demand_after_storage.values > lims_8760)[0])[0]
        else:
            total_overs = np.shape(np.where(self.future.demand.demand.values > lims_8760)[0])[0]
        print(penlevel, ':', total_overs)
        if total_overs == 0:
            mid = copy.copy(penlevel)
            penlevel = np.round((mid+1)/2, 2)
            self.set_up_scenario(year=year, solar=solar, wind=wind, fuel=fuel, ev_scenario=ev_scenario,
                                 ev_timers=ev_timers, ev_pen=penlevel, ev_workplace_control=ev_workplace_control,
                                 ev_workplace_bool=ev_workplace_bool, evs_bool=evs_bool, ev_scenario_date=ev_scenario_date,
                                 weekend_timers=weekend_timers, weekend_date=weekend_date)
            self.check_overgeneration()
            if with_storage_before:
                self.run_storage_before_capacitydispatch(cap, max_rate)
                total_overs = np.shape(np.where(self.storage.df.comb_demand_after_storage.values > lims_8760)[0])[0]
            else:
                total_overs = np.shape(np.where(self.future.demand.demand.values > lims_8760)[0])[0]
            print(penlevel, ':', total_overs)
            if total_overs == 0:
                start_pen = copy.copy(penlevel)
            else:
                start_pen = copy.copy(mid)
        else:
            mid = copy.copy(penlevel)
            penlevel = np.round((mid+minpen)/2, 2)
            self.set_up_scenario(year=year, solar=solar, wind=wind, fuel=fuel, ev_scenario=ev_scenario,
                                 ev_timers=ev_timers, ev_pen=penlevel, ev_workplace_control=ev_workplace_control,
                                 ev_workplace_bool=ev_workplace_bool, evs_bool=evs_bool,
                                 ev_scenario_date=ev_scenario_date, weekend_timers=weekend_timers, weekend_date=weekend_date)
            self.check_overgeneration()
            if with_storage_before:
                self.run_storage_before_capacitydispatch(cap, max_rate)
                total_overs = np.shape(np.where(self.storage.df.comb_demand_after_storage.values > lims_8760)[0])[0]
            else:
                total_overs = np.shape(np.where(self.future.demand.demand.values > lims_8760)[0])[0]
            print(penlevel, ':', total_overs)
            if total_overs == 0:
                start_pen = copy.copy(penlevel)
            else:
                start_pen = copy.copy(minpen)

        print('Linear search from starting point: ', start_pen)
        for penlevel in np.arange(start_pen, 1.01, 0.01):
            print(penlevel)
            penint = int(100*penlevel)
            self.set_up_scenario(year=year, solar=solar, wind=wind, fuel=fuel, ev_scenario=ev_scenario,
                                 ev_timers=ev_timers, ev_pen=penlevel, ev_workplace_control=ev_workplace_control,
                                 ev_workplace_bool=ev_workplace_bool, evs_bool=evs_bool,
                                 ev_scenario_date=ev_scenario_date, weekend_timers=weekend_timers, weekend_date=weekend_date)
            self.check_overgeneration()
            if with_storage_before:
                self.run_storage_before_capacitydispatch(cap, max_rate)
                total_overs = np.shape(np.where(self.storage.df.comb_demand_after_storage.values > lims_8760)[0])[0]
            else:
                total_overs = np.shape(np.where(self.future.demand.demand.values > lims_8760)[0])[0]
            if (total_overs == 1) and (violated1 == False):
                print('Total overs: ', total_overs)
                limit1 = copy.copy(penlevel)
                print('Violation 1: ', penlevel)
                violated1 = True
            elif (total_overs >= 1) and (violated1 == False):
                print('Total overs: ', total_overs)
                limit1 = copy.copy(penlevel)
                print('Violation 1: ', penlevel)
                violated1 = True
            if total_overs >= 10:
                print('Total overs: ', total_overs)
                limit2 = copy.copy(penlevel)
                print('Violation 10: ', penlevel)
                violated2 = True
                break
        if not violated1:
            limit1 = 1.0
        if not violated2:
            limit2 = 1.0
            
        self.limit1 = limit1
        self.limit2 = limit2

        return limit1, limit2
    

class FutureDemand(object):
    """By Siobhan Powell. This class manages the model of the future grid demand.

    :param baseline_demand: Demand data before making any adjustments
    :type baseline_demand: Dataframe
    :param demand: Modeled future demand
    :type demand: Dataframe

    :param ev_pen_level: Level of EV adoption (between 0 and 1)
    :type ev_pen_level: float

    :param year: Year for the future grid
    :type year: int

    :param all_generation_2019: Generation from non-fossil fuel sources in 2019, collected from the EIA.
    :type all_generation_2019: Dataframe
    :param not_combustion: Generation from non-fossil fuel sources in 2019
    :type not_combustion: Dataframe

    :param electrification_scaling: Amount to scale each year for electrification in other sectors
    :type electrification_scaling: dict
    :param solar_multiplier: Amount to scale solar
    :type solar_multiplier: dict
    :param wind_multiplier: Amount to scale wind
    :type wind_multiplier: dict

    :param ev_load: Weekday EV demand, unscaled
    :type ev_load: Dataframe
    :param ev_load_add: Weekday EV demand, scaled by pen level
    :type ev_load_add: Dataframe

    :param ev_load_weekend: Weekend EV demand, unscaled
    :type ev_load_weekend: Dataframe
    :param ev_load_weekend_add: Weekend EV demand, scaled by pen level
    :type ev_load_weekend_add: Dataframe

    :param wind_update_status: Track whether wind has been adjusted
    :type wind_update_status: bool
    :param solar_update_status: Track whether solar has been adjusted
    :type solar_update_status: bool
    :param electrification_update_status: Track whether electrification has been adjusted
    :type electrification_update_status: bool
    :param evs_update_status: Track whether EVs have been added
    :type evs_update_status: bool
    """

    def __init__(self, gd_short, year=2030, renewables_case='projected'):

        self.year = year
        self.baseline_demand = gd_short.demand_data.copy(deep=True)
        self.all_generation_2019 = pd.read_csv('IntermediateOutputs/region_eia_generation_data_2019.csv', index_col=0)
        self.baseline_demand['datetime'] = pd.to_datetime(self.baseline_demand['datetime'])
        self.baseline_demand['total_incl_noncombustion'] = self.baseline_demand['demand'].values + self.all_generation_2019['WECC_notcombustion'].values
        # Note how calculation was made, where "all_data" is "all_generation_2019": 
#         all_data['WECC_combustion'] = all_data.loc[:, ['WECC_COL', 'WECC_NG', 'WECC_OIL', 'WECC_OTH']].sum(axis=1)
#         all_data['WECC_notcombustion'] = all_data.loc[:, ['WECC_WAT', 'WECC_NUC', 'WECC_WND', 'WECC_SUN']].sum(axis=1)
        self.demand = self.baseline_demand.copy(deep=True)

        self.not_combustion = pd.DataFrame({'dt': self.all_generation_2019['dt'],
                                            'generation': self.all_generation_2019['WECC_notcombustion']})

        self.electrification_scaling = {2020: 1.0, 2025: 1.05, 2030: 1.1, 2035: 1.15, 2040: 1.2, 2050: 1.3}

        if renewables_case == 'announced':
            # source: for 2020-2030 https://www.wecc.org/ePubs/GenerationResourceAdequacyForecast/Pages/Nameplate.aspx
            # and https://www.wecc.org/epubs/StateOfTheInterconnection/Pages/Resource-Portfolio.aspx
            self.solar_multiplier = {2020: 1.3, 2025: 2, 2030: 2.1, 2035: 2.1, 2040: 2.1}  # multiplier on wecc 2019 level
            self.wind_multiplier = {2020: 1.2, 2025: 1.4, 2030: 1.5, 2035: 1.5, 2040: 1.5}
        elif renewables_case == 'projected':
            self.solar_multiplier = {2020: 1.3, 2025: 2, 2030: 2.5, 2035: 4, 2040: 5}
            self.wind_multiplier = {2020: 1.2, 2025: 2, 2030: 2.5, 2035: 4, 2040: 5}
        else:
            print('Missing renewables case.')

        self.ev_load = None
        self.ev_load_add = None
        self.ev_load_weekend = None
        self.ev_load_weekend_add = None

        self.efs_loads = None

        self.wind_update_status = False
        self.solar_update_status = False
        self.electrification_update_status = False
        self.evs_update_status = False

        self.ev_pen_level = 0

    def electrification(self, scale_vs_given=False):

        self.electrification_update_status = True

        self.demand['demand'] = self.electrification_scaling[self.year] * self.baseline_demand['total_incl_noncombustion'] - self.all_generation_2019['WECC_notcombustion'].values

    def solar(self, multiplier=True):

        self.solar_update_status = True

        solar_2019 = self.all_generation_2019['WECC_SUN'].values
        self.not_combustion['generation'] += (self.solar_multiplier[self.year]-1) * solar_2019
        self.demand['demand'] -= (self.solar_multiplier[self.year]-1) * solar_2019

    def wind(self):

        self.wind_update_status = True

        wind_2019 = self.all_generation_2019['WECC_WND'].values
        self.not_combustion['generation'] += (self.wind_multiplier[self.year]-1) * wind_2019
        self.demand['demand'] -= (self.wind_multiplier[self.year]-1) * wind_2019

    def evs(self, pen_level, scenario_name='HighHOme', timers_extra_info='', wp_control='', scenario_date='20211119', timers_extra_info_weekends='', weekend_date='20211119'):
        """Example scenarios: HighHome, UniversalHome, LowHome_HighWork, LowHome_LowWork.
        Example timer extras: '', '_midnighttimer', '_NoTimers'. Example wp_control: 'minpeak', 'solar'. """
        self.evs_update_status = True
        self.ev_pen_level = pen_level

        folder = '../EVDemandModel_EVScenarios/RunningModel/Outputs/'
        
        if wp_control != '':
            key1 = folder + scenario_name + '_100p' + timers_extra_info + '_WPcontrol_' + wp_control + '_WECC_' + scenario_date + '.csv'
        else:
            key1 = folder + scenario_name + '_100p' + timers_extra_info + '_WECC_' + scenario_date + '.csv'
        key2 = folder + scenario_name + '_100p' + timers_extra_info_weekends + '_weekend_WECC_' + weekend_date + '.csv'

        self.ev_load = pd.read_csv(key1, index_col=0) 
        if 'Total' not in self.ev_load.columns:
            self.ev_load['Total'] = self.ev_load.sum(axis=1)
        self.ev_load_weekend = pd.read_csv(key2, index_col=0)
        if 'Total' not in self.ev_load_weekend.columns:
            self.ev_load_weekend['Total'] = self.ev_load_weekend.sum(axis=1)

        # apply pen level and convert to MW
        self.ev_load_add = pen_level * (1/1000) * self.ev_load['Total'].values
        self.ev_load_weekend_add = pen_level * (1/1000) * self.ev_load_weekend['Total'].values

        for i in range(365):
            if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_add[np.arange(0, 1440, 60)]
            else:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_weekend_add[np.arange(0, 1440, 60)]

    def update_total(self):

        self.demand['total_incl_noncombustion'] = self.demand['demand'] + self.not_combustion['generation']
