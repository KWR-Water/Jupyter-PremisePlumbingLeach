# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:08:52 2024

@author: dasham
"""
# TODO @Amitosh: First code review like this. I am going to add a little here about how I do it. I will put TODO @Amitosh everywhere.
# TODO @Amitosh Also I will put numbers 1 - 4. where 1 means, breaking, dont do this to 4 which means convention is slightly different, but fine to leave like this
# @Bram: For comments you feel are sufficiently resolved, please feel free to mark as done or delete comments.

# TODO 4 @Amitosh you use what they call CamelCase for your variable names. It looks consistent which is good. However according to python standards this is the convention for
# classes and not variables or functions (https://peps.python.org/pep-0008/#class-names) I dont care one bit, so leave it for now. But know that it might come back one day.
# @Bram: Fixed in script but not yet on Jupyter Notebook

# TODO 3 @Amitosh I think you should remove all these run cells since it is now a jupyter nodebook
# @Bram: Yes, but I have let these stay because I might sometimes run the function on Spyder for QC.
# %%

# TODO 2 @Amitosh remove all these commented out code.
# TODO 1 @Amitosh this hard coded sys path should ofcourse go since no one has that.
# @Bram: Let us solve this by planning how we want to test this out.

import sys
# sys.path
# sys.path.append('C:\\Users\\dasham\\Anaconda3\\envs\\Aquarellus\\Lib\\site-packages')
# sys.path

#!python3
#!python3 -m pip install -r Requirements.txt

# TODO @Amitosh Vs code shows many imports here that are actually not used, does spyder do so too?
# @Bram: Done
import numpy as np
import wntr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import pickle

import datetime
from datetime import datetime
import pandas as pd
import os
from os import listdir


# %%

def extract_demand_nodes(input_file):
    """
    This function extracts all nodes to which a demand is attributed.

    Parameters
    ----------
    input_file : String
        This is a link to the location where all the EPANET input files are present (EPANET input file = geometry + water demand patterns designated to consumption points). The format is “Directory\\Subdirectory\\Subsubdirectory\\ABC”. The recommended naming of the input files is “ABCxx.inp”. “ABC” can be a name of the users choice, for example “Apartment” or “PlumbingInput”. This prefix is followed by “xx” which refers to a two digit number. In case the user chooses to make 25 input files (each input file has same plumbing geometry but variation in water demand patterns), “xx” runs from “01” to “25”.

    Returns
    -------
    demand_nodes : List
        This list contains names of all nodes at which water demand is present (consumption points).

    """

    wn = wntr.network.WaterNetworkModel(input_file)

    demand_nodes = []
    for junction_name, junction in wn.junctions():
        if junction.demand_timeseries_list[-1].base_value > 0:
            # TODO 3 @Amitosh, I think junctionname is a string always already.
            demand_nodes = np.append(demand_nodes, (junction_name))
    # @Bram: Done
    return demand_nodes

# %%

# TODO 2 @Amitosh below you explain inputfile, outputfile, numberofinputfiles but your input is only Files. So explain that instead of these


def run_model(files,
              consumption_properties,
              time_properties, quality_properties,
              leaching_properties, factor):
    """
    This function runs the model with the trifecta of geometry, water demand, and metal leaching.
    This function uses the inputs to calculate metal concentrations at the demand points.
    Two files are saved. 
    One wherein the water demand and metal concentrations at all consumption points are stored. 
    Another wherein all the input parameters are saved for reference.

    Parameters
    ----------
    files : Dictionary
        This variable has multiple fields related to input/output.
    consumption_properties : Dictionary
        This variable has multiple fields that are related to the generated consumption patterns.
    time_properties : Dictionary
        This variable has multiple fields that are related to the desired timesteps for the simulations.
    quality_properties : Dictionary
        This variable has multiple fields regarding the parameters surrounding water quality simulations.
    leaching_properties : Dictionary
        This variable has fields that determines the nature and locations of metal dissolution into water.
    factor : Integer
        This is a numerical value with which the time and demand properties of simulations are altered. All water demands are reduced by the factor while all timesteps are increased with the same factor. This is necessary to combat the high displacement of water in short pipes (caused by high velocities) which can lead to inaccuracies in water quality computations. Therefore, the timestep for water quality calculations is not adjusted to allow for a greater separation between hydraulic and water quality timescales. The higher the factor, the longer the computational times.

    Returns
    -------
    None.

    """

    start_time = time.time()
    timesteps = int(time_properties['Duration'] /
                    time_properties['Hydraulic Timestep']+1)  # Calculate number of timesteps in the simulation

    for file in range(files['Number of files']):
        input_file = files['Input Directory'] + \
            files['Input Prefix'] + '%02d.inp' % (file+1)
        print('Processing file ' + str(file+1) +
              '/' + str(files['Number of files']))

        if file == 0:
            demand_nodes = extract_demand_nodes(input_file)
            consumption_properties['ConsumptionPoints'] = demand_nodes
            # Water quality #TODO 3 @Amitosh call it quality or even water_quality nota qual
            water_quality = np.zeros(
                [len(demand_nodes), timesteps, files['Number of files']])
            # @Bram: Done
            # Demand #TODO 3 @Amitosh call is demand not dem
            demand = np.zeros(
                [len(demand_nodes), timesteps, files['Number of files']])
            # @Bram: Done

        wn = wntr.network.WaterNetworkModel(input_file)

        wn.options.time.hydraulic_timestep = time_properties['Hydraulic Timestep']*factor
        # No multiplication with factor since this value is to be kept as low as possible.
        wn.options.time.quality_timestep = time_properties['Quality Timestep']
        wn.options.time.duration = time_properties['Duration']*factor
        wn.options.time.report_timestep = time_properties['Report Timestep']*factor
        wn.options.time.pattern_timestep = time_properties['Pattern Timestep']*factor

        for junction_name, junction in wn.junctions():  # TODO 3 @Amitosh call it junction not junc
            # @Bram: Done
            junction.demand_timeseries_list[-1].base_value /= factor

        wn.options.quality.parameter = quality_properties['Parameter']
        wn.options.quality.inpfile_units = quality_properties['Units']
        wn.options.quality.tolerance = quality_properties['Tolerance']
        wn.options.quality.diffusivity = quality_properties['Diffusivity']

        wn.options.reaction.bulk_order = 1  # First order bulk reaction at the wall
        # / 1000000 # ug/L -> kg/m3
        wn.options.reaction.limiting_potential = leaching_properties['Equilibrium Concentration']
        for location in leaching_properties['Locations']:
            # Convert leaching rate to an inverse timescale
            wn.links[location].bulk_coeff = ((leaching_properties['Rate']/(
                leaching_properties['Equilibrium Concentration']*1000)) * (4/(1000*wn.links[location].diameter)) / factor)  # /s

        simulator = wntr.sim.EpanetSimulator(wn)
        results = simulator.run_sim()

        for node in range(len(demand_nodes)):
            water_quality[node:node+1, :, file:file+1] = np.reshape(np.array(
                results.node['quality'][demand_nodes[node]]), (1, timesteps, 1))  # kg/m3
            demand[node:node+1, :, file:file+1] = np.reshape(np.array(
                results.node['demand'][demand_nodes[node]]), (1, timesteps, 1))  # m3/s
    # TODO 3 @Amitosh, here I would also prefer full names. It is good practice to only use abbreviations for things we all agree on (like loc or temp) and use full names for anything else. It will make sure that someone better understands what is happening
    # @Bram: Done
    water_quality_full = water_quality
    demand_full = demand*factor  # Correct for rescaling
    water_quality_save = water_quality_full[:, int((time_properties['Duration']-consumption_properties['Pattern Duration']) /
                                                   time_properties['Pattern Timestep'])+1:, :]  # HARDCODING - Use only second week of simulation
    demand_save = demand_full[:, int((time_properties['Duration']-consumption_properties['Pattern Duration']) /
                                     time_properties['Pattern Timestep'])+1:, :]  # HARDCODING - Use only second week of simulation

    run_settings = dict()
    run_settings['Files'] = files
    run_settings['ConsumptionProperties'] = consumption_properties
    run_settings['TimeProperties'] = time_properties
    run_settings['QualityProperties'] = quality_properties
    run_settings['LeachingProperties'] = leaching_properties
    run_settings['Factor'] = factor
    run_settings['ComputationalTime'] = int(time.time() - start_time)
    run_settings['Date'] = datetime.now()
    run_settings['Version'] = '2024.09.04'

    np.savez(files['Output Directory']+files['Output Prefix'],
             water_quality=water_quality_save, demand=demand_save)
    with open(files['Output Directory']+files['Output Prefix']+'Settings.pkl', 'wb') as fp:
        pickle.dump(run_settings, fp)

    # pd.DataFrame(RunSettings).to_csv(Files['Output Directory']+Files['Output Prefix']+'.csv',index=False)

    print('Time elapsed: ' + str(int(time.time() - start_time)) + ' seconds')

# %%


def copy_demand_from_existing_geometry(input_file, junction_input, pattern_input,
                                       output_file, junction_output, pattern_output):
    """
    This function copies demand patterns from an existing input file and writes it to another input file where demand patterns do not exist.

    Parameters
    ----------
    input_file : String
        This is a link to the EPANET input file from which water demand patterns are copied.
    junction_input : List of strings
        This is a list containing the names of all nodes in InputFile to which water demand is attributed.
    pattern_input : List of strings
        Each node has a certain demand pattern attributed to it. This variable contains a list with the names of the demand patterns in InputFile. The value at location X in this list corresponds to location X in the list Junc_Input
    output_file : String
        This is a link to the EPANET input file to which water demand patterns are copied.
    junction_output : List of strings
        This is a list containing the names of all nodes in OutputFile to which water demand is to be attributed.
    pattern_output : List of strings
        Each node has a certain demand pattern attributed to it. This variable contains a list with the names of the demand patterns in OutputFile. The value at location X in this list corresponds to location X in the list Junc_Output

    Returns
    -------
    None.

    """

    # TODO 3 @Amitosh wn_in and wn_out are clearer
    wn_in = wntr.network.WaterNetworkModel(input_file)
    # @Bram: Done
    wn_out = wntr.network.WaterNetworkModel(output_file)

    for node in range(len(junction_input)):
        # Add a pattern to the model with the name of the node.
        wn_out.add_pattern(pattern_output[node], wn_in.get_pattern(
            pattern_input[node]).multipliers)
        wn_out.get_node(junction_output[node]).add_demand(
            base=1, pattern_name=pattern_output[node])  # Add demand pattern to the node

    for junction_name, junction in wn_out.junctions():
        # The patterns are added as the second index of the demand time series and the first one is turned to zero
        junction.demand_timeseries_list[0].base_value = 0

    wn_out.write_inpfile(output_file)

# %%


# TODO 2 @Amitosh what is MWE? Use a different name if possible.
# @Bram: Done
def minimal_working_example_copy_demand_from_existing_geometry():
    """
    This function executes a simple example wherein demand patterns from one series of input files is copied to another using the function "copy_demand_from_existing_geometry".

    Returns
    -------
    None.

    """

    number_of_input_files = 1
    input_file = 'Examples\\CopyExample\\Input\\Example'
    output_file = 'Examples\\CopyExample\\Output\\Output_'

    for file in range(number_of_input_files):
        input_loc = input_file + '%02d.inp' % (file+1)
        output_loc = output_file + '%02d.inp' % (file+1)
        copy_demand_from_existing_geometry(input_loc,
                                           extract_demand_nodes(input_loc),
                                           ['SIMDEUM_' +
                                               m for m in extract_demand_nodes(input_loc)],
                                           output_loc,
                                           extract_demand_nodes(input_loc),
                                           ['SIMDEUM_'+m for m in extract_demand_nodes(input_loc)])

# %%


def load_output(output_file):
    '''
    This function loads previously saved files into the workspace.

    Parameters
    ----------
    output_file : String
        This is a link to the location where all the output files will be saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Two files will be saved with names “XYZ.npz” and “XYZSettings.pkl”.

    Returns
    -------
    demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    water_quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    run_settings : Dictionary
        Contains all fields used to run the corresponding simulations

    '''
    data = np.load(output_file+".npz", allow_pickle=True)
    demand = data['demand']
    water_quality = data['water_quality']
    with open(output_file+'Settings.pkl', 'rb') as fp:
        run_settings = pickle.load(fp)

    return demand, water_quality, run_settings

# %%


def npz_to_xlsx(output_file):
    '''
    This function converts the outputs of a previous simulation (saved in npz format) to a CSV format.

    Parameters
    ----------
    output_file : String
        This is a link to the location where the previously generated output files have been saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Ideally, there should already be two files with names “XYZ.npz” and “XYZSettings.pkl”.

    Returns
    -------
    None.

    '''

    demand, water_quality, run_settings = load_output(output_file)
    number_of_input_files = run_settings['Files']['Number of files']
    demand_nodes = extract_demand_nodes(
        run_settings['Files']['Input Directory']+run_settings['Files']['Input Prefix']+'01.inp')

    with pd.ExcelWriter(output_file+'.xlsx') as writer:
        for file in range(number_of_input_files):
            print('Dumping data ' + str(file+1) +
                  '/' + str(number_of_input_files))
            df_dump_water_quality = pd.DataFrame(
                np.transpose(water_quality[:, :, file]))
            df_dump_demand = pd.DataFrame(np.transpose(demand[:, :, file]))
            df_dump_water_quality.to_excel(writer, sheet_name='Water Quality '+'%02d.inp' % (file+1),
                                           header=demand_nodes)
            df_dump_demand.to_excel(writer, sheet_name='Demand '+'%02d.inp' % (file+1),
                                    header=demand_nodes)

# %%


def pkl_to_csv(output_file):
    '''
    This function converts the run settings of a previous simulation (saved in pkl format) to a CSV format.

    Parameters
    ----------
    output_file : String
        This is a link to the location where the previously generated output files have been saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Ideally, there should already be two files with names “XYZ.npz” and “XYZSettings.pkl”.

    Returns
    -------
    None.

    '''

    with open(output_file+"Settings.pkl", "rb") as f:
        object = pickle.load(f)

    df = pd.DataFrame(object)
    df.to_csv(output_file+"Settings.csv")

# %%


def compute_tap_outputs(demand, water_quality, timestep):
    '''
    This function calculates the total mass water, total mass metal and average concentration metal at each consumption point. 

    Parameters
    ----------
    demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    water_quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    timestep : Integer
        Timestep with which water demand patterns have been generated.

    Returns
    -------
    tap_outputs : Dictionary
        Contains three arrays - one for total mass water, one for total mass metal, one for average concentration metal.
        Each array is two dimensional. First dimension for all consumption points. Second dimension for all input files with unique water consumption patterns.
    '''
    water_quality_output = np.multiply(np.where(
        demand > 0, 1, 0), water_quality)    # Quality of water collected [kg/m3]
    # Total water consumed per week per node [L]
    total_water = np.trapz(demand, dx=timestep, axis=1)*(10**3)
    # Total lead consumed per week per node [ug]
    total_metal = np.trapz(water_quality_output*demand,
                           dx=timestep, axis=1)*(10**9)
    total_concentration = total_metal/total_water

    tap_outputs = dict()
    tap_outputs['TotalWater'] = total_water
    tap_outputs['TotalMetal'] = total_metal
    tap_outputs['TotalConcentration'] = total_concentration

    return tap_outputs

# %%


def compute_CDF(demand, water_quality, node, plot):
    '''
    This function plots the cumulative distribution of metal concentration at a demand point.
    The plot is based on the compilation of all input files.

    Parameters
    ----------
    demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    water_quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    node : Integer
        The index of the consumption point for which the plot is to be made.
        Refer to the key 'Consumption Points' in the dictionary 'ConsumptionProperties'
    plot : Boolean
        Choice to plot the cumulative distribution function

    Returns
    -------
    CDF_properties : Array of floats
        A two dimensional array which contains properties of the cumulative distribution function
        Each array is two dimensional. First dimension for all consumption points. Second dimension for all input files with unique water consumption patterns.
    '''
    demand_CDF = (np.squeeze(demand[node, :, :])).flatten()  # For node make CDF of all weeks
    # For node make CDF of all weeks
    water_quality_CDF = (np.squeeze(water_quality[node, :, :])).flatten()
    consumption_CDF = water_quality_CDF[np.where(demand_CDF > 0)[0]]
    consumption_sort = np.sort(consumption_CDF)*1000000
    consumption_probability = np.arange(
        len(consumption_CDF)) / float(len(consumption_CDF))
    CDF_properties = [consumption_sort, consumption_probability]

    if plot == True:
        fig, axs = plt.subplots()
        plt.rc('figure', titlesize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=18)
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels

        axs.plot(consumption_sort, consumption_probability, linestyle='none',
                 marker='o', linewidth=1, markerfacecolor="black", markeredgecolor="none")
        axs.set_xlabel('Metal Concentration [$\mu$g/L]')
        axs.set_ylabel('Probability [-]')
        axs.grid(True)
        axs.set_xscale('log')
        axs.set_aspect(3)

    return CDF_properties

# %%


def compute_fraction_exceedance(demand, water_quality, threshold):
    '''
    This function calculates the fraction of timesteps where the metal concentration at the tap exceeds a threshold value.
    The fraction is based on the total number of timesteps for which the consumption point is active and there is no correction for the volumetric flow rate.
    The fractions are for all input files combined.

    Parameters
    ----------
    demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    water_quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    threshold : Float
        This is the threshold concentration value on basis whereof the fractions are calculated.

    Returns
    -------
    fraction_exceedance : Array of float
        The fraction of times the concentration at the consumption points exceeds a threshold value.
        The array represents all the consumption points.
    '''
    water_quality_output = np.multiply(np.where(
        demand > 0, 1, 0), water_quality)    # Quality of water collected [kg/m3]
    water_quality_output_good = np.multiply(np.where(
        water_quality_output < threshold, 1, 0), water_quality_output)    # Quality of water collected
    water_quality_output_bad = np.multiply(np.where(
        water_quality_output >= threshold, 1, 0), water_quality_output)    # Quality of water collected

    instances_water_quality_good = np.sum(
        np.where(water_quality_output_good > 0, 1, 0), axis=(1, 2))
    instances_water_quality_bad = np.sum(
        np.where(water_quality_output_bad > 0, 1, 0), axis=(1, 2))

    fraction_exceedance = instances_water_quality_bad / \
        (instances_water_quality_bad+instances_water_quality_good)
    return instances_water_quality_good,instances_water_quality_bad,fraction_exceedance

# %%

# TODO 2 @Amitosh in the explanation you call them quality and demand but as input dem and qual. use the same name (and I prefer demand and quality)


def plot_demand_and_quality(demand, water_quality, node, input_file, threshold=5,
                            duration=86400*7,
                            timestep=10,
                            xlims=[0, 7],
                            ylims_demand=[-2*(10**-2), 25*(10**-2)],
                            ylims_water_quality=[0.01, 100],
                            xticks=[0, 1, 2, 3, 4, 5, 6, 7],
                            xtick_labels=([0, 1, 2, 3, 4, 5, 6, 7]),
                            format_time='Days',
                            font_size=16):
    '''
    This function plots the water demand and metal concentrations at a consumption point corresponding to a certain input file (one out of the multiple).
    Limited options are offered to tweak the visualization. However, the user can modify the visualization by modifying hardcoded choices.

    Parameters
    ----------
    demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    water_quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    node : Integer
        The index of the consumption point for which the plot is to be made.
        Refer to the key 'Consumption Points' in the dictionary 'ConsumptionProperties'.
    input_file : Integer
        Typically, multiple input files are present (each with a different water demand pattern).
        This parameter selects which of the numbered input files is to be considered.
    threshold : Integer, optional
        A threshold metal concentration value used for visualization purposes. 
        Values below and above the threshold are visualized with different colored markers. 
        The default is 5 $\mu$g/L.
    duration : Integer, optional
        This refers to the duration of the water demand patterns. 
        The default is 86400*7 seconds.
    timestep : Integer, optional
        This refers to the resolution of the water demand patterns. 
        The default is 10 seconds.
    xlims : Array of floats, optional
        This sets the limits of the plot (horizontal axis). 
        The default is [0,7].
    ylims_demand : Array of floats, optional
        This sets the limits of the plot (vertical axis - demand).  
        The default is [-2*(10**-2),25*(10**-2)].
    ylims_water_quality : Array of floats, optional
        This sets the limits of the plot (vertical axis - concentrations). Note that the plot is logarthmic scale.  
        The default is [0.01,100].
    xticks : Array of floats, optional
        This determines the locations where the ticks on the horizontal axis will appear. 
        The default is [0,1,2,3,4,5,6,7].
    xtick_labels : Array  of floats, optional
        This determines the labels of the ticks on the horizontal axis. 
        The default is ([0,1,2,3,4,5,6,7]).
    format_time : String, optional
        This determines whether time will be shown with units of days or hours:minutes:seconds.
        The input should be either 'Days' or 'HMS'.
        The default is 'Days'.
    font_size : Integer, optional
        This determines the font sizes of the axes and the labels in the figures.
        The default is 16.

    Returns
    -------
    None.

    '''

    threshold = threshold/1000000
    # Quality of water collected [kg/m3]
    water_quality_output = np.multiply(
        np.where(demand > 0, 1, 0), water_quality)
    water_quality_output_good = np.multiply(np.where(
        water_quality_output < threshold, 1, 0), water_quality_output)    # Quality of water collected
    water_quality_output_bad = np.multiply(np.where(
        water_quality_output >= threshold, 1, 0), water_quality_output)    # Quality of water collected

    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.rc('figure', titlesize=font_size)
    plt.rc('axes', titlesize=font_size*0.75)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
    plt.style.use('default')

    axs[0].set_xlim(xlims[0], xlims[1])
    axs[0].set_ylim(ylims_demand[0], ylims_demand[1])

    usage = axs[0].stem(np.linspace(0, int(duration/86400), int(duration/timestep)),
                        1000*demand[node, :, input_file-1], linefmt='k-', markerfmt='bo', label="Usage")
    axs[0].set_ylabel('Water demand \n [liters per second]')
    axs[0].grid(True)
    axs[0].set_aspect('auto')
    if format_time == 'HMS':
        xfmt = mdates.DateFormatter('%H:%M:%S')
        axs[0].xaxis.set_major_formatter(xfmt)

    axs[1].plot(xlims, [threshold*1000000, threshold*1000000],
                ':k', linewidth=2)
    wntr, = axs[1].plot(np.linspace(0, int(duration/86400), int(duration/timestep)), 1000000*water_quality[node,
                        :, input_file-1], linewidth=3, color="lightseagreen", alpha=0.2, label="Concentration at tap")
    good, = axs[1].plot(np.linspace(0, int(duration/86400), int(duration/timestep)), 1000000*water_quality_output_good[node, :, input_file-1], linewidth=1,
                        linestyle='none', marker='o', markeredgecolor="cornflowerblue", markerfacecolor="cornflowerblue", markersize=10, label="Concentration below threshold")
    bad, = axs[1].plot(np.linspace(0, int(duration/86400), int(duration/timestep)), 1000000*water_quality_output_bad[node, :, input_file-1], linewidth=1,
                       linestyle='none', marker='o', markeredgecolor="lightcoral", markerfacecolor="lightcoral", markersize=10, label="Concentration above threshold")

    if format_time == 'Days':
        axs[1].set_xlabel('Time [days]')
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xtick_labels)
    else:
        axs[1].set_xlabel('Time')
        axs[1].set_xticks(xticks)
    axs[1].set_ylim([ylims_water_quality[0], ylims_water_quality[1]])
    axs[1].set_ylabel('Metal \n concentration [$\mu$g/L]')
    axs[1].set_yscale('log')
    axs[1].grid(True)

    plt.rc('figure', titlesize=font_size)
    plt.rc('axes', titlesize=font_size*0.75)
    plt.rc('axes', labelsize=font_size)
    plt.rc('xtick', labelsize=font_size)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_size)    # fontsize of the tick labels
