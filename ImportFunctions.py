# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:08:52 2024

@author: dasham
"""
# %% 

import sys
#sys.path
sys.path.append('C:\\Users\\dasham\\Anaconda3\\envs\\Aquarellus\\Lib\\site-packages')
#sys.path

#!python3
#!python3 -m pip install -r Requirements.txt

import numpy as np
import wntr
import matplotlib
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib import style, cm
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from wntr.graphics.network import plot_network 
import seaborn as sns
import time
import pickle
import matplotlib.patches as mpatches
import matplotlib.animation as animation
# sns.set_theme(style="whitegrid")
import csv
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage

import os
import datetime
from datetime import datetime
# os.chdir('D:\\Users\\dasham\\Documents\\Projects\\Archives\\LoodAfgifte - Fase I\\Fase I\\Codes\\')
import pandas as pd
from os import listdir
import networkx as nx
from tempfile import TemporaryFile
import winsound
# from MSXBinaryReader import Model_modify_ALL_Diams,getVolumeBetweenNodes,ModifyDurationAndOverwrite, ReturnLengthLinksAndMaxvelocitiesModel, MSXBinReader, ModifyINP_timeAndBaseDemand,RunMSXcommandLine, CheckDFwithINPfile, ProcessWithDemand, Model_multiply_ALL_LENGTHS, Model_SplitPipe

from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

# %%

def ExtractDemandNodes(InputFile):
    '''
    This function extracts all nodes to which a demand is attributed

    Parameters
    ----------
    InputFile : String
        This is a link to the location where all the EPANET input files are present (EPANET input file = geometry + water demand patterns designated to consumption points). The format is “Directory\\Subdirectory\\Subsubdirectory\\ABC”. The recommended naming of the input files is “ABCxx.inp”. “ABC” can be a name of the users choice, for example “Apartment” or “PlumbingInput”. This prefix is followed by “xx” which refers to a two digit number. In case the user chooses to make 25 input files (each input file has same plumbing geometry but variation in water demand patterns), “xx” runs from “01” to “25”.

    Returns
    -------
    DemandNodes : List
        This list contains names of all nodes at which water demand is present (consumption points).
    '''
    
    inp_file = InputFile
    wn = wntr.network.WaterNetworkModel(inp_file)
    
    DemandNodes = []
    for junctionname, junction in wn.junctions():
        # print(junctionname)
        if junction.demand_timeseries_list[-1].base_value > 0:
            DemandNodes = np.append(DemandNodes,str(junctionname))
            
    return DemandNodes

# %%

def RunModel(Files,
             ConsumptionProperties,
             TimeProperties,QualityProperties,
             LeachingProperties,Factor):
    '''
    This function runs the model with the trifecta of geometry, water demand and metal leaching.
    This function uses the inputs to calculate metal concentrations at the consumption points.
    Two files are saved. 
    One wherein the water demand and metal concentrations at all consumption points are stored. 
    Another wherein all the input parameters are saved for reference.

    Parameters
    ----------
    InputFile : String
        This is a link to the location where all the EPANET input files are present (EPANET input file = geometry + water demand patterns designated to consumption points). The format is “Directory\\Subdirectory\\Subsubdirectory\\ABC”. The recommended naming of the input files is “ABCxx.inp”. “ABC” can be a name of the users choice, for example “Apartment” or “PlumbingInput”. This prefix is followed by “xx” which refers to a two digit number. In case the user chooses to make 25 input files (each input file has same plumbing geometry but variation in water demand patterns), “xx” runs from “01” to “25”.
    OutputFile : String
        This is a link to the location where all the output files will be saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Two files will be saved with names “XYZ.npz” and “XYZSettings.pkl”.
    NumberOfInputFiles : Integer
        This defines the number of input files that will simulated and is related to the suffix for the input files.
    ConsumptionProperties : Dictionary
        This variable has multiple fields that are related to the generated consumption patterns
    TimeProperties : Dictionary
        This variable has multiple fields that are related to the desired timesteps for the simulations.
    QualityProperties : Dictionary
        This variable has multiple fields regarding the parameters surrounding water quality simulations.
    LeachingProperties : Dictionary
        This variable has fields that determines the nature and locations of metal dissolution into water.
    Factor : Integer
        This is a numerical value with which the time and demand properties of simulations are altered. All water demands are reduced by the factor while all timesteps are increased with the same factor. This is necessary to combat the high displacement of water in short pipes (caused by high velocities) which can lead to inaccuracies in water quality computations. Therefore, the timestep for water quality calculations is not adjusted to allow for a greater separation between hydraulic and water quality timescales. The higher the factor, the longer the computational times.

    Returns
    -------
    None.

    '''
    
    startTime = time.time()
    Timesteps = int(TimeProperties['Duration']/
                    TimeProperties['Hydraulic Timestep']+1) # Calculate number of timesteps in the simulation
    
    for file in range(Files['Number of files']):
        inp_file = Files['Input Directory'] + Files['Input Prefix'] + '%02d.inp' % (file+1)
        print('Processing file ' + str(file+1) + '/' + str(Files['Number of files']))

        if file == 0:
            DemandNodes = ExtractDemandNodes(inp_file)
            ConsumptionProperties['ConsumptionPoints'] = DemandNodes
            Qual = np.zeros([len(DemandNodes), Timesteps, Files['Number of files']])   # Water quality
            Dem = np.zeros([len(DemandNodes), Timesteps, Files['Number of files']])   # Demand
        
        wn = wntr.network.WaterNetworkModel(inp_file)
        
        wn.options.time.hydraulic_timestep = TimeProperties['Hydraulic Timestep']*Factor 
        wn.options.time.quality_timestep = TimeProperties['Quality Timestep'] # No multiplication with factor since this value is to be kept as low as possible.
        wn.options.time.duration = TimeProperties['Duration']*Factor
        wn.options.time.report_timestep = TimeProperties['Report Timestep']*Factor 
        wn.options.time.pattern_timestep = TimeProperties['Pattern Timestep']*Factor
        
        for junc_name,junc in wn.junctions():
            junc.demand_timeseries_list[-1].base_value /= Factor
            
        wn.options.quality.parameter = QualityProperties['Parameter']
        wn.options.quality.inpfile_units = QualityProperties['Units']
        wn.options.quality.tolerance = QualityProperties['Tolerance']
        wn.options.quality.diffusivity = QualityProperties['Diffusivity']
        
        wn.options.reaction.bulk_order = 1 # First order bulk reaction at the wall
        wn.options.reaction.limiting_potential = LeachingProperties['Equilibrium Concentration'] #/ 1000000 # ug/L -> kg/m3
        for loc in LeachingProperties['Locations']:
            # Convert leaching rate to an inverse timescale
            wn.links[loc].bulk_coeff =  ((LeachingProperties['Rate']/(LeachingProperties['Equilibrium Concentration']*1000)) *(4/(1000*wn.links[loc].diameter)) / Factor) # /s 
        
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        
        for node in range(len(DemandNodes)):
            Qual[node:node+1, :,file:file+1] = np.reshape(np.array(results.node['quality'][DemandNodes[node]]), (1, Timesteps, 1)) # kg/m3
            Dem[node:node+1, :,file:file+1] = np.reshape(np.array(results.node['demand'][DemandNodes[node]]), (1, Timesteps, 1)) # m3/s
            
    QualFull = Qual
    DemFull = Dem*Factor # Correct for rescaling
    Qual = QualFull[:,int((TimeProperties['Duration']-ConsumptionProperties['Pattern Duration'])/TimeProperties['Pattern Timestep'])+1:,:] # HARDCODING - Use only second week of simulation
    Dem = DemFull[:,int((TimeProperties['Duration']-ConsumptionProperties['Pattern Duration'])/TimeProperties['Pattern Timestep'])+1:,:] # HARDCODING - Use only second week of simulation

    RunSettings = dict()
    RunSettings['Files']=Files
    RunSettings['ConsumptionProperties']=ConsumptionProperties
    RunSettings['TimeProperties']=TimeProperties
    RunSettings['QualityProperties']=QualityProperties
    RunSettings['LeachingProperties']=LeachingProperties
    RunSettings['Factor']=Factor
    RunSettings['ComputationalTime']=int(time.time() - startTime)
    RunSettings['Date']=datetime.now()
    RunSettings['Version']='2024.09.04'
    
    np.savez(Files['Output Directory']+Files['Output Prefix'], Quality=Qual, Demand=Dem)
    with open(Files['Output Directory']+Files['Output Prefix']+'Settings.pkl', 'wb') as fp:
        pickle.dump(RunSettings, fp)
    
    #pd.DataFrame(RunSettings).to_csv(Files['Output Directory']+Files['Output Prefix']+'.csv',index=False)
        
    print('Time elapsed: ' + str(int(time.time() - startTime)) + ' seconds')
    
# %% 

def CopyDemandFromExistingGeometry(InputFile,Junc_Input,Pattern_Input,
                                   OutputFile,Junc_Output,Pattern_Output):
    '''
    This function copies demand patterns from an existing input file and writes it to another input file where demand patterns do not exist.

    Parameters
    ----------
    InputFile : String
        This is a link to the EPANET input file from which water demand patterns are copied.
    Junc_Input : List of strings
        This is a list containing the names of all nodes in InputFile to which water demand is attributed.
    Pattern_Input : List of strings
        Each node has a certain demand pattern attributed to it. This variable contains a list with the names of the demand patterns in InputFile. The value at location X in this list corresponds to location X in the list Junc_Input
    OutputFile : String
        This is a link to the EPANET input file to which water demand patterns are copied.
    Junc_Output : List of strings
        This is a list containing the names of all nodes in OutputFile to which water demand is to be attributed.
    Pattern_Output : List of strings
        Each node has a certain demand pattern attributed to it. This variable contains a list with the names of the demand patterns in OutputFile. The value at location X in this list corresponds to location X in the list Junc_Output

    Returns
    -------
    None.

    '''
        
    wnin = wntr.network.WaterNetworkModel(InputFile)
    wnout = wntr.network.WaterNetworkModel(OutputFile)
    
    for node in range(len(Junc_Input)):
        wnout.add_pattern(Pattern_Output[node], wnin.get_pattern(Pattern_Input[node]).multipliers) # Add a pattern to the model with the name of the node.
        wnout.get_node(Junc_Output[node]).add_demand(base=1, pattern_name = Pattern_Output[node]) # Add demand pattern to the node

    for junc_name,junc in wnout.junctions():
        junc.demand_timeseries_list[0].base_value = 0 # The patterns are added as the second index of the demand time series and the first one is turned to zero
        
    wnout.write_inpfile(OutputFile)
    
# %%

def MWE_CopyDemandFromExistingGeometry():
    NumberOfInputFiles = 1
    InputFile = 'Original\\CopyExample\\Input\\Example'    
    OutputFile = 'Original\\CopyExample\\Output\\Output_'

    for file in range(NumberOfInputFiles):
        inp_loc = InputFile + '%02d.inp' % (file+1)
        out_loc = OutputFile + '%02d.inp' % (file+1)
        CopyDemandFromExistingGeometry(inp_loc,
                                       ExtractDemandNodes(inp_loc),
                                       ['SIMDEUM_'+m for m in ExtractDemandNodes(inp_loc)],
                                       out_loc,
                                       ExtractDemandNodes(inp_loc),
                                       ['SIMDEUM_'+m for m in ExtractDemandNodes(inp_loc)])
        
# %% 

def LoadOutput(OutputFile):
    '''
    This function loads previously saved files into the workspace.

    Parameters
    ----------
    OutputFile : String
        This is a link to the location where all the output files will be saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Two files will be saved with names “XYZ.npz” and “XYZSettings.pkl”.

    Returns
    -------
    Demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    Quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    RunSettings : Dictionary
        Contains all fields used to run the corresponding simulations

    '''
    data = np.load(OutputFile+".npz", allow_pickle=True)
    Demand = data['Demand']
    Quality = data['Quality']
    with open(OutputFile+'Settings.pkl', 'rb') as fp:
        RunSettings = pickle.load(fp)
    
    return Demand, Quality, RunSettings

# %% 

def NPZtoXLSX(OutputFile): 
    '''
    This function converts the outputs of a previous simulation (saved in npz format) to a CSV format.

    Parameters
    ----------
    OutputFile : String
        This is a link to the location where the previously generated output files have been saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Ideally, there should already be two files with names “XYZ.npz” and “XYZSettings.pkl”.
    
    Returns
    -------
    None.

    '''
    
    Demand, Quality, RunSettings = LoadOutput(OutputFile)
    NumberOfInputFiles = RunSettings['Files']['Number of files']
    DemandNodes = ExtractDemandNodes(RunSettings['Files']['Input Directory']+RunSettings['Files']['Input Prefix']+'01.inp')
    
    with pd.ExcelWriter(OutputFile+'.xlsx') as writer:
        for file in range(NumberOfInputFiles):
            print('Dumping data ' + str(file+1) + '/' + str(NumberOfInputFiles))
            df_DumpQual = pd.DataFrame(np.transpose(Quality[:,:,file]))
            df_DumpDem = pd.DataFrame(np.transpose(Demand[:,:,file]))
            df_DumpQual.to_excel(writer,sheet_name='Qual'+'%02d.inp' % (file+1),
                                 header=DemandNodes)
            df_DumpDem.to_excel(writer,sheet_name='Dem'+'%02d.inp' % (file+1),
                                 header=DemandNodes)
            
# %%

def PKLtoCSV(OutputFile):
    '''
    This function converts the run settings of a previous simulation (saved in pkl format) to a CSV format.

    Parameters
    ----------
    OutputFile : String
        This is a link to the location where the previously generated output files have been saved. The format is “Directory\\Subdirectory\\Subsubdirectory\\XYZ”. Ideally, there should already be two files with names “XYZ.npz” and “XYZSettings.pkl”.
    
    Returns
    -------
    None.

    '''
    
    with open(OutputFile+"Settings.pkl", "rb") as f:
        object = pickle.load(f)
    
    df = pd.DataFrame(object)
    df.to_csv(OutputFile+"Settings.csv")
    
# %% 

def ComputeTapOutputs(Dem,Qual,timestep):
    '''
    This function calculates the total mass water, total mass metal and average concentration metal at each consumption point. 

    Parameters
    ----------
    Dem : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    Qual : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    timestep : Integer
        Timestep with which water demand patterns have been generated.

    Returns
    -------
    TapOutputs : Dictionary
        Contains three arrays - one for total mass water, one for total mass metal, one for average concentration metal.
        Each array is two dimensional. First dimension for all consumption points. Second dimension for all input files with unique water consumption patterns.
    '''
    QualOutput = np.multiply(np.where(Dem>0, 1, 0), Qual)    # Quality of water collected [kg/m3]
    TotalWater = np.trapz(Dem, dx=timestep,axis=1)*(10**3) # Total water consumed per week per node [L]
    TotalMetal = np.trapz(QualOutput*Dem, dx=timestep, axis=1)*(10**9) # Total lead consumed per week per node [ug]
    TotalConcentration = TotalMetal/TotalWater
    
    TapOutputs = dict()
    TapOutputs['TotalWater']=TotalWater
    TapOutputs['TotalMetal']=TotalMetal
    TapOutputs['TotalConcentration']=TotalConcentration
    
    return TapOutputs

# %% 

def ComputeCDF(Dem,Qual,node,plot):
    '''
    This function plots the cumulative distribution of metal concentration at a consumption point.
    The plot is based on the compilation of all input files.

    Parameters
    ----------
    Dem : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    Qual : Array of floats
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
    CDFproperties : Array of floats
        A two dimensional array which contains properties of the cumulative distribution function
        Each array is two dimensional. First dimension for all consumption points. Second dimension for all input files with unique water consumption patterns.
    '''
    DemCDF = (np.squeeze(Dem[node,:,:])).flatten() # For node make CDF of all weeks
    QualCDF = (np.squeeze(Qual[node,:,:])).flatten() # For node make CDF of all weeks
    ConsumptionCDF = QualCDF[np.where(DemCDF > 0)[0]] 
    xconsumption = np.sort(ConsumptionCDF)*1000000
    yconsumption = np.arange(len(ConsumptionCDF)) / float(len(ConsumptionCDF))
    CDFproperties = [xconsumption,yconsumption]
    
    if plot == True:
        fig, axs = plt.subplots()
        plt.rc('figure', titlesize=18)
        plt.rc('axes', titlesize=18)
        plt.rc('axes', labelsize=18)
        plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
    
        axs.plot(xconsumption, yconsumption, linestyle='none', marker='o', linewidth=1, markerfacecolor="black", markeredgecolor="none")
        axs.set_xlabel('Metal Concentration [$\mu$g/L]')
        axs.set_ylabel('Probability [-]')
        axs.grid(True)
        axs.set_xscale('log')
        axs.set_aspect(3)
        
        figManager = plt.get_current_fig_manager()
        #figManager.window.showMaximized()

    return CDFproperties

# %%

def ComputeFractionExceedance(Dem,Qual,threshold):
    '''
    This function calculates the fraction of timesteps where the metal concentration at the tap exceeds a threshold value.
    The fraction is based on the total number of timesteps for which the consumption point is active and there is no correction for the volumetric flow rate.
    The fractions are for all input files combined.

    Parameters
    ----------
    Dem : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    Qual : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns.
    threshold : Float
        This is the threshold concentration value on basis whereof the fractions are calculated.

    Returns
    -------
    FractionExceedance : Array of float
        The fraction of times the concentration at the consumption points exceeds a threshold value.
        The array represents all the consumption points.
    '''
    QualOutput = np.multiply(np.where(Dem>0, 1, 0), Qual)    # Quality of water collected [kg/m3]
    QualOutputGood = np.multiply(np.where(QualOutput<threshold, 1, 0), QualOutput)    # Quality of water collected
    QualOutputBad = np.multiply(np.where(QualOutput>=threshold, 1, 0), QualOutput)    # Quality of water collected 
    
    InstancesQualGood = np.sum(np.where(QualOutputGood>0, 1, 0), axis=(1, 2))
    InstancesQualBad = np.sum(np.where(QualOutputBad>0, 1, 0), axis=(1, 2))

    FractionExceedance = InstancesQualBad/(InstancesQualBad+InstancesQualGood)
    return FractionExceedance

# %%

def PlotDemandAndQuality(Dem,Qual,node,inputfile,threshold=5,
                         duration=86400*7,
                         timestep=10,
                         xlims=[0,7],
                         ylimsDemand=[-2*(10**-2),25*(10**-2)],
                         ylimsQuality=[0.01,100],
                         xticks=[0,1,2,3,4,5,6,7],
                         xticklabels=([0,1,2,3,4,5,6,7]),
                         formatTime='Days',
                        fontsize=16):
    '''
    This function plots the water demand and metal concentrations at a consumption point corresponding to a certain input file (one out of the multiple).
    Limited options are offered to tweak the visualization. However, the user can modify the visualization by modifying hardcoded choices.

    Parameters
    ----------
    Demand : Array of floats
        Contains water demand patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    Quality : Array of floats
        Contains metal concentration patterns. This array has three dimensions.
        First dimension - all consumption points
        Second dimension - all timesteps
        Third dimension - all files with unique water demand patterns
    node : Integer
        The index of the consumption point for which the plot is to be made.
        Refer to the key 'Consumption Points' in the dictionary 'ConsumptionProperties'.
    inputfile : Integer
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
    ylimsDemand : Array of floats, optional
        This sets the limits of the plot (vertical axis - demand).  
        The default is [-2*(10**-2),25*(10**-2)].
    ylimsQuality : Array of floats, optional
        This sets the limits of the plot (vertical axis - concentrations). Note that the plot is logarthmic scale.  
        The default is [0.01,100].
    xticks : Array of floats, optional
        This determines the locations where the ticks on the horizontal axis will appear. 
        The default is [0,1,2,3,4,5,6,7].
    xticklabels : Array  of floats, optional
        This determines the labels of the ticks on the horizontal axis. 
        The default is ([0,1,2,3,4,5,6,7]).
    formatTime : String, optional
        This determines whether time will be shown with units of days or hours:minutes:seconds.
        The input should be either 'Days' or 'HMS'.
        The default is 'Days'.
    fontsize : Integer, optional
        This determines the font sizes of the axes and the labels in the figures.
        The default is 16.

    Returns
    -------
    None.

    '''
    
    threshold = threshold/1000000
    QualOutput = np.multiply(np.where(Dem>0, 1, 0), Qual)    # Quality of water collected [kg/m3]
    QualOutputGood = np.multiply(np.where(QualOutput<threshold, 1, 0), QualOutput)    # Quality of water collected
    QualOutputBad = np.multiply(np.where(QualOutput>=threshold, 1, 0), QualOutput)    # Quality of water collected 

    fig, axs = plt.subplots(2, 1, sharex=True)
    plt.rc('figure', titlesize=fontsize)
    plt.rc('axes', titlesize=fontsize*0.75)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels
    plt.style.use('default')

    axs[0].set_xlim(xlims[0],xlims[1])
    axs[0].set_ylim(ylimsDemand[0],ylimsDemand[1])

    Usage = axs[0].stem(np.linspace(0,int(duration/86400),int(duration/timestep)), 1000*Dem[node,:,inputfile-1],linefmt = 'k-', markerfmt = 'bo', label="Usage")
    axs[0].set_ylabel('Water demand \n [liters per second]')
    axs[0].grid(True)
    axs[0].set_aspect('auto')
    if formatTime=='HMS':
        xfmt = mdates.DateFormatter('%H:%M:%S')
        axs[0].xaxis.set_major_formatter(xfmt)

    
    axs[1].plot(xlims,[threshold*1000000,threshold*1000000],':k',linewidth=2)
    WNTR, = axs[1].plot(np.linspace(0,int(duration/86400),int(duration/timestep)), 1000000*Qual[node,:,inputfile-1], linewidth=3, color="lightseagreen", alpha=0.2, label="Concentration at tap")
    Good, = axs[1].plot(np.linspace(0,int(duration/86400),int(duration/timestep)), 1000000*QualOutputGood[node,:,inputfile-1],linewidth=1, linestyle='none', marker='o', markeredgecolor="cornflowerblue" , markerfacecolor="cornflowerblue", markersize=10 , label="Concentration below threshold")
    Bad, = axs[1].plot(np.linspace(0,int(duration/86400),int(duration/timestep)), 1000000*QualOutputBad[node,:,inputfile-1],linewidth=1, linestyle='none', marker='o', markeredgecolor="lightcoral", markerfacecolor="lightcoral", markersize=10 , label="Concentration above threshold")
    
    if formatTime == 'Days':
        axs[1].set_xlabel('Time [days]')
        axs[1].set_xticks(xticks)
        axs[1].set_xticklabels(xticklabels)
    else:
        axs[1].set_xlabel('Time')
        axs[1].set_xticks(xticks)
    axs[1].set_ylim([ylimsQuality[0],ylimsQuality[1]])
    axs[1].set_ylabel('Metal \n concentration [$\mu$g/L]')
    axs[1].set_yscale('log')
    axs[1].grid(True)
    
    figManager = plt.get_current_fig_manager()
    #figManager.window.showMaximized()

    plt.rc('figure', titlesize=fontsize)
    plt.rc('axes', titlesize=fontsize*0.75)
    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsize)    # fontsize of the tick labels