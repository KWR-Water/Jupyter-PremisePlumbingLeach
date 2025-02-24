# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:25:04 2025

@author: dasham
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel('D:\\Users\\dasham\\Documents\\Projects\\PARC\\'+
                   'Simulator\\LeadGo_allTaps.xlsx',
                   sheet_name='shift_10sec')

sns.set_style(style='whitegrid')

fig, ax = plt.subplots()
plt.rc('figure', titlesize=24)
plt.rc('axes', titlesize=24)
plt.rc('axes', labelsize=24)
plt.rc('xtick', labelsize=24)    # fontsize of the tick labels
plt.rc('ytick', labelsize=24)    # fontsize of the tick labels
plt.style.use('default')

sns.scatterplot(
    data=df, x='Unnamed: 3', y='Unnamed: 4',
    hue='Unnamed: 2',s=200
    )

ax.plot([0.1,0.1],[100,100],'--k')

# plt.title('Exploring Physical Attributes of Different Penguins')
ax.set_xlabel('Lead concentration EPANET-MSX [$\mu$g/L]',fontsize=24)
ax.set_ylabel('Lead concentration LEadGO [$\mu$g/L]',fontsize=24)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.set_aspect('equal')
ax.tick_params(axis='both', labelsize=16)
plt.plot([0.1,100],[0.1,100],'--k',linewidth=5)
plt.plot([0.1,100],[0.1*1.5,100*1.5],':k',linewidth=3)
plt.plot([0.1,100],[0.1*0.5,100*0.5],':k',linewidth=3)

plt.show()