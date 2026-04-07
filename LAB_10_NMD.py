# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 11:52:07 2026

@author: user
"""

import xarray as xr

data = r"C:\Users\user\Desktop\25CL05014-NMD\LAB_10_NMD\Data\sst.oisst.mon.mean.1982.nc"
ds = xr.open_dataset(data)
print(ds)


sst = ds['sst']
# Q1 
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd 

sst_tim = sst.mean(dim = ('lat','lon'))


sst_clim = sst_tim.groupby('time.month').mean(dim = 'time')

plt.figure(figsize = (12,8))
sst_clim.plot(label = 'SST')
plt.xlabel("Time")
plt.ylabel('SST')
plt.grid(True)
plt.legend()
plt.title('SST climatology')

#Q2 
year = [2005]

year_2005 = sst_tim.sel(time=slice('2005-01-01','2005-12-01'))
sst_anom = year_2005.groupby('time.month').mean(dim = 'time') - sst_clim



sst_anom.plot()

year_2005 = sst_tim.sel(time=slice('2005-01-01','2005-12-31'))
sst_clim = sst_tim.groupby('time.month').mean('time')
sst_anom = year_2005.groupby('time.month') - sst_clim
#sst_2005 = sst_anom.mean(dim=('lat','lon'))

sst_anom.plot()