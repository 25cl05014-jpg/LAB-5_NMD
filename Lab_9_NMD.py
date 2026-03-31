# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:42:34 2026

@author: user
"""


'''
Q1
'''
import xarray as xr
import numpy as np
data = r"C:\Users\user\Desktop\25CL05014-NMD\lab_23mar\coads_climatology.cdf"
ds = xr.open_dataset(data,decode_times=False)
print(ds)

data1 = r"C:\Users\user\Desktop\25CL05014-NMD\lab_23mar\esku_heat_budget.cdf"
ds1 = xr.open_dataset(data1,decode_times=False)
print(ds1)



temp = ds['SST']
press = ds['SLP']
air_temp = ds['AIRT']
w_speed = ds['WSPD']
qa_1 = ds['SPEH']*1e-3
'''
Q2
'''
es = 6.112*np.exp((17.67*temp)/(243.5+temp))

'''
Q3
'''

qs = (0.622*es)/(press - 0.378*es)

'''
Q4
'''
ea = 6.112*np.exp((17.67*air_temp)/(243.5+air_temp))


qa = ((0.622*ea)/(press - 0.378*ea))


rho_air = 1.2 
Lv = 2.5*10**6
Ce = 1.2*10**-3
   # converts TIME into proper datetime64
lhf = rho_air*Lv*Ce*w_speed*(qs-qa_1)

'''
Q5
'''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cartopy.crs as cc
import cartopy.feature as cf
lhf_5 = lhf.sel(COADSY = slice(-30,30),COADSX = slice(30,120)).mean(dim = ('COADSX','COADSY'))


plt.figure(figsize = (12,8))
lhf_5.plot(label = 'LHF')
'''
xs1 = pd.date_range(start="1970-01-01", end="1970-12-31", freq="15D")
xs2 = xs1.strftime("%d/%m")
plt.xticks(xs1,xs2)
'''
xs = np.arange(366,8401,700)
xs1 = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
plt.xticks(xs,xs1)
plt.xlabel('Time')
plt.ylabel('LHF')
plt.grid(True)
plt.title('Temporal variation of LHF over Tropical Indian Ocean')
plt.legend()

lhf_5_spat = lhf.sel(COADSY = slice(-30,30),COADSX = slice(30,120)).mean(dim = 'TIME')
plt.figure(figsize = (12,8))
ax = plt.axes(projection = cc.Mercator())
sp1 = lhf_5_spat.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar= False,extend = 'both')
plt.colorbar(sp1, label = 'LHF')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)
ax.gridlines(draw_labels = False)
plt.title('Spatial variability of LHF')

'''
Q6
'''

Cp = 1004


SHF = rho_air*Cp*Ce*w_speed*(temp-air_temp)


SHF_6 = SHF.sel(COADSY = slice(-30,30),COADSX = slice(30,120)).mean(dim = ('COADSX','COADSY'))
SHF_6.plot()
