# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 09:57:08 2026

@author: user
"""

import xarray as xr
import numpy as np

data1 = r"C:\Users\user\Desktop\Aaroksh_FWO\LAB_27MAR\Flux_params_JJAS_BoB.nc"
ds1 = xr.open_dataset(data1)
print(ds1)


data2 = r"C:\Users\user\Desktop\Aaroksh_FWO\LAB_27MAR\RH_2024_JJAS_BoB.nc"
ds2 = xr.open_dataset(data2)
print(ds2)


u = ds1['u10']
v = ds1['v10']
Ts = ds1['sst'] - 273.15
Ta = ds1['t2m'] - 273.15
rel = ds2['r'].mean(dim = 'pressure_level')
time = ds1['valid_time']

U = np.sqrt(u**2 + v**2)


Ess = 6.112*np.exp((17.67*Ts)/(Ts + 243.5))
Esa = 6.112*np.exp((17.67*Ta)/(Ta + 243.5))


Es = (Ess*rel)/100
Ea = (Esa*rel)/100


qs = (0.622*Ess)/(1013-(0.378*Ess))
qa = (0.622*Esa)/(1013-(0.378*Esa))*(rel/100)


'''
a)
'''
SHF = 1.2*1004*(1.2*10**-3)*U*(Ts - Ta)
LHF = 1.2*(2.5*10**-6)*(1.2*10**-3)*U*(qs-qa)


'''
b)
'''
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as cc
import cartopy.feature as cf
SHF_tim_1 = SHF.mean(dim = ('latitude','longitude'))
LHF_tim_1 = LHF.mean(dim = ('latitude','longitude'))

plt.figure(figsize = (20,12))
plt.suptitle('Temporal variation of SHF and LHF in Monsoon 2024')

plt.subplot(2,1,1)
SHF_tim_1.plot(label = 'SHF')
xs1 = pd.to_datetime(['2024-06-01', '2024-06-15', '2024-07-01', '2024-07-15', '2024-08-01', '2024-08-15', '2024-09-01', '2024-09-15', '2024-09-30'])
xs2 = ['01/06', '15/06', '01/07', '15/07', '01/08', '15/08', '01/09', '15/09', '30/09']
plt.xticks(xs1,xs2)
plt.xlabel('Time')
plt.ylabel('Fluxes')
plt.grid(True)
plt.title('Temporal variation of SHF in Monsoon 2024')
plt.legend()


plt.subplot(2,1,2)
LHF_tim_1.plot(label = 'LHF')
xs1 = pd.to_datetime(['2024-06-01', '2024-06-15', '2024-07-01', '2024-07-15', '2024-08-01', '2024-08-15', '2024-09-01', '2024-09-15', '2024-09-30'])
xs2 = ['01/06', '15/06', '01/07', '15/07', '01/08', '15/08', '01/09', '15/09', '30/09']
plt.xticks(xs1,xs2)
plt.xlabel('Time')
plt.ylabel('Fluxes')
plt.grid(True)
plt.title('Temporal variation of LHF in Monsoon 2024')
plt.legend()

LHF_spat = LHF.mean(dim = 'valid_time')
SHF_spat = SHF.mean(dim = 'valid_time')
Jun = [6]
Jul = [7]
Aug = [8]
Sep = [9]


LHF_JUN = LHF.sel(valid_time = LHF['valid_time.month'].isin(Jun)).mean(dim = 'valid_time')
LHF_JUL = LHF.sel(valid_time = LHF['valid_time.month'].isin(Jul)).mean(dim = 'valid_time')
LHF_AUG = LHF.sel(valid_time = LHF['valid_time.month'].isin(Aug)).mean(dim = 'valid_time')
LHF_SEP = LHF.sel(valid_time = LHF['valid_time.month'].isin(Sep)).mean(dim = 'valid_time')

SHF_JUN = SHF.sel(valid_time = SHF['valid_time.month'].isin(Jun)).mean(dim = 'valid_time')
SHF_JUL = SHF.sel(valid_time = SHF['valid_time.month'].isin(Jul)).mean(dim = 'valid_time')
SHF_AUG = SHF.sel(valid_time = SHF['valid_time.month'].isin(Aug)).mean(dim = 'valid_time')
SHF_SEP = SHF.sel(valid_time = SHF['valid_time.month'].isin(Sep)).mean(dim = 'valid_time')



plt.figure(figsize = (20,12))
plt.suptitle('Spatial varitation of LHF during Monsoon season')

ax = plt.subplot(2,2,1 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
q1 = LHF_JUN.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(q1 , label = 'LHF',pad = 0.07)
plt.title('LHF in June')

ax = plt.subplot(2,2,2 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
q2 = LHF_JUL.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(q2 , label = 'LHF',pad = 0.07)
plt.title('LHF in July')

ax = plt.subplot(2,2,3 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
q3 = LHF_AUG.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(q3 , label = 'LHF',pad = 0.07)
plt.title('LHF in August')

ax = plt.subplot(2,2,4 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
q4 = LHF_SEP.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(q4 , label = 'LHF',pad = 0.07)
plt.title('LHF in September')









plt.figure(figsize = (20,12))
plt.suptitle('Spatial varitation of SHF during Monsoon season')

ax = plt.subplot(2,2,1 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
S1 = SHF_JUN.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(S1 , label = 'SHF',pad = 0.07)
plt.title('SHF in June')

ax = plt.subplot(2,2,2 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
S2 = SHF_JUL.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(S2 , label = 'SHF',pad = 0.07)
plt.title('SHF in July')

ax = plt.subplot(2,2,3 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
S3 = SHF_AUG.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(S3 , label = 'SHF',pad = 0.07)
plt.title('SHF in August')

ax = plt.subplot(2,2,4 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
S4 = SHF_SEP.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(S4 , label = 'SHF',pad = 0.07)
plt.title('SHF in September')

'''
c)
'''

LHF_hour = LHF_tim_1.groupby('valid_time.hour').mean(dim='valid_time')
SHF_hour = SHF_tim_1.groupby('valid_time.hour').mean(dim='valid_time')

plt.figure(figsize = (20,12))
plt.suptitle('Temporal variability of LHF and SHF in hourly mean')

plt.subplot(2,1,1)
LHF_hour.plot()
plt.xlabel('Time')
plt.ylabel('LHF')
plt.grid(True)
plt.title('Temporal variation of LHF (hourly mean)')
plt.legend()

plt.subplot(2,1,2)
SHF_hour.plot()
plt.xlabel('Time')
plt.ylabel('SHF')
plt.grid(True)
plt.title('Temporal variation of SHF (hourly mean)')
plt.legend()



'''
Q2

a)
'''

B = SHF/LHF


B_tim = B.mean(dim = ('latitude','longitude'))
B_tim_roll = B_tim.rolling(valid_time=10, center=True).mean()

plt.figure(figsize = (20,12))
B_tim_roll.plot(label = 'Bowen\'s Ratio')
xs1 = pd.to_datetime(['2024-06-01', '2024-06-15', '2024-07-01', '2024-07-15', '2024-08-01', '2024-08-15', '2024-09-01', '2024-09-15', '2024-09-30'])
xs2 = ['01/06', '15/06', '01/07', '15/07', '01/08', '15/08', '01/09', '15/09', '30/09']
plt.xticks(xs1,xs2)
plt.xlabel('Time')
plt.ylabel('B value')
plt.grid(True)
plt.title('Temporal variation of Bowen\'s ratio')
plt.legend()

'''
b)
'''

B_spat = B.mean(dim = 'valid_time')



plt.figure(figsize = (12,8))
ax = plt.axes(projection = cc.Mercator())
ax.gridlines(draw_labels = True)
Q3 = B_spat.plot.contourf(transform = cc.PlateCarree(),levels = 100, add_colorbar = False, extend = 'both')
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(Q3 , label = 'B value',pad = 0.07)
plt.title('Spatial variation of B value')


'''
Q3

a)
'''


LHF_hov_10 = LHF.sel(latitude = 10,longitude = slice(86,93))#.mean(dim = (''))
LHF_hov_20 = LHF.sel(latitude = 20,longitude = slice(86,93))


SHF_hov_10 = SHF.sel(latitude = 10,longitude = slice(86,93))
SHF_hov_20 = SHF.sel(latitude = 20,longitude = slice(86,93))

SHF_hov_10.plot()
LHF_hov_10.plot()
LHF_hov_20.plot()
