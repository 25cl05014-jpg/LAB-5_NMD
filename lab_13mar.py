# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 10:37:27 2026

@author: user
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data1 = r"C:\Users\user\Desktop\Aaroksh_FWO\lab_13mar\geopot_temp.nc"
ds1 = xr.open_dataset(data1)
print(ds1)


ds1.data_vars
ds1['t'].long_name


time_1 = ds1['valid_time']
lat_1 = ds1['latitude']
long_1 = ds1['longitude']
geo = ds1['z']
temp = ds1['t']
temp_C = temp-273.15


# For Lapse rate , we will firstly find the heights ar respective pressure levels 

phi_850 = geo.sel(pressure_level = 850.0).mean(dim = ('valid_time','latitude','longitude'))
phi_500 = geo.sel(pressure_level = 500.0).mean(dim = ('valid_time','latitude','longitude'))

temp_850 = temp_C.sel(pressure_level = 850.0).mean(dim = ('valid_time','latitude','longitude'))
temp_500 = temp_C.sel(pressure_level = 500.0).mean(dim = ('valid_time','latitude','longitude'))
g = 9.81
z_850 = phi_850/g
z_500 = phi_500/g
print(z_850)

lapse_rate = -((temp_850-temp_500)/(z_850-z_500))

print(lapse_rate.values)

# For vertical temperature profile 

temp_2 = temp_C.mean(dim = ('valid_time','latitude','longitude'))
phi_2 = geo.mean(dim = ('valid_time','latitude','longitude'))
z_2 = phi_2/g


plt.plot(temp_2,z_2,label = 'Lapse Rate')
plt.xlabel("Temperature (in °C)")
plt.ylabel('Height (in m)')
plt.grid(True)
plt.title('Vertical Temperature profile')
plt.legend()


# For time series of Lapse rate 

phi_850_2 = geo.sel(pressure_level = 850.0).mean(dim = ('latitude','longitude'))
phi_500_2 = geo.sel(pressure_level = 500.0).mean(dim = ('latitude','longitude'))

temp_850_2 = temp_C.sel(pressure_level = 850.0).mean(dim = ('latitude','longitude'))
temp_500_2 = temp_C.sel(pressure_level = 500.0).mean(dim = ('latitude','longitude'))

z_850_2 = phi_850_2/g
z_500_2 = phi_500_2/g

lapse_rate_2 = -((temp_850_2-temp_500_2)/(z_850_2-z_500_2))



lapse_rate_2.plot(label = 'Lapse rate')
plt.xlabel("Time")
plt.ylabel("Lapse rate")
plt.grid(True)
plt.title('Time series plot of Lapse Rate in Monsoon period')
xs1 = pd.to_datetime(['2005-06-01', '2005-06-15', '2005-07-01', '2005-07-15', '2005-08-01', '2005-08-15', '2005-09-01', '2005-09-15', '2005-09-30'])
xs2 = ['01/06', '15/06', '01/07', '15/07', '01/08', '15/08', '01/09', '15/09', '30/09']
plt.xticks(xs1,xs2)
plt.legend()




'''
Q3
'''


data3 = r"C:\Users\user\Desktop\Aaroksh_FWO\lab_13mar\1991-2021_vws.nc"
ds3 = xr.open_dataset(data3)
print(ds3)


U = ds3['u']
V = ds3['v']


net_wind = np.sqrt(U**2+V**2)

wind_200 = net_wind.sel(level = 200).mean(dim = ('longitude','latitude','time'))
wind_850 = net_wind.sel(level = 850).mean(dim = ('longitude','latitude','time'))


wind_shear = wind_200-wind_850
print(wind_shear.values)


wind_200_2 = net_wind.sel(level = 200).mean(dim = ('longitude','latitude'))
wind_850_2 = net_wind.sel(level = 850).mean(dim = ('longitude','latitude'))

wind_shear_2 = wind_200_2-wind_850_2

plt.figure(figsize = (12,8))
wind_shear_2.plot(label = 'Wind Shear')
plt.xlabel("Time")
plt.ylabel("Wind Shear")
plt.grid(True)
plt.title('Time series plot of Wind Shear')
plt.legend()


'''
Q2
'''
import math
data2 = r"C:\Users\user\Desktop\Aaroksh_FWO\lab_13mar\1991-2021_p.nc"
ds2 = xr.open_dataset(data2)
print(ds2)

rel = ds2['r']

JJAS = [6,7,8,9]
# For temporal resolution
rel_JJAS = rel.sel(time =slice('2005-06-01','2005-09-01'),latitude = slice(28,29),longitude = slice(76.8,77.3),level = 850).mean(dim = ('latitude','longitude'))

temp_850_2 = temp_C.sel(pressure_level = 850.0).mean(dim = ('latitude','longitude'))
temp_850_3 = temp_850_2.groupby('valid_time.month')

a = (17.67*temp_850_3)/(243.5+temp_850_3)
es = 6.112*np.exp(a)


e = (rel_JJAS*es)/100


q = (0.622*e)/(850-0.378*e)
