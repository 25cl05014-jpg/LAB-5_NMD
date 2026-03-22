# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 18:40:29 2026

@author: hp
"""

import xarray as xr
 
data = r"E:\IIT\SEM_2\FWO\LAB_20MAR\subset_FANI_2019-04-16_to_2019-05-15\subset_FANI_2019-04-16_to_2019-05-15.nc"
ds = xr.open_dataset(data)
print(ds)


temp = ds['to']
salin = ds['so']
height = ds['zo']
ug = ds['ugo']
vg = ds['vgo']
lat = ds['latitude']
depth = ds['depth']
'''
Q1
'''
import cartopy.crs as cc
import cartopy.feature as cf
import matplotlib.pyplot as plt

temp_1 = temp.mean(dim =('depth','time')) - 25
salin_1 = salin.mean(dim = ('depth','time')) - 35

rho = 1025*(1-(2*10**-4)*(temp_1)+(8*10**-4)*(salin_1))

ax = plt.axes(projection = cc.Mercator())
sp = rho.plot.contourf(transform =cc.PlateCarree(),levels = 100,add_colorbar = False, extend = 'both')
ax.gridlines(draw_labels = True)
ax.add_feature(cf.BORDERS)              #HIGHLIGHTING THE BORDERS
ax.add_feature(cf.COASTLINE)            #HIGHLIGHTING THE COASTLINES
ax.add_feature(cf.LAND, color = 'gray')
plt.colorbar(sp,label = "Observed density",pad = 0.13)
plt.title('Spatial distribution Of Observed Density')


'''
Q2
'''
import numpy as np

Omega = 7.2921e-5        # Earth's rotation rate [rad/s]
rho_air = 1.225          # air density [kg/m^3]
Cd = 1.2e-3              # drag coefficient (approx.)
Re = 6.371e6             # Earth radius [m]

# -----------------------------
# 3. Compute Coriolis parameter f
# -----------------------------
f = 2 * Omega * np.sin(np.deg2rad(lat))

u = ug.sel(depth = slice(0,10)).mean(dim = ('time','depth'))
v = vg.sel(depth = slice(0,10)).mean(dim = ('time','depth'))


U = np.sqrt(u**2 + v**2)


tau_x = 1.2*(1.2*10**-3)*u*U
tau_y = 1.2*(1.2*10**-3)*v*U

M_X = tau_x/(f*rho_air)
M_Y = tau_y/(f*rho_air)


tau = np.sqrt(M_X**2 + M_Y**2)

ax = plt.axes(projection = cc.Mercator())
ax.gridlines(draw_labels = True)
sp1 = tau.plot.contourf(transform = cc.PlateCarree(),vmax = np.nanmax(tau),levels = 100,add_colorbar = False,extend = 'both')
ax.add_feature(cf.BORDERS)              #HIGHLIGHTING THE BORDERS
ax.add_feature(cf.COASTLINE)            #HIGHLIGHTING THE COASTLINES
ax.add_feature(cf.LAND, color = 'gray')
plt.colorbar(sp1,label = "Ekman Transport",pad = 0.13)
plt.title('Spatial distribution of Ekman Transport (in $m^2/s$)')



u_1= ug.sel(depth = slice(0,10)).mean(dim = ('latitude','longitude','depth'))
v_1 = vg.sel(depth = slice(0,10)).mean(dim = ('latitude','longitude','depth'))


U_1 = np.sqrt(u_1**2 + v_1**2)


tau_x_1 = 1.2*(1.2*10**-3)*u_1*U_1
tau_y_1 = 1.2*(1.2*10**-3)*v_1*U_1

M_X_1 = tau_x_1/(f*rho_air)
M_Y_1 = tau_y_1/(f*rho_air)

tau_1 = np.sqrt(M_X_1**2 + M_Y_1**2).mean(dim = 'latitude')

import pandas as pd
plt.figure(figsize = (22,6))
tau_1.plot()
xs = pd.date_range(start="2019-04-16", end="2019-05-15")
xs2 = xs.strftime("%d/%m")

#xs = pd.to_datetime(['2019-04-16', '2019-04-17', '2019-04-18', '2019-04-19', '2019-04-20', '2019-04-21', '2019-04-21', '2019-04-22', '2019-04-23','2019-04-24','2019-04-25','2019-04-26','2019-04-27','2019-04-16'])  # DEFINING THE INTERVALS IN TERMS OF DATES TAKEN 
#xs2 = ['01/06', '15/06', '01/07', '15/07', '01/08', '15/08', '01/09', '15/09', '30/09']                                                              # DEFINING THE TITLE FOR THOSE INTERVALS 
plt.xticks(xs,xs2)  #PUTTING THE TICKS TOGETHER
plt.title('Time series plot of Ekman Transport in year 2019')
plt.xlabel('Time')
plt.ylabel('Ekman Transport (in $m^2/s$)')
plt.grid(True)
plt.show()

'''
Q3
'''

rho_w = 1025 
cp = 4000

depth_1 = np.array([0, 5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60, 65,  70,  80,  90, 100, 125, 150, 175, 200, 225, 250, 275, 300,350, 400, 450, 500, 550, 600, 700])
temp_2 = temp.mean(dim = ('time','longitude','latitude'))

o = rho_w*cp*np.trapz(temp_2,depth_1)
