# -*- coding: utf-8 -*-
"""
Created on Mon Feb  9 14:36:42 2026

@author: user
"""
import xarray as xr
import cartopy.crs as cc
import cartopy.feature as cf
import matplotlib.pyplot as plt 
import numpy as np

data1 = r"C:\Users\user\Desktop\25CL05014-NMD\lab5_9feb\winds_io.nc"
data2 = r"C:\Users\user\Desktop\25CL05014-NMD\lab5_9feb\currents_io.nc"

ds_wind = xr.open_dataset(data1)
ds_cur = xr.open_dataset(data2)

print(ds_wind)

u = ds_wind['u']
v = ds_wind['v']
U_comp =u.mean(dim = ('valid_time','pressure_level'))
V_comp =v.mean(dim = ('valid_time','pressure_level'))
lat = ds_wind['latitude']
lon = ds_wind['longitude']
net_w = ((u)**2 + (v)**2)**0.5


lon_2,lat_2 = np.meshgrid(lon,lat)
skip = 4
lon_q = lon_2[::skip,::skip]
lat_q = lat_2[::skip,::skip]

u_q = U_comp[::skip,::skip]
v_q = V_comp[::skip,::skip]

wind = net_w.mean(dim = ('valid_time','pressure_level'))
plt.figure(figsize = (12,8))
ax = plt.axes(projection = cc.Mercator())
cs = wind.plot.contourf(ax=ax,transform = cc.PlateCarree(), cmap = 'Blues',levels = 180)
ax.gridlines(draw_labels = True)
ax.add_feature(cf.LAND, color = 'gray')
ax.add_feature(cf.BORDERS)              #HIGHLIGHTING THE BORDERS
ax.add_feature(cf.COASTLINE)            #HIGHLIGHTING THE COASTLINES

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q), np.ravel(v_q), transform=cc.PlateCarree(), scale=100)
ax.quiverkey(q,1.3,0.5,0.5, label = '0.2 m/s',labelpos = 'E')

'''
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import numpy as np

u = ds_wind['u']
v = ds_wind['v']

U_comp = u.mean(dim=('valid_time','pressure_level'))
V_comp = v.mean(dim=('valid_time','pressure_level'))

lat = ds_wind['latitude']
lon = ds_wind['longitude']

net_w = ((u)**2 + (v)**2)**0.5
wind = net_w.mean(dim=('valid_time','pressure_level'))

lon_2, lat_2 = np.meshgrid(lon, lat)
skip = 10
lon_q = lon_2[::skip, ::skip]
lat_q = lat_2[::skip, ::skip]
u_q = U_comp[::skip, ::skip]
v_q = V_comp[::skip, ::skip]

plt.figure(figsize=(12,8))
ax = plt.axes(projection=ccrs.Mercator())

cs1 = wind.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                         cmap='cubehelix_r', levels=180)

ax.gridlines(draw_labels=True)
ax.add_feature(cf.LAND, color='gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q), np.ravel(v_q), transform=ccrs.PlateCarree(), scale=40)
ax.quiverkey(q, 0.9, 0.1, 0.5, label='0.5 m/s',labelpos='E')

plt.colorbar(cs1, ax=ax, orientation='horizontal', label='Wind Speed (m/s)')
plt.title("Seasonal Wind Speed and Vectors")
plt.show()
'''
'''
Q2
'''
import cartopy.crs as cc
import cartopy.feature as cf
import matplotlib.pyplot as plt 
import numpy as np

DJF = [12,1,2]
MAM = [3,4,5]
JJAS = [6,7,8,9]
ON = [10,11]

wind_djf = net_w.sel(valid_time = net_w['valid_time.month'].isin(DJF)).mean(dim=('valid_time','pressure_level'))
wind_mam = net_w.sel(valid_time = net_w['valid_time.month'].isin(MAM)).mean(dim=('valid_time','pressure_level'))
wind_jjas = net_w.sel(valid_time = net_w['valid_time.month'].isin(JJAS)).mean(dim=('valid_time','pressure_level'))
wind_on = net_w.sel(valid_time = net_w['valid_time.month'].isin(ON)).mean(dim=('valid_time','pressure_level'))


plt.figure(figsize = (20,12))
plt.suptitle('Seasonal plot for Wind speed Intensity')

ax = plt.subplot(2,2,1, projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs1 = wind_djf.plot.contourf(transform = cc.PlateCarree(),cmap = 'viridis',levels = 200, extend = 'max',add_colorbar = False) 
ax.add_feature(cf.LAND)
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)
plt.colorbar(cs1,label = "Wind speed in (m/s)",pad = 0.07)
plt.title('Wind speed in Winters')



U_comp_djf =u.sel(valid_time = net_w['valid_time.month'].isin(DJF)).mean(dim = ('valid_time','pressure_level'))
V_comp_djf =v.sel(valid_time = net_w['valid_time.month'].isin(DJF)).mean(dim = ('valid_time','pressure_level'))

u_q_djf = U_comp_djf[::skip,::skip]
v_q_djf = V_comp_djf[::skip,::skip]

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q_djf), np.ravel(v_q_djf), transform=cc.PlateCarree(), scale=100)
ax.quiverkey(q,1.3,0.5,0.5, label = ' ',labelpos = 'E')



ax = plt.subplot(2,2,2, projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs2 = wind_mam.plot.contourf(transform = cc.PlateCarree(),cmap = 'viridis',levels = 200, extend = 'max',add_colorbar = False) 
ax.add_feature(cf.LAND)
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)
plt.colorbar(cs2,label = "Wind speed in (m/s)",pad = 0.07)
plt.title('Wind speed in Summer')


U_comp_mam =u.sel(valid_time = net_w['valid_time.month'].isin(MAM)).mean(dim = ('valid_time','pressure_level'))
V_comp_mam =v.sel(valid_time = net_w['valid_time.month'].isin(MAM)).mean(dim = ('valid_time','pressure_level'))

u_q_mam = U_comp_mam[::skip,::skip]
v_q_mam = V_comp_mam[::skip,::skip]

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q_mam), np.ravel(v_q_mam), transform=cc.PlateCarree(), scale=100)
ax.quiverkey(q,1.3,0.5,0.5, label = ' ',labelpos = 'E')




ax = plt.subplot(2,2,3, projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs3 = wind_jjas.plot.contourf(transform = cc.PlateCarree(),cmap = 'viridis',levels = 200, extend = 'max',add_colorbar = False) 
ax.add_feature(cf.LAND)
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)
plt.colorbar(cs3,label = "Wind speed in (m/s)",pad = 0.07)
plt.title('Wind speed in Monsoon')


U_comp_jjas =u.sel(valid_time = net_w['valid_time.month'].isin(JJAS)).mean(dim = ('valid_time','pressure_level'))
V_comp_jjas =v.sel(valid_time = net_w['valid_time.month'].isin(JJAS)).mean(dim = ('valid_time','pressure_level'))

u_q_jjas = U_comp_jjas[::skip,::skip]
v_q_jjas = V_comp_jjas[::skip,::skip]

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q_jjas), np.ravel(v_q_jjas), transform=cc.PlateCarree(), scale=300)
ax.quiverkey(q,1.3,0.5,0.5, label = ' ',labelpos = 'E')




ax = plt.subplot(2,2,4, projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs4 = wind_on.plot.contourf(transform = cc.PlateCarree(),cmap = 'viridis',levels = 200, extend = 'max',add_colorbar = False) 
ax.add_feature(cf.LAND)
ax.add_feature(cf.COASTLINE)
ax.add_feature(cf.BORDERS)
plt.colorbar(cs4,label = "Wind speed in (m/s)",pad = 0.07)
plt.title('Wind speed in Post Monsoon')


U_comp_on =u.sel(valid_time = net_w['valid_time.month'].isin(ON)).mean(dim = ('valid_time','pressure_level'))
V_comp_on =v.sel(valid_time = net_w['valid_time.month'].isin(ON)).mean(dim = ('valid_time','pressure_level'))

u_q_on = U_comp_on[::skip,::skip]
v_q_on = V_comp_on[::skip,::skip]

q = ax.quiver(np.ravel(lon_q),np.ravel(lat_q), np.ravel(u_q_on), np.ravel(v_q_on), transform=cc.PlateCarree(), scale=100)
ax.quiverkey(q,1.3,0.5,0.5, label = ' ',labelpos = 'E')


'''
Q3
'''
print(ds_cur)

u_oras = ds_cur['uo_oras']
v_oras = ds_cur['vo_oras']
lat_c = ds_cur['latitude']
lon_c = ds_cur['longitude']

net_c = ((u_oras)**2 + (v_oras)**2)

lon_c2,lat_c2 = np.meshgrid(lon_c,lat_c)
skip = 5
lon_q2 = lon_c2[::skip,::skip]
lat_q2 = lat_c2[::skip,::skip]

cur_djf = net_c.sel(time = net_c['time.month'].isin(DJF)).mean(dim=('time','depth'))
cur_mam = net_c.sel(time = net_c['time.month'].isin(MAM)).mean(dim=('time','depth'))
cur_jjas = net_c.sel(time = net_c['time.month'].isin(JJAS)).mean(dim=('time','depth'))
cur_on = net_c.sel(time = net_c['time.month'].isin(ON)).mean(dim=('time','depth'))


U_comp_DJF = u_oras.sel(time = net_c['time.month'].isin(DJF)).mean(dim = ('time','depth'))
U_comp_MAM = u_oras.sel(time = net_c['time.month'].isin(MAM)).mean(dim = ('time','depth'))
U_comp_JJAS = u_oras.sel(time = net_c['time.month'].isin(JJAS)).mean(dim = ('time','depth'))
U_comp_ON = u_oras.sel(time = net_c['time.month'].isin(ON)).mean(dim = ('time','depth'))


V_comp_DJF = v_oras.sel(time = net_c['time.month'].isin(DJF)).mean(dim = ('time','depth'))
V_comp_MAM = v_oras.sel(time = net_c['time.month'].isin(MAM)).mean(dim = ('time','depth'))
V_comp_JJAS = v_oras.sel(time = net_c['time.month'].isin(JJAS)).mean(dim = ('time','depth'))
V_comp_ON = v_oras.sel(time = net_c['time.month'].isin(ON)).mean(dim = ('time','depth'))


plt.figure(figsize = (20,12),dpi = 600)
plt.suptitle("Seasonal Variation of Oceanic Currents")


ax = plt.subplot(2,2,1 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs1_c = cur_djf.plot.contourf(transform = cc.PlateCarree(),cmap = 'cool',vmin = 0,vmax = 0.8,levels = 150, extend = 'max', add_colorbar = False,alpha = 0.7)
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(cs1_c , label = "Ocean currents [in m/s]",pad = 0.07)

u_c_djf = U_comp_DJF[::skip,::skip]
v_c_djf = V_comp_DJF[::skip,::skip]

q = ax.quiver(np.ravel(lon_q2),np.ravel(lat_q2),np.ravel(u_c_djf),np.ravel(v_c_djf),transform = cc.PlateCarree(),scale = 7)
ax.quiverkey(q,1.01,1.05,0.5,label = " 0.072 m/s ",labelpos = "E")



ax = plt.subplot(2,2,2 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs2_c = cur_mam.plot.contourf(transform = cc.PlateCarree(),cmap = 'cool',vmin = 0,vmax = 0.8,levels = 150, extend = 'max', add_colorbar = False,alpha = 0.7)
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(cs2_c , label = "Ocean currents [in m/s]",pad = 0.07)

u_c_mam = U_comp_MAM[::skip,::skip]
v_c_mam = V_comp_MAM[::skip,::skip]

q = ax.quiver(np.ravel(lon_q2),np.ravel(lat_q2),np.ravel(u_c_mam),np.ravel(v_c_mam),transform = cc.PlateCarree(),scale = 7)
ax.quiverkey(q,1.01,1.05,0.5,label = " 0.064 m/s ",labelpos = "E")



ax = plt.subplot(2,2,3 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs3_c = cur_jjas.plot.contourf(transform = cc.PlateCarree(),cmap = 'cool',vmin = 0,vmax = 0.8,levels = 150, extend = 'max', add_colorbar = False,alpha = 0.7)
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(cs3_c , label = "Ocean currents [in m/s]",pad = 0.07)

u_c_jjas = U_comp_JJAS[::skip,::skip]
v_c_jjas = V_comp_JJAS[::skip,::skip]

q = ax.quiver(np.ravel(lon_q2),np.ravel(lat_q2),np.ravel(u_c_jjas),np.ravel(v_c_jjas),transform = cc.PlateCarree(),scale = 7)
ax.quiverkey(q,1.01,1.05,0.5,label = " 0.079 m/s ",labelpos = "E")



ax = plt.subplot(2,2,4 , projection = cc.Mercator())
ax.gridlines(draw_labels = True)
cs4_c = cur_on.plot.contourf(transform = cc.PlateCarree(),cmap = 'cool',vmin = 0,vmax = 0.8,levels = 150, extend = 'max', add_colorbar = False,alpha = 0.7)
ax.add_feature(cf.LAND , color = 'gray')
ax.add_feature(cf.BORDERS)
ax.add_feature(cf.COASTLINE)
plt.colorbar(cs4_c , label = "Ocean currents [in m/s]",pad = 0.07)

u_c_on = U_comp_ON[::skip,::skip]
v_c_on = V_comp_ON[::skip,::skip]

q = ax.quiver(np.ravel(lon_q2),np.ravel(lat_q2),np.ravel(u_c_on),np.ravel(v_c_on),transform = cc.PlateCarree(),scale = 7)
ax.quiverkey(q,1.01,1.05,0.5,label = " 0.074 m/s ",labelpos = "E")