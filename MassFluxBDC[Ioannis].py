import numpy as N
import netCDF4 as S
import matplotlib.pyplot as plt
#import mpl_toolkits.basemap as bm
import math as M
import scipy.io as io
#from mpl_toolkits.basemap import Basemap
#import Basemap as bm
direcInput = "C:\Users\John\Desktop\Script_Ioannis\AnalysisBDC\ERA-I-Data_AnalysisIsentropicMassFlux" # set working directory
direcOutput = "C:\Users\John\Desktop\Script_Ioannis\AnalysisBDC\ERA-I-Data_AnalysisIsentropicMassFlux" # set directory containing plots of model results

pref = 100000.
omega = 0.00007292
dy = 1.0
g = 9.81
radius_earth = 6370000.0
pi = M.pi
Rd = 287.
cp = 1005.
kappa = Rd/cp


# resolution is 1.0 deg and 6 hours
dphi = 2.5
nx_len = 144  # -180 - +180   ; we use "nx" as index of longitude
ny_len = 37  # SP - NP    ; we use "ny" as index of latitude
nlevel_len = 10   #   265 K to 430 K
latnorth = 75.
lonwest = -180.
loneast = 180.
latsouth = -15.

year=2010
month="Jan"

# resolution is 1.0 deg and 6 hours
if month == "Jan": ntmon_len = 124  # 
if month == "Feb": ntmon_len = 112  # 
if month == "Mar": ntmon_len = 124  # 
if month == "Apr": ntmon_len = 120  # 
if month == "May": ntmon_len = 124  #  
if month == "Jun": ntmon_len = 120  # 
if month == "Jul": ntmon_len = 124  # 
if month == "Aug": ntmon_len = 124  # 
if month == "Sep": ntmon_len = 120  # 
if month == "Oct": ntmon_len = 124  # 
if month == "Nov": ntmon_len = 120  # 
if month == "Dec": ntmon_len = 124  # 

if month == "Feb" and year==2016: ntmon_len = 116  #
if month == "Feb" and year==2012: ntmon_len = 116  #
if month == "Feb" and year==2008: ntmon_len = 116  #
if month == "Feb" and year==2004: ntmon_len = 116  #
if month == "Feb" and year==2000: ntmon_len = 116  #
if month == "Feb" and year==1996: ntmon_len = 116  #
if month == "Feb" and year==1992: ntmon_len = 116  #
if month == "Feb" and year==1988: ntmon_len = 116  #
if month == "Feb" and year==1984: ntmon_len = 116  #
if month == "Feb" and year==1980: ntmon_len = 116  #

nt_len = 124*7

title = 'mass below 285 K - isentrope'

dy = radius_earth * pi * dphi / 180.
dx = N.zeros((ny_len), dtype='d')
latitude = N.zeros((ny_len), dtype='d')
for ny in range(ny_len):
 latitude[ny] = (latnorth - (ny * dphi)) #latitude in degrees
 dx[ny] = dy * M.cos(pi * latitude[ny] / 180.)

#open file

a = S.Dataset("C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Desktop\data\January\ERA-I_vp265-430K_Jan_4xd_2.5deg_15S-75N1.nc", mode='r') #ERA-Int data of v along a zonal strip between fixed latitudes
b = S.Dataset("C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Desktop\data\January\ERA-I_psurfT2mv10m_Jan_4xd_2.5deg_15S-75N1.nc", mode='r') #ERA-Int data of v along a zonal strip between fixed latitudes

#create arrays time, lat and lon for plotting
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
time = nt_len * (time - N.min(time)) / ( N.max(time) - N.min(time)) + 1
theta = a.variables["level"][:]

Theta_axis = N.zeros(nlevel_len, dtype='d')
Theta_axis_p = N.zeros(nlevel_len, dtype='d')   # to plot pressure
for nlevel in range(nlevel_len):
 Theta_axis_p[nlevel] = theta[nlevel]
 if nlevel == 0: Theta_axis[nlevel] = theta[nlevel] - 5.
 if nlevel > 0: Theta_axis[nlevel] = (theta[nlevel-1] + theta[nlevel])/2
 #print (nlevel, Theta_axis[nlevel], theta[nlevel], (1. /(theta[nlevel] - theta[nlevel-1])))

v = a.variables["v"][:,:,:]   #time, level, lat, lon
p = a.variables["pres"][:,:,:,:] 
ps = b.variables["sp"][:,:,:]
t2m = b.variables["t2m"][:,:,:]
v10m = b.variables["v10"][:,:,:]

#indices: time, lat, lon 
#axis: 0, 1, 2

# massflux = N.zeros((nt_len,nlevel_len,ny_len,nx_len), dtype='d')
# thetas = N.zeros((nt_len,ny_len,nx_len), dtype='d')

# Compute mass-flux below theta=275 K
# nlevel = 0 corresponds to 265 K-level
# 
# for nt in range(nt_len):
#     for nx in range(nx_len):
#         for ny in range(ny_len):
#             thetas[nt,ny,nx] = t2m[nt,ny,nx] * M.pow((pref/ps[nt,ny,nx]),kappa)
#             for nlevel in range(nlevel_len):
#                 if nlevel == 0:
#                     if thetas[nt,ny,nx] < theta[nlevel]: massflux[nt,nlevel,ny,nx] = (-1./g) * ((p[nt,nlevel,ny,nx] - ps[nt,ny,nx]) * (v[nt,nlevel,ny,nx] + v10m[nt,ny,nx]) / 2.) * (1. /(theta[nlevel] - thetas[nt,ny,nx]))
#                     if thetas[nt,ny,nx] >= theta[nlevel]: massflux[nt,nlevel,ny,nx] = 0.   #  [kg s^-1 m^-1 K^-1]
#                 if nlevel>0: massflux[nt,nlevel,ny,nx] = (-1./g) * ((p[nt,nlevel,ny,nx] - p[nt,nlevel-1,ny,nx]) * (v[nt,nlevel,ny,nx] + v[nt,nlevel-1,ny,nx]) / 2.) * (1. /(theta[nlevel] - theta[nlevel-1]))
#       

# massflux_zm = N.zeros((nt_len,nlevel_len,ny_len), dtype='d')
# massflux_zm_tm = N.zeros((nlevel_len,ny_len), dtype='d')
#indicate=N.arange(23,132)
#massflux1=N.delete(massflux,indicate,axis=3)
# indicate=N.arange(23,132)
# massflux1=N.delete(massflux,indicate,axis=3)
# massflux_zm[:,:,:] = N.mean(massflux1,axis=3)   # average along x-axis
#massflux_zm_tm[:,:] = N.mean(massflux_zm,axis=0)   # average along x-axis
#massflux_zm_thm=N.sum(massflux_zm,axis=1)
# massflux307=massflux[:,4,:,:]

#p_zm = N.zeros((nt_len,nlevel_len,ny_len), dtype='d')
#p_zm_tm = N.zeros((nlevel_len,ny_len), dtype='d')
#p1=N.delete(p,indicate,axis=3)
#p_zm[:,:,:] = N.mean(p[:,:,:,48:69],axis=3)   # average along x-axis
#p_zm_tm[:,:] = N.mean(p_zm,axis=0)   # average along x-axis

#thetas_zm = N.zeros((nt_len,ny_len), dtype='d')
#thetas_zm_tm = N.zeros((ny_len), dtype='d')
#thetas_zm[:,:] = N.mean(thetas, axis=2)
#thetas_zm_tm[:] = N.mean(thetas_zm, axis=0)

#print ('massflux_zm_tm[:,ny]')
#for ny in range(ny_len):
#  print (massflux_zm_tm[:,ny])

###########################  Plot zonal mean time mean mass flux

#plt.figure(figsize=(10,8))
#plt.axis([-15,75.,270,380])
#N.savetxt('MzmF1985.txt',massflux_zm_tm)
#N.savetxt('lat.txt',lat)
#N.savetxt('TpF.txt',Theta_axis_p)
#N.savetxt('TF.txt',Theta_axis)
#N.savetxt('PztJ5Atlantic.txt',p_zm_tm)
# N.savetxt('Mzm322J4Pacific.txt',massflux322)
#plt.xticks(N.arange(-15,80,15), fontsize=14)
#plt.yticks(N.arange(270,390.,10), fontsize=14)
#
#CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[-200, -150, -50], linestyles='solid', colors='blue')
#
#CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[-100], linestyles='solid', colors='blue')
#plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )
#
#CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=1 , levels=[50, 150, 200], linestyles='solid', colors='red')
#
#CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[100], linestyles='solid', colors='red')
#plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )
#
#CS = plt.contour(lat,Theta_axis_p,  p_zm_tm/100., linewidths=2 , levels=[100,200,300,500,700,850,950], linestyles='solid', colors='black')
#plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )
#
#plt.xlabel('Latitude [degrees North]', fontsize=18) # label along x-axes
#plt.ylabel('Potential temperature [K]', fontsize=18) # label along y-axes
#
#plt.text(30,375, month+" "+str(year), fontsize=14)
#plt.text(30,370,'Contour interval: 50 kg m^-1 s^-1 K^-1 (red: northward)', fontsize=10)
#plt.text(30,365,'Isobars (black) labled in hPa', fontsize=10)
#plt.text(30,360,'(MassFluxBDC.py)', fontsize=10)
#plt.title("Zonal mean meridional isentropic mass flux in "+month+" "+str(year), fontsize=18)
#
#plt.grid(True)
#filename = "TroposphericBDC-nh_"+month+str(year)
#plt.savefig(direcOutput+filename) # save plot as png-file in directory, DYME
#plt.show()

# Interesting websites for NAM :http://ljp.gcess.cn/dct/page/65569
# io.savemat('massflux292J1.mat',{"mf1":massflux[:,3,:,:] })
# N.savetxt('Theta_axis_p1.txt',Theta_axis_p)