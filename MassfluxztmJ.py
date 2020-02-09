# -*- coding: utf-8 -*-
import numpy as np
import netCDF4 as S
import matplotlib.pyplot as plt
#import mpl_toolkits.basemap as bm
import math as M
MFJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\MzmJ1.txt')
MFJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\MzmJ2.txt')
MFJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\MzmJ3.txt')
MFJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\MzmJ4.txt')
MFJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\MzmJ5.txt')
lat=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\latJ.txt')
Theta_axis=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\TJ.txt')
Theta_axis_p=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\TpJ.txt')
PztJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\PztJ1.txt')
PztJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\PztJ2.txt')
PztJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\PztJ3.txt')
PztJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\PztJ4.txt')
PztJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\PztJ5.txt')
massflux_zm_tm=(7*np.array(MFJ1)+8*np.array(MFJ2)+8*np.array(MFJ3)+8*np.array(MFJ4)+7*np.array(MFJ5))/38.
p_zm_tm=(7*np.array(PztJ1)+8*np.array(PztJ2)+8*np.array(PztJ3)+8*np.array(PztJ4)+8*np.array(PztJ5))/38.
fig=plt.figure(figsize=(10,6))
plt.axis([-15,75.,270,380])

plt.xticks(np.arange(-15,80,15), fontsize=14)
plt.yticks(np.arange(270,390.,10), fontsize=14)

CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[-200, -150, -50], linestyles='solid', colors='blue')

CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[-100], linestyles='solid', colors='blue')
plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )

CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=1 , levels=[50, 150, 200], linestyles='solid', colors='red')

CS = plt.contour(lat,Theta_axis,  massflux_zm_tm, linewidths=2 , levels=[100], linestyles='solid', colors='red')
plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )

CS = plt.contour(lat,Theta_axis_p,  p_zm_tm/100., linewidths=2 , levels=[100,200,300,500,700,850,950], linestyles='solid', colors='black')
plt.clabel(CS, fontsize=12, inline=1,fmt = '%1.0f' )

plt.xlabel('Latitude [degrees North]', fontsize=22) # label along x-axes
plt.ylabel('Potential temperature [K]', fontsize=22) # label along y-axes

plt.text(30,375, "January"+" "+"1980-2017", fontsize=14)
plt.text(10,365,'Contour interval: 50 kg m^-1 s^-1 K^-1 (red: northward)', fontsize=14)
plt.text(30,355,'Isobars (black) labled in hPa', fontsize=14)
# plt.text(30,360,'(MassFluxBDC.py)', fontsize=10)
plt.title("Zonal mean meridional isentropic mass flux in "+"January"+" "+"1980-2017", fontsize=18)

plt.grid(True)

plt.show()
plt.savefig("mfzmtmJ.png",bbox_inches='tight')    

# Interesting websites for NAM :http://ljp.gcess.cn/dct/page/65569
