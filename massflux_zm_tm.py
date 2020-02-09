# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S


a = S.Dataset("C:\Users\John\Desktop\data\January\MSLPJ38.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
psl=a.variables["msl"][:]
zpsl=np.mean(psl,axis=2)
month_len=4*31*38
time1 = month_len * (time - np.min(time)) / ( np.max(time) - np.min(time))
time2=time1/3+1979+2/3.
MFJ1=np.loadtxt('C:\Users\John\Mzm360J1.txt')
MFJ2=np.loadtxt('C:\Users\John\Mzm360J2.txt')
MFJ3=np.loadtxt('C:\Users\John\Mzm360J3.txt')
MFJ4=np.loadtxt('C:\Users\John\Mzm360J4.txt')
MFJ5=np.loadtxt('C:\Users\John\Mzm360J5.txt')

MFJ451=MFJ1[:,12]
MFJ452=MFJ2[:,12]
MFJ453=MFJ3[:,12]
MFJ454=MFJ4[:,12]
MFJ455=MFJ5[:,12]
MF=[]

MF.extend(MFJ451)
MF.extend(MFJ452)  
MF.extend(MFJ453) 
MF.extend(MFJ454) 
MF.extend(MFJ455)
MF=np.array(MF)
Mfj=np.zeros(38)
for n in range(0,38):
    MFm=[]
    for k in range(0,124):
        MFm.append(MF[n*124+k])
    Mfj[n]=np.mean(MFm)
zpsl38=np.zeros(38)
zpsl65=zpsl[:,4]
zpsl35=zpsl[:,16]
Pl35=np.zeros(38)
Pl65=np.zeros(38)
for n in range(0,38):
    Plm35=[]
    Plm65=[]
    for k in range(0,124):
        Plm35.append(zpsl35[n*124+k])
        Plm65.append(zpsl65[n*124+k])
    Pl35[n]=np.mean(Plm35)
    Pl65[n]=np.mean(Plm65)
Pl35m=np.mean(Pl35)
Pl65m=np.mean(Pl65)
Pl35dev=Pl35m-Pl35
Pl65dev=Pl65m-Pl65
Pldif=Pl35dev-Pl65dev

  
    
plt.figure(1)
plt.plot(Pldif/100,Mfj,'.')
#fit1=np.polyfit(difp,MF,deg=1)
#plt.plot(difp,fit1[0]*difp+fit1[1],label='s=%f' %fit1[0])
#plt.legend()
plt.show()
#plt.figure(2)
#plt.plot(time2,difp,'.')
#fit2=np.polyfit(time2,difp,deg=1)
#plt.plot(time2,fit2[0]*time2+fit2[1],label='s=%f' %fit2[0])
#plt.legend()
#plt.figure(3)
#plt.plot(time2,MF,'.')
#fit3=np.polyfit(time2,MF,deg=1)
#plt.plot(time2,fit3[0]*time2+fit3[1],label='s=%f' %fit3[0])
#plt.legend()
#plt.show()
#r=np.corrcoef(difp,MF)[0,1]
#print r
#cd=r**2
#print cd

