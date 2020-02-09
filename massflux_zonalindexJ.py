# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S


a = S.Dataset("C:\Users\JohnChris\Desktop\Thesis\Results\data\Mean sea level pressure\MSP38.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
psl=a.variables["msl"][:]
zpsl=np.mean(psl,axis=2)
month_len=4*31*38
time1 = month_len * (time - np.min(time)) / ( np.max(time) - np.min(time))
time2=np.arange(1980,2018)
MFJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Mzm307J1.txt')
MFJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Mzm307J2.txt')
MFJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Mzm307J3.txt')
MFJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Mzm307J4.txt')
MFJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Mzm307J5.txt')
sigmaJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\sigmazm307J1.txt')
sigmaJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\sigmazm307J2.txt')
sigmaJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\sigmazm307J3.txt')
sigmaJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\sigmazm307J4.txt')
sigmaJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\sigmazm307J5.txt')
sigma501=sigmaJ1[:,10]
sigma502=sigmaJ2[:,10]
sigma503=sigmaJ3[:,10]
sigma504=sigmaJ4[:,10]
sigma505=sigmaJ5[:,10]
sigma=[]
sigma.extend(sigma501)
sigma.extend(sigma502)
sigma.extend(sigma503)
sigma.extend(sigma504)
sigma.extend(sigma505)
sigma=np.array(sigma)
vmerJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Vzm307J1.txt')
vmerJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Vzm307J2.txt')
vmerJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Vzm307J3.txt')
vmerJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Vzm307J4.txt')
vmerJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Results\Thesis-Data\January\Vzm307J5.txt')
vmer501=vmerJ1[:,10]
vmer502=vmerJ2[:,10]
vmer503=vmerJ3[:,10]
vmer504=vmerJ4[:,10]
vmer505=vmerJ5[:,10]
vmer=[]
vmer.extend(vmer501)
vmer.extend(vmer502)
vmer.extend(vmer503)
vmer.extend(vmer504)
vmer.extend(vmer505)
vmer=np.array(vmer)
massfluxmean=np.zeros(len(sigma))

massfluxmean=sigma*vmer

MFJ501=MFJ1[:,10]
MFJ502=MFJ2[:,10]
MFJ503=MFJ3[:,10]
MFJ504=MFJ4[:,10]
MFJ505=MFJ5[:,10]
MF=[]

MF.extend(MFJ501)
MF.extend(MFJ502)  
MF.extend(MFJ503) 
MF.extend(MFJ504) 
MF.extend(MFJ505)
MF=np.array(MF)
MFdist=MF-massfluxmean
Mfj=np.zeros(38)
MFJdist=np.zeros(38)
massfluxmeanJ=np.zeros(38)
for n in range(0,38):
    MFm=[]
    MFmeanm=[]
    MFdistm=[]
    for k in range(0,124):
        MFm.append(MF[n*124+k])
        MFmeanm.append(massfluxmean[n*124+k])
        MFdistm.append(MFdist[n*124+k])
    Mfj[n]=np.mean(MFm)
    MFJdist[n]=np.mean(MFdistm)
    massfluxmeanJ[n]=np.mean(MFmeanm)
zpsl38=np.zeros(38)
zpsl65=zpsl[:,10]
zpsl35=zpsl[:,22]
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
Pl35dev=Pl35-Pl35m
Pl65dev=Pl65-Pl65m
Pldif=Pl35dev-Pl65dev
Pldif=Pldif/100.
std=np.std(Pldif)
Pldif=Pldif/std

R=np.corrcoef(Pldif,Mfj)[0,1]

fig=plt.figure(1,figsize=(10,6))
# fig, ax = plt.subplots()
# ax.scatter(Pldif, Mfj)
# n=np.arange(1980,2018)
# for i, txt in enumerate(n):
#     ax.annotate(txt, (Pldif[i], Mfj[i]))
# plt.scatter(Pldif,Mfj,color='g')
plt.scatter(Pldif,Mfj,color='g',label='R=%f'%R)
# plt.scatter(Pldif[0],Mfj[0],color='black')
# plt.scatter(Pldif[16],Mfj[16],color='black')
# plt.scatter(Pldif[24],Mfj[24],color='black')
# plt.scatter(Pldif[30],Mfj[30],color='black')
# plt.scatter(Pldif[31],Mfj[31],color='black')
# plt.scatter(Pldif[18],Mfj[18],color='black')
# plt.scatter(Pldif[27],Mfj[27],color='r')
# plt.scatter(Pldif[9],Mfj[9],color='r')
# plt.scatter(Pldif[10],Mfj[10],color='r')
# plt.scatter(Pldif[12],Mfj[12],color='r')
# plt.scatter(Pldif[13],Mfj[13],color='r')
# plt.scatter(Pldif[22],Mfj[22],color='r')
# # plt.scatter(Pldif,Mfj,color='g',label='R=%f'%R)
# plt.scatter(Pldif[30],Mfj[30],color='black')
# plt.scatter(Pldif[27],Mfj[27],color='r')
# fit1=np.polyfit(Pldif,Mfj,deg=1)
# plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R)
fit=np.polyfit(Pldif,Mfj,deg=1)
plt.plot(Pldif,Pldif*fit[0]+fit[1],label='y=%f'%fit[0]+'*x '+str(fit[1]))
plt.xlabel('NAM index [hPa]',fontsize=15)
plt.ylabel('Total Meridional Mass Flux  (kg*m^-1*s^-1*K^-1)',fontsize=15)
plt.title('NAM index -TMF at 307.5K and at 50N in January 1980-2017',fontsize=18)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("tmfnamJ50",bbox_inches='tight')   
#plt.figure(2)
#plt.plot(time2,Pldif,'.')
#fit2=np.polyfit(time2,Pldif,deg=1)
#plt.plot(time2,fit2[0]*time2+fit2[1],label='s=%f' %fit2[0])
#plt.xlabel('Years')
#plt.ylabel('NAM-index')
#plt.legend()
#plt.show()
#plt.figure(3)
#plt.plot(time2,Mfj,'.')
#fit3=np.polyfit(time2,Mfj,deg=1)
#plt.plot(time2,fit3[0]*time2+fit3[1],label='s=%f' %fit3[0])
#plt.xlabel('Years')
#plt.ylabel('Meridional Mass flux')
#R=np.corrcoef(Mfj,fit3[0]*time2+fit3[1])[0,1]
#plt.legend()
#plt.show()

fig=plt.figure(4,figsize=(10,6))
# fig, ax = plt.subplots()
# ax.scatter(Pl35dev/100, Pl65dev/100.)
# n=np.arange(1980,2018)
# for i, txt in enumerate(n):
#     ax.annotate(txt, (Pl35dev[i]/100,Pl65dev[i]/100))
r=np.corrcoef(Pl35dev/100.,Pl65dev/100.)[0,1]
plt.scatter(Pl35dev/100.,Pl65dev/100.,color='g',label='R=%f'%r)
# fit4=np.polyfit(Pl35dev/100.,Pl65dev/100.,deg=1)
# plt.plot(Pl35dev/100.,fit4[0]*Pl35dev/100.+fit4[1])
# plt.scatter(Pl35dev[30]/100.,Pl65dev[30]/100.,color='b')
# plt.scatter(Pl35dev[0]/100.,Pl65dev[0]/100.,color='b')
# plt.scatter(Pl35dev[36]/100.,Pl65dev[36]/100.,color='b')
# plt.scatter(Pl35dev[18]/100.,Pl65dev[18]/100.,color='b')
# plt.scatter(Pl35dev[24]/100.,Pl65dev[24]/100.,color='b')
# plt.scatter(Pl35dev[16]/100.,Pl65dev[16]/100.,color='b')
# plt.scatter(Pl35dev[27]/100.,Pl65dev[27]/100.,color='r')
# plt.scatter(Pl35dev[9]/100.,Pl65dev[9]/100.,color='r')
# plt.scatter(Pl35dev[10]/100.,Pl65dev[10]/100.,color='r')
# plt.scatter(Pl35dev[12]/100.,Pl65dev[12]/100.,color='r')
# plt.scatter(Pl35dev[13]/100.,Pl65dev[13]/100.,color='r')
# plt.scatter(Pl35dev[22]/100.,Pl65dev[22]/100.,color='r')
plt.xlabel('monthly mean sea level pressure anomaly at 35N[hPa]',fontsize=18)
plt.ylabel('monthly mean sea level pressure anomaly at 65N[hPa]',fontsize=15)
plt.title('January (1980-2017)',fontsize=20)
# plt.text(0,6,'R='+str(r),fontsize=12)
plt.legend(prop={'size': 20})
plt.show()
plt.savefig("namindexJ",bbox_inches='tight')   
plt.figure(5,figsize=(10,6))
r2=np.corrcoef(Pldif,MFJdist)[0,1]
plt.scatter(Pldif,MFJdist,color='green',label='R=%f'%r2)
# plt.scatter(Pldif[0],MFJdist[0],color='b')
# plt.scatter(Pldif[16],MFJdist[16],color='b')
# plt.scatter(Pldif[24],MFJdist[24],color='b')
# plt.scatter(Pldif[30],MFJdist[30],color='b')
# plt.scatter(Pldif[31],MFJdist[31],color='b')
# plt.scatter(Pldif[18],MFJdist[18],color='b')
# plt.scatter(Pldif[27],MFJdist[27],color='r')
# plt.scatter(Pldif[9],MFJdist[9],color='r')
# plt.scatter(Pldif[10],MFJdist[10],color='r')
# plt.scatter(Pldif[12],MFJdist[12],color='r')
# plt.scatter(Pldif[13],MFJdist[13],color='r')
# plt.scatter(Pldif[22],MFJdist[22],color='r')
# plt.scatter(Pldif,Mfj,color='g',label='R=%f'%R)
# plt.scatter(Pldif[30],MFJdist[30],color='blue')
# plt.scatter(Pldif[27],MFJdist[27],color='red')
# fit4=np.polyfit(Pldif,MFJdist,deg=1)
# plt.plot(Pldif,fit4[0]*Pldif+fit4[1],label='R=%f' %r2)
fit=np.polyfit(Pldif,MFJdist,deg=1)
plt.plot(Pldif,Pldif*fit[0]+fit[1],label='y=%f'%fit[0]+'*x +'+str(fit[1]))
plt.xlabel('NAM index [hPa]',fontsize=15)
plt.ylabel('Eddy meridional mass flux  (kg*m^-1*s^-1*K^-1)',fontsize=15)
plt.title('NAM index-Eddy mass flux at 307.5 and at 50N in January 1980-2017',fontsize=18)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("emfnamJ50",bbox_inches='tight')   
# plt.figure(6)
# R2=np.corrcoef(Pldif,massfluxmeanJ)[0,1]
# plt.scatter(Pldif,massfluxmeanJ,color='green')
# plt.scatter(Pldif[30],massfluxmeanJ[30],color='blue')
# plt.scatter(Pldif[27],massfluxmeanJ[27],color='red')
# fit5=np.polyfit(Pldif,massfluxmeanJ,deg=1)
# plt.plot(Pldif,fit5[0]*Pldif+fit5[1],label='R=%f' %R2)
# plt.xlabel('NAM-INDEX')
# plt.ylabel('Mean of the mass flux ')
# plt.title(' Mean of mass flux in January 1980--2017')
# plt.legend()
# plt.show()
# #Jan=np.zeros((38,4*31))
# #Janeddies=np.zeros((38,4*31))
# #for n in range(0,38):
# #    Janeddies[n,:]=MFdist[n*4*31:(n+1)*4*31]
# #for n in range(0,38):
# #    Jan[n,:]=MF[n*4*31:(n+1)*4*31]
# #
# #np.savetxt('MFjan.txt',Jan)
# #np.savetxt('MFJaneddies.txt',Janeddies)
# #
# #Janmean=np.zeros((38,4*31))
# #for n in range(0,38):
# #    Janmean[n,:]=massfluxmean[n*4*31:(n+1)*4*31]
# #
# #np.savetxt('MFJanmean.txt',Janmean)
# 
# MF1=[]
# MF1.extend(MFJ1)
# MF1.extend(MFJ2)
# MF1.extend(MFJ3)
# MF1.extend(MFJ4)
# MF1.extend(MFJ5)
# MF1=np.array(MF1)
# MF1=np.mean(MF1[:,5:21],axis=1)
# Mfj2=np.zeros(38)
# for n in range(0,38):
#     MFm=[]
#     for k in range(0,124):
#         MFm.append(MF1[n*124+k])        
#     Mfj2[n]=np.mean(MFm)
#     
#     
# R=np.corrcoef(Pldif,Mfj2)[0,1]
# 
# plt.figure(8)
# plt.scatter(Pldif,Mfj2,color='g')
# plt.scatter(Pldif[30],Mfj2[30],color='b')
# plt.scatter(Pldif[27],Mfj2[27],color='r')
# fit7=np.polyfit(Pldif,Mfj2,deg=1)
# plt.plot(Pldif,fit7[0]*Pldif+fit7[1],label='R=%f' %R)
# plt.xlabel('NAM-index')
# plt.ylabel('Meridional Mass Flux')
# plt.legend()
# plt.show()
# 
# 
# sigma1=[]
# sigma1.extend(sigmaJ1)
# sigma1.extend(sigmaJ2)
# sigma1.extend(sigmaJ3)
# sigma1.extend(sigmaJ4)
# sigma1.extend(sigmaJ5)
# sigma1=np.array(sigma1)
# 
# sigma1=np.mean(sigma1[:,5:21],axis=1)
# 
# vmer1=[]
# vmer1.extend(vmerJ1)
# vmer1.extend(vmerJ2)
# vmer1.extend(vmerJ3)
# vmer1.extend(vmerJ4)
# vmer1.extend(vmerJ5)
# vmer1=np.array(vmer1)
# vmer1=np.mean(vmer1[:,5:21],axis=1)
# 
# massfluxmean1=np.zeros(len(sigma1))
# massfluxmean1=sigma1*vmer1
# 
# MFdist1=MF1-massfluxmean1
# MFJdist1=np.zeros(38)
# massfluxmeanJ1=np.zeros(38)
# for n in range(0,38):
#     MFmeanm=[]
#     MFdistm=[]
#     for k in range(0,124):
#         MFmeanm.append(massfluxmean1[n*124+k])
#         MFdistm.append(MFdist1[n*124+k])
#     MFJdist1[n]=np.mean(MFdistm)
#     massfluxmeanJ1[n]=np.mean(MFmeanm)
#     
# plt.figure(9)
# r2=np.corrcoef(Pldif,MFJdist1)[0,1]
# plt.scatter(Pldif,MFJdist1,color='green')
# plt.scatter(Pldif[30],MFJdist1[30],color='blue')
# plt.scatter(Pldif[27],MFJdist1[27],color='red')
# fit4=np.polyfit(Pldif,MFJdist1,deg=1)
# plt.plot(Pldif,fit4[0]*Pldif+fit4[1],label='R=%f' %r2)
# plt.xlabel('NAM-INDEX')
# plt.ylabel('Eddies of mass flux')
# plt.title('Eddies of mass flux in January 1980-2017')
# plt.legend()
# plt.show()
# plt.figure(10)
# R2=np.corrcoef(Pldif,massfluxmeanJ1)[0,1]
# plt.scatter(Pldif,massfluxmeanJ1,color='green')
# plt.scatter(Pldif[30],massfluxmeanJ1[30],color='blue')
# plt.scatter(Pldif[27],massfluxmeanJ1[27],color='red')
# fit5=np.polyfit(Pldif,massfluxmeanJ1,deg=1)
# plt.plot(Pldif,fit5[0]*Pldif+fit5[1],label='R=%f' %R2)
# plt.xlabel('NAM-INDEX')
# plt.ylabel('Mean of the mass flux ')
# plt.title(' Mean of mass flux in January 1980--2017')
# plt.legend()
# plt.show()
# 
# 
# f=MFJdist1/Mfj2
