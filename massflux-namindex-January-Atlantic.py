import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S

a=S.Dataset("C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Desktop\data\Mean sea level pressure\MSP38.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
psl=a.variables["msl"][:]
zpsl=np.mean(psl[:,:,48:69],axis=2)


month_len=4*31*38
time1 = month_len * (time - np.min(time)) / ( np.max(time) - np.min(time))
time2=np.arange(1980,2018)
MFJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J1Atlantic.txt')
MFJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J2Atlantic.txt')
MFJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J3Atlantic.txt')
MFJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J4Atlantic.txt')
MFJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J5Atlantic.txt')

sigmaJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J1Atlantic.txt')
sigmaJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J2Atlantic.txt')
sigmaJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J3Atlantic.txt')
sigmaJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J4Atlantic.txt')
sigmaJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J5Atlantic.txt')
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
vmerJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J1Atlantic.txt')
vmerJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J2Atlantic.txt')
vmerJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J3Atlantic.txt')
vmerJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J4Atlantic.txt')
vmerJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J5Atlantic.txt')
vmer501=vmerJ1[:,12]
vmer502=vmerJ2[:,12]
vmer503=vmerJ3[:,12]
vmer504=vmerJ4[:,12]
vmer505=vmerJ5[:,12]
vmer=[]
vmer.extend(vmer501)
vmer.extend(vmer502)
vmer.extend(vmer503)
vmer.extend(vmer504)
vmer.extend(vmer505)
vmer=np.array(vmer)
massfluxmean=np.zeros(len(sigma))

massfluxmean=sigma*vmer

MFJ501=MFJ1[:,6]
MFJ502=MFJ2[:,6]
MFJ503=MFJ3[:,6]
MFJ504=MFJ4[:,6]
MFJ505=MFJ5[:,6]
MF=[]

MF.extend(MFJ501)
MF.extend(MFJ502)  
MF.extend(MFJ503) 
MF.extend(MFJ504) 
MF.extend(MFJ505)
MF=np.array(MF)
np.savetxt('MFzmtm.txt',MF)
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
zpsl65=zpsl[:,11]
zpsl35=zpsl[:,25]
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
  
R=np.corrcoef(Pldif,Mfj)[0,1]

plt.figure(1)
plt.scatter(Pldif,Mfj,color='g',label='R=%f'%R)
# plt.scatter(Pldif[30],Mfj[30],color='b')
# plt.scatter(Pldif[27],Mfj[27],color='r')
# fit1=np.polyfit(Pldif,Mfj,deg=1)
# plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R)
plt.xlabel('NAM index [hPa]')
plt.ylabel('Total Meridional Mass Flux (kg*m^-1*s^-1*K^-1)')
plt.title('NAM index-Total Meridional mass flux in January 1980-2017 Atlantic ocean')
plt.legend()
plt.show()

plt.figure(2)
r2=np.corrcoef(Pldif,MFJdist)[0,1]
plt.scatter(Pldif,MFJdist,color='green',label='R=%f'%r2)
# plt.scatter(Pldif[30],MFJdist[30],color='blue')
# plt.scatter(Pldif[27],MFJdist[27],color='red')
# fit4=np.polyfit(Pldif,MFJdist,deg=1)
# plt.plot(Pldif,fit4[0]*Pldif+fit4[1],label='R=%f' %r2)
plt.xlabel('NAM index [hPa]')
plt.ylabel('Total Eddy  mass flux (kg*m^-1*s^-1*K^-1)')
plt.title('NAM index -Total Eddy mass flux in January 1980-2017 Atlantic ocean')
plt.legend()
plt.show()


# #method2
# 
# 
# MF2=[]
# MF2.extend(MFJ1)
# MF2.extend(MFJ2)
# MF2.extend(MFJ3)
# MF2.extend(MFJ4)
# MF2.extend(MFJ5)
# MF2=np.array(MF2)
# MF2=np.mean(MF2[:,5:19],axis=1)
# 
# 
# Mfj2=np.zeros(38)
# for n in range(0,38):
#     MFm=[]
#     for k in range(0,124):
#         MFm.append(MF2[n*124+k])
#     Mfj2[n]=np.mean(MFm)
#     
# R2=np.corrcoef(Pldif,Mfj2)[0,1]
# 
# plt.figure(3)
# plt.scatter(Pldif,Mfj2,color='g',label='R=%f'%R2)
# # plt.scatter(Pldif[30],Mfj2[30],color='b')
# # plt.scatter(Pldif[27],Mfj2[27],color='r')
# # fit1=np.polyfit(Pldif,Mfj2,deg=1)
# # plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R2)
# plt.xlabel('NAM index [hPa]')
# plt.ylabel('Total Meridional Mass Flux (kg*m^-1*s^-1*K^-1)')
# plt.title('NAM index-Total Meridional mass flux in January 1980-2017 Atlantic ocean')
# plt.legend()
# plt.show()
# 
# sigma1=[]
# sigma1.extend(sigmaJ1)
# sigma1.extend(sigmaJ2)
# sigma1.extend(sigmaJ3)
# sigma1.extend(sigmaJ4)
# sigma1.extend(sigmaJ5)
# sigma1=np.array(sigma1)
# sigma1=np.mean(sigma1[:,5:19],axis=1)
#     
# vmer1=[]
# vmer1.extend(vmerJ1)
# vmer1.extend(vmerJ2)
# vmer1.extend(vmerJ3)
# vmer1.extend(vmerJ4)
# vmer1.extend(vmerJ5)
# vmer1=np.array(vmer1)
# vmer1=np.mean(vmer1[:,5:19],axis=1)
# 
# massfluxmean2=sigma1*vmer1
# 
# MFdist1=MF2-massfluxmean2
# MFJdist1=np.zeros(38)
# 
# for n in range(0,38):
#     MFdistm=[]
#     for k in range(0,124):
#         MFdistm.append(MFdist1[n*124+k])
#     MFJdist1[n]=np.mean(MFdistm)
#     
# plt.figure(4)
# r2=np.corrcoef(Pldif,MFJdist1)[0,1]
# plt.scatter(Pldif,MFJdist1,color='green')
# plt.scatter(Pldif[30],MFJdist1[30],color='blue')
# plt.scatter(Pldif[27],MFJdist1[27],color='red')
# fit4=np.polyfit(Pldif,MFJdist1,deg=1)
# plt.plot(Pldif,fit4[0]*Pldif+fit4[1],label='R=%f' %r2)
# plt.xlabel('NAM-INDEX')
# plt.ylabel('Eddies of mass flux')
# plt.title('Eddies of mass flux in January 1980-2017 Atlantic ocean')
# plt.legend()
# plt.show()