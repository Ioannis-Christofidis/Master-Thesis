import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S

a=S.Dataset("C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Desktop\data\Mean sea level pressure\MSP38.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
psl=a.variables["msl"][:]
zpsl=np.mean(psl,axis=2)
indicate=np.arange(23,132)
zpsl1=np.delete(psl,indicate,axis=2)
zpsl=np.mean(zpsl1,axis=2)


month_len=4*31*38
time1 = month_len * (time - np.min(time)) / ( np.max(time) - np.min(time))
time2=np.arange(1980,2018)
MFJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J1Pacific.txt')
MFJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J2Pacific.txt')
MFJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J3Pacific.txt')
MFJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J4Pacific.txt')
MFJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Mzm307J5Pacific.txt')

sigmaJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J1Pacific.txt')
sigmaJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J2Pacific.txt')
sigmaJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J3Pacific.txt')
sigmaJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J4Pacific.txt')
sigmaJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\sigmazm307J5Pacific.txt')

sigma501=sigmaJ1[:,11]
sigma502=sigmaJ2[:,11]
sigma503=sigmaJ3[:,11]
sigma504=sigmaJ4[:,11]
sigma505=sigmaJ5[:,11]

sigma=[]
sigma.extend(sigma501)
sigma.extend(sigma502)
sigma.extend(sigma503)
sigma.extend(sigma504)
sigma.extend(sigma505)
sigma=np.array(sigma)

vmerJ1=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J1Pacific.txt')
vmerJ2=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J2Pacific.txt')
vmerJ3=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J3Pacific.txt')
vmerJ4=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J4Pacific.txt')
vmerJ5=np.loadtxt('C:\Users\JohnChris\Desktop\Old-laptop\Program Files (x86)\Users\John\Vzm307J5Pacific.txt')

Vmer501=vmerJ1[:,11]
Vmer502=vmerJ2[:,11]
Vmer503=vmerJ3[:,11]
Vmer504=vmerJ4[:,11]
Vmer505=vmerJ5[:,11]

vmer=[]
vmer.extend(Vmer501)
vmer.extend(Vmer502)
vmer.extend(Vmer503)
vmer.extend(Vmer504)
vmer.extend(Vmer505)
vmer=np.array(vmer)

massfluxm=vmer*sigma


MFJ501=MFJ1[:,11]
MFJ502=MFJ2[:,11]
MFJ503=MFJ3[:,11]
MFJ504=MFJ4[:,11]
MFJ505=MFJ5[:,11]
MF=[]

MF.extend(MFJ501)
MF.extend(MFJ502)  
MF.extend(MFJ503) 
MF.extend(MFJ504) 
MF.extend(MFJ505)
MF=np.array(MF)
np.savetxt('MFpacific.txt',MF)
MFd=MF-massfluxm

Mfj=np.zeros(38)
MFdm=np.zeros(38)
MFmm=np.zeros(38)
for n in range(0,38):
    MFm=[]
    MFmeanm=[]
    MFdistm=[]
    for k in range(0,124):
        MFm.append(MF[n*124+k])
        MFmeanm.append(massfluxm[n*124+k])
        MFdistm.append(MFd[n*124+k])
    Mfj[n]=np.mean(MFm)
    MFmm[n]=np.mean(MFmeanm)
    MFdm[n]=np.mean(MFdistm)
    
zpsl38=np.zeros(38)
zpsl65=zpsl[:,9]
zpsl35=zpsl[:,27]
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
plt.scatter(Pldif,Mfj,color='g')
plt.scatter(Pldif[30],Mfj[30],color='b')
plt.scatter(Pldif[27],Mfj[27],color='r')
fit1=np.polyfit(Pldif,Mfj,deg=1)
plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R)
plt.xlabel('NAM-index')
plt.ylabel('Total Meridional Mass Flux')
plt.title('January 1980-2017 Pacific Ocean (Method1)')
plt.legend()
plt.show()

R2=np.corrcoef(Pldif,MFdm)[0,1]
plt.figure(2)
plt.scatter(Pldif,MFdm,color='g')
plt.scatter(Pldif[30],MFdm[30],color='b')
plt.scatter(Pldif[27],MFdm[27],color='r')
fit1=np.polyfit(Pldif,MFdm,deg=1)
plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R2)
plt.xlabel('NAM-index')
plt.ylabel('Eddy Mass Flux')
plt.title('January 1980-2017 Pacific Ocean (Method1)')
plt.legend()
plt.show()


R5=np.corrcoef(Pldif,MFmm)[0,1]
plt.figure(5)
plt.scatter(Pldif,MFmm,color='g')
plt.scatter(Pldif[30],MFmm[30],color='b')
plt.scatter(Pldif[27],MFmm[27],color='r')
fit1=np.polyfit(Pldif,MFmm,deg=1)
plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R5)
plt.xlabel('NAM-index')
plt.ylabel('Mean meridional Mass Flux')
plt.title('January 1980-2017 Pacific Ocean ')
plt.legend()
plt.show()
#method2
# MF=[]
# MF.extend(MFJ1)
# MF.extend(MFJ2)  
# MF.extend(MFJ3) 
# MF.extend(MFJ4) 
# MF.extend(MFJ5)
# MF=np.array(MF)
# MF1=np.mean(MF[:,5:20],axis=1)
# 
# sigma=[]
# sigma.extend(sigmaJ1)
# sigma.extend(sigmaJ2)  
# sigma.extend(sigmaJ3) 
# sigma.extend(sigmaJ4) 
# sigma.extend(sigmaJ5)
# sigma=np.array(sigma)
# sigma1=np.mean(sigma[:,5:20],axis=1)
# 
# vmer=[]
# vmer.extend(vmerJ1)
# vmer.extend(vmerJ2)  
# vmer.extend(vmerJ3) 
# vmer.extend(vmerJ4) 
# vmer.extend(vmerJ5)
# vmer=np.array(vmer)
# vmer1=np.mean(vmer[:,5:20],axis=1)
# 
# massfluxmean=vmer1*sigma1
# 
# eddymassflux=MF1-massfluxmean
# 
# MFj2=np.zeros(38)
# MFm=np.zeros(38)
# MFd=np.zeros(38)
# for n in range(0,38):
#     mfj=[]
#     mfm=[]
#     mfd=[]
#     for k in range(0,124):
#          mfj.append(MF1[n*124+k])
#          mfm.append(massfluxmean[n*124+k])
#          mfd.append(eddymassflux[n*124+k])
#          
#     MFj2[n]=np.mean(mfj)
#     MFm[n]=np.mean(mfm)
#     MFd[n]=np.mean(mfd)
#     
# plt.figure(3)
# R=np.corrcoef(Pldif,MFj2)[0,1]
# plt.scatter(Pldif,MFj2,color='g')
# plt.scatter(Pldif[30],MFj2[30],color='b')
# plt.scatter(Pldif[27],MFj2[27],color='r')
# fit1=np.polyfit(Pldif,MFj2,deg=1)
# plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R)
# plt.xlabel('NAM-index')
# plt.ylabel('Total Meridional Mass Flux')
# plt.title('January 1980-2017 Pacific Ocean (Method2)')
# plt.legend()
# plt.show()
# 
# R2=np.corrcoef(Pldif,MFd)[0,1]
# plt.figure(4)
# plt.scatter(Pldif,MFd,color='g')
# plt.scatter(Pldif[30],MFd[30],color='b')
# plt.scatter(Pldif[27],MFd[27],color='r')
# fit1=np.polyfit(Pldif,MFd,deg=1)
# plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R2)
# plt.xlabel('NAM-index')
# plt.ylabel('Eddy Mass Flux')
# plt.title('January 1980-2017 Pacific Ocean (Method2)')
# plt.legend()
# plt.show()
# 
# R2=np.corrcoef(Pldif,MFm)[0,1]
# plt.figure(6)
# plt.scatter(Pldif,MFd,color='g')
# plt.scatter(Pldif[30],MFm[30],color='b')
# plt.scatter(Pldif[27],MFm[27],color='r')
# fit1=np.polyfit(Pldif,MFm,deg=1)
# plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='R=%f' %R2)
# plt.xlabel('NAM-index')
# plt.ylabel('Mean Meridional mass flux')
# plt.title('January 1980-2017 Pacific Ocean (Method2)')
# plt.legend()
# plt.show()
#     