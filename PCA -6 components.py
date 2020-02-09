# PCA analysic of Eddy PVS Flux,Mean Mass flux,Mean PVS flux,Eddy mass flux,NAM INDEX
import numpy as np
import netCDF4 as S
import matplotlib.pyplot as plt
import math as M
import scipy.io 
Jmean=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Thesis-Code\kthesis-deskstop\February thesis\krelative vorticity- vocomponent data\Jmean307.txt')
Jmean=Jmean[:,12]
Jeddy=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\Thesis-Code\kthesis-deskstop\February thesis\krelative vorticity- vocomponent data\Jeddy307.txt')
Jeddy=Jeddy[:,12]
cimf=np.loadtxt('C:\Users\JohnChris\Desktop\Thesis\CIMF\cimf-map\cimf300.txt')
a=S.Dataset("C:\Users\JohnChris\Desktop\Thesis\Results\data\Mean sea level pressure\MSP38.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
psl=a.variables["msl"][:]
zpsl=np.mean(psl,axis=2)

psl38=np.zeros(38)
zpsl65=zpsl[:,11]
zpsl35=zpsl[:,26]
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
plt.figure(1)

MF1=scipy.io.loadmat('C:\Users\JohnChris\Desktop\Thesis\Results\January\MFtotal307.mat')
MF=MF1['MF']
Vmer1=scipy.io.loadmat('C:\Users\JohnChris\Desktop\Thesis\Results\January\Vmertotal307.mat')
Vmer=Vmer1['Vmer']
Sigma1=scipy.io.loadmat('C:\Users\JohnChris\Desktop\Thesis\Results\January\Sigmatotal307.mat')
Sigma=Sigma1['Sigma']
MFy=np.zeros((38,124,37,144))
Vmery=np.zeros((38,124,37,144))
Sigmay=np.zeros((38,124,37,144))
for n in range(0,38):
    for j in range(0,124):
        for k in range(0,37):
            for l in range(0,144):
                MFy[n,j,k,l]=MF[n*124+j,k,l]
                Vmery[n,j,k,l]=Vmer[n*124+j,k,l]
                Sigmay[n,j,k,l]=Sigma[n*124+j,k,l]
                
MFyzm=np.mean(MFy,axis=3)
MFyzmtm=np.mean(MFyzm,axis=1)
                
# zonal mean of time mean of each component

Vmeryzm=np.mean(Vmery,axis=3)
Vmeryzmtm=np.mean(Vmeryzm,axis=1)

Sigmayzm=np.mean(Sigmay,axis=3)
Sigmayzmtm=np.mean(Sigmayzm,axis=1)
mf1=Vmeryzmtm*Sigmayzmtm


eddy=MFyzmtm-mf1
eddy=eddy[:,12]
mf1=mf1[:,12]
a=np.corrcoef(Jmean,Jeddy)[0,1]
b=np.corrcoef(Jmean,mf1)[0,1]
c=np.corrcoef(Jmean,eddy)[0,1]
d=np.corrcoef(Jmean,Pldif)[0,1]
e=np.corrcoef(Jmean,cimf)[0,1]
f=np.corrcoef(Jeddy,mf1)[0,1]
g=np.corrcoef(Jeddy,eddy)[0,1]
h=np.corrcoef(Jeddy,Pldif)[0,1]
i=np.corrcoef(Jeddy,cimf)[0,1]
k=np.corrcoef(mf1,eddy)[0,1]
l=np.corrcoef(mf1,Pldif)[0,1]
m=np.corrcoef(mf1,cimf)[0,1]
n=np.corrcoef(eddy,Pldif)[0,1]
o=np.corrcoef(eddy,cimf)[0,1]
p=np.corrcoef(Pldif,cimf)[0,1]
f=[[1.,a,b,c,d,e],[a,1.,f,g,h,i],[b,f,1.,k,l,m],[c,g,k,1.,n,o],[d,h,l,n,1.,p],[e,i,m,o,p,1.]]
f=np.array(f)
l,E=np.linalg.eigh(f)
E=np.array(E)
# K=sum(l)
# # ldev=l/K
# 
# mean mass flux deviation
MMF=np.mean(mf1)
sd=np.std(mf1)
MF_dev=(mf1-MMF)/sd
# mean PVS flux deviation
MPVS=np.mean(Jmean)
sd=np.std(Jmean)
Jmean_dev=(Jmean-MPVS)/sd
# eddy PVS flux deviation
EPVS=np.mean(Jeddy)
sd=np.std(Jeddy)
Jeddy_dev=(Jeddy-EPVS)/sd
# eddy mass flux deviation
MEMF=np.mean(eddy)
sd=np.std(eddy)
eddy_dev=(eddy-MEMF)/sd
# NAM INDEX deviation
MNM=np.mean(Pldif)
sd=np.std(Pldif)
Pldif_dev=(Pldif-MNM)/sd
# cimf deviation
meanc=np.mean(cimf)
sd=np.std(cimf)
cimf_dev=(cimf-meanc)/sd
X=[Jmean_dev,Jeddy_dev,MF_dev,eddy_dev,Pldif_dev,cimf_dev]
X=np.array(X)
X=X.transpose()
lmax=max(l)
lmin=min(l)
l_max=5
l_max2=4
l_max3=3
l_min3=2
l_min2=1
l_min=0
Y=np.dot(X,E)
Y1=Y[:,l_max]
Y2=Y[:,l_max2]
Y3=Y[:,l_max3]
Y4=Y[:,l_min3]
Y5=Y[:,l_min2]
Y6=Y[:,l_min3]

plt.figure(1)
time=np.arange(1980,2018)
plt.scatter(time,Y1,color='red',label='Y1')
plt.scatter(time,Y2,color='black',label='Y2')
plt.scatter(time,Y3,color='green',label='Y3')
plt.scatter(time,Y4,color='red',label='Y4')
plt.scatter(time,Y5,color='cyan',label='Y5')
plt.scatter(time,Y6,color='magenta',label='Y6')
plt.legend()
plt.xlabel('Years',fontsize=15)
plt.ylabel('Amplitude of principal components',fontsize=15)
plt.title('Principal component analysis',fontsize=20)
plt.show()

plt.figure(2,figsize=(10,6))
time=np.arange(1980,2018)
plt.scatter(time,Y1,color='red',label='Y1')
plt.scatter(time,Y2,color='black',label='Y2')
# plt.scatter(time,Y3,color='black',label='Y3')
# plt.scatter(time,Y4,color='red',label='Y4')
# plt.scatter(time,Y5,color='cyan',label='Y5')
plt.legend()
plt.xlabel('Years',fontsize=20)
plt.ylabel('Amplitude of  the 2 first principal components',fontsize=17)
plt.title('Principal component analysis',fontsize=20)
plt.savefig("epvsmmf.png",bbox_inches='tight')    
plt.savefig("pca2.png",bbox_inches='tight')    

plt.show()

# plt.figure(4)
# r=np.corrcoef(Jeddy,eddy)[0,1]
# plt.scatter(Jeddy,eddy,label='r=%f'%r)
# plt.xlabel('Eddy PVS flux [10^-5 m s^-2]')
# plt.ylabel('Eddy meridional mass flux[kg m^-1 K^-1 s^-1]')
# plt.title('Relation Eddy PVS-mass flux at 307K and 50N')
# plt.legend()

fig=plt.figure(5,figsize=(10,6))
r1=np.corrcoef(mf1,Jeddy)[0,1]
plt.scatter(Jeddy,mf1,label='r=%f'%r1)
plt.scatter(Jeddy[27],mf1[27],color='red',label='2007')
plt.scatter(Jeddy[30],mf1[30],color='black',label='2010')
fit1=np.polyfit(Jeddy,mf1,deg=1)
plt.plot(Jeddy,fit1[0]*Jeddy+fit1[1],label='y=%f'%fit1[0]+'*x +'+str(fit1[1]))
plt.ylabel('Mean meridional mass flux [10^-5 m s^-2]',fontsize=15)
plt.xlabel('Eddy PVS flux [10^-5 m s^-2]',fontsize=15)
plt.title('Eddy PVS-mean mass flux at 307.5K and 45N January (1980-2017)',fontsize=17)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("epvsmmf.png",bbox_inches='tight')    




fig=plt.figure(6,figsize=(10,6))
r2=np.corrcoef(eddy,cimf)[0,1]
plt.scatter(eddy,cimf,label='r=%f'%r2)
fit1=np.polyfit(eddy,cimf,deg=1)
plt.plot(eddy,fit1[0]*eddy+fit1[1],label='y=%f'%fit1[0]+'*x +'+str(fit1[1]))
plt.ylabel('CIMF [ Kg m^-1 s^1]',fontsize=18)
plt.xlabel('Eddy meridional mass flux[kg m^-1 K^-1 s^-1]',fontsize=15)
plt.title('Relation CIMF- eddy mass flux at 307K and 45N',fontsize=17)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("emfcimf.png",bbox_inches='tight')   


fig=plt.figure(7,figsize=(10,6))
r3=np.corrcoef(MFyzmtm[:,12],cimf)[0,1]
plt.scatter(MFyzmtm[:,12],cimf,label='r=%f'%r3)
fit1=np.polyfit(MFyzmtm[:,12],cimf,deg=1)
plt.plot(MFyzmtm[:,12],fit1[0]*MFyzmtm[:,12]+fit1[1],label='y=%f'%fit1[0]+'*x +'+str(fit1[1]))
plt.ylabel('CIMF [ Kg m^-1 s^1] at 300K',fontsize=15)
plt.xlabel('Total meridional mass flux[kg m^-1 K^-1 s^-1]',fontsize=13)
plt.title('Relation CIMF- total mass flux at 307K and 45N January (1980-2017)',fontsize=17)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("tmfcimf.png",bbox_inches='tight')   



# plt.figure(8)
# r4=np.corrcoef(mf1,cimf)[0,1]
# plt.scatter(mf1,cimf,label='r=%f'%r4)
# plt.ylabel('CIMF [ Kg m^-1 s^1] at 300K')
# plt.xlabel('Mean meridional mass flux[kg m^-1 K^-1 s^-1]')
# plt.title('Relation CIMF- mean mass flux at 307K and 45N')
# plt.legend()
# plt.show()
# 
fig=plt.figure(9,figsize=(10,6))
r5=np.corrcoef(Jeddy,Jmean)[0,1]
plt.scatter(Jmean,Jeddy,label='r=%f'%r5)
fit1=np.polyfit(Jmean,Jeddy,deg=1)
plt.plot(Jmean,fit1[0]*Jmean+fit1[1],label='y=%f'%fit1[0]+'*x '+str(fit1[1]))
plt.xlabel('Mean PVS flux [10^-5 m s^-2]',fontsize=18)
plt.ylabel('Eddy PVS flux [10^-5 m s^-2]',fontsize=18)
plt.title('Relation Eddy-Mean PVS flux at 307.5K and 45N January (1980-2017)',fontsize=20)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("empvs.png",bbox_inches='tight')   



fig=plt.figure(10,figsize=(10,6))
r5=np.corrcoef(Pldif,cimf)[0,1]
plt.scatter(Pldif,cimf,label='r=%f'%r5)
fit1=np.polyfit(Pldif,cimf,deg=1)
plt.plot(Pldif,fit1[0]*Pldif+fit1[1],label='y=%f'%fit1[0]+'*x '+str(fit1[1]))
plt.xlabel('NAM index[hPa]',fontsize=20)
plt.ylabel('CIMF [ Kg m^-1 s^1]',fontsize=15)
plt.title('Relation of NAM index and CIMF(polar area) for January (1980-2017)',fontsize=15)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("cimfnam.png",bbox_inches='tight') 

  
fig=plt.figure(11,figsize=(10,6))
r1=np.corrcoef(mf1,Jmean)[0,1]
plt.scatter(Jmean,mf1,label='r=%f'%r1)

fit1=np.polyfit(Jmean,mf1,deg=1)
plt.plot(Jmean,fit1[0]*Jmean+fit1[1],label='y=%f'%fit1[0]+'*x +'+str(fit1[1]))
plt.ylabel('Mean meridional mass flux [10^-5 m s^-2]',fontsize=15)
plt.xlabel('Mean PVS flux [10^-5 m s^-2]',fontsize=15)
plt.title('Mean PVS-mass flux at 307.5K and 45N January (1980-2017)',fontsize=15)
plt.legend(prop={'size': 15})
plt.show()
plt.savefig("mpvsmf.png",bbox_inches='tight')    
  


