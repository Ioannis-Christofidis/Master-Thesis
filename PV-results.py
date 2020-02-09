import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S

PV=np.loadtxt('C:\Users\John\PotentialVorticityarea1.txt')
PV1=np.loadtxt('C:\Users\John\PotentialVorticityarea.txt')
Area=11765371728401.629
PV2=PV/Area
PVf=PV2[:,2]
PVd=np.zeros(1074)
for n in range(0,1074):
    pvd=[]
    for j in range(0,4):
        pvd.append(PVf[n*4+j])
    PVd[n]=np.mean(pvd)
    
PVm=np.zeros(38)
k=4
for n in range(0,38):
    pvm=[]
    if k!=4:
        for j in range(0,28):
            pvm.append(PVd[n*28+j])
        k=k+1
    elif k==4:
        for j in range(0,29):
            pvm.append(PVd[n*28+j])
    PVm[n]=np.mean(pvm)
PVm=PVm*10**6
time=np.arange(1980,2018)
plt.scatter(time,PVm)
plt.scatter(time[5],PVm[5],color='red')
plt.scatter(time[14],PVm[14],color='red')
plt.scatter(time[16],PVm[16],color='red')
plt.scatter(time[17],PVm[17],color='red')
plt.scatter(time[36],PVm[36],color='red')
plt.ylabel('Potential Vorticity (PVU 10^6)')
plt.xlabel('Years')
plt.title('Potential Vorticity at 395K over the north polar area')
plt.show()
        
        
