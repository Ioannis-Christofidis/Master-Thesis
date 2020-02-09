import numpy as np
import matplotlib.pyplot as plt
import math as m
import netCDF4 as S

a = S.Dataset("C:\Users\John\Downloads\Potential Vorticity  430,475,530.nc", mode='r')
lat = a.variables["latitude"][:] 
lon = a.variables["longitude"][:] 
time = a.variables["time"][:]
level=a.variables["level"][:]
PV=a.variables["pv"][:]
#PV1=np.zeros((4296,3,37,144))
#R=6371221
#pi=3.14
#Da=np.zeros((37,144))
#for n in range(0,4296):
#    for j in range(0,3):
#        for k in range(0,36):
#            for l in range(0,143):
#                Da[k,l]=R**2*2.5*pi/180.*(np.sin(pi*lat[k]/180.)-np.sin(pi*lat[k+1]/180.))
#                PV1[n,j,k,l]=(PV[n,j,k+1,l+1]-PV[n,j,k,l])*Da[k,l]
#                
#PVz=np.sum(PV1,axis=3)
#PVzm=np.sum(PVz[:,:,0:7],axis=2)
#Area1=np.sum(Da,axis=1)
#Area=np.sum(Area1[0:7])
#np.savetxt('PotentialVorticityarea1.txt',PVzm)
#np.savetxt('Area',Area)
PVzm=np.mean(PV,axis=3)
PVztm=np.mean(PVzm,axis=0)
#PVzm350=PVzm[:,0,:]
#PVzm370=PVzm[:,1,:]
#PVzm395=PVzm[:,2,:]
#np.savetxt('PVzm350.txt',PVzm350)
#np.savetxt('PVzm370.txt',PVzm370)
#np.savetxt('PVzm395.txt',PVzm395)
#np.savetxt('latitude.txt',lat)
np.savetxt('PVztm1.txt',PVztm)
