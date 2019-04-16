import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


##########################################################################
#Initialization

#1. Initialize the state of the filter
#2. Initialize our belief in the state
#Predict

#1. Use system behavior to predict state at the next time step
#2. Adjust belief to account for the uncertainty in prediction
#Update

#1. Get a measurement and associated belief about its accuracy
#2. Compute residual between estimated state and measurement
#3. New estimate is somewhere on the residual line
##########################################################################

def g_h_filter(data, x0, dx, g, h, dt=1.):
    #The first estimation should be the same with state x 
    x_est = x0
    results = []
    #With each measurement data coming, do the prediction and update steps
    for z in data:
        # prediction step: need to prediction values for state and gain
        x_pred = x_est + (dx*dt)
        #The gain weight is varying actually, need some code here!!!
        dx = dx

        # update step
        residual = z - x_pred
        #update the gain according to measurement
        dx = dx + h * (residual) / dt
        #choose best estimation somewhere in between pre-estimation and measurement!
        x_est = x_pred + g * residual
        results.append(x_est)
    return np.array(results)

#delta_accl means with time going, dx increases by the value of delta_accl
def gen_data(x0,dx,count,noise_factor,delta_accl=0.0):
    zs = []
    for i in range(count):
        zs.append(x0 + dx*i + randn()*noise_factor)
        dx += delta_accl
    return zs

#############################################################
#common part
count = 100  # how many data of measurement 
x = np.arange(0, count, 1)

#experiments:
#1.100 measurement data are good: x0 = 0, dx = 1, std_dev = 1
#measurements = gen_data(0,1,count,1)
#2.100 measurement data are bad: x0 = 5, dx = 2, std_dev = 10
#measurements = gen_data(5.,2,count,10)
#3.100 measurement data are extremely noisy
#measurements = gen_data(5.,2,count,100)
#4.100 measurement data have no noise but the dx with some acceleration=2
#measurements = gen_data(5,0,count,0,2.)
#5.100 measurement data have dx and 50 for noise factor. No accl.
measurements = gen_data(5,5,count,50)


#1.g_h filter measurement data: init state:x0 = 0, init state prediction delta:dx = 1,
#scale factor of next delta_x h=0.02, scale factor of next update g=0.2
#data = g_h_filter(data = measurements,x0=0,dx=1.,g=.2,h=0.02)
#2.g_h filter measurement data: init state:x0 = 100, init state prediction delta:dx = 2,
#scale factor of next delta_x h=0.02, scale factor of next update g=0.2
#data = g_h_filter(data = measurements,x0=100,dx=2.,g=0.2,h=0.02)
#3.g_h filter measurement data: init state:x0 = 100, init state prediction delta:dx = 2,
#scale factor of next delta_x h=0.02, scale factor of next update g=0.2
#data = g_h_filter(data = measurements,x0=100,dx=2.,g=0.2,h=0.02)
#4.g_h filter measurement data: init state:x0 = 10, init state prediction delta:dx = 2,
#scale factor of next delta_x h=0.02, scale factor of next update g=0.2
#The result lag error always exists because when update dx, not considiering the derivative of dx, just
#use h to catch up dx, but no enough.
data = g_h_filter(data = measurements,x0=10,dx=0.,g=0.2,h=0.02)
#5.Adjust g: 0.1, 0.4, 0.8
#data = g_h_filter(data = measurements,x0=0,dx=5.,g=0.1,h=0.01)
#data = g_h_filter(data = measurements,x0=0,dx=5.,g=0.4,h=0.01)
data = g_h_filter(data = measurements,x0=0,dx=5.,g=0.8,h=0.01)



###############################################################################
#plot data and results
plt.figure()
measur = plt.scatter(x, measurements,alpha=0.6)
g_h_filter = plt.plot(x, data, color='red', linewidth=2.0, linestyle='--')
plt.legend([measur,g_h_filter[0]],['Measurement','g_h_filter'])

plt.show()
