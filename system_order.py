'''
1.A lower order filter can filter a higher order system if the process noise is 
set to a large value.
2.A higher order filter may work well if a certain set of data is given. But may 
diverge for significant periods of time. May not fit for different data set.
Higher order filter take acc into consideration which actually doesn't exits. Then
with update() step it will mistake measurement noise and add it into velocity and 
position.
3.Right order filter to the right order sysyem is important. But in the beginning,
 try lower order filter with large Q is also good.
 4.Don't evaluate a filter using the P matrix. Sometimes the matrix tells lies. Designer
 told something wrong about the system to the filter and the filter overconfident in 
 its performance. So use 3/sigma method and plot the residual to check.
'''

import matplotlib.pyplot as plt
import numpy as np
import logging
from numpy.random import randn
import math
from scipy.linalg import block_diag

#from Multivariable_Kalman import KalmanFilter
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
'''
We will see how the order of the system affects the filter's performance
'''

#plot filled using mean and covariance
#from true value and variance of one variable
def plot_residual(truth,data,cov,dt=1.,stds=1):
    num = len(truth)
    res = data - truth
    #print("residuals:",res)
    x = np.arange(0, num, dt)
    y1,y2 = [], []
    for i in range(num):
        y1.append(0-stds*math.sqrt(cov[i])) 
        y2.append(stds*math.sqrt(cov[i]))

    fig, ax = plt.subplots()
    ax.plot(x, y1, x, y2, color='black')
    ax.fill_between(x, y1, y2, where=y2 >y1, facecolor='yellow', alpha=0.5)
    ax.fill_between(x, y1, y2, where=y2 <=y1, facecolor='green', alpha=0.5)
    plt.plot(x, res, marker='s',color='red', linewidth=2.0, linestyle='--',label='residuals')

def plot_filter_output(xs,zs,data,dt=1.):
    plt.xlabel('Time')
    N = len(zs)
    x = np.arange(0,N,dt)
    ground_truth = plt.scatter(x, xs,s=200, facecolors='none', edgecolors='y',label='ground truth')
    measur = plt.scatter(x, zs,s=200, facecolors='none', edgecolors='b',label='measurements')
    kf_mean = plt.plot(x, data[:,0], marker='s',color='green', linewidth=2.0, linestyle='--',label='kf_mean')
    plt.legend(loc='lower right')
    plt.show()

def plot_all(truth,zs,data,cov,dt=1.,stds=1.):
    fig, axs = plt.subplots(2, 1)

    num = len(truth)
    res = data - zs
    x = np.arange(0, num, dt)
    y1,y2 = [], []
    for i in range(num):
        y1.append(0-stds*math.sqrt(cov[i])) 
        y2.append(stds*math.sqrt(cov[i]))

    axs[0].plot(x, y1, x, y2, color='black')
    axs[0].fill_between(x, y1, y2, where=y2 >y1, facecolor='yellow', alpha=0.5)
    axs[0].fill_between(x, y1, y2, where=y2 <=y1, facecolor='green', alpha=0.5)
    axs[0].plot(x, res, marker='s',color='red', linewidth=2.0, linestyle='--',label='residuals')


    axs[1].scatter(x, truth,s=200, facecolors='none', edgecolors='y',label='ground truth')
    axs[1].scatter(x, zs,s=200, facecolors='none', edgecolors='b',label='measurements')
    axs[1].plot(x, data, marker='s',color='green', linewidth=2.0, linestyle='--',label='kf_mean')
    plt.legend(loc='lower right')
    plt.show()

#constant velocity object
''''''
class Constant_velocity(object):
    def __init__(self, x0=0, vel=1., vel_noise=0.03):
        self.x = x0
        self.vel = vel
        self.noise_scale = noise_scale

    def update(self):
        self.vel += randn() * self.noise_scale
        self.x += self.vel
        
        return (self.x,self.vel)

''' End Class '''   

#constant acc system 
class Constant_acc(object):
    def __init__(self, x0=0, vel=1., acc=0.1, acc_noise=.1):
        self.x = x0
        self.vel = vel
        self.acc_base = acc
        self.acc_noise_scale = acc_noise
    
    def update(self):
        self.acc = self.acc_base + randn() * self.acc_noise_scale       
        self.vel += self.acc
        self.x += self.vel
        return (self.x, self.vel, self.acc)


#sensor on the object
#input pos has (x,y) position
''''''
class PosSensor(object):
    def __init__(self, pos, noise_scale=1.):
        self.pos_no_noise = pos
        self.data_len = len(pos)
        self.noise_scale = noise_scale
        self.pos_noise = []
    def read(self):
        for i in range(self.data_len):
            self.pos_noise.append(self.pos_no_noise[i] + randn() * self.noise_scale)
        return self.pos_noise

''' End Class ''' 

def ZeroOrderKF(R, Q, P=20):
    """ Create zero order Kalman filter.
    Specify R and Q as floats."""
    
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.x = np.array([0.])
    kf.R *= R
    kf.Q *= Q
    kf.P *= P
    kf.F = np.eye(1)
    kf.H = np.eye(1)
    return kf

    
def FirstOrderKF(R, Q, dt):
    """ Create first order Kalman filter. 
    Specify R and Q as floats."""
    
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.zeros(2)
    kf.P *= np.array([[100, 0], [0, 1]])
    kf.R *= R
    kf.Q = Q_discrete_white_noise(2, dt, Q)
    kf.F = np.array([[1., dt],
                     [0., 1]])
    kf.H = np.array([[1., 0]])
    return kf


def SecondOrderKF(R_std, Q, dt, P=100):
    """ Create second order Kalman filter. 
    Specify R and Q as floats."""
    
    kf = KalmanFilter(dim_x=3, dim_z=1)
    kf.x = np.zeros(3)
    kf.P[0, 0] = P
    kf.P[1, 1] = 1
    kf.P[2, 2] = 1
    kf.R *= R_std**2
    kf.Q = Q_discrete_white_noise(3, dt, Q)
    kf.F = np.array([[1., dt, .5*dt*dt],
                     [0., 1.,       dt],
                     [0., 0.,       1.]])
    kf.H = np.array([[1., 0., 0.]])
    return kf



def simulate_velocity_system(Q,R,dt,count):
    xs, zs = [], []
    obj = Constant_velocity(x0=0.,vel=0.5,vel_noise=Q)
    #xs:=(x,velocity)
    xs = np.array([obj.update() for _ in range(count)])
    obj_sensor = PosSensor(xs[:,0])
    zs = obj_sensor.read()
    return xs,zs
#

def simulate_acc_system(Q,R,dt,count):
    xs, zs = [], []
    obj = Constant_acc(x0=0.,vel=0.5,acc=0.1,acc_noise=Q)
    #xs:=(x,velocity)
    xs = np.array([obj.update() for _ in range(count)])
    obj_sensor = PosSensor(xs[:,0])
    zs = obj_sensor.read()
    return xs,zs 


N = 40
R = 6.
Q = 0.02
dt = 1.
xs, zs = [], []
#xs,zs = simulate_velocity_system(Q=Q,R=R,dt=dt,count=N)
xs,zs = simulate_acc_system(Q=Q,R=R,dt=dt,count=N)
print("xs",xs)
print("zs",zs)
tracker = FirstOrderKF(R,Q,dt=1.)
#tracker = SecondOrderKF(R,Q,dt=1.)
#tracker = ZeroOrderKF(R,Q)


#filter the whole measurement
mu, cov, _, _ = tracker.batch_filter(zs)

#Plot result 
x_cov = []
for each_cov in cov:
    x_cov.append(each_cov[0,0])
plot_all(xs[:,0],zs,mu[:,0],x_cov,dt=1.,stds=1.)
#plot_residual(zs,mu[:,0],x_cov,dt=1.,stds=1.)

#plt.legend(loc='lower right')
#plot_filter_output(xs[:,0],zs,mu,dt=1.)
#plt.show()






