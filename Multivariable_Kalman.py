from __future__ import absolute_import, division
import numpy as np
import random as rd
import matplotlib
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from copy import deepcopy
from math import log, exp, sqrt
import sys
import warnings
from numpy import dot, zeros, eye, isscalar, shape
import numpy.linalg as linalg
from scipy.linalg import block_diag
from filterpy.stats import plot_covariance_ellipse
#from Process_noise_model import Q_continuous_white_noise
#from Process_noise_model import Q_discrete_white_noise
from filterpy.common import Q_discrete_white_noise
from tools.plot import plot_covariance
import logging

#logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.DEBUG,
                    filename='./log/Multivariable_kalman.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)


def reshape_z(z, dim_z, ndim):
    """ ensure z is a (dim_z, 1) shaped vector"""

    z = np.atleast_2d(z)
    if z.shape[1] == dim_z:
        z = z.T

    if z.shape != (dim_z, 1):
        raise ValueError('z must be convertible to shape ({}, 1)'.format(dim_z))

    if ndim == 1:
        z = z[:, 0]

    if ndim == 0:
        z = z[0, 0]

    return z

#sensor on the robot
''''''
class PosSensor(object):
    def __init__(self, pos=(0, 0), vel=(0, 0), noise_std=1.):
        self.vel = vel
        self.noise_std = noise_std
        self.pos = [pos[0], pos[1]]
        
    def read(self):
        self.pos[0] += self.vel[0]
        self.pos[1] += self.vel[1]
        
        return [self.pos[0] + rd.random() * self.noise_std,
                self.pos[1] + rd.random() * self.noise_std]

''' End Class '''    

#Kalman Filter Implementation
''''''
class KalmanFilter(object):
    def __init__(self,dim_x,dim_z,dim_u=0):
        if dim_x < 1:
            raise ValueError('dim_x must be 1 or larger.')
        if dim_z < 1:
            raise ValueError('dim_z must be 1 or larger.')
        if dim_u < 0:
            raise ValueError('dim_u must be 0 or larger.')

        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_u = dim_u

        self.x = zeros((dim_x,1))                   #state
        self.P = eye(dim_x)                         #state uncertainty covariance
        self.Q = eye(dim_x)                         #process uncertainty
        self.B = None                               #control transmition matrix
        self.F = eye(dim_x)                         #state trainsmition matrix

        self.H = zeros((dim_z,dim_x))               #measurement function
        self.R = eye(dim_z)                        #measurement uncertainty
        self._alpha_sq = 1.                         #fading memory control
        self.M = np.zeros((dim_z,dim_z))            #process-measurement cross correlation
        self.z = np.array([[None]*self.dim_z]).T    #

        #Kalman gain and residual
        self.K = np.zeros((dim_x,dim_z))            #Kalman gain
        self.y = zeros((dim_z,1))                   #residual between predicted x and measurement
        self.S = np.zeros((dim_z,dim_z))            #System uncertainty
        self.SI = np.zeros((dim_z,dim_z))           #inverse system uncertainty


        #Identity matrix. 
        self._I = np.eye(dim_x)
        #These will be copy of x and P after update is called
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()

        # Only computed only if requested via property
        self._log_likelihood = log(sys.float_info.min)
        self._likelihood = sys.float_info.min
        self._mahalanobis = None

        self.inv = np.linalg.inv
        

    def predict(self,u=None,B=None,F=None,Q=None):
        '''
        Predict prior using the Kalman filter state propagation equations.
        '''
        if B is None:
            B = self.B
        if F is None:
            F = self.F
        if Q is None:
            Q = self.Q
        elif isscalar(Q):
            Q = eye(self.dim_x)*Q
        
        if B is not None and u is not None:
            self.x = dot(F,self.x) + dot(B,u)
        else:
            self.x = dot(F,self.x)

        #P=FPF' + Q
        self.P = self._alpha_sq * dot(dot(F,self.P),F.T) + Q

        #save prior
        self.x_prior = self.x.copy()
        
        self.P_prior = self.P.copy()

    def update(self,z,R=None,H=None):
        '''
        Add a new measurement to the Kalman filter.

        If z is None, nothing to compute. However, x_post and P_post are
        updated with the prior(x_prior and P_prior), and self.z is set to None.
        '''
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

        if z is None:
            self.z = np.array([None]*self.dim_z).T
            self.x_post = self.x.copy()
            self.P_post = self.P.copy()
            self.y = zeros((self.dim_z,1))
            return
        #reshape z as 2*N matrix
        z = reshape_z(z,self.dim_z,self.x.ndim)
        #for i in range(N):
        #    logger.debug("UPDATE: z after reshape: x:%f y:%f",z[])
        
        if R is None:
            R = self.R
        elif isscalar(R):
            R = eye(self.dim_z) * R

        if H is None:
            H = self.H
        
        #residual between measurement and prediction
        self.y = z - dot(H,self.x)

        #common subexpression for speed
        PHT = dot(self.P,H.T)
        #S = HpH' + R
        #project system uncertainty into measurement space
        self.S = dot(H,PHT) + R
        self.SI = self.inv(self.S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        self.K = dot(PHT, self.SI)

        # x = x + Ky
        # predict new x with residual scaled by the kalman gain
        self.x = self.x + dot(self.K, self.y)


        # P = (I-KH)P(I-KH)' + KRK'
        # This is more numerically stable
        # and works for non-optimal K vs the equation
        # P = (I-KH)P usually seen in the literature.

        I_KH = self._I - dot(self.K, H)
        self.P = dot(dot(I_KH, self.P), I_KH.T) + dot(dot(self.K, R), self.K.T)

        # save measurement and posterior state
        self.z = deepcopy(z)
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()


    def batch_filter(self,zs,Fs=None,Qs=None,Hs=None,
                    Rs=None,Bs=None,us=None,update_first=False,
                    saver=None):
        ''' Batch processes a sequences of measurements.
        '''
        #pylint: disable=too-many-statements
        n = np.size(zs, 0)
        if Fs is None:
            Fs = [self.F] * n
        if Qs is None:
            Qs = [self.Q] * n
        if Hs is None:
            Hs = [self.H] * n
        if Rs is None:
            Rs = [self.R] * n
        if Bs is None:
            Bs = [self.B] * n
        if us is None:
            us = [0] * n

        # mean estimates from Kalman Filter
        if self.x.ndim == 1:
            means = zeros((n, self.dim_x))
            means_p = zeros((n, self.dim_x))
        else:
            means = zeros((n, self.dim_x, 1))
            means_p = zeros((n, self.dim_x, 1))

        # state covariances from Kalman Filter
        covariances = zeros((n, self.dim_x, self.dim_x))
        covariances_p = zeros((n, self.dim_x, self.dim_x))

        if update_first:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.update(z, R=R, H=H)
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                self.predict(u=u, B=B, F=F, Q=Q)
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                if saver is not None:
                    saver.save()
        else:
            for i, (z, F, Q, H, R, B, u) in enumerate(zip(zs, Fs, Qs, Hs, Rs, Bs, us)):

                self.predict(u=u, B=B, F=F, Q=Q)
                logger.debug("Kalman filter x-prior is: %s",np.array_repr(tracker.x).replace('\n', ''))
                logger.debug("Kalman filter P is: %s",np.array_repr(tracker.P).replace('\n', ''))
                means_p[i, :] = self.x
                covariances_p[i, :, :] = self.P

                self.update(z, R=R, H=H)
                logger.debug("Kalman filter z is: %s",np.array_repr(z).replace('\n', ''))
                logger.debug("Kalman filter x-post is: %s",np.array_repr(tracker.x).replace('\n', ''))
                logger.debug("Kalman filter P-post is: %s",np.array_repr(tracker.P).replace('\n', ''))
                means[i, :] = self.x
                covariances[i, :, :] = self.P

                if saver is not None:
                    saver.save()

        return (means, covariances, means_p, covariances_p)

''' End Class '''    


if __name__== '__main__':
    #sensor data prepare
    R_std = 3.5
    #R_std = 1.4
    Q_std = 1.2
    N = 30
    pos = (0,0)
    vel = (2.,2.)
    dt = 1.
    sensor = PosSensor(pos, vel, noise_std=R_std)
    zs = np.array([sensor.read() for _ in range(N)])

    logger.debug("mesurement every %s seconds, total number is : %d" % (dt,N))
   
    #initialize kalman tracker
    tracker = KalmanFilter(dim_x=4, dim_z=2)
    logger.debug("Kalman filter dimension is: %d, measurement dimension is: %d",tracker.dim_x,tracker.dim_z)
    tracker.F = np.array([[1, dt, 0,  0],
                        [0,  1, 0,  0],
                        [0,  0, 1, dt],
                        [0,  0, 0,  1]])
    logger.debug("Kalman filter F is: %s",np.array_repr(tracker.F).replace('\n', ''))                  
    q = Q_discrete_white_noise(dim=2, dt=dt, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    logger.debug("Kalman filter Q is: %s",np.array_repr(tracker.Q).replace('\n', ''))
    tracker.H = np.array([ 
                            [1/0.3048, 0, 0,        0],
                            [0,        0, 1/0.3048, 0]
                        ])
    logger.debug("Kalman filter H is: %s",np.array_repr(tracker.H).replace('\n', ''))
    tracker.R = np.eye(2) * R_std**2
    logger.debug("Kalman filter R is: %s",np.array_repr(tracker.R).replace('\n', ''))
    tracker.x = np.array([[0, 0, 0, 0]]).T
    logger.debug("Kalman filter init x is: %s",np.array_repr(tracker.x).replace('\n', ''))
    tracker.P = np.eye(4) * 500.
    logger.debug("Kalman filter init P is: %s",np.array_repr(tracker.P).replace('\n', ''))

    #filter the whole measurement
    mu, cov, _, _ = tracker.batch_filter(zs)
    #change measurement unit from feet to meter!
    zs *= .3048
    #Plot result 
    '''
    fig,axs = plt.subplots(1,1)
    axs[0].plot(zs[:,0],zs[:,1],mu[0,:], mu[2,:])
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].grid(True)
    fig.tight_layout()
    plt.show()
    '''
    plt.xlabel('X position')
    plt.ylabel('Y position')
    plt.title('Kalman filter')
    measur = plt.scatter(zs[:,0], zs[:,1],s=200, facecolors='none', edgecolors='b',label='measurements')
    '''
    plt.plot(x1,y, marker='s', linestyle='-', color='y',label='tv')
    plt.plot(x2,y, marker='o', linestyle='--', color='g', label='raddio')
    '''
    kf_mean = plt.plot(mu[:,0], mu[:,2], marker='s',color='red', linewidth=2.0, linestyle='--',label='kf_mean')
    for x, P in zip(mu, cov):
        # covariance of x and y
        cov = np.array([[P[0, 0], P[2, 0]], 
                        [P[0, 2], P[2, 2]]])
        mean = (x[0, 0], x[2, 0])
        plot_covariance(mean, cov=cov, show_semiaxis=True,fc='w',edgecolor='r', std=3, alpha=0.5)
    #kf_cov = plt.plot(x, ux, color='blue', linewidth=2.0, linestyle='--')
    
    #postteroir = plt.scatter(mu[0,:], mu[2,:],alpha=1)

    #posteroir = plt.scatter(mu[:,0], mu[:,2],alpha=1)
    #plt.legend([measur,kf_mean],['Measurement','kf_mean'])
    plt.legend(loc='lower right')
    plt.show()