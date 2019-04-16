import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy.random import randn

x = np.arange(-100, 100, .1)
'''
discrete_bayes calculates the 
'''
class One_dimensional_kf():
    def __init__(self,init_gaussian_u,init_gaussian_sg,time_interval=1.,pu=1.):
        self.process_var = 2. #process uncertainty
        self.process_u = 1.0
        #self.sensor_var = 4.5 #sensor ucertainty
        self.sensor_var = .5 #sensor ucertainty
        #use pos and var to describe the gaussian distribution
        self.pos = 50.
        self.var = 0.
        self.dt = time_interval
        #x = norm(loc = init_gaussian_u, scale = init_gaussian_sg)#init status variable gausian distribution
        #process_model = norm(process_u * dt,process_var)

    #update pos and var
    def predict(self):
        dx = self.process_u * self.dt#process_u is considered as the expectation of movement(it is a distribution)
        #the sum of two gaussian distributed variables is another gaussian distributed varible with u=u1+u2, var=var1+var2
        self.pos = self.pos + dx
        self.var = self.var + self.process_var
        return self.pos
    #update 
    def update(self,z):
        #In discrete_bayes, likelihood mutiplies prior is actually equal to below two steps because of the attributes of gaussian 
        self.pos = (self.var * z + self.sensor_var * self.pos) / (self.var + self.sensor_var)#z is considered as the expectation of sensor reading(it is a distribution)
        self.var = (self.var * self.sensor_var) / (self.var + self.sensor_var)
        return self.pos

    def sensor_generator(self,x0,N):
        zs = []
        addition = 0.
        for i in range(N):
            zs.append(x0 + addition + randn()*self.sensor_var**.5)#randn()*std_var
            addition += self.process_u * self.dt
        return zs
        

#result and visualization
num = 25
x = np.arange(0, num, 1)
kf = One_dimensional_kf(0.,400.)
zs = kf.sensor_generator(0,num)
px = np.zeros(num)
ux = np.zeros(num)

for i in range(num):
    px[i] = kf.predict()
    ux[i] = kf.update(zs[i])


plt.figure()
measur = plt.scatter(x, zs,alpha=0.6)
kf_prediction = plt.plot(x, px, color='red', linewidth=2.0, linestyle='--')
kf_update = plt.plot(x, ux, color='blue', linewidth=2.0, linestyle='--')
plt.legend([measur,kf_prediction[0],kf_update[0]],['Measurement','prediction','update'])

plt.show()




