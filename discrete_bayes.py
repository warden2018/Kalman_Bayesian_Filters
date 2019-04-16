import numpy as np
#initial belief It is a probability prior. Also called probability distribution.
#Bayesian statistics takes past information (the prior) into account.This belief does not 
#consider sensor reading into account!
belief = np.array([1./10]*10)
#1 is door, and 0 is wall Think it as a MAP
hallway = np.array([1,1,0,0,0,0,0,0,1,0])

#First read from the sensor: the dog is in front of a door! So assign each door with 0.3333. (sensor is very accurate!)
#Right now have no idea of which door!
#what if sensor is noisy? We can't believe it but should use the information
#How to use the information? 
#belief = np.array([.31, .31, .01, .01, .01, .01, .01, .01, .31, .01])
def normalize(v):
    norm = np.linalg.norm(v,1)
    if norm == 0: 
       return v
    #print("normalize triggered: v/norm = ",v / norm)
    #print("sum of v/norm is:",sum(v/norm))
    return v / norm
    

'''def update_belief(hall,b,z,z_prob):
    scale = z_prob / (1. - z_prob)
    b[hall==z] *= scale
    #print("belief before normalize: ",b)
    b = normalize(b)
    #print("belief after normalize: ",b)
    return b
    '''
#Incorporating movement:
'''
Another term is system propagation. It refers to how the state of the system changes over time. 
For filters, time is usually a discrete step, such as 1 second. For our dog tracker the system state is the position of the dog, 
and the state evolution is the position after a discrete amount of time has passed.
We model the system behavior with process model.
So the movement data we get is from sensors with noises. The uncertainty of the reading needs to be considered. 
So predict is the sum of x and dx . These two varibles are independently and convolution is used for getting the 
new distibution!
'''
def predict(pdf,offset,kernel):
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i  - offset + (width - k)) % N
            prior += pdf[index] * kernel[k]
    return prior

#actually more general way is to calculate the likelihood and update
#So function update_belief no longer used
def lh_hallway(hall,z,z_prob):
    '''
    compute likelihood that a measurement matches positions in the hallway
    '''
    try:
        scale = z_prob / (1 - z_prob)
    except ZeroDivisionError:
        scale = 1e8
    likelihood = np.ones(len(hall))
    likelihood[hall==z] *= scale
    return likelihood

def update(lh,prior):
    return normalize(lh * prior)

#belief = update_belief(hallway,belief,z=1,z_prob=.75)
kernel = (.1,.8,.1)
prior = predict(posterior,1,kernel)
likelihood = lh_hallway(prior,z=1,z_prob=0.75)
posterior = update(likelihood,prior)
print("updated belief are:",posterior)
