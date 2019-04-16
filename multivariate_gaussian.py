#from scipy import stats
import scipy
x = [2.5,7.3]
mu = [2.0,7.0]
P = [
    [8.,0.],
    [0.,3.]
]

try:
    print('{:.4f}'.format(scipy.stats.multivariate_normal(mu, P).pdf(x)))
except:
    print('scipy version old, please update!')