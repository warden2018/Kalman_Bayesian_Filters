import numpy as np
import sympy 
from sympy import (init_printing, Matrix, MatMul, 
                   integrate, symbols)
'''
z=[1,2,3,4]
#z = [[1,2,3,4],[0,3],[0,1,4,7]]
a = np.atleast_2d(z)
print("a:",a)
print("a.shape:",a.shape)
'''


init_printing(use_latex='mathjax')
dt, phi = symbols('\Delta{t} \Phi_s')

# dim==4 (third order, constant jerk)

F_k = Matrix([
                [1,dt,dt**2/2,dt**3/6],
                [0,1, dt,dt**2/2],
                [0,0,1,dt],
                [0,0,0,1]
            ])
Q_c = Matrix([
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,0],
                [0,0,0,1]
            ]) * phi


# dim==3 (second order, constant acceleration)
'''
F_k = Matrix([
                [1,dt,dt**2/2],
                [0,1,       dt],
                [0,0,1]
            ])
Q_c = Matrix([
                [0,0,0],
                [0,0,0],
                [0,0,1]
            ]) * phi
'''
#dim==2 (first order, constant velocity)
'''
F_k = Matrix([
                [1,dt],
                [0,1,],
            ])
Q_c = Matrix([
                [0,0],
                [0,1],
            ]) * phi
'''
#dim==1 (0 order, constant position)
'''
F_k = Matrix([
                [1]
            ])
Q_c = Matrix([
                [1]
            ]) * phi
'''

Q = integrate(F_k*Q_c*F_k.T,(dt,0,dt))
Q = Q/phi
MatMul(Q,phi)
print(str(Q))
#print("Q:",Q)

