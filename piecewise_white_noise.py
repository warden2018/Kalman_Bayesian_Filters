import numpy as np
import sympy 
from sympy import (init_printing, Matrix, MatMul, 
                   integrate, symbols)

init_printing(use_latex='mathjax')
var = symbols('\sigma^2_v')
dt = symbols('\Delta{t}')

# dim==4 (third order, constant jerk)
'''
F_k = Matrix([
                [1,dt,dt**2/2,dt**3/6],
                [0,1, dt,dt**2/2],
                [0,0,1,dt],
                [0,0,0,1]
            ])

v = Matrix([
                [dt**3/6],
                [dt**2/2],
                [dt],
                [1]
            ])
Q = v * var * v.T
Q = Q / var
MatMul(Q,var)
'''


# dim==3 (second order, constant acceleration)
'''
F_k = Matrix([
                [1,dt,dt**2/2],
                [0,1, dt],
                [0,0,1],
            ])

v = Matrix([
                [dt**2/2],
                [dt],
                [1]
            ])

Q = v * var * v.T
Q = Q / var
MatMul(Q,var)
'''

#dim==2 (first order, constant velocity)

F_k = Matrix([
                [1,dt],
                [0,1],
            ])

v = Matrix([
                [dt**2/2],
                [dt]
            ])

Q = v * var * v.T
Q = Q / var
MatMul(Q,var)

print(str(Q))

