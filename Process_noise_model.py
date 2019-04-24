from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from numpy import zeros, vstack, eye, array
from numpy.linalg import inv
from scipy.linalg import expm, block_diag

def order_by_derivative(Q, dim, block_size):
    """
    Given a matrix Q, ordered assuming state space
        [x y z x' y' z' x'' y'' z''...]
    return a reordered matrix assuming an ordering of
       [ x x' x'' y y' y'' z z' y'']
    This works for any covariance matrix or state transition function
    Parameters
    ----------
    Q : np.array, square
        The matrix to reorder
    dim : int >= 1
       number of independent state variables. 3 for x, y, z
    block_size : int >= 0
        Size of derivatives. Second derivative would be a block size of 3
        (x, x', x'')
    """

    N = dim * block_size

    D = zeros((N, N))

    Q = array(Q)
    for i, x in enumerate(Q.ravel()):
        f = eye(block_size) * x

        ix, iy = (i // dim) * block_size, (i % dim) * block_size
        D[ix:ix+block_size, iy:iy+block_size] = f

    return D


def Q_continuous_white_noise(dim,dt=1.,spectral_density=1.,block_size=1,
                            order_by_dim=True):
    '''         delta t
                ----
        Q=      -       F(t)QcF(t).Tdt
                - 
            ----
              0
        Qc is continuous noise. Meaning the accelertion is not constant/ It may variate
        with time passing by due to external and unmodeled forces. The integral above is 
        the integration of Qc projection based on F(t) over delta t period of time.
            -               -
            |   0   0   0    |
        Qc =|   0   0   0    | * [spectral density]
            |   0   0   1    |
            |                |
            -               -
        In practice, we don't know the spectral density of the noise. So the best way is to fine 
        tune the params in it until the filter's performance acts as we expect.
    Parameters
    ----------
    dim : int (2 or 3 or 4)
        dimension for Q, where the final dimension is (dim x dim). 
        2 is constant velocity (first order)
        3 is constant accerleration (second order)
        4 is constant jerk (third order)
    
    dt : float, default=1.0
        time step in whatever units the filter is using for time. The amount of time between innovations.
    
    spectral density : float, default=1.0
        spectral density for the continuous process
    
    block_size : int >=1
        If the state vector contains more than 1 dimension, such as a 3D vector [x,x',x'',y,y',y'',z,z',z'']T
        Then we need a block diagnal matrix.
    '''

    if not (dim==2 or dim==3 or dim==4):
        raise ValueError("dim must be either 2 or 3 or 4.")
    
    if dim == 2:
        Q = [
            [(dt**3)/3.,(dt**2)/2.],
            [(dt**2)/2.,        dt]
            ]
    elif dim == 3:
        Q = [
            [(dt**5)/20., (dt**4)/8., (dt**3)/6.],
             [ (dt**4)/8., (dt**3)/3., (dt**2)/2.],
             [ (dt**3)/6., (dt**2)/2.,        dt]
             ]
    else:
        Q = [[(dt**7)/252., (dt**6)/72., (dt**5)/30., (dt**4)/24.],
             [(dt**6)/72.,  (dt**5)/20., (dt**4)/8.,  (dt**3)/6.],
             [(dt**5)/30.,  (dt**4)/8.,  (dt**3)/3.,  (dt**2)/2.],
             [(dt**4)/24.,  (dt**3)/6.,  (dt**2/2.),   dt]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * spectral_density
    
    return order_by_derivative(array(Q),dim,block_size) * spectral_density




def Q_discrete_white_noise(dim,dt=1.,var=1.,block_size=1,order_by_dim=True):
    """
    Returns the Q matrix for the Discrete Constant White Noise
    Model. dim may be either 2, 3, or 4 dt is the time step, and sigma
    is the variance in the noise.
    Q is computed as the G * G^T * variance, where G is the process noise per
    time step. In other words, G = [[.5dt^2][dt]]^T for the constant velocity
    model.

    Parameters
    ----------
    dim : int (2, 3, or 4)
        dimension for Q, where the final dimension is (dim x dim)
    dt : float, default=1.0
        time step in whatever units your filter is using for time. i.e. the
        amount of time between innovations
    var : float, default=1.0
        variance in the noise
    block_size : int >= 1
        If your state variable contains more than one dimension, such as
        a 3d constant velocity model [x x' y y' z z']^T, then Q must be
        a block diagonal matrix.
    order_by_dim : bool, default=True
        Defines ordering of variables in the state vector. `True` orders
        by keeping all derivatives of each dimensions)
        [x x' x'' y y' y'']
        whereas `False` interleaves the dimensions
        [x y z x' y' z' x'' y'' z'']

    """
    if not (dim == 2 or dim == 3 or dim == 4):
        raise ValueError("dim must be between 2 and 4")

    if dim == 2:
        Q = [[.25*dt**4, .5*dt**3],
             [ .5*dt**3,    dt**2]]
    elif dim == 3:
        Q = [[.25*dt**4, .5*dt**3, .5*dt**2],
             [ .5*dt**3,    dt**2,       dt],
             [ .5*dt**2,       dt,        1]]
    else:
        Q = [[(dt**6)/36, (dt**5)/12, (dt**4)/6, (dt**3)/6],
             [(dt**5)/12, (dt**4)/4,  (dt**3)/2, (dt**2)/2],
             [(dt**4)/6,  (dt**3)/2,   dt**2,     dt],
             [(dt**3)/6,  (dt**2)/2 ,  dt,        1.]]

    if order_by_dim:
        return block_diag(*[Q]*block_size) * var
    return order_by_derivative(array(Q), dim, block_size) * var

