from io import *
import numpy as np
import segyio

class VelocityModel():
    """
    Base class to implement the velocity model in m/s

    Parameters
    ----------
    vp : float
     Velocity of P wave (m/s)
    shape : (int, ...)
     Size of the base along each axis
    file : str
     Path to the velocity model file
    """
    def __init__(self, vp=1500.0, shape=None, file=None):

        # create a velocity model
        if shape is not None:
            self.model = np.zeros(shape, dtype=np.float32)
            self.model[:] = vp

        # read the velocity model from a file .segy
        if file is not None:
            self.model = io.read_2D_segy(file)

    def shape(self):
        return self.model.shape
