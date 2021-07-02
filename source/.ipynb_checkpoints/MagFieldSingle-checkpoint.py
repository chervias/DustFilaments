import Bpowspec
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def get_MagField(size,pixels):
	Bcube					= np.zeros((pixels,pixels,pixels,3))
	return Bcube
def get_interpolator(size,pixels,Bcube):
	real_units				= np.linspace(-0.5*size,+0.5*size,pixels)
	interp_fn			 	= RegularGridInterpolator((real_units,real_units,real_units),Bcube,method='linear',fill_value=None)
	return interp_fn