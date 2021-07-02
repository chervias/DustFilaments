import healpy as hp
import numpy as np

nside = 65536

first = (1.57954016543073305634E+00 , 1.74014690528168314287E+00) 
second = (1.57953288605539055034E+00 ,1.74023396746201086671E+00) 
third = (1.57916395800194520049E+00 ,1.74511975856811507590E+00) 
fourth = (1.57917093697492050275E+00 , 1.74503672707541679365E+00)

vectors = np.zeros((4,3))

vectors[0,:] = hp.ang2vec(first[0],first[1])
vectors[0,:] = hp.ang2vec(second[0],first[1])
vectors[0,:] = hp.ang2vec(third[0],first[1])
vectors[0,:] = hp.ang2vec(fourth[0],first[1])

pixels = hp.query_polygon(nside,vectors,nest=True)

print(len(pixels))