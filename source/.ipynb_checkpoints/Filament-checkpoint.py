import numpy as np
import healpy as hp

class Filament:
	def __init__(self,center,sizes,angles):
		self.center 										= center
		self.angles											= angles
		self.sizes											= sizes
		self.rot_matrix										= self.rot_matrix()
		self.inv_rot_matrix									= np.linalg.inv(self.rot_matrix)
		self.XYZ_vertices									= self.XYZ_vertices()
		self.xyz_vertices									= self.xyz_vertices()
		self.xyz_faces										= self.xyz_faces()
		self.xyz_normal										= self.xyz_normal_to_faces()
		self.xyz_edges_vectors,self.xyz_edges_vectors_unit	= self.xyz_edge_vectors()
	
	def rot_matrix(self):
		#Given the alpha and beta euler angles, gamma is 0
		(a,b)			= self.angles
		ca = np.cos(a) ; cb = np.cos(b) 
		sa = np.sin(a) ; sb = np.sin(b)
		matrix      = np.zeros((3,3))
		matrix[0,0] = ca*cb
		matrix[1,0] = sa*cb
		matrix[2,0] = -sb
		matrix[0,1] = -sa
		matrix[1,1] = ca
		matrix[2,1] = 0
		matrix[0,2] = ca*sb
		matrix[1,2] = sa*sb
		matrix[2,2] = cb
		return matrix
		
	def XYZ_vertices(self):
		#return the eight vertices defined by the cuboid in the XYZ coordinates
		# Sx, Sy, Sz correspond to a,b,c of each filament
		(Sx,Sy,Sz) = self.sizes
		matrix = np.zeros((8,3))
		# X element of vertices 1,2,3,4
		matrix[0,0] = matrix[1,0] = matrix[2,0]=matrix[3,0] = +5*Sx
		# X element of vertices 5,6,7,8
		matrix[4,0] = matrix[5,0] = matrix[6,0]=matrix[7,0] = -5*Sx
		# Y element of vertices 1,4,5,8
		matrix[0,1] = matrix[3,1] = matrix[4,1]=matrix[7,1] = -5*Sy
		# Y elemeny of vertices 2,3,6,7
		matrix[1,1] = matrix[2,1] = matrix[5,1]=matrix[6,1] = +5*Sy
		# Z element of vertices 1,2,5,6
		matrix[0,2] = matrix[1,2] = matrix[4,2]=matrix[5,2] = +5*Sz
		# Z element of vertices 3,4,7,8
		matrix[2,2] = matrix[3,2] = matrix[6,2]=matrix[7,2] = -5*Sz
		return matrix

	def xyz_vertices(self):
		#rotate them according to euler angles
		vertices_rotated = np.zeros((8,3))
		for n in range(8):
			vertices_rotated[n,:] = np.matmul(self.rot_matrix,self.XYZ_vertices[n,:]) + self.center
		return vertices_rotated

	def xyz_faces(self):
		# np array with faces (6 faces, 4 points per face, 3 dimensions)
		faces	= np.zeros((6,4,3))
		faces[0,:,:] = self.xyz_vertices[np.array([0,1,2,3]),:]
		faces[1,:,:] = self.xyz_vertices[np.array([4,5,6,7]),:]
		faces[2,:,:] = self.xyz_vertices[np.array([0,3,7,4]),:]
		faces[3,:,:] = self.xyz_vertices[np.array([1,2,6,5]),:]
		faces[4,:,:] = self.xyz_vertices[np.array([0,1,5,4]),:]
		faces[5,:,:] = self.xyz_vertices[np.array([2,3,7,6]),:]
		return faces
	def xyz_edge_vectors(self):
		# these are the edge vectors, coming out from vertice 0. These are 2: (v1-v0) and (v3-v0)
		# 6 faces, 2 edges vectors, 3 dimensions
		edges			= np.zeros((6,2,3))
		edges_unit		= np.zeros((6,2,3))
		for k in range(6):
			edges[k,0,:]		= self.xyz_faces[k,1] - self.xyz_faces[k,0]
			edges[k,1,:]		= self.xyz_faces[k,3] - self.xyz_faces[k,0]
			edges_unit[k,0,:]	= edges[k,0] / np.linalg.norm(edges[k,0])
			edges_unit[k,1,:]	= edges[k,1] / np.linalg.norm(edges[k,1])
		return edges,edges_unit
	def xyz_normal_to_faces(self):
		# In the XYZ frame, for face A the normal is +x
		# for face B is -x, for face C is -y, for face D is +y
		# for face E is +z, for face F is -z
		normal_to_faces_unrotated	= np.array([[+1,0,0],[+1,0,0],[0,+1,0],[0,+1,0],[0,0,+1],[0,0,+1]]) 
		normal_to_faces 			= np.zeros((6,3))
		for n in range(6):
			normal_to_faces[n,:]	= np.matmul(self.rot_matrix,normal_to_faces_unrotated[n,:])
		return normal_to_faces
	def do_query_polygon(self,sky):
		# Do a query polygon for the 6 faces with an sky object
		# Return a list with the indices of the pixels with the shadow of the filament
		pix_filament	= np.array([])
		for n in range(6):
			# Get the pixels for each face
			pix_filament	= np.append(pix_filament,hp.query_polygon(sky.nside,self.xyz_faces[n,:,:],nest=False,inclusive=False))
		# Remove duplicates
		pix_filament	= np.unique(pix_filament).astype(int)
		return pix_filament
