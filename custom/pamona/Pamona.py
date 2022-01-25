'''
author: Kai Cao
email: caokai@amss.ac.cn
'''

import time
import numpy as np
import random
import ot
from ot.bregman import sinkhorn
from ot.gromov import init_matrix, gwggrad
from ot.partial import gwgrad_partial
from numpy import linalg as la
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import block_diag

from pamona.eval import *
from pamona.utils import *
from pamona.visualization import visualize

class Pamona(object):

	"""
	Pamona software for single-cell mulit-omics data integration
	Preprint at https://doi.org/10.1101/2020.11.03.366146

	=============================
			parameters:
	=============================
	dataset: list of numpy array, [dataset1, dataset2, ...] (n_datasets, n_samples, n_features). 
	--list of datasets to be integrated, in the form of a numpy array.

	n_shared: int, default as the cell number of the smallest dataset. 
	--shared cell number between datasets.

	epsilon: float, default as 0.001. 
	--the regularization parameter of the partial-GW framework.

	n_neighbors: int, default as 10. 
	--the number of neighborhoods of the k-nn graph.

	Lambda: float, default as 1.0. 
	--the parameter to make a trade-off between aligning corresponding cells and preserving the local geometries

	output_dim: int, default as 30. 
	--output dimension of the common embedding space after the manifold alignment

	M (optionally): numpy array , default as None. 
	--disagreement matrix of prior information.
	
	=============================
			Functions:
	=============================
	run_Pamona(dataset) 				
	--find correspondence between datasets, align multi-omics data in a common embedded space

	entropic_gromov_wasserstein(self, C1, C2, p, q, m, M, loss_fun)			
	--find correspondence between datasets using partial GW

	project_func(self, data)	
	--project multi-omics data into a common embedded space

	Visualize(data, integrated_data, datatype, mode)	
	--Visualization
	
	test_labelTA(data1, data2, type1, type2) 		
	test label transfer accuracy

	alignment_score(data1_shared, data2_shared, data1_specific=None, data2_specific=None) 		
	test alignment score

	=============================
			Examples:
	=============================
	input: numpy arrays with rows corresponding to samples and columns corresponding to features
	output: integrated numpy arrays
	>>> from pamona import Pamona
	>>> import numpy as np
	>>> data1 = np.loadtxt("./scGEM/expression.txt")
	>>> data2 = np.loadtxt("./scGEM/methylation.txt")
	>>> type1 = np.loadtxt("./scGEM/expression_type.txt")
	>>> type2 = np.loadtxt("./scGEM/methylation_type.txt")
	>>> type1 = type1.astype(np.int)
	>>> type2 = type2.astype(np.int)
	>>> uc = Pamona.Pamona()
	>>> integrated_data = uc.fit_transform(dataset=[data1,data2])
	>>> uc.test_labelTA(integrated_data[0], integrated_data[1], type1, type2)
	>>> uc.Visualize([data1,data2], integrated_data, [type1,type2], mode='PCA')
	===============================
	"""

	def __init__(self, n_shared=None, M=None, n_neighbors=10, epsilon=0.001, Lambda=1.0, virtual_cells=1, \
		output_dim=30, max_iter=1000, tol=1e-9, manual_seed=666, mode="distance", metric="minkowski", verbose=True):

		self.n_shared = n_shared
		self.M = M
		self.n_neighbors = n_neighbors
		self.epsilon = epsilon
		self.Lambda = Lambda
		self.virtual_cells = virtual_cells
		self.output_dim = output_dim
		self.max_iter = max_iter
		self.tol = tol
		self.manual_seed = manual_seed
		self.mode = mode
		self.metric = metric
		self.verbose = verbose
		self.dist = []
		self.Gc = []
		self.T = []

	def run_Pamona(self, data):

		print("Pamona start!")
		time1 = time.time()

		init_random_seed(self.manual_seed)

		sampleNo = []
		Max = []
		Min = []
		p = []
		q = []
		n_datasets = len(data)

		for i in range(n_datasets):
			sampleNo.append(np.shape(data[i])[0])
			self.dist.append(Pamona_geodesic_distances(data[i], self.n_neighbors, mode=self.mode, metric=self.metric))

		for i in range(n_datasets-1):
			Max.append(np.maximum(sampleNo[i], sampleNo[-1])) 
			Min.append(np.minimum(sampleNo[i], sampleNo[-1]))

		if self.n_shared is None:
			self.n_shared = Min

		for i in range(n_datasets-1):
			if self.n_shared[i] > Min[i]:
				self.n_shared[i] = Min[i]
			p.append(ot.unif(Max[i])[0:len(data[i])])
			q.append(ot.unif(Max[i])[0:len(data[-1])])

		for i in range(n_datasets-1):
			if self.M is not None:
				T_tmp = self.entropic_gromov_wasserstein(self.dist[i], self.dist[-1], p[i], q[i], \
					self.n_shared[i]/Max[i]-1e-15, self.M[i])
			else:
				T_tmp = self.entropic_gromov_wasserstein(self.dist[i], self.dist[-1], p[i], q[i], \
				self.n_shared[i]/Max[i]-1e-15)
			self.T.append(T_tmp) 
			self.Gc.append(T_tmp[:len(p[i]), :len(q[i])])

		integrated_data = self.project_func(data)

		time2 = time.time()
		print("Pamona Done! takes {:f}".format(time2-time1), 'seconds')

		return integrated_data, self.T


	def entropic_gromov_wasserstein(self, C1, C2, p, q, m, M=None, loss_fun='square_loss'):

		C1 = np.asarray(C1, dtype=np.float32)
		C2 = np.asarray(C2, dtype=np.float32)

		T0 = np.outer(p, q)  # Initialization

		dim_G_extended = (len(p) + self.virtual_cells, len(q) + self.virtual_cells)
		q_extended = np.append(q, [(np.sum(p) - m) / self.virtual_cells] * self.virtual_cells)
		p_extended = np.append(p, [(np.sum(q) - m) / self.virtual_cells] * self.virtual_cells)

		q_extended = q_extended/np.sum(q_extended)
		p_extended = p_extended/np.sum(p_extended)

		constC, hC1, hC2 = init_matrix(C1, C2, p, q, loss_fun)

		cpt = 0
		err = 1

		while (err > self.tol and cpt < self.max_iter):

			Gprev = T0
			# compute the gradient
			if abs(m-1)<1e-10: # full match
				Ck = gwggrad(constC, hC1, hC2, T0)
			else: # partial match
				Ck = gwgrad_partial(C1, C2, T0)
		
			if M is not None:
				Ck = Ck*M

			Ck_emd = np.zeros(dim_G_extended)
			Ck_emd[:len(p), :len(q)] = Ck
			Ck_emd[-self.virtual_cells:, -self.virtual_cells:] = 100*np.max(Ck_emd)
			Ck_emd = np.asarray(Ck_emd, dtype=np.float64)

			# T = sinkhorn(p, q, Ck, epsilon, method = 'sinkhorn')
			T = sinkhorn(p_extended, q_extended, Ck_emd, self.epsilon, method = 'sinkhorn')
			T0 = T[:len(p), :len(q)]

			if cpt % 10 == 0:
				err = np.linalg.norm(T0 - Gprev)

				if self.verbose:
					if cpt % 200 == 0:
						print('{:5s}|{:12s}'.format(
							'Epoch.', 'Loss') + '\n' + '-' * 19)
					print('{:5d}|{:8e}|'.format(cpt, err))

			cpt += 1
	
		return T

	def project_func(self, data):

		n_datasets = len(data)
		H0 = []
		L = []
		for i in range(n_datasets-1):
			self.Gc[i] = self.Gc[i]*np.shape(data[i])[0]

		for i in range(n_datasets):    
			graph_data = kneighbors_graph(data[i], self.n_neighbors, mode="distance")
			graph_data = graph_data + graph_data.T.multiply(graph_data.T > graph_data) - \
				graph_data.multiply(graph_data.T > graph_data)
			W = np.array(graph_data.todense())
			index_pos = np.where(W>0)
			W[index_pos] = 1/W[index_pos] 
			D = np.diag(np.dot(W, np.ones(np.shape(W)[1])))
			L.append(D - W)

		Sigma_x = []
		Sigma_y = []
		for i in range(n_datasets-1):
			Sigma_y.append(np.diag(np.dot(np.transpose(np.ones(np.shape(self.Gc[i])[0])), self.Gc[i])))
			Sigma_x.append(np.diag(np.dot(self.Gc[i], np.ones(np.shape(self.Gc[i])[1]))))

		S_xy = self.Gc[0]
		S_xx = L[0] + self.Lambda*Sigma_x[0]
		S_yy = L[-1] + self.Lambda*Sigma_y[0]
		for i in range(1, n_datasets-1):
			S_xy = np.vstack((S_xy, self.Gc[i]))
			S_xx = block_diag(S_xx, L[i] + self.Lambda*Sigma_x[i])
			S_yy = S_yy + self.Lambda*Sigma_y[i]

		v, Q = la.eig(S_xx)
		v = v + 1e-12   
		V = np.diag(v**(-0.5))
		H_x = np.dot(Q, np.dot(V, np.transpose(Q)))

		v, Q = la.eig(S_yy)
		v = v + 1e-12      
		V = np.diag(v**(-0.5))
		H_y = np.dot(Q, np.dot(V, np.transpose(Q)))

		H = np.dot(H_x, np.dot(S_xy, H_y))
		U, sigma, V = la.svd(H)

		num = [0]
		for i in range(n_datasets-1):
			num.append(num[i]+len(data[i]))

		U, V = U[:,:self.output_dim], np.transpose(V)[:,:self.output_dim]

		fx = np.dot(H_x, U)
		fy = np.dot(H_y, V)

		integrated_data = []
		for i in range(n_datasets-1):
			integrated_data.append(fx[num[i]:num[i+1]])

		integrated_data.append(fy)

		return integrated_data

	def Visualize(self, data, integrated_data, datatype=None, mode='PCA'):
		if datatype == None:
			visualize(data, integrated_data, mode=mode)
		else:
			visualize(data, integrated_data, datatype, mode=mode)

	def test_LabelTA(self, data1, data2, type1, type2):
		label_transfer_acc = test_transfer_accuracy(data1,data2,type1,type2)
		print("label transfer accuracy:")
		print(label_transfer_acc)

	def alignment_score(self, data1_shared, data2_shared, data1_specific=None, data2_specific=None):
		alignment_sco = test_alignment_score(data1_shared, data2_shared, data1_specific=data1_specific, data2_specific=data2_specific)
		print("alignment score:")
		print(alignment_sco)


# if __name__ == '__main__':
# 	### example
# 	data1 = np.loadtxt("./PBMC/ATAC_scaledata.txt")
# 	data2 = np.loadtxt("./PBMC/RNA_scaledata.txt")
# 	type1 = np.loadtxt("./PBMC/ATAC_type.txt")
# 	type2 = np.loadtxt("./PBMC/RNA_type.txt")
# 	data1=zscore_standardize(np.asarray(data1))
# 	data2=zscore_standardize(np.asarray(data2))

# 	type1 = type1.astype(np.int)
# 	type2 = type2.astype(np.int)
# 	data = [data1,data2]
# 	datatype = [type1,type2]

# 	M = []
# 	n_datasets = len(data)
# 	for k in range(n_datasets-1):
# 	    M.append(np.ones((len(data[k]), len(data[-1]))))
# 	    for i in range(len(data[k])):
# 	        for j in range(len(data[-1])):
# 	            if datatype[k][i] == datatype[-1][j]:
# 	                M[k][i][j] = 0.5

# 	Pa = Pamona(n_shared=[1649], M=M, n_neighbors=30)
# 	integrated_data, T = Pa.run_Pamona(data)

# 	####PBMC
# 	index1 = np.argwhere(type1==0).reshape(1,-1).flatten()    
# 	index2 = np.argwhere(type1==1).reshape(1,-1).flatten()
# 	index3 = np.argwhere(type1==2).reshape(1,-1).flatten()
# 	index4 = np.argwhere(type1==3).reshape(1,-1).flatten()
# 	shared1 = np.hstack((index1, index2))
# 	shared1 = np.hstack((shared1, index3))
# 	shared1 = np.hstack((shared1, index4))
# 	print(np.shape(shared1))

# 	index1 = np.argwhere(type1==4).reshape(1,-1).flatten()    
# 	index2 = np.argwhere(type1==5).reshape(1,-1).flatten()
# 	specific1 = np.hstack((index1, index2))
# 	print(np.shape(specific1))

# 	index1 = np.argwhere(type2==0).reshape(1,-1).flatten()    
# 	index2 = np.argwhere(type2==1).reshape(1,-1).flatten()
# 	index3 = np.argwhere(type2==2).reshape(1,-1).flatten()
# 	index4 = np.argwhere(type2==3).reshape(1,-1).flatten()
# 	shared2 = np.hstack((index1, index2))
# 	shared2 = np.hstack((shared2, index3))
# 	shared2 = np.hstack((shared2, index4))
# 	print(np.shape(shared2))

# 	index1 = np.argwhere(type2==6).reshape(1,-1).flatten()    
# 	index2 = np.argwhere(type2==7).reshape(1,-1).flatten()
# 	index3 = np.argwhere(type2==8).reshape(1,-1).flatten()
# 	index4 = np.argwhere(type2==9).reshape(1,-1).flatten()
# 	specific2 = np.hstack((index1, index2))
# 	specific2 = np.hstack((specific2, index3))
# 	specific2 = np.hstack((specific2, index4))
# 	print(np.shape(specific2))

# 	Pa.alignment_score(integrated_data[0][shared1], integrated_data[-1][shared2], \
# 	    data1_specific=integrated_data[0][specific1] , data2_specific=integrated_data[-1][specific2])
# 	Pa.test_LabelTA(integrated_data[0][shared1],integrated_data[-1],type1[shared1],type2)

# 	Pa.Visualize([data1,data2], integrated_data, datatype=datatype, mode='UMAP')
