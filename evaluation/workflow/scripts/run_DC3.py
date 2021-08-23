r"""
Modified from https://github.com/SUwonglab/DC3/blob/360207818eafb8d246e2113f1aeacbdea2ee9140/dc3.py

Added and modified lines are tagged with "# MOD" at the end
"""

import os  # MOD
import sys  # MOD
import numpy as np
import pickle
from sklearn.decomposition import NMF
from sklearn.feature_selection import  SelectFdr,SelectPercentile,f_classif
from numpy import linalg as LA
import math
import argparse
import itertools
import scipy.io as scio
import pandas as pd
import time
import scipy.stats as stats
from statsmodels.stats.weightstats import ttest_ind
from scipy  import sparse
from scipy.stats.stats import pearsonr
import scipy.sparse  # MOD
import h5py  # MOD
import yaml  # MOD
import networkx as nx  # MOD

def quantileNormalize(df_input):
    df = df_input.copy()
    #compute rank
    dic = {}
    for col in df:
        dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
        t = np.searchsorted(np.sort(df[col]), df[col])
        df[col] = [rank[i] for i in t]
    return df

def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

def read_h5ad(file):  # MOD
	with h5py.File(file, "r") as f:  # MOD
		if isinstance(f["X"], h5py.Group):  # MOD
			X = scipy.sparse.csr_matrix((
				f["X"]["data"][...],
				f["X"]["indices"][...],
				f["X"]["indptr"][...]
			), shape=f["X"].attrs["shape"]).T  # MOD
		else:  # MOD
			X = f["X"][...].T  # MOD
		obs_names = f["obs"][f["obs"].attrs["_index"]][...]  # MOD
		var_names = f["var"][f["var"].attrs["_index"]][...]  # MOD
	return X, obs_names, var_names  # MOD


parser = argparse.ArgumentParser(description='dcHiChIP for HiChIP deconvolution and joint clustering scRNA-seq and scATAC-seq.')
parser.add_argument('-k', dest='k', type=int, default=2, help='the number of clusters')
parser.add_argument('-E', type=str, help='the location of singlecell RNA-seq data (.h5ad)')  # MOD
parser.add_argument('-O',type=str, help='the location of singlecell ATAC-seq data (.h5ad')  # MOD
parser.add_argument('-hichip', type=str, help='the location of hichip enhancer-promoter interactions (.graphml[.gz])')  # MOD
parser.add_argument('-lambda1', dest='lambda1', type=float, help='lambda1, hyperparameters to control the term NMF for E')
parser.add_argument('-lambda2', dest='lambda2', type=float, help='lambda2, hyperparameters to control the coupled term')
parser.add_argument('-seed', dest='seed', type=int, help='random seed')  # MOD
parser.add_argument('-E_latent', type=str, help='output of E latent matrix')  # MOD
parser.add_argument('-O_latent', type=str, help='output of O latent matrix')  # MOD
parser.add_argument('-run_info', type=str, help='output of run info')  # MOD
parser.add_argument('-out_dir', type=str, help='additional output directory')  # MOD

args = parser.parse_args()

np.random.seed(args.seed)  # MOD
rep=50

print "Loading data..."

K=args.k

#loading expression and chromatin accessibitily matrix and row labels.
E, E_cell, E_symbol = read_h5ad(args.E)  # MOD
O, O_cell, P_symbol = read_h5ad(args.O)  # MOD
A = nx.read_graphml(args.hichip)  # MOD
used_symbols = set(A.nodes)  # MOD
E_idx = np.where([item in used_symbols for item in E_symbol])[0]  # MOD
O_idx = np.where([item in used_symbols for item in P_symbol])[0]  # MOD
E_symbol, P_symbol = E_symbol[E_idx], P_symbol[O_idx]  # MOD
E, O = E[E_idx], O[O_idx]  # MOD
if scipy.sparse.issparse(E):  # MOD
	E = E.toarray()  # MOD
if scipy.sparse.issparse(O):  # MOD
	O = O.toarray()  # MOD

#loading bulk HiChIP interactions
i, j, d = [], [], []  # MOD
for e in A.edges:  # MOD
	i.append(e[0])  # MOD
	j.append(e[1])  # MOD
	attrs = A.get_edge_data(*e)  # MOD
	d.append(attrs["weight"] * attrs["sign"])  # MOD
i = pd.Index(E_symbol).get_indexer(i)  # MOD
j = pd.Index(P_symbol).get_indexer(j)  # MOD
d = np.asarray(d) * 50  # MOD (50 = average loop count in example data)
d[d < 0] = 0  # Discard negative-sign edges
mask = np.logical_and(i >= 0, j >= 0)  # MOD
i, j, d = i[mask], j[mask], d[mask]  # MOD
A = scipy.sparse.coo_matrix(
	(d, (i, j)), shape=(E.shape[0], O.shape[0])
).toarray()  # MOD

E_symbol = np.asarray(E_symbol)
P_symbol = np.asarray(P_symbol)
E        = pd.DataFrame(E)
O        = pd.DataFrame(O)

start_time = time.time()  # MOD

#perform quantile nomalization on E and O matrix.
E        = quantileNormalize(E)
O        = quantileNormalize(O)

print "Initializing non-negative matrix factorization for E..."
E[E>10000] = 10000
X = np.log(1+E)

err1=np.zeros(rep)
for i in range(0,rep):
        model = NMF(n_components=K, init='random', random_state=i,solver='cd',max_iter=50)
        W20 = model.fit_transform(X)
        H20 = model.components_
        err1[i]=LA.norm(X-np.dot(W20,H20),ord = 'fro')

model = NMF(n_components=K, init='random', random_state=np.argmin(err1),solver='cd',max_iter=1000)
W20 = model.fit_transform(X)
H20 = model.components_
S20=np.argmax(H20,0)

print "Initializing non-negative matrix factorization for O..."
O = np.log(O+1)
err=np.zeros(rep)

for i in range(0,rep):
        model = NMF(n_components=K, init='random', random_state=i,solver='cd',max_iter=50)
        W10 = model.fit_transform(O)
        H10 = model.components_
        err[i]=LA.norm(O-np.dot(W10,H10),ord = 'fro')

model = NMF(n_components=K, init='random', random_state=np.argmin(err),solver='cd',max_iter=1000)
W10 = model.fit_transform(O)
H10 = model.components_
S10=np.argmax(H10,0)

perm = list(itertools.permutations(range(K)))
score = np.zeros(len(perm))
for i in range(len(perm)):
	temp = np.dot(W20[:,perm[i]],np.transpose(W10))
	[score[i],p] =  pearsonr(A[A>0],temp[A>0])

match = np.argmax(score)
W20 = W20[:,perm[match]]
H20 = H20[perm[match],:]
S20=np.argmax(H20,0)

print "Initializing hyperparameters lambda1 and lambda2 ..."
eps = 1e-20
D   = np.zeros((A.shape[0],A.shape[1]))
D[A>0] = 1

s1        = np.zeros((H10.shape[1],H10.shape[1]))
s2        = np.zeros((H20.shape[1],H20.shape[1]))
C         = np.zeros((K,K))

for p in range(H10.shape[1]):
        s1[p,p] = LA.norm(H10[:,p],1)

for p in range(H20.shape[1]):
        s2[p,p] = LA.norm(H20[:,p],1)

H1       = np.dot(H10,LA.inv(s1))
H2       = np.dot(H20,LA.inv(s2))
for p in range(K):
        C[p,p]  = np.sum(H1[p,:]) + np.sum(H2[p,:])
C   = C/np.sum(C)

temp1 = np.dot(np.dot(W20,C), np.transpose(W10))
temp1 = temp1[A>0]
temp2 = A[A>0]
alpha = np.dot(np.transpose(temp1),temp2)/np.dot(np.transpose(temp1),temp1)
lambda10 = pow(LA.norm(O-np.dot(W10,H10),ord = 'fro'),2)/pow(LA.norm(X-np.dot(W20,H20),ord = 'fro'),2)
lambda20 = pow(LA.norm(O-np.dot(W10,H10),ord = 'fro'),2)/pow(LA.norm(A-alpha*D*np.dot(np.dot(W20,C),np.transpose(W10)),ord = 'fro'),2)

if type(args.lambda1) == type(None) and type(args.lambda2) == type(None):
	set1=[lambda10*pow(10,0),lambda10*pow(10,1),lambda10*pow(10,2),lambda10*pow(10,3),lambda10*pow(10,4)]
	set2=[lambda20*pow(10,-5),lambda20*pow(10,-4),lambda20*pow(10,-3),lambda20*pow(10,-2),lambda20*pow(10,-1)]
elif type(args.lambda1) == type(None):
	set1=[lambda10*pow(10,0),lambda10*pow(10,1),lambda10*pow(10,2),lambda10*pow(10,3),lambda10*pow(10,4)]
	set2=[args.lambda2]
elif type(args.lambda2) == type(None):
        set1=[args.lambda1]
        set2=[lambda20*pow(10,-5),lambda20*pow(10,-4),lambda20*pow(10,-3),lambda20*pow(10,-2),lambda20*pow(10,-1)]

else:
	set1=[args.lambda1]
	set2=[args.lambda2]

detr = np.zeros((len(set1),len(set2)))
S1_all = np.zeros((len(set1)*len(set2),O.shape[1]))
S2_all = np.zeros((len(set1)*len(set2),E.shape[1]))
hichip_all = np.zeros((len(set1)*len(set2),K,np.sum(A>0)))
P_all = np.zeros((len(set1)*len(set2),K,O.shape[0]))
E_all = np.zeros((len(set1)*len(set2),K,E.shape[0]))
P_p_all = np.zeros((len(set1)*len(set2),K,O.shape[0]))
E_p_all = np.zeros((len(set1)*len(set2),K,E.shape[0]))
H1_all = np.zeros((len(set1)*len(set2),K,O.shape[1]))  # MOD
H2_all = np.zeros((len(set1)*len(set2),K,E.shape[1]))  # MOD

print "Starting dcHiChIP..."
count = 0
for x in range(len(set1)):
	for y in range(len(set2)):
		lambda1 = set1[x]
		lambda2 = set2[y]
		W1 = W10+np.random.rand(O.shape[0],K)*10
		W2 = W20+np.random.rand(E.shape[0],K)*10
		H1 = H10
		H2 = H20
		print lambda1,lambda2

		print "Iterating dcHiChIP..."
		maxiter   = 500
		err       = 1
		terms     = np.zeros(maxiter)
		it        = 0
		s1        = np.zeros((H1.shape[1],H1.shape[1]))
		s2        = np.zeros((H2.shape[1],H2.shape[1]))
		C         = np.zeros((K,K))

		for p in range(H1.shape[1]):
			s1[p,p] = LA.norm(H1[:,p],1)

		for p in range(H2.shape[1]):
			s2[p,p] = LA.norm(H2[:,p],1)

		H1       = np.dot(H1,LA.inv(s1))
		H2       = np.dot(H2,LA.inv(s2))
		for p in range(K):
			C[p,p]  = np.sum(H1[p,:]) + np.sum(H2[p,:])
		C   = C/np.sum(C)
		temp1 = np.dot(np.dot(W2,C), np.transpose(W1))
		temp1 = temp1[A>0]
		temp2 = A[A>0]
		alpha = np.dot(np.transpose(temp1),temp2)/np.dot(np.transpose(temp1),temp1)
		terms[it] = lambda1/2*pow(LA.norm(X-np.dot(W2,H2),ord = 'fro'),2)+1/2*pow(LA.norm(O-np.dot(W1,H1),ord = 'fro'),2)+lambda2/2*pow(LA.norm(A-alpha*D*np.dot(np.dot(W2,C),np.transpose(W1)),ord = 'fro'),2)
		while it < maxiter-1 and err >1e-4:
			it  = it +1
			T1 = lambda2*alpha*np.dot(np.dot(np.transpose(A),W2),C)
			W1  = W1*((np.dot(O,np.transpose(H1))+T1)/(eps+np.dot(W1,np.dot(H1,np.transpose(H1)))+lambda2*alpha*alpha*np.dot(np.dot(np.transpose(D)*np.dot(np.dot(W1,np.transpose(C)),np.transpose(W2)),W2),C)))
			H1  = H1*((np.dot(np.transpose(W1),O))/(eps+np.dot(np.dot(np.transpose(W1),W1),H1)))
                        T2 = lambda2*alpha*np.dot(np.dot(A,W1),np.transpose(C))
			W2  = W2*((lambda1*np.dot(X,np.transpose(H2))+T2)/(eps+lambda1*np.dot(W2,np.dot(H2,np.transpose(H2)))+lambda2*alpha*alpha*np.dot(np.dot(D*np.dot(np.dot(W2,C),np.transpose(W1)),W1),np.transpose(C))))
			H2  = H2*((np.dot(np.transpose(W2),X))/(eps+np.dot(np.dot(np.transpose(W2),W2),H2)))

			for p in range(H1.shape[1]):
				s1[p,p] = LA.norm(H1[:,p],1)

			for p in range(H2.shape[1]):
				s2[p,p] = LA.norm(H2[:,p],1)

			H1  = np.dot(H1,LA.inv(s1))
			H2  = np.dot(H2,LA.inv(s2))
			H1_all[count] = H1  # MOD
			H2_all[count] = H2  # MOD
			for p in range(K):
        	                C[p,p]  = np.sum(H1[p,:]) + np.sum(H2[p,:])

                	C   = C/np.sum(C)
                	temp1 = np.dot(np.dot(W2,C), np.transpose(W1))
                	temp1 = temp1[A>0]
                	temp2 = A[A>0]
                	alpha = np.dot(np.transpose(temp1),temp2)/np.dot(np.transpose(temp1),temp1)
                	terms[it] = lambda1/2*pow(LA.norm(X-np.dot(W2,H2),ord = 'fro'),2)+1/2*pow(LA.norm(O-np.dot(W1,H1),ord = 'fro'),2)+lambda2/2*pow(LA.norm(A-alpha*D*np.dot(np.dot(W2,C),np.transpose(W1)),ord = 'fro'),2)
			err = abs(terms[it]-terms[it-1])/abs(terms[it-1])


		S2=np.argmax(H2,0)
		S1=np.argmax(H1,0)


		for p in range(K):
			temp = alpha*C[p,p]*np.dot(W2[:,p].reshape(-1,1),np.transpose(W1[:,p].reshape(-1,1)))
			hichip_all[count,p,:] = temp[D>0]

		hichip_final  = hichip_all[count,:,:]
                hichip_revised= A[A>0]*hichip_final/(np.sum(hichip_final,axis = 0)+eps)


		p2 = np.zeros((X.shape[0],K))
		for i in range(K):
        		for j in range(X.shape[0]):
                		statistic, p2[j,i],df  = ttest_ind(X.ix[j,S2!=i], X.ix[j,S2==i] ,alternative='smaller')

		WP2 = np.zeros((W2.shape))
		p2[np.isnan(p2) ] = 1
		scores = -np.log10(p2)
		temp = int(len(E_symbol)/20)
		for i in range(K):
        		indexs = scores[:,i].argsort()[-temp:][::-1]
        		WP2[indexs,i] = 1
			E_all[count,i,indexs] = 1
			E_p_all[count,i,indexs] = p2[indexs,i]

		p1 = np.zeros((O.shape[0],K))
		for i in range(K):
        		for j in range(O.shape[0]):
                		statistic, p1[j,i],df  = ttest_ind(O.ix[j,S1!=i], O.ix[j,S1==i] ,alternative='smaller')

		WP1 = np.zeros((W1.shape))
		p1[np.isnan(p1) ] = 1
		scores = -np.log10(p1)
		temp = int(len(P_symbol)/20)
		for i in range(K):
        		indexs = scores[:,i].argsort()[-temp:][::-1]
        		WP1[indexs,i] = 1
			P_all[count,i,indexs] = 1
			P_p_all[count,i,indexs] = p1[indexs,i]

		for i in range(K):
			temp = np.zeros((A.shape[0],A.shape[1]))
			temp[A>0] = hichip_final[i,:]
			T = np.dot(np.dot(np.transpose(WP2[:,i].reshape(-1,1)),temp),WP1[:,i].reshape(-1,1))
			detr[x,y] = detr[x,y] + T

		#print detr[x,y]
		S1_all[count] = S1
		S2_all[count] = S2
		count = count + 1


[i,j] = npmax(detr)
index = detr.argmax()
S1_final = S1_all[index,:]+1
S2_final = S2_all[index,:]+1
E_final  = E_all[index,:,:]
P_final  = P_all[index,:,:]
E_p_final  = E_p_all[index,:,:]
P_p_final  = P_p_all[index,:,:]
hichip_final = hichip_all[index,:,:]
H1_final = H1_all[index,:,:]  # MOD
H2_final = H2_all[index,:,:]  # MOD

elapsed_time = time.time() - start_time  # MOD

hichip_revised= A[A>0]*hichip_final/(np.sum(hichip_final,axis = 0)+eps)
fout1 = open(os.path.join(args.out_dir,"scATAC-result.txt"),"w")  # MOD
fout2 = open(os.path.join(args.out_dir,"scRNA-result.txt"),"w")  # MOD
fout3 = open(os.path.join(args.out_dir,"cluster-specific-peaks-genes-pairs.txt"),"w")  # MOD
fout4 = open(os.path.join(args.out_dir,"cluster-specific-hichip.txt"),"w")  # MOD

pd.DataFrame(H1_final.T, index=O_cell).to_csv(args.O_latent, header=False)  # MOD
pd.DataFrame(H2_final.T, index=E_cell).to_csv(args.E_latent, header=False)  # MOD
with open(args.run_info, "w") as f:  # MOD
	yaml.dump({
		"cmd": " ".join(sys.argv),
		"args": vars(args),
		"time": elapsed_time,
		"n_cells": E.shape[1] + O.shape[1]
	}, f)  # MOD

print "scATAC-seq clustering assignment:"
print S1_final
print "scRNA-seq clustering assignment:"
print S2_final
print "cluster-specific HiChIP:"
print hichip_revised

for item in S1_final:
	fout1.write(str(item)+"\t")
fout1.write("\n")


for item in S2_final:
        fout2.write(str(item)+"\t")
fout2.write("\n")

p, q = np.nonzero(A)
for i in range(K):
	fout4 = open(os.path.join(args.out_dir,"cluster-"+str(i+1)+".txt"),"w")  # MOD
	for j in range(np.sum(A>0)):
		fout4.write(str(p[j])+"\t"+str(q[j])+"\t"+str(hichip_revised[i,j])+"\n")
	fout4.close()

for i in range(K):
	temp = np.dot(np.reshape(E_final[i,:],(E.shape[0],1)),np.reshape(P_final[i,:],(1,O.shape[0])))*A
	p, q = np.nonzero(temp)
	for j in range(len(p)):
		fout3.write("cluster "+str(i+1)+": "+E_symbol[p[j]]+"\t"+P_symbol[q[j]]+"\t"+str(E_p_final[i,p[j]])+"\t"+str(P_p_final[i,q[j]])+"\n")
