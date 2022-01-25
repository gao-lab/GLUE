#Author: Ritambhara Singh (ritambhara@brown.edu)
#Created on: 5 Feb 2020

#Script to calculate performance scores for learnt embeddings of MMD algorithm

import numpy as np
import math
import os
import sys
import random

from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA

import matplotlib.cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def calc_sil(x1_mat,x2_mat,x1_lab,x2_lab): #function to calculate Silhouette scores

    x = np.concatenate((x1_mat,x2_mat))
    lab = np.concatenate((x1_lab,x2_lab))

    sil_score = silhouette_samples(x,lab)


    avg = np.mean(sil_score)
    
    return avg

def calc_frac(x1_mat,x2_mat): #function to calculate FOSCTTM values
	nsamp = x1_mat.shape[0]
	total_count = nsamp * (nsamp -1)
	rank=0
	for row_idx in range(nsamp):
		euc_dist = np.sqrt(np.sum(np.square(np.subtract(x1_mat[row_idx,:], x2_mat)), axis=1))
		true_nbr = euc_dist[row_idx]
		sort_euc_dist = sorted(euc_dist)
		rank+=sort_euc_dist.index(true_nbr)
	
	frac = float(rank)/total_count
	return frac
	


def get_score(curr_dir,nfeat,c,l1,l2,seed_idx,matrix_a,matrix_b,lab_a,lab_b,eval_type): #function to obtain the embeddings and call functions for score calulations
    
    i=10000
    print("Loading alphas and betas...")
    
    if os.path.exists(curr_dir+'seed_'+str(seed_idx)+'/alpha_hat_'+str(seed_idx)+'_'+str(i)+'.txt'):
        
        alpha = np.loadtxt(curr_dir+'seed_'+str(seed_idx)+'/alpha_hat_'+str(seed_idx)+'_'+str(i)+'.txt')
        beta = np.loadtxt(curr_dir+'seed_'+str(seed_idx)+'/beta_hat_'+str(seed_idx)+'_'+str(i)+'.txt')
        
        x1 = np.matmul(matrix_a,alpha)
        x2 = np.matmul(matrix_b,beta)
        
        
        
        print("Calculating scores...")
        
        if eval_type == 0:
            frac1 = calc_frac(x1,x2)
            frac2 = calc_frac(x2,x1)
            score = (frac1+frac2)/2
        else:
            
            sil1 = calc_sil(x1,x2,lab_a,lab_b)
            sil2 = calc_sil(x2,x1,lab_b,lab_a)
            score = (sil1+sil2)/2
            
    else:
        score = 999
    
    print(score)
    
    return score

result_dir = 'results/' #modify to add the result directory with weights

eval_type = 0 #set value to 0 for FOSCTTM and 1 for Sil Score

print("Loading matrices ...")
x = np.load('input_k1.npy').astype(np.float32) #specify file name for input kernel 1 for single-cell dataset 1
y = np.load('input_k2.npy').astype(np.float32) #pecify file name for input kernel 2 for single-cell dataset 2

if eval_type == 0 : #labels not required as the data should have 1-1 correspondence
    lab_x  = []
    lab_y = []
else:
    lab_x = np.load('labels1.npy') #specify cluster label file for single-cell dataset 1
    lab_y = np.load('labels2.npy') #specify cluster label file for single-cell dataset 2


#Loop over recommended tuning values
nfeats = [4,5,6]
cs = [0.0]
lambda1s = [1e-03,1e-04,1e-05,1e-06,1e-07]
lambda2s = [1e-03,1e-04,1e-05,1e-06,1e-07]


hyper_scores = {}
c_scores = {}

outfile=open("scores_mmd_all.txt","w") #Text file to save all the scores

for c in cs:
    print("c:",c)
    for l1 in lambda1s:
        print("lambda1:",l1)
        for l2 in lambda2s:
            print("lambda2:",l2)
            for nfeat in nfeats:
                print("nfeat:",nfeat)
                curr_dir = result_dir+'results_nfeat_'+str(nfeat)+'_sigma_'+str(c)+'_lam1_'+str(l1)+'_lam2_'+str(l2)+'/'
                file_in = curr_dir+'objective_0.txt'
                
                if os.path.exists(file_in):
                    obj_values=[]
                    for seed in range(20): #Pick the seed with the lowest objective function value
                        if os.path.exists(curr_dir+'objective_'+str(seed)+'.txt'):
                            file_in = open(curr_dir+'objective_'+str(seed)+'.txt')
                            obj = float(file_in.read())
                            obj_values.append(obj)
                    obj_np_values = np.asarray(obj_values)
                    seed_idx = np.argmin(obj_np_values)
                    print("Lowest obj value:",obj_np_values[seed_idx])
                    print("for seed:",seed_idx)
                    
                    score=get_score(curr_dir,nfeat,c,l1,l2,seed_idx,x,y, lab_x,lab_y,eval_type)

                    outfile.write("{}\n".format(score)) 
                    
                    if score != 999:
                        dict_key = '(d='+str(nfeat)+', c='+str(c)+', l1='+str(l1)+', l2='+str(l2)+')'
                        hyper_scores[dict_key]=score
                        if c in c_scores.keys():
                            c_scores[c].append(score)
                        else:
                            c_scores[c] = [score]
                  
sorted_scores = sorted(hyper_scores.values())

if eval_type==0:
    final_scores = sorted_scores[:5]
else:
    final_scores = sorted_scores[-5:]

for i in final_scores: #print the 5 best hyperparameter performances 
    best_hyp = list(hyper_scores.keys())[list(hyper_scores.values()).index(i)]        
    print('Best hyp:'+best_hyp+' for score:'+str(i))

