#Author: Ritambhara Singh (ritambhara@brown.edu)
#Created on: 21 Nov 2018

#Pytorch implementation of MMD-MA Tensorflow code (https://bitbucket.org/noblelab/2019_mmd_wabi/src/master/)
#Script to perform manifold alignment using Maximum Mean Discrepancy (MMD)
#Includes GPU capability

import numpy as np
import math
import sys
import os

import matplotlib.cm
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils import data
import torch.optim
from torch.autograd import Variable
import torch.nn as nn
import torch.cuda

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F


USAGE = """USAGE: manifold_align_mmd_pytorch.py <input_k1> <input_k2> <result_dir> <num_feat> <sigma> <lambda1> <lambda2>

Run MMD-MA algorithm training to align single-cell datasets:
<input_k1>: Input kernel for single-cell dataset 1
<input_k2>: Input kernel for single-cell dataset 2
<result_dir>: Directory for saving the alpha and beta weights learned by the algorithm
<num_feat>: Dimension size of the learned low-dimensional space [Recommended tuning values : 4,5,6]
<sigma>: Bandwidth paramteter for gaussian kernel calculation, set value to 0.0 to perform automatic calculation
<lambda1>: Parameter for penalty term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
<lambda2>: Parameter for distortion term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]

The outputs of the code are alpha and beta weight matrices learned by the algorithm

To obtain the final embeddings:
Embeddings for single-cell dataset 1 = input_k1 x alpha matrix
Embeddings for single-cell dataset 2 = input_k2 x beta matrix
"""

if (len(sys.argv) < 8):
  sys.stderr.write(USAGE)
  sys.exit(1)


if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device ="cpu"

print("Running on",device)

def compute_pairwise_distances(x, y): #function to calculate the pairwise distances
  if not len(x.size()) == len(y.size()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if list(x.size())[1] != list(y.size())[1]:
    raise ValueError('The number of features should be the same.')

  diff =  (x.unsqueeze(2) - y.t())
  diff = (diff ** 2).sum(1)
  return diff.t()

def gaussian_kernel_matrix(x, y, sigmas): #function to calculate Gaussian kernel
  beta = 1.0 / (2.0 * (sigmas.unsqueeze(1)))
  dist = compute_pairwise_distances(x, y)
  s = beta * (dist.contiguous()).view(1,-1)
  result =  ((-s).exp()).sum(0)
  return (result.contiguous()).view(dist.size())


def stream_maximum_mean_discrepancy(x, y,  sigmas, kernel=gaussian_kernel_matrix): #This function has been implemented  to caculate MMD value for large number of samples (N>5,000)
  n_x = x.shape[0]
  n_y = y.shape[0]

  n_small = np.minimum(n_x, n_y)
  n = (n_small // 2) * 2

  cost = (kernel(x[:n:2], x[1:n:2], sigmas)  + kernel(y[:n:2], y[1:n:2], sigmas)
          - kernel(x[:n:2], y[1:n:2], sigmas) - kernel(x[1:n:2], y[:n:2], sigmas)).mean()
  if cost.data.item()<0:
    cost = torch.FloatTensor([0.0]).to(device)
  return cost

def maximum_mean_discrepancy(x, y, sigmas, kernel=gaussian_kernel_matrix): #Function to calculate MMD value

  cost = (kernel(x, x, sigmas)).mean()
  cost += (kernel(y, y, sigmas)).mean()
  cost -= 2.0 * (kernel(x, y, sigmas)).mean()

  if cost.data.item()<0:
    cost = torch.FloatTensor([0.0]).to(device)

  return cost

def calc_sigma(x1,x2): #Automatic sigma calculation
    const = 8
    mat = np.concatenate((x1,x2))
    dist = []
    nsamp = mat.shape[0]
    for i in range(nsamp):
        euc_dist = np.sqrt(np.sum(np.square(np.subtract(mat[i,:], mat)), axis=1))
        dist.append(sorted(euc_dist)[1])

    sigma = np.square(const*np.median(dist))
    print("Calculated sigma:",sigma)
    return sigma

class manifold_alignment(nn.Module): #MMD objective function

  def __init__(self, nfeat, num_k1, num_k2, seed):
    super(manifold_alignment, self).__init__()
    #Initializing alpha and beta
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
    else:
      torch.manual_seed(seed)
    self.alpha = Parameter(torch.FloatTensor(num_k1, nfeat).uniform_(0.0,0.1).to(device))
    self.beta = Parameter(torch.FloatTensor(num_k2, nfeat).uniform_(0.0,0.1).to(device))


  def forward(self, k1, k2, ip, sigmas, lambda1, lambda2):


    if sigmas == 0: #If the user does not specify sigma values for the kernel calculation, they will be caclulated automatically
      x1 = (torch.matmul(k1,self.alpha)).detach().cpu().numpy()
      x2 = (torch.matmul(k2,self.beta)).detach().cpu().numpy()

      sigma = calc_sigma(x1,x2)
      sigmas = torch.FloatTensor([sigma]).to(device)

    mmd = maximum_mean_discrepancy(torch.matmul(k1,self.alpha),torch.matmul(k2,self.beta), sigmas)
    #mmd = stream_maximum_mean_discrepancy(torch.matmul(k1,self.alpha),torch.matmul(k2,self.beta), sigmas) #remove comment and comment the previous line if number of samples are large (N>5,000)

    penalty = lambda1 * ((((torch.matmul(self.alpha.t(),torch.matmul(k1,self.alpha))) - ip).norm(2))
              + (((torch.matmul(self.beta.t(),torch.matmul(k2,self.beta))) - ip).norm(2)))

    distortion = lambda2 * ((((torch.matmul((torch.matmul(k1,self.alpha)),(torch.matmul(self.alpha.t(),k1.t()))))-k1).norm(2))
              + (((torch.matmul((torch.matmul(k2,self.beta)),(torch.matmul(self.beta.t(),k2.t()))))-k2).norm(2)))

    return mmd, penalty, distortion, sigmas


#Functions to plot function values
def plot_data(filename,k,i,obj,mmd,pen,dist,nfeat,sigma,lambda1,lambda2):
  plt.xlabel('Iteration')
  plt.ylabel('log(Function value)')
  plt.title('nfeat:'+str(nfeat)+',seed:'+str(k)+', sigma:'+str(sigma)+', lambda1:'+str(lambda1)+', lambda2:'+str(lambda2))

  plt.plot(obj, 'k--', label='Objective')
  plt.plot(mmd, 'r--', label='MMD')
  plt.plot(pen, 'b--', label='Penalty')
  plt.plot(dist, 'g--', label='Distortion')
  if i == 10000:
    plt.legend(loc='upper right')
  plt.savefig(filename)
  plt.close()



def main():
  print("Loading data...")

  k1_matrix = np.load(sys.argv[1]).astype(np.float32)
  k2_matrix = np.load(sys.argv[2]).astype(np.float32)

  print(k1_matrix.shape)
  print(k2_matrix.shape)

  print("Size of matrices...")
  num_k1 = k1_matrix.shape[0]                    #number of samples in dataset 1
  num_k2 = k2_matrix.shape[0]                    #number of samples in dataset 2

  nfeat = int(sys.argv[4])      				# number features in joint embedding
  print("Number of dimensions of latent space...",nfeat)
  sigma = float(sys.argv[5])
  sigmas = torch.FloatTensor([sigma]).to(device)
  lambda_1 = float(sys.argv[6])   			# Lambda1 coefficient for penalty term
  lambda_2 = float(sys.argv[7])     				# Lambda2 coefficient for distortion term

  results_dir = sys.argv[3]+"results_nfeat_"+str(nfeat)+"_sigma_"+str(sigma)+"_lam1_"+str(lambda_1)+"_lam2_"+str(lambda_2)+"/"
  if not os.path.exists(results_dir):
      os.makedirs(results_dir)

  Ip = np.identity(nfeat).astype(np.float32)          #identity matrix of size nfeatxnfeat

  K1 = torch.from_numpy(k1_matrix).to(device)
  K2 = torch.from_numpy(k2_matrix).to(device)
  I_p = torch.from_numpy(Ip).to(device)


  for seed in [int(sys.argv[8])]:

    results_dir_seed = results_dir+"seed_"+str(seed)+"/"
    if not os.path.exists(results_dir_seed):
      os.makedirs(results_dir_seed)

    obj_val=[]
    mmd_val=[]
    pen_val=[]
    dist_val=[]

    model = manifold_alignment(nfeat, num_k1, num_k2, seed)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0005,amsgrad=True)

    model.train()

    for i in range(10001): #Training takes place for 10,000 iterations

      optimizer.zero_grad()

      mmd, penalty, distortion, sigmas = model(K1, K2, I_p, sigmas, lambda_1, lambda_2)
      obj = mmd + penalty + distortion

      obj.backward()

      optimizer.step()

      obj_value = obj.data.item()
      mmd_value = mmd.data.item()
      pen_value = penalty.data.item()
      dist_value = distortion.data.item()

      if mmd_value > 0 :
        obj_val.append(math.log(obj_value))
        mmd_val.append(math.log(mmd_value))
        pen_val.append(math.log(pen_value))
        dist_val.append(math.log(dist_value))

      if (i%200 == 0): #the weights can be saved every 200 iterations
        weights=[]

        for p in model.parameters():
          if p.requires_grad:
            weights.append(p.data)

        plot_data(results_dir_seed+"Functions_"+str(seed)+".png",seed,i,obj_val,mmd_val,pen_val,dist_val,nfeat,sigma,lambda_1,lambda_2)

        if (i==0 or i==10000): #This saves the weights at the beginning and end of the training
          np.savetxt(results_dir_seed+"alpha_hat_"+str(seed)+"_"+str(i)+".txt", weights[0].cpu().numpy())
          np.savetxt(results_dir_seed+"beta_hat_"+str(seed)+"_"+str(i)+".txt", weights[1].cpu().numpy())

    np.savetxt(results_dir+"objective_"+str(seed)+".txt",[obj.data.item()])

main()
