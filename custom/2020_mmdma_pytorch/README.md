# README #

MMD-MA implementation in pytorch with following improvements:
(1) GPU capability
(2) Automatic bandwidth parameter selection for gaussian kernel calculation

### What is this repository for? ###

* Running manifold alignment using MMD to align two single-cell datasets in an unsupervised manner
* Version 1.0
* Contents:
	* `manifold_align_mmd_pytorch.py`: Code to run MMD-MA alignment
	* `evaluate.py`: Code with functions to evaluate performance (FOSCTTM or Silhouette scores)


### How do I get set up? ###

* The code is in Python (version 3.7) and uses [Pytorch framework](https://pytorch.org/)
* We recommend installing all the packages using [conda](https://www.anaconda.com/products/individual)
* Dependencies: 
	* `torch`
	* `cuda`
	* `numpy`
	* `matplotlib`
	* `sklearn`
* Datasets: Simulation dataset input kernels to test run the code can be downloaded from [here](https://noble.gs.washington.edu/proj/mmd-ma/)
* Running the MMD-MA code

```
USAGE: manifold_align_mmd_pytorch.py <input_k1> <input_k2> <result_dir> <num_feat> <sigma> <lambda1> <lambda2>

Run MMD-MA algorithm training to align single-cell datasets:
<input_k1>: Input kernel for single-cell dataset 1 (in .npy format)
<input_k2>: Input kernel for single-cell dataset 2 (in .npy format)
<result_dir>: Directory for saving the alpha and beta weights learned by the algorithm
<num_feat>: Dimension size of the learned low-dimensional space [Recommended tuning values : 4,5,6]
<sigma>: Bandwidth parameter for gaussian kernel calculation, set value to 0.0 to perform automatic calculation
<lambda1>: Parameter for penalty term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]
<lambda2>: Parameter for distortion term [Recommended tuning values : 1e-03, 1e-04, 1e-05, 1e-06, 1e-07]

The outputs of the code are alpha and beta weight matrices learned by the algorithm

To obtain the final embeddings:
Embeddings for single-cell dataset 1 = input_k1 x alpha matrix 
Embeddings for single-cell dataset 2 = input_k2 x beta matrix
```

### Who do I talk to? ###

* William S. Noble (william-noble@uw.edu)
* Ritambhara Singh (ritambhara@brown.edu)
