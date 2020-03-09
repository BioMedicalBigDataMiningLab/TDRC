# Tensor Decomposition with Relational Constraints
This is our Python implementation for the unpublished paper.

## Introduction
Tensor Decomposition with Relational Constraints (TDRC) is a tensor decomposition method for predicting multiple types of microRNA-disease associations, which incorporates the miRNA-miRNA similarity and the disease-disease similarity.
## Environment Requirement
The code has been tested running under Python 3.6.8. The required packages are as follows:
   * tensorly == 0.4.3
   * numpy == 1.16.4
## Arguments
* `X`   
    + the input miRNA-disease-type tensor with the size of (_m_\*_n_\*_t_)
* `S_d`
    + the disease-disease similarity matrix with the size of (_n_\*_n_)
* `S_m`
    + the miRNA-miRNA similarity matrix with the size of (_m_\*_m_)
* `r`
    +  the rank of the reconstructed tensor, also the dimensionality of the latent space/representations, int, default=4
* `alpha` and `beta` 
    + control the contributions of miRNA-miRNA similarity and disease-disease similarity, float, default=0.125/0.25  
* `lam` 
    + the regularization coefficient, float, default=0.001 
* `tol` 
    + tolerance of stopping criterion, float, default=1e-6
* `max_iter`
	+ hard limit on iterations for ADMM, int, default=500
