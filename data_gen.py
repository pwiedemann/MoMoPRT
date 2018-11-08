from sklearn.decomposition import PCA
import os
import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('..')
from dataset import get_position_map


# =============================================================================
# 
# =============================================================================
def check_nan(nparray):
    return np.any(np.isnan(nparray))

# =============================================================================
# get all pos data
# =============================================================================
def data(d):
    pos_maps = np.zeros((500,256,256,3))
    for idx in range(500):
        p = get_position_map(d + '/position_map_{}'.format(idx))
        pos_maps[idx] = p
    return pos_maps

    
# =============================================================================
# X has to be of shape (nb_shapes, nb_vertices)
# =============================================================================
def save_pca_params(X, data_file):
    
    X = np.reshape(X,(500,256**2*3))
    X = np.swapaxes(X,0,1)
    Xmean = np.mean(X,axis=1)[:,np.newaxis]
    
    # center data to mean val.
    Xc = X - Xmean
    
    pca = PCA(n_components=500)

    print('solving trafo...')
    Xnew = pca.fit_transform(Xc)
    
    Cov = pca.get_covariance()
    
    print('saving params...')
    np.savez_compressed(data_file+'.npz', 
                        mean =Xmean, 
                        covariance=Cov,
                        Xpca=Xnew)

# =============================================================================
#  https://gravis.dmi.unibas.ch/publications/2009/BFModel09.pdf
# =============================================================================
def MoMo_model(model_name, alpha):
    
    Xpca  = np.load(model_name+'.npz')['Xpca']
    print('bases: ',Xpca.shape)
    Xmean = np.load(model_name+'.npz')['mean']
    print('mean: ',Xmean.shape)
    Cov   = np.load(model_name+'.npz')['covariance']
    var = np.diag(Cov)[:,np.newaxis]
    print('cov: ',var.shape)
    del Cov
    var = np.sqrt(var)
#    
    Xnew = Xmean + ( Xpca.dot(var) )*alpha
    del Xmean
    del Xpca
    del var

    return Xnew 


# =============================================================================
# Ax=b
# =============================================================================
def get_alpha_GT_shape(model_name,shape_idx):
    
    alpha = np.zeros((500,256**2*3))
    
    X = data('C:/Users/pablo/Dev/DeepPRT/Training/data/pirate_head' )
    X = np.reshape(X,(500,256**2*3))
    
    Xpca  = np.load(model_name+'.npz')['Xpca']
    Xpca = np.swapaxes(Xpca,0,1)

    b = np.zeros(shape=(256**2*3,1))
    
#    for i in range(500):
#
    return alpha
        
    
# =============================================================================
# 
# =============================================================================
def generate_data():  

#    X = data('C:/Users/pablo/Dev/DeepPRT/Training/data/OLD_flag_sim' )
#    save_pca_params(X, save_as)
    
    
    # PARAMETERS
    #----------------------------------
    params_file = './pirate_MoMoParams'
    out_data_file ='./lin_Pirate_shapes_std'
    
    nb_samples = 500
    #----------------------------------
    
    
    alpha = np.random.uniform(low=-8, high=8, size=(nb_samples))
    Snew = np.zeros((len(alpha),256**2*3,1))
    
    for i,a in enumerate(alpha):
        print(i)
        Snew[i] = MoMo_model(params_file,a)
    
    print('saving...')
    np.savez(out_data_file+'.npz',Snew=Snew, alphas=alpha)

