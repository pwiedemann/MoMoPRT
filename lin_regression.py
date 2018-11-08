from sklearn.decomposition import PCA
import os
import numpy as np
import shutil

import keras.backend as K
from keras.callbacks import *
from keras.models import Model
from keras import optimizers
from keras import losses
from keras.layers import Dense, Input
import matplotlib.pyplot as plt



# =============================================================================
# 
# =============================================================================
def check_nan(nparray):
    return np.any(np.isnan(nparray))

# =============================================================================
# 
# =============================================================================
def lin_model(inp_shape, out_size, name):
    inp = Input(shape=inp_shape)
    x = Dense(out_size)(inp) 
    return Model(inputs=inp, outputs=x, name=name)

# =============================================================================
# 
# =============================================================================
def train(X,Y, model):
    
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'Models/' + model.name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    
    # Callbacks:
#    loss_history = LossHistory()
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'weights.best.hdf5'), 
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min')#.{epoch:d}
    
    # compile
    model.compile('adam',loss='mse')
    
    # training
    history = model.fit(X,Y,batch_size=32,epochs=1,validation_split=0.1, 
                        callbacks=[checkpoint])
    
    #saving  model and training properties
    save_path = os.path.join(current_dir, 'Models/' + model.name)
    numpy_loss_history = np.array(history.history['val_loss'])
    min_val_loss = np.min(numpy_loss_history)
    np.savetxt(os.path.join(save_path, 'val_loss.txt'), numpy_loss_history, delimiter=",")
    model.save_weights(save_path+'/model.hdf5')
    shutil.copy(os.path.join(save_path, 'weights.best.hdf5'),
        os.path.join(save_path, 'weights_val_loss{0:.5e}.hdf5'.format(min_val_loss)))


# =============================================================================
# 
# =============================================================================
def main():
    # Load training data     
    Snew = np.load('C:/Users/pablo/Dev/DeepPRT/Training/MoMoPRT/lin_Pirate_shapes_std.npz')['Snew']
    alpha = np.load('C:/Users/pablo/Dev/DeepPRT/Training/MoMoPRT/lin_Pirate_shapes_std.npz')['alphas']
   
    # declare lin model
    model = lin_model((1,),256**2*3, name='test')
    print(model.summary())
    print(model.name)
    # train (lin regression) 
    train(X=alpha,Y=Snew[...,0], model=model)

    
#    
#    Y_model = model.predict(X)
#    
#    
#    vis(X,Y,Y_model)

main()
    


