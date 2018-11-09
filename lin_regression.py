import os
import numpy as np
import shutil
import time

import keras.backend as K
from keras.callbacks import *
from keras.models import Model
from  keras.optimizers import Adam
from keras import losses
from keras.layers import Dense, Input
import matplotlib.pyplot as plt
from MoMo_dataset import train_set_generator, validation_set_generator, get_transfer_coeffs
import tensorflow as tf


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
def train1(model, params):
    
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'Models/' + model.name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    
    # Callbacks:
#    loss_history = LossHistory()
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'weights.best.hdf5'), 
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min')#.{epoch:d}
    
    # optimiser
    opt =Adam(lr=params['lr'])
    # compile
    model.compile(opt,loss='mse')
    
    # training
    
    history = model.fit_generator(train_set_generator(params['batch_size']), 
                                  epochs=params['epochs'], verbose=1, 
                                  validation_data=validation_set_generator(params['batch_size']), 
                                  callbacks=[checkpoint],steps_per_epoch=90,validation_steps=10)
    
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
def train(X,Y, model, params):
    
    current_dir = os.getcwd()
    save_path = os.path.join(current_dir, 'Models/' + model.name)
    if os.path.exists(save_path) is False:
        os.mkdir(save_path)
    
    # Callbacks:
    # ------------------------------
    def lr_scheduler(epoch):
        lr = 1e-3
        if epoch > 0.9 * params['epochs']:
            lr /=2.0
        elif epoch > 0.75 * params['epochs']:
            lr /=2.0
        elif epoch > 0.3 * params['epochs']:
            lr = 1e-4
        return lr
    scheduler = LearningRateScheduler(lr_scheduler)
    checkpoint = ModelCheckpoint(filepath=os.path.join(save_path, 'weights.best.hdf5'), 
                                 monitor='val_loss', verbose=1, save_best_only=True, mode='min')#.{epoch:d}
    # ------------------------------
    
    # optimiser
    opt =Adam(lr=params['lr'])
    # compile
    model.compile(opt,loss='mse')
    
    # training
    history = model.fit(X,Y,batch_size=params['batch_size'],epochs=params['epochs'],
                        validation_split=0.1, callbacks=[checkpoint, scheduler])
    
    
    #saving  model and training properties
    save_path = os.path.join(current_dir, 'Models/' + model.name)
    numpy_loss_history = np.array(history.history['val_loss'])
    min_val_loss = np.min(numpy_loss_history)
    np.savetxt(os.path.join(save_path, 'val_loss.txt'), numpy_loss_history, delimiter=",")
    model.save_weights(save_path+'/model.hdf5')
    shutil.copy(os.path.join(save_path, 'weights.best.hdf5'),
        os.path.join(save_path, 'weights_val_loss{0:.5e}.hdf5'.format(min_val_loss)))



def data(d,nb_samples = 500):
    T_maps = np.zeros((nb_samples,256,256,16),dtype='float32')
    print('loading data...')
    t1 = time.time()
    for idx in range(nb_samples):
        T_coeff = get_transfer_coeffs(d + '/transfer_coefficients_{}'.format(idx)).astype('float32')
        T_maps[idx] = T_coeff
    print('loading duration: ', time.time()-t1)
    return T_maps

# =============================================================================
# 
# =============================================================================
def lin_regression1():
    
    nb_samples = 500
    
    # Load training data 
    data_prefix = "C:/Users/pablo/Dev/DeepPRT/Training/data/MoMo_basis/pirate_plane_basis"    
    Y = data(data_prefix, nb_samples)
    print('type: ', Y.dtype)
    Y = np.reshape(Y,(len(Y),-1,16))
    Y = np.swapaxes(Y,0,1)
    
    alpha = np.eye(nb_samples).astype('float32')
    print('shape: ', Y.shape)
    
    '''
    --------------------------------------------------
    TRAINING PARAMS
    --------------------------------------------------
    '''
    params ={
            'epochs': 100,
            'batch_size': 32,
            'lr': 1e-3
            }
    
    t1 = time.time()
    i=0
    # Lin model for each vertex 
    for v in Y[:1,...]:
#        # declare lin model
        model = lin_model((nb_samples,),16, name='blendshape_pirate_vrtx_{}'.format(i))
        print(model.summary())
        print(model.name)
        i+=1
    
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        session = tf.Session(config=config)
        K.set_session(session)
        
        # train (lin regression) 
    #    train1(model=model, params=params)
        train(alpha,v, model=model, params=params)
    print('\n--------')
    print('duration: ', time.time()-t1)
    
# =============================================================================
# 
# =============================================================================
def lin_regression():
    nb_samples = 500
    
    # Load training data 
#    data_prefix = "C:/Users/pablo/Dev/DeepPRT/Training/data/MoMo_basis/pirate_plane_basis"
#    Y = data(data_prefix, nb_samples)
    t1 = time.time()
    Y = np.load('C:/Users/pablo/Dev/DeepPRT/Training/data/MoMo_basis/sh_gt_trainset.npz')['train']
    print(time.time()-t1)
    print('type: ', Y.dtype)
    Y = np.reshape(Y,(len(Y),-1))
    
    alpha =np.load("C:/Users/pablo/Dev/DeepPRT/Training/MoMoPRT/pirate_plane_basis.npz")['alphas'][:nb_samples]
    print('shape: ', Y.shape)
    print('alphas: ', alpha.shape)
    
    
    # TRAINING PARAMS
    #--------------------------------------------------
    params ={
            'epochs': 500,
            'batch_size': 12,
            'lr': 1e-3
            }
    #--------------------------------------------------

    # declare lin model
    model = lin_model((3,),256**2*16, name='pirate_plane_basis')
    print(model.summary())
    print(model.name)

    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    session = tf.Session(config=config)
    K.set_session(session)
    
    # train (lin regression) 
    train(alpha,Y, model=model, params=params)


# =============================================================================
# 
# =============================================================================
if __name__ == '__main__':
    lin_regression()

    


