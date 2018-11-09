#import cv2
import numpy as np
import struct
from random import randint
import random

image_width = 256

bands = 4

#animation = 'band'+str(bands)+'/pirate_band'+str(bands)
#data_prefix = "C:/Users/pablo/Dev/DeepPRT/Training/data/"+animation
data_prefix = "C:/Users/pablo/Dev/DeepPRT/Training/data/pirate_head"


# =============================================================================
# 
# =============================================================================
def get_transfer_coeffs(file_name, diffuse=True):
    f = open(file_name, 'rb')
    n_coeff_per_vertex = struct.unpack('i', f.read(4))[0] # n_bands * n_bands
    data_list = []
    for i in range(n_coeff_per_vertex):
        if (not diffuse):
            for j in range(3):
                data = np.array(struct.unpack('d'*image_width*image_width, f.read(8*image_width*image_width))).reshape(image_width,image_width)
                data_list.append(data*0.01+0.5) # glossy
                # data_list.append(data) # no shifft
                del data
        else:
            data = np.array(struct.unpack('d'*image_width*image_width, f.read(8*image_width*image_width))).reshape(image_width,image_width)
            data_list.append(data+0.5)
#            data_list.append(data)
            del data
    f.close()
    trans_coeffs = data_list[0][:,:,np.newaxis]
    
    # Glossy 
    if (not diffuse):
        for i in range(1, bands**2*3): #glossy
            trans_coeffs = np.concatenate((trans_coeffs,data_list[i][:,:,np.newaxis]),axis=2)
    else:
    # Diffuse
        for i in range(1, bands**2):
            trans_coeffs = np.concatenate((trans_coeffs,data_list[i][:,:,np.newaxis]),axis=2)
    return trans_coeffs

# =============================================================================
# 
# =============================================================================
def get_position_map(file_name):
	f = open(file_name, 'rb')
	total_len = image_width * image_width
	data_x = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	data_y = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	data_z = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	
	position_map = np.concatenate((data_x[:,:,np.newaxis], data_y[:,:,np.newaxis], data_z[:,:,np.newaxis]), axis=2)
	
	del data_x
	del data_y
	del data_z
	return position_map


# =============================================================================
# 
# =============================================================================
def get_normal_map(file_name):
	f = open(file_name, 'rb')
	total_len = image_width * image_width
	data_x = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	data_y = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	data_z = np.array(struct.unpack('d'*total_len, f.read(8*total_len))).reshape(image_width, image_width)
	
	normal_map = np.concatenate((data_x[:,:,np.newaxis], data_y[:,:,np.newaxis], data_z[:,:,np.newaxis]), axis=2)
	
	del data_x
	del data_y
	del data_z
	return normal_map


# =============================================================================
# 
# =============================================================================
def train_set_generator(batch_size):
    
    alpha = np.eye(500).astype('float32')
    while 1:
        trans_coeffs = []
        for i in range(batch_size):
            idx = str(random.randint(0, 450))
            trans_coeff = get_transfer_coeffs(data_prefix + '/transfor_coefficients_{}'.format(idx))
            trans_coeffs.append(trans_coeff)
        
        trans_coeffs=np.reshape(np.array(trans_coeffs),(batch_size,-1)).astype('float32')
        print('type: ', trans_coeffs.dtype)
        print('transf coeff: ', trans_coeffs.shape)
        print('alpha: ', alpha.shape)
        yield alpha[:batch_size,...], trans_coeffs

# =============================================================================
# 
# =============================================================================
def validation_set_generator(batch_size):
    alpha = np.eye(500).astype('float32')
    while 1:
        trans_coeffs = []
        for i in range(batch_size):
            idx = str(random.randint(450, 500))
            trans_coeff = get_transfer_coeffs(data_prefix + '/transfor_coefficients_{}'.format(idx))
            trans_coeffs.append(trans_coeff)
        
        trans_coeffs=np.reshape(np.array(trans_coeffs),(batch_size,-1)).astype('float32')
        
        yield alpha[:batch_size,...], trans_coeffs



