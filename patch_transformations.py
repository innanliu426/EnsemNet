############### from list of image patches to transformed tensors
import numpy as np

input_r, input_c = 12, 12
scale = 4
show_step = 10
input_shape = (input_r, input_c)

def transx(x, g):
    fftx=[]
    for i in range(len(x)):
        trans = g(x[i])
        fftx.append(trans)
    return fftx

def transback(x, ig):
    length = x.shape[0]
    shaper, shapec = ig(x[0,:,:,0]).shape
    y = np.zeros((length,shaper, shapec,1))
    for i in range(length):
        y[i,:,:,0] = ig(x[i,:,:,0])
    return y

def intolayers(x):
      x_layers = []
      for i in range(len(x)):
          v = x[i]
          cut_r, cut_c = v.shape
          cut_r, cut_c = cut_r//2, cut_c//2
          v_layers = np.zeros((cut_r, cut_c, 4))
          cA, cH, cV, cD = v[:cut_r, :cut_c], v[:cut_r, cut_c:], v[cut_r:, :cut_c], v[cut_r:, cut_c:]
          v_layers[:,:,0],v_layers[:,:,1],v_layers[:,:,2],v_layers[:,:,3] = cA, cH, cV, cD 
          x_layers.append(v_layers)
      return x_layers
    
def padding(x, padding_shape):
    r, c, t = x.shape
    temp = np.zeros((padding_shape,padding_shape, t))
    temp[:r, :c, :] = x
    return temp

def wave_patch_back(x):
    large_patch = []
    if x.shape[-1]>1:
        for i in range(x.shape[0]):
            v_layers = x[i,:,:,:]
            cut_r, cut_c = v_layers.shape[0], v_layers.shape[1]
            whole = np.zeros((cut_r*2, cut_c*2))
            whole[:cut_r, :cut_c], whole[:cut_r, cut_c:], whole[cut_r:, :cut_c], whole[cut_r:, cut_c:] = \
                    v_layers[:,:,0],v_layers[:,:,1],v_layers[:,:,2],v_layers[:,:,3]
            large_patch.append(whole)
        large_patch = np.reshape(np.asarray(large_patch), (len(large_patch), cut_r*2, cut_c*2,1))
    else:
        large_patch = x
    return large_patch

 #### space and dct domains
 def process_data(x_data,y_data, g):  
    x_train = transx(x_data, g)
    y_train = transx(y_data, g)

    input_r, input_c = x_train[0].shape
    output_r, output_c = y_train[0].shape 

    x_train = np.reshape(x_train , (len(x_train), input_r,input_c, 1))  # adapt this if using `channels_first` image data format
    y_train = np.reshape(y_train , (len(y_train), output_r,output_c, 1))  # adapt this if using `channels_first` image data format

    #print(x_train.shape, y_train.shape)
    return x_train[0,:,:,:].shape, y_train[0,:,:,:].shape, x_train, y_train

#### wvt domains
def process_data_wave(x_data,y_data, g):
    x_train = intolayers(transx(x_data, g))
    y_train = intolayers(transx(y_data, g))

    input_r, input_c, _ = x_train[0].shape
    output_r, output_c, _ = y_train[0].shape 
    x_train = np.reshape(x_train , (len(x_train), input_r,input_c, _))  
    y_train = np.reshape(y_train , (len(y_train), output_r,output_c, _))  

    return x_train[0,:,:,:].shape, y_train[0,:,:,:].shape, x_train, y_train
