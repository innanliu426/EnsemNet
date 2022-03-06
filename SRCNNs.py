################ EDSR modified from the repository krasserm/super-resolution
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Multiply
from tensorflow.keras.layers import Dense, Concatenate, Add, Lambda, Reshape, Dropout, LeakyReLU, Conv2DTranspose
from wavetf import WaveTFFactory

################ basic EDSR ################
def edsr(x_in, scale, num_filters=64, num_res_blocks=8, res_block_scaling=0.1):
    x = b = Conv2D(num_filters, 3, padding='same')(x_in)
    for i in range(num_res_blocks):
        b = res_block(b, num_filters, res_block_scaling)
    b = Conv2D(num_filters, 3, padding='same')(b)
    x = Add()([x, b])
    x = upsample(x, scale, num_filters)
    x = Conv2D(num_filters, 3, padding='same')(x)
    return x


def res_block(x_in, filters, scaling):
    """Creates an EDSR residual block."""
    x = Conv2D(filters, 3, padding='same', activation='relu')(x_in)
    x = Conv2D(filters, 3, padding='same')(x)
    if scaling:
        x = Lambda(lambda t: t * scaling)(x)
    x = Add()([x_in, x])
    return x


def upsample(x, scale, num_filters):
    def upsample_1(x, factor, **kwargs):
        """Sub-pixel convolution."""
        x = Conv2D(num_filters * (factor ** 2), 3, padding='same', **kwargs)(x)
        return Lambda(pixel_shuffle(scale=factor))(x)

    if scale == 2:
        x = upsample_1(x, 2)
    elif scale == 3:
        x = upsample_1(x, 3)
    elif scale == 4:
        x = upsample_1(x, 2)
        x = upsample_1(x, 2)
    elif scale == 6:
        x = upsample_1(x, 3)
        x = upsample_1(x, 2)
    return x

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def normalize(x):
    return (x - DIV2K_RGB_MEAN) / 127.5

def denormalize(x):
    return x * 127.5 + DIV2K_RGB_MEAN


def k_dct(x):
    tf.signal.dct(x, type=2, n=None, axis=-1, norm='ortho')
    
def k_dct(x):
    X1 = tf.signal.dct(x, type=2, norm='ortho')
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.dct(X1_t, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    return X2_t
def k_idct(x):
    X1 = tf.signal.idct(x, type=2, norm='ortho')
    X1_t = tf.transpose(X1, perm=[0, 1, 3, 2])
    X2 = tf.signal.idct(X1_t, type=2, norm='ortho')
    X2_t = tf.transpose(X2, perm=[0, 1, 3, 2])
    return X2_t


####################################  edsr_wave_dct ################################### 
def model(input_img):
    x1 = Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    x1 = k_dct(x1)
    x1 = edsr(x1, scale =4, num_filters=64, num_res_blocks=8, res_block_scaling=0.1)
    x1 = SubpixelConv2D((input_r, input_c, 1), scale=4)(x1)
    x1 = Conv2D(1, 3, padding='same')(x1)
    x1 = k_idct(x1)
    
    x2= Reshape((input_shape[0], input_shape[1],input_shape[2]-1))(input_img[:,:,:,1:])
    x2= edsr(x2, scale =4, num_filters=64, num_res_blocks=8, res_block_scaling=0.1)
    x2 = SubpixelConv2D((input_r, input_c, 1), scale=4)(x2) 
    x2 = Conv2D(4, 3, padding='same')(x2)
    x2 = Conv2D(1, 3, padding='same')(x2)
    
    x = Add()([x1, x2])
    out = Conv2D(filters=output_shape[-1] , kernel_size=(3,3) , strides=(1, 1) , padding='same')(x)
    return out


####################################  edsr_space ################################### 
def model_edsr(input_img):
    x1= Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    x1= edsr(x1, scale =4, num_filters=64, num_res_blocks=8, res_block_scaling=0.1)
    x1 = upsample(x1, scale =4, num_filters=64)
    out = Conv2D(1, 3, padding='same')(x1)
    return out


####################################  dct ################################### 
def frame_dct(input_img, output_shape):
    x = Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    x = k_dct(x)
    pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x)
    pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
    
    ### RDB layers ###
    x = RDBlocks(pass2, 'RDB1')
    RDBlocks_list = [x]
    for i in range(2, 8):
        x = RDBlocks(x, 'RDB'+ str(i))
        RDBlocks_list.append(x)
    x = Concatenate(axis = 3)(RDBlocks_list)
    x = Conv2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(x)
    x = Conv2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(x) 
    
    ### Upsample layer ###
    x = Add()([x , pass1])
    x = SubpixelConv2D((input_r, input_c, 1), scale=4)(x)
    x = Conv2D(filters =1 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(x)
    out = k_idct(x)
    return out



################################### basic SRResNet ################################### 
def SubpixelConv2D(input_shape, scale=4):
    """
    Keras layer to do subpixel convolution.
    NOTE: Tensorflow backend only. Uses tf.depth_to_space
    Ref:
        [1] Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
            Shi et Al.
            https://arxiv.org/abs/1609.05158
    :param input_shape: tensor shape, (batch, height, width, channel)
    :param scale: upsampling scale. Default=4
    :return:
    """
    # upsample using depth_to_space
    def subpixel_shape(input_shape):
        dims = [input_shape[0],
                input_shape[1] * scale,
                input_shape[2] * scale,
                int(input_shape[3] / (scale ** 2))]
        output_shape = tuple(dims)
        return output_shape

    def subpixel(x):
        return tf.nn.depth_to_space(x, scale)


    return Lambda(subpixel, output_shape=subpixel_shape)


def RDBlocks(x, name, count = 6, g=32):
    ## 6 layers of RDB block
    ## this thing need to be in a damn loop for more customisability
    li = [x]
    pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu' , name = name+'_conv1')(x)

    for i in range(2 , count+1):
        li.append(pas)
        out = Concatenate(axis = 3)(li) # conctenated out put
        pas = Conv2D(filters=g, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu', name = name+'_conv'+str(i))(out)

    # feature extractor from the dense net
    li.append(pas)
    out = Concatenate(axis = 3)(li)
    feat = Conv2D(filters=64, kernel_size=(1,1), strides=(1, 1), padding='same',activation='relu' , name = name+'_Local_Conv')(out)

    feat = Add()([feat , x])
    return feat


def resnet(input_img):
    x = Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x)
    pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
    
    ### RDB layers ###
    x = RDBlocks(pass2, 'RDB1')
    RDBlocks_list = [x]
    for i in range(2, 8):
        x = RDBlocks(x, 'RDB'+ str(i))
        RDBlocks_list.append(x)
    x = Concatenate(axis = 3)(RDBlocks_list)
    x = Conv2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(x)
    x = Conv2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(x) 
    
    ### Upsample layer ###
    x = Add()([x , pass1])
    x = SubpixelConv2D((input_r, input_c, 1), scale=4)(x)
    out = Conv2D(filters =1 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(x)
    return out




################################### rcan ################################### 
def Residual_Group(x, n_RCAB, filter_size, ratio, n_feats):
    skip_connection = x
    
    for i in range(n_RCAB):
        x = RCA_Block(x, filter_size, ratio, n_feats)
    
    x =  Conv2D(n_feats, filter_size, padding='same')(x)
    
    x = x + skip_connection
    
    return x

def RCA_Block(x, filter_size, ratio, n_feats):
    
    _res = x
    
    x = Conv2D(n_feats, filter_size, padding='same', activation='relu')(x)
    x = Conv2D(n_feats, filter_size, padding='same')(x)
    
    x = Channel_attention(x, ratio, filter_size, n_feats)
    
    x = x + _res
    
    return x

def Channel_attention(x, ratio, filter_size, n_feats):
    
    _res = x
    
    #x = tf.reduce_mean(x, axis = [1,2], keep_dims = True)
    x = Conv2D(n_feats // ratio, filter_size, padding='same', activation='relu')(x)
    x = Conv2D(n_feats, filter_size, padding='same', activation='sigmoid')(x)
    x = tf.multiply(x, _res)
    
    return x

def Up_scaling(x, filter_size, n_feats, scale):
    
    ## if scale is 2^n
    if (scale & (scale -1) == 0):
        for i in range(int(np.log2(scale))):
            x = Conv2D( 2 * 2 * n_feats, filter_size, padding='same')(x)
            x = tf.nn.depth_to_space(x, 2)
            
    elif scale == 3:
        x = Conv2D( 3 * 3 * n_feats, filter_size, padding='same')(x)
        x = tf.nn.depth_to_space(x, 3)
        
    else:
        x = Conv2D( scale * scale * n_feats, filter_size, padding='same')(x)
        x = tf.nn.depth_to_space(x, scale)
        
    return x

def rcan(x, scale, n_feats, n_RG, n_RCAB, filter_size, channel, ratio):
    x= Reshape((input_shape[0], input_shape[1],4))(input_img[:,:,:,1:])
    x = LongSkipConnection = Conv2D(n_feats, filter_size, padding='same')(x)
    
    for i in range(n_RG):
        x = Residual_Group(x, n_RCAB, filter_size, ratio, n_feats)

    x = Conv2D(n_feats, filter_size, padding='same')(x)
    x = x + LongSkipConnection
    
    x = Up_scaling(x, filter_size, n_feats, scale)
    out = Conv2D(1, filter_size, padding='same')(x)
    
    return out

def rcanmulti(input_img, scale = 4, n_feats = 64, n_RG= 3, n_RCAB= 3, filter_size = 3, channel = 4, ratio = 16):
    x = LongSkipConnection = Conv2D(n_feats, filter_size, padding='same')(input_img)    
    for i in range(n_RG):
        x = Residual_Group(x, n_RCAB, filter_size, ratio, n_feats)
    x = Conv2D(n_feats, filter_size, padding='same')(x)
    x = x + LongSkipConnection 

    x = Up_scaling(x, filter_size, n_feats, scale)
    out = Conv2D(1, filter_size, padding='same')(x)
    
    return out


##################### edsr+resnet #####################
def rcan_block(x, scale, n_feats, n_RG, n_RCAB, filter_size,channel, ratio):
    x = LongSkipConnection = Conv2D(n_feats, filter_size, padding='same')(x)    
    for i in range(n_RG):
        x = Residual_Group(x, n_RCAB, filter_size, ratio, n_feats)
    x = Conv2D(n_feats, filter_size, padding='same')(x)
    x = x + LongSkipConnection    
    x = SubpixelConv2D((input_r, input_c, 1), scale=4)(x)
    out = Conv2D(channel, filter_size, padding='same')(x)
    return out   
    
def rcan_resnet(input_img):
    x2= Reshape((input_shape[0], input_shape[1],input_shape[2]-1))(input_img[:,:,:,1:])
    x2 = rcan_block(x2, 1, 64, 3, 3, 3, 1, 16)
    
    x3= Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x3)
    pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
    x3 = RDBlocks(pass2, 'RDB1')
    RDBlocks_list = [x3]
    for i in range(2, 8):
        x3 = RDBlocks(x3, 'RDB'+ str(i))
        RDBlocks_list.append(x3)
    x3 = Concatenate(axis = 3)(RDBlocks_list)
    x3 = Conv2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(x3)
    x3 = Conv2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(x3) 
    x3 = Add()([x3 , pass1])
    x3 = SubpixelConv2D((input_r, input_c, 1), scale=4)(x3)
    x3 = Conv2D(filters =1 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(x3)

    x = Add()([x3 , x2])
    out = Conv2D(filters=output_shape[-1] , kernel_size=(3,3) , strides=(1, 1) , padding='same')(x)
    return out


##################### edsr+resnet ##################### 
def edsr_resnet(input_img):    
    x2= Reshape((input_shape[0], input_shape[1],input_shape[2]-1))(input_img[:,:,:,1:])
    x2= edsr(x2, scale =4, num_filters=64, num_res_blocks=8, res_block_scaling=0.1)
    x2 = SubpixelConv2D((input_r, input_c, 1), scale=4)(x2)
    x2 = Conv2D(1, 3, padding='same')(x2)    
    
    x3= Reshape((input_shape[0], input_shape[1],1))(input_img[:,:,:,0])
    pass1 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(x3)
    pass2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1, 1), padding='same' , activation='relu')(pass1)
    x3 = RDBlocks(pass2, 'RDB1')
    RDBlocks_list = [x3]
    for i in range(2, 8):
        x3 = RDBlocks(x3, 'RDB'+ str(i))
        RDBlocks_list.append(x3)
    x3 = Concatenate(axis = 3)(RDBlocks_list)
    x3 = Conv2D(filters=64 , kernel_size=(1,1) , strides=(1,1) , padding='same')(x3)
    x3 = Conv2D(filters=64 , kernel_size=(3,3) , strides=(1,1) , padding='same')(x3) 
    x3 = Add()([x3 , pass1])
    x3 = SubpixelConv2D((input_r, input_c, 1), scale=4)(x3)
    x3 = Conv2D(filters =1 , kernel_size=(3,3) , strides=(1 , 1) , padding='same')(x3)

    x = Add()([x3 , x2])
    out = Conv2D(filters=output_shape[-1] , kernel_size=(3,3) , strides=(1, 1) , padding='same')(x)
    return out
