import numpy as np
import scipy
import pandas as pd
import imageio
import pywt
import time
from matplotlib import pyplot as plt
from glob import glob
from skimage.metrics import structural_similarity as ssim
from skimage.transform import pyramid_reduce


def psnr(hr,lr):
    mse_ = mse(hr,lr)
    if mse_ == 0.:
        psnr = 100
    else:
        psnr = 20 * np.log10(hr.max() / np.sqrt(mse_))
    return psnr 

def mse(x,y):
    return np.mean((y-x)**2)
  
def combine_domain(weights,a,b,d):
    patch = []
    for i in range(weights.shape[0]):
        patch.append(a[i,:,:,:]*weights[i,0] + b[i,:,:,:]*weights[i,1] + d[i,:,:,:]*weights[i,2])
#                                            d[i,:,:,:]*weights[i,3] + e[i,:,:,:]*weights[i,4])
    return np.asarray(patch)

def bicubic(x_data):
    result =[]
    r, c = x_data[0].shape[1] * scale, x_data[1].shape[1] * scale
    for i in range(len(x_data)):
        result.append(cv2.resize(x_data[i], (c,r), interpolation=cv2.INTER_CUBIC))
    return np.asarray(result)

def crop(img_lr, input_r, input_c):
    r, c, _ = img_lr.shape
    n_col = c-input_c+1 ## number of patch/column
    n_row = r-input_r+1 ## number of patch/row
    patch = []
    for i in range(n_row):
        for j in range(n_col):
            patch.append(img_lr[i:i+input_r, j:j+input_c, :])
    return patch

def assemble(predY, predCR, predCB):
    result = []
    n, r, c,_ = predY.shape
    def trim(x):
        if x.ndim == 4:
            x = np.reshape(x, x.shape[:-1])
        return x
    def roundto(x):
        x[x<0] = 0
        x[x>255] = 255
        return x
    predY = trim(predY)
    predCR = trim(predCR)
    predCB = trim(predCB)
    for i in range(n):
        whole_pic = np.zeros((r,c,3))
        whole_pic[:,:,0] = roundto(np.rint(predY[i,:,:]))
        whole_pic[:,:,1] = roundto(np.rint(predCR[i,:,:]))
        whole_pic[:,:,2] = roundto(np.rint(predCB[i,:,:])) 
        result.append(whole_pic)
    return result

def reconstruct(img_lr, patch, input_r, input_c, scale):
    r, c, _ = img_lr.shape
    R, C = r*scale, c*scale
    n_col = c-input_c+1 ## number of patch/column
    n_row = r-input_r+1 ## number of patch/row
    output_r, output_c = input_r*scale, input_c*scale
    img_hr = np.zeros((R, C, 3))
    index = np.zeros((R, C, 3)) ## count # of times added for each pixel
    k = 0
    for i in range(n_row):
        for j in range(n_col):
            img_hr[i*scale:i*scale+output_r, j*scale:j*scale+output_c,:] += patch[k]
            index[i*scale:i*scale+output_r, j*scale:j*scale+output_c,:] += np.ones((output_r, output_c, 3))
            k += 1
    return np.rint(img_hr/index)


def predict_this_set(model, list_files_HR, version, remainder):
    psnr_Ychannel = []
    ssim_Ychannel = []
    for i in range(len(list_files_HR)):
        file_HR = list_files_HR[i]
        img_hr = cv2.imread(file_HR, cv2.IMREAD_COLOR)
        img_hr = cv2.cvtColor(img_hr, cv2.COLOR_BGR2YCR_CB)
        hr_shape = img_hr.shape[:2]
        img_hr = img_hr[:hr_shape[0]//scale*scale, : hr_shape[1]//scale*scale]
        
        if version == 1:
            img_lr = cv2.resize(img_hr, (hr_shape[1] // (scale//2),
                                        hr_shape[0] // (scale//2)), interpolation=cv2.INTER_LINEAR)
            img_lr = cv2.GaussianBlur(img_lr,(5,5),0)
            img_lr = cv2.resize(img_lr, (hr_shape[1] // scale,
                                        hr_shape[0] // scale), interpolation=cv2.INTER_NEAREST)
            img_lr = cv2.GaussianBlur(img_lr,(5,5),0)
            
        elif version ==2:
            img_lr = cv2.resize(img_hr, (hr_shape[1] // (scale//2),
                                        hr_shape[0] // (scale//2)), interpolation=cv2.INTER_LANCZOS4)
            img_lr = cv2.resize(img_lr, (hr_shape[1] // scale,
                                        hr_shape[0] // scale), interpolation=cv2.INTER_NEAREST)
            img_lr = cv2.GaussianBlur(img_lr,(5,5),0)
            
        elif version ==3:
            img_lr = pyramid_reduce(img_hr, downscale=4, sigma=None, order=1, mode='reflect', cval=0, multichannel=True)
            img_lr = cv2.GaussianBlur(img_lr,(5,5),0)
            
        elif version ==4:
            img_lr = pyramid_reduce(img_hr, downscale=2, sigma=None, order=1, mode='reflect', cval=0, multichannel=True)
            img_lr = cv2.GaussianBlur(img_lr,(7,7),0)
            img_lr = pyramid_reduce(img_lr, downscale=2, sigma=None, order=1, mode='constant', cval=0, multichannel=True)
            img_lr = cv2.GaussianBlur(img_lr,(3,3),0)
        
        
        ##crop patches
        lr = crop(img_lr, input_r, input_c)
        Yx_data= np.asarray(lr)[:,:,:,0]
        CRx_data = np.asarray(lr)[:,:,:,1]
        CBx_data = np.asarray(lr)[:,:,:,2]
        hr = crop(img_hr, input_r*scale, input_c*scale)
        Yy_data= np.asarray(hr)[:,:,:,0]
        
        input_shape_dct, output_shape_dct, x_test_dct, y_test_dct = process_data(Yx_data,Yy_data,space)        
        input_shape_db6, output_shape_db6, x_test_db6, y_test_db6 = process_data_wave(Yx_data,Yy_data,db6)

        padding_shape = np.max([input_shape_dct[0], input_shape_db6[0]])
        test_unpad = [x_test_dct, x_test_db6]
        test = []
        ## pad to the same shape
        for k in range(len(test_unpad)):
            if test_unpad[k].shape[1] == padding_shape:
                test.append(test_unpad[k])
            else:
                target = test_unpad[k]
                v = []
                for j in range(target.shape[0]):
                    temp = padding(target[j, :,:,:], padding_shape)
                    v.append(temp)
                test.append(np.asarray(v))
        ## fold to one tensor
        for k in range(1,len(test)):
            test[0] = np.concatenate((test[0], test[k]), axis=3)
        x_test = test[0]
        y_test = np.reshape(Yy_data, (Yy_data.shape[0],Yy_data.shape[1],Yy_data.shape[2],1))


        #######  predict and assemble reconstructed patches  ####### 
        ## Y channel
        pred_patch = model.predict(x_test)
########################################

        predY = pred_patch
        ## CR channel 
        predCR = bicubic(CRx_data)
        ## CB channel
        predCB = bicubic(CBx_data)
        ####### append results  ####### 
        pred_patches = assemble(predY, predCR, predCB)
        reconstruction = reconstruct(img_lr, pred_patches, input_r, input_c, scale)
        psnr_Ychannel.append(psnr(img_hr[:,:,0], reconstruction[:,:,0]))
        ssim_Ychannel.append(ssim(img_hr[:,:,0], reconstruction[:,:,0]))


        if i % show_step == remainder:
            # bicubic for channel CR/CB does not work for version 4
            if version == 3 or version == 4 :
                reconstruction[:,:,1] = img_hr[:,:,1]
                reconstruction[:,:,2] = img_hr[:,:,2]
            def cov_type(x):
                if isinstance(x[0,0,0], float) and x.max()<=1:
                    return 'float32'
                else:
                    return 'uint8'
            lr_rgb = cv2.cvtColor(img_lr.astype(cov_type(img_lr)), cv2.COLOR_YCR_CB2RGB)
            recons_rgb = cv2.cvtColor(reconstruction.astype(cov_type(reconstruction)), cv2.COLOR_YCR_CB2RGB)
            hr_rgb = cv2.cvtColor(img_hr.astype(cov_type(img_hr)), cv2.COLOR_YCR_CB2RGB)
            fig = plt.figure(figsize=(16,5))
            plt.subplot(1, 3, 1)
            fig.gca().set_title('Low Resolution')
            imgplot = plt.imshow(lr_rgb)
            plt.subplot(1, 3, 2)
            fig.gca().set_title('Reconstruction')
            imgplot = plt.imshow(recons_rgb)
            plt.subplot(1, 3, 3)
            fig.gca().set_title('Ground Truth')
            imgplot = plt.imshow(hr_rgb)
            plt.show()

    print('Ychannel PSNR = ' + str(np.mean(psnr_Ychannel)))
    print('Ychannel SSIM = ' + str(np.mean(ssim_Ychannel)))
    return np.mean(psnr_Ychannel)
