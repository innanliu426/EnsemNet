import numpy as np
import pywt
from scipy import fftpack



######## trasformations ########        
def dct2(x):
    return fftpack.dct(fftpack.dct(x, norm = 'ortho' ,axis=0),  norm = 'ortho' ,axis=1)
def idct2(x):
    return fftpack.idct(fftpack.idct(x, norm = 'ortho' ,axis=1),  norm = 'ortho' ,axis=0)

def space(x):
    return x
def ispace(x):
    return x

def db6(x):
    cA, (cH, cV, cD) = pywt.dwt2(x,'db6')
    return np.concatenate((np.concatenate((cA, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
def idb6(dwt):
    r, c = dwt.shape
    r, c = int(r/2), int(c/2)
    cA, cH, cV, cD = dwt[:r, :c], dwt[:r, c:], dwt[r:, :c], dwt[r:, c:]
    return pywt.idwt2((cA, (cH, cV, cD)), 'db6') 

def bior(x):
    cA, (cH, cV, cD) = pywt.dwt2(x,'bior5.5')
    return np.concatenate((np.concatenate((cA, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
def ibior(dwt):
    r, c = dwt.shape
    r, c = int(r/2), int(c/2)
    cA, cH, cV, cD = dwt[:r, :c], dwt[:r, c:], dwt[r:, :c], dwt[r:, c:]
    return pywt.idwt2((cA, (cH, cV, cD)), 'bior5.5') 

def rbio(x):
    cA, (cH, cV, cD) = pywt.dwt2(x,'rbio6.8')
    #return cA
    return np.concatenate((np.concatenate((cA, cH), axis=1), np.concatenate((cV, cD), axis=1)), axis=0)
    
def irbio(dwt):
    r, c = dwt.shape
    r, c = int(r/2), int(c/2)
    cA, cH, cV, cD = dwt[:r, :c], dwt[:r, c:], dwt[r:, :c], dwt[r:, c:]
    return pywt.idwt2((cA, (cH, cV, cD)), 'rbio6.8') 
    
    
def psnr(hr,lr):
    mse_ = mse(hr,lr)
    if mse_ == 0.:
        psnr = 100
    else:
        psnr = 20 * np.log10(hr.max() / np.sqrt(mse_))
    return psnr 

def mse(x,y):
    return np.mean((y-x)**2)

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
