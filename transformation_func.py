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
    
    
