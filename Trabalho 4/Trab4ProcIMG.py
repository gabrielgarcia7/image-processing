# -----------------------------------------------------------
# Assignment 4: recovery and fourier
# SCC0251 — Image Processing
# 
# Prof. Lizeth Andrea Castellanos Beltran **********
# Aluna PAE: Leo Sampaio Ferraz Ribeiro
# Monitor: Flavio Salles
#
# 2021.1
# 
# Trabalho feito por:
# Caio Augusto Duarte Basso - NUSP 10801173
# Gabriel Garcia Lorencetti - NUSP 10691891
# -----------------------------------------------------------


import numpy as np
import imageio
import scipy
import math

####### functions definitions ######


# Function for method 2
def Constrained_least_square(gamma, img, filter):
    p = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]) # inverse of Laplacian filter
    
    # adding padding
    N,M = img.shape
    a = int((N-3)/2)
    b = int((M-2)/2)
    p = np.pad(p, (a, b))

    n,m = filter.shape
    a = int((N-n)/2)
    b = int((M-m+1)/2)
    filter = np.pad(filter, (a, b))


    # moving to fourrier's domain
    img = scipy.fft.rfft2(img)
    p = scipy.fft.rfft2(p)
    filter = scipy.fft.rfft2(filter)
    h_ = np.conj(filter)


    F = ( h_ / (filter**2 + gamma * (p**2) ) ) * img

    # move back to visual domain
    F = scipy.fft.irfft2(F)
    F = np.fft.fftshift(F)

    # clip image between [0,255]
    F = np.clip(F, 0, 255) 

    return F

def Gaussian_filter (k = 3, sigma = 1.0):
    arx = np.arange ((-k // 2 ) + 1.0 , ( k // 2 ) + 1.0 )
    x, y = np.meshgrid ( arx , arx )
    filt = np.exp ( -(1/2)*( np.square ( x ) + np.square ( y ) ) / np.square ( sigma ) )
    return filt /np.sum( filt )


# Function that calculates the Root-mean-square deviation (RSME)
def RSME(g, r):
	x, y = g.shape
	return math.sqrt((np.sum(np.square(g - r)))/(x*y))
####################################

####### input reading #######
_img_ref_name = input().rstrip() # read reference image's name
_img_deg_name = input().rstrip() # read degraded image's name
_func_used = int(input()) # read chosen function's id 1 <= x <= 2
_gamma = float(input()) # read parameter γ


if _func_used == 1: # if using method 1 (Adaptive Denoising)
    #_row = ***************
    _filter_size = int(input())
    _mode = input().rstrip() # read denoising mode: "average" or "robust"

elif _func_used == 2: # if using method 2 (Constrained Least Squares)

    _filter_size = int (input())
    _sigma = float (input()) # reads σ for the gaussian filter

##############################

####### reading image #######
img = imageio.imread(_img_deg_name)
ref_img = imageio.imread(_img_ref_name)
#############################

###### restoring image ######
if _func_used == 1:


elif _func_used == 2:
    filter = Gaussian_filter(_filter_size, _sigma)
    img = Constrained_least_square(_gamma, img, filter)


#### Comparing with reference image ####


print("%.4f" % RSME(img, ref_img)) # printing the rsme value
#################################