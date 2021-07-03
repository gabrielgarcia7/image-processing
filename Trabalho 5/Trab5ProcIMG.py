# -----------------------------------------------------------
# Assignment 5:  image descriptors
# SCC0251 — Image Processing
# 
# SCC5830/0251 — Prof. Moacir Ponti
# Teaching Assistant: Leo Sampaio Ferraz Ribeiro
# Teaching Assistant: Flavio Salles
#
# 2021.1
# 
# Made by:
# Caio Augusto Duarte Basso - NUSP 10801173
# Gabriel Garcia Lorencetti - NUSP 10691891
# -----------------------------------------------------------


import numpy as np
import imageio
import math
import scipy
#import scipy.ndimage
import sys


####### functions definitions ######

# Function for transforming RGB to gray scale
def To_grayscale(img):
    N, M = img.shape[:-1]
    for i in range(N):
        for j in range(M):
            img[i, j, 0] = np.floor(img[i, j, 0] * 0.299 + img[i, j, 1] * 0.587 + img[i, j, 2] * 0.114)

    img = img[:, :, 0:1].squeeze()
    return img

# Quantisation to b bits
def Quantisation (img, b):
    #print(img)
    img = np.right_shift(img, 8-b)
    #print(img)
    return img

# Calculates and concatenates the descriptors
def Create_descriptors(img, b):
    


    dc = Normalized_histogram(img, (2**b))
    dt = Haralick_descriptor(img, (2**b)-1)
    dg = Gradients_histogram(img)


    # print('descriptor dc')
    # print(dc)
    # print('descriptor dt')
    # print(dt)
    # print('descriptor dg')
    # print(dg)
    # print('concatenado')
    desc = np.concatenate((dc, dt, dg))

    # print("imprimindo vetorzao concatenado")
    # print(desc)

    return desc

# Calculates normalized histogram, returns vector
def Normalized_histogram(img, b):
    hist = np.zeros(b, dtype=np.int32)
    for i in range(b):
		# sum all positions in which A == i is true
        pixels_value_i = np.sum(img == i)
		# store it in the histogram array
        hist[i] += pixels_value_i
    
    N,M = img.shape
    #simply divide h(k) by the total number of pixels in the image.
    p = hist / (N*M)
    """ print('d_c')
    print(p) """
    p = p / np.linalg.norm(p, b)
    #print('imprimindo dc linalg')
    #print(p)
    return p

def energy(c):
    return np.sum(c**2)

def entropy(c):
    return -np.sum(c*np.log(c+0.001))

def contrast(c):    
    linha, coluna = c.shape
    sum = 0
    
    for i in range(linha):
        for j in range(coluna):
            sum += ((i-j)**2)*c[i, j]

    return sum/(linha*coluna)

def correlation(c):

    mi_i = 0 
    mi_j = 0
    delta_i = 0
    delta_j = 0
    correlation = 0
    linha, coluna = c.shape

    # mi_i
    for i in range(linha):
        alt = 0
        for j in range(coluna):
            alt += c[i, j]

        mi_i += i * alt

    # mi_j
    for j in range(linha):
        alt = 0
        for i in range(coluna):
            alt += c[i, j]

        mi_j += j * alt
    

    # delta_i
    for i in range(linha):
        alt = 0
        for j in range(coluna):
            alt += c[i, j]
        delta_i += ((i-mi_i)**2) * alt

    # delta_j
    for j in range(linha):
        alt = 0
        for i in range(coluna):
            alt += c[i, j]
        delta_j = ((j-mi_j)**2) * alt

    # correlation
    for i in range(linha):
        for j in range(coluna):
            correlation += ((i*j*c[i,j])-(mi_i*mi_j))/(delta_i*delta_j)


    return correlation

def homogeneity(c):
    linha, coluna = c.shape
    sum = 0
    
    for i in range(linha):
        for j in range(coluna):
            sum += c[i, j]/(1 + abs(i -j))

    return sum

def coocurrence(g, intensities):
    linha, coluna = g.shape

    #print(g)
    c = np.zeros([intensities+1, intensities+1])

    for i in range(linha-1):
        for j in range(coluna-1):
            c[int(g[i, j]), int(g[i+1, j+1])] += 1

    c = c / np.sum(c)
    return c

# Calculates haralick texture descriptor, returns vector
def Haralick_descriptor(img, intensities):
    
    c = coocurrence(img, intensities)

    dt = np.zeros(5)

    dt[0] = energy(c)
    dt[1] = entropy(c)
    dt[2] = contrast(c)
    dt[3] = correlation(c)
    dt[4] = homogeneity(c)
    """ print("d_t")
    print(dt) """

    dt = dt / np.linalg.norm(dt)

    return dt

# Calculates histogram of oriented gradients, returns vector
def Gradients_histogram(img):

    img = img.astype('float64')

    wsx = [ [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]]

    wsy = [ [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]]

    gradientX = scipy.ndimage.convolve(img, wsx, mode='grid-mirror')
    gradientY = scipy.ndimage.convolve(img, wsy, mode='grid-mirror')
    #np.set_printoptions(threshold=sys.maxsize)
    #print("gradientX: ", gradientX)
    #print("gradientY: ", gradientY)

    soma = np.sum(np.sqrt(gradientX**2 + gradientY**2))
    #print("soma: ", soma)

    m = np.sqrt(gradientX**2 + gradientY**2)/soma

    #print("matriz M: ", m)

    np.seterr(divide='ignore', invalid='ignore')

    fi = np.arctan(gradientY/gradientX)
    fi = fi + np.pi/2
    fi = np.degrees(fi)
    #print("fi: ", fi)

    # digitise the angles into 9 bins
    bins = np.arange(20, 180, 20)
    fi_d = np.digitize(fi, bins, right=False)

    linhas, colunas = img.shape

    dg = np.zeros(9)

    for i in range(linhas):
        for j in range(colunas):
            dg[fi_d[i,j]] += m[i,j]

    """ print("d_g")
    print(dg) """

    dg = dg / np.linalg.norm(dg)

    return dg

# Function that calculates the Root-mean-square deviation (RSME)
def RSME(g, r):
	x, y = g.shape
	return math.sqrt((np.sum(np.square(g - r)))/(x*y))

# Function that compares two descriptors
def Difference(a, b):
    sum = 0.0
    # print("imprimindo a")
    # print(a)
    # print("imprimindo b")
    # print(b)
    for i in a:
        for j in b:
            sum = (i - j)**2
    return math.sqrt(sum)
####################################

####### input reading #######
_img_obj = input().rstrip() # read object image's name
_img_ref = input().rstrip() # read reference image's name
_quant_par = int(input()) # read quantisation parameter b <= 8
##############################

####### reading image #######
obj_img = imageio.imread(_img_obj)
ref_img = imageio.imread(_img_ref)
#############################

###### Preprocessing and Quantisation ######
grayscale_img = To_grayscale(obj_img)
grayscale_img = Quantisation(grayscale_img, _quant_par)

ref_img = To_grayscale(ref_img)
ref_img = Quantisation(ref_img, _quant_par)
############################################

###### Creating Image Descriptors ##########
descriptor = Create_descriptors(grayscale_img, _quant_par)
############################################

###### Finding the object #########
N,M = ref_img.shape
#print(N,M)
windows = np.empty((int(N/16),int(M/16),32,32))
wind_descr = np.empty((int(((N/16)) * ((M/16))), (2**_quant_par)+5+9))
for i in range (int(N/16)-1): # creating windows and their descriptors
    for j in range(int(M/16)-1):

        #print("(X) imagem de ", i*16, " a ", (i+2)*16)
        #print("(Y) imagem de ", j*16, " a ", (j+2)*16)
        windows[i][j] = ref_img[i*16:(i+2)*16,j*16:(j+2)*16]
        
        #print("imprimindo os wind")
        #print(windows[i][j])
        wind_descr[int(i* N/16 + j)] = Create_descriptors(windows[i][j], _quant_par)
        #print(wind_descr)

min_dist = sys.maxsize
close_x = -1
close_y = -1
for x in range (len(descriptor)): # comparing descriptors with descriptors
    for y in range (int(N/16 * M/16)):
        a = Difference(descriptor, wind_descr[y])
        if min_dist > a:
            min_dist = a
            close_x = int(y / N/16)
            close_y = int((y % N/16))

###################################

###### Printing the results #####
print(close_x, " ", close_y)
#################################