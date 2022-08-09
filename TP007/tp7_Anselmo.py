import numpy as np
from scipy import stats
from scipy.signal import convolve2d

def test(num1, num2):
	print('Numero 1 es: ' + str(num1) )
	print('Numero 2 es: ' + str(num2) )
	return num1 * num2

def nn_downsampling_x2(img):
	img2 = img[::2, ::2]
	return img2

def avg_downsampling_x2(img):
	tam = img.shape[0]//2
	img2 = img.reshape(-1, 2, tam, 2).sum((-1, -3)) / 2
	return img2

def kerGaus(img, tam, std):
    mu, sigma = 0, std # media y desvio estandar
    normal = stats.norm(mu, sigma)
    tam = np.floor(tam / 2)
    X,Y = np.mgrid[-tam:tam+1:1, -tam:tam+1:1]
    X = normal.pdf(X)
    Y = normal.pdf(Y)
    kernel = np.matmul(X,Y)
    kernel = kernel / kernel.sum()
    img_filt = convolve2d(img, kernel , 'same')
    return img_filt

def gaussian_nn_downsampling_x2(img, std):
	tam = 3
	img2 = kerGaus(img, tam, std)
	return nn_downsampling_x2(img)

def nn_upsampling_x2(img):
	N, M = img.shape
	img2 = np.zeros( (N*2, M*2) )
	img2[::2, ::2] = img
	img2[::2,1::2] = img
	img2[1::2,::2] = img
	img2[1::2, 1::2] = img
	return img2

def bilinear_upsampling_x2(img):
	N, M = img.shape
	img2 = np.zeros( (N*2, M*2) )
	img2[::2, ::2] = img
	kerAux = np.array([ [1/4, 1/2, 1/4], [1/2, 1, 1/2], [1/4, 1/2, 1/4] ])
	img2 = convolve2d(img2, kerAux , 'same')
	return img2

def bicubic_upsampling_x2(img):
	N, M = img.shape
	img2 = np.zeros( (N*2, M*2) )
	img2[::2, ::2] = img
#	0.0156        0  -0.0781  -0.1250  -0.0781        0   0.0156
#        0        0        0        0        0        0        0
#  -0.0781        0   0.3906   0.6250   0.3906        0  -0.0781
#  -0.1250        0   0.6250   1.0000   0.6250        0  -0.1250
#  -0.0781        0   0.3906   0.6250   0.3906        0  -0.0781
#        0        0        0        0        0        0        0
#   0.0156        0  -0.0781  -0.1250  -0.0781        0   0.0156
	kerAux = np.array([[0.0156,0,-0.0781,-0.1250,-0.0781,0,0.0156],[0,0,0,0,0,0,0],[-0.0781,0,0.3906,0.6250,0.3906,0,-0.0781],[-0.1250,0,0.6250,1,0.6250,0,-0.1250],[-0.0781,0,0.3906,0.6250,0.3906,0,-0.0781],[0,0,0,0,0,0,0],[0.0156,0,-0.0781,-0.1250,-0.0781,0,0.0156] ] )
	img2 = convolve2d(img2, kerAux , 'same')
	return img2

class tp7:
    def __init__(self):
    	print()