import numpy as np
from scipy import stats
from scipy.signal import convolve2d

def test(num1, num2):
	print('Numero 1 es: ' + str(num1) )
	print('Numero 2 es: ' + str(num2) )
	return num1 * num2

def nn_downsampling_x2(img):
	return img[::2, ::2]

def avg_downsampling_x2(img):
	tam = img.shape[0]//2
	return img.reshape(-1, 2, tam, 2).sum((-1, -3)) / 2

def kerGaus(img, tam, std):
    mu, sigma = 0, std # media y desvio estandar
    normal = stats.norm(mu, sigma)
    tam = np.floor(tam / 2)
    X,Y = np.mgrid[-tam:tam+1:1, -tam:tam+1:1]
    X = normal.pdf(X)
    Y = normal.pdf(Y)
    kernel = np.matmul(X,Y)
    kernel = kernel / kernel.sum()
    return convolve2d(img, kernel , 'same')

def gaussian_nn_downsampling_x2(img, std):
	tam = 3 # Tamaño del kernel gausiano
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
	# kerAux
	#	/  0.0156   0   -0.0781  -0.1250  -0.0781   0   0.0156  \
	#   |       0   0         0        0        0   0        0  |
	#   | -0.0781   0    0.3906   0.6250   0.3906   0   -0.0781 |
	#   | -0.1250   0    0.6250   1.0000   0.6250   0   -0.1250 |
	#   | -0.0781   0    0.3906   0.6250   0.3906   0   -0.0781 |
	#   |       0   0         0        0        0   0         0 |
	#   \  0.0156   0   -0.0781  -0.1250  -0.0781   0    0.0156 /
	kerAux = np.array([[0.0156,0,-0.0781,-0.1250,-0.0781,0,0.0156],[0,0,0,0,0,0,0],[-0.0781,0,0.3906,0.6250,0.3906,0,-0.0781],[-0.1250,0,0.6250,1,0.6250,0,-0.1250],[-0.0781,0,0.3906,0.6250,0.3906,0,-0.0781],[0,0,0,0,0,0,0],[0.0156,0,-0.0781,-0.1250,-0.0781,0,0.0156] ] )
	img2[::2, ::2] = img
	return convolve2d(img2, kerAux , 'same')

def fft_resampling(img, new_shape):
	fft = np.fft.fftshift( np.fft.fft2(img) ) / img.size # tranformada de fourier de la imagen original
	newImg = np.zeros( new_shape, complex)# Creo una imagen con ceros complejos y de las dimensiones de la imegen de salida
	x0 = 0
	y0 = 0
	x1, y1 = fft.shape
	# 1) Si alguna diemension es mas grande, recorto simetricamente
	if fft.shape[0] > new_shape[0]:
		x0 = ( fft.shape[0] - new_shape[0] ) // 2
		x1 = ( fft.shape[0] + new_shape[0] ) // 2
	if fft.shape[1] > new_shape[1]:
		y0 = ( fft.shape[1] - new_shape[1] ) // 2
		y1 = ( fft.shape[1] + new_shape[1] ) // 2
	fft = fft[x0:x1, y0:y1]
	# 2) Si alguna dimension es mas pequeña, completo con ceros
	x0 = 0
	y0 = 0
	x1, y1 = fft.shape
	if new_shape[0] > fft.shape[0]:
		x0 = ( new_shape[0] - fft.shape[0] ) // 2
		x1 = ( new_shape[0] + fft.shape[0] ) // 2
	if new_shape[1] > fft.shape[1]:
		y0 = ( new_shape[1] - fft.shape[1] ) // 2
		y1 = ( new_shape[1] + fft.shape[1] ) // 2

	newImg[x0:x1, y0:y1] = fft

	return newImg.size * np.abs( np.fft.ifft2( np.fft.ifftshift( newImg ) ) )

# Cuantizacion
def cuantize_uniform(img, levels):
	return np.round( img * ( levels - 1 ) ) / ( levels - 1 ) 

def cuantize_dithering_scanline(img, levels):
	newImg = np.zeros( img.shape )
	for numFil, fila in enumerate(img):
		error = 0
		for numCol, pixel in enumerate(fila):
			newImg[numFil, numCol] = np.round( (pixel + error ) * ( levels - 1 ) ) / ( levels - 1 )
			error = error - (newImg[numFil, numCol] - pixel)
	return newImg

def cuantize_floyd_steinberg(img, levels):
	N, M = img.shape
	newImg = img
	error = 0
	for x in range(1, N - 1):
		for y in range(1, M - 1):
			oldPixel = newImg[x, y]
			newPixel = np.round( oldPixel * ( levels - 1 ) ) / ( levels - 1 )
			newImg[x,y] = newPixel
			error = oldPixel - newPixel
			newImg[x + 1, y    ] = newImg[x + 1, y    ] + error * ( 7/16)
			newImg[x - 1, y + 1] = newImg[x - 1, y + 1] + error * ( 3/16)
			newImg[x    , y + 1] = newImg[x    , y + 1] + error * ( 5/16)
			newImg[x + 1, y + 1] = newImg[x + 1, y + 1] + error * ( 1/16)
	return newImg

class tp7:
    def __init__(self):
    	print()