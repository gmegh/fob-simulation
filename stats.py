import batoid

import galsim
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm.notebook import tqdm
import trace_dataset as trace
import scipy.stats as st
import seaborn as sns
from scipy import fftpack, ndimage
import skimage.measure    
import warnings
warnings.filterwarnings('ignore')


#We can check that the result is the same if we use the full 2D data array
def image_statistics_2D(Z):
    y,x = np.mgrid[:Z.shape[0],:Z.shape[1]]
    
    #Centroid (mean)
    mx = np.sum(x*Z)/np.sum(Z) # x-center of gravity
    my = np.sum(y*Z)/np.sum(Z) # y-center of gravity

    mu02 = np.sum(Z*(y - my)**2)/np.sum(Z) # variance/width along y-axis
    mu20 = np.sum(Z*(x - mx)**2)/np.sum(Z) # variance/width along x-axis

    mu03 = np.sum((y - my)**3*Z)/np.sum(Z)
    mu30 = np.sum((x - mx)**3*Z)/np.sum(Z)

    mu04 = np.sum((y - my)**4*Z)/np.sum(Z)
    mu40 = np.sum((x - mx)**4*Z)/np.sum(Z)

    #SD is the sqrt of the variance
    sx,sy = np.sqrt(mu20),np.sqrt(mu02)

    #Skewness is the third central moment divided by SD cubed
    skx = mu30/sx**3
    sky = mu03/sy**3

    #Kurtosis is the fourth central moment divided by SD to the fourth power
    kx = mu40/sx**4 - 3
    ky = mu04/sy**4 - 3

    return mx,my,sx,sy,skx,sky,kx,ky

# +
def chi_square(image, gaussian):
    chi = (image - gaussian)**2/gaussian
    
    return np.sum(chi)

def crop_normalize(image, length, noise):
    image[image < 0] = 0
    center = np.where(image == np.max(image))
    image = image[center[0][0] - length : center[0][0] + length, center[1][0] - length : center[1][0] + length]
    if noise:
        image = add_noise(image)
    image = image/np.sum(image)
    return image
    
def add_noise(image, level = 1_000):
    image = image + np.random.normal(loc = level, size = (image.shape[0], image.shape[1]))
    
    return image


# -

def load_data(name, method, duration, num_simulations, num_phot):
    dat = {
        'mean_x' : [],
        'mean_y' : [],
        'var_x' : [],
        'var_y' : [],
        'skew_x' : [],
        'skew_y' : [],
        'kurt_x' : [],
        'kurt_y' : [],
        'fft_var_x' : [],
        'fft_var_y' : [],
        'entropy': [],
        'hu1': [],
        'hu2': [],
        'hu3': [],
        'hu4': [],
        'hu5': [],        
        'hu6': [],
        'hu7': [],
        'chi_150': [],
        'chi_gaus': [],
        'chi_150_noise': [],
        'chi_gaus_noise': []
    }
    
    fob_150s = np.loadtxt('data/{}s/{}_{}_{}_{}_{}.txt'.format('FRB', 'FRB', 'FFT', '150s', 750000, 0), delimiter = ',')
    fob_150s = crop_normalize(fob_150s, 6, noise = False)
    fob_150s = fob_150s/np.sum(fob_150s)

    x, y = np.mgrid[0:12:1, 0:12:1]
    pos = np.dstack((x, y))
    rv = st.multivariate_normal([6, 6], [[2.0, 0], [0, 2]])
    gaus = rv.pdf(pos)
    gaus = gaus/np.sum(gaus)
    
    for it in tqdm(range(num_simulations)):
        image = np.loadtxt('data/{}s/{}_{}_{}_{}_{}.txt'.format(name, name, method, duration, num_phot, it), delimiter = ',')
        image[image < 0] = 0
        cx,cy,sx,sy,skx,sky,kx,ky = image_statistics_2D(image)

        W, H = 20, 20
        fft2 = fftpack.fft2(image)/(W*H)
        ff = np.abs(fftpack.fftshift(fft2))

        cx_fft,cy_fft,sx_fft,sy_fft,skx_fft,sky_fft,kx_fft,ky_fft = image_statistics_2D(ff)
        en = skimage.measure.shannon_entropy(image)
        [hu1, hu2, hu3, hu4, hu5, hu6, hu7] = skimage.measure.moments_hu(image)
        
        image_c = crop_normalize(image, 6, noise = False)
        image_c = image_c/np.sum(image_c)
        
        image_n = crop_normalize(image, 6, noise = False)
        image_n = image_n/np.sum(image_n)

        dat['chi_150'].append(chi_square(image_c, fob_150s))
        dat['chi_gaus'].append(chi_square(image_c, gaus))
        dat['chi_150_noise'].append(chi_square(image_n, fob_150s))
        dat['chi_gaus_noise'].append(chi_square(image_n, gaus))

        dat['mean_x'].append(cx)
        dat['mean_y'].append(cy)
        dat['var_x'].append(sx)
        dat['var_y'].append(sy)
        dat['skew_x'].append(skx)
        dat['skew_y'].append(sky)
        dat['kurt_x'].append(kx)
        dat['kurt_y'].append(ky)
        dat['fft_var_x'].append(sx_fft)
        dat['fft_var_y'].append(sy_fft)
        dat['entropy'].append(en)
        dat['hu1'].append(hu1)
        dat['hu2'].append(hu2)
        dat['hu3'].append(hu3)
        dat['hu4'].append(hu4)
        dat['hu5'].append(hu5)
        dat['hu6'].append(hu6)
        dat['hu7'].append(hu7)
    
    return dat

def customize_plot():
    plt.grid()
    plt.legend(fontsize = 16)
    plt.tick_params(labelsize = 16)
    plt.xlabel('', fontsize = 20)
    plt.ylabel('', fontsize = 20)

def plot_entropy(label, *args):
    plt.figure(figsize = (15,7))
    plt.subplot(1,2,1)
    for arg in args:
        sns.distplot(arg[0][attribute], hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    plt.grid()
    plt.legend(fontsize = 16)
    plt.tick_params(labelsize = 16)
    plt.xlabel('', fontsize = 20)
    plt.ylabel('', fontsize = 20)
    plt.xlabel(label)
    plt.ylabel('Probability')


def plot_stat(attribute, label, *args):
    # -------- Plot X and Y axis ---------
    # X Axis
    plt.figure(figsize = (15,7))
    plt.subplot(1,2,1)
    for arg in args:
        sns.distplot(arg[0][attribute + '_x'], hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    customize_plot()
    plt.xlabel(label + ' - X axis')
    plt.ylabel('Probability')

    # Y Axis
    plt.subplot(1,2,2)
    for arg in args:
        sns.distplot(arg[0][attribute + '_y'], hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    customize_plot()
    plt.xlabel(label + ' - Y axis')
    plt.ylabel('Probability')
    # -------------------------------------

    
    # -------- Plot X, Y and main axis ---------
    # X axis
    plt.figure(figsize = (23,7))
    plt.subplot(1,3,1)
    for arg in args:
        sns.distplot(arg[0][attribute + '_x'], hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    customize_plot()
    plt.xlabel(label + ' - X axis')
    plt.ylabel('Probability')

    
    # Y axis
    plt.subplot(1,3,2)
    for arg in args:
        sns.distplot(arg[0][attribute + '_y'], hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    customize_plot()
    plt.xlabel(label + ' - Y axis')
    plt.ylabel('Probability')


    # Main axis
    plt.subplot(1,3,3)
    for arg in args:
        th = np.arctan(np.array(arg[0][attribute + '_y'])/np.array(arg[0][attribute + '_x']))
        main_axis = np.array(arg[0][attribute + '_x'])/np.cos(th)
    
        sns.distplot(abs(main_axis), hist = False, kde = True,
                         kde_kws = {'shade': True, 'linewidth': 3}, label = arg[1])
    customize_plot()
    plt.xlabel(label)
    plt.ylabel('Probability')
    # -------------------------------------

def printSensorImage(traced_photons, wavelengths, STAMP_SIZE, PIXEL_SIZE, SKY_LEVEL, noise = False):
    # Convert rays to pixels for galsim sensor object. Put batoid results back into photons.
    photons = galsim.PhotonArray(len(wavelengths))
    photons.x = traced_photons.x/PIXEL_SIZE
    photons.y = traced_photons.y/PIXEL_SIZE
    photons.dxdz = traced_photons.vx/traced_photons.vz
    photons.dydz = traced_photons.vy/traced_photons.vz
    photons.wavelength = wavelengths
    photons.flux = ~traced_photons.vignetted

    image = galsim.Image(STAMP_SIZE, STAMP_SIZE)

    # Set image center at the center of the photons
    image.setCenter(
        int(np.mean(photons.x[~traced_photons.vignetted])),
        int(np.mean(photons.y[~traced_photons.vignetted]))
    )

    # Add photons to the image
    sensor = galsim.Sensor()
    sensor.accumulate(photons, image)

    # Add Gaussian background.
    if noise == True:
        rng = galsim.BaseDeviate()
        gd = galsim.GaussianDeviate(rng, sigma = np.sqrt(100_000))
        gd.add_generate(image.array)

    return image.array
