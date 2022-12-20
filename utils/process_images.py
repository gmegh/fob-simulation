import numpy as np
import galsim
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def add_noise(image, sky_level = 1000):
    """
    Add noise to FOB image

    Parameters  
    ----------
    image : numpy.ndarray
        FOB image

    Returns
    -------
    image_array : numpy.ndarray
        FOB image with noise
    """

    image[image < 0] = 0
    image += sky_level
    
    return np.random.normal(loc = image, scale = np.sqrt(image))

def remove_zero_frequency(image):
    """
    Remove zero frequency from FOB image

    Parameters
    ----------
    image : numpy.ndarray
        FOB image

    Returns
    -------
    image_new : numpy.ndarray
        FOB image with zero frequency removed
    """

    image[image < 0] = 0

    rng = galsim.BaseDeviate()
    gd = galsim.GaussianDeviate(rng, sigma = np.sqrt(1_000))
    gd.add_generate(image)

    fft2 = np.fft.fftshift(np.fft.fft2(image))
    ff_rem = fft2
    ff_rem[8:13, 8:13] = 0
    image_new = np.fft.ifft2(ff_rem)

    return image_new

def normalize_image(image):
    """
    Normalize image

    Parameters
    ----------
    image : numpy.ndarray
        FOB image
    """
    
    return image/sum(sum(image))

def clip_image(image):
    """
    Clip image with saturation limit

    Parameters
    ----------
    image : numpy.ndarray
        FOB image
    """

    return np.clip(image, 0, 1e5)


def recenter_image(image):
    """
    Recenter FOB image and crop to 16 pxels size. 

    Parameters
    ----------
    image : numpy.ndarray
        FOB image

    Returns
    -------
    image_new : numpy.ndarray
        FOB image with center at (8, 8)
    """
    cmy, cmx = np.where(image == np.max(image))
    cy = cmy[0]
    cx = cmx[0]

    image_new = image[int(np.round(cy)) - 8 : int(np.round(cy)) + 8 + 1, int(np.round(cx)) - 8 : int(np.round(cx)) + 8 + 1]

    return image_new

def process_dataset(duration_label, num_phot, num_simulations):
    for it in tqdm(range(num_simulations)):
        image = np.loadtxt(f'data/FOBs/FOB_{duration_label}_{num_phot}_{it}.txt', delimiter = ',')

        image = add_noise(image, 1000)
        image = recenter_image(image)
        image = clip_image(image)
        image_array = normalize_image(image)
    
        plt.imsave(f'data/processed/FOB_{duration_label}_{num_phot}_{it}.png', image_array)
        np.savetxt(f'data/processed/FOB_{duration_label}_{num_phot}_{it}.txt', image_array)

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--duration_label',
                        metavar='duration_label',
                        type=str)

    parser.add_argument('--num_phot',
                        metavar='num_phot',
                        type=int)  

    parser.add_argument('--num_simulations',
                        metavar='num_simulations',
                        type=int)                                                                    

    # Execute the parse_args() method
    args = parser.parse_args()

    process_dataset(
        args.duration_label,
        args.num_phot,
        args.num_simulations,
    )