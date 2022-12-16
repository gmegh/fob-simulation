import trace
import numpy as np
import argparse
from tqdm.notebook import tqdm
import galsim
import pickle
import matplotlib.pyplot as plt
import multiprocessing as mp


def update(*a):
    pbar.update()
    pbar.refresh()
    
class Arguments:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
      
def generate_dataset(it, args):
    print('in here')
    np.random.seed(int(it*1000*np.random.random()))
    rng = galsim.BaseDeviate()
    traced_photons, wavelengths = trace.tracePhotons(args.num_phot, rng, args.observation, args.wavelength, args.atmosphere, args.duration)
    print('hehe')
    PIXEL_SIZE = 10e-6
    STAMP_SIZE = 10
    SKY_LEVEL = 1000.0
    image_array = trace.generateImage(traced_photons, wavelengths, STAMP_SIZE, PIXEL_SIZE, SKY_LEVEL)

    plt.imsave(f'{args.folder}/FOB_{args.duration_label}_{args.num_phot}_{it + args.offset}.png', image_array)
    np.save(f'{args.folder}/FOB_{args.duration_label}_{args.num_phot}_{it + args.offset}.npy', image_array)
    np.savetxt(f'{args.folder}/FOB_{args.duration_label}_{args.num_phot}_{it + args.offset}.txt', image_array, delimiter=',')  
   
def generate_dataset_parallel(argums, pbar):
    pool = mp.Pool(2)
    
    pbar.reset(total=argums.num_simulations) 
    for it in range(argums.num_simulations):
        print('there')
        pool.apply_async(
            generate_dataset, 
            args = (it, argums), 
            callback = update
        )

    pool.close()
    pool.join()
    
    # Close progressbar
    pbar.close()


def main(duration, duration_label, num_phot, num_simulations, atm = 'full', offset = 0):

    # Define wavelength dictionary
    wavelength_dict = dict(
        u = 365.49,
        g = 480.03,
        r = 622.20,
        i = 754.06,
        z = 868.21,
        y = 991.66
    )

    # Define the observation parameters.
    observation = {
        'boresight': galsim.CelestialCoord(
            30*galsim.degrees, 10*galsim.degrees
        ),
        'zenith': 30*galsim.degrees,
        'airmass': 1.1547,
        'rotTelPos': 0.0*galsim.degrees,  # zenith measured CCW from up
        'rotSkyPos': 0.0*galsim.degrees,  # N measured CCW from up
        'rawSeeing': 0.7*galsim.arcsec,
        'band': 'i',
        'exptime': 15.0,
        'frbtime': 0.005,
        'temperature': 293.15,  # K
        'pressure': 69.328,  # kPa
        'H2O_pressure': 1.067,  # kPa
    }

    # Set the wavelenght from the observation band. 
    wavelength = wavelength_dict[observation['band']]

    if (atm == 'truncated'):
        # Load atmosphere phase screen list created from Elleboerk model.
        with open("policy/truncated_atm.pkl", 'rb') as f:
            atmosphere = pickle.load(f)
    else:
        # Load atmosphere phase screen list created from Elleboerk model.
        with open("policy/full_atm.pkl", 'rb') as f:
            atmosphere = pickle.load(f)
        
    pbar = tqdm(total = num_simulations)

    # Generate dataset
    generate_dataset_parallel(
        Arguments(
            atmosphere = atmosphere,
            duration = duration,
            duration_label = duration_label,
            num_simulations = num_simulations,
            observation = observation, 
            wavelength = wavelength,
            num_phot = int(num_phot),
            offset = offset,
            folder = 'data'
        ),
        pbar
    )

if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument('--atm',
                        metavar='atm',
                        type=str)

    parser.add_argument('--duration',
                        metavar='duration',
                        type=float)  

    parser.add_argument('--duration_label',
                        metavar='duration_label',
                        type=str)

    parser.add_argument('--num_phot',
                        metavar='num_phot',
                        type=int)           
    parser.add_argument('--num_simulations',
                        metavar='num_simulations',
                        type=int) 
    parser.add_argument('--offset',
                        metavar='offset',
                        type=int)                                                                      

    # Execute the parse_args() method
    args = parser.parse_args()

    main(
        args.duration,
        args.duration_label,
        args.num_phot,
        args.num_simulations,
        args.atm,
        args.offset,
    )
