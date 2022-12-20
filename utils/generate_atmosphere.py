import galsim
import multiprocessing as mp
import numpy as np
import pickle
from scipy.optimize import bisect


def main(folder = '/home/guillemmh/fob-simulation/policy/'):
    """
    Generate an atmosphere model truncated to low modes and a full model.

    """

    # Some settings here from Vera C. Rubin Observatory.
    wavelength_dict = dict(
        u=365.49,
        g=480.03,
        r=622.20,
        i=754.06,
        z=868.21,
        y=991.66
    )

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
        'exptime': 30.0,
        'temperature': 293.15,  # K
        'pressure': 69.328,  # kPa
        'H2O_pressure': 1.067,  # kPa
    }

    # Some atmospheric settings from Vera C. Rubin Observatory.
    atmSettings = {
        'kcrit': 0.2,
        'screen_size': 819.2,
        'screen_scale': 0.1,
        'nproc': 6,
    }

    # Get the wavelength.
    wavelength = wavelength_dict[observation['band']]

    # Set the random number generator.
    rng = galsim.BaseDeviate()

    # Generate the atmosphere model.
    atmosphere = generate_atmosphere(
        atmSettings, 
        wavelength, 
        observation, 
        rng
    )

    # Generate the truncated atmosphere model.
    truncated_atm = get_atm(
        atmosphere,
        wavelength,
        kcrit = atmSettings['kcrit'],
        nproc = atmSettings['nproc'],
        truncate=False,
    )

    # Save the truncated atmosphere model.
    with open(folder + "truncated_atm.pkl", 'wb') as f:
        with galsim.utilities.pickle_shared():
            pickle.dump(truncated_atm, f)
        
    # Generate the full atmosphere model.
    full_atm = get_atm(
        atmosphere,
        wavelength,
        kcrit = atmSettings['kcrit'],
        nproc = atmSettings['nproc'],
        truncate=False
    )

    # Save the full atmosphere model.
    with open(folder + "full_atm.pkl", 'wb') as f:
        with galsim.utilities.pickle_shared():
            pickle.dump(full_atm, f)

    return full_atm, truncated_atm

def generate_atmosphere(atmSettings, wavelength, observation, rng):
    """
    Generate atmosphere model.

    Parameters
    ----------
    atmSettings : dict
        Atmosphere settings
    wavelength : float
        Wavelength in nm
    observation : dict  
        Observation settings
    rng : galsim.BaseDeviate
        Random number generator

    Returns
    -------
    atmosphere : galsim.Atmosphere
        Atmosphere model
    """
    targetFWHM = (
        observation['rawSeeing']/galsim.arcsec *
        observation['airmass']**0.6 *
        (wavelength/500.0)**(-0.3)
    )

    ud = galsim.UniformDeviate(rng)
    gd = galsim.GaussianDeviate(rng)

    # Use values measured from Ellerbroek 2008.
    altitudes = [0.0, 2.58, 5.16, 7.73, 12.89, 15.46]
    # Elevate the ground layer though.  Otherwise, PSFs come out too correlated
    # across the field of view.
    altitudes[0] = 0.2

    # Use weights from Ellerbroek too, but add some random perturbations.
    weights = [0.652, 0.172, 0.055, 0.025, 0.074, 0.022]
    weights = [np.abs(w*(1.0 + 0.1*gd())) for w in weights]
    weights = np.clip(weights, 0.01, 0.8)  # keep weights from straying too far.
    weights /= np.sum(weights)  # renormalize

    # Draw outer scale from truncated log normal
    L0 = 0
    while L0 < 10.0 or L0 > 100:
        L0 = np.exp(gd() * 0.6 + np.log(25.0))

    # Given the desired targetFWHM and randomly selected L0, determine
    # appropriate r0_500
    r0_500 = _r0_500(wavelength, L0, targetFWHM)

    # Broadcast common outer scale across all layers
    L0 = [L0]*6

    # Uniformly draw layer speeds between 0 and max_speed.
    maxSpeed = 20.0
    speeds = [ud()*maxSpeed for _ in range(6)]

    # Isotropically draw directions.
    directions = [ud()*360.0*galsim.degrees for _ in range(6)]

    atmosphere = galsim.Atmosphere(
        r0_500 = r0_500, 
        L0 = 25.0, 
        speed = speeds,
        direction = directions, 
        altitude = altitudes, 
        r0_weights = weights,
        rng = rng,
        screen_size=atmSettings['screen_size'],
        screen_scale=atmSettings['screen_scale']
    )
    
    return atmosphere

def get_atm(atm, wavelength, kcrit = 0.2, nproc = 6, truncate = False):
    """
    Generate instantiated atmosphere model.

    Parameters
    ----------
    atm: Object
        Atmosphere model
    wavelength : float
        Wavelength
    kcrit : float
        Critical k value
    nproc : int
        Number of processes
    truncate : bool
        Whether to truncate the atmosphere

    Returns
    -------
    atm: Object
        Instantiated atmosphere model
    """
    
    # Use fork with multiprocessing.
    ctx = mp.get_context('fork')

    # Determine the kmax to use for the atmosphere.
    r0_500 = atm.r0_500_effective
    r0 = r0_500 * (wavelength/500)**(6./5)
    if truncate:
        kmax = kcrit / r0
    else:
        kmax = np.inf

    # Instantiate the atmosphere
    with ctx.Pool(
        nproc, 
        initializer=galsim.phase_screens.initWorker,
        initargs=galsim.phase_screens.initWorkerArgs()
    ) as pool:
        atm.instantiate(pool=pool, kmax=kmax)
        
    return atm

def _r0_500(wavelength, L0, targetSeeing):
    """
    Given a desired FWHM, outer scale, and wavelength, determine the appropriate
    Fried parameter r0_500.

    Parameters
    ----------
    wavelength : float
        Wavelength
    L0 : float
        Outer scale
    targetSeeing : float
        Desired Seeing

    Returns
    -------
    r0_500 : float
        Returns r0_500 to use to get target seeing
    """
    r0_500_max = min(1.0, L0*(1./2.183)**(-0.356)*(wavelength/500.)**1.2)
    r0_500_min = 0.01

    return bisect(
        _seeingResid,
        r0_500_min,
        r0_500_max,
        args=(wavelength, L0, targetSeeing)
    )

def _vkSeeing(r0_500, wavelength, L0):
    """
    Generate von Karman calculated seeing.

    Parameters
    ----------
    r0_500 : float
        Fried parameter
    wavelength : float
        Wavelength
    L0 : float
        Outer scale

    Returns
    -------
    residue : float
        Returns von Karman calculated seeing.
    """
    # von Karman profile FWHM from Tokovinin fitting formula (Kolmogorov)
    kolm_seeing = galsim.Kolmogorov(r0_500=r0_500, lam=wavelength).fwhm
    r0 = r0_500 * (wavelength/500)**1.2
    arg = 1. - 2.183*(r0/L0)**0.356
    factor = np.sqrt(arg) if arg > 0.0 else 0.0

    return kolm_seeing*factor


def _seeingResid(r0_500, wavelength, L0, targetSeeing):
    """
    Generate seeing residue from targetseeing and von Karman calculated seeing.

    Parameters
    ----------
    r0_500 : float
        Fried parameter
    wavelength : float
        Wavelength
    L0 : float
        Outer scale
    targetSeeing : float
        Desired Seeing

    Returns
    -------
    residue : float
        Returns residue from targetseeing and von Karman calculated.
    """
    return _vkSeeing(r0_500, wavelength, L0) - targetSeeing


if __name__ == "__main__":
    main()