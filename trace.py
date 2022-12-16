import batoid
import galsim
import matplotlib.pyplot as plt
import numpy as np

def tracePhotons(num_phot, rng, observation, wavelength, atm, duration):
    '''
    Generate and trace photons through atmosphere and telescope.

    Parameters
    ----------
    num_phot : int
        Number of photons to trace.
    rng : galsim.UniformDeviate
        Random number generator.
    observation : dict
        Dictionary containing observation metadata.
    wavelength : float
        Wavelength of telescope filter in nm.
    atm : np.array
        Atmospheric model.
    duration : float
        Exposure time in seconds.

    Returns
    -------
    traced_photons : np.array
        Traced photons.
    wavelengths : np.array
        Wavelengths of photons.
    '''
    # Define FRB as a delta function profile.
    frb_delta = galsim.DeltaFunction(flux = 2e5)

    # Generate atmospheric PSF for a 15s expsoure. Includes first_kick and second_kick.
    frb_atm_psf = atm.makePSF(
                wavelength, # wavelength in nm.
                t0 = np.random.random()*100, # initial time in seconds.
                diam = 8.36, # diameter of telescope in meters.
                exptime = duration, # exposure time in seconds.
                flux = 1.0, 
                geometric_shooting = False,
                second_kick = False,
            )

    # Convolve with point-source psf
    frb_total_psf = galsim.Convolve([frb_delta, frb_atm_psf])

    frb_photons = frb_total_psf.shoot(num_phot, rng)
    
    traced_photons, wavelengths = atmosphericTrace(frb_photons, rng, observation, wavelength, atm, first_kick = False)
    print('hello3')

    return traced_photons, wavelengths


def projection(observation):
    '''
    Generate a projection from celestial coordinates to field angle.

    Parameters
    ----------
    observation : dict\
        Dictionary containing observation metadata.

    Returns
    -------
    fieldAngle : np.array
        Field angles.
    '''
    # Develop gnomonic projection from ra/dec to field angle using
    # GalSim TanWCS class.
    q = observation['rotTelPos'] - observation['rotSkyPos']
    cq, sq = np.cos(q), np.sin(q)
    affine = galsim.AffineTransform(cq, -sq, sq, cq)

    # Generate field angles
    radecToField = galsim.TanWCS(
                        affine,
                        observation['boresight'],
                        units = galsim.radians
                    )

    # Generate celestial coordinates
    dist = 100*galsim.degrees
    while (dist > 1.75*galsim.degrees):   # whole focal plane
        ra = np.random.uniform(26.0, 34.0)
        dec = np.random.uniform(7.0, 13.0)
        coord = galsim.CelestialCoord(
            ra*galsim.degrees, dec*galsim.degrees
        )
        dist = observation['boresight'].distanceTo(coord)

    return radecToField.toImage(coord)

def populatePupil(num_photons, rng):
    """
    Assign pupil position to traced photons.
    
    Parameters
    ----------
    num_photons : int
        Number of photons to populate pupil with.
    rng : galsim.UniformDeviate
        Random number generator.

    Returns
    -------
    u, v : np.ndarray
        Pupil positions of photons.
    """

    # Populate pupil
    r_outer = 8.36/2

    # purposely underestimate inner radius a bit.
    # Rays that miss will be marked vignetted.
    r_inner = 8.36/2*0.58
    ud = galsim.UniformDeviate(rng)
    r = np.empty(num_photons)
    ud.generate(r)
    r *= (r_outer**2 - r_inner**2)
    r += r_inner**2
    r = np.sqrt(r)

    # Sample angle around the pupil.
    th = np.empty(num_photons)
    ud.generate(th)
    th *= 2*np.pi

    # Assign pupil position given radius and theta.
    u = r*np.cos(th)
    v = r*np.sin(th)

    return u,v

def BBSED(T):
    """
    (unnormalized) Blackbody SED for temperature T in Kelvin. 
    Generates SED of a star.

    Parameters
    ----------
    T : float
        Temperature of star in Kelvin.

    Returns
    -------
    galsim.SED
        SED of star.
    """

    # Define wavelength vector
    waves_nm = np.arange(330.0, 1120.0, 10.0)

    # Compute Blackbody flux given temperature and wavelength.
    def planck(t, w):
        # T := temperature in K
        # w := wavelength in m
        c = 2.99792458e8  # speed of light in m/s
        kB = 1.3806488e-23  # Boltzmann's constant J per Kelvin
        h = 6.62607015e-34  # Planck's constant in J s
        return w**(-5) / (np.exp(h*c/(w*kB*t))-1)

    flambda = planck(T, waves_nm*1e-9)
    
    return galsim.SED(
        galsim.LookupTable(waves_nm, flambda),
        wave_type='nm',
        flux_type='flambda'
    )

def generateWavelenghts(num_phot, observation, rng):
    """
    Generate wavelenghts for all photons.

    Parameters
    ----------
    num_phot : int
        Number of photons to generate wavelenghts for.
    observation : dict
        Dictionary containing observation information.
    rng : galsim.UniformDeviate
        Random number generator.

    Returns
    -------
    wavelengths : np.ndarray
        Wavelenghts of all photons.
    """
    # Define bandpass
    bandpass = galsim.Bandpass('policy/LSST_{}.dat'.format(observation['band']), wave_type='nm')

    # Initialize the used SED and Temperature of the source.
    T = np.random.uniform(4000, 10000)
    sed = BBSED(T)

    # Assign wavelengths to all photons sampling the SED distribution.
    wavelengths = sed.sampleWavelength(num_phot, bandpass, rng)

    # Generate vector with all photons wavelenghts
    return wavelengths

def generatePhaseGradients(u, v, photons, wavelengths, observation, fieldAngle, atm, rng, wavelength, first_kick):   
    """
    Generate phase gradients for all photons.

    Parameters
    ----------
    u, v : np.ndarray
        Pupil positions of photons.
    photons : np.ndarray
        Traced photons.
    wavelengths : np.ndarray
        Wavelenghts of all photons.
    observation : dict
        Dictionary containing observation information.
    fieldAngle : galsim.CelestialCoord
        Field angle of photons.
    atm : galsim.Atmosphere
        Atmosphere object.
    rng : galsim.UniformDeviate
        Random number generator.
    wavelength : float
        Wavelength of filter in telescope
    first_kick : bool
        If True, apply atmospheric first kick.

    Returns
    -------
    dku, dkv : np.ndarray
        Phase gradients of all photons.
    """ 
    dku = np.zeros(len(wavelengths))
    dkv = np.zeros(len(wavelengths))

    if first_kick == True:
        # Generate time vector for all photons.
        t = np.zeros(len(wavelengths))

        # Uniformly distribute galaxy photon times throughout 15s exposure.
        ud = galsim.UniformDeviate(rng)
        ud.generate(t)
        t += observation['exptime']
                
        dku, dkv = atm.wavefront_gradient(
            u, v, t, (fieldAngle.x*galsim.radians, fieldAngle.y*galsim.radians)
        )
        
        # Since the output is in nm per m, convert to radians.
        dku *= 1e-9 
        dkv *= 1e-9

    # Add photons position from atmosphere to the phase gradients.
    # This is the convolution, so should just be for the galaxy also. 
    # this is also for frb, but would be without the +.
    dku += photons.x*(galsim.arcsec/galsim.radians)
    dkv += photons.y*(galsim.arcsec/galsim.radians)

    dku *= (wavelengths/500)**(-0.3)
    dkv *= (wavelengths/500)**(-0.3)

    # DCR.  dkv is aligned along meridian, so only need to shift in this
    # direction
    # Compute refraction due to observation wavelength.
    # This is also applies to FRB
    
    base_refraction = galsim.dcr.get_refraction(
        wavelength,
        observation['zenith'],
        temperature = observation['temperature'],
        pressure = observation['pressure'],
        H2O_pressure = observation['H2O_pressure'],
    )

    # Compute refraction due to wavelenght for each photon.
    refraction = galsim.dcr.get_refraction(
        wavelengths,
        observation['zenith'],
        temperature = observation['temperature'],
        pressure = observation['pressure'],
        H2O_pressure = observation['H2O_pressure'],
    )

    # Add the refraction difference to the phase gradient.
    refraction -= base_refraction
    dkv += refraction

    # Add the tangent plane coordinates to the refraction difference.
    # This also applies to frb
    dku += fieldAngle.x
    dkv += fieldAngle.y
    
    return dku, dkv

def GalsimToBatoid(u, v, dku, dkv, telescope, wavelengths, num_phot):
    """
    Generate batoid rays from Galsim photons.

    Parameters
    ----------
    u, v : np.ndarray
        Pupil positions of photons.
    dku, dkv : np.ndarray
        Phase gradients of all photons.
    telescope : dict
        Dictionary containing telescope information.
    wavelengths : np.ndarray
        Wavelenghts of all photons.
    num_phot : int
        Number of photons.

    Returns 
    -------
    rays : batoid.RayVector
        Batoid rays.
    """
    # Generate rays velocity components from atmospheric phase gradient computed in the previous section.
    vx, vy, vz = batoid.utils.fieldToDirCos(dku, dkv, projection='gnomonic')

    # Place rays on entrance pupil - the planar cap coincident with the rim
    # of M1.  Eventually may want to back rays up further so that they can
    # be obstructed by struts, e.g.
    x = u
    y = v
    zPupil = telescope["M1"].surface.sag(0, 0.5*telescope.pupilSize)
    z = np.zeros_like(x) + zPupil

    # Rescale velocities so that they're consistent with the current
    # refractive index.
    n = []
    for idx in range(len(wavelengths)):
        n.append(telescope.inMedium.getN(wavelengths[idx]))

    vx /= n
    vy /= n
    vz /= n

    # Adapt wavelength units.
    wavelengths_arr = wavelengths*1e-9
    
    rays = batoid.RayVector(
        x.tolist(), y.tolist(), z.tolist(),
        vx.tolist(), vy.tolist(), vz.tolist(),
        np.zeros(num_phot),
        wavelengths_arr.tolist(),
        np.zeros(num_phot) + 1.0
    )
    
    return rays

def TelescopeTrace(rays, telescope):
    """
    Trace rays through telescope.

    Parameters
    ----------
    rays : batoid.RayVector
        Batoid rays.
    telescope : dict
        Dictionary containing telescope information.

    Returns
    -------
    rays_traced : batoid.RayVector
        Traced batoid rays.
    """
    # Trace rays trhough telescope
    rays_traced = telescope.trace(rays)
    
    #print('Number of rays that reached the detector: {}'.format(sum(i for i in ~rays_traced.vignetted)))
    
    # Initialize Silicon detector
    silicon = batoid.TableMedium.fromTxt("silicon_dispersion.txt")

    # Refract the output beam into the silicon detector.
    telescope['Detector'].surface.refract(
        rays_traced,
        telescope['Detector'].inMedium,
        silicon
    )
    
    return rays_traced

def atmosphericTrace(photons, rng, observation, wavelength, atm, first_kick = False):
    '''
    Trace photons through atmosphere.
    
    Parameters
    ----------
    photons : galsim.GSObject
        Photons to trace.
    rng : np.random.RandomState
        Random number generator.
    observation : dict
        Dictionary containing observation information.
    wavelength : float
        Wavelength of filter.
    atm : dict
        Dictionary containing atmosphere information.
    first_kick : bool
        If True, trace photons using first kick in atmospheric modelling.

    Returns
    -------
    rays_traced : batoid.RayVector
        Traced batoid rays.
    wavelengths : np.ndarray
        Wavelengths of photons.
    '''
    # Initialize telescope 
    telescope = batoid.Optic.fromYaml(f"policy/LSST_{observation['band']}.yml")

    fieldAngle = projection(observation)
    phot_num_tot = photons.x.shape[0]
    
    u, v = populatePupil(phot_num_tot, rng)
    
    #t0 = time.time()
    wavelengths = generateWavelenghts(phot_num_tot, observation, rng)
    #print('Wavelengths generated, took {} seconds.'.format(time.time() - t0))
    
    #t0 = time.time()
    dku, dkv = generatePhaseGradients(
        u, 
        v, 
        photons, 
        wavelengths, 
        observation, 
        fieldAngle, 
        atm,
        rng, 
        wavelength, 
        first_kick
    )
    #print('Phase gradients generated, took {} seconds.'.format(time.time() - t0))
    
    #t0 = time.time()
    rays = GalsimToBatoid(u, v, dku, dkv, telescope, wavelengths, phot_num_tot)
    #print('Rays transferred to Batoid, took {} seconds.'.format(time.time() - t0))
    
    #t0 = time.time()
    traced_rays = TelescopeTrace(rays, telescope)
    #print('Rays traced through telescope, took {} seconds.'.format(time.time() - t0))
    
    return traced_rays, wavelengths

def generateImage(traced_photons, wavelengths, STAMP_SIZE, PIXEL_SIZE, noise):
    """
    Generate sensor image from traced photons.

    Parameters
    ----------
    traced_photons : batoid.RayVector
        Traced batoid rays.
    wavelengths : np.ndarray
        Wavelengths of photons.
    STAMP_SIZE : int
        Size of image.
    PIXEL_SIZE : float
        Size of pixel.
    noise : bool
        If True, add noise to image.

    Returns
    -------
    image : np.ndarray
        Sensor image.
    """
    print('hello 2')
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
    treering_func = galsim.SiliconSensor.simple_treerings(r_max = 1e7)
    # Include tree ring and brighter fatter effect
    sensor = galsim.SiliconSensor(treering_func=treering_func, treering_center=galsim.PositionD(0.0,0.0))
    sensor.accumulate(photons, image)

    # Add Gaussian background.
    if noise == True:
        rng = galsim.BaseDeviate()
        gd = galsim.GaussianDeviate(rng, sigma = np.sqrt(100_000))
        gd.add_generate(image.array)
    
    return image.array

def simulateFOB(num_phot, rng, observation, wavelength, atm, duration, STAMP_SIZE, PIXEL_SIZE, noise = False):
    traced_photons, wavelengths = tracePhotons(num_phot, rng, observation, wavelength, atm, duration)

    print('nanos')

    image_array = generateImage(traced_photons, wavelengths, STAMP_SIZE, PIXEL_SIZE, noise)

    return image_array
