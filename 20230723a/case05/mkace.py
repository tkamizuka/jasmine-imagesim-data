#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""Make a galactic center image.
   
    This example generates a mock JASMINE dectector image of the Galactic center using Aizawa's jscon catalog.

"""

import tqdm
import numpy as np
from jis.binutils.save import save_outputs
from jis.binutils.runphotonsim import run_calc_wfe, run_calc_psf, run_calc_ace, apply_gaussian
from jis.binutils.runpixsim import init_pix, uniform_flat, init_images, set_positions, make_local_flat
from jis.binutils.runpixsim import index_control_trajectory, calc_theta, scaling_pixar, run_simpix
from jis.binutils.runpixsim import global_dark
from jis.binutils.scales import get_pixelscales
from jis.binutils.check import check_ace_length
from jis.pixsim.integrate import integrate
from jis.pixsim.addnoise import addnoise
import astropy.io.ascii as asc
import matplotlib.pylab as plt
from jis.pixsim.wcs import set_wcs
from jis.photonsim.extract_json import Detector, ControlParams, Telescope, AcePsd
from jis.galcen.read_galcen_position import load_jscon_random_stars
from jis.galcen.read_galcen_position import random_stars_to_starplate
from jis.galcen.read_galcen_position import maximum_separation
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

if __name__ == '__main__':
    """
    Notes:
        When performing the PSF integration (simpix),
        Output= array of images in each time bin in the exposure.
        When the PSF is given in e/fp-cell/sec,
        simpix/(psfscale*psfscale) is in e/pix/(1./Nts_per_plate sec).

    """
    import os
    ver = "ver1_2_"
    region = "B"
    dirname_params = "./"
    filenames = {}
    #filenames['starplate'] = os.path.join(dirname_params, "gelcen_random_stars.json")
    filenames['starplate'] = os.path.join(dirname_params,
                                          "gelcen_random_stars.json")
    filenames['detjson'] = os.path.join(dirname_params, "det.json")
    filenames['teljson'] = os.path.join(dirname_params, "tel.json")
    filenames['acejson'] = os.path.join(dirname_params, "ace_001.json")
    filenames['ctljson'] = os.path.join(dirname_params, "ctl.json")
    filenames['acex']    = "./acex.fits"
    filenames['acey']    = "./acey.fits"

    detector = Detector.from_json(filenames['detjson'])
    control_params = ControlParams.from_json(filenames['ctljson'])
    telescope = Telescope.from_json(filenames['teljson'])
    ace_params = AcePsd.from_json(filenames['acejson']).parameters

    # Running calculations. ########################################
    acex, acey, Nts_per_plate = run_calc_ace(control_params, detector,
                                             ace_params)

    # Save x-ACE data. #############################################
    hdu = fits.PrimaryHDU(acex)
    hdu.header['ACE-FILE'] = filenames['acejson']
    hdu.header['ACE-TOTT'] = control_params.ace_control['tace']
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filenames['acex'], overwrite=True)

    # Save y-ACE data. #############################################
    hdu = fits.PrimaryHDU(acey)
    hdu.header['ACE-FILE'] = filenames['acejson']
    hdu.header['ACE-TOTT'] = control_params.ace_control['tace']
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(filenames['acey'], overwrite=True)
