#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-
"""Make an image.

.. code-block:: bash

  usage:
    mkimage.py [-h|--help] [--pd paramdir] --starplate star_plate.csv [--var variability.json] --det det.json --tel tel.json --ace ace.json --ctl ctl.json [--dft drift.json] --format format [--od outdir] [--overwrite]

  options:
   -h --help                   show this help message and exit.
   --pd paramdir               name of the directory containing parameter files.
   --starplate star_plate.csv  csv file containing star info (plate index, star index, x pixel, y pixel, lambda, beta, Hwmag)
   --var variability.json      json file for stellar variability/transit (optional). The input variability will be shown in variability_input().png
   --det det.json              json file containing detector related parameters.
   --tel tel.json              json file containing telescope related parameters.
   --ace ace.json              json file containing ace parameters.
   --ctl ctl.json              json file containing control parameters.
   --dft drift.json            json file containing drift parameterers.
   --format format             format of the output file (platefits, fitscube, hdfcube).
   --od outdir                 name of the directory to put the outputs.
   --overwrite                 if set, overwrite option activated.


    Note:
        In the process to make pixar, Nts_per_plate is multiplied to the result of simpix to make the units of pixar to be e/pix/control_params.ace_control['dtace']. But, in none/gauss mode, the scaling is not correct for simulating a single-shot image. Therefore, we divide pixar by Nts_per_plate for correction.

    Note:
        Curretly, the time resolution should be prepared in the unit of control_params.tplate + detector.readparams.t_scan. We do not support the finest time resolution yet (control_params.ace_control['dtace']).
"""

from docopt import docopt
import tqdm
import numpy as np
from jis.photonsim.extract_json import Variability
from jis.binutils.setfiles import set_filenames_from_args, set_filenames_output
from jis.binutils.setcontrol import load_parameters
from jis.binutils.save import save_outputs
from jis.binutils.runphotonsim import run_calc_wfe, run_calc_psf, run_calc_ace, apply_gaussian
from jis.binutils.runpixsim import init_pix, uniform_flat, init_images, set_positions, make_local_flat, index_control_trajectory, calc_theta, normalize_pixar, add_varability, add_dark_current
from jis.binutils.scales import get_pixelscales, get_tday
from jis.binutils.check import check_ace_length
from jis.binutils.binplot import plot_variability
from jis.pixsim.integrate import integrate
import matplotlib.pylab as plt
from jis.pixsim import simpix_stable as sp
import astropy.io.fits as pf

if __name__ == '__main__':
    """
    Notes:
        When performing the PSF integration (simpix),
        Output= array of images in each time bin in the exposure.
        When the PSF is given in e/fp-cell/sec,
        simpix/(psfscale*psfscale) is in e/pix/(1./Nts_per_plate sec).

    """
    args = docopt(__doc__)
    filenames, dirname_output = set_filenames_from_args(args)
    table_starplate, detector, control_params, telescope, ace_params = load_parameters(
        filenames)
    filenames, output_format, overwrite = set_filenames_output(
        args, filenames, control_params, dirname_output)

    # Selecting the data for the first plate. ######################
    pos = np.where(table_starplate['plate index'] == 0)
    table_starplate = table_starplate[pos]

    wfe = run_calc_wfe(control_params, telescope)
    psf = run_calc_psf(control_params, telescope, detector, wfe)
    acex, acey, Nts_per_plate = run_calc_ace(control_params, detector, ace_params)
    detpix_scale, fp_cellsize_rad, fp_scale, psfscale = get_pixelscales(
        control_params, telescope, detector)
    theta_full, pixdim, Npixcube = init_pix(
        control_params, detector, acex, acey, detpix_scale, args['--dft'])
    check_ace_length(Nts_per_plate, control_params, theta_full)
    if args['--var']:
        variability = Variability.from_json(filenames['varjson'])
        tday = get_tday(control_params, detector)
        plot_variability(variability, filenames['starplate'], tday)

    if control_params.effect.ace == 'gauss':
        psf = apply_gaussian(psf, control_params.ace_control['acex_std'],\
                             control_params.ace_control['acey_std'], fp_scale)

    uniform_flat_interpix, uniform_flat_intrapix = uniform_flat(detector)
    pixcube_global = init_images(control_params, detector)

    pcg_array = np.tile(pixcube_global, (6,1,1,1))
    # Making data around each star.
    for i_star, line in enumerate(table_starplate):
        print('StarID: {}'.format(line['star index']))
        mag = line['Hwmag']
        xc_local, yc_local, x0_global, y0_global, xc_global, yc_global = set_positions(
            line, Npixcube)
        interpix_local = make_local_flat(
            control_params, detector, x0_global, y0_global, pixdim)

        # Making a cube containing plate data for a local region (small frame for a local region).
        # Initialize (Axis order: X, Y, Z)
        pixcube = np.zeros((Npixcube, Npixcube, control_params.nplate))

        # Load variability
        if args['--var']:
            varsw, injlc, b = variability.read_var(tday, line['star index'])

        # Loop to take each plate.
        for iplate in tqdm.tqdm(range(0, control_params.nplate)):
            # picking temporary trajectory and local position update
            istart, iend = index_control_trajectory(
                control_params, iplate, Nts_per_plate)
            theta = calc_theta(theta_full, istart, iend, xc_local, yc_local)

            if control_params.effect.flat_intrapix:
                flat_intrapix = detector.flat.intrapix
            else:
                flat_intrapix = uniform_flat_intrapix

            if control_params.effect.wfe != 'fringe37':
                psfin = psf
                psfcenter = (np.array(np.shape(psfin))-1.0)*0.5
            else:
                psfin = psf[i_star]
                psfcenter = (np.array(np.shape(psfin)[1:])-1.0)*0.5

            pixar_org = sp.simpix(theta, interpix_local, flat_intrapix,
                                  psfarr=psfin, psfcenter=psfcenter, psfscale=psfscale)\
                / (psfscale*psfscale)*control_params.ace_control['dtace']/(1./Nts_per_plate)

            mags = np.arange(12.0, 15.0, 0.5)
            for i_mag in range(0,6):
                pixar = normalize_pixar(control_params, pixar_org, mag, Nts_per_plate)

                if args['--var']:
                    if varsw:
                        pixar = add_varability(pixar, injlc_iplate)

                pixar = add_dark_current(control_params, detector, pixar)
                integrated = integrate(pixar, x0_global, y0_global, control_params.tplate,
                                       control_params.ace_control['dtace'], detector)
                # integrated is in adu/pix/plate.
                pixcube[:, :, iplate] = integrated

                pcg_array[i_mag, x0_global:x0_global+Npixcube, y0_global:y0_global+Npixcube, iplate]=\
                    pixcube[:,:,iplate]


    save_outputs(filenames, output_format, control_params, telescope, detector, wfe, psf, pixcube_global,
                 control_params.tplate, uniform_flat_interpix, uniform_flat_intrapix, acex, acey, overwrite)

    pcg_array = np.swapaxes(pcg_array, 1, 3)
    for i in range(0, control_params.nplate):
        for i_mag in range(6):
            pf.writeto(filenames['images'][i].replace('image', 'image.{:4.1f}.'.format(mags[i_mag])),\
                       pcg_array[i_mag, i].astype('int32'), overwrite=overwrite)

