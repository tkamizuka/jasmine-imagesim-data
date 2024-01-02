import os
import numpy as np
from astropy.io import fits

input_dir = "03_converted/"
output_dir = "04_stacked/"

# Define [mag] values
mag_values = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0, 14.5]

# Iterate over [mag] and [dd]
for mag in mag_values:
    for dd in range(10, 17):
        for num in range(11):
            # Initialize an empty variable to store the sum of images
            stacked_data = None
            
            # Iterate from 0 to num and add images
            for i in range(num + 1):
                # Construct the input filename
                input_filename = f"image.{mag:.1f}.{i:02d}.{dd}bit.fits"
                input_path = os.path.join(input_dir, input_filename)
                
                # Check if the input file exists
                if os.path.exists(input_path):
                    # Read the FITS file and add the image data to the sum
                    with fits.open(input_path) as hdul:
                        image_data = hdul[0].data.astype('float')
                        if stacked_data is None:
                            stacked_data = image_data
                        else:
                            stacked_data += image_data
            stacked_data = stacked_data/(num+1.)
            if stacked_data is not None:
                # Create the output filename with [nn] = num + 1
                output_filename = f"image.{mag:.1f}.{num+1:02d}.{dd}bit.fits"
                output_path = os.path.join(output_dir, output_filename)
                
                # Write the stacked data to the output FITS file
                hdu = fits.PrimaryHDU(stacked_data)
                hdu.writeto(output_path, overwrite=True)
                print(f"Stacked and saved: {output_filename}")
