import sys
import numpy as np
from astropy.io import fits

def convert_fits(input_file, conversion_factor, output_file):
    try:
        # FITSファイルを読み込み
        hdul = fits.open(input_file)
        
        # データを変換係数で割り算して整数に変換
        data = hdul[0].data
        converted_data = (data / conversion_factor + 0.5).astype(int)
        
        # 新しいFITSファイルを作成してデータを書き込み
        new_hdul = fits.HDUList([fits.PrimaryHDU(converted_data)])
        new_hdul.writeto(output_file, overwrite=True)
        
        print(f"変換が完了し、{output_file} に保存されました。")

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    mags = np.arange(10.0, 15.0, 0.5)
    bits = [10,11,12,13,14,15,16]
    inputdir = "02_results/"
    outputdir = "03_converted/"
    for mag in mags:
        for iplate in range(0,11):
            inputfile  = inputdir+"image.{:4.1f}.{:02d}.fits".format(mag, iplate)
            for bit in bits:
                convf = 1.53/2**bit/7.28e-6
                outputfile = outputdir+"image.{:4.1f}.{:02d}.{:2d}bit.fits".format(mag, iplate, bit)
                convert_fits(inputfile, convf, outputfile)
#    if len(sys.argv) != 4:
#        print("Usage: python script.py <input_file.fits> <conversion_factor> <output_file.fits>")
#    else:
#        input_file = sys.argv[1]
#        conversion_factor = float(sys.argv[2])
#        output_file = sys.argv[3]
#        convert_fits(input_file, conversion_factor, output_file)
