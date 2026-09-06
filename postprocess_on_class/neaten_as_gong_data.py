import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from astropy.io import fits
from scipy.stats import skew, kurtosis

from smp.config import load_config

config = load_config()

for cr in range(2239,2240):
    # raw GONG fits: flipud to get synoptic maps
    raw_file = config.repository_root / "determine_order" / f"mrzqs_c{cr}.fits"
    # raw_file = path + 'raw/' + 'mrzqs_c2277.fits'
    # raw_file = path + 'neaten/format_as_gong/' + 'mrzql231120t0104c2277_000.fits'
    raw_fits = fits.open(raw_file)

    raw_Br = raw_fits[0].data
    raw_header = raw_fits[0].header
    # print(raw_header)

    plt.figure()
    plt.imshow(raw_Br,cmap='RdBu',vmin=-4,vmax=4)
    plt.colorbar()
    
    # plt.show()
    plt.close()

    # predicted WSO mat: flipud to get synoptic maps
    pred_file = config.repository_root / "determine_order" / f"{cr}_WSO_9_interp.mat"
    pred_mat = scio.loadmat(pred_file)
    pred_Br = pred_mat['pred_Br_interp']

    plt.figure()
    plt.imshow(pred_Br,cmap='RdBu',vmin=-4,vmax=4)
    plt.colorbar()

    # replace original GONG data with predicted data
    neaten_fits = raw_fits
    neaten_fits[0].data = pred_Br

    # plt.show()
    plt.close()
    
    # replace original header with that based on predicted data
    pred_Br_1d = pred_Br.flatten()
    nan_indices = np.isnan(pred_Br_1d)
    pred_Br_1d = pred_Br_1d[~nan_indices]
    # data related
    neaten_fits[0].header['IMGMN01'] = np.mean(pred_Br_1d) # mean
    neaten_fits[0].header['IMGRMS01'] = np.std(pred_Br_1d) # RMS
    neaten_fits[0].header['IMGSKW01'] = skew(pred_Br_1d) # skewness
    neaten_fits[0].header['IMGMIN01'] = np.min(pred_Br_1d) # Min
    neaten_fits[0].header['IMGMAX01'] = np.max(pred_Br_1d) # Max
    neaten_fits[0].header['IMGADV01'] = np.mean(np.abs(pred_Br_1d-np.mean(pred_Br_1d))) # Average Deviation
    neaten_fits[0].header['IMGVAR01'] = np.var(pred_Br_1d) # Variance
    neaten_fits[0].header['IMGKUR01'] = kurtosis(pred_Br_1d) # Kurtosis
    
    # # time
    # neaten_fits[0].header['DATE'] = '2024-05-29T23:16:00'
    # neaten_fits[0].header['DATE-OBS'] = '2024-05-29'
    # neaten_fits[0].header['TIME-OBS'] = '23:16   '
    # # Carrington Rotation
    # neaten_fits[0].header['CAR_ROT'] = cr
    # neaten_fits[0].header['CAVAL1A'] = neaten_fits[0].header['CAR_ROT'] + 0.5
    # neaten_fits[0].header['CARROT'] = neaten_fits[0].header['CAR_ROT']
    # neaten_fits[0].header['CREDGE'] = neaten_fits[0].header['CAR_ROT'] + 1
    # neaten_fits[0].header['CRNOW'] = neaten_fits[0].header['CAR_ROT'] + 0.8333333
    # neaten_fits[0].header['CR60'] = neaten_fits[0].header['CRNOW']
    # neaten_fits[0].header['CRCENTER'] = neaten_fits[0].header['CAR_ROT'] + 0.5
    # # filename related
    # neaten_fits[0].header['MAPNAME'] = '2284_000'
    # neaten_fits[0].header['MAPDATE'] = neaten_fits[0].header['DATE-OBS']
    # neaten_fits[0].header['MAPTIME'] = neaten_fits[0].header['TIME-OBS']
    # neaten_fits[0].header['FILELIST'] = 'mrzql240529t2316c2284_000'
    
    # save .fits
    save_file = config.repository_root / "determine_order" / f"{cr}_WSO_9.fits"
    # save_file = path + 'neaten/' + 'cr' + str(cr) + '_neaten.fits'
    # save_file = path + 'neaten/' + neaten_fits[0].header['FILELIST'] + '.fits'
    # neaten_fits.writeto(save_file,overwrite=True)
