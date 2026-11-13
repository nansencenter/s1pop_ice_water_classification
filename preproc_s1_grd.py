from s1denoise import Sentinel1Image
import os
import glob
import gc

from multiprocessing import Pool

from tqdm import tqdm
from nansat import Nansat
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter


force = False

idir = '/Volumes/sat_downloads_sentinel1/2026/s1pop'
ifiles = glob.iglob(f'{idir}/S1?_EW_GRDM_1SDH_*SAFE')
for ifile in ifiles:
    print(ifile)
    break

tif_out_dir = 's1_grd_preproc_tif'
png_out_dir = 's1_grd_preproc_png'
inc_out_dir = 's1_grd_preproc_inc'
os.makedirs(tif_out_dir, exist_ok=True)
os.makedirs(png_out_dir, exist_ok=True)
os.makedirs(inc_out_dir, exist_ok=True)

def get_outut_filename(s1file, out_dir, out_ext):
    basename = os.path.basename(s1file)
    platform = basename.split('_')[0]
    year = basename.split('_')[4][:4]
    month = basename.split('_')[4][4:6]
    odir = f'{out_dir}/{platform}_{year}_{month}'
    os.makedirs(odir, exist_ok=True)
    return f'{odir}/{os.path.basename(s1file)}.{out_ext}'


def preproc_s1_grd(s1file):
    tif_out_file = get_outut_filename(s1file, tif_out_dir, 'tif')
    if os.path.exists(tif_out_file) and not force:
        return
    
    n1 = Nansat(s1file)
    n1.reproject_gcps()

    n1.resize(0.25)
    i = Sentinel1Image(s1file)
    s0 = {}
    for pol in ['HH', 'HV']:
        s0[pol] = i.remove_texture_noise(pol)
        s0[pol] = 10*np.log10(s0[pol])
        s0[pol] = gaussian_filter(s0[pol], 1, truncate=2)[::4, ::4]
        
        s0_out = np.zeros(n1.shape(), np.float32)
        min_rows, min_cols = min(n1.shape()[0], s0[pol].shape[0]), min(n1.shape()[1], s0[pol].shape[1])
        s0_out[:min_rows, :min_cols] = s0[pol][:min_rows, :min_cols]
        s0[pol] = s0_out

    n_out = Nansat.from_domain(n1)
    n_out.add_bands([s0['HH'], s0['HV']], [{'name': 'sigma0_HH'}, {'name': 'sigma0_HV'}])
    n_out.export(tif_out_file, driver='GTiff', options='COMPRESS=LZW')

    png_out_file = get_outut_filename(s1file, png_out_dir, 'png')
    fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    clim0 = np.nanpercentile(s0['HH'][:1500, :1500], (0.1, 99.9))
    clim1 = np.nanpercentile(s0['HV'][:1500, :1500], (0.1, 99.9))
    plt.colorbar(axs[0].imshow(s0['HH'], cmap='gray', clim=clim0), shrink=0.5)
    plt.colorbar(axs[1].imshow(s0['HV'], cmap='gray', clim=clim1), shrink=0.5)
    for ax in axs:
        ax.axis('off')
    plt.savefig(png_out_file, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    del n1, n_out, s0
    gc.collect()
    print(s1file)


def preproc_s1_grd_inc(s1file):
    """ Extract only incidenc agle from the geolocation """
    inc_out_file = get_outut_filename(s1file, inc_out_dir, 'npz')
    if os.path.exists(inc_out_file) and not force:
        return
    
    i = Sentinel1Image(s1file)
    data = dict(
        incidence_angle = i.geolocation('HV')['incidenceAngle'],
        pixel = i.geolocation('HV')['pixel'],
        line = i.geolocation('HV')['line'],
    )
    np.savez(inc_out_file, **data)


if __name__ == '__main__':
    with Pool(2) as pool:
        #pool.map(preproc_s1_grd, ifiles)
        pool.map(preproc_s1_grd_inc, ifiles)
