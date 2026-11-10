from s1denoise import Sentinel1Image
import time
import os
import glob

from tqdm import tqdm
from warping import *
from nansat import Nansat, Domain, NSR

from cartopy import crs as ccrs
from cartopy.feature import LAND

import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from skimage.util import view_as_windows

import glob

import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, feature, future
from sklearn.ensemble import RandomForestClassifier
from functools import partial


sigma_min = 0.5
sigma_max = 4
features_func = partial(
    feature.multiscale_basic_features,
    intensity=True,
    edges=False,
    texture=True,
    sigma_min=sigma_min,
    sigma_max=sigma_max,
    channel_axis=-1,
)
clf = RandomForestClassifier(n_estimators=50, n_jobs=-1, max_depth=10, max_samples=0.05)

load_v1 = False
force = False
x_res, y_res = 500, 500
ps = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)
asip_dir = '/Volumes/KINGSTON/dmi_asip/SEAICE_ARC_PHY_AUTO_L3_MYNRT_011_023/cmems_obs-si_arc_phy_nrt_l3_P1D_202411'

idir = '/Volumes/sat_downloads_sentinel1/2026/s1pop'
ifiles = glob.iglob(f'{idir}/S1?_EW_GRDM_1SDH_202505*SAFE')
#print(len(ifiles), os.path.basename(ifiles[0]), os.path.basename(ifiles[-1]))
for ifile in ifiles:
    print(ifile)
    break


odir = 'collocated_v3'

for s1file in tqdm(ifiles):
    ofile = f'{odir}/{os.path.basename(s1file)}#1_1.png'
    if os.path.exists(ofile) and not force:
        continue
    
    #t0 = time.time()
    n1 = Nansat(s1file)
    n1.reproject_gcps()
    n1.vrt.tps = True
    n1c_lon, n1c_lat = n1.get_corners()
    n1cx, n1cy = ps.transform_points(ccrs.PlateCarree(), n1c_lon, n1c_lat)[:, :2].T
    #print(f"Open {s1file} in {time.time() - t0:.2f} seconds")

    xmin, xmax = float(n1cx.min() - x_res), float(n1cx.max() + x_res)
    ymin, ymax = float(n1cy.min() - y_res), float(n1cy.max() + y_res)

    asip_mask = n1.time_coverage_start.strftime('%Y/%m/*_%Y%m%d.nc')
    asip_file = glob.glob(f'{asip_dir}/{asip_mask}')[0]
    
    #t0 = time.time()
    with xr.open_dataset(asip_file) as ds:
        xc_ascending = ds.xc.values[0] < ds.xc.values[-1]
        yc_ascending = ds.yc.values[0] < ds.yc.values[-1]

        xc_slice = slice(xmin, xmax) if xc_ascending else slice(xmax, xmin)
        yc_slice = slice(ymin, ymax) if yc_ascending else slice(ymax, ymin)

        sic_subset = ds["sic"].sel(xc=xc_slice, yc=yc_slice)[0].values + 2
        status_flag = ds['status_flag'].sel(xc=xc_slice, yc=yc_slice)[0].values
        sic_subset[np.isnan(sic_subset)] = 0
        sic_subset[status_flag & 1 == 1] = 1
    #print(f"Open {asip_file} in {time.time() - t0:.2f} seconds")

    #t0 = time.time()
    src_dom = Domain(
        '+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs',
        f'-te {(xmin - x_res/2):.2f} {(ymin - y_res/2):.2f} {(xmax + x_res/2):.2f} {(ymax + y_res/2):.2f} -tr {x_res} {y_res}'
        )
    n1.resize(0.25)
    blon, blat = n1.get_border()
    dst_img = warp(src_dom, sic_subset, n1)
    #print(f"Warp in {time.time() - t0:.2f} seconds")
    windows = {}
    sub_size = sub_step = np.array(dst_img.shape) // 2

    if not load_v1:
        i = Sentinel1Image(s1file)
        s0 = {}
        loaded_data = True
        for pol in ['HH', 'HV']:
            try:
                s0[pol] = i.remove_thermal_noise(pol)
            except Exception as e:
                print(f"Error processing {s1file} {pol} polarization: {e}")
                loaded_data = False
        #print(f"Load and denoise in {time.time() - t0:.2f} seconds")
        if not loaded_data:
            continue
        #t0 = time.time()
        s0db = {}
        for pol in ['HH', 'HV']:
            s0db[pol] = 10*np.log10(s0[pol])
        s0dbf = {}
        for pol in ['HH', 'HV']:
            s0dbf[pol] = gaussian_filter(s0db[pol], 2, truncate=2)[::4, ::4]
        
        for pol in ['HH', 'HV']:
            windows[pol] = view_as_windows(s0dbf[pol], sub_size, sub_step)
    
    windows['sic'] = view_as_windows(dst_img, sub_size, sub_step)
    #print(f"Split into windows in {time.time() - t0:.2f} seconds")

    p_lim = 0.2
    for i in range(windows['sic'].shape[0]):
        for j in range(windows['sic'].shape[1]):

            if load_v1:
                # NB: Load data from preprocessed V1
                ifile_v1 = f'collocated_v1/{os.path.basename(s1file)}#{i}_{j}.png'
                img_v1 = plt.imread(ifile_v1)
                r = (img_v1[:, :, 0] * 255).astype(np.uint8)
                b = (img_v1[:, :, 2] * 255).astype(np.uint8)
            else:
                landmask = windows['sic'][i, j]==1
                r = windows['HV'][i, j].copy()
                r_min, r_max = np.nanpercentile(r[(r > -40) & ~landmask], [p_lim, 100-p_lim])
                r = np.clip((r - r_min) / (r_max - r_min), 0, 1)
                r = np.where(np.isnan(r), 0, 1+254*r).astype(np.uint8)
                print(r_min, r_max)
                b = windows['HH'][i, j].copy()
                b_min, b_max = np.nanpercentile(b[~landmask], [p_lim, 100-p_lim])
                b = np.clip((b - b_min) / (b_max - b_min), 0, 1)
                b = np.where(np.isnan(b), 0, 1+254*b).astype(np.uint8)
                print(b_min, b_max)

            #g = r * (r + 2 * b * (1 - r))
            g = windows['sic'][i, j].astype(np.uint8)

            # Supervised segmentation using Random Forest Classifier
            rr, bb = r / 255., b / 255.
            gg = (rr * (rr + 2 * bb * (1 - rr)) * 255).astype(np.uint8)
            img_seg = np.dstack([r, gg, b])

            invalid_mask = (g <= 1) | ((r == 0) & (b == 0))
            sic = (g - 2).astype(float)

            training_labels = np.zeros(sic.shape, dtype=np.uint8)
            training_labels = 1 + np.floor(sic / 20).astype(np.uint8)
            training_labels[invalid_mask] = 0

            features = features_func(img_seg)        
            clf = future.fit_segmenter(training_labels, features, clf)
            result = future.predict_segmenter(features, clf)
            result[invalid_mask] = 0
            img = np.dstack([r, g, b, 255-result])

            #img = np.dstack([r, g, b])
            plt.imsave(f'{odir}/{os.path.basename(s1file)}#{i}_{j}.png', img[:, ::-1, :])

            row_start = i * sub_size[0]
            col_start = j * sub_size[1]
            row_end = row_start + sub_size[0]
            col_end = col_start + sub_size[1]
            col_vec = np.arange(col_start, col_end, 100)
            row_vec = np.arange(row_start, row_end, 100)
            col_bnd = np.concatenate([col_vec, [col_end] * len(row_vec), col_vec[::-1], [col_start] * len(row_vec)])
            row_bnd = np.concatenate([[row_start] * len(col_vec), row_vec, [row_end] * len(col_vec), row_vec[::-1]])
            lon, lat = n1.transform_points(col_bnd, row_bnd)
            fig, axs = plt.subplots(1, 1, figsize=(10, 10), subplot_kw={'projection': ps})
            sic_subset[sic_subset < 2] = np.nan
            plt.imshow(sic_subset, 
                    extent=(xmin, xmax, ymin, ymax), 
                    origin='upper', 
                    cmap='Blues_r', 
                    zorder=0,
                    interpolation='nearest',
                    clim=[0, 120])
            plt.plot(blon, blat, color='red', linewidth=1, zorder=2, transform=ccrs.PlateCarree())
            plt.plot(lon, lat, transform=ccrs.PlateCarree())
            axs.add_feature(LAND, zorder=1)        
            plt.tight_layout()
            plt.savefig(f'{odir}_maps/{os.path.basename(s1file)}_map#{i}_{j}.png', dpi=150, bbox_inches='tight', pad_inches=0.1)
            plt.close('all')
