import os
import glob

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from cmocean import cm
from tqdm import tqdm
from multiprocessing import Pool

from pixel_classify import classify_pixels

def load_image(ifile):
    """ Load sea ice concentration, valid pixels mask, water pixel mask, RGB image
    
    Parameters
    ----------
    ifile : str
        Path to the input image file.
    """
    try:
        raw = plt.imread(ifile)
    except SyntaxError as e:
        print(f'Cannot read {ifile}')
        raise e

    r = raw[:,:,0].astype(float) # HH
    b = raw[:,:,2].astype(float) # HV
    g = (r * (r + 2 * b * (1 - r))) # ratio
    mask = r != 0 # mask showing where the values for the HH channel exist 
    sic = ((255 * (raw[:, :, 1]))-2)
    water_mask = sic != -1 # masking out land
    rgb_img = (255 * np.dstack([r, g, b])).astype(np.uint8)
    return sic, mask, water_mask, rgb_img

class Reclassify:
    def __init__(self, 
                 tif_dir='./s1_grd_preproc_tif', 
                 png_dir='./v3labeled/png', 
                 msk_dir='./v3labeled/ice_masks',
                 out_dir='./v3labeled/reclass',
                 label_confidence=0.8,
                 sic_weight=1.0, 
                 label_weight=1.0,
                 force=False
                 ):
        self.tif_dir = tif_dir
        self.png_dir = png_dir
        self.msk_dir = msk_dir
        self.out_dir = out_dir
        self.label_confidence = label_confidence
        self.sic_weight = sic_weight
        self.label_weight = label_weight
        self.force = force

    def get_scene_ids(self, proc_file_step):
        sub_dirs = sorted(glob.glob(f'{self.msk_dir}/*'))
        ice_mask_files = []
        for sub_dir in sub_dirs:
            ice_mask_files.extend(sorted(glob.glob(f'{sub_dir}/*npz'))[::proc_file_step])
        scene_ids = [os.path.basename(f).split('#')[0].split('.')[0] for f in ice_mask_files]
        return scene_ids

    def run_many(self, proc_file_step=1):
        scene_ids = self.get_scene_ids(proc_file_step)
        for scene_id in tqdm(scene_ids):
            try:
                self.run_one(scene_id)
            except:
                print(f'Error processing {scene_id}')

    def load_labeled(self):
        sic, mask, water_mask = self.load_png()
        ice_mask, segment_label = self.load_ice_mask()
        ice_mask[sic < 0] = 0
        return sic, mask, water_mask, ice_mask, segment_label

    def set_scene_details(self, scene_id):
        self.scene_id = scene_id
        self.platform = scene_id.split('_')[0]
        self.date_str = scene_id.split('_')[4]
        self.plt_date_dir = f'{self.platform}_{self.date_str[:4]}_{self.date_str[4:6]}'

    def run_one(self, scene_id, plot6=False):
        self.set_scene_details(scene_id)
        
        out_file = self.get_out_file()
        if os.path.exists(out_file) and not self.force:
            return

        sic, _, _, ice_mask, segment_label = self.load_labeled()
        s1_sigma0_HH, s1_sigma0_HV = self.load_tif(sic)

        ice_prob, ice_class, details = classify_pixels(
            s1_sigma0_HH, s1_sigma0_HV, sic, segment_label, ice_mask,
            n_neighbours=9,              # segments pooled for local class statistics
            smooth_size=3,               # boxcar speckle filter on sigma0
            label_confidence=self.label_confidence,
            sic_weight=self.sic_weight, 
            label_weight=self.label_weight,
            n_iter=5,                    # EM iterations
            posterior_smooth_sigma=1.0,  # spatial smoothing of posterior log-odds
            return_details=True,
        )
        if plot6:
            self.plot6(sic, s1_sigma0_HV, ice_mask, details, ice_prob, ice_class)
        self.plot_reclass_sigma0(s1_sigma0_HH, ice_class)

    def load_tif(self, sic):
        # Load HH and HV backscatter from Sentinel-1 SAR image taken over mixed ice and water areas
        sentinel1_tif_file = f'{self.tif_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE.tif'
        s1_ds = gdal.Open(sentinel1_tif_file)
        s1_sigma0_HH, s1_sigma0_HV = s1_ds.ReadAsArray()        
        s1_sigma0_HH, s1_sigma0_HV = [a[:sic.shape[0], :sic.shape[1]] for a in (s1_sigma0_HH, s1_sigma0_HV)]
        return s1_sigma0_HH, s1_sigma0_HV

    def load_png(self):
        # Load sea ice concentration and valid pixels masks from pre-processed PNG files
        png_files = [
            f'{self.png_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#0_0.png',
            f'{self.png_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#0_1.png',
            f'{self.png_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#1_0.png',
            f'{self.png_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#1_1.png',
        ]
        sics = []
        masks = []
        water_masks = []
        for png_file in png_files:
            sic, mask, water_mask, _ = load_image(png_file)
            sics.append(sic)
            masks.append(mask)
            water_masks.append(water_mask)

        # Stack four pieces into one array matching the sentinel-1 image
        sic, mask, water_mask = [np.vstack([
            np.hstack([i[0][:, ::-1], i[1][:, ::-1]]),
            np.hstack([i[2][:, ::-1], i[3][:, ::-1]]),
        ]) for i in [sics, masks, water_masks]]
        return sic, mask, water_mask

    def load_ice_mask(self):
        # Load the automatically detected segments and manuall classification of segements into ice and water classes
        ice_mask_files = [
            f'{self.msk_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#0_0_ice_mask.npz',
            f'{self.msk_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#0_1_ice_mask.npz',
            f'{self.msk_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#1_0_ice_mask.npz',
            f'{self.msk_dir}/{self.plt_date_dir}/{self.scene_id}.SAFE#1_1_ice_mask.npz',
        ]
        ice_masks = []
        segment_labels = []
        label_start_number = 0
        for ice_mask_file in ice_mask_files:
            with np.load(ice_mask_file) as ds:
                ice_mask = ds['ice_mask']
                segment_label = ds['labels']
                segment_label += label_start_number
                label_start_number = segment_label.max() + 1
                ice_masks.append(ice_mask)
                segment_labels.append(segment_label)

        # Stack the ice mask (0 - invalid, 1 - water, 2 - ice) and segment labels into a single array that matches the sigma0 image
        ice_mask, segment_label = [np.vstack([
            np.hstack([i[0][:, ::-1], i[1][:, ::-1]]),
            np.hstack([i[2][:, ::-1], i[3][:, ::-1]]),
        ]) for i in [ice_masks, segment_labels]]
        return ice_mask, segment_label

    def plot6(self, sic, s1_sigma0_HV, ice_mask, details, ice_prob, ice_class):
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), sharex=True, sharey=True)
        axs = axs.ravel()
        axs[0].imshow(sic, vmin=0, vmax=100, cmap=cm.ice); axs[0].set_title('SIC, %')
        axs[1].imshow(s1_sigma0_HV, vmin=-31, vmax=-18, cmap='gray'); axs[1].set_title('HV, dB')
        axs[2].imshow(ice_mask, interpolation='nearest', vmin=0, vmax=2); axs[2].set_title('Manual segment classification')
        axs[3].imshow(details['likelihood_logodds'], vmin=-10, vmax=10, cmap='RdBu_r'); axs[3].set_title('Backscatter log-odds (ice vs water)')
        axs[4].imshow(ice_prob, vmin=0, vmax=1, cmap='RdBu_r'); axs[4].set_title('Posterior ice probability')
        axs[5].imshow(ice_class, interpolation='nearest', vmin=0, vmax=2); axs[5].set_title('Pixel-wise classification')
        plt.tight_layout()
        plt.show()

    def plot_reclass_sigma0(self, s1_sigma0_HH, ice_class):
        plt.figure(figsize=(20,20))
        plt.imshow(s1_sigma0_HH, vmin=-25, vmax=-5, cmap='gray')
        plt.imshow(ice_class, interpolation='nearest', vmin=0, vmax=2, alpha=0.3)
        plt.axis('off')
        plt.tight_layout()
        out_file = self.get_out_file()
        plt.savefig(out_file, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close('all')

    def get_out_file(self):
        exp_dir = self.get_exp_dir()
        out_dir = f'{self.out_dir}/{exp_dir}/{self.plt_date_dir}'
        os.makedirs(out_dir, exist_ok=True)
        out_file = f'{out_dir}/{self.scene_id}_s0_class.png'
        return out_file

    def get_exp_dir(self):
        return f'{int(self.label_confidence*10):02}_{int(self.sic_weight*10):02}_{int(self.label_weight*10):02}'



def run_experiment(exp):
    proc_file_step = 25
    r = Reclassify(**exp)
    print(len(r.get_scene_ids(proc_file_step=proc_file_step)))
    r.run_many(proc_file_step=proc_file_step)

if __name__ == '__main__':
    experiments = [
        dict(
            label_confidence=0.8,
            sic_weight=1.0, 
            label_weight=1.0,
        ),
        dict(
            label_confidence=0.7,
            sic_weight=0.7,
            label_weight=1.0,
        ),
        dict(
            label_confidence=0.6,
            sic_weight=0.5,
            label_weight=1.0,
        )
    ]

    with Pool(3) as p:
        p.map(run_experiment, experiments)