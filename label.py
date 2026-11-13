import os
import glob

import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import mark_boundaries, slic
from PIL import Image
from matplotlib.widgets import Button, Slider, CheckButtons
from skimage import graph


def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (
        graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count']
    )

class Clicker:
    """ Interactive image labeling tool for ice mask creation. """
    def __init__(self, ifile, outdir, sigma=1, compactness=10, thresh=10, n_segments=400):
        """ Initializes the Clicker with image file and parameters. 
        
        Parameters
        ----------
        ifile : str
            Path to the input image file.
        outdir : str
            Directory to save the output ice mask.
        sigma : float, optional
            Sigma value for image smoothing. Default is 1.
        compactness : float, optional
            Compactness parameter for SLIC segmentation. Default is 10.
        thresh : float, optional
            Threshold value for segmentation. Default is 10.
        n_segments : int, optional
            Number of segments for SLIC. Default is 400.
        """
        self.out_file = f'{outdir}/{os.path.basename(ifile).replace(".png", "_ice_mask.npz")}'
        self.alpha = 0.5
        self.sigma = sigma
        self.compactness = compactness
        self.thresh = thresh
        self.n_segments = n_segments
        self.min_sic = 30 # minimum procentage of sic for initial classification (NOT USED NOW)
        self.min_ice_mask_mean = 1.25
        self.min_invalid = 0.1        
        self.move_counter = 0

    def load_image(self, ifile):
        """ Loads the image and initializes the necessary attributes.
        
        Parameters
        ----------
        ifile : str
            Path to the input image file.
        """
        raw = plt.imread(ifile)
        r = raw[:,:,0].astype(float) # HH
        b = raw[:,:,2].astype(float) # HV
        g = (r * (r + 2 * b * (1 - r))) # ratio
        self.classification = (255 - 255*raw[:,:,3]) # classification of ice concentrations (values 0 to 5)
        self.mask = r != 0 # mask showing where the values for the HH channel exist 
        self.sic = ((255 * (raw[:, :, 1]))-2)
        self.water_mask = self.sic != -1 # masking out land
        self.img = (255 * np.dstack([r, g, b])).astype(np.uint8)
        self.ice_mask = np.zeros_like(self.sic, dtype=np.uint8)
        self.ice_mask[self.classification <= 1] = 1 # using ice classification for the initial classification 
        self.ice_mask[self.classification  > 1] = 2 # using ice classification for the initial classification 
        self.ice_mask[self.sic == -2] = 0
        self.ice_mask[self.sic == -1] = 0

    def onclick(self, event):
        """ Handles mouse click events on the image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        """
        if (self.ice_mask_layer.get_visible() == True):  # allows to change the ice mask only when it's visible
            if event.inaxes != self.ax:  # Ignore clicks outside main axes
                return
            if self.fig.canvas.toolbar.mode == 'zoom rect' or self.fig.canvas.toolbar.mode == 'pan/zoom': # ignors the clicks when zoom or pan is turned on
                return
            print(f'Clicked at: ({event.xdata}, {event.ydata} {event.button}) {self.img[int(event.ydata), int(event.xdata)]}')
            label_id = self.labels[int(event.ydata), int(event.xdata)]
            mask_value = self.ice_mask[int(event.ydata), int(event.xdata)]
            if mask_value == 0:
                return
            self.ice_mask[self.labels == label_id] = {0: 0, 1: 2, 2: 1}[mask_value]
            self.ax.images[2].set_data(self.ice_mask)
            self.fig.canvas.draw()
            
    
    
    def on_press(self, event): 
        """ Handles key press events on the image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.KeyEvent
            The key event.
        """
        print(f"Key pressed: {event.key} {event.ydata} {event.xdata}")
        if event.key == ' ':
            print("You pressed 'space'!")
        elif event.key == 'q':
            plt.close(self.fig) # Close the figure window
            
        # Enables switching the mask layer on and off on the segmented image with the 'm' key
        elif event.key == 'm':
            self.ice_mask_layer.set_visible(not self.ice_mask_layer.get_visible())
            print(f"Mask visible: {self.ice_mask_layer.get_visible()}")
            self.fig.canvas.draw()
        
        # Enables switching the segments layer on and off with the 'r' key
        elif event.key == 'r':
            self.regions_layer.set_visible(not self.regions_layer.get_visible())
            print(f"Segments (regions) visible: {self.regions_layer.get_visible()}")
            self.fig.canvas.draw()

        else:
            return

    def on_move(self, event):
        """ Handles mouse move events on the image.
        
        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        """
        # Check if the mouse is over the axes
        if event.inaxes and event.key in ['z', 'x']:
            label_id = self.labels[int(event.ydata), int(event.xdata)]
            self.ice_mask[(self.labels == label_id) & (self.ice_mask != 0)] = { 'z': 1, 'x': 2 }[event.key]
            print(f'Moving at: ({event.xdata}, {event.ydata}) {self.img[int(event.ydata), int(event.xdata)]}')
            self.move_counter += 1
            if self.move_counter % 10 == 0:  # Update every 10 moves
                self.ax.images[2].set_data(self.ice_mask)
                self.fig.canvas.draw()

    def update_sigma(self, val):
        """ Updates the sigma parameter and re-segments the image. """
        current_zoom_x = self.ax.get_xlim()
        current_zoom_y = self.ax.get_ylim()
        self.sigma = val
        self.segment_image()
        self.update_ice_mask()
        self.imshow()
        self.ax.set_xlim(current_zoom_x)
        self.ax.set_ylim(current_zoom_y)

    def update_compactness(self, val):
        """ Updates the compactness parameter and re-segments the image. """
        current_zoom_x = self.ax.get_xlim()
        current_zoom_y = self.ax.get_ylim()
        self.compactness = val
        self.segment_image()
        self.update_ice_mask()
        self.imshow()
        self.ax.set_xlim(current_zoom_x)
        self.ax.set_ylim(current_zoom_y)
    
    def update_thresh(self, val):
        """ Updates the threshold parameter and re-segments the image. """
        current_zoom_x = self.ax.get_xlim()
        current_zoom_y = self.ax.get_ylim()
        self.thresh = val
        self.segment_image()
        self.update_ice_mask()        
        self.imshow()
        self.ax.set_xlim(current_zoom_x)
        self.ax.set_ylim(current_zoom_y)

    def update_n_segments(self, val):
        """ Updates the n_segments parameter and re-segments the image. """
        current_zoom_x = self.ax.get_xlim()
        current_zoom_y = self.ax.get_ylim()
        self.n_segments = int(val)
        self.segment_image()
        self.update_ice_mask()        
        self.imshow()
        self.ax.set_xlim(current_zoom_x)
        self.ax.set_ylim(current_zoom_y)
    
    def imshow(self):   
        """ Displays the image with the current labels and ice mask. """
        self.ax.clear()
        self.image_layer = self.ax.imshow(self.img)
        self.regions_layer = self.ax.imshow(mark_boundaries(self.img, self.labels, color=(1,1,1)))
        self.ice_mask_layer = self.ax.imshow(self.ice_mask, alpha=self.alpha, cmap='gray')
        self.fig.canvas.draw()


    def figure(self, file_name):
        """ Creates the figure and sets up the interactive elements. """
        self.fig = plt.figure(f'{file_name}', figsize=(15, 15))

        # Main image axes
        self.ax = plt.axes([0.1, 0.3, 0.8, 0.6])
        self.imshow()
        
        # Add slider for sigma control
        ax_sigma = plt.axes([0.1, 0.15, 0.65, 0.03])
        self.slider_sigma = Slider(ax_sigma, 'Sigma', 0.0, 5.0, valinit=self.sigma)
        self.slider_sigma.on_changed(self.update_sigma)

        # Add slider for compactness  control
        ax_compactness = plt.axes([0.1, 0.1, 0.65, 0.03])
        self.slider_compactness = Slider(ax_compactness, 'Compactness', 0.0, 100.0, valinit=self.compactness)
        self.slider_compactness.on_changed(self.update_compactness)
        
        # Add slider for thresh control
        ax_thresh = plt.axes([0.1, 0.05, 0.65, 0.03])
        self.slider_thresh = Slider(ax_thresh, 'Thresh', 0.0, 100.0, valinit=self.thresh)
        self.slider_thresh.on_changed(self.update_thresh)

        # Add slider for n_segments control
        ax_n_segments = plt.axes([0.1, 0.0, 0.65, 0.03])
        self.slider_n_segments = Slider(ax_n_segments, 'N Segments', 200, 1200, valinit=self.n_segments, valstep=1)
        self.slider_n_segments.on_changed(self.update_n_segments)

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.fig.canvas.mpl_connect('key_press_event', self.on_press)

        plt.show()

    def segment_image(self):
        """ Segments the image using SLIC and hierarchical merging. """
        labels0 = slic(self.img, n_segments=self.n_segments, compactness=self.compactness, sigma=self.sigma, start_label=1, mask = (self.mask & self.water_mask), enforce_connectivity = True, )
        g = graph.rag_mean_color(self.img, labels0)
        self.labels = graph.merge_hierarchical(
            labels0,
            g,
            thresh=self.thresh,
            rag_copy=False,
            in_place_merge=True,
            merge_func=merge_mean_color,
            weight_func=_weight_mean_color,
        )

    def update_ice_mask(self):
        """ Updates the ice mask based on the current labels. """
        label_ids = np.unique(self.labels)
        new_ice_mask = np.zeros_like(self.ice_mask, dtype=np.uint8)
        for label_id in label_ids:
            label_mask = self.labels == label_id
            label_size = np.sum(label_mask)
            valid_mask = self.mask & label_mask & self.water_mask   # works for all pixels with RGB values that aren't land
            valid_size = valid_mask.sum()
            valid_size_rel = valid_size / label_size
            if valid_size_rel > self.min_invalid:
                ice_mask_mean = np.mean(self.ice_mask[valid_mask])
                if ice_mask_mean < self.min_ice_mask_mean:
                    new_ice_mask[valid_mask] = 1
                else:
                    new_ice_mask[valid_mask] = 2
        self.ice_mask = new_ice_mask

    def save_ice_mask(self):
        """ Saves the ice mask and labels to a file. """
        np.savez(self.out_file, ice_mask=self.ice_mask.data, labels=self.labels, sigma = self.sigma, compactness = self.compactness, thresh = self.thresh, n_segments = self.n_segments)

def main():
    """ Main function to run the interactive image labeling tool. """
    
    # TODO: 
    # Add argument parsing for input and output directories, and parameters
    
    idir = 'collocated'
    outdir = 'ice_masks'
    sigma=2
    compactness=10
    thresh=5
    n_segments=800

    ifiles = sorted(glob.glob(f'{idir}/*.png'))
    for i, ifile in enumerate(ifiles):
        clicker = Clicker(ifile, outdir, sigma=sigma, compactness=compactness, thresh=thresh, n_segments=n_segments)
        clicker.load_image(ifile)
        if os.path.exists(clicker.out_file):
            continue      # skips the png files which alredy have an ice_mask created and updated
            clicker.ice_mask = np.load(clicker.out_file)['ice_mask']
            clicker.sigma = float(np.load(clicker.out_file)["sigma"])
            clicker.compactness = float(np.load(clicker.out_file)["compactness"])
            clicker.thresh = float(np.load(clicker.out_file)["thresh"])
            clicker.n_segments = int(np.load(clicker.out_file)["n_segments"])
        
        clicker.segment_image()
        clicker.update_ice_mask()
        clicker.figure(ifile)
        clicker.save_ice_mask()
        #break

if __name__ == "__main__":
    main()
