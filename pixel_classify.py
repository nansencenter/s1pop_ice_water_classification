"""Pixel-wise ice/water classification of Sentinel-1 HH/HV imagery.

Refines a coarse segment-wise manual classification into a pixel-wise one using
a spatially local two-class Gaussian model fitted with EM:

* Features: sigma0 HH and HV in dB (optionally speckle-smoothed).
* Likelihoods: for every segment, a local ice and a local water 2D Gaussian
  (mean + covariance of [HH, HV]) are estimated by pooling backscatter
  statistics from the segment itself and its K nearest neighbour segments
  (Gaussian distance weights). Local pooling accounts for incidence-angle
  dependence and spatially varying wind roughening of open water.
* Prior: log-odds combination of the pixel's sea ice concentration (sic / 100)
  and the manual segment label (ice_mask).
* EM: posterior ice probabilities are used as soft weights to re-estimate the
  per-segment class statistics, so that e.g. leads inside ice segments no
  longer contaminate the ice statistics, and floes inside water segments
  contribute to local ice statistics.
"""
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter
from scipy.spatial import cKDTree


def _nan_uniform_filter(a, size):
    """Mean filter that ignores NaNs."""
    valid = np.isfinite(a)
    num = uniform_filter(np.where(valid, a, 0).astype(float), size)
    den = uniform_filter(valid.astype(float), size)
    with np.errstate(invalid='ignore', divide='ignore'):
        out = num / den
    out[~valid] = np.nan
    return out


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _segment_moments(seg, x, w, nseg):
    """Weighted zeroth, first and second moments of features x per segment.

    seg : (N,) int segment index, x : (N, 2) features, w : (N,) weights
    Returns S0 (nseg,), S1 (nseg, 2), S2 (nseg, 2, 2)
    """
    S0 = np.bincount(seg, w, nseg)
    S1 = np.stack([np.bincount(seg, w * x[:, i], nseg) for i in range(2)], axis=1)
    S2 = np.empty((nseg, 2, 2))
    for i in range(2):
        for j in range(i, 2):
            S2[:, i, j] = S2[:, j, i] = np.bincount(seg, w * x[:, i] * x[:, j], nseg)
    return S0, S1, S2


def _pooled_gaussians(S0, S1, S2, nbr_idx, nbr_w, n_prior, min_var):
    """Pool segment moments over spatial neighbours and convert to Gaussians.

    A weak global prior (n_prior pseudo-pixels with the global class
    distribution) regularises segments with few neighbouring class pixels.
    """
    P0 = (S0[nbr_idx] * nbr_w).sum(1)
    P1 = (S1[nbr_idx] * nbr_w[..., None]).sum(1)
    P2 = (S2[nbr_idx] * nbr_w[..., None, None]).sum(1)
    g0 = S0.sum()
    P0 = P0 + n_prior
    P1 = P1 + n_prior * S1.sum(0) / g0
    P2 = P2 + n_prior * S2.sum(0) / g0
    mean = P1 / P0[:, None]
    cov = P2 / P0[:, None, None] - mean[:, :, None] * mean[:, None, :]
    cov = cov + np.eye(2) * min_var
    return mean, cov


def _gauss_loglik(x, seg, mean, cov):
    """Per-pixel log-likelihood under the Gaussian of the pixel's segment."""
    det = cov[:, 0, 0] * cov[:, 1, 1] - cov[:, 0, 1] ** 2
    inv = np.empty_like(cov)
    inv[:, 0, 0] = cov[:, 1, 1] / det
    inv[:, 1, 1] = cov[:, 0, 0] / det
    inv[:, 0, 1] = inv[:, 1, 0] = -cov[:, 0, 1] / det
    d = x - mean[seg]
    i = inv[seg]
    maha = i[:, 0, 0] * d[:, 0] ** 2 + 2 * i[:, 0, 1] * d[:, 0] * d[:, 1] + i[:, 1, 1] * d[:, 1] ** 2
    return -0.5 * (maha + np.log(det[seg]))


def classify_pixels(
    s1_sigma0_HH, s1_sigma0_HV, sic, segment_label, ice_mask,
    n_neighbours=9, smooth_size=3, label_confidence=0.8, sic_clip=0.05,
    sic_weight=1.0, label_weight=1.0, n_prior=200, min_var=0.25,
    n_iter=5, posterior_smooth_sigma=1.0, return_details=False,
):
    """Pixel-wise ice/water classification.

    Parameters
    ----------
    s1_sigma0_HH, s1_sigma0_HV : 2D float arrays
        Sigma0 in dB (NaN = no data).
    sic : 2D array
        Sea ice concentration 0..100; negative values (land / no data) are
        treated as "no SIC information".
    segment_label : 2D int array
        Segment ids from automatic segmentation.
    ice_mask : 2D uint8 array
        Manual segment classification: 0 - invalid, 1 - water, 2 - ice.
    n_neighbours : int
        Number of segments (including itself) pooled for local class statistics.
    smooth_size : int
        Size of the NaN-aware boxcar filter applied to sigma0 (speckle
        reduction). 1 disables smoothing.
    label_confidence : float
        Prior probability that a pixel belongs to the class of its manual label.
    sic_clip : float
        SIC probability is clipped to [sic_clip, 1 - sic_clip] before logit.
    sic_weight, label_weight : float
        Weights of the SIC and manual-label log-odds in the prior.
    n_prior : float
        Pseudo-pixel count of the global class distribution added to local stats.
    min_var : float
        Variance (dB^2) added to the covariance diagonal for stability.
    n_iter : int
        Number of EM iterations (0 = use hard manual labels for statistics only).
    posterior_smooth_sigma : float
        Sigma (pixels) of Gaussian smoothing of posterior log-odds (simple
        spatial regularisation). 0 disables.
    return_details : bool
        If True, also return a dict with prior / likelihood log-odds.

    Returns
    -------
    ice_prob : 2D float array
        Posterior probability of ice (NaN where not classified).
    ice_class : 2D uint8 array
        0 - invalid, 1 - water, 2 - ice (same convention as ice_mask).
    details : dict, optional
    """
    hh, hv = s1_sigma0_HH, s1_sigma0_HV
    if smooth_size > 1:
        hh = _nan_uniform_filter(hh, smooth_size)
        hv = _nan_uniform_filter(hv, smooth_size)

    valid = (ice_mask > 0) & np.isfinite(hh) & np.isfinite(hv)
    seg_ids, seg = np.unique(segment_label[valid], return_inverse=True)
    nseg = seg_ids.size
    x = np.column_stack([hh[valid], hv[valid]]).astype(float)

    # --- Prior log-odds from SIC and manual labels ---
    sic_v = sic[valid].astype(float)
    sic_ok = sic_v >= 0
    sic_lo = np.zeros_like(sic_v)
    sic_lo[sic_ok] = _logit(np.clip(sic_v[sic_ok] / 100, sic_clip, 1 - sic_clip))
    lab_lo = np.where(ice_mask[valid] == 2, 1.0, -1.0) * _logit(label_confidence)
    prior_lo = sic_weight * sic_lo + label_weight * lab_lo

    # --- Spatial neighbourhood of segments (centroids) ---
    rows, cols = np.nonzero(valid)
    cnt = np.bincount(seg, minlength=nseg)
    cent = np.column_stack([np.bincount(seg, rows, nseg), np.bincount(seg, cols, nseg)]) / cnt[:, None]
    k = min(n_neighbours, nseg)
    dist, nbr_idx = cKDTree(cent).query(cent, k=k)
    dist = dist.reshape(nseg, k)
    nbr_idx = nbr_idx.reshape(nseg, k)
    scale = np.median(dist[:, -1]) if k > 1 else 1.0
    nbr_w = np.exp(-0.5 * (dist / scale) ** 2)

    # --- EM: start from hard manual labels ---
    p_ice = (ice_mask[valid] == 2).astype(float)
    for it in range(n_iter + 1):
        stats = {}
        for name, w in (('ice', p_ice), ('water', 1 - p_ice)):
            S0, S1, S2 = _segment_moments(seg, x, w, nseg)
            stats[name] = _pooled_gaussians(S0, S1, S2, nbr_idx, nbr_w, n_prior, min_var)
        llr = _gauss_loglik(x, seg, *stats['ice']) - _gauss_loglik(x, seg, *stats['water'])
        post_lo = llr + prior_lo
        p_ice = 1 / (1 + np.exp(-np.clip(post_lo, -50, 50)))

    # --- Back to 2D, optional spatial regularisation of log-odds ---
    lo2d = np.full(ice_mask.shape, np.nan)
    lo2d[valid] = post_lo
    if posterior_smooth_sigma > 0:
        num = gaussian_filter(np.where(valid, lo2d, 0), posterior_smooth_sigma)
        den = gaussian_filter(valid.astype(float), posterior_smooth_sigma)
        with np.errstate(invalid='ignore', divide='ignore'):
            lo2d = np.where(valid, num / den, np.nan)

    ice_prob = 1 / (1 + np.exp(-np.clip(lo2d, -50, 50)))
    ice_class = np.zeros(ice_mask.shape, np.uint8)
    ice_class[valid] = np.where(ice_prob[valid] > 0.5, 2, 1)

    if not return_details:
        return ice_prob, ice_class

    def to2d(v):
        a = np.full(ice_mask.shape, np.nan)
        a[valid] = v
        return a

    details = dict(
        prior_logodds=to2d(prior_lo),
        likelihood_logodds=to2d(llr),
        ice_mean=stats['ice'][0], water_mean=stats['water'][0],
        ice_cov=stats['ice'][1], water_cov=stats['water'][1],
        segment_ids=seg_ids,
    )
    return ice_prob, ice_class, details

