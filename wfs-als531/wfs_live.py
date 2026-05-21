"""
wfs_online.py
─────────────

Soft X-ray wavefront sensor analysis pipeline for live beamline use.

Pipeline
--------
For a single detector image / 1-D fringe profile, the full workflow is::

    image  → extract_profile        →  1-D profile
           → find_carrier            →  k_peak, FFT
           → extract_envelopes       →  I0, A1, complex_1st
           → find_phase_centroid     →  x_c, beam_mask
           → reconstruct_wavefront   →  W(x) in radians
           → parabolic_focal_fit     →  f_pred, residual aberration
           → propagate_to_focus      →  caustic, z_focus  (optional)
           → wavefront_at_focus      →  W at focal spot   (optional)

For convenience, ``reconstruct_single`` runs all of the above in one call and
``quick_look`` additionally generates a diagnostic plot.

Usage at the beamline
---------------------
The typical Jupyter workflow::

    from wfs_online import quick_look

    # ROI was picked by eye from a raw image
    y_roi = slice(470, 486)

    # One frame in one line
    result = quick_look(
        images[idx], dx_m, grating_pitch_m, wavelength_m, z_gd_m,
        y_roi=y_roi, axis=1,
        propagate=True,
        title=f'E = {energy_eV[idx]:.0f} eV',
    )

Dependencies
------------
- numpy, matplotlib                  (standard)
- monoplus                           (in-house Fresnel propagator;
                                      provides propTF and secondmomt)
- tiled  ==  0.2.1                   (required for live data access at the
                                      beamline; newer versions (e.g. 0.2.9)
                                      have been observed to fail — pin to 0.2.1
                                      until verified otherwise)

Acknowledgements
----------------
This pipeline builds on work and discussions with:
    - Dr. Antoine İşlegen-Wojdyla
    - Dr. Xiaoya Chong
    - Dr. Ka Hung (Henry) Chan

Author : Wei "Francis" He (francisho@lbl.gov / whorwhey@gmail.com)
Date   : May 2026
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import monoplus as mp   # in-house Fresnel propagator (propTF, secondmomt)


# ─── Helpers ────────────────────────────────────────────────────────────────
def energy_eV_to_wavelength_m(energy_eV):
    """Convert photon energy [eV] to wavelength [m]."""
    return 1.23984e-6 / np.asarray(energy_eV, dtype=float)


# ════════════════════════════════════════════════════════════════════════════
# PRIMITIVES
# ════════════════════════════════════════════════════════════════════════════

def extract_profile(image_2d, roi, axis=1):
    """Average a 2-D detector image to a 1-D fringe profile.

    Parameters
    ----------
    image_2d : ndarray, shape (n0, n1)
        Raw detector image.
    roi : slice
        Slice along the averaging axis (the no-fringe direction).
        E.g. slice(470, 486) keeps rows 470–485 if axis=1, or columns
        470–485 if axis=0.
    axis : int, optional (default 1)
        Axis to average over. The fringe direction is `1 - axis`.
        - axis=1 → average over columns, profile runs along rows.
        - axis=0 → average over rows,    profile runs along columns.

    Returns
    -------
    profile : ndarray, shape (n_{1-axis},)
        1-D intensity profile along the fringe direction.
    """
    slicer = [slice(None), slice(None)]
    slicer[axis] = roi
    sub = image_2d[tuple(slicer)]
    return np.mean(sub, axis=axis).astype(float)


def find_carrier(profile, dx, grating_pitch, band_factor=1.5):
    """Locate the carrier peak in the FFT of a fringe profile.

    The carrier is the +1-order spatial frequency of the shearing grating,
    nominally at f = 1/p. The actual peak shifts slightly with wavefront
    tilt, so we search a band around 1/p rather than picking it as known.

    Parameters
    ----------
    profile : array_like, shape (N,)
        1-D fringe intensity profile from extract_profile.
    dx : float  [m]
        Pixel pitch in the fringe direction.
    grating_pitch : float  [m]
        Nominal grating pitch (fabrication spec).
    band_factor : float, optional (default 1.5)
        Search band is (1/p / band_factor,  band_factor / p).
        Set higher (e.g. 1.5–2.0) if strong tilt pulls the peak far from 1/p.

    Returns
    -------
    out : dict
        k_peak     [cyc/m]  measured carrier frequency
        k_amp      [a.u.]   |FFT| at the carrier — quality indicator
        k_ideal    [cyc/m]  1/grating_pitch
        fft_avg    ndarray  centered FFT of the profile (complex)
        freq       ndarray  centered frequency axis [cyc/m]
        band       (lo, hi) the search band actually used [cyc/m]
    """
    profile = np.asarray(profile, dtype=float).ravel()
    N = len(profile)

    fft_avg = np.fft.fftshift(np.fft.fft(profile))
    freq    = np.fft.fftshift(np.fft.fftfreq(N, dx))

    k_ideal = 1.0 / grating_pitch
    band_lo = k_ideal / band_factor
    band_hi = k_ideal * band_factor

    search_mask = (freq >= band_lo) & (freq <= band_hi)
    if not search_mask.any():
        raise ValueError(
            f"No FFT frequencies fall inside search band "
            f"[{band_lo*1e-3:.1f}, {band_hi*1e-3:.1f}] cyc/mm. "
            f"Check dx, grating_pitch, and N."
        )

    carrier_idx = np.argmax(np.abs(fft_avg) * search_mask)
    k_peak      = freq[carrier_idx]
    k_amp       = float(np.abs(fft_avg[carrier_idx]))

    return dict(
        k_peak  = k_peak,
        k_amp   = k_amp,
        k_ideal = k_ideal,
        fft_avg = fft_avg,
        freq    = freq,
        band    = (band_lo, band_hi),
    )


def extract_envelopes(fft_avg, freq, k_peak,
                      sigma_ratio_0=5, sigma_ratio_1=10):
    """Extract 0th-order and +1-order complex envelopes by Gaussian filtering.

    Parameters
    ----------
    fft_avg : ndarray, shape (N,) complex
        Centered FFT of the fringe profile (from find_carrier).
    freq : ndarray, shape (N,)
        Centered frequency axis [cyc/m] (from find_carrier).
    k_peak : float  [cyc/m]
        Measured carrier frequency (from find_carrier).
    sigma_ratio_0 : float, optional (default 5)
        0th-order Gaussian width: σ_0 = k_peak / sigma_ratio_0.
        Smaller ratio → wider filter.
    sigma_ratio_1 : float, optional (default 10)
        +1-order Gaussian width: σ_1 = k_peak / sigma_ratio_1.

    Returns
    -------
    out : dict
        complex_0th : ndarray (N,) complex   0th-order complex field
        complex_1st : ndarray (N,) complex   +1-order complex field
        I0          : ndarray (N,)           |complex_0th|² — beam intensity
        A1          : ndarray (N,)           |complex_1st|  — fringe amplitude
        sigma_0     : float [cyc/m]          0th-order filter width
        sigma_1     : float [cyc/m]          +1-order filter width
        gauss_lp    : ndarray (N,)           0th-order filter (for plotting)
        gauss_bp    : ndarray (N,)           +1-order filter (for plotting)
    """
    sigma_0 = k_peak / sigma_ratio_0
    sigma_1 = k_peak / sigma_ratio_1

    # 0th-order: low-pass Gaussian centered at f = 0
    gauss_lp    = np.exp(-0.5 * (freq / sigma_0)**2)
    fft_filt_0  = fft_avg * gauss_lp
    complex_0th = np.fft.ifft(np.fft.ifftshift(fft_filt_0))
    I0          = np.abs(complex_0th)**2

    # +1-order: bandpass Gaussian centered at f = k_peak
    gauss_bp    = np.exp(-0.5 * ((freq - k_peak) / sigma_1)**2)
    fft_filt_1  = fft_avg * gauss_bp
    complex_1st = np.fft.ifft(np.fft.ifftshift(fft_filt_1))
    A1          = np.abs(complex_1st)

    return dict(
        complex_0th = complex_0th,
        complex_1st = complex_1st,
        I0          = I0,
        A1          = A1,
        sigma_0     = sigma_0,
        sigma_1     = sigma_1,
        gauss_lp    = gauss_lp,
        gauss_bp    = gauss_bp,
    )


def fill_mask_gaps(mask):
    """Fill False gaps between the leftmost and rightmost True values.

    A "trust region" beam mask should be a single contiguous block —
    if the threshold criterion (e.g. I0 > threshold) has small dips
    in the middle of the beam, fill_mask_gaps closes them.

    Parameters
    ----------
    mask : array_like of bool, shape (N,)
        Boolean mask, typically from I0 > threshold.

    Returns
    -------
    filled : ndarray of bool, shape (N,)
        Mask with all False values between the first and last True
        promoted to True. Returns the input unchanged if mask is all False.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return mask.copy()
    i_first = np.argmax(mask)                          # first True
    i_last  = len(mask) - 1 - np.argmax(mask[::-1])    # last True
    filled = mask.copy()
    filled[i_first:i_last + 1] = True
    return filled


def find_phase_centroid(I0, A1, threshold=None):
    """Find the beam centroid for use as phase reference.

    Builds a beam mask from I0 > threshold * max(I0), fills gaps,
    then computes the centroid weighted by |A1|² inside that mask.

    Parameters
    ----------
    I0 : array_like, shape (N,)
        0th-order beam intensity (from extract_envelopes).
    A1 : array_like, shape (N,)
        +1-order fringe amplitude (from extract_envelopes).
    threshold : float, optional (default 1/e² ≈ 0.135)
        Fraction of I0 peak above which pixels are kept in the beam mask.

    Returns
    -------
    out : dict
        x_c          : int        centroid pixel index
        beam_mask    : ndarray    gap-filled boolean mask (N,)
        threshold    : float      threshold actually used
        n_mask_px    : int        number of True pixels in mask
        n_filled_px  : int        number of False→True conversions by gap-fill
    """
    if threshold is None:
        threshold = np.exp(-2)   # 1/e² ≈ 0.135

    I0 = np.asarray(I0, dtype=float)
    A1 = np.asarray(A1, dtype=float)

    raw_mask  = I0 > threshold * I0.max()
    beam_mask = fill_mask_gaps(raw_mask)

    if not beam_mask.any():
        raise ValueError(
            f"Beam mask is empty at threshold={threshold:.3g}. "
            f"Check I0 (max={I0.max():.3g}) or lower threshold."
        )

    weights = A1[beam_mask] ** 2
    x_idx   = np.arange(len(I0))
    x_c     = int(np.average(x_idx[beam_mask], weights=weights))

    return dict(
        x_c         = x_c,
        beam_mask   = beam_mask,
        threshold   = threshold,
        n_mask_px   = int(beam_mask.sum()),
        n_filled_px = int(beam_mask.sum() - raw_mask.sum()),
    )


def reconstruct_wavefront(complex_1st, dx, grating_pitch, wavelength, z_gd, x_c):
    """Reconstruct wavefront W(x) in radians from the +1-order complex envelope.

    Implements the corrected May-2026 pipeline:
        1. Unwrap phase of complex_1st.
        2. Center unwrapped phase at x_c (beam centroid, not array midpoint).
        3. Subtract ideal carrier ramp 2π/p · (x − x[x_c]).
        4. Cumulative-integrate the differential phase.
        5. Re-center result at x_c.

    Parameters
    ----------
    complex_1st : ndarray (N,) complex
        +1-order complex envelope (from extract_envelopes).
    dx : float [m]
        Pixel pitch.
    grating_pitch : float [m]
        Ideal grating pitch (used for the carrier ramp, NOT the filter).
    wavelength : float [m]
        X-ray wavelength.
    z_gd : float [m]
        Grating-to-detector distance.
    x_c : int
        Phase reference pixel index (from find_phase_centroid).

    Returns
    -------
    out : dict
        W_rad        : ndarray (N,) [rad]   Reconstructed W(x), centered at x_c
        delta_phi    : ndarray (N,) [rad]   Differential phase after carrier removal
        phase_unwrap : ndarray (N,) [rad]   Unwrapped phase, centered at x_c
        x_m          : ndarray (N,) [m]     Spatial coordinate (x = 0 at x_c)
        carrier_ramp : ndarray (N,) [rad]   Ideal carrier ramp (for diagnostics)
    """
    N = len(complex_1st)

    # Spatial coordinate: x = 0 at the phase centroid
    x_m = (np.arange(N) - x_c) * dx

    # 1. Unwrap and re-center at x_c
    phase_unwrap = np.unwrap(np.angle(complex_1st))
    phase_unwrap = phase_unwrap - phase_unwrap[x_c]

    # 2. Ideal carrier ramp (centered at x_c, so it's zero there)
    carrier_ramp = 2 * np.pi / grating_pitch * x_m

    # 3. Differential phase
    delta_phi = phase_unwrap - carrier_ramp

    # 4. Integrate (sign follows May 19 convention; flip if your +1/−1 order differs)
    W_temp = -np.cumsum(delta_phi) * dx * grating_pitch / wavelength / z_gd

    # 5. Re-center at x_c
    W_rad = W_temp - W_temp[x_c]

    return dict(
        W_rad        = W_rad,
        delta_phi    = delta_phi,
        phase_unwrap = phase_unwrap,
        x_m          = x_m,
        carrier_ramp = carrier_ramp,
    )


def parabolic_focal_fit(W_rad, x_m, wavelength, mask=None):
    """Fit a parabola to W(x) and predict focal length from defocus.

    For a converging wave, W(x) = -k x² / (2f) where k = 2π/λ.
    Fitting W = a₂ x² + a₁ x + a₀ gives:
        f_pred = -k / (2 a₂)
    Positive f_pred  → focus is downstream of the detector (converging beam).
    Negative f_pred  → focus is upstream (diverging or virtual focus).

    Parameters
    ----------
    W_rad : array_like (N,)  [rad]
        Wavefront from reconstruct_wavefront.
    x_m : array_like (N,)  [m]
        Spatial coordinate (x = 0 at the phase centroid).
    wavelength : float  [m]
        X-ray wavelength.
    mask : array_like (N,) of bool, optional
        Fit region. If None, fits over the entire array.

    Returns
    -------
    out : dict
        f_pred           : float [m]    Predicted focal length offset
        a2, a1, a0       : float        Polynomial coefficients (rad units)
        W_para_rad       : ndarray (N,) [rad]   Fitted parabola
        W_resid_rad      : ndarray (N,) [rad]   W − parabola (residual aberration)
        rms_resid_nm     : float [nm]   RMS of residual inside the mask
        pv_resid_nm      : float [nm]   PV of residual inside the mask
    """
    W_rad = np.asarray(W_rad)
    x_m   = np.asarray(x_m)

    if mask is None:
        mask = np.ones_like(W_rad, dtype=bool)

    # Fit parabola inside mask only
    a2, a1, a0 = np.polyfit(x_m[mask], W_rad[mask], 2)

    # Evaluate fit and residual everywhere
    W_para_rad  = a2 * x_m**2 + a1 * x_m + a0
    W_resid_rad = W_rad - W_para_rad

    # Predicted focal length offset (uses defocus coefficient a2)
    k = 2 * np.pi / wavelength
    f_pred = -k / (2 * a2)

    # Residual statistics inside the mask (the meaningful region)
    W_resid_nm   = W_resid_rad * wavelength / (2 * np.pi) * 1e9
    rms_resid_nm = float(np.std(W_resid_nm[mask]))
    pv_resid_nm  = float(W_resid_nm[mask].max() - W_resid_nm[mask].min())

    return dict(
        f_pred       = f_pred,
        a2           = a2,
        a1           = a1,
        a0           = a0,
        W_para_rad   = W_para_rad,
        W_resid_rad  = W_resid_rad,
        rms_resid_nm = rms_resid_nm,
        pv_resid_nm  = pv_resid_nm,
    )


def propagate_to_focus(A, W_rad, dx, wavelength, z_range_m, x_m=None):
    """Fresnel-propagate the reconstructed field to find the focus.

    Constructs the field at the detector plane
        E_det(x) = A(x) · exp[i W(x)]
    and propagates it over a range of z values using monoplus.propTF.
    The focus is identified as the z at which the RMS beam size σ(z)
    is minimum.

    Parameters
    ----------
    A : array_like (N,)  [a.u.]
        Field amplitude at the detector plane. Use A1 from extract_envelopes
        — it naturally tapers to ~0 outside the beam, which suppresses
        edge-diffraction artifacts during propagation.
    W_rad : array_like (N,)  [rad]
        Reconstructed wavefront (from reconstruct_wavefront).
    dx : float  [m]
        Pixel pitch.
    wavelength : float  [m]
        X-ray wavelength.
    z_range_m : array_like  [m]
        Propagation distances to evaluate. e.g. np.linspace(-3, 3, 601).
    x_m : array_like (N,) [m], optional
        Spatial coordinate — needed for σ_rms computation. If None,
        uses (np.arange(N) - N//2) * dx.

    Returns
    -------
    out : dict
        E_caustic     : ndarray (N, Z) complex   field at each z
        I_caustic     : ndarray (N, Z)           |E|² intensity caustic
        sigma_rms     : ndarray (Z,) [m]         σ_rms vs z
        z_range_m     : ndarray (Z,) [m]
        z_focus       : float [m]                z minimizing σ_rms
        sigma_focus   : float [m]                σ_rms at focus
        i_focus       : int                      index of focus in z_range_m
        E_focus       : ndarray (N,) complex     field at focus
        I_focus       : ndarray (N,)             |E|² at focus
    """
    N = len(A)
    if x_m is None:
        x_m = (np.arange(N) - N // 2) * dx

    L_m = N * dx                      # physical array length

    # Build detector-plane complex field
    E_det = A * np.exp(1j * W_rad)

    # Propagate over z_range
    z_range_m = np.asarray(z_range_m, dtype=float)
    Z = len(z_range_m)

    E_caustic = np.zeros((N, Z), dtype=complex)
    for i_z, z in enumerate(z_range_m):
        E_caustic[:, i_z] = mp.propTF(E_det, L_m, wavelength, z)

    I_caustic = np.abs(E_caustic)**2
    sigma_rms = mp.secondmomt(Z, x_m, E_caustic)

    # Locate focus
    i_focus     = int(np.argmin(sigma_rms))
    z_focus     = float(z_range_m[i_focus])
    sigma_focus = float(sigma_rms[i_focus])

    return dict(
        E_caustic   = E_caustic,
        I_caustic   = I_caustic,
        sigma_rms   = sigma_rms,
        z_range_m   = z_range_m,
        z_focus     = z_focus,
        sigma_focus = sigma_focus,
        i_focus     = i_focus,
        E_focus     = E_caustic[:, i_focus],
        I_focus     = I_caustic[:, i_focus],
    )


def wavefront_at_focus(prop, wavelength, dx, threshold=None):
    """Extract the wavefront W(x) at the focus, restricted to the focal spot.

    The focal spot is defined as |E_focus|² > threshold·max(|E_focus|²),
    with gap-filling. The wavefront is the unwrapped phase of E_focus,
    centered at the intensity centroid of the focus.

    Parameters
    ----------
    prop : dict
        Output of propagate_to_focus.
    wavelength : float [m]
        X-ray wavelength (for nm conversion).
    dx : float [m]
        Pixel pitch.
    threshold : float, optional (default 1/e²)
        Intensity-mask threshold for the focal spot.

    Returns
    -------
    out : dict
        W_focus_rad     : ndarray (N,) [rad]    unwrapped phase at focus, centered
        W_focus_nm      : ndarray (N,) [nm]     W_focus_rad in nm OPL
        focus_mask      : ndarray (N,) of bool  gap-filled focal-spot mask
        x_c_focus       : int                   centroid pixel of focus intensity
        rms_at_focus_nm : float [nm]            RMS inside focus_mask
        pv_at_focus_nm  : float [nm]            PV  inside focus_mask
    """
    if threshold is None:
        threshold = np.exp(-2)

    E = prop['E_focus']
    I = prop['I_focus']
    N = len(E)

    # Focus mask: intensity > threshold × peak, gap-filled
    raw_mask   = I > threshold * I.max()
    focus_mask = fill_mask_gaps(raw_mask)

    if not focus_mask.any():
        raise ValueError("Empty focus mask — lower threshold or check propagation result.")

    # Centroid of focus intensity (use I-weighting, not A²-weighting,
    # since at focus there's no separate fringe-amplitude concept)
    x_idx     = np.arange(N)
    x_c_focus = int(np.average(x_idx[focus_mask], weights=I[focus_mask]))

    # Unwrap phase and center at the focus centroid
    W_focus_rad = np.unwrap(np.angle(E))
    W_focus_rad = W_focus_rad - W_focus_rad[x_c_focus]

    # nm conversion
    W_focus_nm = W_focus_rad * wavelength / (2 * np.pi) * 1e9

    # Statistics inside focus mask
    rms_at_focus_nm = float(np.std(W_focus_nm[focus_mask]))
    pv_at_focus_nm  = float(W_focus_nm[focus_mask].max() - W_focus_nm[focus_mask].min())

    return dict(
        W_focus_rad     = W_focus_rad,
        W_focus_nm      = W_focus_nm,
        focus_mask      = focus_mask,
        x_c_focus       = x_c_focus,
        rms_at_focus_nm = rms_at_focus_nm,
        pv_at_focus_nm  = pv_at_focus_nm,
    )


# ════════════════════════════════════════════════════════════════════════════
# ORCHESTRATORS
# ════════════════════════════════════════════════════════════════════════════

def reconstruct_single(
    image_or_profile,
    dx, grating_pitch, wavelength, z_gd,
    *,
    y_roi=None, axis=1,
    band_factor=1.5,
    sigma_ratio_0=5, sigma_ratio_1=10,
    centroid_threshold=None,
    propagate=False,
    z_range_m=None,
    focus_threshold=None,
    k_amp_good=2000, k_amp_bad=500,
    verbose=False,
):
    """Full single-frame WFS pipeline: image → wavefront (+ optional propagation).

    Steps:
        1. extract_profile (if 2-D input)
        2. find_carrier
        3. extract_envelopes
        4. find_phase_centroid
        5. reconstruct_wavefront
        6. parabolic_focal_fit
        7. (optional) propagate_to_focus + wavefront_at_focus

    Parameters
    ----------
    image_or_profile : ndarray
        Either a 2-D detector image (will be reduced to 1-D via y_roi),
        or an already-extracted 1-D fringe profile.
    dx, grating_pitch, wavelength, z_gd : float
        Required physical parameters [m].
    y_roi : slice, optional
        Required if image_or_profile is 2-D; passed to extract_profile.
    axis : int
        Averaging axis for extract_profile (1 = average over columns/rows
        depending on convention). Default 1.
    band_factor : float
        Carrier search half-band width factor (search = 1/p / f to 1/p * f).
    sigma_ratio_0, sigma_ratio_1 : float
        Gaussian filter widths in units of k_peak.
    centroid_threshold : float
        I0 mask threshold (default 1/e²).
    propagate : bool
        If True, run steps 8a + 8b (propagation + W at focus).
    z_range_m : array_like, optional
        Custom z range for propagation. If None, auto-set from f_pred.
    focus_threshold : float
        Focus mask threshold (default 1/e²).
    k_amp_good, k_amp_bad : float
        Quality thresholds on carrier amplitude:
        - amp >= k_amp_good → 'good'
        - k_amp_bad <= amp < k_amp_good → 'ok'
        - amp < k_amp_bad → 'bad' (likely near-focus, no fringes)
    verbose : bool
        Print a one-line summary.

    Returns
    -------
    result : dict
        Flat dict containing keys from all pipeline stages, plus:
        - quality : 'good' / 'ok' / 'bad'
        - W_nm    : reconstructed W in nm OPL (convenience)
        - x_mm    : x_m in mm (convenience)
    """
    arr = np.asarray(image_or_profile, dtype=float)

    # --- Step 1: profile ---
    if arr.ndim == 2:
        if y_roi is None:
            raise ValueError("y_roi is required for 2-D input.")
        profile = extract_profile(arr, y_roi, axis=axis)
    elif arr.ndim == 1:
        profile = arr
    else:
        raise ValueError(f"image_or_profile must be 1-D or 2-D, got {arr.ndim}-D.")

    # --- Step 2: carrier ---
    car = find_carrier(profile, dx, grating_pitch, band_factor=band_factor)

    # --- Quality flag based on carrier amplitude ---
    if car['k_amp'] >= k_amp_good:
        quality = 'good'
    elif car['k_amp'] >= k_amp_bad:
        quality = 'ok'
    else:
        quality = 'bad'

    # --- Step 3: envelopes ---
    env = extract_envelopes(car['fft_avg'], car['freq'], car['k_peak'],
                            sigma_ratio_0=sigma_ratio_0,
                            sigma_ratio_1=sigma_ratio_1)

    # --- Step 4: centroid ---
    cen = find_phase_centroid(env['I0'], env['A1'],
                              threshold=centroid_threshold)

    # --- Step 5: wavefront ---
    wfr = reconstruct_wavefront(env['complex_1st'], dx, grating_pitch,
                                wavelength, z_gd, x_c=cen['x_c'])

    # --- Step 6: parabolic fit + f_pred ---
    fit = parabolic_focal_fit(wfr['W_rad'], wfr['x_m'], wavelength,
                              mask=cen['beam_mask'])

    # Convenience conversions
    W_nm = wfr['W_rad'] * wavelength / (2 * np.pi) * 1e9
    x_mm = wfr['x_m'] * 1e3

    # In-mask statistics on the raw W (defocus-dominated)
    m = cen['beam_mask']
    rms_W_nm = float(np.std(W_nm[m]))
    pv_W_nm  = float(W_nm[m].max() - W_nm[m].min())

    result = dict(
        # raw inputs / step 1
        profile           = profile,
        # step 2
        k_peak            = car['k_peak'],
        k_amp             = car['k_amp'],
        k_ideal           = car['k_ideal'],
        fft_avg           = car['fft_avg'],
        freq              = car['freq'],
        carrier_band      = car['band'],
        quality           = quality,
        # step 3
        I0                = env['I0'],
        A1                = env['A1'],
        complex_0th       = env['complex_0th'],
        complex_1st       = env['complex_1st'],
        sigma_0           = env['sigma_0'],
        sigma_1           = env['sigma_1'],
        # step 4
        x_c               = cen['x_c'],
        beam_mask         = cen['beam_mask'],
        n_mask_px         = cen['n_mask_px'],
        # step 5
        W_rad             = wfr['W_rad'],
        W_nm              = W_nm,
        x_m               = wfr['x_m'],
        x_mm              = x_mm,
        delta_phi         = wfr['delta_phi'],
        phase_unwrap      = wfr['phase_unwrap'],
        # step 6
        f_pred            = fit['f_pred'],
        a2                = fit['a2'],
        W_para_rad        = fit['W_para_rad'],
        W_resid_rad       = fit['W_resid_rad'],
        rms_resid_nm      = fit['rms_resid_nm'],
        pv_resid_nm       = fit['pv_resid_nm'],
        # raw stats
        rms_W_nm          = rms_W_nm,
        pv_W_nm           = pv_W_nm,
        # passed-through parameters
        dx                = dx,
        grating_pitch     = grating_pitch,
        wavelength        = wavelength,
        z_gd              = z_gd,
    )

    # --- Step 7: optional propagation ---
    if propagate:
        if z_range_m is None:
            half = max(abs(fit['f_pred']), 2.0)
            n_pts = int(half / 0.02) * 2 + 1   # ~20 mm step, odd count
            z_range_m = np.linspace(fit['f_pred'] - half,
                                    fit['f_pred'] + half, n_pts)
        prop = propagate_to_focus(env['A1'], wfr['W_rad'], dx, wavelength,
                                  z_range_m, x_m=wfr['x_m'])
        wff  = wavefront_at_focus(prop, wavelength, dx, threshold=focus_threshold)

        result.update(dict(
            # step 7
            z_range_m       = prop['z_range_m'],
            sigma_rms       = prop['sigma_rms'],
            z_focus         = prop['z_focus'],
            sigma_focus     = prop['sigma_focus'],
            i_focus         = prop['i_focus'],
            E_focus         = prop['E_focus'],
            I_focus         = prop['I_focus'],
            I_caustic       = prop['I_caustic'],
            # step 8
            W_focus_rad     = wff['W_focus_rad'],
            W_focus_nm      = wff['W_focus_nm'],
            focus_mask      = wff['focus_mask'],
            x_c_focus       = wff['x_c_focus'],
            rms_at_focus_nm = wff['rms_at_focus_nm'],
            pv_at_focus_nm  = wff['pv_at_focus_nm'],
        ))

    if verbose:
        msg = (f"[{quality:>4}]  λ={wavelength*1e9:.2f}nm  "
               f"k_peak={car['k_peak']*1e-3:.1f} cyc/mm  "
               f"k_amp={car['k_amp']:.0f}  "
               f"f_pred={fit['f_pred']:+.2f}m  "
               f"RMS_W={rms_W_nm:.1f}nm  RMS_resid={fit['rms_resid_nm']:.3f}nm")
        if propagate:
            msg += (f"  z_focus={prop['z_focus']:+.2f}m  "
                    f"RMS@focus={wff['rms_at_focus_nm']:.3f}nm")
        print(msg)

    return result


def quick_look(
    image_or_profile,
    dx, grating_pitch, wavelength, z_gd,
    *,
    y_roi=None, axis=1,
    propagate=False,
    title='',
    **kwargs,
):
    """One-call beamline workflow: image → reconstruction + diagnostic plot.

    Equivalent to:
        result = reconstruct_single(..., verbose=True, **kwargs)
        plot_result(result, title=title)
        return result

    Parameters
    ----------
    image_or_profile : ndarray
        2-D detector image or 1-D pre-extracted profile.
    dx, grating_pitch, wavelength, z_gd : float
        Physical parameters [m].
    y_roi : slice
        Required if input is 2-D.
    axis : int
        Averaging axis for extract_profile (default 1).
    propagate : bool
        If True, also propagate to find focus (~5–10 s overhead).
    title : str
        Figure suptitle.
    **kwargs
        Any other reconstruct_single kwargs (band_factor, sigma_ratio_0/1,
        centroid_threshold, k_amp_good/bad, z_range_m, ...).

    Returns
    -------
    result : dict
        Full reconstruct_single output dict.
    """
    result = reconstruct_single(
        image_or_profile,
        dx, grating_pitch, wavelength, z_gd,
        y_roi=y_roi, axis=axis,
        propagate=propagate,
        verbose=True,
        **kwargs,
    )
    plot_result(result, title=title)
    return result


# ════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════════════════

def plot_result(result, title='', figsize=None):
    """Diagnostic figure for a single-frame reconstruction.

    Panels (always shown):
        1. Fringe profile (raw)
        2. FFT |spectrum| with carrier and filters
        3. Envelopes I0, |A1|² and beam mask
        4. Wavefront W(x) with parabolic fit
        5. Residual (W − fit) inside beam mask

    Additional panels (if propagation was run):
        6. Caustic |E(x,z)|²
        7. σ_rms(z)
        8. Intensity at focus + wavefront at focus (twin y-axis)

    Parameters
    ----------
    result : dict
        Output of reconstruct_single.
    title : str
        Optional figure suptitle.
    figsize : tuple, optional
        Auto-sized if None.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    has_prop = 'z_focus' in result

    if figsize is None:
        figsize = (14, 14) if has_prop else (14, 9)

    fig = plt.figure(figsize=figsize)
    n_rows = 5 if has_prop else 3
    gs = fig.add_gridspec(n_rows, 2, hspace=0.55, wspace=0.25)

    x_mm  = result['x_mm']
    N     = len(result['profile'])
    freq  = result['freq']
    m     = result['beam_mask']
    lam   = result['wavelength']

    # ─── Panel 1: raw fringe profile ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    x_full_mm = np.arange(N) * result['dx'] * 1e3
    ax.plot(x_full_mm, result['profile'], lw=0.7, color='steelblue')
    ax.axvline(x_full_mm[result['x_c']], color='C3', ls=':', lw=1,
               label=f"x_c (px {result['x_c']})")
    ax.set_xlabel('Position [mm]')
    ax.set_ylabel('Intensity [a.u.]')
    ax.set_title('Fringe profile')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 2: FFT spectrum with carrier ──────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    ax.semilogy(freq * 1e-3, np.abs(result['fft_avg']), lw=0.8,
                color='steelblue', label='|FFT|')
    ax.axvspan(result['carrier_band'][0]*1e-3, result['carrier_band'][1]*1e-3,
               color='gold', alpha=0.2, label='search band')
    ax.axvline(result['k_ideal']*1e-3, color='red',    ls=':', lw=1,
               label=f"k_ideal={result['k_ideal']*1e-3:.1f}")
    ax.axvline(result['k_peak']*1e-3,  color='tomato', ls='--', lw=1.2,
               label=f"k_peak={result['k_peak']*1e-3:.1f}")
    ax.set_xlabel('Spatial frequency [cyc/mm]')
    ax.set_ylabel('|FFT| (log)')
    ax.set_title(f"FFT spectrum  (quality: {result['quality']}, "
                 f"k_amp={result['k_amp']:.0f})")
    ax.set_xlim(-result['k_ideal']*1e-3, 3*result['k_ideal']*1e-3)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    # ─── Panel 3: envelopes & beam mask ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    I0_n   = result['I0'] / result['I0'].max()
    A1sq_n = (result['A1']**2) / (result['A1']**2).max()
    ax.plot(x_mm, I0_n,   'C0', lw=1.3, label=r'$I_0$ (norm)')
    ax.plot(x_mm, A1sq_n, 'C3', lw=1.3, label=r'$|A_{+1}|^2$ (norm)')
    ax.fill_between(x_mm, 0, 1.05, where=m, color='gold', alpha=0.15,
                    label='beam_mask')
    ax.axvline(0, color='C3', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel('Position [mm] (x=0 at centroid)')
    ax.set_ylabel('Norm intensity')
    ax.set_title(f"Envelopes & beam mask  ({result['n_mask_px']} px)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 4: wavefront + parabolic fit ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    W_para_nm   = result['W_para_rad'] * lam / (2*np.pi) * 1e9
    W_plot      = np.where(m, result['W_nm'], np.nan)
    W_para_plot = np.where(m, W_para_nm,      np.nan)
    ax.plot(x_mm, result['W_nm'], 'C0', lw=0.6, alpha=0.3, label='W (outside)')
    ax.plot(x_mm, W_plot,         'C0', lw=1.5, label='W measured')
    ax.plot(x_mm, W_para_plot,    'C3', lw=1.2, ls='--',
            label=f"parabolic (f={result['f_pred']:+.2f} m)")
    if m.any():
        lo, hi = result['W_nm'][m].min(), result['W_nm'][m].max()
        rng = hi - lo if hi > lo else 1.0
        ylo, yhi = lo - 0.15*rng, hi + 0.15*rng
        ax.set_ylim(ylo, yhi)
        ax.fill_between(x_mm, ylo, yhi, where=m, color='gold', alpha=0.12)
        # X-limit: ±1.5 × beam_mask half-width
        x_beam   = x_mm[m]
        x_half   = (x_beam.max() - x_beam.min()) / 2
        x_center = (x_beam.max() + x_beam.min()) / 2
        ax.set_xlim(x_center - 1.5*x_half, x_center + 1.5*x_half)
    ax.axvline(0, color='gray', ls=':', lw=0.7, alpha=0.6)
    ax.set_xlabel('Position [mm]')
    ax.set_ylabel('W [nm OPL]')
    ax.set_title('Wavefront')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 5: residual (defocus removed) ─────────────────────────────────
    ax = fig.add_subplot(gs[2, :])
    W_resid_nm   = result['W_resid_rad'] * lam / (2*np.pi) * 1e9
    W_resid_plot = np.where(m, W_resid_nm, np.nan)
    ax.plot(x_mm, W_resid_plot, 'C4', lw=1.4)
    ax.axhline(0, color='gray', lw=0.7, alpha=0.5)
    ax.set_xlabel('Position [mm]')
    ax.set_ylabel('Residual [nm]')
    ax.set_title('Residual (defocus removed)')
    if m.any():
        lo, hi = W_resid_nm[m].min(), W_resid_nm[m].max()
        rng = max(hi - lo, 0.01)
        ax.set_ylim(lo - 0.2*rng, hi + 0.2*rng)
    ax.grid(alpha=0.3)

    # ─── Propagation panels (if available) ───────────────────────────────────
    if has_prop:
        z   = result['z_range_m']
        m_f = result['focus_mask']

        # Panel 6: caustic (full width)
        ax = fig.add_subplot(gs[3, :])
        I_norm = result['I_caustic'] / result['I_caustic'].max(axis=0, keepdims=True)
        im = ax.imshow(I_norm,
                       extent=(z[0], z[-1], x_mm[0], x_mm[-1]),
                       aspect='auto', origin='lower', cmap='inferno')
        ax.axvline(result['z_focus'], color='cyan', lw=1.2, ls='--',
                   label=f"z_focus = {result['z_focus']:+.2f} m")
        ax.axvline(result['f_pred'], color='lime',  lw=1.0, ls=':',
                   label=f"f_pred  = {result['f_pred']:+.2f} m")
        ax.set_xlabel('z [m]')
        ax.set_ylabel('x [mm]')
        ax.set_title('Caustic |E(x,z)|² (per-column normalized)')
        if m.any():
            x_beam = x_mm[m]
            ax.set_ylim(x_beam.min()*1.2, x_beam.max()*1.2)
        ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

        # Panel 7: σ_rms(z)
        ax = fig.add_subplot(gs[4, 0])
        ax.plot(z, result['sigma_rms']*1e6, 'C0', lw=1.4)
        ax.axvline(result['z_focus'], color='cyan', lw=1, ls='--')
        ax.scatter([result['z_focus']], [result['sigma_focus']*1e6],
                   color='red', s=40, zorder=5,
                   label=f"σ_min = {result['sigma_focus']*1e6:.1f} µm")
        ax.set_xlabel('z [m]')
        ax.set_ylabel('σ_rms [µm]')
        ax.set_title('Beam size vs z')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 8: focus intensity + W at focus (twin axis)
        ax = fig.add_subplot(gs[4, 1])
        I_focus_norm = result['I_focus'] / result['I_focus'].max()
        ax.plot(x_mm, I_focus_norm, 'C1', lw=1.4, label='I (norm)')
        ax.fill_between(x_mm, 0, 1.05, where=m_f, color='gold', alpha=0.15)
        ax.set_xlabel('x [mm]')
        ax.set_ylabel('Norm intensity', color='C1')
        ax.tick_params(axis='y', labelcolor='C1')
        ax.set_ylim(0, 1.05)
        # X-limit: ±1.5 × focus_mask half-width
        if m_f.any():
            x_f      = x_mm[m_f]
            x_half   = (x_f.max() - x_f.min()) / 2
            x_center = (x_f.max() + x_f.min()) / 2
            ax.set_xlim(x_center - 1.5*x_half, x_center + 1.5*x_half)

        # Wavefront on twin axis
        ax2 = ax.twinx()
        W_focus_plot = np.where(m_f, result['W_focus_nm'], np.nan)
        ax2.plot(x_mm, W_focus_plot, 'C4', lw=1.3, label='W at focus')
        ax2.set_ylabel('W at focus [nm]', color='C4')
        ax2.tick_params(axis='y', labelcolor='C4')
        if m_f.any():
            lo, hi = result['W_focus_nm'][m_f].min(), result['W_focus_nm'][m_f].max()
            rng = max(hi - lo, 0.01)
            ax2.set_ylim(lo - 0.3*rng, hi + 0.3*rng)

        ax.set_title('Focus')
        ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=1)
    fig.subplots_adjust(top=0.95, bottom=0.05, left=0.07, right=0.96)
    plt.show()
    return fig
