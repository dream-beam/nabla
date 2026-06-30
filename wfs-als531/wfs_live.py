"""
wfs_live.py
───────────

X-ray wavefront sensor analysis pipeline for live beamline use.

Author       : Wei "Francis" He (francisho@lbl.gov / whorwhey@gmail.com)
Created      : May 2026
Last updated : 2026-06-28

Reconstructs the X-ray wavefront from a single shearing-interferometer
image using grating-based lateral shearing interferometry. The pipeline
takes a 2-D detector image (Talbot-plane interferogram), extracts the
+1-order phase, removes the carrier, integrates to obtain W(x), and
optionally propagates the reconstructed field to locate the focus.
Works for both soft and hard X-rays; the only energy-dependent
quantities are the wavelength and the optimal grating-to-detector
distance.

Pipeline
--------
For a single detector image / 1-D fringe profile, the full workflow is::

    image  → extract_profile         →  1-D profile
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

    from wfs_live import quick_look

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

Acknowledgements
----------------
This pipeline builds on work and discussions with:
    - Dr. Antoine İşlegen-Wojdyla (awojdyla@lbl.gov)
    - Dr. Xiaoya Chong
    - Dr. Ka Hung (Henry) Chan
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

import monoplus as mp   # in-house Fresnel propagator (propTF, secondmomt)

import warnings

# ─── Helpers ────────────────────────────────────────────────────────────────
def energy_eV_to_wavelength_m(energy_eV):
    """Convert photon energy [eV] to wavelength [m]."""
    return 1.23984e-6 / np.asarray(energy_eV, dtype=float)


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


def compute_visibility_masks(I0, A1, V_threshold=0.4, I0_threshold=0.02,
                             beam_mask=None):
    """Compute normalized visibility V and the trust mask V > V_threshold.

    Single source of truth for V/trust_mask construction. Used by both
    parabolic_focal_fit (via fit_mask = beam_mask ∩ trust_mask) and
    build_caustic_amplitude (gated-mode amplitude support).

    V is computed inside safety_mask (an I0 floor guarding division by 
    tiny intensities), then normalized to its peak inside beam_mask if 
    provided, else inside safety_mask.

    When `beam_mask` is provided, raw (V > V_threshold) is intersected with 
    it before fill_mask_gaps, preventing out-of-beam spikes from anchoring 
    the fill. Recommended whenever a beam_mask is available.

    Parameters
    ----------
    I0, A1 : ndarray (N,)
        0th-order intensity envelope and +1-order fringe amplitude
        from extract_envelopes.
    V_threshold : float
        Fraction of peak visibility above which fringes are trusted.
    I0_threshold : float
        Fraction of max(I0) defining safety_mask (V-computation floor).
    beam_mask : ndarray (N,) of bool, optional
        Main-beam support (from find_phase_centroid). If provided, fills
        are anchored only to True points inside beam_mask.

    Returns
    -------
    V           : ndarray (N,)         Normalized visibility (0 outside safety_mask)
    trust_mask  : ndarray (N,) of bool Gap-filled (V > V_threshold), optionally
                                       constrained to beam_mask before fill.
    safety_mask : ndarray (N,) of bool Gap-filled (I0 > I0_threshold * max)
    """
    I0 = np.asarray(I0, dtype=float)
    A1 = np.asarray(A1, dtype=float)

    safety_mask = fill_mask_gaps(I0 > I0_threshold * I0.max())
    V = np.zeros_like(I0)
    V[safety_mask] = 2 * A1[safety_mask] / I0[safety_mask]

    # Normalize to in-beam peak when available; else fall back to safety_mask.
    norm_mask = beam_mask if beam_mask is not None else safety_mask
    if V[norm_mask].size and V[norm_mask].max() > 0:
        V /= V[norm_mask].max()
    raw_trust = V > V_threshold
    if beam_mask is not None:
        raw_trust = raw_trust & beam_mask
    trust_mask = fill_mask_gaps(raw_trust)
    return V, trust_mask, safety_mask


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


def find_carrier(profile, dx, grating_pitch, band_factor=1.5, plot=False):
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
        Search band is (1/p / band_factor,  1/p * band_factor).
        Set higher (e.g. 1.5–2.0) if strong tilt pulls the peak far from 1/p.
    plot : bool, optional (default False)
        If True, calls plot_carrier on the returned dict for a quick
        diagnostic figure.
    
    Returns
    -------
    out : dict
        k_peak     [cyc/m]  measured carrier frequency
        k_amp      [a.u.]   |FFT| at the carrier — quality indicator
        k_ideal    [cyc/m]  1/grating_pitch
        fft_avg    ndarray  centered FFT of the profile (complex)
        freq       ndarray  centered frequency axis [cyc/m]
        band       (lo, hi) the search band actually used [cyc/m]
        dc_amp     [a.u.]   |FFT| at zero frequency
        bg_amp     [a.u.]   mean |FFT| over [1.2, 1.8] × k_peak
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

    # Band-edge proximity check: warn if k_peak lands within 5% of k_ideal
    # from either edge of the search band — usually a sign band_factor is too small.
    margin = 0.05 * k_ideal
    if (k_peak - band_lo) < margin or (band_hi - k_peak) < margin:
        warnings.warn(
            f"k_peak={k_peak*1e-3:.1f} cyc/mm landed within "
            f"5% of k_ideal of a search-band edge "
            f"[{band_lo*1e-3:.1f}, {band_hi*1e-3:.1f}] cyc/mm. "
            f"Consider increasing band_factor (currently {band_factor}).",
            stacklevel=2,
        )

    # DC amplitude and band background for quality metrics (Phase G).
    # bg_amp: mean |F| over [1.2*k_peak, 1.8*k_peak], avoiding the carrier
    # and its 2nd harmonic.
    dc_amp = float(np.abs(fft_avg[freq == 0])[0])
    bg_mask = (freq >= 1.2 * k_peak) & (freq <= 1.8 * k_peak)
    bg_amp = float(np.abs(fft_avg[bg_mask]).mean()) if bg_mask.any() else float('nan')

    out = dict(
        k_peak  = k_peak,
        k_amp   = k_amp,
        k_ideal = k_ideal,
        dc_amp  = dc_amp,
        bg_amp  = bg_amp,
        fft_avg = fft_avg,
        freq    = freq,
        band    = (band_lo, band_hi),
    )
    if plot:
        plot_carrier(out)
    return out


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
        I0          : ndarray (N,)           |complex_0th| — intensity envelope a(x)
        A1          : ndarray (N,)           |complex_1st| — intensity envelope b/2
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
    I0          = np.abs(complex_0th)        # linear magnitude (Decision A)
    
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


def build_caustic_amplitude(
    I0, A1,
    mode='field_envelope',
    V_threshold=0.4,
    I0_threshold=0.02,
    V=None,
    trust_mask=None,
    safety_mask=None,
    beam_mask=None,
    debug=False,
    plot=None,
):
    """Construct the amplitude A(x) for propagate_to_focus.

    A(x) = √I₀(x) in 'field_envelope' mode (smooth physical envelope).
    A(x) = √I₀(x) · trust_mask in 'gated' mode (zeros wings outside
    fringe coverage; produces a sharp aperture penalty on propagation
    but correctly excludes off-axis intensity humps).

    Parameters
    ----------
    I0, A1 : ndarray (N,)
        Envelopes from extract_envelopes.
    mode : {'field_envelope', 'gated'}
        Amplitude construction mode.
    V_threshold : float, default 0.4
        Fraction of peak normalized visibility above which fringes are
        trusted. Used only if trust_mask is computed here.
    I0_threshold : float, default 0.02
        I0 floor for V computation (passed to compute_visibility_masks).
        Used only if V/trust_mask/safety_mask are computed here.
    V, trust_mask, safety_mask : ndarray, optional
        Precomputed outputs of compute_visibility_masks. If any is None,
        all three are recomputed internally. Pass these from
        reconstruct_single to avoid redundant computation.
    beam_mask : ndarray (N,) of bool, optional
        Main-beam support. Used only when V/trust_mask/safety_mask are
        recomputed here; passed through to compute_visibility_masks to
        prevent out-of-beam anchors from corrupting trust_mask.
    debug : bool, optional (default False)
        If True, return a dict containing A plus diagnostic arrays
        (V, safety_mask, trust_mask, mode_used) instead of just A.
    plot : bool or None, optional (default None)
        If True, show plot_caustic_amplitude figure. If False, never plot.
        If None, default is True when mode='gated' and False when
        mode='field_envelope'. Set plot=False explicitly to suppress
        when calling from a loop or orchestrator.

    Returns
    -------
    A : ndarray, shape (N,)
        Returned when debug=False (current behavior).
    debug_dict : dict
        Returned when debug=True. Keys: A, V, safety_mask, trust_mask,
        mode_used, I0, A1, V_threshold.
    """
    I0 = np.asarray(I0, dtype=float)
    A1 = np.asarray(A1, dtype=float)

    if plot is None:
        plot = (mode == 'gated')

    if V is None or trust_mask is None or safety_mask is None:
        V, trust_mask, safety_mask = compute_visibility_masks(
            I0, A1, V_threshold=V_threshold, I0_threshold=I0_threshold,
            beam_mask=beam_mask
        )

    if mode == 'field_envelope':
        A = np.sqrt(np.clip(I0, 0, None))
    elif mode == 'gated':
        A = np.sqrt(np.clip(I0, 0, None)) * trust_mask
    else:
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'field_envelope' or 'gated'."
        )

    if debug or plot:
        debug_dict = dict(
            A           = A,
            V           = V,
            safety_mask = safety_mask,
            trust_mask  = trust_mask,
            mode_used   = mode,
            I0          = I0,
            A1          = A1,
            V_threshold = V_threshold,
        )
        if plot:
            plot_caustic_amplitude(debug_dict)
        if debug:
            return debug_dict

    return A


def find_phase_centroid(I0, A1, threshold=None, fill_gaps=True):
    """Find the beam centroid for use as phase reference.

    Builds a beam mask from I0 > threshold * max(I0), optionally fills
    interior gaps, then computes the centroid weighted by A1 inside that
    mask.

    Parameters
    ----------
    I0 : array_like, shape (N,)
        0th-order beam intensity (from extract_envelopes).
    A1 : array_like, shape (N,)
        +1-order fringe amplitude (from extract_envelopes).
    threshold : float, optional (default 1/e² ≈ 0.135)
        Fraction of I0 peak above which pixels are kept in the beam mask.
        I0 is the linear intensity envelope, so this selects the 1/e²
        intensity radius (standard beam-radius convention).
    fill_gaps : bool, optional (default True)
        If True, close interior holes in the raw threshold mask.
        Set False when off-axis structure above threshold would be
        incorrectly bridged into the main beam by gap-fill.

    Returns
    -------
    out : dict
        x_c          : int        centroid pixel index
        beam_mask    : ndarray    boolean mask (N,), gap-filled if fill_gaps=True
        threshold    : float      threshold actually used
        n_mask_px    : int        number of True pixels in mask
        n_filled_px  : int        False→True conversions by gap-fill (0 if fill_gaps=False)
    """
    if threshold is None:
        threshold = np.exp(-2)   # 1/e² ≈ 0.135

    I0 = np.asarray(I0, dtype=float)
    A1 = np.asarray(A1, dtype=float)

    raw_mask = I0 > threshold * I0.max()
    if fill_gaps:
        beam_mask = fill_mask_gaps(raw_mask)
    else:
        beam_mask = raw_mask

    if not beam_mask.any():
        raise ValueError(
            f"Beam mask is empty at threshold={threshold:.3g}. "
            f"Check I0 (max={I0.max():.3g}) or lower threshold."
        )

    weights = A1[beam_mask]        # linear A1 weighting (Decision F)
    x_idx   = np.arange(len(I0))
    x_c     = int(np.average(x_idx[beam_mask], weights=weights))

    return dict(
        x_c         = x_c,
        beam_mask   = beam_mask,
        threshold   = threshold,
        n_mask_px   = int(beam_mask.sum()),
        n_filled_px = int(beam_mask.sum() - raw_mask.sum()),  # 0 when fill_gaps=False
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
        Pixel used as the phase and wavefront reference: Δφ and W are both
        zero here, and x_m = 0 here. Pass a fixed value across frames for
        gauge-consistent comparisons within a scan series.

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

    # 4. Integrate (negative cumsum; flip if your +1/−1 order differs)
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


def parabolic_focal_fit(W_rad, x_m, wavelength, mask=None, n_min_fit=50):
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
    n_min_fit : int, optional (default 50)
        Soft warning threshold on mask size. Below 3, raises ValueError
        (polyfit ill-posed). Between 3 and n_min_fit, emits a
        RuntimeWarning; the fit still runs but may be unreliable.

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

    n_fit = int(mask.sum())
    if n_fit < 3:
        raise ValueError(
            f"parabolic_focal_fit: mask has {n_fit} pixels, need ≥3 for a "
            f"quadratic fit. Likely V_threshold too high, no fringes "
            f"detected, or empty beam_mask."
        )
    if n_fit < n_min_fit:
        warnings.warn(
            f"parabolic_focal_fit: mask has only {n_fit} pixels "
            f"(< n_min_fit={n_min_fit}). Fit may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    a2, a1, a0 = np.polyfit(x_m[mask], W_rad[mask], 2)

    W_para_rad  = a2 * x_m**2 + a1 * x_m + a0
    W_resid_rad = W_rad - W_para_rad

    k = 2 * np.pi / wavelength
    f_pred = -k / (2 * a2)

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


def propagate_to_focus(
    A, W_rad, dx, wavelength, z_range_m,
    x_m=None,
    f_pred_m=None,
    focus_search_halfwidth_m=None,
    focus_locator='fwhm_min',
):
    """Fresnel-propagate the reconstructed field and locate the focus.

    Builds E_det(x) = A(x) · exp[i W(x)], propagates over z_range_m via
    monoplus.propTF, and returns the caustic and a chosen focal plane.

    Two diagnostic curves are computed across the full z_range:
        sigma_rms(z)   — second moment of |E(x,z)|², wing-sensitive.
        fwhm_z(z)      — from monoplus.fwhm, returned in σ-equivalent
                         units (FWHM / 2.35), so directly comparable
                         to sigma_rms numerically. Less wing-sensitive,
                         but noisy in regions with fragmented intensity.

    Focus localization is restricted to a window centered on f_pred_m,
    width 2 × focus_search_halfwidth_m. This prevents off-axis wing
    structure from pulling the minimum to a spurious upstream location.
    If the minimum lands on the window boundary, `focus_at_edge=True`
    is returned along with a RuntimeWarning — likely a too-tight window.

    Parameters
    ----------
    A : array_like (N,)
        Detector-plane field amplitude (from build_caustic_amplitude).
    W_rad : array_like (N,)  [rad]
        Reconstructed wavefront.
    dx : float  [m]
        Pixel pitch.
    wavelength : float  [m]
    z_range_m : array_like  [m]
        Propagation distances to evaluate.
    x_m : array_like (N,) [m], optional
        Spatial coordinate. Defaults to (np.arange(N) - N//2) * dx.
    f_pred_m : float [m], optional
        Geometric focus from parabolic_focal_fit. Centers the bounded
        search window. If None, falls back to global argmin on the
        chosen locator curve with a warning.
    focus_search_halfwidth_m : float [m], optional
        Half-width of the search window around f_pred_m. If None,
        auto-computed as max(0.15 * |f_pred_m|, 0.05 m).
    focus_locator : {'fwhm_min', 'sigma_min', 'geometric'}, default 'fwhm_min'
        Curve to minimize for z_focus.

    Returns
    -------
    out : dict
        E_caustic        : ndarray (N, Z) complex
        I_caustic        : ndarray (N, Z)
        sigma_rms        : ndarray (Z,) [m]
        fwhm_z           : ndarray (Z,) [m]   FWHM/2.35 (σ-equivalent)
        z_range_m        : ndarray (Z,) [m]
        z_focus          : float [m]
        sigma_focus      : float [m]
        fwhm_focus       : float [m]
        i_focus          : int
        z_focus_locator  : str
        focus_at_edge    : bool
        E_focus, I_focus : ndarray (N,)
    """
    if focus_locator not in ('sigma_min', 'fwhm_min', 'geometric'):
        raise ValueError(
            f"Unknown focus_locator: {focus_locator!r}. "
            f"Use 'sigma_min', 'fwhm_min', or 'geometric'."
        )

    if focus_search_halfwidth_m is None:
        focus_search_halfwidth_m = (max(0.15 * abs(f_pred_m), 0.05)
                                    if f_pred_m is not None else 0.0)

    N = len(A)
    if x_m is None:
        x_m = (np.arange(N) - N // 2) * dx
    L_m = N * dx

    # Detector-plane field, propagated over z_range_m
    E_det = A * np.exp(1j * W_rad)
    z_range_m = np.asarray(z_range_m, dtype=float)
    Z = len(z_range_m)

    E_caustic = np.zeros((N, Z), dtype=complex)
    for i_z, z in enumerate(z_range_m):
        E_caustic[:, i_z] = mp.propTF(E_det, L_m, wavelength, z)

    I_caustic = np.abs(E_caustic)**2

    # Diagnostic curves — both computed across the full range.
    # Outside the focus search window, fwhm_z may be unreliable on
    # wing-dominated frames (multi-peak structure inflates the metric),
    # but the values are kept for plotting and post-hoc inspection.
    sigma_rms = mp.secondmomt(Z, x_m, E_caustic)
    fwhm_z    = mp.fwhm(Z, x_m, E_caustic)

    # --- Focus localization ---
    if focus_locator == 'geometric':
        if f_pred_m is None:
            raise ValueError(
                "focus_locator='geometric' requires f_pred_m to be provided."
            )
        i_focus = int(np.argmin(np.abs(z_range_m - f_pred_m)))
        z_focus = float(z_range_m[i_focus])
        focus_at_edge = (i_focus == 0) or (i_focus == Z - 1)
        if abs(z_focus - f_pred_m) > 0.5 * focus_search_halfwidth_m:
            warnings.warn(
                f"propagate_to_focus: f_pred_m={f_pred_m:+.3f} m is far from "
                f"the nearest z in z_range_m ({z_focus:+.3f} m). "
                f"Consider widening z_range_m around f_pred.",
                RuntimeWarning, stacklevel=2,
            )

    else:
        # 'sigma_min' or 'fwhm_min'
        curve = sigma_rms if focus_locator == 'sigma_min' else fwhm_z

        if f_pred_m is None:
            # No anchor: fall back to global sigma_min, regardless of request.
            warnings.warn(
                "propagate_to_focus: f_pred_m not provided; using global "
                "argmin on sigma_rms. Pass f_pred_m to enable bounded "
                "fwhm_min.",
                RuntimeWarning, stacklevel=2,
            )
            i_focus = int(np.argmin(sigma_rms))
            focus_at_edge = (i_focus == 0) or (i_focus == Z - 1)
        else:
            idx_window = np.where(
                (z_range_m >= f_pred_m - focus_search_halfwidth_m) &
                (z_range_m <= f_pred_m + focus_search_halfwidth_m)
            )[0]
            if len(idx_window) == 0:
                warnings.warn(
                    f"propagate_to_focus: search window "
                    f"[{f_pred_m - focus_search_halfwidth_m:+.3f}, "
                    f"{f_pred_m + focus_search_halfwidth_m:+.3f}] m "
                    f"does not overlap z_range_m. Falling back to geometric.",
                    RuntimeWarning, stacklevel=2,
                )
                i_focus = int(np.argmin(np.abs(z_range_m - f_pred_m)))
                focus_at_edge = True
            else:
                sub_curve = curve[idx_window]
                if np.all(np.isnan(sub_curve)):
                    warnings.warn(
                        f"propagate_to_focus: {focus_locator} curve is all "
                        f"NaN inside the search window. Falling back to "
                        f"geometric.",
                        RuntimeWarning, stacklevel=2,
                    )
                    i_focus = int(np.argmin(np.abs(z_range_m - f_pred_m)))
                    focus_at_edge = True
                else:
                    sub_min = int(np.nanargmin(sub_curve))
                    i_focus = int(idx_window[sub_min])
                    focus_at_edge = (sub_min == 0) or \
                                    (sub_min == len(sub_curve) - 1)
                    if focus_at_edge:
                        warnings.warn(
                            f"propagate_to_focus: {focus_locator} minimum "
                            f"landed on the search window boundary "
                            f"(z={z_range_m[i_focus]:+.3f} m, window "
                            f"±{focus_search_halfwidth_m:.3f} m around "
                            f"f_pred={f_pred_m:+.3f} m).",
                            RuntimeWarning, stacklevel=2,
                        )

        z_focus = float(z_range_m[i_focus])

    sigma_focus = float(sigma_rms[i_focus])
    fwhm_focus  = float(fwhm_z[i_focus]) if not np.isnan(fwhm_z[i_focus]) else np.nan

    return dict(
        E_caustic       = E_caustic,
        I_caustic       = I_caustic,
        sigma_rms       = sigma_rms,
        fwhm_z          = fwhm_z,
        z_range_m       = z_range_m,
        z_focus         = z_focus,
        sigma_focus     = sigma_focus,
        fwhm_focus      = fwhm_focus,
        i_focus         = i_focus,
        z_focus_locator = focus_locator,
        focus_search_halfwidth_m = focus_search_halfwidth_m,
        focus_at_edge   = focus_at_edge,
        E_focus         = E_caustic[:, i_focus],
        I_focus         = I_caustic[:, i_focus],
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
    fill_gaps=True,
    propagate=False,
    z_range_m=None,
    focus_threshold=None,
    caustic_amplitude='field_envelope',
    V_threshold=0.4,
    I0_threshold=0.02,
    focus_search_halfwidth_m=None,
    focus_locator='fwhm_min',
    r_good=50, r_bad=200, 
    verbose=False,
    x_c_override=None,
):
    """Full single-frame WFS pipeline: image → wavefront (+ optional propagation).

    Steps:
        1. extract_profile (if 2-D input)
        2. find_carrier
        3. extract_envelopes
        4. find_phase_centroid
        5. reconstruct_wavefront
        6. parabolic_focal_fit  (may soft-fail; pipeline continues with NaN
                                 sentinels and fit_status='fail')
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
    fill_gaps : bool
        Whether find_phase_centroid gap-fills the beam mask (default True).
        Set False to disable gap-fill when off-axis structure would be
        incorrectly bridged into the main beam (Stage B off-axis hump case).
    propagate : bool
        If True, run steps 7–8 (propagation + W at focus).
    z_range_m : array_like, optional
        Custom z range for propagation. If None, auto-set from f_pred.
    focus_threshold : float
        Focus mask threshold (default 1/e²).
    caustic_amplitude : {'field_envelope', 'gated'}
        Mode for build_caustic_amplitude (default 'field_envelope').
        See that function for mode semantics.
    V_threshold, I0_threshold : float
        Forwarded to build_caustic_amplitude when caustic_amplitude='gated'.
    focus_search_halfwidth_m : float [m], optional
        Half-width of the bounded search window for focus localization.
        z_focus is selected as the argmin of the chosen locator curve
        within [f_pred - halfwidth, f_pred + halfwidth]. If None
        (default), auto-computed as max(0.15 * |f_pred|, 0.05) m.
    focus_locator : {'sigma_min', 'fwhm_min', 'geometric'}, default 'fwhm_min'
        Curve to minimize for z_focus. 'sigma_min' uses σ_rms(z),
        'fwhm_min' uses FWHM(z), 'geometric' skips the search and
        sets z_focus = f_pred.
    r_good, r_bad : float
        Quality thresholds on the DC/carrier amplitude ratio
        r = |F(0)| / |F(k_peak)|. Lower r means stronger fringes
        relative to the unmodulated background.
        - r < r_good           → 'good'
        - r_good ≤ r < r_bad   → 'ok'
        - r ≥ r_bad            → 'bad' (likely near-focus, fringes lost)
        Defaults (50, 200) are working values pending degraded-frame
        calibration.
    verbose : bool
        Print a one-line summary.
    x_c_override : int or None, optional
        Fix the phase and wavefront reference pixel for all frames in a scan
        series. The per-frame centroid is still computed (for beam_mask), but
        x_c_override is used as the gauge anchor instead. result['x_c'] always
        reports the value actually used. Default None uses the per-frame
        centroid.

    Returns
    -------
    result : dict
        Flat dict containing keys from all pipeline stages, plus:
        - quality          : 'good' / 'ok' / 'bad'  (from carrier r_DC)
        - fit_status       : 'ok' / 'fail'          (parabolic fit outcome)
        - fit_fail_reason  : str                    (empty on 'ok')
        - W_nm             : reconstructed W in nm OPL (convenience)
        - x_mm             : x_m in mm (convenience)

        On fit failure, all step-6 numeric fields (f_pred, a2, rms_resid_nm,
        pv_resid_nm) are NaN and W_para_rad / W_resid_rad are NaN-filled
        arrays of the input shape. Propagation is skipped regardless of
        the `propagate` flag; step-7/8 fields are absent from the result
        dict. All upstream fields (W_rad, phase_unwrap, delta_phi,
        envelopes, carrier stats, masks, quality) remain valid.
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

    # --- Quality flag based on DC/carrier amplitude ratio ---
    dc_carrier_ratio = car['dc_amp'] / car['k_amp']
    carrier_snr      = car['k_amp']  / car['bg_amp']
    if dc_carrier_ratio < r_good:
        quality = 'good'
    elif dc_carrier_ratio < r_bad:
        quality = 'ok'
    else:
        quality = 'bad'

    # --- Step 3: envelopes ---
    env = extract_envelopes(car['fft_avg'], car['freq'], car['k_peak'],
                            sigma_ratio_0=sigma_ratio_0,
                            sigma_ratio_1=sigma_ratio_1)

    # --- Step 4: centroid ---
    cen = find_phase_centroid(env['I0'], env['A1'],
                              threshold=centroid_threshold,
                              fill_gaps=fill_gaps)

    # --- Step 5: wavefront ---
    x_c = x_c_override if x_c_override is not None else cen['x_c']
    wfr = reconstruct_wavefront(env['complex_1st'], dx, grating_pitch,
                                wavelength, z_gd, x_c=x_c)
    # --- Prepare fit_mask: beam_mask ∩ trust_mask ---
    # Trust mask excludes off-axis humps (high I0 but low V) that would
    # otherwise pollute the parabolic fit. Reused by Step 7 (caustic).
    # beam_mask is passed to constrain fill_mask_gaps anchors to the main beam.
    V, trust_mask, safety_mask = compute_visibility_masks(
        env['I0'], env['A1'],
        V_threshold=V_threshold, I0_threshold=I0_threshold,
        beam_mask=cen['beam_mask'],
    )
    fit_mask = cen['beam_mask'] & trust_mask

    # --- Step 6: parabolic fit + f_pred ---
    # Soft-fail: parabolic_focal_fit raises ValueError when fit_mask has <3
    # pixels. Catch it so the pipeline still returns upstream results
    # (W_rad, phase_unwrap, envelopes, carrier stats). Sentinel values are
    # NaN to avoid any false-success appearance downstream.
    try:
        fit = parabolic_focal_fit(wfr['W_rad'], wfr['x_m'], wavelength,
                                  mask=fit_mask)
        fit_status, fit_fail_reason = 'ok', ''
    except ValueError as exc:
        nan_arr = np.full_like(wfr['W_rad'], np.nan)
        fit = dict(
            f_pred       = np.nan,
            a2           = np.nan,
            a1           = np.nan,
            a0           = np.nan,
            W_para_rad   = nan_arr,
            W_resid_rad  = nan_arr,
            rms_resid_nm = np.nan,
            pv_resid_nm  = np.nan,
        )
        fit_status, fit_fail_reason = 'fail', str(exc)

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
        dc_amp            = car['dc_amp'],
        bg_amp            = car['bg_amp'],
        dc_carrier_ratio  = dc_carrier_ratio,
        carrier_snr       = carrier_snr,
        quality           = quality,
        # step 3
        I0                = env['I0'],
        A1                = env['A1'],
        complex_0th       = env['complex_0th'],
        complex_1st       = env['complex_1st'],
        sigma_0           = env['sigma_0'],
        sigma_1           = env['sigma_1'],
        # step 4
        x_c               = x_c,
        beam_mask         = cen['beam_mask'],
        n_mask_px         = cen['n_mask_px'],
        # fit-mask preparation (used by step 6 and step 7)
        V                 = V,
        trust_mask        = trust_mask,
        safety_mask       = safety_mask,
        fit_mask          = fit_mask,
        n_fit_px          = int(fit_mask.sum()),
        V_threshold       = V_threshold,
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
        fit_status        = fit_status,
        fit_fail_reason   = fit_fail_reason,
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
    # Skipped on fit-fail (fit_mask empty → no coherent trust region to
    # propagate). Caller must check fit_status before consuming step-7 fields.
    if propagate and fit_status == 'ok':
        if z_range_m is None:
            half = max(abs(fit['f_pred']), 2.0)
            n_pts = int(half / 0.02) * 2 + 1   # ~20 mm step, odd count
            z_range_m = np.linspace(fit['f_pred'] - half,
                                    fit['f_pred'] + half, n_pts)
        A_caustic = build_caustic_amplitude(
            env['I0'], env['A1'],
            mode=caustic_amplitude,
            V_threshold=V_threshold, I0_threshold=I0_threshold,
            V=V, trust_mask=trust_mask, safety_mask=safety_mask,
            plot=False,
        )
        prop = propagate_to_focus(
            A_caustic, wfr['W_rad'], dx, wavelength, z_range_m,
            x_m=wfr['x_m'],
            f_pred_m=fit['f_pred'],
            focus_search_halfwidth_m=focus_search_halfwidth_m,
            focus_locator=focus_locator,
        )
        wff = wavefront_at_focus(prop, wavelength, dx,
                                 threshold=focus_threshold)

        result.update(dict(
            # step 7
            A_caustic       = A_caustic,
            caustic_mode    = caustic_amplitude,
            z_range_m       = prop['z_range_m'],
            sigma_rms       = prop['sigma_rms'],
            fwhm_z          = prop['fwhm_z'],
            z_focus         = prop['z_focus'],
            sigma_focus     = prop['sigma_focus'],
            fwhm_focus      = prop['fwhm_focus'],
            i_focus         = prop['i_focus'],
            z_focus_locator = prop['z_focus_locator'],
            focus_search_halfwidth_m = prop['focus_search_halfwidth_m'],
            focus_at_edge   = prop['focus_at_edge'],
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
               f"r_DC={dc_carrier_ratio:.1f}  "
               f"SNR={carrier_snr:.1f}  "
               f"f_pred={fit['f_pred']:+.2f}m  "
               f"RMS_W={rms_W_nm:.1f}nm  RMS_resid={fit['rms_resid_nm']:.3f}nm")
        if propagate and fit_status == 'ok':
            edge_tag = ' *EDGE*' if prop['focus_at_edge'] else ''
            msg += (f"  z_focus={prop['z_focus']:+.2f}m"
                    f"({prop['z_focus_locator']}){edge_tag}  "
                    f"RMS@focus={wff['rms_at_focus_nm']:.3f}nm")
        if fit_status == 'fail':
            msg += f"  *FIT FAIL: {fit_fail_reason}*"
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
        centroid_threshold, r_good/bad, z_range_m, ...).

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

def plot_result(result, title='', figsize=None, x_axis='centered'):
    """Diagnostic figure for a single-frame reconstruction.

    Panels (always shown):
        1. Fringe profile (raw)
        2. FFT |spectrum| with carrier and filters
        3. Envelopes I0, A1 and beam mask
        4. Wavefront W(x) with parabolic fit
        5. Residual (W − fit) inside beam mask

    Additional panels (if propagation was run):
        6. Caustic |E(x,z)|²
        7. Beam size vs z (locator-dependent: σ_rms or FWHM/2.35)
        8. Intensity at focus + wavefront at focus (twin y-axis)

    Parameters
    ----------
    result : dict
        Output of reconstruct_single.
    title : str
        Optional figure suptitle.
    figsize : tuple, optional
        Auto-sized if None.
    x_axis : {'centered', 'raw'}, default 'centered'
        Spatial coordinate convention for panels 3–8. 'centered' puts
        x=0 at the centroid (uses result['x_mm']); 'raw' uses detector
        coordinates (np.arange(N)*dx). Panel 1 is always raw.

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

    N         = len(result['profile'])
    x_full_mm = np.arange(N) * result['dx'] * 1e3   # raw detector mm
    x_c_mm    = x_full_mm[result['x_c']]            # centroid in raw mm

    if x_axis == 'centered':
        x_plot = result['x_mm']
        x_ref  = 0.0
        xlabel = 'Position [mm] (x=0 at centroid)'
    elif x_axis == 'raw':
        x_plot = x_full_mm
        x_ref  = x_c_mm
        xlabel = 'Position [mm] (detector frame)'
    else:
        raise ValueError(f"x_axis must be 'centered' or 'raw', got {x_axis!r}")

    freq  = result['freq']
    m     = result['beam_mask']
    fm    = result['fit_mask']
    rejected = m & ~result['trust_mask']
    lam   = result['wavelength']

    # ─── Panel 1: raw fringe profile ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
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
                 f"r_DC={result['dc_carrier_ratio']:.1f}, "
                 f"SNR={result['carrier_snr']:.1f})")
    ax.set_xlim(-result['k_ideal']*1e-3, 3*result['k_ideal']*1e-3)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    # ─── Panel 3: envelopes & beam mask ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    I0_n = result['I0'] / result['I0'].max()
    A1_n = result['A1'] / result['A1'].max()
    ax.plot(x_plot, I0_n, 'C0', lw=1.3, label=r'$I_0$ (norm)')
    ax.plot(x_plot, A1_n, 'C3', lw=1.3, label=r'$A_{+1}$ (norm)')
    ax.fill_between(x_plot, 0, 1.05, where=m, color='gold', alpha=0.15,
                    label='beam_mask')
    if rejected.any():
        ax.fill_between(x_plot, 0, 1.05, where=rejected,
                        color='lightcoral', alpha=0.35, hatch='///',
                        edgecolor='firebrick', linewidth=0,
                        label=r'excluded ($V < V_{\mathrm{thr}}$)')
    ax.axvline(x_ref, color='C3', ls=':', lw=1, alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Norm intensity')
    ax.set_title(f"Envelopes  ({result['n_mask_px']} px beam, "
                 f"{result['n_fit_px']} px fit)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 4: wavefront + parabolic fit ──────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    W_para_nm   = result['W_para_rad'] * lam / (2*np.pi) * 1e9
    W_plot      = np.where(m, result['W_nm'], np.nan)
    W_para_plot = np.where(m, W_para_nm,      np.nan)
    ax.plot(x_plot, result['W_nm'], 'C0', lw=0.6, alpha=0.3, label='W (outside)')
    ax.plot(x_plot, W_plot,         'C0', lw=1.5, label='W measured')
    ax.plot(x_plot, W_para_plot,    'C3', lw=1.2, ls='--',
            label=f"parabolic (f={result['f_pred']:+.2f} m)")
    if m.any():
        lo, hi = result['W_nm'][m].min(), result['W_nm'][m].max()
        rng = hi - lo if hi > lo else 1.0
        ylo, yhi = lo - 0.15*rng, hi + 0.15*rng
        ax.set_ylim(ylo, yhi)
        ax.fill_between(x_plot, ylo, yhi, where=m, color='gold', alpha=0.12)
        if rejected.any():
            ax.fill_between(x_plot, ylo, yhi, where=rejected,
                            color='lightcoral', alpha=0.25, hatch='///',
                            edgecolor='firebrick', linewidth=0)
        x_beam   = x_plot[m]
        x_half   = (x_beam.max() - x_beam.min()) / 2
        x_center = (x_beam.max() + x_beam.min()) / 2
        ax.set_xlim(x_center - 1.5*x_half, x_center + 1.5*x_half)
    ax.axvline(x_ref, color='gray', ls=':', lw=0.7, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('W [nm OPL]')
    ax.set_title('Wavefront')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 5: residual (defocus removed, plotted over fit_mask) ──────────
    ax = fig.add_subplot(gs[2, :])
    W_resid_nm   = result['W_resid_rad'] * lam / (2*np.pi) * 1e9
    W_resid_plot = np.where(fm, W_resid_nm, np.nan)
    ax.plot(x_plot, W_resid_plot, 'C4', lw=1.4)
    ax.axhline(0, color='gray', lw=0.7, alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Residual [nm]')
    fit_fail_tag = ' *FIT FAIL*' if result.get('fit_status') == 'fail' else ''
    ax.set_title(f"Residual over fit_mask  "
                 f"(rms={result['rms_resid_nm']:.3f} nm){fit_fail_tag}")
    if fm.any():
        lo, hi = W_resid_nm[fm].min(), W_resid_nm[fm].max()
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
                       extent=(z[0], z[-1], x_plot[0], x_plot[-1]),
                       aspect='auto', origin='lower', cmap='inferno')
        ax.axvline(result['z_focus'], color='cyan', lw=1.2, ls='--',
                   label=f"z_focus = {result['z_focus']:+.2f} m")
        ax.axvline(result['f_pred'], color='lime',  lw=1.0, ls=':',
                   label=f"f_pred  = {result['f_pred']:+.2f} m")
        ax.set_xlabel('z [m]')
        ax.set_ylabel('x [mm]')
        ax.set_title('Caustic |E(x,z)|² (per-column normalized)')
        if m.any():
            x_beam = x_plot[m]
            x_half = max(abs(x_beam.min() - x_ref), abs(x_beam.max() - x_ref)) * 1.2
            ax.set_ylim(x_ref - x_half, x_ref + x_half)
        ax.legend(fontsize=8, loc='upper right')
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01)

        # Panel 7: beam size vs z (curve chosen by locator)
        ax = fig.add_subplot(gs[4, 0])
        locator = result.get('z_focus_locator', 'fwhm_min')
        if locator == 'fwhm_min':
            curve     = result['fwhm_z']
            curve_val = result['fwhm_focus']
            ylabel    = 'FWHM/2.35 [µm]'
            marker_lbl = (f"FWHM/2.35 at focus = {curve_val*1e6:.1f} µm"
                          if not np.isnan(curve_val) else 'FWHM/2.35 = NaN')
        else:
            curve     = result['sigma_rms']
            curve_val = result['sigma_focus']
            ylabel    = 'σ_rms [µm]'
            marker_lbl = f"σ_min = {curve_val*1e6:.1f} µm"

        ax.plot(z, curve*1e6, 'C0', lw=1.4)
        ax.axvline(result['z_focus'], color='cyan', lw=1, ls='--')
        # Mark search-window edges if a bounded search was actually used
        if result.get('focus_search_halfwidth_m', 0) > 0 and 'f_pred' in result:
            fp = result['f_pred']
            hw = result['focus_search_halfwidth_m']
            ax.axvspan(fp - hw, fp + hw, color='gray', alpha=0.08,
                       label=f'search window (±{hw*1e3:.0f} mm)')
        if not np.isnan(curve_val):
            ax.scatter([result['z_focus']], [curve_val*1e6],
                       color='red', s=40, zorder=5, label=marker_lbl)
        ax.set_xlabel('z [m]')
        ax.set_ylabel(ylabel)
        title_suffix = ' *EDGE*' if result.get('focus_at_edge', False) else ''
        ax.set_title(f"Beam size vs z  (locator: {locator}){title_suffix}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Panel 8: focus intensity + W at focus (twin axis)
        ax = fig.add_subplot(gs[4, 1])
        I_focus_norm = result['I_focus'] / result['I_focus'].max()
        ax.plot(x_plot, I_focus_norm, 'C1', lw=1.4, label='I (norm)')
        ax.fill_between(x_plot, 0, 1.05, where=m_f, color='gold', alpha=0.15)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Norm intensity', color='C1')
        ax.tick_params(axis='y', labelcolor='C1')
        ax.set_ylim(0, 1.05)
        # X-limit: ±1.5 × focus_mask half-width
        if m_f.any():
            x_f      = x_plot[m_f]
            x_half   = (x_f.max() - x_f.min()) / 2
            x_center = (x_f.max() + x_f.min()) / 2
            ax.set_xlim(x_center - 1.5*x_half, x_center + 1.5*x_half)

        # Wavefront on twin axis
        ax2 = ax.twinx()
        W_focus_plot = np.where(m_f, result['W_focus_nm'], np.nan)
        ax2.plot(x_plot, W_focus_plot, 'C4', lw=1.3, label='W at focus')
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


def plot_carrier(out, title='', figsize=(8, 4)):
    """One-panel diagnostic for find_carrier output.

    Shows |FFT| (log), the search band, k_ideal, and the located k_peak.

    Parameters
    ----------
    out : dict
        Output of find_carrier. Uses keys: freq, fft_avg, band, k_ideal,
        k_peak, k_amp, dc_amp, bg_amp.
    title : str, optional
        Figure title.
    figsize : tuple, optional
        Default (8, 4).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    freq    = out['freq']
    fft_avg = out['fft_avg']
    band_lo, band_hi = out['band']
    k_ideal = out['k_ideal']
    k_peak  = out['k_peak']
    k_amp   = out['k_amp']
    dc_amp  = out['dc_amp']
    bg_amp  = out['bg_amp']
    r_dc    = dc_amp / k_amp
    snr     = k_amp / bg_amp

    fig, ax = plt.subplots(figsize=figsize)
    ax.semilogy(freq * 1e-3, np.abs(fft_avg), lw=0.8,
                color='steelblue', label='|FFT|')
    ax.axvspan(band_lo * 1e-3, band_hi * 1e-3,
               color='gold', alpha=0.2, label='search band')
    ax.axvline(k_ideal * 1e-3, color='red',    ls=':',  lw=1,
               label=f"k_ideal = {k_ideal*1e-3:.1f} cyc/mm")
    ax.axvline(k_peak  * 1e-3, color='tomato', ls='--', lw=1.2,
               label=f"k_peak  = {k_peak*1e-3:.1f} cyc/mm")
    ax.axvline(0, color='gray', ls='--', lw=1, alpha=0.8, label='DC')
    ax.set_xlabel('Spatial frequency [cyc/mm]')
    ax.set_ylabel('|FFT| (log)')
    ax.set_xlim(-k_ideal * 1e-3, 3 * k_ideal * 1e-3)
    ax.set_title(
        f"Carrier  (r_DC={r_dc:.1f}, SNR={snr:.1f}, "
        f"k_peak/k_ideal={k_peak/k_ideal:.3f})"
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.85)

    return fig


def plot_caustic_amplitude(debug, title='', figsize=(12, 3.5)):
    """Three-panel diagnostic for build_caustic_amplitude(debug=True) output.

    Panels:
        1. Envelopes I0 and 2·A1 (normalized) with safety_mask shaded.
        2. Visibility V = 2·A1/I0 (normalized) with V_threshold reference line.
        3. Returned amplitude A with trust_mask shaded (gated mode only).

    Parameters
    ----------
    debug : dict
        Output of build_caustic_amplitude(..., debug=True). Uses keys:
        A, V, safety_mask, trust_mask, mode_used, I0, A1, V_threshold.
    title : str, optional
        Figure suptitle.
    figsize : tuple, optional
        Default (12, 3.5).

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    A          = debug['A']
    V          = debug['V']
    I0         = debug['I0']
    A1         = debug['A1']
    safety_mask = debug['safety_mask']
    trust_mask = debug['trust_mask']
    mode_used  = debug['mode_used']
    V_thr      = debug['V_threshold']

    N = len(A)
    px = np.arange(N)

    fig, axs = plt.subplots(1, 3, figsize=figsize)

    # ─── Panel 1: envelopes ─────────────────────────────────────────────────
    ax = axs[0]
    I0_n = I0 / I0.max() if I0.max() > 0 else I0
    A1_n = (2*A1) / (2*A1).max() if A1.max() > 0 else 2*A1
    ax.plot(px, I0_n, 'C0', lw=1.3, label=r'$I_0$ (norm)')
    ax.plot(px, A1_n, 'C3', lw=1.3, label=r'$2\,A_{+1}$ (norm)')
    ax.fill_between(px, 0, 1.05, where=safety_mask, color='gold', alpha=0.15,
                    label='safety_mask')
    ax.set_xlabel('Pixel index')
    ax.set_ylabel('Norm intensity')
    ax.set_ylim(0, 1.05)
    ax.set_title('Envelopes')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 2: visibility ────────────────────────────────────────────────
    ax = axs[1]
    ax.plot(px, V, 'C2', lw=1.3, label='V (norm)')
    ax.axhline(V_thr, color='gray', ls='--', lw=1,
               label=f'V_threshold = {V_thr:.2f}')
    ax.fill_between(px, 0, 1.05, where=safety_mask, color='gold', alpha=0.10)
    ax.set_xlabel('Pixel index')
    ax.set_ylabel('Visibility')
    ax.set_ylim(0, 1.05)
    ax.set_title(r'Visibility $V = 2\,A_{+1}/I_0$')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    # ─── Panel 3: returned amplitude ────────────────────────────────────────
    ax = axs[2]
    A_n = A / A.max() if A.max() > 0 else A
    ax.plot(px, A_n, 'C1', lw=1.3, label='A (norm)')
    if mode_used == 'gated':
        ax.fill_between(px, 0, 1.05, where=trust_mask, color='lime', alpha=0.18,
                        label='trust_mask')
    ax.set_xlabel('Pixel index')
    ax.set_ylabel('Norm amplitude')
    ax.set_ylim(0, 1.05)
    ax.set_title(f'Amplitude (mode={mode_used})')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12, fontweight='bold')
        fig.subplots_adjust(top=0.82)
    else:
        fig.tight_layout()
    plt.show()
    return fig
