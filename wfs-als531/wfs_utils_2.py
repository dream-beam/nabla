"""
Wavefront Analysis Utilities for Lateral Shearing Interferometry

This module provides core functions for analyzing X-ray wavefront sensor data
using grating-based lateral shearing interferometry.

Author: "Francis" Wei He (francisho@lbl.gov, wehe@ucsd.edu)
Date: October 2025
"""

import numpy as np
# from scipy import signal
import matplotlib.pyplot as plt


def calculate_phase_from_fringes(intensity_profile, pixel_size, 
                                   dc_exclude_width=5000, 
                                   filter_width_ratio=5.0,
                                   plot=False):
    """
    Extract phase from interferometric fringe pattern using automatic carrier detection.
    
    This function automatically detects the carrier frequency from the FFT peak,
    applies a Gaussian filter, and extracts the phase.
    
    Parameters
    ----------
    intensity_profile : array_like
        1D intensity profile from interferogram
    pixel_size : float
        Pixel size in meters [m]
    dc_exclude_width : float, optional
        Width around DC (zero frequency) to exclude when finding carrier peak [cycles/m]
        Default is 5000 cycles/m
    filter_width_ratio : float, optional
        Ratio of carrier frequency to filter width (higher = narrower filter)
        Default is 5.0 (filter width = carrier_freq / 5)
    plot : bool, optional
        If True, plot FFT spectrum with filter and detected carrier. Default is False.
    
    Returns
    -------
    phase_unwrapped : ndarray
        Unwrapped phase centered at middle point [radians]
    x_coords : ndarray
        Spatial coordinates in meters [m]
    k_carrier : float
        Detected carrier frequency [cycles/m]
    fig : matplotlib.figure.Figure or None
        Figure handle if plot=True, otherwise None
        
    Notes
    -----
    Method:
    1. FFT the intensity profile
    2. Exclude DC region (±dc_exclude_width)
    3. Find peak in positive frequencies → carrier frequency
    4. Apply Gaussian filter: exp(-0.5 * ((f - f_c) / σ)²)
       where σ = f_c / filter_width_ratio
    5. IFFT to get complex field, extract and unwrap phase
    
    Examples
    --------
    >>> dx_m = 1.21e-6  # 1.21 µm pixel
    >>> phase, x, k_c, fig = calculate_phase_from_fringes(
    ...     I_detector, pixel_size=dx_m, plot=True
    ... )
    >>> print(f"Detected carrier: {k_c*1e-3:.1f} cycles/mm")
    """
    # Ensure 1D array
    if intensity_profile.ndim > 1:
        intensity_profile = intensity_profile.ravel()
    
    Nx = len(intensity_profile)
    
    # Step 1: FFT with proper shifts
    fft_result = np.fft.fftshift(np.fft.fft(np.fft.fftshift(intensity_profile)))
    freq_x = np.fft.fftshift(np.fft.fftfreq(Nx, pixel_size))
    fft_magnitude = np.abs(fft_result)
    
    # Step 2: Exclude DC component
    fft_magnitude_no_dc = fft_magnitude.copy()
    dc_index = Nx // 2
    dc_width = int(dc_exclude_width * pixel_size * Nx)  # Convert to index width
    if dc_width < 1:
        dc_width = 50  # Default to 50 pixels if calculated width too small
    fft_magnitude_no_dc[dc_index - dc_width : dc_index + dc_width] = 0
    
    # Step 3: Find carrier frequency (highest peak in positive frequencies)
    search_mask = (freq_x > 0) & (np.abs(freq_x) > dc_exclude_width)
    fft_search = np.zeros_like(fft_magnitude_no_dc)
    fft_search[search_mask] = fft_magnitude_no_dc[search_mask]
    carrier_index = np.argmax(fft_search)
    k_carrier = freq_x[carrier_index]
    
    # Step 4: Apply Gaussian filter centered at carrier
    filter_width = k_carrier / filter_width_ratio
    gaussian_filter = np.exp(-0.5 * ((freq_x - k_carrier) / filter_width)**2)
    fft_filtered = fft_result * gaussian_filter
    
    # Step 5: IFFT to get complex field
    E_reconstructed = np.fft.ifftshift(np.fft.ifft(np.fft.ifftshift(fft_filtered)))
    
    # Extract wrapped phase
    phase_wrapped = np.angle(E_reconstructed)
    
    # Unwrap and center
    phase_unwrapped = np.unwrap(phase_wrapped)
    phase_unwrapped = phase_unwrapped - phase_unwrapped[Nx//2]
    
    # Spatial coordinates
    x_coords = (np.arange(Nx) - Nx//2) * pixel_size
    
    # Plotting if requested
    fig = None
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Top: Gaussian filtering on +1 order (original FFT vs filtered)
        ax = axes[0]
        ax.plot(freq_x * 1e-3, fft_magnitude, linewidth=1, alpha=0.7, label='Original FFT')
        ax.plot(freq_x * 1e-3, np.abs(fft_filtered), linewidth=1.5, label='Filtered')
        ax.axvline(k_carrier * 1e-3, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Spatial Frequency [cycles/mm]', fontsize=11)
        ax.set_ylabel('FFT Magnitude', fontsize=11)
        ax.set_title(f'Gaussian Filtering on +1 Order (width={filter_width*1e-3:.1f} cyc/mm)', fontsize=12)
        ax.set_xlim(-100, 100)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Middle: Wrapped phase
        ax = axes[1]
        ax.plot(x_coords * 1e6, phase_wrapped, linewidth=1)
        ax.set_xlabel('Position x [µm]', fontsize=11)
        ax.set_ylabel('Phase [rad]', fontsize=11)
        ax.set_title('Wrapped Phase', fontsize=12)
        ax.set_ylim(-np.pi, np.pi)
        ax.grid(True, alpha=0.3)
        
        # Bottom: Unwrapped & centered phase
        ax = axes[2]
        ax.plot(x_coords * 1e6, phase_unwrapped, linewidth=1.5)
        ax.set_xlabel('Position x [µm]', fontsize=11)
        ax.set_ylabel('Phase [rad]', fontsize=11)
        ax.set_title('Unwrapped & Centered Phase Difference', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    return phase_unwrapped, x_coords, k_carrier, fig


def reconstruct_wavefront(phase_centered, dx, grating_pitch, wavelength, z):
    """
    Reconstruct wavefront from unwrapped phase by carrier removal and cumulative integration.

    Parameters
    ----------
    phase_centered : array_like
        Unwrapped phase centered at midpoint [radians]. Still contains carrier ramp.
    dx : float
        Spatial sampling interval [m]
    grating_pitch : float
        Ideal grating pitch from fabrication [m]
    wavelength : float
        X-ray wavelength [m]
    z_T : float
        Grating-to-detector distance [m]

    Returns
    -------
    wavefront_reconstructed : ndarray
        Reconstructed wavefront W(x) at the detector plane [rad]

    Notes
    -----
    Carrier removal uses the ideal pitch (1/p), not the FFT-measured carrier frequency.
    The difference between the ideal and measured carrier encodes wavefront curvature;
    subtracting the measured carrier would erase it.

    Reconstruction formula:
        W(x) = (p / λ z_T) ∫ Δφ(x) dx
    """
    Nx = len(phase_centered)
    x_coords = (np.arange(Nx) - Nx // 2) * dx

    carrier_ramp = 2 * np.pi / grating_pitch * x_coords
    carrier_ramp_centered = carrier_ramp - carrier_ramp[Nx // 2]
    delta_phi = phase_centered - carrier_ramp_centered

    W_temp = -np.cumsum(delta_phi) * dx * grating_pitch / wavelength / z
    wavefront_reconstructed = W_temp - W_temp[Nx // 2]

    return wavefront_reconstructed


def detrending(wavefront_rad, x_coords, fit_order=2):
    """
    Remove large-scale curvature from wavefront by polynomial fitting.
    
    Parameters
    ----------
    wavefront_rad : array_like
        Wavefront in radians [rad]
    x_coords : array_like
        Spatial coordinates in meters [m]
    fit_order : int, optional
        Order of polynomial fit (default=2 for parabolic curvature)
    
    Returns
    -------
    fitted_curvature : ndarray
        Fitted polynomial curvature in radians [rad]
    aberrations_rad : ndarray
        Residual aberrations after removing fit [rad]
    poly_coeffs : ndarray
        Polynomial coefficients (highest degree first)
        
    Notes
    -----
    Typical use: fit_order=2 removes defocus (parabolic curvature)
    Higher orders can remove additional low-frequency components.
    
    All outputs are in radians. Statistics and plotting should be done separately.
    """
    # Fit polynomial
    poly_coeffs = np.polyfit(x_coords, wavefront_rad, fit_order)
    fitted_curvature = np.polyval(poly_coeffs, x_coords)
    
    # Calculate residual aberrations
    aberrations_rad = wavefront_rad - fitted_curvature
    
    return fitted_curvature, aberrations_rad, poly_coeffs


def calculate_rms_across_dataset(wavefront_list, x_coords_list=None):
    """
    Calculate point-by-point RMS from multiple wavefront measurements.
    
    This function computes the standard deviation at each spatial position
    across multiple measurements, revealing the stability/repeatability.
    
    Parameters
    ----------
    wavefront_list : list of arrays
        List of wavefront arrays from multiple measurements
    x_coords_list : list of arrays, optional
        List of corresponding x-coordinate arrays. If provided, will interpolate
        all wavefronts to a common grid. If None, assumes all wavefronts share
        the same coordinate system.
    
    Returns
    -------
    mean_wavefront : ndarray
        Mean wavefront across all measurements
    rms_map : ndarray
        Point-by-point RMS (standard deviation) in same units as input
    x_coords : ndarray
        Spatial coordinates
        
    Notes
    -----
    RMS values indicate measurement precision. Values < 1 pm are excellent.
    
    Examples
    --------
    >>> wavefronts = [wf1, wf2, wf3, ...]  # From 27 measurements
    >>> mean_wf, rms, x = calculate_rms_across_dataset(wavefronts)
    >>> print(f"Average RMS: {np.mean(rms):.3f} pm")
    """
    # Convert to array for easier manipulation
    wavefront_array = np.array(wavefront_list)  # Shape: (n_measurements, n_points)
    
    # Calculate mean and std at each point
    mean_wavefront = np.mean(wavefront_array, axis=0)
    rms_map = np.std(wavefront_array, axis=0)
    
    # Use x_coords from first measurement if not provided
    if x_coords_list is not None:
        x_coords = x_coords_list[0]
    else:
        x_coords = np.arange(len(mean_wavefront))
    
    return mean_wavefront, rms_map, x_coords

