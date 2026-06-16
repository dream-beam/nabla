"""
X-ray Beam Propagation and Optical Simulation Tools

This module provides propagators, optical element models, and beam
characterization functions for simulating soft X-ray optical systems.

Author: "Francis" Wei He (francisho@lbl.gov, wehe@ucsd.edu)
Date: October 2025
"""

import numpy as np

#----------------------------------------------------------------
# Propagators in 1D
#----------------------------------------------------------------

#Fresnel Transfer Function (TF) Propagator in 1D
def propTF(uin_V_m, L_m, lambda_m, z_m):
    """Fresnel Transfer Function (TF) Propagator in 1D.
    
    Best used when dx_m >= wavelength_m*z_m/L_m. Efficient for near-field 
    propagation where the output grid size matches input grid size.
    
    Note: Contains a slight numerical offset (~11nm at 12mm propagation) due to
    FFT frequency grid asymmetry.
    
    Parameters
    ----------
    uin_V_m : array_like
        Input field amplitude
    L_m : float
        Grid size [m]
    lambda_m : float
        Wavelength [m]
    z_m : float
        Propagation distance [m]
        
    Returns
    -------
    array_like
        Propagated field amplitude
    """
    M = uin_V_m.size
    dx_m = L_m/M
    # k_1_m = 2*np.pi/lambda_m    # wavenumber
    
    # Frequency coordinates
    fx_1_m = np.linspace(-1/(2*dx_m), 1/(2*dx_m) - (1/L_m), M)
    
    # Transfer function (quadratic phase in frequency space 
    # -- this is the Fresnel approximation of the angular spectrum method)
    H = np.exp(-1j * np.pi * lambda_m * z_m * (fx_1_m**2))
    H = np.fft.fftshift(H)  # Center the transfer function
    
    # FFT implementation
    Uin_V_m = np.fft.fft(np.fft.fftshift(uin_V_m))      # Forward FFT of centered input
    Uout_V_m = H * Uin_V_m                              # Apply propagation in frequency space
    uout_V_m = np.fft.ifftshift(np.fft.ifft(Uout_V_m))  # Inverse FFT and center
    
    return uout_V_m

# Huygens-Fresnel Propagator in 1D (original non-vectorized version — kept for reference)
# def propHF(xo_m, xi_m, Eo, k_1_m, z_m):
#     """Huygens-Fresnel Propagator in 1D.
#
#     Implements direct integration of Huygens-Fresnel principle. More computationally
#     intensive than propTF but allows arbitrary output grid size. Best used for
#     far-field propagation where dx_m < wavelength_m*z_m/L_m.
#
#     Computation time scales with len(xo_m) * len(xi_m).
#     """
#     M = np.size(xo_m)
#     N = np.size(xi_m)
#     Eii = np.zeros(M, dtype="complex")
#     Ei = np.zeros(N, dtype="complex")
#     for j in range(N):
#         for i in range(M):
#             roi_m = np.sign(z_m) * np.sqrt((xo_m[i] - xi_m[j])**2 + z_m**2)
#             Eii[i] = Eo[i] * np.exp(+1j * k_1_m * roi_m) / roi_m
#         Ei[j] = np.sum(Eii[:])
#     return Ei[:]

def propHF(xo_m, xi_m, Eo, k_1_m, z_m):
    """Vectorized Huygens-Fresnel Propagator in 1D.
    
    Implements direct integration of Huygens-Fresnel principle. 
    Best used for far-field propagation where dx_m < wavelength_m*z_m/L_m.
    
    Parameters
    ----------
    xo_m : array_like
        Source plane coordinates [m]
    xi_m : array_like
        Target plane coordinates [m]
    Eo : array_like
        Input field at source plane
    k_1_m : float
        Wavenumber (2π/λ) [1/m]
    z_m : float
        Propagation distance [m]
        
    Returns
    -------
    array_like
        Propagated field at target plane
    
    Notes
    -----
    Same as the original propHF but uses numpy broadcasting for faster computation.
    Warning: Uses more memory but is significantly faster.
    """
    # Create coordinate matrices using broadcasting
    # This creates a matrix of all combinations of xo_m and xi_m
    X_o, X_i = np.meshgrid(xo_m, xi_m, sparse=True)
    
    # Calculate all distances at once
    roi_m = np.sign(z_m) * np.sqrt((X_o - X_i)**2 + z_m**2)
    
    # Calculate propagation for all points simultaneously
    propagator = np.exp(1j * k_1_m * roi_m) / roi_m
    
    # Apply field and sum contributions
    return np.sum(Eo * propagator, axis=1)

#----------------------------------------------------------------
# Gaussian generator in 1D
#----------------------------------------------------------------
#Gaussian function in 1D
def gaussfunc(x, mean_x, sigma_x):
    """Generate a 1D Gaussian beam profile.

    Parameters
    ----------
    x : array_like
        Spatial coordinates [m]
    mean_x : float
        Center position [m]
    sigma_x : float
        1/e field radius (standard deviation) [m]

    Returns
    -------
    ndarray
        Gaussian amplitude profile: exp(-((x - mean_x) / (√2 · σ))²)
    """
    gaussF = np.exp(-((x-mean_x)/(np.sqrt(2)*sigma_x))**2)
    return gaussF

#----------------------------------------------------------------
# Grating and lens
#----------------------------------------------------------------
#VLS grating equation
def VLS(alpha_rad, m, lambda_m, a_m, k_lpm, p_m, q_m, w_m):
    beta_rad = np.arcsin(np.sin(alpha_rad) - m *lambda_m/a_m)
    b2 = -((np.cos(alpha_rad)**2)/p_m + (np.cos(beta_rad)**2)/q_m) / (2 * k_lpm * lambda_m) 
    b3 = -(np.cos(alpha_rad)**2 * np.sin(alpha_rad)/p_m**2 - np.cos(beta_rad)**2 * np.sin(beta_rad)/q_m**2) / (2 * k_lpm * lambda_m) 
    n = k_lpm * (w_m + b2 * w_m**2 + b3* w_m**3)
    return n

#VLS grating equation for reflection
def VLSrefl(alpha_rad, m, lambda_m, a_m, k_lpm, p_m, q_m, w_m):
    beta_rad = np.arcsin(m *lambda_m/a_m - np.sin(alpha_rad))
    b2 = -((np.cos(alpha_rad)**2)/p_m + (np.cos(beta_rad)**2)/q_m) / (2 * k_lpm * lambda_m) 
    b3 = -(np.cos(alpha_rad)**2 * np.sin(alpha_rad)/p_m**2 - np.cos(beta_rad)**2 * np.sin(beta_rad)/q_m**2) / (2 * k_lpm * lambda_m) 
    n = k_lpm * (w_m + b2 * w_m**2 + b3* w_m**3)
    return n


#Focus in 1D
def focus(uin_V_m, L_m, lambda_m, zf_m):
    M = uin_V_m.size
    dx_m = L_m/M
    k_1_m = 2 * np.pi/lambda_m
    
    x_m = np.linspace(-L_m/2, L_m/2 - dx_m, M)
    uout_V_m = uin_V_m * np.exp(-1j * k_1_m/(2*zf_m)*(x_m**2))
    return uout_V_m

def polyfunc(x, Dx, degree):
    """Generate a random polynomial wavefront for testing.

    Constructs a sum of monomials (x/Dx)^n with random coefficients in [0, 1).

    Parameters
    ----------
    x : array_like
        Spatial coordinates [m]
    Dx : float
        Normalization length scale [m] (typically the grid half-width)
    degree : int
        Number of polynomial terms (powers 0 through degree-1)

    Returns
    -------
    ndarray
        Random polynomial wavefront in the same units as x
    """
    shape_wave = x * 0
    for i_p in range(degree):
        shape_wave = shape_wave + np.random.rand(1) * (x/Dx) ** (i_p)
    return shape_wave

def secondmomt(Z, x_m, E):
    """Compute beam size at multiple propagation planes using the second moment (RMS) method.

    Parameters
    ----------
    Z : int
        Number of propagation planes (columns in E)
    x_m : array_like
        Transverse spatial coordinates [m]
    E : ndarray, shape (len(x_m), Z)
        Complex field amplitude at each plane

    Returns
    -------
    sigma_mrms : ndarray, shape (Z,)
        RMS beam radius (1σ) at each plane [m]
    """
    sigma_mrms = np.zeros(Z)
    for i_z in range(Z):
        I_z = np.abs(E[:,i_z] **2)/np.max(np.abs(E[:,i_z] **2))
        mu_m = np.sum(x_m * I_z)/np.sum(I_z)
        sigma_mrms[i_z]= np.sqrt(np.sum((x_m - mu_m)**2 * I_z)/np.sum(I_z))
    return sigma_mrms

def fwhm(Z, x_m, E):
    """Compute beam size at multiple propagation planes using the Full Width Half Maximum method.

    Parameters
    ----------
    Z : int
        Number of propagation planes (columns in E)
    x_m : array_like
        Transverse spatial coordinates [m]
    E : ndarray, shape (len(x_m), Z)
        Complex field amplitude at each plane

    Returns
    -------
    sigma2_mrms : ndarray, shape (Z,)
        FWHM beam size converted to 1σ equivalent (FWHM / 2.35) at each plane [m]
    """
    sigma2_mrms = np.zeros(Z)
    px_m = (x_m[1] - x_m[0])
    for i_z in range(Z):
        Iz = np.abs(E[:,i_z] **2)
        sigma2_mrms[i_z] = (np.size(np.where(Iz >= np.max(Iz/2))))/2.35 * px_m
    return sigma2_mrms

def divg(sigma_rms, z):
    """Compute beam divergence (half-angle) at each propagation plane.

    Parameters
    ----------
    sigma_rms : array_like
        Beam radius (1σ or FWHM/2.35) at each plane [m]
    z : array_like
        Propagation distances corresponding to each plane [m]

    Returns
    -------
    div : ndarray
        Half-angle divergence σ/z at each plane [rad]
    """
    div = np.zeros(np.size(z))
    for i_z in range(np.size(z)):
        div[i_z] = sigma_rms[i_z]/z[i_z]
    return div
