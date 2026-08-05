"""
Shared Shadow4 beamline simulation logic for the ALS 5.3.1 digital twin.

This module extracts the beamline-construction-and-trace logic that used
to live inline in ``als531_shadow4.ipynb`` into a single reusable
function, :func:`build_and_trace`, which maps a dictionary of "degree of
freedom" (DOF) values onto a simulated camera image.

It is used by:

* ``als531_shadow4_ioc.py`` -- a standalone caproto IOC that exposes the
  DOFs as EPICS process variables and the resulting camera image as an
  areaDetector-style PV interface.
* ``als531_shadow4.ipynb`` -- which imports only the lightweight camera
  geometry constants (for plotting) and otherwise talks to the IOC as a
  pure EPICS client.

Note that the ``shadow4``/``syned`` imports are deferred to inside
:func:`build_and_trace` so that importing this module (e.g. just to get
the camera geometry constants) does not require those (possibly heavy)
optics packages to be installed.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Degree-of-freedom definitions
# ---------------------------------------------------------------------------

#: Names of the 10 degrees of freedom exposed by this beamline model.
DOF_NAMES = [
    "m1_rel_pitch_deg",
    "m1_bending_m",
    "mono_height_m",
    "mono_angle_rel_angle_deg",
    "hexapod_rel_x_m",
    "hexapod_rel_y_m",
    "hexapod_rel_z_m",
    "hexapod_rel_Rx_deg",
    "hexapod_rel_Ry_deg",
    "hexapod_rel_Rz_deg",
]

#: Default ("as-built") values for each degree of freedom, matching the
#: original values hard-coded in als531_shadow4.ipynb.
DEFAULT_DOF_VALUES = {
    "m1_rel_pitch_deg": -0.0001,
    "m1_bending_m": 2400,
    "mono_height_m": 1e-3,
    "mono_angle_rel_angle_deg": 0.00,
    "hexapod_rel_x_m": 0.0,
    "hexapod_rel_y_m": 0.0,
    "hexapod_rel_z_m": 0.0,
    "hexapod_rel_Rx_deg": 0.0,
    "hexapod_rel_Ry_deg": 0.0,
    "hexapod_rel_Rz_deg": 0.0,
}

# ---------------------------------------------------------------------------
# Simulated camera / detector geometry (mirrors the "Pilatus" camera used
# in als531_shadow4.ipynb).
# ---------------------------------------------------------------------------

#: Simulated camera image width, in pixels.
CAMERA_NX = 1200
#: Simulated camera image height, in pixels.
CAMERA_NY = 1000
#: Physical pixel size, in metres.
CAMERA_PIXEL_SIZE_M = 2.4e-6
#: Optical magnification between the sample/focus plane and the camera.
CAMERA_MAGNIFICATION = 0.25

#: Physical width/height of the field of view at the camera, in metres.
CAMERA_X_M = CAMERA_NX * CAMERA_PIXEL_SIZE_M / CAMERA_MAGNIFICATION
CAMERA_Y_M = CAMERA_NY * CAMERA_PIXEL_SIZE_M / CAMERA_MAGNIFICATION


def build_and_trace(dof_values: dict | None = None, nrays: int = 10_000, seed: int = 0) -> dict:
    """Build the ALS 5.3.1 Shadow4 beamline with the given DOF values and
    trace it, returning a dict of simulation results.

    Parameters
    ----------
    dof_values:
        Mapping of DOF name (see :data:`DOF_NAMES`) to its numeric
        value. Any DOF not present in the mapping falls back to
        :data:`DEFAULT_DOF_VALUES`. May be ``None`` to use all defaults.
    nrays:
        Number of rays to trace from the source.
    seed:
        Random seed for the source ray generation.

    Returns
    -------
    dict with keys:
        ``camera_image``: 2D numpy array of shape ``(ny, nx)`` with the
        binned intensity on the simulated camera (row-major, matching
        the areaDetector ``ArrayData`` convention used elsewhere in
        this repository).
        ``intensity_total``: sum of all (non-lost) ray intensities.
        ``nx``, ``ny``: camera image dimensions in pixels.
        ``pixel_size_m``: camera pixel size in metres.
        ``grid_x_m``, ``grid_y_m``: 1D bin-centre coordinate arrays.
        ``x_m``, ``y_m``, ``intensity``: raw (non-lost) ray coordinates
        and per-ray intensity, useful for further analysis/plotting.
    """
    dofs = dict(DEFAULT_DOF_VALUES)
    dofs.update(dof_values or {})

    from shadow4.beamline.s4_beamline import S4Beamline
    from shadow4.sources.source_geometrical.source_geometrical import (
        SourceGeometrical,
    )
    from syned.beamline.shape import Rectangle
    from syned.beamline.element_coordinates import ElementCoordinates
    from shadow4.beamline.s4_beamline_element_movements import (
        S4BeamlineElementMovements,
    )
    from shadow4.beamline.optical_elements.mirrors.s4_toroid_mirror import (
        S4ToroidMirror,
        S4ToroidMirrorElement,
    )
    from shadow4.beamline.optical_elements.absorbers.s4_screen import (
        S4Screen,
        S4ScreenElement,
    )
    from shadow4.beamline.optical_elements.crystals.s4_plane_crystal import (
        S4PlaneCrystal,
        S4PlaneCrystalElement,
    )
    from shadow4.beamline.optical_elements.mirrors.s4_plane_mirror import (
        S4PlaneMirror,
        S4PlaneMirrorElement,
    )

    beamline = S4Beamline()

    light_source = SourceGeometrical(name="Geometrical Source", nrays=nrays, seed=seed)
    light_source.set_spatial_type_gaussian(sigma_h=0.00025, sigma_v=0.000020)
    light_source.set_depth_distribution_off()
    light_source.set_angular_distribution_gaussian(sigdix=0.000020, sigdiz=0.000020)
    light_source.set_energy_distribution_singleline(8000.000000, unit="eV")
    light_source.set_polarization(
        polarization_degree=1.000000, phase_diff=0.000000, coherent_beam=0
    )
    beam = light_source.get_beam()

    beamline.set_light_source(light_source)

    # -- M1 toroidal mirror ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.01, x_right=0.01, y_bottom=-0.25, y_top=0.25)
    optical_element = S4ToroidMirror(
        name="M1 Toroidal mirror V-deflecting",
        boundary_shape=boundary_shape,
        surface_calculation=1,
        min_radius=0.06,  # min_radius = sagittal
        maj_radius=dofs["m1_bending_m"],  # maj_radius = tangential
        f_torus=0,
        p_focus=12,
        q_focus=12,
        grazing_angle=0.005,
        f_reflec=0,
        f_refl=0,
        file_refl="<none>",
        refraction_index=0.99999 + 0.001j,
        coating_material="Si",
        coating_density=2.33,
        coating_roughness=0,
    )
    coordinates = ElementCoordinates(
        p=12, q=0, angle_radial=1.565796327, angle_azimuthal=0, angle_radial_out=1.565796327
    )
    movements = S4BeamlineElementMovements(
        f_move=1,
        offset_x=1e-09,
        offset_y=1e-09,
        offset_z=1e-09,
        rotation_x=dofs["m1_rel_pitch_deg"],
        rotation_y=1.74533e-11,
        rotation_z=1.74533e-11,
    )
    beamline_element = S4ToroidMirrorElement(
        optical_element=optical_element,
        coordinates=coordinates,
        movements=movements,
        input_beam=beam,
    )
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- M1 photon slit ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.5, x_right=0.5, y_bottom=-0.000249999, y_top=0.000250001)
    optical_element = S4Screen(
        name="M1 slits V-defining",
        boundary_shape=boundary_shape,
        i_abs=0,
        i_stop=0,
        thick=0,
        file_abs="<specify file name>",
        material="Au",
        density=19.3,
    )
    coordinates = ElementCoordinates(p=0.5, q=0, angle_radial=0, angle_azimuthal=0, angle_radial_out=3.141592654)
    beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- differential pump slits ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.005, x_right=0.005, y_bottom=-0.005, y_top=0.005)
    optical_element = S4Screen(
        name="differential pump Slits",
        boundary_shape=boundary_shape,
        i_abs=0,
        i_stop=0,
        thick=0,
        file_abs="<specify file name>",
        material="Au",
        density=19.3,
    )
    coordinates = ElementCoordinates(p=8.5, q=0, angle_radial=0, angle_azimuthal=0, angle_radial_out=3.141592654)
    beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- dcm1 ------------------------------------------------
    optical_element = S4PlaneCrystal(
        name="DCM1",
        boundary_shape=None,
        material="Si",
        miller_index_h=1,
        miller_index_k=1,
        miller_index_l=1,
        f_bragg_a=False,
        asymmetry_angle=0.0,
        is_thick=1,
        thickness=0.001,
        f_central=0,
        f_phot_cent=0,
        phot_cent=8000.0,
        file_refl="bragg.dat",
        f_ext=0,
        material_constants_library_flag=0,  # 0=xraylib,1=dabax,2=preprocessor v1,3=preprocessor v2
        method_efields_management=0,  # 0=new in S4; 1=like in S3
    )
    coordinates = ElementCoordinates(
        p=3, q=0, angle_radial=1.321032031, angle_azimuthal=0, angle_radial_out=1.321032031
    )
    movements = S4BeamlineElementMovements(
        f_move=1,
        offset_x=1e-09,
        offset_y=1e-09,
        offset_z=dofs["mono_height_m"],
        rotation_x=dofs["mono_angle_rel_angle_deg"],
        rotation_y=1.74533e-11,
        rotation_z=1.74533e-11,
    )
    beamline_element = S4PlaneCrystalElement(
        optical_element=optical_element, coordinates=coordinates, movements=movements, input_beam=beam
    )
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- dcm2 ------------------------------------------------
    optical_element = S4PlaneCrystal(
        name="DCM2",
        boundary_shape=None,
        material="Si",
        miller_index_h=1,
        miller_index_k=1,
        miller_index_l=1,
        f_bragg_a=False,
        asymmetry_angle=0.0,
        is_thick=1,
        thickness=0.001,
        f_central=0,
        f_phot_cent=0,
        phot_cent=8000.0,
        file_refl="bragg.dat",
        f_ext=0,
        material_constants_library_flag=0,
        method_efields_management=0,
    )
    coordinates = ElementCoordinates(
        p=0.025, q=0, angle_radial=1.321032031, angle_azimuthal=3.141592654, angle_radial_out=1.321032031
    )
    movements = S4BeamlineElementMovements(
        f_move=1,
        offset_x=1e-09,
        offset_y=1e-09,
        offset_z=1e-09,
        rotation_x=1.74533e-11,
        rotation_y=1.74533e-11,
        rotation_z=1.74533e-11,
    )
    beamline_element = S4PlaneCrystalElement(
        optical_element=optical_element, coordinates=coordinates, movements=movements, input_beam=beam
    )
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- white beam slits (Harmonic suppressor) ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.5, x_right=0.5, y_bottom=-0.005, y_top=0.005)
    optical_element = S4Screen(
        name="White beam slits (Harmonic suppressor)",
        boundary_shape=boundary_shape,
        i_abs=0,
        i_stop=0,
        thick=0,
        file_abs="<specify file name>",
        material="Au",
        density=19.3,
    )
    coordinates = ElementCoordinates(p=0.075, q=0, angle_radial=0, angle_azimuthal=0, angle_radial_out=3.141592654)
    beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- beam defining slits ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.5, x_right=0.5, y_bottom=-0.5, y_top=0.5)
    optical_element = S4Screen(
        name="Generic Beam Screen/Slit/Stopper/Attenuator",
        boundary_shape=boundary_shape,
        i_abs=0,
        i_stop=0,
        thick=0,
        file_abs="<specify file name>",
        material="Au",
        density=19.3,
    )
    coordinates = ElementCoordinates(p=0.5, q=0, angle_radial=0, angle_azimuthal=0, angle_radial_out=3.141592654)
    beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- hexapod mirror ------------------------------------------------
    boundary_shape = Rectangle(x_left=-0.152, x_right=0.152, y_bottom=-0.152, y_top=0.152)
    optical_element = S4PlaneMirror(
        name="hexapod",
        boundary_shape=boundary_shape,
        f_reflec=0,
        f_refl=0,
        file_refl="<none>",
        refraction_index=0.99999 + 0.001j,
        coating_material="Si",
        coating_density=2.33,
        coating_roughness=0,
    )
    coordinates = ElementCoordinates(
        p=1, q=0, angle_radial=0.7853981634, angle_azimuthal=0, angle_radial_out=0.7853981634
    )
    movements = S4BeamlineElementMovements(
        f_move=1,
        offset_x=dofs["hexapod_rel_x_m"],
        offset_y=dofs["hexapod_rel_y_m"],
        offset_z=dofs["hexapod_rel_z_m"],
        rotation_x=dofs["hexapod_rel_Rx_deg"],
        rotation_y=dofs["hexapod_rel_Ry_deg"],
        rotation_z=dofs["hexapod_rel_Rz_deg"],
    )
    beamline_element = S4PlaneMirrorElement(
        optical_element=optical_element, coordinates=coordinates, movements=movements, input_beam=beam
    )
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -- Pilatus / camera screen ------------------------------------------------
    optical_element = S4Screen(
        name="Pilatus",
        boundary_shape=None,
        i_abs=0,
        i_stop=0,
        thick=0,
        file_abs="<specify file name>",
        material="Au",
        density=19.3,
    )
    coordinates = ElementCoordinates(p=1, q=1, angle_radial=0, angle_azimuthal=0, angle_radial_out=3.141592654)
    beamline_element = S4ScreenElement(optical_element=optical_element, coordinates=coordinates, input_beam=beam)
    beam, footprint = beamline_element.trace_beam()
    beamline.append_beamline_element(beamline_element)

    # -----------------------------------------------------------------
    # Bin rays onto the simulated camera.
    # -----------------------------------------------------------------
    x_m = beam.get_column(1, nolost=1)
    y_m = beam.get_column(3, nolost=1)
    intensity = beam.get_column(23, nolost=1)

    grid_x_edges_m = np.linspace(-CAMERA_X_M / 2, CAMERA_X_M / 2, CAMERA_NX + 1)
    grid_y_edges_m = np.linspace(-CAMERA_Y_M / 2, CAMERA_Y_M / 2, CAMERA_NY + 1)

    camera_image, _xedges, _yedges = np.histogram2d(
        x_m, y_m, bins=[grid_x_edges_m, grid_y_edges_m], weights=(intensity)
    )
    # np.histogram2d returns shape (nx, ny); transpose to the (ny, nx)
    # row-major image convention used for plotting and for the
    # areaDetector-style ArrayData PV.
    camera_image = camera_image.T

    grid_x_m = 0.5 * (grid_x_edges_m[:-1] + grid_x_edges_m[1:])
    grid_y_m = 0.5 * (grid_y_edges_m[:-1] + grid_y_edges_m[1:])

    return {
        "camera_image": camera_image,
        "intensity_total": float(np.sum(intensity)),
        "nx": camera_image.shape[1],
        "ny": camera_image.shape[0],
        "pixel_size_m": CAMERA_PIXEL_SIZE_M,
        "grid_x_m": grid_x_m,
        "grid_y_m": grid_y_m,
        "x_m": x_m,
        "y_m": y_m,
        "intensity": intensity,
    }
