"""
formfactors.py

This module defines lists of form factor models for Small Angle X-ray Scattering (SAXS) simulations.
The models are sourced from SasView (https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html).

The form factors are categorized into different geometric shapes:
- Cylinders
- Ellipsoids
- Lamellae
- Spheres

Two main lists are created:
1. formfactors_original: Contains all the form factor models
2. formfactors: A copy of formfactors_original, which can be modified without affecting the original list

These lists can be imported and used in other parts of the SAXS simulation and analysis pipeline.
"""

# List of all interesting models from: https://www.sasview.org/docs/user/qtgui/Perspectives/Fitting/models/index.html
Cylinder_functions_essentials = ['cylinder', 'elliptical_cylinder', 'flexible_cylinder', 'flexible_cylinder_elliptical', 'hollow_cylinder', 'stacked_disks']
Ellipsoid_functions = ['core_shell_ellipsoid', 'ellipsoid', 'triaxial_ellipsoid']
Lamellae_functions_essentials = ['lamellar']
Sphere_functions_essentials = ['binary_hard_sphere', 'core_shell_sphere', 'fuzzy_sphere', 'sphere', 'superball', 'vesicle']
formfactors_original = Cylinder_functions_essentials + Ellipsoid_functions + Lamellae_functions_essentials + Sphere_functions_essentials
formfactors = formfactors_original.copy()

formfactor_params = {
'sphere': ['background', 'radius', 'radius_pd'],
'cylinder': ['background', 'radius', 'radius_pd', 'length', 'length_pd'],
'ellipsoid': ['background', 'radius_polar', 'radius_equatorial'],
'elliptical_cylinder': ['background', 'radius_minor', 'axis_ratio', 'length', 'length_pd'],
'flexible_cylinder': ['background', 'radius', 'radius_pd', 'length', 'length_pd', 'kuhn_length'],
'flexible_cylinder_elliptical': ['background', 'radius', 'radius_pd', 'axis_ratio', 'length', 'length_pd', 'kuhn_length'],
'stacked_disks': ['background', 'radius', 'radius_pd', 'thick_layer', 'thick_core', 'n_stacking'],
'core_shell_ellipsoid': ['background', 'radius_equat_core', 'x_core', 'thick_shell', 'x_polar_shell', 'sld_core', 'sld_shell'],
'binary_hard_sphere': ['background', 'radius_lg', 'radius_sm', 'volfraction_lg', 'volfraction_sm'],
'vesicle': ['background', 'radius', 'radius_pd', 'thickness', 'volfraction'],
'core_shell_sphere': ['background', 'radius', 'radius_pd', 'thickness'],
'triaxial_ellipsoid': ['background', 'radius_equat_minor', 'radius_equat_major', 'radius_polar'],
'superball': ['background', 'length_a', 'exponent_p'],
'fuzzy_sphere': ['background', 'radius', 'radius_pd', 'fuzziness'],
'hollow_cylinder': ['background', 'radius', 'radius_pd', 'thickness', 'length', 'length_pd'],
'lamellar': ['background', 'thickness'],
                }