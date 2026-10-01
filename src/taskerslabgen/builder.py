from .core import apply_vacuum_to_slab, build_surface


def build_cut_slabs(bulk_atoms, miller, layer_thickness_list, zbot, ztop, L, vacuum=15.0):
    """
    Build Tasker I/II slabs of various thicknesses by cutting between
    *zbot* and *ztop*.

    Parameters
    ----------
    bulk_atoms : Atoms
        Bulk unit cell.
    miller : tuple of int
        Miller index ``(h, k, l)``.
    layer_thickness_list : list of int
        Slab thicknesses in bulk repeat units.
    zbot : float
        Bottom cut z-coordinate (angstrom), in the frame of
        ``build_surface(bulk_atoms, miller)``.
    ztop : float
        Top cut z-coordinate (angstrom), same frame.
    L : float
        Lattice-plane spacing (angstrom).
    vacuum : float
        Vacuum to add (angstrom, per side).

    Returns
    -------
    list of Atoms
        One slab per requested thickness, sorted by atom count.
    """
    slabs = []
    for layer_thickness in layer_thickness_list:
        surf_bulk_n = build_surface(bulk_atoms, miller, layers=layer_thickness + 2)
        zmin = zbot
        zmax = ztop + layer_thickness * L
        mask = [(zmin <= atom.position[2] <= zmax) for atom in surf_bulk_n]
        slab = surf_bulk_n[mask]
        slab.set_pbc((True, True, True))
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=2)
        slabs.append(slab)
    slabs.sort(key=len)
    return slabs
