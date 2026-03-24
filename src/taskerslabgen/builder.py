from ase.build import surface
from .core import apply_vacuum_to_slab


def build_cut_slabs(bulk_atoms, miller, layer_thickness_list, zbot, ztop, L, vacuum=15.0):
    """Build Tasker I/II slabs of various thicknesses by cutting between zbot and ztop."""
    slabs = []
    for layer_thickness in layer_thickness_list:
        surf_bulk_n = surface(bulk_atoms, miller, layers=layer_thickness + 2, vacuum=0.0)
        zmin = zbot
        zmax = ztop + layer_thickness * L
        mask = [(zmin <= atom.position[2] <= zmax) for atom in surf_bulk_n]
        slab = surf_bulk_n[mask]
        slab.set_pbc((True, True, True))
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=2)
        slabs.append(slab)
    return slabs
