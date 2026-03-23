from pathlib import Path

from ase.build import surface
from ase.io import write

from .core import apply_vacuum_to_slab


def build_cut_slabs(
    bulk_atoms,
    miller,
    layer_thickness_list,
    zbot,
    ztop,
    L,
    repeat=(1, 1, 1),
    vacuum=15.0,
    out_dir=".",
    filename_template="surf_bulk_layers_{layer_thickness}.xyz",
    output_ext="xyz",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    slabs = []

    for layer_thickness in layer_thickness_list:
        surf_bulk_n = surface(bulk_atoms, miller, layers=layer_thickness + 2, vacuum=0.0)
        zmin = zbot
        zmax = ztop + layer_thickness * L
        mask = [(zmin <= atom.position[2] <= zmax) for atom in surf_bulk_n]
        slab = surf_bulk_n[mask]
        if repeat is not None:
            slab = slab.repeat(tuple(repeat))
        slab.set_pbc((True, True, True))
        apply_vacuum_to_slab(slab, vacuum=vacuum, axis=2)

        if output_ext is not None and filename_template is not None:
            out_path = out_dir / filename_template.format(layer_thickness=layer_thickness)
            write(out_path.as_posix(), slab)
            outputs.append(out_path.as_posix())
        slabs.append(slab)

    return slabs, outputs
