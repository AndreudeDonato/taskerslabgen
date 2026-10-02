"""
Regenerate the figures in docs/images/ used by README.md and example/TUTORIAL.md.

Every figure is a plot made by the library itself (plot=True), so the docs
show exactly what users get.  Run from anywhere:

    python3 docs/make_figures.py
"""
import shutil
import tempfile
from pathlib import Path

from ase.io import read

from taskerslabgen import cutslab, generate_slabs_for_miller

HERE = Path(__file__).resolve().parent
BULKS = HERE.parent / "bulk_files"
IMAGES = HERE / "images"
Q_IR = {"Ir": 4.0, "O": -2.0}
Q_CE = {"Ce": 4.0, "O": -2.0}
Q_TI = {"Ti": 4.0, "O": -2.0}


def one(folder, pattern):
    found = sorted(Path(folder).glob(pattern))
    if len(found) != 1:
        raise RuntimeError(f"expected one plot matching {pattern} in {folder}, got {found}")
    return found[0]


def main():
    IMAGES.mkdir(exist_ok=True)
    iro2 = read((BULKS / "IrO2_rutile.cif").as_posix())
    ceo2 = read((BULKS / "CeO2_fluorite.cif").as_posix())
    anatase = read((BULKS / "TiO2_anatase.cif").as_posix())
    figures = {}
    with tempfile.TemporaryDirectory() as tmp:
        # genslab: a Tasker I/II termination and a Tasker III reconstruction
        d = f"{tmp}/g1"
        term = generate_slabs_for_miller(iro2, Q_IR, (1, 1, 0), [3], bulk_name="IrO2",
                                         plot=True, plot_out_dir=d)[(1, 1, 0)][0]
        figures["genslab_IrO2_110.png"] = one(d, "*.png")
        d = f"{tmp}/g2"
        generate_slabs_for_miller(ceo2, Q_CE, (0, 0, 1), [2], bulk_name="CeO2", prefer_plane="O",
                                  bond_distances={"Ce-Ce": None, "O-O": None, "Ce-O": 2.35},
                                  candidates="best", plot=True, plot_out_dir=d)
        figures["genslab_CeO2_001_recon.png"] = one(d, "*.png")

        # cutslab: thickness series of the IrO2 (110) slab, relative and absolute
        d = f"{tmp}/c1"
        cutslab(term["atoms"][0], Q_IR, plot=True, plot_out_dir=d)
        figures["cutslab_IrO2_110.png"] = one(d, "*_cut_1_*.png")
        d = f"{tmp}/c2"
        cutslab(term["atoms"][0], Q_IR, selection="absolute", plot=True, plot_out_dir=d)
        figures["phases_IrO2_110_absolute.png"] = one(d, "*_cut_0_*.png")

        # phases: anatase (101), relative keeps the termination, shape mixes
        ta = generate_slabs_for_miller(anatase, Q_TI, (1, 0, 1), [2], bulk_name="anatase",
                                       dipole_tol=0.05)[(1, 0, 1)][0]
        d = f"{tmp}/p1"
        cutslab(ta["atoms"][0], Q_TI, dipole_tol=0.05, plot=True, plot_out_dir=d)
        figures["phases_anatase101_relative.png"] = one(d, "*_cut_0_*.png")
        d = f"{tmp}/p2"
        cutslab(ta["atoms"][0], Q_TI, dipole_tol=0.05, selection="shape", plot=True, plot_out_dir=d)
        figures["phases_anatase101_shape.png"] = one(d, "*_cut_0_*.png")

        for name, src in figures.items():
            shutil.copy(src, IMAGES / name)
            print(f"docs/images/{name}  <-  {src.name}")


if __name__ == "__main__":
    main()
