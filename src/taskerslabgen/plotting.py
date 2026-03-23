import numpy as np
import matplotlib.pyplot as plt
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors

from .core import identify_planes


def _composition_label(counts):
    """Build a short composition string like '2Ir+4O' from a counts dict."""
    parts = []
    for Z in sorted(counts):
        sym = chemical_symbols[Z]
        c = counts[Z]
        parts.append(f"{c}{sym}" if c > 1 else sym)
    return "+".join(parts)


def plot_unitcell_atoms(
    atoms_z,
    L,
    miller,
    out_png="atoms_z_unitcell.png",
    plane_tol=0.1,
    planes=None,
    zbot=None,
    ztop=None,
    dipole=None,
    matched_planes=None,
    plane_names=None,
    title=None,
):
    """
    Plot atoms along the stacking axis with plane annotations.

    Parameters
    ----------
    matched_planes : set of int or None
        Plane indices (into the sorted planes list) that match a reference
        termination.  Matched planes are drawn in green, others in gray.
        When None, all planes are gray (original behaviour).
    plane_names : list of str or None
        Per-plane symbolic names (e.g. "P0", "P1-recon").  When provided,
        used instead of the auto-generated "P{i}" labels.
    title : str or None
        Custom title for the plot.  When None the default Miller-index
        based title is used.
    """
    z_uc = atoms_z[:, 1] % L
    z_uc_types = atoms_z[:, 0].astype(int)
    z_uc_colors = jmol_colors[z_uc_types]
    z_uc_y = np.zeros_like(z_uc)

    z_tol_uc = 0.02 * L
    offset_step_uc = 0.06
    sorted_uc_idx = np.argsort(z_uc)
    group_uc = [sorted_uc_idx[0]]
    for idx in sorted_uc_idx[1:]:
        if abs(z_uc[idx] - z_uc[group_uc[-1]]) <= z_tol_uc:
            group_uc.append(idx)
        else:
            n = len(group_uc)
            offsets = (np.arange(n) - (n - 1) / 2) * offset_step_uc
            z_uc_y[group_uc] = offsets
            group_uc = [idx]

    if group_uc:
        n = len(group_uc)
        offsets = (np.arange(n) - (n - 1) / 2) * offset_step_uc
        z_uc_y[group_uc] = offsets

    if planes is None:
        planes = identify_planes(atoms_z, L, plane_tol=plane_tol)

    planes_sorted = sorted(planes, key=lambda p: p["z_center"] % L)

    fig, ax = plt.subplots(figsize=(10, 3.0))
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.8)
    ax.axvline(L, color="black", lw=1.0, alpha=0.8)
    ax.scatter(z_uc, z_uc_y, c=z_uc_colors, s=50, alpha=0.85)

    for i, plane in enumerate(planes_sorted):
        zc = plane["z_center"] % L
        q_total = plane["q_total"]

        if matched_planes is not None and i in matched_planes:
            line_color = "#2ca02c"
            text_color = "#2ca02c"
            line_alpha = 0.9
            lw = 1.6
        else:
            line_color = "gray"
            text_color = "gray"
            line_alpha = 0.7
            lw = 1.0

        ax.axvline(zc, color=line_color, lw=lw, alpha=line_alpha, zorder=1)

        comp = _composition_label(plane["counts"])
        id_label = plane_names[i] if plane_names is not None else f"P{i}"
        charge_label = f"Q={q_total:.2f}"

        y_top = 10 + (i % 2) * 14
        ax.annotate(
            f"{id_label}  {comp}",
            xy=(zc, 0.0),
            xytext=(0, y_top),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=5,
            color=text_color,
            rotation=45,
        )
        ax.annotate(
            charge_label,
            xy=(zc, 0.0),
            xytext=(0, -8 - (i % 2) * 10),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=5,
            color=text_color,
        )

    if zbot is not None or ztop is not None:
        offset = 0.01 * L
        zbot_plot = zbot % L if zbot is not None else None
        ztop_plot = ztop % L if ztop is not None else None
        if zbot_plot is not None and ztop_plot is not None and abs(zbot_plot - ztop_plot) < 1e-6:
            zbot_plot = zbot_plot + offset
            ztop_plot = ztop_plot - offset
        if zbot_plot is not None:
            ax.axvline(
                zbot_plot, color="red", lw=1.2, linestyle="--", alpha=0.6, zorder=2, label="bottom cut"
            )
        if ztop_plot is not None:
            ax.axvline(
                ztop_plot, color="blue", lw=1.2, linestyle="--", alpha=0.6, zorder=2, label="top cut"
            )
        ax.legend(loc="upper right")

    if dipole is not None:
        ax.annotate(
            f"mu = {dipole:+.4e}",
            xy=(0.99, 0.04),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=10,
            color="black",
        )

    ax.set_yticks([])
    max_abs_y_uc = np.max(np.abs(z_uc_y)) if len(z_uc_y) else 0.1
    ax.set_ylim(-max_abs_y_uc - 0.3, max_abs_y_uc + 0.4)
    ax.set_xlabel("z (Å)")
    ax.set_title(title if title is not None else f"Atoms along z (Miller index {miller})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)
