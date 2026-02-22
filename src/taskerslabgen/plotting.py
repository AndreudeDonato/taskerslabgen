import numpy as np
import matplotlib.pyplot as plt
from ase.data.colors import jmol_colors

from .core import identify_planes


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
):
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

    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.axvline(0.0, color="black", lw=1.0, alpha=0.8)
    ax.axvline(L, color="black", lw=1.0, alpha=0.8)
    ax.scatter(z_uc, z_uc_y, c=z_uc_colors, s=50, alpha=0.85)
    for plane in planes:
        zc = plane["z_center"] % L
        q_total = plane["q_total"]
        ax.axvline(zc, color="gray", lw=1.0, alpha=0.7, zorder=1)
        ax.annotate(
            f"{q_total:.2f}",
            xy=(zc, 0.0),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            color="gray",
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
    ax.set_ylim(-max_abs_y_uc - 0.1, max_abs_y_uc + 0.1)
    ax.set_xlabel("z (Å)")
    ax.set_title(f"Unit-cell atoms along z (Miller index {miller})")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    #plt.show()
