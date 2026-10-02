from pathlib import Path

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
    plane_tol=None,
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

    Produces a 1-D projection of the unit cell showing atom positions,
    plane labels, compositions, charges, and (optionally) cut positions
    and dipole value.

    Parameters
    ----------
    atoms_z : ndarray, shape (N, 3)
        ``[atomic_number, z_position, charge]`` matrix.
    L : float
        Lattice-plane spacing (angstrom).
    miller : tuple of int
        Miller index, used in the default title.
    out_png : str
        Output image path.
    plane_tol : float or None
        Tolerance for plane identification (only used if *planes* is
        None).  ``None`` uses the default 0.1 Å.
    planes : list of dict or None
        Pre-computed planes.  When None, :func:`identify_planes` is
        called internally.
    zbot : float or None
        z-coordinate of the bottom cut line (dashed red).
    ztop : float or None
        z-coordinate of the top cut line (dashed blue).
    dipole : float or None
        Dipole value shown in the lower-right corner.
    matched_planes : set of int or None
        Plane indices that match a reference termination.  Matched
        planes are drawn in green, others in gray.
    plane_names : list of str or None
        Per-plane labels (e.g. ``"O4"``, ``"O4-recon"``).
    title : str or None
        Custom title.  When None the default Miller-index title is used.
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
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _spread(values, gap):
    """Positions near *values* (sorted) that are at least *gap* apart."""
    pos = list(values)
    for _ in range(50):
        moved = False
        for k in range(1, len(pos)):
            d = pos[k] - pos[k - 1]
            if d < gap:
                shift = 0.5 * (gap - d)
                pos[k - 1] -= shift
                pos[k] += shift
                moved = True
        if not moved:
            break
    return pos


def plot_slab(
    atoms,
    planes,
    plane_names,
    out_png,
    bottom=0,
    top=None,
    axis=2,
    bottom_ok=None,
    top_ok=None,
    removed=(),
    title=None,
):
    """
    Side view of a slab with its planes labelled: the plot of both
    generate_slabs_for_miller and cutslab.

    Left: the atoms, position along the first cell vector (two cells) against
    height.  Right: the label of every plane at its height.  The planes
    *bottom* to *top* are the slab shown: its two surfaces are in bold, and
    planes outside it (a sub-slab cut from a thicker one) are grey, with the
    cuts as dashed lines (red bottom, blue top).  Planes in *bottom_ok* /
    *top_ok* (those a selection allowed as bottom / top surface) are coloured
    like the cut lines, green if allowed as either.  Atoms in *removed*
    (deleted by a reconstruction) are drawn as open circles.
    """
    from ase.data import covalent_radii

    n = len(planes)
    top = n - 1 if top is None else top
    bottom_ok = set(bottom_ok or ())
    top_ok = set(top_ok or ())
    removed = set(int(i) for i in removed)
    in_plane = [i for i in range(3) if i != axis]
    frac = atoms.get_scaled_positions(wrap=False)
    a1 = float(np.linalg.norm(atoms.cell[in_plane[0]]))
    x = (frac[:, in_plane[0]] % 1.0) * a1
    z = atoms.positions[:, axis]
    kept = {i for p in planes[bottom:top + 1] for i in p["indices"]} - removed

    z_lo, z_hi = z.min() - 1.0, z.max() + 1.0
    height = float(np.clip(0.28 * (z_hi - z_lo), 3.0, 14.0))
    fig, (ax, lad) = plt.subplots(1, 2, figsize=(7.0, height), sharey=True,
                                  gridspec_kw={"width_ratios": [1.4, 1.0]})
    sizes = 200.0 * covalent_radii[atoms.numbers] ** 2
    gone = np.array([i in removed for i in range(len(atoms))], dtype=bool)
    keep = np.array([i in kept for i in range(len(atoms))], dtype=bool)
    out = ~keep & ~gone
    for shift in (0.0, a1):
        ax.scatter(x[out] + shift, z[out], s=sizes[out], c="#dddddd", edgecolors="#c4c4c4",
                   linewidths=0.4, zorder=1)
        ax.scatter(x[keep] + shift, z[keep], s=sizes[keep], c=jmol_colors[atoms.numbers[keep]],
                   edgecolors="black", linewidths=0.4, zorder=2)
        ax.scatter(x[gone] + shift, z[gone], s=sizes[gone], facecolors="none",
                   edgecolors=jmol_colors[atoms.numbers[gone]], linewidths=1.2, zorder=3)
    ax.set_xlim(-0.6, 2 * a1 + 0.6)
    ax.set_xticks([])
    ax.set_ylabel("z (Å)")

    zc = [p["z_center"] for p in planes]
    gap = 1.3 * 8.0 / 72.0 * (z_hi - z_lo) / (0.85 * height)
    order = np.argsort(zc)
    label_z = np.empty(n)
    label_z[order] = _spread([zc[k] for k in order], gap)
    for k in range(n):
        inside = bottom <= k <= top
        if k in bottom_ok and k in top_ok:
            color = "#2ca02c"
        elif k in bottom_ok:
            color = "#d62728"
        elif k in top_ok:
            color = "#1f77b4"
        else:
            color = "black" if inside else "#aaaaaa"
        surface = k in (bottom, top)
        lad.plot([0.0, 0.1, 0.18], [zc[k], zc[k], label_z[k]], color=color, lw=0.8)
        lad.text(0.2, label_z[k], plane_names[k], va="center", fontsize=8, color=color,
                 fontweight="bold" if surface else "normal")
    if bottom > 0 or top < n - 1:
        def extent(k):
            zz = z[planes[k]["indices"]]
            return zz.min(), zz.max()

        lo, hi = extent(bottom)[0], extent(top)[1]
        cuts = [(0.5 * (extent(bottom - 1)[1] + lo) if bottom > 0 else lo - 0.5, "#d62728"),
                (0.5 * (hi + extent(top + 1)[0]) if top < n - 1 else hi + 0.5, "#1f77b4")]
        for zcut, color in cuts:
            for panel in (ax, lad):
                panel.axhline(zcut, color=color, ls="--", lw=1.0)
    lad.set_xlim(0.0, 1.0)
    lad.axis("off")
    ax.set_ylim(z_lo, z_hi)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
