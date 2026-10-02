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


def plot_cut(
    atoms,
    planes,
    plane_names,
    bottom,
    top,
    out_png,
    axis=2,
    candidates=None,
    removed=(),
    notes=(),
    title=None,
    top_candidates=None,
):
    """
    Picture of one cut of a slab: which planes the sub-slab keeps, where it
    was cut, and which planes could have been its surfaces.

    Left panel: side view of the input slab (position along the first cell
    vector, two cells wide, against height).  Atoms of the sub-slab are
    drawn in colour, atoms cut away faded, and atoms removed by a
    reconstruction as open circles.  Right panel: every plane at its height
    with its label, coloured like the cut lines: planes the selection allows
    as the bottom surface in red, as the top surface in blue, as either in
    green; the sub-slab's bottom and top planes in bold.  Dashed lines
    mark the two cuts; *notes* go in a caption below.

    Parameters
    ----------
    atoms : Atoms
        The input slab.
    planes : list of dict
        Its planes (``indices``, ``z_center``, ``counts``), bottom to top.
    plane_names : list of str
        Label of every plane.
    bottom, top : int
        Indices of the sub-slab's bottom and top planes.
    out_png : str
        Output image path.
    axis : int
        Surface normal (cell vector index).
    candidates : set of int or None
        Planes the selection allowed as the bottom surface (as either
        surface if *top_candidates* is None).
    removed : iterable of int
        Atoms of the sub-slab's planes deleted by a reconstruction.
    notes : iterable of str
        Caption lines (selection, dipole, ...).
    title : str or None
        Figure title.
    top_candidates : set of int or None
        Planes the selection allowed as the top surface.
    """
    from ase.data import covalent_radii

    bottom_ok = set(candidates or ())
    top_ok = bottom_ok if top_candidates is None else set(top_candidates)
    removed = set(int(i) for i in removed)
    in_plane = [i for i in range(3) if i != axis]
    frac = atoms.get_scaled_positions(wrap=False)
    a1 = float(np.linalg.norm(atoms.cell[in_plane[0]]))
    x = (frac[:, in_plane[0]] % 1.0) * a1
    z = atoms.positions[:, axis]
    kept = {i for p in planes[bottom:top + 1] for i in p["indices"]} - removed

    def plane_extent(k):
        zz = z[planes[k]["indices"]]
        return zz.min(), zz.max()

    lo, hi = plane_extent(bottom)[0], plane_extent(top)[1]
    z_bottom_cut = 0.5 * (plane_extent(bottom - 1)[1] + lo) if bottom > 0 else lo - 0.6
    z_top_cut = 0.5 * (hi + plane_extent(top + 1)[0]) if top < len(planes) - 1 else hi + 0.6

    z_lo, z_hi = z.min() - 1.5, z.max() + 1.5
    height = float(np.clip(0.3 * (z_hi - z_lo), 4.0, 16.0))
    fig, (ax, lad) = plt.subplots(
        1, 2, figsize=(9.0, height), sharey=True, gridspec_kw={"width_ratios": [1.3, 1.0]}
    )
    sizes = 260.0 * covalent_radii[atoms.numbers] ** 2
    gone = np.array([i in removed for i in range(len(atoms))], dtype=bool)
    keep = np.array([i in kept for i in range(len(atoms))], dtype=bool)
    out = ~keep & ~gone
    for shift in (0.0, a1):
        ax.scatter(x[out] + shift, z[out], s=sizes[out], c="#d9d9d9", edgecolors="#bfbfbf",
                   linewidths=0.5, zorder=1)
        ax.scatter(x[keep] + shift, z[keep], s=sizes[keep], c=jmol_colors[atoms.numbers[keep]],
                   edgecolors="black", linewidths=0.5, zorder=2)
        ax.scatter(x[gone] + shift, z[gone], s=sizes[gone], facecolors="none",
                   edgecolors=jmol_colors[atoms.numbers[gone]], linewidths=1.4, zorder=3)
    ax.axvline(a1, color="#999999", lw=0.8, ls=":")
    ax.set_xlim(-0.6, 2 * a1 + 0.6)
    ax.set_xlabel("position along the first cell vector (Å), two cells")
    ax.set_ylabel("height z (Å)")

    # Labels spread apart where planes are close, with leader lines.
    zc = [p["z_center"] for p in planes]
    gap = 1.35 * 8.5 / 72.0 * (z_hi - z_lo) / (0.8 * height)
    order = np.argsort(zc)
    spread = _spread([zc[k] for k in order], gap)
    label_z = np.empty(len(planes))
    label_z[order] = spread
    for k, plane in enumerate(planes):
        inside = bottom <= k <= top
        surface = k in (bottom, top)
        if k in bottom_ok and k in top_ok:
            color = "#2ca02c"
        elif k in bottom_ok:
            color = "#d62728"
        elif k in top_ok:
            color = "#1f77b4"
        else:
            color = "black" if inside else "#b0b0b0"
        lad.plot([0.0, 0.12], [zc[k], zc[k]], color=color, lw=2.4 if surface else 1.0)
        lad.plot([0.12, 0.2], [zc[k], label_z[k]], color=color, lw=0.6)
        n_out = sum(1 for i in plane["indices"] if i in removed)
        text = plane_names[k]
        if n_out and inside:
            text += f"  ({len(plane['indices']) - n_out} of {len(plane['indices'])} atoms kept)"
        if k == bottom == top:
            text += "   \u25c0 bottom and top surface"
        elif k == bottom:
            text += "   \u25c0 bottom surface"
        elif k == top:
            text += "   \u25c0 top surface"
        lad.text(0.22, label_z[k], text, va="center", fontsize=8.5, color=color,
                 fontweight="bold" if surface else "normal")
    for panel in (ax, lad):
        panel.axhline(z_bottom_cut, color="red", ls="--", lw=1.2)
        panel.axhline(z_top_cut, color="blue", ls="--", lw=1.2)
    lad.text(0.99, z_bottom_cut, "bottom cut", color="red", fontsize=8, ha="right", va="top",
             transform=lad.get_yaxis_transform())
    lad.text(0.99, z_top_cut, "top cut", color="blue", fontsize=8, ha="right", va="bottom",
             transform=lad.get_yaxis_transform())
    lad.set_xlim(0.0, 1.0)
    lad.set_xticks([])
    for side in ("top", "right", "bottom"):
        lad.spines[side].set_visible(False)
    ax.set_ylim(z_lo, z_hi)

    legend = ("planes allowed as surface: red bottom, blue top, green either;  "
              "grey: cut away")
    if removed:
        legend += ";  open circles: removed by the reconstruction"
    caption = "\n".join(list(notes) + [legend])
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.text(0.01, 0.0, caption, fontsize=8.5, va="top", ha="left", family="monospace")
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
