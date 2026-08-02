# Automated Generation of Non-Polar Slab Models for Surface Convergence Studies Using ASE

Andreu A. de Donato

Constructing slab models for density functional theory (DFT) surface
calculations requires careful attention to stoichiometry, charge neutrality,
and dipole cancellation — particularly for polar (Tasker III) surfaces where
surface reconstruction is necessary. Existing tools (e.g. pymatgen cleavage
and Tasker-2 half-ion moves, Surfaxe zero-dipole filters, and crystallographic
nonpolar-slab algorithms) address parts of this problem, but researchers often
still lack an ASE-native path that prefers stoichiometric nonpolar cuts, scores
Tasker III reconstructions when needed, and builds thickness-convergence series
with consistent surface chemistry.

We present `taskerslabgen`, an open-source Python library built entirely on
ASE `Atoms` objects that automates generation of stoichiometric non-polar slab
terminations for arbitrary Miller indices. The library classifies surfaces
according to Tasker's criteria using formal or computed charges, enumerates
stoichiometric cut sequences (Tasker I/II when possible), and for Tasker III
surfaces performs systematic symmetric reconstruction by scoring deletion
patterns based on broken bonds and pairwise distance metrics that penalize
deviations from target bond lengths while maximizing separation between
species that should not bond. All valid terminations are presented to the user
with stacking-axis visualizations showing plane compositions, charges, and
symbolic identifiers, enabling informed selection rather than black-box
automation.

A key feature is the tandem workflow for convergence testing: from a single
bulk structure, the code first generates a thick reference slab with the
desired termination, then systematically cuts it into a series of thinner
slabs of decreasing thickness — all preserving the same surface termination
and reconstruction pattern. This produces ready-to-use convergence sets
directly as ASE `Atoms` objects, compatible with any ASE-based DFT calculator
interface.

We demonstrate the workflow on CeO2 fluorite and IrO2 rutile surfaces,
including Tasker III reconstructed (001) and Tasker I/II (110) terminations
with stacking-aware plane identification for supercells.
