---
name: pde-visualization
description: >-
  Shared VTK/ParaView (+ matplotlib) knowledge base for simulation
  visualization and analysis: volume rendering, isosurfacing, slices,
  streamlines/pathlines, glyphs, warping, mesh rendering; perceptually sound
  colormaps and out-of-range highlighting; parallel data formats (VTU/VTS,
  HDF5+XDMF); in-situ vs post-hoc pipelines; and analytics (NaN/Inf detection,
  range checks, solver- and mesh-convergence plots, conservation history). Load
  when specifying visualizations or writing standalone post-processors
  (visualization agent), or implementing in-situ/output code (code-generation
  agent).
---

# PDE visualization & analysis (VTK / ParaView)

The single knowledge base loaded by both the `visualization` agent (to SPECIFY,
and to IMPLEMENT standalone post-processors) and the `code-generation` agent (to
IMPLEMENT in-situ/output code inside the simulation binary), so specs and code
share vocabulary.

## Rendering techniques
- **Volume rendering** — scalar fields via a transfer function; for interactive
  human exploration of 3D structure.
- **Isosurfaces** — level sets of a scalar; pick isovalues from the field range.
- **Slices/clips** — planar cross-sections through 3D data.
- **Streamlines/pathlines** — vector-field integral curves (steady vs unsteady).
- **Glyphs** — oriented/scaled markers for vectors/tensors.
- **Warp** — deform geometry by a scalar/vector (e.g. displacement).
- **Mesh rendering** — the grid itself (quality, refinement, partition).

## Color & highlighting
- Use perceptually uniform colormaps (e.g. viridis); reserve diverging maps for
  signed fields about a meaningful midpoint. Avoid rainbow.
- **Out-of-range highlighting**: mark values outside a nominal range with
  saturated above/below colors so violations are obvious.

## Data formats & scale
- Small/serial: `.vtu`/`.vts`. At scale/parallel: **HDF5 + XDMF** (ParaView
  reads it, supports partial/parallel reads).
- **Post-hoc**: write files, explore in ParaView / `pvpython`.
- **In-situ** (Catalyst): render during the run when data is too big to store —
  relevant for large campaigns.

## Analytics (factual)
- **NaN/Inf detection**: scan solution fields; report location (cell/step).
- **Range checks / sanity checks**: verify fields stay within physical bounds;
  support user-specified diagnostics.
- **Solver-convergence plots**: residual vs iteration.
- **Mesh-convergence plots**: error vs h across a refinement family (log-log
  with reference slopes).
- **Conservation history**: invariant vs time.
- Particle tracing and other auxiliary computations feed these views.

## Division of labor
The split is by WHERE the code lives:
- **In-situ / output code** (solution `VecView` → VTU/HDF5/XDMF, Catalyst
  rendering) lives inside the simulation binary → `code-generation` implements
  it, from Vis Spec items marked `execution: "in_situ"`.
- **Standalone post-processors** that read the output files (pvpython/ParaView,
  matplotlib, pyvista) → the `visualization` agent writes AND runs them, from
  items marked `execution: "post_hoc"`.
Keep heavy rendering in VTK/ParaView; use matplotlib for line/convergence plots.
