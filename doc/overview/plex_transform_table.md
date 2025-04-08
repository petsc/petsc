(plex_transform_table)=

# Summary of Unstructured Mesh Transformations

```{eval-rst}
.. list-table::
   :widths: auto
   :align: center
   :header-rows: 1

   * -
     - ``DMPlexTransformType``
     - Accepts Active Label
     - Description
   * - Mesh filtering
     - transform_filter
     - Yes
     - Preserve a subset of the mesh marked by a `DMLabel`
   * - Regular Refinement
     - refine_regular
     - No
     - Splits all k-cells into :math:`2^k` pieces
   * - Alfeld Refinement
     - refine_alfeld
     - No
     - Barycentric refinement for simplicies
   * - Skeleton-based Refinement (SBR)
     - refine_sbr
     - Yes
     - Simplicial refinement from Plaza and Carey
   * - 1D Refinement
     - refine_1d
     - No
     - Optimized refinement for 1D meshes that preserves the canonical ordering
   * - Simplex-to-Box transform
     - refine_tobox
     - No
     - Replaces each simplex cell with :math:`2^d` box cells
   * - Box-to-Simplex transform
     - refine_tosimplex
     - No
     - Replaces each box cell with simplex cells
   * - Mesh extrusion
     - extrude
     - Yes
     - Extrude n layers of cells from a surface
   * - Boundary Layer Extrusion
     - refine_boundary_layer
     - Yes
     - Creates n layers of tensor cells along marked boundaries
   * - Cohesive cell extrusion
     - cohesive_extrude
     - Yes
     - Extrude a layer of cells into a mesh from an internal surface
```
