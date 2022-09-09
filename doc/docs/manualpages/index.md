# Manual Pages

* [API Changes in each release](../changes/index.rst)
* [MPI](http://www.mpich.org/static/docs/latest/)
* [Vector Operations (Vec)](Vec/index.md)

  -  [Defining your own mathematical functions (PF)](PF/index.md)
* [Matrix Operations (Mat)](Mat/index.md)

  -  [Matrix colorings (MatColoring), orderings (MatOrdering), and partitionings (MatPartitioning)](MatOrderings/index.md)
  -  [Finite difference computation of Jacobians (MatFD)](MatFD/index.md)
* Data and grid management

  - [Index sets (IS)](IS/index.md)
  - [Star Forest Communication (PetscSF)](PetscSF/index.md)
  -  [Section Data Layout (PetscSection)](PetscSection/index.md)
* [Data Management (DM) between Vec and Mat, and the distributed mesh datastructures](DM/index.md)

  -  [Structured Grids (DMDA)](DMDA/index.md)
  -  [Staggered, Structured Grids (DMStag)](DMStag/index.md)
  -  [Unstructured Grids (DMPlex)](DMPlex/index.md)
  -  [Graphs and Networks (DMNetwork)](DMNetwork/index.md)
  -  [A Forest of Trees (DMFOREST)](DMForest/index.md)
  -  [DMPATCH](DMPatch/index.md)
  -  [Particles (DMSWARM)](DMSwarm/index.md)
  -  [DMMOAB](DMMOAB/index.md)
  -  [Selecting Parts of Meshes (DMLABEL)](DMLabel/index.md)

* [Discretization Technology (DT)](DT/index.md)

  -  [Function Space Technology (PetscSpace)](SPACE/index.md)
  -  [Dual Space Technology (PetscDualSpace)](DUALSPACE/index.md)
  -  [Finite Element Technology (PetscFE)](FE/index.md)
  -  [Finite Volume Technology (PetscFV)](FV/index.md)
* [Application Orderings (AO)](AO/index.md)
* [Linear Solvers (KSP)](KSP/index.md)

  -  [Preconditioners (PC)](PC/index.md)
  -  [Krylov Subspace Methods (KSP)](KSP/index.md)
* [Nonlinear Solvers (SNES)](SNES/index.md)

  - [Full Approximation Storage (FAS) nonlinear multigrid](SNESFAS/index.md)
  - [Matrix-free nonlinear solvers (MATMFFD)](SNES/MatCreateSNESMF.md)
* [Time Stepping (TS) ODE solvers](TS/index.md)

  -  [Sensitivity analysis](Sensitivity/index.md)
  -  [Method of characteristics](Characteristic/index.md)
* [Optimization Solvers (Tao)](Tao/index.md)

  -  [Optimization LineSearch Solver (TaoLineSearch)](TaoLineSearch/index.md)
* Utilities

  -  [Viewing Objects](Viewer/index.md)
  -  [Graphics (Draw)](Draw/index.md)
  -  [System Routines (Options, IO, utilities)](Sys/index.md)
  -  [Profiling and Logging](Profiling/index.md)

* [Single Index of all Manual Pages](singleindex.md)

The manual pages are split into four categories; we recommend that
you begin with basic functionality and then gradually explore more
sophisticated library features.

- *Beginner* - Basic usage
- *Intermediate* - Setting options for algorithms and data structures
- *Advanced* - Setting more advanced options and customization
- *Developer* - Interfaces intended primarily for library developers

## List of all Manual Sections

```{toctree}
:maxdepth: 1

AO/index
Characteristic/index
DM/index
DMDA/index
DMForest/index
DMLabel/index
DMMOAB/index
DMNetwork/index
DMPatch/index
DMPlex/index
DMPRODUCT/index
DMStag/index
DMSwarm/index
DT/index
DUALSPACE/index
Draw/index
FE/index
FV/index
IS/index
KSP/index
LANDAU/index
Mat/index
MatFD/index
MatOrderings/index
PC/index
PF/index
PetscSF/index
PetscSection/index
Profiling/index
SNES/index
SNESFAS/index
SPACE/index
Sensitivity/index
Sys/index
TS/index
Tao/index
TaoLineSearch/index
Vec/index
Viewer/index
```
