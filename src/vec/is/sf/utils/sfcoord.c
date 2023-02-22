#include <petsc/private/sfimpl.h> /*I  "petscsf.h"   I*/

static PetscErrorCode GetBoundingBox_Internal(PetscInt npoints, PetscInt dim, const PetscReal *coords, PetscReal *bbox)
{
  PetscFunctionBegin;
  for (PetscInt d = 0; d < dim; d++) {
    bbox[0 * dim + d] = PETSC_MAX_REAL;
    bbox[1 * dim + d] = PETSC_MIN_REAL;
  }
  for (PetscInt i = 0; i < npoints; i++) {
    for (PetscInt d = 0; d < dim; d++) {
      bbox[0 * dim + d] = PetscMin(bbox[0 * dim + d], coords[i * dim + d]);
      bbox[1 * dim + d] = PetscMax(bbox[1 * dim + d], coords[i * dim + d]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscBool IntersectBoundingBox_Internal(PetscInt dim, const PetscReal *a, const PetscReal *b, PetscReal tol)
{
  for (PetscInt d = 0; d < dim; d++) {
    if (a[1 * dim + d] + tol < b[0 * dim + d] || b[1 * dim + d] + tol < a[0 * dim + d]) return PETSC_FALSE;
  }
  return PETSC_TRUE;
}

static PetscBool InBoundingBox_Internal(PetscInt dim, const PetscReal *x, const PetscReal *bbox, PetscReal tol)
{
  for (PetscInt d = 0; d < dim; d++) {
    if (x[d] + tol < bbox[0 * dim + d] || bbox[1 * dim + d] + tol < x[d]) return PETSC_FALSE;
  }
  return PETSC_TRUE;
}

/*@
  PetscSFSetGraphFromCoordinates - Create SF by fuzzy matching leaf coordinates to root coordinates

  Collective

  Input Parameters:
+ sf - PetscSF to set graph on
. nroots - number of root coordinates
. nleaves - number of leaf coordinates
. dim - spatial dimension of coordinates
. tol - positive tolerance for matching
. rootcoords - array of root coordinates in which root i component d is [i*dim+d]
- leafcoords - array of root coordinates in which leaf i component d is [i*dim+d]

  Notes:
  The tolerance typically represents the rounding error incurred by numerically computing coordinates via
  possibly-different procedures. Passing anything from `PETSC_SMALL` to `100 * PETSC_MACHINE_EPSILON` is appropriate for
  most use cases.

  Example:
  As a motivating example, consider fluid flow in the x direction with y (distance from a wall). The spanwise direction,
  z, has periodic boundary conditions and needs some spanwise length to allow turbulent structures to develop. The
  distribution is stationary with respect to z, so you want to average turbulence variables (like Reynolds stress) over
  the z direction. It is complicated in a 3D simulation with arbitrary partitioner to uniquely number the nodes or
  quadrature point coordinates to average these quantities into a 2D plane where they will be visualized, but it's easy
  to compute the projection of each 3D point into the 2D plane.

  Suppose a 2D target mesh and 3D source mesh (logically an extrusion of the 2D, though perhaps not created in that way)
  are distributed independently on a communicator. Each rank passes its 2D target points as root coordinates and the 2D
  projection of its 3D source points as leaf coordinates. Calling `PetscSFReduceBegin()`/`PetscSFReduceEnd()` on the
  result will sum data from the 3D sources to the 2D targets.

  As a concrete example, consider three MPI ranks with targets (roots)
.vb
Rank 0: (0, 0), (0, 1)
Rank 1: (0.1, 0), (0.1, 1)
Rank 2: (0.2, 0), (0.2, 1)
.ve
  Note that targets must be uniquely owned. Suppose also that we identify the following leaf coordinates (perhaps via projection from a 3D space).
.vb
Rank 0: (0, 0), (0.1, 0), (0, 1), (0.1, 1)
Rank 1: (0, 0), (0.1, 0), (0.2, 0), (0, 1), (0.1, 1)
Rank 2: (0.1, 0), (0.2, 0), (0.1, 1), (0.2, 1)
.ve
  Leaf coordinates may be repeated, both on a rank and between ranks. This example yields the following `PetscSF` capable of reducing from sources to targets.
.vb
Roots by rank
[0]  0:   0.0000e+00   0.0000e+00   0.0000e+00   1.0000e+00
[1]  0:   1.0000e-01   0.0000e+00   1.0000e-01   1.0000e+00
[2]  0:   2.0000e-01   0.0000e+00   2.0000e-01   1.0000e+00
Leaves by rank
[0]  0:   0.0000e+00   0.0000e+00   1.0000e-01   0.0000e+00   0.0000e+00
[0]  5:   1.0000e+00   1.0000e-01   1.0000e+00
[1]  0:   0.0000e+00   0.0000e+00   1.0000e-01   0.0000e+00   2.0000e-01
[1]  5:   0.0000e+00   0.0000e+00   1.0000e+00   1.0000e-01   1.0000e+00
[1] 10:   2.0000e-01   1.0000e+00
[2]  0:   1.0000e-01   0.0000e+00   2.0000e-01   0.0000e+00   1.0000e-01
[2]  5:   1.0000e+00   2.0000e-01   1.0000e+00
PetscSF Object: 3 MPI processes
  type: basic
  [0] Number of roots=2, leaves=4, remote ranks=2
  [0] 0 <- (0,0)
  [0] 1 <- (1,0)
  [0] 2 <- (0,1)
  [0] 3 <- (1,1)
  [1] Number of roots=2, leaves=6, remote ranks=3
  [1] 0 <- (0,0)
  [1] 1 <- (1,0)
  [1] 2 <- (2,0)
  [1] 3 <- (0,1)
  [1] 4 <- (1,1)
  [1] 5 <- (2,1)
  [2] Number of roots=2, leaves=4, remote ranks=2
  [2] 0 <- (1,0)
  [2] 1 <- (2,0)
  [2] 2 <- (1,1)
  [2] 3 <- (2,1)
.ve

  Level: advanced

.seealso: `PetscSFCreate()`, `PetscSFSetGraph()`, `PetscSFCreateByMatchingIndices()`
@*/
PetscErrorCode PetscSFSetGraphFromCoordinates(PetscSF sf, PetscInt nroots, PetscInt nleaves, PetscInt dim, PetscReal tol, const PetscReal *rootcoords, const PetscReal *leafcoords)
{
  PetscReal    bbox[6], *bboxes, *target_coords;
  PetscMPIInt  size, *ranks_needed, num_ranks;
  PetscInt    *root_sizes, *root_starts;
  PetscSFNode *premote, *lremote;
  PetscSF      psf;
  MPI_Datatype unit;
  MPI_Comm     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sf, &comm));
  PetscCall(GetBoundingBox_Internal(nroots, dim, rootcoords, bbox));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscMalloc1(size * 2 * dim, &bboxes));
  PetscCallMPI(MPI_Allgather(bbox, 2 * dim, MPIU_REAL, bboxes, 2 * dim, MPIU_REAL, comm));
  PetscCall(GetBoundingBox_Internal(nleaves, dim, leafcoords, bbox));
  PetscCall(PetscMalloc1(size, &root_sizes));
  PetscCallMPI(MPI_Allgather(&nroots, 1, MPIU_INT, root_sizes, 1, MPIU_INT, comm));

  PetscCall(PetscMalloc2(size, &ranks_needed, size + 1, &root_starts));
  root_starts[0] = 0;
  num_ranks      = 0;
  for (PetscMPIInt r = 0; r < size; r++) {
    if (IntersectBoundingBox_Internal(dim, bbox, &bboxes[2 * dim * r], tol)) {
      ranks_needed[num_ranks++] = r;
      root_starts[num_ranks]    = root_starts[num_ranks - 1] + root_sizes[r];
    }
  }
  PetscCall(PetscFree(root_sizes));
  PetscCall(PetscMalloc1(root_starts[num_ranks], &premote));
  for (PetscInt i = 0; i < num_ranks; i++) {
    for (PetscInt j = root_starts[i]; j < root_starts[i + 1]; j++) {
      premote[j].rank  = ranks_needed[i];
      premote[j].index = j - root_starts[i];
    }
  }
  PetscCall(PetscSFCreate(comm, &psf));
  PetscCall(PetscSFSetGraph(psf, nroots, root_starts[num_ranks], NULL, PETSC_USE_POINTER, premote, PETSC_USE_POINTER));
  PetscCall(PetscMalloc1(root_starts[num_ranks] * dim, &target_coords));
  PetscCallMPI(MPI_Type_contiguous(dim, MPIU_REAL, &unit));
  PetscCallMPI(MPI_Type_commit(&unit));
  PetscCall(PetscSFBcastBegin(psf, unit, rootcoords, target_coords, MPI_REPLACE));
  PetscCall(PetscSFBcastEnd(psf, unit, rootcoords, target_coords, MPI_REPLACE));
  PetscCallMPI(MPI_Type_free(&unit));
  PetscCall(PetscSFDestroy(&psf));

  // Condense targets to only those that lie within our bounding box
  PetscInt num_targets = 0;
  for (PetscInt i = 0; i < root_starts[num_ranks]; i++) {
    if (InBoundingBox_Internal(dim, &target_coords[i * dim], bbox, tol)) {
      premote[num_targets] = premote[i];
      for (PetscInt d = 0; d < dim; d++) target_coords[num_targets * dim + d] = target_coords[i * dim + d];
      num_targets++;
    }
  }
  PetscCall(PetscFree(bboxes));
  PetscCall(PetscFree2(ranks_needed, root_starts));

  PetscCall(PetscMalloc1(nleaves, &lremote));
  for (PetscInt i = 0; i < nleaves; i++) {
    for (PetscInt j = 0; j < num_targets; j++) {
      PetscReal sum = 0;
      for (PetscInt d = 0; d < dim; d++) sum += PetscSqr(leafcoords[i * dim + d] - target_coords[j * dim + d]);
      if (sum < tol * tol) {
        lremote[i] = premote[j];
        goto matched;
      }
    }
    switch (dim) {
    case 1:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "No target found for leaf coordinate %g", (double)leafcoords[i * dim + 0]);
    case 2:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "No target found for leaf coordinate (%g, %g)", (double)leafcoords[i * dim + 0], (double)leafcoords[i * dim + 1]);
    case 3:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "No target found for leaf coordinate (%g, %g, %g)", (double)leafcoords[i * dim + 0], (double)leafcoords[i * dim + 1], (double)leafcoords[i * dim + 2]);
    }
  matched:;
  }
  PetscCall(PetscFree(premote));
  PetscCall(PetscFree(target_coords));
  PetscCall(PetscSFSetGraph(sf, nroots, nleaves, NULL, PETSC_USE_POINTER, lremote, PETSC_OWN_POINTER));
  PetscFunctionReturn(PETSC_SUCCESS);
}
