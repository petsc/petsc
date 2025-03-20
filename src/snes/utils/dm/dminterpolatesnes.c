#include <petsc/private/dmpleximpl.h> /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>   /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

/*@
  DMInterpolationCreate - Creates a `DMInterpolationInfo` context

  Collective

  Input Parameter:
. comm - the communicator

  Output Parameter:
. ctx - the context

  Level: beginner

  Developer Note:
  The naming is incorrect, either the object should be named `DMInterpolation` or all the routines should begin with `DMInterpolationInfo`

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationDestroy()`
@*/
PetscErrorCode DMInterpolationCreate(MPI_Comm comm, DMInterpolationInfo *ctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(ctx, 2);
  PetscCall(PetscNew(ctx));

  (*ctx)->comm   = comm;
  (*ctx)->dim    = -1;
  (*ctx)->nInput = 0;
  (*ctx)->points = NULL;
  (*ctx)->cells  = NULL;
  (*ctx)->n      = -1;
  (*ctx)->coords = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationSetDim - Sets the spatial dimension for the interpolation context

  Not Collective

  Input Parameters:
+ ctx - the context
- dim - the spatial dimension

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationGetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ctx, PetscInt dim)
{
  PetscFunctionBegin;
  PetscCheck(!(dim < 1) && !(dim > 3), ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %" PetscInt_FMT, dim);
  ctx->dim = dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationGetDim - Gets the spatial dimension for the interpolation context

  Not Collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. dim - the spatial dimension

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationSetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationGetDim(DMInterpolationInfo ctx, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscAssertPointer(dim, 2);
  *dim = ctx->dim;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationSetDof - Sets the number of fields interpolated at a point for the interpolation context

  Not Collective

  Input Parameters:
+ ctx - the context
- dof - the number of fields

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationGetDof()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ctx, PetscInt dof)
{
  PetscFunctionBegin;
  PetscCheck(dof >= 1, ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %" PetscInt_FMT, dof);
  ctx->dof = dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationGetDof - Gets the number of fields interpolated at a point for the interpolation context

  Not Collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. dof - the number of fields

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationSetDof()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationGetDof(DMInterpolationInfo ctx, PetscInt *dof)
{
  PetscFunctionBegin;
  PetscAssertPointer(dof, 2);
  *dof = ctx->dof;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationAddPoints - Add points at which we will interpolate the fields

  Not Collective

  Input Parameters:
+ ctx    - the context
. n      - the number of points
- points - the coordinates for each point, an array of size `n` * dim

  Level: intermediate

  Note:
  The input coordinate information is copied into the object.

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationSetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ctx, PetscInt n, PetscReal points[])
{
  PetscFunctionBegin;
  PetscCheck(ctx->dim >= 0, ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  PetscCheck(!ctx->points, ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  ctx->nInput = n;

  PetscCall(PetscMalloc1(n * ctx->dim, &ctx->points));
  PetscCall(PetscArraycpy(ctx->points, points, n * ctx->dim));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationSetUp - Compute spatial indices for point location during interpolation

  Collective

  Input Parameters:
+ ctx                 - the context
. dm                  - the `DM` for the function space used for interpolation
. redundantPoints     - If `PETSC_TRUE`, all processes are passing in the same array of points. Otherwise, points need to be communicated among processes.
- ignoreOutsideDomain - If `PETSC_TRUE`, ignore points outside the domain, otherwise return an error

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationSetUp(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain)
{
  MPI_Comm           comm = ctx->comm;
  PetscScalar       *a;
  PetscInt           p, q, i;
  PetscMPIInt        rank, size;
  Vec                pointVec;
  PetscSF            cellSF;
  PetscLayout        layout;
  PetscReal         *globalPoints;
  PetscScalar       *globalPointsScalar;
  const PetscInt    *ranges;
  PetscMPIInt       *counts, *displs;
  const PetscSFNode *foundCells;
  const PetscInt    *foundPoints;
  PetscMPIInt       *foundProcs, *globalProcs, in;
  PetscInt           n, N, numFound;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCheck(ctx->dim >= 0, comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetLocalSize(layout, n));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(PetscLayoutGetSize(layout, &N));
    /* Communicate all points to all processes */
    PetscCall(PetscMalloc3(N * ctx->dim, &globalPoints, size, &counts, size, &displs));
    PetscCall(PetscLayoutGetRanges(layout, &ranges));
    for (p = 0; p < size; ++p) {
      PetscCall(PetscMPIIntCast((ranges[p + 1] - ranges[p]) * ctx->dim, &counts[p]));
      PetscCall(PetscMPIIntCast(ranges[p] * ctx->dim, &displs[p]));
    }
    PetscCall(PetscMPIIntCast(n * ctx->dim, &in));
    PetscCallMPI(MPI_Allgatherv(ctx->points, in, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm));
  } else {
    N            = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout          = NULL;
  }
#if 0
  PetscCall(PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs));
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
  #if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(N * ctx->dim, &globalPointsScalar));
  for (i = 0; i < N * ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
  #else
  globalPointsScalar = globalPoints;
  #endif
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N * ctx->dim, globalPointsScalar, &pointVec));
  PetscCall(PetscMalloc2(N, &foundProcs, N, &globalProcs));
  for (p = 0; p < N; ++p) foundProcs[p] = size;
  cellSF = NULL;
  PetscCall(DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF));
  PetscCall(PetscSFGetGraph(cellSF, NULL, &numFound, &foundPoints, &foundCells));
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  PetscCallMPI(MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm));
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      PetscCheck(ignoreOutsideDomain, comm, PETSC_ERR_PLIB, "Point %" PetscInt_FMT ": %g %g %g not located in mesh", p, (double)globalPoints[p * ctx->dim + 0], (double)(ctx->dim > 1 ? globalPoints[p * ctx->dim + 1] : 0),
                 (double)(ctx->dim > 2 ? globalPoints[p * ctx->dim + 2] : 0));
      if (rank == 0) ++ctx->n;
    } else if (globalProcs[p] == rank) ++ctx->n;
  }
  /* Create coordinates vector and array of owned cells */
  PetscCall(PetscMalloc1(ctx->n, &ctx->cells));
  PetscCall(VecCreate(comm, &ctx->coords));
  PetscCall(VecSetSizes(ctx->coords, ctx->n * ctx->dim, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(ctx->coords, ctx->dim));
  PetscCall(VecSetType(ctx->coords, VECSTANDARD));
  PetscCall(VecGetArray(ctx->coords, &a));
  for (p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = globalPoints[p * ctx->dim + d];
      ctx->cells[q] = foundCells[q].index;
      ++q;
    }
    if (globalProcs[p] == size && rank == 0) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = 0.;
      ctx->cells[q] = -1;
      ++q;
    }
  }
  PetscCall(VecRestoreArray(ctx->coords, &a));
#if 0
  PetscCall(PetscFree3(foundCells,foundProcs,globalProcs));
#else
  PetscCall(PetscFree2(foundProcs, globalProcs));
  PetscCall(PetscSFDestroy(&cellSF));
  PetscCall(VecDestroy(&pointVec));
#endif
  if ((void *)globalPointsScalar != (void *)globalPoints) PetscCall(PetscFree(globalPointsScalar));
  if (!redundantPoints) PetscCall(PetscFree3(globalPoints, counts, displs));
  PetscCall(PetscLayoutDestroy(&layout));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationGetCoordinates - Gets a `Vec` with the coordinates of each interpolation point

  Collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. coordinates - the coordinates of interpolation points

  Level: intermediate

  Note:
  The local vector entries correspond to interpolation points lying on this process, according to the associated `DM`.
  This is a borrowed vector that the user should not destroy.

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationGetCoordinates(DMInterpolationInfo ctx, Vec *coordinates)
{
  PetscFunctionBegin;
  PetscAssertPointer(coordinates, 2);
  PetscCheck(ctx->coords, ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  *coordinates = ctx->coords;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationGetVector - Gets a `Vec` which can hold all the interpolated field values

  Collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. v - a vector capable of holding the interpolated field values

  Level: intermediate

  Note:
  This vector should be returned using `DMInterpolationRestoreVector()`.

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationRestoreVector()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationGetVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscFunctionBegin;
  PetscAssertPointer(v, 2);
  PetscCheck(ctx->coords, ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  PetscCall(VecCreate(ctx->comm, v));
  PetscCall(VecSetSizes(*v, ctx->n * ctx->dof, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(*v, ctx->dof));
  PetscCall(VecSetType(*v, VECSTANDARD));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationRestoreVector - Returns a `Vec` which can hold all the interpolated field values

  Collective

  Input Parameters:
+ ctx - the context
- v   - a vector capable of holding the interpolated field values

  Level: intermediate

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationGetVector()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationRestoreVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscFunctionBegin;
  PetscAssertPointer(v, 2);
  PetscCheck(ctx->coords, ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  PetscCall(VecDestroy(v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMInterpolate_Segment_Private(DMInterpolationInfo ctx, DM dm, PetscInt p, Vec xLocal, Vec v)
{
  const PetscInt     c   = ctx->cells[p];
  const PetscInt     dof = ctx->dof;
  PetscScalar       *x   = NULL;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscReal          v0, J, invJ, detJ, xir[1];
  PetscInt           xSize, comp;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0, &J, &invJ, &detJ));
  PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double)detJ, c);
  xir[0] = invJ * PetscRealPart(coords[p] - v0);
  PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
  if (2 * dof == xSize) {
    for (comp = 0; comp < dof; ++comp) a[p * dof + comp] = x[0 * dof + comp] * (1 - xir[0]) + x[1 * dof + comp] * xir[0];
  } else if (dof == xSize) {
    for (comp = 0; comp < dof; ++comp) a[p * dof + comp] = x[0 * dof + comp];
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input closure size %" PetscInt_FMT " must be either %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 2 * dof, dof);
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMInterpolate_Triangle_Private(DMInterpolationInfo ctx, DM dm, PetscInt p, Vec xLocal, Vec v)
{
  const PetscInt     c = ctx->cells[p];
  PetscScalar       *x = NULL;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscReal         *v0, *J, *invJ, detJ;
  PetscReal          xi[4];

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(ctx->dim, &v0, ctx->dim * ctx->dim, &J, ctx->dim * ctx->dim, &invJ));
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double)detJ, c);
  PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
  for (PetscInt comp = 0; comp < ctx->dof; ++comp) a[p * ctx->dof + comp] = x[0 * ctx->dof + comp];
  for (PetscInt d = 0; d < ctx->dim; ++d) {
    xi[d] = 0.0;
    for (PetscInt f = 0; f < ctx->dim; ++f) xi[d] += invJ[d * ctx->dim + f] * 0.5 * PetscRealPart(coords[p * ctx->dim + f] - v0[f]);
    for (PetscInt comp = 0; comp < ctx->dof; ++comp) a[p * ctx->dof + comp] += PetscRealPart(x[(d + 1) * ctx->dof + comp] - x[0 * ctx->dof + comp]) * xi[d];
  }
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMInterpolate_Tetrahedron_Private(DMInterpolationInfo ctx, DM dm, PetscInt p, Vec xLocal, Vec v)
{
  const PetscInt     c        = ctx->cells[p];
  const PetscInt     order[3] = {2, 1, 3};
  PetscScalar       *x        = NULL;
  PetscReal         *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscReal          xi[4];

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(ctx->dim, &v0, ctx->dim * ctx->dim, &J, ctx->dim * ctx->dim, &invJ));
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
  PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double)detJ, c);
  PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
  for (PetscInt comp = 0; comp < ctx->dof; ++comp) a[p * ctx->dof + comp] = x[0 * ctx->dof + comp];
  for (PetscInt d = 0; d < ctx->dim; ++d) {
    xi[d] = 0.0;
    for (PetscInt f = 0; f < ctx->dim; ++f) xi[d] += invJ[d * ctx->dim + f] * 0.5 * PetscRealPart(coords[p * ctx->dim + f] - v0[f]);
    for (PetscInt comp = 0; comp < ctx->dof; ++comp) a[p * ctx->dof + comp] += PetscRealPart(x[order[d] * ctx->dof + comp] - x[0 * ctx->dof + comp]) * xi[d];
  }
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode QuadMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar *)ctx;
  const PetscScalar  x0       = vertices[0];
  const PetscScalar  y0       = vertices[1];
  const PetscScalar  x1       = vertices[2];
  const PetscScalar  y1       = vertices[3];
  const PetscScalar  x2       = vertices[4];
  const PetscScalar  y2       = vertices[5];
  const PetscScalar  x3       = vertices[6];
  const PetscScalar  y3       = vertices[7];
  const PetscScalar  f_1      = x1 - x0;
  const PetscScalar  g_1      = y1 - y0;
  const PetscScalar  f_3      = x3 - x0;
  const PetscScalar  g_3      = y3 - y0;
  const PetscScalar  f_01     = x2 - x1 - x3 + x0;
  const PetscScalar  g_01     = y2 - y1 - y3 + y0;
  const PetscScalar *ref;
  PetscScalar       *real;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref, &ref));
  PetscCall(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_01 * p0 * p1;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_01 * p0 * p1;
  }
  PetscCall(PetscLogFlops(28));
  PetscCall(VecRestoreArrayRead(Xref, &ref));
  PetscCall(VecRestoreArray(Xreal, &real));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#include <petsc/private/dmimpl.h>
static inline PetscErrorCode QuadJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar *)ctx;
  const PetscScalar  x0       = vertices[0];
  const PetscScalar  y0       = vertices[1];
  const PetscScalar  x1       = vertices[2];
  const PetscScalar  y1       = vertices[3];
  const PetscScalar  x2       = vertices[4];
  const PetscScalar  y2       = vertices[5];
  const PetscScalar  x3       = vertices[6];
  const PetscScalar  y3       = vertices[7];
  const PetscScalar  f_01     = x2 - x1 - x3 + x0;
  const PetscScalar  g_01     = y2 - y1 - y3 + y0;
  const PetscScalar *ref;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref, &ref));
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscInt    rows[2] = {0, 1};
    PetscScalar       values[4];

    values[0] = (x1 - x0 + f_01 * y) * 0.5;
    values[1] = (x3 - x0 + f_01 * x) * 0.5;
    values[2] = (y1 - y0 + g_01 * y) * 0.5;
    values[3] = (y3 - y0 + g_01 * x) * 0.5;
    PetscCall(MatSetValues(J, 2, rows, 2, rows, values, INSERT_VALUES));
  }
  PetscCall(PetscLogFlops(30));
  PetscCall(VecRestoreArrayRead(Xref, &ref));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMInterpolate_Quad_Private(DMInterpolationInfo ctx, DM dm, PetscInt p, Vec xLocal, Vec v)
{
  const PetscInt     c   = ctx->cells[p];
  PetscFE            fem = NULL;
  DM                 dmCoord;
  SNES               snes;
  KSP                ksp;
  PC                 pc;
  Vec                coordsLocal, r, ref, real;
  Mat                J;
  PetscTabulation    T = NULL;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscReal          xir[2] = {0., 0.};
  PetscInt           Nf;
  const PetscInt     dof = ctx->dof;
  PetscScalar       *x = NULL, *vertices = NULL;
  PetscScalar       *xi;
  PetscInt           coordSize, xSize;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  if (Nf) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, 0, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      fem = (PetscFE)obj;
      PetscCall(PetscFECreateTabulation(fem, 1, 1, xir, 0, &T));
    }
  }
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateDM(dm, &dmCoord));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "quad_interp_"));
  PetscCall(VecCreate(PETSC_COMM_SELF, &r));
  PetscCall(VecSetSizes(r, 2, 2));
  PetscCall(VecSetType(r, dm->vectype));
  PetscCall(VecDuplicate(r, &ref));
  PetscCall(VecDuplicate(r, &real));
  PetscCall(MatCreate(PETSC_COMM_SELF, &J));
  PetscCall(MatSetSizes(J, 2, 2, 2, 2));
  PetscCall(MatSetType(J, MATSEQDENSE));
  PetscCall(MatSetUp(J));
  PetscCall(SNESSetFunction(snes, r, QuadMap_Private, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, QuadJacobian_Private, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  PetscCall(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
  PetscCheck(4 * 2 == coordSize, ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %" PetscInt_FMT " should be %d", coordSize, 4 * 2);
  PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
  PetscCall(SNESSetFunction(snes, NULL, NULL, vertices));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
  PetscCall(VecGetArray(real, &xi));
  xi[0] = coords[p * ctx->dim + 0];
  xi[1] = coords[p * ctx->dim + 1];
  PetscCall(VecRestoreArray(real, &xi));
  PetscCall(SNESSolve(snes, real, ref));
  PetscCall(VecGetArray(ref, &xi));
  xir[0] = PetscRealPart(xi[0]);
  xir[1] = PetscRealPart(xi[1]);
  if (4 * dof == xSize) {
    for (PetscInt comp = 0; comp < dof; ++comp) a[p * dof + comp] = x[0 * dof + comp] * (1 - xir[0]) * (1 - xir[1]) + x[1 * dof + comp] * xir[0] * (1 - xir[1]) + x[2 * dof + comp] * xir[0] * xir[1] + x[3 * dof + comp] * (1 - xir[0]) * xir[1];
  } else if (dof == xSize) {
    for (PetscInt comp = 0; comp < dof; ++comp) a[p * dof + comp] = x[0 * dof + comp];
  } else {
    PetscCheck(fem, ctx->comm, PETSC_ERR_ARG_WRONG, "Cannot have a higher order interpolant if the discretization is not PetscFE");
    xir[0] = 2.0 * xir[0] - 1.0;
    xir[1] = 2.0 * xir[1] - 1.0;
    PetscCall(PetscFEComputeTabulation(fem, 1, xir, 0, T));
    for (PetscInt comp = 0; comp < dof; ++comp) {
      a[p * dof + comp] = 0.0;
      for (PetscInt d = 0; d < xSize / dof; ++d) a[p * dof + comp] += x[d * dof + comp] * T->T[0][d * dof + comp];
    }
  }
  PetscCall(VecRestoreArray(ref, &xi));
  PetscCall(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&real));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode HexMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar *)ctx;
  const PetscScalar  x0       = vertices[0];
  const PetscScalar  y0       = vertices[1];
  const PetscScalar  z0       = vertices[2];
  const PetscScalar  x1       = vertices[9];
  const PetscScalar  y1       = vertices[10];
  const PetscScalar  z1       = vertices[11];
  const PetscScalar  x2       = vertices[6];
  const PetscScalar  y2       = vertices[7];
  const PetscScalar  z2       = vertices[8];
  const PetscScalar  x3       = vertices[3];
  const PetscScalar  y3       = vertices[4];
  const PetscScalar  z3       = vertices[5];
  const PetscScalar  x4       = vertices[12];
  const PetscScalar  y4       = vertices[13];
  const PetscScalar  z4       = vertices[14];
  const PetscScalar  x5       = vertices[15];
  const PetscScalar  y5       = vertices[16];
  const PetscScalar  z5       = vertices[17];
  const PetscScalar  x6       = vertices[18];
  const PetscScalar  y6       = vertices[19];
  const PetscScalar  z6       = vertices[20];
  const PetscScalar  x7       = vertices[21];
  const PetscScalar  y7       = vertices[22];
  const PetscScalar  z7       = vertices[23];
  const PetscScalar  f_1      = x1 - x0;
  const PetscScalar  g_1      = y1 - y0;
  const PetscScalar  h_1      = z1 - z0;
  const PetscScalar  f_3      = x3 - x0;
  const PetscScalar  g_3      = y3 - y0;
  const PetscScalar  h_3      = z3 - z0;
  const PetscScalar  f_4      = x4 - x0;
  const PetscScalar  g_4      = y4 - y0;
  const PetscScalar  h_4      = z4 - z0;
  const PetscScalar  f_01     = x2 - x1 - x3 + x0;
  const PetscScalar  g_01     = y2 - y1 - y3 + y0;
  const PetscScalar  h_01     = z2 - z1 - z3 + z0;
  const PetscScalar  f_12     = x7 - x3 - x4 + x0;
  const PetscScalar  g_12     = y7 - y3 - y4 + y0;
  const PetscScalar  h_12     = z7 - z3 - z4 + z0;
  const PetscScalar  f_02     = x5 - x1 - x4 + x0;
  const PetscScalar  g_02     = y5 - y1 - y4 + y0;
  const PetscScalar  h_02     = z5 - z1 - z4 + z0;
  const PetscScalar  f_012    = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar  g_012    = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar  h_012    = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  const PetscScalar *ref;
  PetscScalar       *real;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref, &ref));
  PetscCall(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];
    const PetscScalar p2 = ref[2];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_4 * p2 + f_01 * p0 * p1 + f_12 * p1 * p2 + f_02 * p0 * p2 + f_012 * p0 * p1 * p2;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_4 * p2 + g_01 * p0 * p1 + g_01 * p0 * p1 + g_12 * p1 * p2 + g_02 * p0 * p2 + g_012 * p0 * p1 * p2;
    real[2] = z0 + h_1 * p0 + h_3 * p1 + h_4 * p2 + h_01 * p0 * p1 + h_01 * p0 * p1 + h_12 * p1 * p2 + h_02 * p0 * p2 + h_012 * p0 * p1 * p2;
  }
  PetscCall(PetscLogFlops(114));
  PetscCall(VecRestoreArrayRead(Xref, &ref));
  PetscCall(VecRestoreArray(Xreal, &real));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode HexJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar *)ctx;
  const PetscScalar  x0       = vertices[0];
  const PetscScalar  y0       = vertices[1];
  const PetscScalar  z0       = vertices[2];
  const PetscScalar  x1       = vertices[9];
  const PetscScalar  y1       = vertices[10];
  const PetscScalar  z1       = vertices[11];
  const PetscScalar  x2       = vertices[6];
  const PetscScalar  y2       = vertices[7];
  const PetscScalar  z2       = vertices[8];
  const PetscScalar  x3       = vertices[3];
  const PetscScalar  y3       = vertices[4];
  const PetscScalar  z3       = vertices[5];
  const PetscScalar  x4       = vertices[12];
  const PetscScalar  y4       = vertices[13];
  const PetscScalar  z4       = vertices[14];
  const PetscScalar  x5       = vertices[15];
  const PetscScalar  y5       = vertices[16];
  const PetscScalar  z5       = vertices[17];
  const PetscScalar  x6       = vertices[18];
  const PetscScalar  y6       = vertices[19];
  const PetscScalar  z6       = vertices[20];
  const PetscScalar  x7       = vertices[21];
  const PetscScalar  y7       = vertices[22];
  const PetscScalar  z7       = vertices[23];
  const PetscScalar  f_xy     = x2 - x1 - x3 + x0;
  const PetscScalar  g_xy     = y2 - y1 - y3 + y0;
  const PetscScalar  h_xy     = z2 - z1 - z3 + z0;
  const PetscScalar  f_yz     = x7 - x3 - x4 + x0;
  const PetscScalar  g_yz     = y7 - y3 - y4 + y0;
  const PetscScalar  h_yz     = z7 - z3 - z4 + z0;
  const PetscScalar  f_xz     = x5 - x1 - x4 + x0;
  const PetscScalar  g_xz     = y5 - y1 - y4 + y0;
  const PetscScalar  h_xz     = z5 - z1 - z4 + z0;
  const PetscScalar  f_xyz    = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar  g_xyz    = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar  h_xyz    = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  const PetscScalar *ref;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref, &ref));
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscScalar z       = ref[2];
    const PetscInt    rows[3] = {0, 1, 2};
    PetscScalar       values[9];

    values[0] = (x1 - x0 + f_xy * y + f_xz * z + f_xyz * y * z) / 2.0;
    values[1] = (x3 - x0 + f_xy * x + f_yz * z + f_xyz * x * z) / 2.0;
    values[2] = (x4 - x0 + f_yz * y + f_xz * x + f_xyz * x * y) / 2.0;
    values[3] = (y1 - y0 + g_xy * y + g_xz * z + g_xyz * y * z) / 2.0;
    values[4] = (y3 - y0 + g_xy * x + g_yz * z + g_xyz * x * z) / 2.0;
    values[5] = (y4 - y0 + g_yz * y + g_xz * x + g_xyz * x * y) / 2.0;
    values[6] = (z1 - z0 + h_xy * y + h_xz * z + h_xyz * y * z) / 2.0;
    values[7] = (z3 - z0 + h_xy * x + h_yz * z + h_xyz * x * z) / 2.0;
    values[8] = (z4 - z0 + h_yz * y + h_xz * x + h_xyz * x * y) / 2.0;

    PetscCall(MatSetValues(J, 3, rows, 3, rows, values, INSERT_VALUES));
  }
  PetscCall(PetscLogFlops(152));
  PetscCall(VecRestoreArrayRead(Xref, &ref));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode DMInterpolate_Hex_Private(DMInterpolationInfo ctx, DM dm, PetscInt p, Vec xLocal, Vec v)
{
  const PetscInt     c = ctx->cells[p];
  DM                 dmCoord;
  SNES               snes;
  KSP                ksp;
  PC                 pc;
  Vec                coordsLocal, r, ref, real;
  Mat                J;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscScalar       *x = NULL, *vertices = NULL;
  PetscScalar       *xi;
  PetscReal          xir[3];
  PetscInt           coordSize, xSize;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateDM(dm, &dmCoord));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "hex_interp_"));
  PetscCall(VecCreate(PETSC_COMM_SELF, &r));
  PetscCall(VecSetSizes(r, 3, 3));
  PetscCall(VecSetType(r, dm->vectype));
  PetscCall(VecDuplicate(r, &ref));
  PetscCall(VecDuplicate(r, &real));
  PetscCall(MatCreate(PETSC_COMM_SELF, &J));
  PetscCall(MatSetSizes(J, 3, 3, 3, 3));
  PetscCall(MatSetType(J, MATSEQDENSE));
  PetscCall(MatSetUp(J));
  PetscCall(SNESSetFunction(snes, r, HexMap_Private, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, HexJacobian_Private, NULL));
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCLU));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  PetscCall(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
  PetscCheck(8 * 3 == coordSize, ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid coordinate closure size %" PetscInt_FMT " should be %d", coordSize, 8 * 3);
  PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
  PetscCheck((8 * ctx->dof == xSize) || (ctx->dof == xSize), ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input closure size %" PetscInt_FMT " should be %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 8 * ctx->dof, ctx->dof);
  PetscCall(SNESSetFunction(snes, NULL, NULL, vertices));
  PetscCall(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
  PetscCall(VecGetArray(real, &xi));
  xi[0] = coords[p * ctx->dim + 0];
  xi[1] = coords[p * ctx->dim + 1];
  xi[2] = coords[p * ctx->dim + 2];
  PetscCall(VecRestoreArray(real, &xi));
  PetscCall(SNESSolve(snes, real, ref));
  PetscCall(VecGetArray(ref, &xi));
  xir[0] = PetscRealPart(xi[0]);
  xir[1] = PetscRealPart(xi[1]);
  xir[2] = PetscRealPart(xi[2]);
  if (8 * ctx->dof == xSize) {
    for (PetscInt comp = 0; comp < ctx->dof; ++comp) {
      a[p * ctx->dof + comp] = x[0 * ctx->dof + comp] * (1 - xir[0]) * (1 - xir[1]) * (1 - xir[2]) + x[3 * ctx->dof + comp] * xir[0] * (1 - xir[1]) * (1 - xir[2]) + x[2 * ctx->dof + comp] * xir[0] * xir[1] * (1 - xir[2]) + x[1 * ctx->dof + comp] * (1 - xir[0]) * xir[1] * (1 - xir[2]) +
                               x[4 * ctx->dof + comp] * (1 - xir[0]) * (1 - xir[1]) * xir[2] + x[5 * ctx->dof + comp] * xir[0] * (1 - xir[1]) * xir[2] + x[6 * ctx->dof + comp] * xir[0] * xir[1] * xir[2] + x[7 * ctx->dof + comp] * (1 - xir[0]) * xir[1] * xir[2];
    }
  } else {
    for (PetscInt comp = 0; comp < ctx->dof; ++comp) a[p * ctx->dof + comp] = x[0 * ctx->dof + comp];
  }
  PetscCall(VecRestoreArray(ref, &xi));
  PetscCall(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
  PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&real));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationEvaluate - Using the input from `dm` and `x`, calculates interpolated field values at the interpolation points.

  Input Parameters:
+ ctx - The `DMInterpolationInfo` context obtained with `DMInterpolationCreate()`
. dm  - The `DM`
- x   - The local vector containing the field to be interpolated, can be created with `DMCreateGlobalVector()`

  Output Parameter:
. v - The vector containing the interpolated values, obtained with `DMInterpolationGetVector()`

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationGetVector()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`, `DMInterpolationGetCoordinates()`
@*/
PetscErrorCode DMInterpolationEvaluate(DMInterpolationInfo ctx, DM dm, Vec x, Vec v)
{
  PetscDS   ds;
  PetscInt  n, p, Nf, field;
  PetscBool useDS = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == ctx->n * ctx->dof, ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %" PetscInt_FMT " should be %" PetscInt_FMT, n, ctx->n * ctx->dof);
  if (!n) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMGetDS(dm, &ds));
  if (ds) {
    useDS = PETSC_TRUE;
    PetscCall(PetscDSGetNumFields(ds, &Nf));
    for (field = 0; field < Nf; ++field) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(PetscDSGetDiscretization(ds, field, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id != PETSCFE_CLASSID && id != PETSCFV_CLASSID) {
        useDS = PETSC_FALSE;
        break;
      }
    }
  }
  if (useDS) {
    const PetscScalar *coords;
    PetscScalar       *interpolant;
    PetscInt           cdim, d;

    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCall(VecGetArrayRead(ctx->coords, &coords));
    PetscCall(VecGetArrayWrite(v, &interpolant));
    for (p = 0; p < ctx->n; ++p) {
      PetscReal    pcoords[3], xi[3];
      PetscScalar *xa   = NULL;
      PetscInt     coff = 0, foff = 0, clSize;

      if (ctx->cells[p] < 0) continue;
      for (d = 0; d < cdim; ++d) pcoords[d] = PetscRealPart(coords[p * cdim + d]);
      PetscCall(DMPlexCoordinatesToReference(dm, ctx->cells[p], 1, pcoords, xi));
      PetscCall(DMPlexVecGetClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      for (field = 0; field < Nf; ++field) {
        PetscTabulation T;
        PetscObject     obj;
        PetscClassId    id;

        PetscCall(PetscDSGetDiscretization(ds, field, &obj));
        PetscCall(PetscObjectGetClassId(obj, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE)obj;

          PetscCall(PetscFECreateTabulation(fe, 1, 1, xi, 0, &T));
          {
            const PetscReal *basis = T->T[0];
            const PetscInt   Nb    = T->Nb;
            const PetscInt   Nc    = T->Nc;

            for (PetscInt fc = 0; fc < Nc; ++fc) {
              interpolant[p * ctx->dof + coff + fc] = 0.0;
              for (PetscInt f = 0; f < Nb; ++f) interpolant[p * ctx->dof + coff + fc] += xa[foff + f] * basis[(0 * Nb + f) * Nc + fc];
            }
            coff += Nc;
            foff += Nb;
          }
          PetscCall(PetscTabulationDestroy(&T));
        } else if (id == PETSCFV_CLASSID) {
          PetscFV  fv = (PetscFV)obj;
          PetscInt Nc;

          // TODO Could use reconstruction if available
          PetscCall(PetscFVGetNumComponents(fv, &Nc));
          for (PetscInt fc = 0; fc < Nc; ++fc) interpolant[p * ctx->dof + coff + fc] = xa[foff + fc];
          coff += Nc;
          foff += Nc;
        }
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      PetscCheck(coff == ctx->dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total components %" PetscInt_FMT " != %" PetscInt_FMT " dof specified for interpolation", coff, ctx->dof);
      PetscCheck(foff == clSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total FE/FV space size %" PetscInt_FMT " != %" PetscInt_FMT " closure size", foff, clSize);
    }
    PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
    PetscCall(VecRestoreArrayWrite(v, &interpolant));
  } else {
    for (PetscInt p = 0; p < ctx->n; ++p) {
      const PetscInt cell = ctx->cells[p];
      DMPolytopeType ct;

      PetscCall(DMPlexGetCellType(dm, cell, &ct));
      switch (ct) {
      case DM_POLYTOPE_SEGMENT:
        PetscCall(DMInterpolate_Segment_Private(ctx, dm, p, x, v));
        break;
      case DM_POLYTOPE_TRIANGLE:
        PetscCall(DMInterpolate_Triangle_Private(ctx, dm, p, x, v));
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        PetscCall(DMInterpolate_Quad_Private(ctx, dm, p, x, v));
        break;
      case DM_POLYTOPE_TETRAHEDRON:
        PetscCall(DMInterpolate_Tetrahedron_Private(ctx, dm, p, x, v));
        break;
      case DM_POLYTOPE_HEXAHEDRON:
        PetscCall(DMInterpolate_Hex_Private(ctx, dm, cell, x, v));
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for cell type %s", DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMInterpolationDestroy - Destroys a `DMInterpolationInfo` context

  Collective

  Input Parameter:
. ctx - the context

  Level: beginner

.seealso: [](ch_dmbase), `DM`, `DMInterpolationInfo`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ctx)
{
  PetscFunctionBegin;
  PetscAssertPointer(ctx, 1);
  PetscCall(VecDestroy(&(*ctx)->coords));
  PetscCall(PetscFree((*ctx)->points));
  PetscCall(PetscFree((*ctx)->cells));
  PetscCall(PetscFree(*ctx));
  *ctx = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}
