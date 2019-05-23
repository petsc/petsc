#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>     /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petscblaslapack.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

/************************** Interpolation *******************************/

static PetscErrorCode DMSNESConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex);CHKERRQ(ierr);
  if (isPlex) {
    *plex = dm;
    ierr = PetscObjectReference((PetscObject) dm);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex);CHKERRQ(ierr);
    if (!*plex) {
      ierr = DMConvert(dm,DMPLEX,plex);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex);CHKERRQ(ierr);
      if (copy) {
        PetscInt    i;
        PetscObject obj;
        const char *comps[3] = {"A","dmAux","dmCh"};

        ierr = DMCopyDMSNES(dm, *plex);CHKERRQ(ierr);
        for (i = 0; i < 3; i++) {
          ierr = PetscObjectQuery((PetscObject) dm, comps[i], &obj);CHKERRQ(ierr);
          ierr = PetscObjectCompose((PetscObject) *plex, comps[i], obj);CHKERRQ(ierr);
        }
      }
    } else {
      ierr = PetscObjectReference((PetscObject) *plex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationCreate - Creates a DMInterpolationInfo context

  Collective

  Input Parameter:
. comm - the communicator

  Output Parameter:
. ctx - the context

  Level: beginner

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationDestroy()
@*/
PetscErrorCode DMInterpolationCreate(MPI_Comm comm, DMInterpolationInfo *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ctx, 2);
  ierr = PetscNew(ctx);CHKERRQ(ierr);

  (*ctx)->comm   = comm;
  (*ctx)->dim    = -1;
  (*ctx)->nInput = 0;
  (*ctx)->points = NULL;
  (*ctx)->cells  = NULL;
  (*ctx)->n      = -1;
  (*ctx)->coords = NULL;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationSetDim - Sets the spatial dimension for the interpolation context

  Not collective

  Input Parameters:
+ ctx - the context
- dim - the spatial dimension

  Level: intermediate

.seealso: DMInterpolationGetDim(), DMInterpolationEvaluate(), DMInterpolationAddPoints()
@*/
PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ctx, PetscInt dim)
{
  PetscFunctionBegin;
  if ((dim < 1) || (dim > 3)) SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %D", dim);
  ctx->dim = dim;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationGetDim - Gets the spatial dimension for the interpolation context

  Not collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. dim - the spatial dimension

  Level: intermediate

.seealso: DMInterpolationSetDim(), DMInterpolationEvaluate(), DMInterpolationAddPoints()
@*/
PetscErrorCode DMInterpolationGetDim(DMInterpolationInfo ctx, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidIntPointer(dim, 2);
  *dim = ctx->dim;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationSetDof - Sets the number of fields interpolated at a point for the interpolation context

  Not collective

  Input Parameters:
+ ctx - the context
- dof - the number of fields

  Level: intermediate

.seealso: DMInterpolationGetDof(), DMInterpolationEvaluate(), DMInterpolationAddPoints()
@*/
PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ctx, PetscInt dof)
{
  PetscFunctionBegin;
  if (dof < 1) SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %D", dof);
  ctx->dof = dof;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationGetDof - Gets the number of fields interpolated at a point for the interpolation context

  Not collective

  Input Parameter:
. ctx - the context

  Output Parameter:
. dof - the number of fields

  Level: intermediate

.seealso: DMInterpolationSetDof(), DMInterpolationEvaluate(), DMInterpolationAddPoints()
@*/
PetscErrorCode DMInterpolationGetDof(DMInterpolationInfo ctx, PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidIntPointer(dof, 2);
  *dof = ctx->dof;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationAddPoints - Add points at which we will interpolate the fields

  Not collective

  Input Parameters:
+ ctx    - the context
. n      - the number of points
- points - the coordinates for each point, an array of size n * dim

  Note: The coordinate information is copied.

  Level: intermediate

.seealso: DMInterpolationSetDim(), DMInterpolationEvaluate(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ctx, PetscInt n, PetscReal points[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->dim < 0) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  if (ctx->points)  SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  ctx->nInput = n;

  ierr = PetscMalloc1(n*ctx->dim, &ctx->points);CHKERRQ(ierr);
  ierr = PetscArraycpy(ctx->points, points, n*ctx->dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationSetUp - Computea spatial indices that add in point location during interpolation

  Collective on ctx

  Input Parameters:
+ ctx - the context
. dm  - the DM for the function space used for interpolation
- redundantPoints - If PETSC_TRUE, all processes are passing in the same array of points. Otherwise, points need to be communicated among processes.

  Level: intermediate

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationSetUp(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints)
{
  MPI_Comm          comm = ctx->comm;
  PetscScalar       *a;
  PetscInt          p, q, i;
  PetscMPIInt       rank, size;
  PetscErrorCode    ierr;
  Vec               pointVec;
  PetscSF           cellSF;
  PetscLayout       layout;
  PetscReal         *globalPoints;
  PetscScalar       *globalPointsScalar;
  const PetscInt    *ranges;
  PetscMPIInt       *counts, *displs;
  const PetscSFNode *foundCells;
  const PetscInt    *foundPoints;
  PetscMPIInt       *foundProcs, *globalProcs;
  PetscInt          n, N, numFound;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  if (ctx->dim < 0) SETERRQ(comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    ierr = PetscLayoutCreate(comm, &layout);CHKERRQ(ierr);
    ierr = PetscLayoutSetBlockSize(layout, 1);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(layout, n);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(layout);CHKERRQ(ierr);
    ierr = PetscLayoutGetSize(layout, &N);CHKERRQ(ierr);
    /* Communicate all points to all processes */
    ierr = PetscMalloc3(N*ctx->dim,&globalPoints,size,&counts,size,&displs);CHKERRQ(ierr);
    ierr = PetscLayoutGetRanges(layout, &ranges);CHKERRQ(ierr);
    for (p = 0; p < size; ++p) {
      counts[p] = (ranges[p+1] - ranges[p])*ctx->dim;
      displs[p] = ranges[p]*ctx->dim;
    }
    ierr = MPI_Allgatherv(ctx->points, n*ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm);CHKERRQ(ierr);
  } else {
    N = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout = NULL;
  }
#if 0
  ierr = PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs);CHKERRQ(ierr);
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscMalloc1(N*ctx->dim,&globalPointsScalar);CHKERRQ(ierr);
  for (i=0; i<N*ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
#else
  globalPointsScalar = globalPoints;
#endif
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N*ctx->dim, globalPointsScalar, &pointVec);CHKERRQ(ierr);
  ierr = PetscMalloc2(N,&foundProcs,N,&globalProcs);CHKERRQ(ierr);
  for (p = 0; p < N; ++p) {foundProcs[p] = size;}
  cellSF = NULL;
  ierr = DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(cellSF,NULL,&numFound,&foundPoints,&foundCells);CHKERRQ(ierr);
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  ierr   = MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm);CHKERRQ(ierr);
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) SETERRQ4(comm, PETSC_ERR_PLIB, "Point %d: %g %g %g not located in mesh", p, (double)globalPoints[p*ctx->dim+0], (double)(ctx->dim > 1 ? globalPoints[p*ctx->dim+1] : 0.0), (double)(ctx->dim > 2 ? globalPoints[p*ctx->dim+2] : 0.0));
    else if (globalProcs[p] == rank) ctx->n++;
  }
  /* Create coordinates vector and array of owned cells */
  ierr = PetscMalloc1(ctx->n, &ctx->cells);CHKERRQ(ierr);
  ierr = VecCreate(comm, &ctx->coords);CHKERRQ(ierr);
  ierr = VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(ctx->coords, ctx->dim);CHKERRQ(ierr);
  ierr = VecSetType(ctx->coords,VECSTANDARD);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->coords, &a);CHKERRQ(ierr);
  for (p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = globalPoints[p*ctx->dim+d];
      ctx->cells[q] = foundCells[q].index;
      ++q;
    }
  }
  ierr = VecRestoreArray(ctx->coords, &a);CHKERRQ(ierr);
#if 0
  ierr = PetscFree3(foundCells,foundProcs,globalProcs);CHKERRQ(ierr);
#else
  ierr = PetscFree2(foundProcs,globalProcs);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&cellSF);CHKERRQ(ierr);
  ierr = VecDestroy(&pointVec);CHKERRQ(ierr);
#endif
  if ((void*)globalPointsScalar != (void*)globalPoints) {ierr = PetscFree(globalPointsScalar);CHKERRQ(ierr);}
  if (!redundantPoints) {ierr = PetscFree3(globalPoints,counts,displs);CHKERRQ(ierr);}
  ierr = PetscLayoutDestroy(&layout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationGetCoordinates - Gets a Vec with the coordinates of each interpolation point

  Collective on ctx

  Input Parameter:
. ctx - the context

  Output Parameter:
. coordinates  - the coordinates of interpolation points

  Note: The local vector entries correspond to interpolation points lying on this process, according to the associated DM. This is a borrowed vector that the user should not destroy.

  Level: intermediate

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationGetCoordinates(DMInterpolationInfo ctx, Vec *coordinates)
{
  PetscFunctionBegin;
  PetscValidPointer(coordinates, 2);
  if (!ctx->coords) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  *coordinates = ctx->coords;
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationGetVector - Gets a Vec which can hold all the interpolated field values

  Collective on ctx

  Input Parameter:
. ctx - the context

  Output Parameter:
. v  - a vector capable of holding the interpolated field values

  Note: This vector should be returned using DMInterpolationRestoreVector().

  Level: intermediate

.seealso: DMInterpolationRestoreVector(), DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationGetVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  if (!ctx->coords) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  ierr = VecCreate(ctx->comm, v);CHKERRQ(ierr);
  ierr = VecSetSizes(*v, ctx->n*ctx->dof, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetBlockSize(*v, ctx->dof);CHKERRQ(ierr);
  ierr = VecSetType(*v,VECSTANDARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationRestoreVector - Returns a Vec which can hold all the interpolated field values

  Collective on ctx

  Input Parameters:
+ ctx - the context
- v  - a vector capable of holding the interpolated field values

  Level: intermediate

.seealso: DMInterpolationGetVector(), DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationRestoreVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  if (!ctx->coords) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  ierr = VecDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMInterpolate_Triangle_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (p = 0; p < ctx->n; ++p) {
    PetscInt     c = ctx->cells[p];
    PetscScalar *x = NULL;
    PetscReal    xi[4];
    PetscInt     d, f, comp;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D", (double)detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x);CHKERRQ(ierr);
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[(d+1)*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMInterpolate_Tetrahedron_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (p = 0; p < ctx->n; ++p) {
    PetscInt       c = ctx->cells[p];
    const PetscInt order[3] = {2, 1, 3};
    PetscScalar   *x = NULL;
    PetscReal      xi[4];
    PetscInt       d, f, comp;

    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D", (double)detJ, c);
    ierr = DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x);CHKERRQ(ierr);
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[order[d]*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    ierr = DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = PetscFree3(v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode QuadMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar*) ctx;
  const PetscScalar x0        = vertices[0];
  const PetscScalar y0        = vertices[1];
  const PetscScalar x1        = vertices[2];
  const PetscScalar y1        = vertices[3];
  const PetscScalar x2        = vertices[4];
  const PetscScalar y2        = vertices[5];
  const PetscScalar x3        = vertices[6];
  const PetscScalar y3        = vertices[7];
  const PetscScalar f_1       = x1 - x0;
  const PetscScalar g_1       = y1 - y0;
  const PetscScalar f_3       = x3 - x0;
  const PetscScalar g_3       = y3 - y0;
  const PetscScalar f_01      = x2 - x1 - x3 + x0;
  const PetscScalar g_01      = y2 - y1 - y3 + y0;
  const PetscScalar *ref;
  PetscScalar       *real;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecGetArray(Xreal, &real);CHKERRQ(ierr);
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_01 * p0 * p1;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_01 * p0 * p1;
  }
  ierr = PetscLogFlops(28);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xreal, &real);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petsc/private/dmimpl.h>
PETSC_STATIC_INLINE PetscErrorCode QuadJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar*) ctx;
  const PetscScalar x0        = vertices[0];
  const PetscScalar y0        = vertices[1];
  const PetscScalar x1        = vertices[2];
  const PetscScalar y1        = vertices[3];
  const PetscScalar x2        = vertices[4];
  const PetscScalar y2        = vertices[5];
  const PetscScalar x3        = vertices[6];
  const PetscScalar y3        = vertices[7];
  const PetscScalar f_01      = x2 - x1 - x3 + x0;
  const PetscScalar g_01      = y2 - y1 - y3 + y0;
  const PetscScalar *ref;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Xref,  &ref);CHKERRQ(ierr);
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscInt    rows[2] = {0, 1};
    PetscScalar       values[4];

    values[0] = (x1 - x0 + f_01*y) * 0.5; values[1] = (x3 - x0 + f_01*x) * 0.5;
    values[2] = (y1 - y0 + g_01*y) * 0.5; values[3] = (y3 - y0 + g_01*x) * 0.5;
    ierr      = MatSetValues(J, 2, rows, 2, rows, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(30);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMInterpolate_Quad_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  DM             dmCoord;
  PetscFE        fem = NULL;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  Vec            coordsLocal, r, ref, real;
  Mat            J;
  PetscTabulation    T;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscReal      xir[2];
  PetscInt       Nf, p;
  const PetscInt dof = ctx->dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  if (Nf) {ierr = DMGetField(dm, 0, NULL, (PetscObject *) &fem);CHKERRQ(ierr);}
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "quad_interp_");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
  ierr = VecSetSizes(r, 2, 2);CHKERRQ(ierr);
  ierr = VecSetType(r,dm->vectype);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &ref);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &real);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &J);CHKERRQ(ierr);
  ierr = MatSetSizes(J, 2, 2, 2, 2);CHKERRQ(ierr);
  ierr = MatSetType(J, MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, QuadMap_Private, NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, J, J, QuadJacobian_Private, NULL);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecGetArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  ierr = PetscFECreateTabulation(fem, 1, 1, xir, 0, &T);CHKERRQ(ierr);
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    ierr = DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices);CHKERRQ(ierr);
    if (4*2 != coordSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %D should be %d", coordSize, 4*2);
    ierr   = DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x);CHKERRQ(ierr);
    ierr   = SNESSetFunction(snes, NULL, NULL, (void*) vertices);CHKERRQ(ierr);
    ierr   = SNESSetJacobian(snes, NULL, NULL, NULL, (void*) vertices);CHKERRQ(ierr);
    ierr   = VecGetArray(real, &xi);CHKERRQ(ierr);
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    ierr   = VecRestoreArray(real, &xi);CHKERRQ(ierr);
    ierr   = SNESSolve(snes, real, ref);CHKERRQ(ierr);
    ierr   = VecGetArray(ref, &xi);CHKERRQ(ierr);
    xir[0] = PetscRealPart(xi[0]);
    xir[1] = PetscRealPart(xi[1]);
    if (4*dof != xSize) {
      PetscInt d;

      xir[0] = 2.0*xir[0] - 1.0; xir[1] = 2.0*xir[1] - 1.0;
      ierr = PetscFEComputeTabulation(fem, 1, xir, 0, T);CHKERRQ(ierr);
      for (comp = 0; comp < dof; ++comp) {
        a[p*dof+comp] = 0.0;
        for (d = 0; d < xSize/dof; ++d) {
          a[p*dof+comp] += x[d*dof+comp]*T->T[0][d*dof+comp];
        }
      }
    } else {
      for (comp = 0; comp < dof; ++comp)
        a[p*dof+comp] = x[0*dof+comp]*(1 - xir[0])*(1 - xir[1]) + x[1*dof+comp]*xir[0]*(1 - xir[1]) + x[2*dof+comp]*xir[0]*xir[1] + x[3*dof+comp]*(1 - xir[0])*xir[1];
    }
    ierr = VecRestoreArray(ref, &xi);CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices);CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x);CHKERRQ(ierr);
  }
  ierr = PetscTabulationDestroy(&T);CHKERRQ(ierr);
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->coords, &coords);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  ierr = VecDestroy(&real);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode HexMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar*) ctx;
  const PetscScalar x0        = vertices[0];
  const PetscScalar y0        = vertices[1];
  const PetscScalar z0        = vertices[2];
  const PetscScalar x1        = vertices[9];
  const PetscScalar y1        = vertices[10];
  const PetscScalar z1        = vertices[11];
  const PetscScalar x2        = vertices[6];
  const PetscScalar y2        = vertices[7];
  const PetscScalar z2        = vertices[8];
  const PetscScalar x3        = vertices[3];
  const PetscScalar y3        = vertices[4];
  const PetscScalar z3        = vertices[5];
  const PetscScalar x4        = vertices[12];
  const PetscScalar y4        = vertices[13];
  const PetscScalar z4        = vertices[14];
  const PetscScalar x5        = vertices[15];
  const PetscScalar y5        = vertices[16];
  const PetscScalar z5        = vertices[17];
  const PetscScalar x6        = vertices[18];
  const PetscScalar y6        = vertices[19];
  const PetscScalar z6        = vertices[20];
  const PetscScalar x7        = vertices[21];
  const PetscScalar y7        = vertices[22];
  const PetscScalar z7        = vertices[23];
  const PetscScalar f_1       = x1 - x0;
  const PetscScalar g_1       = y1 - y0;
  const PetscScalar h_1       = z1 - z0;
  const PetscScalar f_3       = x3 - x0;
  const PetscScalar g_3       = y3 - y0;
  const PetscScalar h_3       = z3 - z0;
  const PetscScalar f_4       = x4 - x0;
  const PetscScalar g_4       = y4 - y0;
  const PetscScalar h_4       = z4 - z0;
  const PetscScalar f_01      = x2 - x1 - x3 + x0;
  const PetscScalar g_01      = y2 - y1 - y3 + y0;
  const PetscScalar h_01      = z2 - z1 - z3 + z0;
  const PetscScalar f_12      = x7 - x3 - x4 + x0;
  const PetscScalar g_12      = y7 - y3 - y4 + y0;
  const PetscScalar h_12      = z7 - z3 - z4 + z0;
  const PetscScalar f_02      = x5 - x1 - x4 + x0;
  const PetscScalar g_02      = y5 - y1 - y4 + y0;
  const PetscScalar h_02      = z5 - z1 - z4 + z0;
  const PetscScalar f_012     = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar g_012     = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar h_012     = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  const PetscScalar *ref;
  PetscScalar       *real;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecGetArray(Xreal, &real);CHKERRQ(ierr);
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];
    const PetscScalar p2 = ref[2];

    real[0] = x0 + f_1*p0 + f_3*p1 + f_4*p2 + f_01*p0*p1 + f_12*p1*p2 + f_02*p0*p2 + f_012*p0*p1*p2;
    real[1] = y0 + g_1*p0 + g_3*p1 + g_4*p2 + g_01*p0*p1 + g_01*p0*p1 + g_12*p1*p2 + g_02*p0*p2 + g_012*p0*p1*p2;
    real[2] = z0 + h_1*p0 + h_3*p1 + h_4*p2 + h_01*p0*p1 + h_01*p0*p1 + h_12*p1*p2 + h_02*p0*p2 + h_012*p0*p1*p2;
  }
  ierr = PetscLogFlops(114);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xreal, &real);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode HexJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
{
  const PetscScalar *vertices = (const PetscScalar*) ctx;
  const PetscScalar x0        = vertices[0];
  const PetscScalar y0        = vertices[1];
  const PetscScalar z0        = vertices[2];
  const PetscScalar x1        = vertices[9];
  const PetscScalar y1        = vertices[10];
  const PetscScalar z1        = vertices[11];
  const PetscScalar x2        = vertices[6];
  const PetscScalar y2        = vertices[7];
  const PetscScalar z2        = vertices[8];
  const PetscScalar x3        = vertices[3];
  const PetscScalar y3        = vertices[4];
  const PetscScalar z3        = vertices[5];
  const PetscScalar x4        = vertices[12];
  const PetscScalar y4        = vertices[13];
  const PetscScalar z4        = vertices[14];
  const PetscScalar x5        = vertices[15];
  const PetscScalar y5        = vertices[16];
  const PetscScalar z5        = vertices[17];
  const PetscScalar x6        = vertices[18];
  const PetscScalar y6        = vertices[19];
  const PetscScalar z6        = vertices[20];
  const PetscScalar x7        = vertices[21];
  const PetscScalar y7        = vertices[22];
  const PetscScalar z7        = vertices[23];
  const PetscScalar f_xy      = x2 - x1 - x3 + x0;
  const PetscScalar g_xy      = y2 - y1 - y3 + y0;
  const PetscScalar h_xy      = z2 - z1 - z3 + z0;
  const PetscScalar f_yz      = x7 - x3 - x4 + x0;
  const PetscScalar g_yz      = y7 - y3 - y4 + y0;
  const PetscScalar h_yz      = z7 - z3 - z4 + z0;
  const PetscScalar f_xz      = x5 - x1 - x4 + x0;
  const PetscScalar g_xz      = y5 - y1 - y4 + y0;
  const PetscScalar h_xz      = z5 - z1 - z4 + z0;
  const PetscScalar f_xyz     = x6 - x0 + x1 - x2 + x3 + x4 - x5 - x7;
  const PetscScalar g_xyz     = y6 - y0 + y1 - y2 + y3 + y4 - y5 - y7;
  const PetscScalar h_xyz     = z6 - z0 + z1 - z2 + z3 + z4 - z5 - z7;
  const PetscScalar *ref;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(Xref,  &ref);CHKERRQ(ierr);
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscScalar z       = ref[2];
    const PetscInt    rows[3] = {0, 1, 2};
    PetscScalar       values[9];

    values[0] = (x1 - x0 + f_xy*y + f_xz*z + f_xyz*y*z) / 2.0;
    values[1] = (x3 - x0 + f_xy*x + f_yz*z + f_xyz*x*z) / 2.0;
    values[2] = (x4 - x0 + f_yz*y + f_xz*x + f_xyz*x*y) / 2.0;
    values[3] = (y1 - y0 + g_xy*y + g_xz*z + g_xyz*y*z) / 2.0;
    values[4] = (y3 - y0 + g_xy*x + g_yz*z + g_xyz*x*z) / 2.0;
    values[5] = (y4 - y0 + g_yz*y + g_xz*x + g_xyz*x*y) / 2.0;
    values[6] = (z1 - z0 + h_xy*y + h_xz*z + h_xyz*y*z) / 2.0;
    values[7] = (z3 - z0 + h_xy*x + h_yz*z + h_xyz*x*z) / 2.0;
    values[8] = (z4 - z0 + h_yz*y + h_xz*x + h_xyz*x*y) / 2.0;

    ierr = MatSetValues(J, 3, rows, 3, rows, values, INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscLogFlops(152);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xref,  &ref);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode DMInterpolate_Hex_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  DM             dmCoord;
  SNES           snes;
  KSP            ksp;
  PC             pc;
  Vec            coordsLocal, r, ref, real;
  Mat            J;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinatesLocal(dm, &coordsLocal);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(dm, &dmCoord);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_SELF, &snes);CHKERRQ(ierr);
  ierr = SNESSetOptionsPrefix(snes, "hex_interp_");CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &r);CHKERRQ(ierr);
  ierr = VecSetSizes(r, 3, 3);CHKERRQ(ierr);
  ierr = VecSetType(r,dm->vectype);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &ref);CHKERRQ(ierr);
  ierr = VecDuplicate(r, &real);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF, &J);CHKERRQ(ierr);
  ierr = MatSetSizes(J, 3, 3, 3, 3);CHKERRQ(ierr);
  ierr = MatSetType(J, MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = SNESSetFunction(snes, r, HexMap_Private, NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes, J, J, HexJacobian_Private, NULL);CHKERRQ(ierr);
  ierr = SNESGetKSP(snes, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = VecGetArrayRead(ctx->coords, &coords);CHKERRQ(ierr);
  ierr = VecGetArray(v, &a);CHKERRQ(ierr);
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscReal    xir[3];
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    ierr = DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices);CHKERRQ(ierr);
    if (8*3 != coordSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %D should be %d", coordSize, 8*3);
    ierr = DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x);CHKERRQ(ierr);
    if (8*ctx->dof != xSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %D should be %D", xSize, 8*ctx->dof);
    ierr   = SNESSetFunction(snes, NULL, NULL, (void*) vertices);CHKERRQ(ierr);
    ierr   = SNESSetJacobian(snes, NULL, NULL, NULL, (void*) vertices);CHKERRQ(ierr);
    ierr   = VecGetArray(real, &xi);CHKERRQ(ierr);
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    xi[2]  = coords[p*ctx->dim+2];
    ierr   = VecRestoreArray(real, &xi);CHKERRQ(ierr);
    ierr   = SNESSolve(snes, real, ref);CHKERRQ(ierr);
    ierr   = VecGetArray(ref, &xi);CHKERRQ(ierr);
    xir[0] = PetscRealPart(xi[0]);
    xir[1] = PetscRealPart(xi[1]);
    xir[2] = PetscRealPart(xi[2]);
    for (comp = 0; comp < ctx->dof; ++comp) {
      a[p*ctx->dof+comp] =
        x[0*ctx->dof+comp]*(1-xir[0])*(1-xir[1])*(1-xir[2]) +
        x[3*ctx->dof+comp]*    xir[0]*(1-xir[1])*(1-xir[2]) +
        x[2*ctx->dof+comp]*    xir[0]*    xir[1]*(1-xir[2]) +
        x[1*ctx->dof+comp]*(1-xir[0])*    xir[1]*(1-xir[2]) +
        x[4*ctx->dof+comp]*(1-xir[0])*(1-xir[1])*   xir[2] +
        x[5*ctx->dof+comp]*    xir[0]*(1-xir[1])*   xir[2] +
        x[6*ctx->dof+comp]*    xir[0]*    xir[1]*   xir[2] +
        x[7*ctx->dof+comp]*(1-xir[0])*    xir[1]*   xir[2];
    }
    ierr = VecRestoreArray(ref, &xi);CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices);CHKERRQ(ierr);
    ierr = DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(v, &a);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->coords, &coords);CHKERRQ(ierr);

  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&ref);CHKERRQ(ierr);
  ierr = VecDestroy(&real);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationEvaluate - Using the input from dm and x, calculates interpolated field values at the interpolation points.

  Input Parameters:
+ ctx - The DMInterpolationInfo context
. dm  - The DM
- x   - The local vector containing the field to be interpolated

  Output Parameters:
. v   - The vector containing the interpolated values

  Note: A suitable v can be obtained using DMInterpolationGetVector().

  Level: beginner

.seealso: DMInterpolationGetVector(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationEvaluate(DMInterpolationInfo ctx, DM dm, Vec x, Vec v)
{
  PetscInt       dim, coneSize, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  if (n != ctx->n*ctx->dof) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %D should be %D", n, ctx->n*ctx->dof);
  if (n) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, ctx->cells[0], &coneSize);CHKERRQ(ierr);
    if (dim == 2) {
      if (coneSize == 3) {
        ierr = DMInterpolate_Triangle_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else if (coneSize == 4) {
        ierr = DMInterpolate_Quad_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %D for point interpolation", dim);
    } else if (dim == 3) {
      if (coneSize == 4) {
        ierr = DMInterpolate_Tetrahedron_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else {
        ierr = DMInterpolate_Hex_Private(ctx, dm, x, v);CHKERRQ(ierr);
      }
    } else SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %D for point interpolation", dim);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationDestroy - Destroys a DMInterpolationInfo context

  Collective on ctx

  Input Parameter:
. ctx - the context

  Level: beginner

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ctx, 2);
  ierr = VecDestroy(&(*ctx)->coords);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->points);CHKERRQ(ierr);
  ierr = PetscFree((*ctx)->cells);CHKERRQ(ierr);
  ierr = PetscFree(*ctx);CHKERRQ(ierr);
  *ctx = NULL;
  PetscFunctionReturn(0);
}

/*@C
  SNESMonitorFields - Monitors the residual for each field separately

  Collective on SNES

  Input Parameters:
+ snes   - the SNES context
. its    - iteration number
. fgnorm - 2-norm of residual
- vf  - PetscViewerAndFormat of type ASCII

  Notes:
  This routine prints the residual norm at each iteration.

  Level: intermediate

.seealso: SNESMonitorSet(), SNESMonitorDefault()
@*/
PetscErrorCode SNESMonitorFields(SNES snes, PetscInt its, PetscReal fgnorm, PetscViewerAndFormat *vf)
{
  PetscViewer        viewer = vf->viewer;
  Vec                res;
  DM                 dm;
  PetscSection       s;
  const PetscScalar *r;
  PetscReal         *lnorms, *norms;
  PetscInt           numFields, f, pStart, pEnd, p;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  ierr = SNESGetFunction(snes, &res, NULL, NULL);CHKERRQ(ierr);
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &s);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(s, &numFields);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(s, &pStart, &pEnd);CHKERRQ(ierr);
  ierr = PetscCalloc2(numFields, &lnorms, numFields, &norms);CHKERRQ(ierr);
  ierr = VecGetArrayRead(res, &r);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < numFields; ++f) {
      PetscInt fdof, foff, d;

      ierr = PetscSectionGetFieldDof(s, p, f, &fdof);CHKERRQ(ierr);
      ierr = PetscSectionGetFieldOffset(s, p, f, &foff);CHKERRQ(ierr);
      for (d = 0; d < fdof; ++d) lnorms[f] += PetscRealPart(PetscSqr(r[foff+d]));
    }
  }
  ierr = VecRestoreArrayRead(res, &r);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(lnorms, norms, numFields, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) dm));CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(viewer,vf->format);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer, ((PetscObject) snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "%3D SNES Function norm %14.12e [", its, (double) fgnorm);CHKERRQ(ierr);
  for (f = 0; f < numFields; ++f) {
    if (f > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "%14.12e", (double) PetscSqrtReal(norms[f]));CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPrintf(viewer, "]\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIISubtractTab(viewer, ((PetscObject) snes)->tablevel);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  ierr = PetscFree2(lnorms, norms);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/********************* Residual Computation **************************/


/*@
  DMPlexSNESGetGeometryFVM - Return precomputed geometric data

  Input Parameter:
. dm - The DM

  Output Parameters:
+ facegeom - The values precomputed from face geometry
. cellgeom - The values precomputed from cell geometry
- minRadius - The minimum radius over the mesh of an inscribed sphere in a cell

  Level: developer

.seealso: DMPlexTSSetRHSFunctionLocal()
@*/
PetscErrorCode DMPlexSNESGetGeometryFVM(DM dm, Vec *facegeom, Vec *cellgeom, PetscReal *minRadius)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(plex, NULL, cellgeom, facegeom, NULL);CHKERRQ(ierr);
  if (minRadius) {ierr = DMPlexGetMinRadius(plex, minRadius);CHKERRQ(ierr);}
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESGetGradientDM - Return gradient data layout

  Input Parameters:
+ dm - The DM
- fv - The PetscFV

  Output Parameter:
. dmGrad - The layout for gradient values

  Level: developer

.seealso: DMPlexSNESGetGeometryFVM()
@*/
PetscErrorCode DMPlexSNESGetGradientDM(DM dm, PetscFV fv, DM *dmGrad)
{
  DM             plex;
  PetscBool      computeGradients;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidHeaderSpecific(fv,PETSCFV_CLASSID,2);
  PetscValidPointer(dmGrad,3);
  ierr = PetscFVGetComputeGradients(fv, &computeGradients);CHKERRQ(ierr);
  if (!computeGradients) {*dmGrad = NULL; PetscFunctionReturn(0);}
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDataFVM(plex, fv, NULL, NULL, dmGrad);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComputeBdResidual_Single_Internal(DM dm, PetscReal t, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, Vec locF, DMField coordField, IS facetIS)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DM               plex = NULL, plexA = NULL;
  DMEnclosureType  encAux;
  PetscDS          prob, probAux = NULL;
  PetscSection     section, sectionAux = NULL;
  Vec              locA = NULL;
  PetscScalar     *u = NULL, *u_t = NULL, *a = NULL, *elemVec = NULL;
  PetscInt         v;
  PetscInt         totDim, totDimAux = 0;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    DM dmAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetDS(plexA, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plexA, &sectionAux);CHKERRQ(ierr);
  }
  for (v = 0; v < numValues; ++v) {
    PetscFEGeom    *fgeom;
    PetscInt        maxDegree;
    PetscQuadrature qGeom = NULL;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numFaces, face, Nq;

    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a priori if one is a superset of the other */
      ierr = ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      pointIS = isectIS;
    }
    ierr = ISGetLocalSize(pointIS,&numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS,&points);CHKERRQ(ierr);
    ierr = PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim, &elemVec, locA ? numFaces*totDimAux : 0, &a);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetFaceQuadrature(fe, &qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support, *cone;
      PetscScalar   *x     = NULL;
      PetscInt       i, coneSize, faceLoc;

      ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
      for (faceLoc = 0; faceLoc < coneSize; ++faceLoc) if (cone[faceLoc] == point) break;
      if (faceLoc == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of support[0] %D", point, support[0]);
      fgeom->face[face][0] = faceLoc;
      ierr = DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      if (locX_t) {
        ierr = DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
      }
      if (locA) {
        PetscInt subp;

        ierr = DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
      }
    }
    ierr = PetscArrayzero(elemVec, numFaces*totDim);CHKERRQ(ierr);
    {
      PetscFE         fe;
      PetscInt        Nb;
      PetscFEGeom     *chunkGeom = NULL;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscInt        Nr, offset;

      ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      /* TODO: documentation is unclear about what is going on with these numbers: how should Nb / Nq factor in ? */
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      ierr = PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateBdResidual(prob, field, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(fgeom, 0, offset, &chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateBdResidual(prob, field, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, &elemVec[offset*totDim]);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "BdResidual", totDim, &elemVec[face*totDim]);CHKERRQ(ierr);}
      ierr = DMPlexGetSupport(plex, point, &support);CHKERRQ(ierr);
      ierr = DMPlexVecSetClosure(plex, NULL, locF, support[0], &elemVec[face*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
    }
    ierr = DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree4(u, u_t, elemVec, a);CHKERRQ(ierr);
  }
  if (plex)  {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  if (plexA) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidualSingle(DM dm, PetscReal t, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, Vec locF)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMPlexComputeBdResidual_Single_Internal(dm, t, label, numValues, values, field, locX, locX_t, locF, coordField, facetIS);CHKERRQ(ierr);
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidual_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  PetscDS        prob;
  PetscInt       numBd, bd;
  DMField        coordField = NULL;
  IS             facetIS    = NULL;
  DMLabel        depthLabel;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel,dim - 1,&facetIS);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    DMBoundaryConditionType type;
    const char             *bdLabel;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                field, numValues;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &type, NULL, &bdLabel, &field, NULL, NULL, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    if (!facetIS) {
      DMLabel  depthLabel;
      PetscInt dim;

      ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
      ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(depthLabel, dim - 1, &facetIS);CHKERRQ(ierr);
    }
    ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    ierr = DMPlexComputeBdResidual_Single_Internal(dm, t, label, numValues, values, field, locX, locX_t, locF, coordField, facetIS);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Internal(DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Residual";
  DM               dmAux      = NULL;
  DM               dmGrad     = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          prob       = NULL;
  PetscDS          probAux    = NULL;
  PetscSection     section    = NULL;
  PetscBool        useFEM     = PETSC_FALSE;
  PetscBool        useFVM     = PETSC_FALSE;
  PetscBool        isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFV          fvm        = NULL;
  PetscFVCellGeom *cgeomFVM   = NULL;
  PetscFVFaceGeom *fgeomFVM   = NULL;
  DMField          coordField = NULL;
  Vec              locA, cellGeometryFVM = NULL, faceGeometryFVM = NULL, grad, locGrad = NULL;
  PetscScalar     *u = NULL, *u_t, *a, *uL, *uR;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, f, totDim, totDimAux, numChunks, cellChunkSize, faceChunkSize, chunk, fStart, fEnd;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
  /* FEM+FVM */
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    PetscInt subcell;
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosurePoint(dmAux, dm, DM_ENC_UNKNOWN, cStart, &subcell);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, subcell, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  /* 2: Get geometric data */
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    PetscBool    fimp;

    ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
    if (isImplicit != fimp) continue;
    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID) {useFEM = PETSC_TRUE;}
    if (id == PETSCFV_CLASSID) {useFVM = PETSC_TRUE; fvm = (PetscFV) obj;}
  }
  if (useFEM) {
    ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&affineQuad);CHKERRQ(ierr);
      if (affineQuad) {
        ierr = DMSNESGetFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      }
    } else {
      ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscObject  obj;
        PetscClassId id;
        PetscBool    fimp;

        ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
        if (isImplicit != fimp) continue;
        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id == PETSCFE_CLASSID) {
          PetscFE fe = (PetscFE) obj;

          ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
          ierr = PetscObjectReference((PetscObject)quads[f]);CHKERRQ(ierr);
          ierr = DMSNESGetFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        }
      }
    }
  }
  if (useFVM) {
    ierr = DMPlexSNESGetGeometryFVM(dm, &faceGeometryFVM, &cellGeometryFVM, NULL);CHKERRQ(ierr);
    ierr = VecGetArrayRead(faceGeometryFVM, (const PetscScalar **) &fgeomFVM);CHKERRQ(ierr);
    ierr = VecGetArrayRead(cellGeometryFVM, (const PetscScalar **) &cgeomFVM);CHKERRQ(ierr);
    /* Reconstruct and limit cell gradients */
    ierr = DMPlexSNESGetGradientDM(dm, fvm, &dmGrad);CHKERRQ(ierr);
    if (dmGrad) {
      ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
      ierr = DMPlexReconstructGradients_Internal(dm, fvm, fStart, fEnd, faceGeometryFVM, cellGeometryFVM, locX, grad);CHKERRQ(ierr);
      /* Communicate gradient values */
      ierr = DMGetLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dmGrad, grad, INSERT_VALUES, locGrad);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmGrad, &grad);CHKERRQ(ierr);
    }
    /* Handle non-essential (e.g. outflow) boundary values */
    ierr = DMPlexInsertBoundaryValues(dm, PETSC_FALSE, locX, time, faceGeometryFVM, cellGeometryFVM, locGrad);CHKERRQ(ierr);
  }
  /* Loop over chunks */
  if (useFEM) {ierr = ISCreate(PETSC_COMM_SELF, &chunkIS);CHKERRQ(ierr);}
  numCells      = cEnd - cStart;
  numChunks     = 1;
  cellChunkSize = numCells/numChunks;
  faceChunkSize = (fEnd - fStart)/numChunks;
  numChunks     = PetscMin(1,numCells);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscScalar     *elemVec, *fluxL, *fluxR;
    PetscReal       *vol;
    PetscFVFaceGeom *fgeom;
    PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;
    PetscInt         fS = fStart+chunk*faceChunkSize, fE = PetscMin(fS+faceChunkSize, fEnd), numFaces = 0, face;

    /* Extract field coefficients */
    if (useFEM) {
      ierr = ISGetPointSubrange(chunkIS, cS, cE, cells);CHKERRQ(ierr);
      ierr = DMPlexGetCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
      ierr = PetscArrayzero(elemVec, numCells*totDim);CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
      ierr = PetscArrayzero(fluxL, numFaces*totDim);CHKERRQ(ierr);
      ierr = PetscArrayzero(fluxR, numFaces*totDim);CHKERRQ(ierr);
    }
    /* TODO We will interlace both our field coefficients (u, u_t, uL, uR, etc.) and our output (elemVec, fL, fR). I think this works */
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscObject  obj;
      PetscClassId id;
      PetscBool    fimp;
      PetscInt     numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset;

      ierr = PetscDSGetImplicit(prob, f, &fimp);CHKERRQ(ierr);
      if (isImplicit != fimp) continue;
      ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
      if (id == PETSCFE_CLASSID) {
        PetscFE         fe = (PetscFE) obj;
        PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
        PetscFEGeom    *chunkGeom = NULL;
        PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
        PetscInt        Nq, Nb;

        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        blockSize = Nb;
        batchSize = numBlocks * blockSize;
        ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(prob, f, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(prob, f, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
        ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        ierr = PetscFVIntegrateRHSFunction(fv, prob, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
      } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %D", f);
    }
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (c = cS; c < cE; ++c) {
        const PetscInt cell = cells ? cells[c] : c;
        const PetscInt cind = c - cStart;

        if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
        if (ghostLabel) {
          PetscInt ghostVal;

          ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
          if (ghostVal > 0) continue;
        }
        ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
      }
    }
    if (useFVM) {
      PetscScalar *fa;
      PetscInt     iface;

      ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     foff, pdim;

        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscDSGetFieldOffset(prob, f, &foff);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
        /* Accumulate fluxes to cells */
        for (face = fS, iface = 0; face < fE; ++face) {
          const PetscInt *scells;
          PetscScalar    *fL = NULL, *fR = NULL;
          PetscInt        ghost, d, nsupp, nchild;

          ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
          ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
          ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
          if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
          ierr = DMPlexGetSupport(dm, face, &scells);CHKERRQ(ierr);
          ierr = DMLabelGetValue(ghostLabel,scells[0],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[0], f, fa, &fL);CHKERRQ(ierr);}
          ierr = DMLabelGetValue(ghostLabel,scells[1],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, scells[1], f, fa, &fR);CHKERRQ(ierr);}
          for (d = 0; d < pdim; ++d) {
            if (fL) fL[d] -= fluxL[iface*totDim+foff+d];
            if (fR) fR[d] += fluxR[iface*totDim+foff+d];
          }
          ++iface;
        }
      }
      ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
    }
    /* Handle time derivative */
    if (locX_t) {
      PetscScalar *x_t, *fa;

      ierr = VecGetArray(locF, &fa);CHKERRQ(ierr);
      ierr = VecGetArray(locX_t, &x_t);CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        PetscFV      fv;
        PetscObject  obj;
        PetscClassId id;
        PetscInt     pdim, d;

        ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
        ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
        if (id != PETSCFV_CLASSID) continue;
        fv   = (PetscFV) obj;
        ierr = PetscFVGetNumComponents(fv, &pdim);CHKERRQ(ierr);
        for (c = cS; c < cE; ++c) {
          const PetscInt cell = cells ? cells[c] : c;
          PetscScalar   *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel, cell, &ghostVal);CHKERRQ(ierr);
            if (ghostVal > 0) continue;
          }
          ierr = DMPlexPointLocalFieldRead(dm, cell, f, x_t, &u_t);CHKERRQ(ierr);
          ierr = DMPlexPointLocalFieldRef(dm, cell, f, fa, &r);CHKERRQ(ierr);
          for (d = 0; d < pdim; ++d) r[d] += u_t[d];
        }
      }
      ierr = VecRestoreArray(locX_t, &x_t);CHKERRQ(ierr);
      ierr = VecRestoreArray(locF, &fa);CHKERRQ(ierr);
    }
    if (useFEM) {
      ierr = DMPlexRestoreCellFields(dm, chunkIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, MPIU_SCALAR, &fluxR);CHKERRQ(ierr);
      if (dmGrad) {ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);}
    }
  }
  if (useFEM) {ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);}
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);

  if (useFEM) {
    ierr = DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user);CHKERRQ(ierr);

    if (maxDegree <= 1) {
      ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
      ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
    } else {
      for (f = 0; f < Nf; ++f) {
        ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);
      }
      ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
    }
  }

  /* FEM */
  /* 1: Get sizes from dm and dmAux */
  /* 2: Get geometric data */
  /* 3: Handle boundary values */
  /* 4: Loop over domain */
  /*   Extract coefficients */
  /* Loop over fields */
  /*   Set tiling for FE*/
  /*   Integrate FE residual to get elemVec */
  /*     Loop over subdomain */
  /*       Loop over quad points */
  /*         Transform coords to real space */
  /*         Evaluate field and aux fields at point */
  /*         Evaluate residual at point */
  /*         Transform residual to real space */
  /*       Add residual to elemVec */
  /* Loop over domain */
  /*   Add elemVec to locX */

  /* FVM */
  /* Get geometric data */
  /* If using gradients */
  /*   Compute gradient data */
  /*   Loop over domain faces */
  /*     Count computational faces */
  /*     Reconstruct cell gradient */
  /*   Loop over domain cells */
  /*     Limit cell gradients */
  /* Handle boundary values */
  /* Loop over domain faces */
  /*   Read out field, centroid, normal, volume for each side of face */
  /* Riemann solve over faces */
  /* Loop over domain faces */
  /*   Accumulate fluxes to cells */
  /* TODO Change printFEM to printDisc here */
  if (mesh->printFEM) {
    Vec         locFbc;
    PetscInt    pStart, pEnd, p, maxDof;
    PetscScalar *zeroes;

    ierr = VecDuplicate(locF,&locFbc);CHKERRQ(ierr);
    ierr = VecCopy(locF,locFbc);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(section,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetMaxDof(section,&maxDof);CHKERRQ(ierr);
    ierr = PetscCalloc1(maxDof,&zeroes);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      ierr = VecSetValuesSection(locFbc,section,p,zeroes,INSERT_BC_VALUES);CHKERRQ(ierr);
    }
    ierr = PetscFree(zeroes);CHKERRQ(ierr);
    ierr = DMPrintLocalVec(dm, name, mesh->printTol, locFbc);CHKERRQ(ierr);
    ierr = VecDestroy(&locFbc);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Hybrid_Internal(DM dm, IS cellIS, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Hybrid Residual";
  DM               dmAux      = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          prob       = NULL;
  PetscDS          probAux    = NULL;
  PetscSection     section    = NULL;
  DMField          coordField = NULL;
  Vec              locA;
  PetscScalar     *u = NULL, *u_t, *a;
  PetscScalar     *elemVec;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt        *faces;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, f, totDim, totDimAux, numChunks, cellChunkSize, chunk;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* FEM */
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, cStart, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  /* 2: Setup geometric data */
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree > 1) {
    ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) quads[f]);CHKERRQ(ierr);
    }
  }
  /* Loop over chunks */
  numCells      = cEnd - cStart;
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  ierr = PetscCalloc1(2*cellChunkSize, &faces);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS);CHKERRQ(ierr);
  /* Extract field coefficients */
  /* NOTE This needs the end cap faces to have identical orientations */
  ierr = DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, cellChunkSize*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    ierr = PetscMemzero(elemVec, cellChunkSize*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      ierr = DMPlexGetCone(dm, cell, &cone);CHKERRQ(ierr);
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    ierr = ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER);CHKERRQ(ierr);
    /* Get geometric data */
    if (maxDegree <= 1) {
      if (!affineQuad) {ierr = DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad);CHKERRQ(ierr);}
      if (affineQuad)  {ierr = DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom);CHKERRQ(ierr);}
    } else {
      for (f = 0; f < Nf; ++f) {
        ierr = DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]);CHKERRQ(ierr);
      }
    }
    /* Loop over fields */
    for (f = 0; f < Nf; ++f) {
      PetscFE         fe;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[f];
      PetscFEGeom    *chunkGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[f];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateHybridResidual(prob, f, Ne, chunkGeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEIntegrateHybridResidual(prob, f, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&chunkGeom);CHKERRQ(ierr);
    }
    /* Add elemVec to locX */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cStart;

      if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cind*totDim]);CHKERRQ(ierr);}
      if (ghostLabel) {
        PetscInt ghostVal;

        ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
        if (ghostVal > 0) continue;
      }
      ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cind*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, numCells*totDim, MPIU_SCALAR, &elemVec);CHKERRQ(ierr);
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
  } else {
    for (f = 0; f < Nf; ++f) {
      if (geoms) {ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE,&geoms[f]);CHKERRQ(ierr);}
      if (quads) {ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);}
    }
    ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESComputeResidualFEM - Form the local residual F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local solution
- user - The user context

  Output Parameter:
. F  - Local output vector

  Level: developer

.seealso: DMPlexComputeJacobianAction()
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM             plex;
  IS             cellIS;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {
    ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeResidual_Internal(plex, cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESComputeBoundaryFEM - Form the boundary values for the local input X

  Input Parameters:
+ dm - The mesh
- user - The user context

  Output Parameter:
. X  - Local solution

  Level: developer

.seealso: DMPlexComputeJacobianAction()
@*/
PetscErrorCode DMPlexSNESComputeBoundaryFEM(DM dm, Vec X, void *user)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexInsertBoundaryValues(plex, PETSC_TRUE, X, PETSC_MIN_REAL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Single_Internal(DM dm, PetscReal t, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt fieldI, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP, DMField coordField, IS facetIS)
{
  DM_Plex        *mesh = (DM_Plex *) dm->data;
  DM              plex = NULL, plexA = NULL, tdm;
  DMEnclosureType encAux;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, sectionAux = NULL;
  PetscSection    globalSection, subSection = NULL;
  Vec             locA = NULL, tv;
  PetscScalar    *u = NULL, *u_t = NULL, *a = NULL, *elemMat = NULL;
  PetscInt        v;
  PetscInt        Nf, totDim, totDimAux = 0;
  PetscBool       isMatISP, transform;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    DM dmAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetDS(plexA, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plexA, &sectionAux);CHKERRQ(ierr);
  }

  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);}
  for (v = 0; v < numValues; ++v) {
    PetscFEGeom    *fgeom;
    PetscInt        maxDegree;
    PetscQuadrature qGeom = NULL;
    IS              pointIS;
    const PetscInt *points;
    PetscInt        numFaces, face, Nq;

    ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
    if (!pointIS) continue; /* No points with that id on this process */
    {
      IS isectIS;

      /* TODO: Special cases of ISIntersect where it is quick to check a prior if one is a superset of the other */
      ierr = ISIntersect_Caching_Internal(facetIS,pointIS,&isectIS);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      pointIS = isectIS;
    }
    ierr = ISGetLocalSize(pointIS, &numFaces);CHKERRQ(ierr);
    ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = PetscMalloc4(numFaces*totDim, &u, locX_t ? numFaces*totDim : 0, &u_t, numFaces*totDim*totDim, &elemMat, locA ? numFaces*totDimAux : 0, &a);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,pointIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetFaceQuadrature(fe, &qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support, *cone;
      PetscScalar   *x     = NULL;
      PetscInt       i, coneSize, faceLoc;

      ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
      ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
      ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
      for (faceLoc = 0; faceLoc < coneSize; ++faceLoc) if (cone[faceLoc] == point) break;
      if (faceLoc == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %D in cone of support[0] %D", point, support[0]);
      fgeom->face[face][0] = faceLoc;
      ierr = DMPlexVecGetClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
      if (locX_t) {
        ierr = DMPlexVecGetClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plex, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
      }
      if (locA) {
        PetscInt subp;
        ierr = DMGetEnclosurePoint(plexA, dm, encAux, support[0], &subp);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
        ierr = DMPlexVecRestoreClosure(plexA, sectionAux, locA, subp, NULL, &x);CHKERRQ(ierr);
      }
    }
    ierr = PetscArrayzero(elemMat, numFaces*totDim*totDim);CHKERRQ(ierr);
    {
      PetscFE         fe;
      PetscInt        Nb;
      /* Conforming batches */
      PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
      /* Remainder */
      PetscFEGeom    *chunkGeom = NULL;
      PetscInt        fieldJ, Nr, offset;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numFaces / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numFaces % (numBatches*batchSize);
      offset    = numFaces - Nr;
      ierr = PetscFEGeomGetChunk(fgeom,0,offset,&chunkGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        ierr = PetscFEIntegrateBdJacobian(prob, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomGetChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        ierr = PetscFEIntegrateBdJacobian(prob, fieldI, fieldJ, Nr, chunkGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      ierr = PetscFEGeomRestoreChunk(fgeom,offset,numFaces,&chunkGeom);CHKERRQ(ierr);
    }
    for (face = 0; face < numFaces; ++face) {
      const PetscInt point = points[face], *support;

      /* Transform to global basis before insertion in Jacobian */
      ierr = DMPlexGetSupport(plex, point, &support);CHKERRQ(ierr);
      if (transform) {ierr = DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, support[0], PETSC_TRUE, totDim, &elemMat[face*totDim*totDim]);CHKERRQ(ierr);}
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(point, "BdJacobian", totDim, totDim, &elemMat[face*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP, &lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = DMSNESRestoreFEGeom(coordField,pointIS,qGeom,PETSC_TRUE,&fgeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
    ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
    ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
    ierr = PetscFree4(u, u_t, elemMat, a);CHKERRQ(ierr);
  }
  if (plex)  {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  if (plexA) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobianSingle(DM dm, PetscReal t, DMLabel label, PetscInt numValues, const PetscInt values[], PetscInt field, Vec locX, Vec locX_t, PetscReal X_tShift, Mat Jac, Mat JacP)
{
  DMField        coordField;
  DMLabel        depthLabel;
  IS             facetIS;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMPlexComputeBdJacobian_Single_Internal(dm, t, label, numValues, values, field, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS);CHKERRQ(ierr);
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdJacobian_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  PetscDS          prob;
  PetscInt         dim, numBd, bd;
  DMLabel          depthLabel;
  DMField          coordField = NULL;
  IS               facetIS;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depthLabel);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(depthLabel, dim-1, &facetIS);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  for (bd = 0; bd < numBd; ++bd) {
    DMBoundaryConditionType type;
    const char             *bdLabel;
    DMLabel                 label;
    const PetscInt         *values;
    PetscInt                fieldI, numValues;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &type, NULL, &bdLabel, &fieldI, NULL, NULL, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, fieldI, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    ierr = DMPlexComputeBdJacobian_Single_Internal(dm, t, label, numValues, values, fieldI, locX, locX_t, X_tShift, Jac, JacP, coordField, facetIS);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&facetIS);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Internal(DM dm, IS cellIS, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Mat Jac, Mat JacP,void *user)
{
  DM_Plex        *mesh  = (DM_Plex *) dm->data;
  const char     *name  = "Jacobian";
  DM              dmAux, plex, tdm;
  DMEnclosureType encAux;
  Vec             A, tv;
  DMField         coordField;
  PetscDS         prob, probAux = NULL;
  PetscSection    section, globalSection, subSection, sectionAux;
  PetscScalar    *elemMat, *elemMatP, *elemMatD, *u, *u_t, *a = NULL;
  const PetscInt *cells;
  PetscInt        Nf, fieldI, fieldJ;
  PetscInt        totDim, totDimAux, cStart, cEnd, numCells, c;
  PetscBool       isMatIS, isMatISP, hasJac, hasPrec, hasDyn, hasFV = PETSC_FALSE, transform;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMHasBasisTransform(dm, &transform);CHKERRQ(ierr);
  ierr = DMGetBasisTransformDM_Internal(dm, &tdm);CHKERRQ(ierr);
  ierr = DMGetBasisTransformVec_Internal(dm, &tv);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);}
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  /* user passed in the same matrix, avoid double contributions and
     only assemble the Jacobian */
  if (hasJac && Jac == JacP) hasPrec = PETSC_FALSE;
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = PetscMalloc5(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,hasJac ? numCells*totDim*totDim : 0,&elemMat,hasPrec ? numCells*totDim*totDim : 0, &elemMatP,hasDyn ? numCells*totDim*totDim : 0, &elemMatD);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    ierr = DMPlexVecGetClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      PetscInt subcell;
      ierr = DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plex, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
    }
  }
  if (hasJac)  {ierr = PetscArrayzero(elemMat,  numCells*totDim*totDim);CHKERRQ(ierr);}
  if (hasPrec) {ierr = PetscArrayzero(elemMatP, numCells*totDim*totDim);CHKERRQ(ierr);}
  if (hasDyn)  {ierr = PetscArrayzero(elemMatD, numCells*totDim*totDim);CHKERRQ(ierr);}
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscClassId    id;
    PetscFE         fe;
    PetscQuadrature qGeom = NULL;
    PetscInt        Nb;
    /* Conforming batches */
    PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt        Nr, offset, Nq;
    PetscInt        maxDegree;
    PetscFEGeom     *cgeomFEM, *chunkGeom = NULL, *remGeom = NULL;

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId((PetscObject) fe, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; continue;}
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom);CHKERRQ(ierr);
    }
    if (!qGeom) {
      ierr = PetscFEGetQuadrature(fe,&qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    ierr = PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      if (hasJac) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasPrec) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasDyn) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
  }
  /*   Add contribution from X_t */
  if (hasDyn) {for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];}
  if (hasFV) {
    PetscClassId id;
    PetscFV      fv;
    PetscInt     offsetI, NcI, NbI = 1, fc, f;

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fv);CHKERRQ(ierr);
      ierr = PetscDSGetFieldOffset(prob, fieldI, &offsetI);CHKERRQ(ierr);
      ierr = PetscObjectGetClassId((PetscObject) fv, &id);CHKERRQ(ierr);
      if (id != PETSCFV_CLASSID) continue;
      /* Put in the identity */
      ierr = PetscFVGetNumComponents(fv, &NcI);CHKERRQ(ierr);
      for (c = cStart; c < cEnd; ++c) {
        const PetscInt cind    = c - cStart;
        const PetscInt eOffset = cind*totDim*totDim;
        for (fc = 0; fc < NcI; ++fc) {
          for (f = 0; f < NbI; ++f) {
            const PetscInt i = offsetI + f*NcI+fc;
            if (hasPrec) {
              if (hasJac) {elemMat[eOffset+i*totDim+i] = 1.0;}
              elemMatP[eOffset+i*totDim+i] = 1.0;
            } else {elemMat[eOffset+i*totDim+i] = 1.0;}
          }
        }
      }
    }
    /* No allocated space for FV stuff, so ignore the zero entries */
    ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);CHKERRQ(ierr);
  }
  /* Insert values into matrix */
  isMatIS = PETSC_FALSE;
  if (hasPrec && hasJac) {
    ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatIS);CHKERRQ(ierr);
  }
  if (isMatIS && !subSection) {
    ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;

    /* Transform to global basis before insertion in Jacobian */
    if (transform) {ierr = DMPlexBasisTransformPointTensor_Internal(dm, tdm, tv, cell, PETSC_TRUE, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
    if (hasPrec) {
      if (hasJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatIS) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(Jac,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    } else {
      if (hasJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (hasFV) {ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);}
  ierr = PetscFree5(u,u_t,elemMat,elemMatP,elemMatD);CHKERRQ(ierr);
  if (dmAux) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  /* Compute boundary integrals */
  ierr = DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, user);CHKERRQ(ierr);
  /* Assemble matrix */
  if (hasJac && hasPrec) {
    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Hybrid_Internal(DM dm, IS cellIS, PetscReal t, PetscReal X_tShift, Vec locX, Vec locX_t, Mat Jac, Mat JacP, void *user)
{
  DM_Plex         *mesh       = (DM_Plex *) dm->data;
  const char      *name       = "Hybrid Jacobian";
  DM               dmAux      = NULL;
  DM               plex       = NULL;
  DM               plexA      = NULL;
  DMLabel          ghostLabel = NULL;
  PetscDS          prob       = NULL;
  PetscDS          probAux    = NULL;
  PetscSection     section    = NULL;
  DMField          coordField = NULL;
  Vec              locA;
  PetscScalar     *u = NULL, *u_t, *a = NULL;
  PetscScalar     *elemMat, *elemMatP;
  PetscSection     globalSection, subSection, sectionAux;
  IS               chunkIS;
  const PetscInt  *cells;
  PetscInt        *faces;
  PetscInt         cStart, cEnd, numCells;
  PetscInt         Nf, fieldI, fieldJ, totDim, totDimAux, numChunks, cellChunkSize, chunk;
  PetscInt         maxDegree = PETSC_MAX_INT;
  PetscQuadrature  affineQuad = NULL, *quads = NULL;
  PetscFEGeom     *affineGeom = NULL, **geoms = NULL;
  PetscBool        isMatIS = PETSC_FALSE, isMatISP = PETSC_FALSE, hasBdJac, hasBdPrec;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = DMConvert(dm, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMGetSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetCellDS(dm, cStart, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasBdJacobian(prob, &hasBdJac);CHKERRQ(ierr);
  ierr = PetscDSHasBdJacobianPreconditioner(prob, &hasBdPrec);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  if (isMatISP) {ierr = DMPlexGetSubdomainSection(plex, &subSection);CHKERRQ(ierr);}
  if (hasBdPrec && hasBdJac) {ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatIS);CHKERRQ(ierr);}
  if (isMatIS && !subSection) {ierr = DMPlexGetSubdomainSection(plex, &subSection);CHKERRQ(ierr);}
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexA);CHKERRQ(ierr);
    ierr = DMGetSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetCellDS(dmAux, cStart, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(coordField, cellIS, NULL, &maxDegree);CHKERRQ(ierr);
  if (maxDegree > 1) {
    PetscInt f;
    ierr = PetscCalloc2(Nf,&quads,Nf,&geoms);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscFE fe;

      ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetQuadrature(fe, &quads[f]);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject) quads[f]);CHKERRQ(ierr);
    }
  }
  cellChunkSize = numCells;
  numChunks     = !numCells ? 0 : PetscCeilReal(((PetscReal) numCells)/cellChunkSize);
  ierr = PetscCalloc1(2*cellChunkSize, &faces);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, cellChunkSize, faces, PETSC_USE_POINTER, &chunkIS);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP);CHKERRQ(ierr);
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscInt cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, c;

    if (hasBdJac)  {ierr = PetscMemzero(elemMat,  numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
    if (hasBdPrec) {ierr = PetscMemzero(elemMatP, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
    /* Get faces */
    for (c = cS; c < cE; ++c) {
      const PetscInt  cell = cells ? cells[c] : c;
      const PetscInt *cone;
      ierr = DMPlexGetCone(plex, cell, &cone);CHKERRQ(ierr);
      faces[(c-cS)*2+0] = cone[0];
      faces[(c-cS)*2+1] = cone[1];
    }
    ierr = ISGeneralSetIndices(chunkIS, cellChunkSize, faces, PETSC_USE_POINTER);CHKERRQ(ierr);
    if (maxDegree <= 1) {
      if (!affineQuad) {ierr = DMFieldCreateDefaultQuadrature(coordField, chunkIS, &affineQuad);CHKERRQ(ierr);}
      if (affineQuad)  {ierr = DMSNESGetFEGeom(coordField, chunkIS, affineQuad, PETSC_TRUE, &affineGeom);CHKERRQ(ierr);}
    } else {
      PetscInt f;
      for (f = 0; f < Nf; ++f) {
        ierr = DMSNESGetFEGeom(coordField, chunkIS, quads[f], PETSC_TRUE, &geoms[f]);CHKERRQ(ierr);
      }
    }

    for (fieldI = 0; fieldI < Nf; ++fieldI) {
      PetscFE         fe;
      PetscFEGeom    *geom = affineGeom ? affineGeom : geoms[fieldI];
      PetscFEGeom    *chunkGeom = NULL, *remGeom = NULL;
      PetscQuadrature quad = affineQuad ? affineQuad : quads[fieldI];
      PetscInt        numChunks, numBatches, batchSize, numBlocks, blockSize, Ne, Nr, offset, Nq, Nb;

      ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
      ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
      ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
      ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
      blockSize = Nb;
      batchSize = numBlocks * blockSize;
      ierr      = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
      numChunks = numCells / (numBatches*batchSize);
      Ne        = numChunks*numBatches*batchSize;
      Nr        = numCells % (numBatches*batchSize);
      offset    = numCells - Nr;
      ierr = PetscFEGeomGetChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomGetChunk(geom,offset,numCells,&remGeom);CHKERRQ(ierr);
      for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
        if (hasBdJac) {
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
        }
        if (hasBdPrec) {
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);
          ierr = PetscFEIntegrateHybridJacobian(prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]);CHKERRQ(ierr);
        }
      }
      ierr = PetscFEGeomRestoreChunk(geom,offset,numCells,&remGeom);CHKERRQ(ierr);
      ierr = PetscFEGeomRestoreChunk(geom,0,offset,&chunkGeom);CHKERRQ(ierr);
    }
    /* Insert values into matrix */
    for (c = cS; c < cE; ++c) {
      const PetscInt cell = cells ? cells[c] : c;
      const PetscInt cind = c - cS;

      if (hasBdPrec) {
        if (hasBdJac) {
          if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
          if (!isMatIS) {
            ierr = DMPlexMatSetClosure(plex, section, globalSection, Jac, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
          } else {
            Mat lJ;

            ierr = MatISGetLocalMat(Jac,&lJ);CHKERRQ(ierr);
            ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
          }
        }
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMatP[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMatP[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      } else if (hasBdJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(cell, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(plex, section, globalSection, JacP, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(plex, section, subSection, lJ, cell, &elemMat[cind*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = DMPlexRestoreCellFields(dm, cellIS, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, hasBdJac  ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMat);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, hasBdPrec ? cellChunkSize*totDim*totDim : 0, MPIU_SCALAR, &elemMatP);CHKERRQ(ierr);
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = ISDestroy(&chunkIS);CHKERRQ(ierr);
  ierr = ISRestorePointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  if (maxDegree <= 1) {
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,affineQuad,PETSC_FALSE,&affineGeom);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&affineQuad);CHKERRQ(ierr);
  } else {
    PetscInt f;
    for (f = 0; f < Nf; ++f) {
      if (geoms) {ierr = DMSNESRestoreFEGeom(coordField,cellIS,quads[f],PETSC_FALSE, &geoms[f]);CHKERRQ(ierr);}
      if (quads) {ierr = PetscQuadratureDestroy(&quads[f]);CHKERRQ(ierr);}
    }
    ierr = PetscFree2(quads,geoms);CHKERRQ(ierr);
  }
  if (dmAux) {ierr = DMDestroy(&plexA);CHKERRQ(ierr);}
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  /* Assemble matrix */
  if (hasBdJac && hasBdPrec) {
    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexComputeJacobianAction - Form the local portion of the Jacobian action Z = J(X) Y at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. cellIS -
. t  - The time
. X_tShift - The multiplier for the Jacobian with repsect to X_t
. X  - Local solution vector
. X_t  - Time-derivative of the local solution vector
. Y  - Local input vector
- user - The user context

  Output Parameter:
. Z - Local output vector

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: FormFunctionLocal()
@*/
PetscErrorCode DMPlexComputeJacobianAction(DM dm, IS cellIS, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Vec Y, Vec Z, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  const char       *name  = "Jacobian";
  DM                dmAux, plex, plexAux = NULL;
  DMEnclosureType   encAux;
  Vec               A;
  PetscDS           prob, probAux = NULL;
  PetscQuadrature   quad;
  PetscSection      section, globalSection, sectionAux;
  PetscScalar      *elemMat, *elemMatD, *u, *u_t, *a = NULL, *y, *z;
  PetscInt          Nf, fieldI, fieldJ;
  PetscInt          totDim, totDimAux = 0;
  const PetscInt   *cells;
  PetscInt          cStart, cEnd, numCells, c;
  PetscBool         hasDyn;
  DMField           coordField;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMSNESConvertPlex(dm, &plex, PETSC_TRUE);CHKERRQ(ierr);
  if (!cellIS) {
    PetscInt depth;

    ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
    ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
    if (!cellIS) {ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);}
  } else {
    ierr = PetscObjectReference((PetscObject) cellIS);CHKERRQ(ierr);
  }
  ierr = DMGetLocalSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = ISGetLocalSize(cellIS, &numCells);CHKERRQ(ierr);
  ierr = ISGetPointRange(cellIS, &cStart, &cEnd, &cells);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetEnclosureRelation(dmAux, dm, &encAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plexAux);CHKERRQ(ierr);
    ierr = DMGetLocalSection(plexAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = VecSet(Z, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*totDim*totDim,&elemMat,hasDyn ? numCells*totDim*totDim : 0, &elemMatD,numCells*totDim,&y,totDim,&z);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt cell = cells ? cells[c] : c;
    const PetscInt cind = c - cStart;
    PetscScalar   *x = NULL,  *x_t = NULL;
    PetscInt       i;

    ierr = DMPlexVecGetClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[cind*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, cell, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[cind*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, cell, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      PetscInt subcell;
      ierr = DMGetEnclosurePoint(dmAux, dm, encAux, cell, &subcell);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(plexAux, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[cind*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plexAux, sectionAux, A, subcell, NULL, &x);CHKERRQ(ierr);
    }
    ierr = DMPlexVecGetClosure(dm, section, Y, cell, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) y[cind*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, Y, cell, NULL, &x);CHKERRQ(ierr);
  }
  ierr = PetscArrayzero(elemMat, numCells*totDim*totDim);CHKERRQ(ierr);
  if (hasDyn)  {ierr = PetscArrayzero(elemMatD, numCells*totDim*totDim);CHKERRQ(ierr);}
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscFE  fe;
    PetscInt Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset, Nq;
    PetscQuadrature qGeom = NULL;
    PetscInt    maxDegree;
    PetscFEGeom *cgeomFEM, *chunkGeom = NULL, *remGeom = NULL;

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = DMFieldGetDegree(coordField,cellIS,NULL,&maxDegree);CHKERRQ(ierr);
    if (maxDegree <= 1) {ierr = DMFieldCreateDefaultQuadrature(coordField,cellIS,&qGeom);CHKERRQ(ierr);}
    if (!qGeom) {
      ierr = PetscFEGetQuadrature(fe,&qGeom);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)qGeom);CHKERRQ(ierr);
    }
    ierr = PetscQuadratureGetData(qGeom, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSNESGetFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    blockSize = Nb;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    ierr = PetscFEGeomGetChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomGetChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
      ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      if (hasDyn) {
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Ne, chunkGeom, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Nr, remGeom, &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]);CHKERRQ(ierr);
      }
    }
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,offset,numCells,&remGeom);CHKERRQ(ierr);
    ierr = PetscFEGeomRestoreChunk(cgeomFEM,0,offset,&chunkGeom);CHKERRQ(ierr);
    ierr = DMSNESRestoreFEGeom(coordField,cellIS,qGeom,PETSC_FALSE,&cgeomFEM);CHKERRQ(ierr);
    ierr = PetscQuadratureDestroy(&qGeom);CHKERRQ(ierr);
  }
  if (hasDyn) {
    for (c = 0; c < numCells*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscInt     cell = cells ? cells[c] : c;
    const PetscInt     cind = c - cStart;
    const PetscBLASInt M = totDim, one = 1;
    const PetscScalar  a = 1.0, b = 0.0;

    PetscStackCallBLAS("BLASgemv", BLASgemv_("N", &M, &M, &a, &elemMat[cind*totDim*totDim], &M, &y[cind*totDim], &one, &b, z, &one));
    if (mesh->printFEM > 1) {
      ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[cind*totDim*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Y",  totDim, &y[cind*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Z",  totDim, z);CHKERRQ(ierr);
    }
    ierr = DMPlexVecSetClosure(dm, section, Z, cell, z, ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree6(u,u_t,elemMat,elemMatD,y,z);CHKERRQ(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Z:\n");CHKERRQ(ierr);
    ierr = VecView(Z, NULL);CHKERRQ(ierr);
  }
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plexAux);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESComputeJacobianFEM - Form the local portion of the Jacobian matrix J at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. X  - Local input vector
- user - The user context

  Output Parameter:
. Jac  - Jacobian matrix

  Note:
  We form the residual one batch of elements at a time. This allows us to offload work onto an accelerator,
  like a GPU, or vectorize on a multicore machine.

  Level: developer

.seealso: FormFunctionLocal()
@*/
PetscErrorCode DMPlexSNESComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP,void *user)
{
  DM             plex;
  PetscDS        prob;
  IS             cellIS;
  PetscBool      hasJac, hasPrec;
  PetscInt       depth;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(plex, &depth);CHKERRQ(ierr);
  ierr = DMGetStratumIS(plex, "dim", depth, &cellIS);CHKERRQ(ierr);
  if (!cellIS) {ierr = DMGetStratumIS(plex, "depth", depth, &cellIS);CHKERRQ(ierr);}
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {ierr = MatZeroEntries(Jac);CHKERRQ(ierr);}
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = DMPlexComputeJacobian_Internal(plex, cellIS, 0.0, 0.0, X, NULL, Jac, JacP, user);CHKERRQ(ierr);
  ierr = ISDestroy(&cellIS);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatComputeNeumannOverlap - Computes an unassembled (Neumann) local overlapping Mat in nonlinear context.

   Input Parameters:
+     X - SNES linearization point
.     ovl - index set of overlapping subdomains

   Output Parameter:
.     J - unassembled (Neumann) local matrix

   Level: intermediate

.seealso: DMCreateNeumannOverlap(), MATIS, PCHPDDMSetAuxiliaryMat()
*/
static PetscErrorCode MatComputeNeumannOverlap_Plex(Mat J, PetscReal t, Vec X, Vec X_t, PetscReal s, IS ovl, void *ctx)
{
  SNES           snes;
  Mat            pJ;
  DM             ovldm,origdm;
  DMSNES         sdm;
  PetscErrorCode (*bfun)(DM,Vec,void*);
  PetscErrorCode (*jfun)(DM,Vec,Mat,Mat,void*);
  void           *bctx,*jctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_MATIS",(PetscObject*)&pJ);CHKERRQ(ierr);
  if (!pJ) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing overlapping Mat");CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ovl,"_DM_Original_HPDDM",(PetscObject*)&origdm);CHKERRQ(ierr);
  if (!origdm) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing original DM");CHKERRQ(ierr);
  ierr = MatGetDM(pJ,&ovldm);CHKERRQ(ierr);
  ierr = DMSNESGetBoundaryLocal(origdm,&bfun,&bctx);CHKERRQ(ierr);
  ierr = DMSNESSetBoundaryLocal(ovldm,bfun,bctx);CHKERRQ(ierr);
  ierr = DMSNESGetJacobianLocal(origdm,&jfun,&jctx);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(ovldm,jfun,jctx);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject*)&snes);CHKERRQ(ierr);
  if (!snes) {
    ierr = SNESCreate(PetscObjectComm((PetscObject)ovl),&snes);CHKERRQ(ierr);
    ierr = SNESSetDM(snes,ovldm);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject)snes);CHKERRQ(ierr);
    ierr = PetscObjectDereference((PetscObject)snes);CHKERRQ(ierr);
  }
  ierr = DMGetDMSNES(ovldm,&sdm);CHKERRQ(ierr);
  ierr = VecLockReadPush(X);CHKERRQ(ierr);
  PetscStackPush("SNES user Jacobian function");
  ierr = (*sdm->ops->computejacobian)(snes,X,pJ,pJ,sdm->jacobianctx);CHKERRQ(ierr);
  PetscStackPop;
  ierr = VecLockReadPop(X);CHKERRQ(ierr);
  /* this is a no-hop, just in case we decide to change the placeholder for the local Neumann matrix */
  {
    Mat locpJ;

    ierr = MatISGetLocalMat(pJ,&locpJ);CHKERRQ(ierr);
    ierr = MatCopy(locpJ,J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexSetSNESLocalFEM - Use DMPlex's internal FEM routines to compute SNES boundary values, residual, and Jacobian.

  Input Parameters:
+ dm - The DM object
. boundaryctx - the user context that will be passed to pointwise evaluation of boundary values (see PetscDSAddBoundary())
. residualctx - the user context that will be passed to pointwise evaluation of finite element residual computations (see PetscDSSetResidual())
- jacobianctx - the user context that will be passed to pointwise evaluation of finite element Jacobian construction (see PetscDSSetJacobian())

  Level: developer
@*/
PetscErrorCode DMPlexSetSNESLocalFEM(DM dm, void *boundaryctx, void *residualctx, void *jacobianctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESSetBoundaryLocal(dm,DMPlexSNESComputeBoundaryFEM,boundaryctx);CHKERRQ(ierr);
  ierr = DMSNESSetFunctionLocal(dm,DMPlexSNESComputeResidualFEM,residualctx);CHKERRQ(ierr);
  ierr = DMSNESSetJacobianLocal(dm,DMPlexSNESComputeJacobianFEM,jacobianctx);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,"MatComputeNeumannOverlap_C",MatComputeNeumannOverlap_Plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckDiscretization - Check the discretization error of the exact solution

  Input Parameters:
+ snes - the SNES object
. dm   - the DM
. u    - a DM vector
. exactFuncs - pointwise functions of the exact solution for each field
. ctxs - contexts for the functions
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
. error - An array which holds the discretization error in each field, or NULL

  Level: developer

.seealso: DNSNESCheckFromOptions(), DMSNESCheckResidual(), DMSNESCheckJacobian()
@*/
PetscErrorCode DMSNESCheckDiscretization(SNES snes, DM dm, Vec u, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx), void **ctxs, PetscReal tol, PetscReal error[])
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ectxs;
  MPI_Comm          comm;
  PetscDS           ds;
  PetscReal        *err;
  PetscInt          Nf, f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (error) PetscValidRealPointer(error, 6);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc3(Nf, &exacts, Nf, &ectxs, PetscMax(1, Nf), &err);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {ierr = PetscDSGetExactSolution(ds, f, &exacts[f], &ectxs[f]);CHKERRQ(ierr);}
  ierr = DMProjectFunction(dm, 0.0, exactFuncs ? exactFuncs : exacts, ctxs ? ctxs : ectxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) u, "exact_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-vec_view");CHKERRQ(ierr);
  if (Nf > 1) {
    ierr = DMComputeL2FieldDiff(dm, 0.0, exactFuncs ? exactFuncs : exacts, ctxs ? ctxs : ectxs, u, err);CHKERRQ(ierr);
    if (tol >= 0.0) {
      for (f = 0; f < Nf; ++f) {
        if (err[f] > tol) SETERRQ3(comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g for field %D exceeds tolerance %g", (double) err[f], f, (double) tol);
      }
    } else if (error) {
      for (f = 0; f < Nf; ++f) error[f] = err[f];
    } else {
      ierr = PetscPrintf(comm, "L_2 Error: [");CHKERRQ(ierr);
      for (f = 0; f < Nf; ++f) {
        if (f) {ierr = PetscPrintf(comm, ", ");CHKERRQ(ierr);}
        ierr = PetscPrintf(comm, "%g", (double)err[f]);CHKERRQ(ierr);
      }
      ierr = PetscPrintf(comm, "]\n");CHKERRQ(ierr);
    }
  } else {
    ierr = DMComputeL2Diff(dm, 0.0, exactFuncs ? exactFuncs : exacts, ctxs ? ctxs : ectxs , u, &err[0]);CHKERRQ(ierr);
    if (tol >= 0.0) {
      if (err[0] > tol) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g exceeds tolerance %g", (double) err[0], (double) tol);
    } else if (error) {
      error[0] = err[0];
    } else {
      ierr = PetscPrintf(comm, "L_2 Error: %g\n", (double)err[0]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree3(exacts, ectxs, err);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckResidual - Check the residual of the exact solution

  Input Parameters:
+ snes - the SNES object
. dm   - the DM
. u    - a DM vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
. residual - The residual norm of the exact solution, or NULL

  Level: developer

.seealso: DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckJacobian()
@*/
PetscErrorCode DMSNESCheckResidual(SNES snes, DM dm, Vec u, PetscReal tol, PetscReal *residual)
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ectxs;
  MPI_Comm          comm;
  PetscDS           ds;
  Vec               r;
  PetscReal         res;
  PetscInt          Nf, f;
  PetscBool         computeSol = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (residual) PetscValidRealPointer(residual, 5);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &exacts, Nf, &ectxs);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscDSGetExactSolution(ds, f, &exacts[f], &ectxs[f]);CHKERRQ(ierr);
    if (exacts[f]) computeSol = PETSC_TRUE;
  }
  if (computeSol) {ierr = DMProjectFunction(dm, 0.0, exacts, ectxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);}
  ierr = PetscFree2(exacts, ectxs);CHKERRQ(ierr);

  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  if (tol >= 0.0) {
    if (res > tol) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
  } else if (residual) {
    *residual = res;
  } else {
    ierr = PetscPrintf(comm, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
    ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) r, "Initial Residual");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"res_");CHKERRQ(ierr);
    ierr = VecViewFromOptions(r, NULL, "-vec_view");CHKERRQ(ierr);
  }
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckJacobian - Check the Jacobian of the exact solution against the residual using the Taylor Test

  Input Parameters:
+ snes - the SNES object
. dm   - the DM
. u    - a DM vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
+ isLinear - Flag indicaing that the function looks linear, or NULL
- convRate - The rate of convergence of the linear model, or NULL

  Level: developer

.seealso: DNSNESCheckFromOptions(), DMSNESCheckDiscretization(), DMSNESCheckResidual()
@*/
PetscErrorCode DMSNESCheckJacobian(SNES snes, DM dm, Vec u, PetscReal tol, PetscBool *isLinear, PetscReal *convRate)
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ectxs;
  MPI_Comm          comm;
  PetscDS           ds;
  Mat               J, M;
  MatNullSpace      nullspace;
  PetscReal         slope, intercept;
  PetscInt          Nf, f;
  PetscBool         hasJac, hasPrec, isLin = PETSC_FALSE, computeSol = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (isLinear) PetscValidBoolPointer(isLinear, 5);
  if (convRate) PetscValidRealPointer(convRate, 5);
  ierr = PetscObjectGetComm((PetscObject) snes, &comm);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  ierr = PetscMalloc2(Nf, &exacts, Nf, &ectxs);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    ierr = PetscDSGetExactSolution(ds, f, &exacts[f], &ectxs[f]);CHKERRQ(ierr);
    if (exacts[f]) computeSol = PETSC_TRUE;
  }
  if (computeSol) {ierr = DMProjectFunction(dm, 0.0, exacts, ectxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);}
  ierr = PetscFree2(exacts, ectxs);CHKERRQ(ierr);

  /* Create and view matrices */
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(ds, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(ds, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {
    ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(snes, u, J, M);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) M, "Preconditioning Matrix");CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_");CHKERRQ(ierr);
    ierr = MatViewFromOptions(M, NULL, "-mat_view");CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
  } else {
    ierr = SNESComputeJacobian(snes, u, J, J);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) J, "Jacobian");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject) J, "jac_");CHKERRQ(ierr);
  ierr = MatViewFromOptions(J, NULL, "-mat_view");CHKERRQ(ierr);
  /* Check nullspace */
  ierr = MatGetNullSpace(J, &nullspace);CHKERRQ(ierr);
  if (nullspace) {
    PetscBool isNull;
    ierr = MatNullSpaceTest(nullspace, J, &isNull);CHKERRQ(ierr);
    if (!isNull) SETERRQ(comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  ierr = MatNullSpaceDestroy(&nullspace);CHKERRQ(ierr);
  /* Taylor test */
  {
    PetscRandom rand;
    Vec         du, uhat, r, rhat, df;
    PetscReal   h;
    PetscReal  *es, *hs, *errors;
    PetscReal   hMax = 1.0, hMin = 1e-6, hMult = 0.1;
    PetscInt    Nv, v;

    /* Choose a perturbation direction */
    ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &du);CHKERRQ(ierr);
    ierr = VecSetRandom(du, rand); CHKERRQ(ierr);
    ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &df);CHKERRQ(ierr);
    ierr = MatMult(J, du, df);CHKERRQ(ierr);
    /* Evaluate residual at u, F(u), save in vector r */
    ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
    ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    ierr = PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &uhat);CHKERRQ(ierr);
    ierr = VecDuplicate(u, &rhat);CHKERRQ(ierr);
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      ierr = VecWAXPY(uhat, h, du, u);CHKERRQ(ierr);
      /* F(\hat u) \approx F(u) + J(u) (uhat - u) = F(u) + h * J(u) du */
      ierr = SNESComputeFunction(snes, uhat, rhat);CHKERRQ(ierr);
      ierr = VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df);CHKERRQ(ierr);
      ierr = VecNorm(rhat, NORM_2, &errors[Nv]);CHKERRQ(ierr);

      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    ierr = VecDestroy(&uhat);CHKERRQ(ierr);
    ierr = VecDestroy(&rhat);CHKERRQ(ierr);
    ierr = VecDestroy(&df);CHKERRQ(ierr);
    ierr = VecDestroy(&r);CHKERRQ(ierr);
    ierr = VecDestroy(&du);CHKERRQ(ierr);
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL)    break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    ierr = PetscLinearRegression(Nv, hs, es, &slope, &intercept);CHKERRQ(ierr);
    ierr = PetscFree3(es, hs, errors);CHKERRQ(ierr);
    /* Slope should be about 2 */
    if (tol >= 0) {
      if (!isLin && PetscAbsReal(2 - slope) > tol) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double) slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) {ierr = PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double) slope);CHKERRQ(ierr);}
      else        {ierr = PetscPrintf(comm, "Function appears to be linear\n");CHKERRQ(ierr);}
    }
  }
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESCheck_Internal(SNES snes, DM dm, Vec u, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx), void **ctxs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESCheckDiscretization(snes, dm, u, exactFuncs, ctxs, -1.0, NULL);CHKERRQ(ierr);
  ierr = DMSNESCheckResidual(snes, dm, u, -1.0, NULL);CHKERRQ(ierr);
  ierr = DMSNESCheckJacobian(snes, dm, u, -1.0, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ snes - the SNES object
. u    - representative SNES vector
. exactFuncs - pointwise functions of the exact solution for each field
- ctxs - contexts for the functions

  Level: developer
@*/
PetscErrorCode DMSNESCheckFromOptions(SNES snes, Vec u, PetscErrorCode (**exactFuncs)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx), void **ctxs)
{
  DM             dm;
  Vec            sol;
  PetscBool      check;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(((PetscObject)snes)->options,((PetscObject)snes)->prefix, "-dmsnes_check", &check);CHKERRQ(ierr);
  if (!check) PetscFunctionReturn(0);
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &sol);CHKERRQ(ierr);
  ierr = SNESSetSolution(snes, sol);CHKERRQ(ierr);
  ierr = DMSNESCheck_Internal(snes, dm, sol, exactFuncs, ctxs);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
