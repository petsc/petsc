#include <petsc/private/dmpleximpl.h>   /*I "petscdmplex.h" I*/
#include <petsc/private/snesimpl.h>     /*I "petscsnes.h"   I*/
#include <petscds.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/petscfeimpl.h>

static void pressure_Private(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar p[])
{
  p[0] = u[uOff[1]];
}

/*
  SNESCorrectDiscretePressure_Private - Add a vector in the nullspace to make the continuum integral of the pressure field equal to zero. This is normally used only to evaluate convergence rates for the pressure accurately.

  Collective on SNES

  Input Parameters:
+ snes      - The SNES
. pfield    - The field number for pressure
. nullspace - The pressure nullspace
. u         - The solution vector
- ctx       - An optional user context

  Output Parameter:
. u         - The solution with a continuum pressure integral of zero

  Notes:
  If int(u) = a and int(n) = b, then int(u - a/b n) = a - a/b b = 0. We assume that the nullspace is a single vector given explicitly.

  Level: developer

.seealso: `SNESConvergedCorrectPressure()`
*/
static PetscErrorCode SNESCorrectDiscretePressure_Private(SNES snes, PetscInt pfield, MatNullSpace nullspace, Vec u, void *ctx)
{
  DM             dm;
  PetscDS        ds;
  const Vec     *nullvecs;
  PetscScalar    pintd, *intc, *intn;
  MPI_Comm       comm;
  PetscInt       Nf, Nv;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject) snes, &comm));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCheck(dm,comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a SNES DM");
  PetscCheck(nullspace,comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a Jacobian nullspace");
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, pfield, pressure_Private));
  PetscCall(MatNullSpaceGetVecs(nullspace, NULL, &Nv, &nullvecs));
  PetscCheck(Nv == 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "Can only handle a single null vector for pressure, not %" PetscInt_FMT, Nv);
  PetscCall(VecDot(nullvecs[0], u, &pintd));
  PetscCheck(PetscAbsScalar(pintd) <= PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g", (double) PetscRealPart(pintd));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCall(PetscMalloc2(Nf, &intc, Nf, &intn));
  PetscCall(DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, ctx));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  PetscCall(VecAXPY(u, -intc[pfield]/intn[pfield], nullvecs[0]));
#if defined (PETSC_USE_DEBUG)
  PetscCall(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  PetscCheck(PetscAbsScalar(intc[pfield]) <= PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g", (double) PetscRealPart(intc[pfield]));
#endif
  PetscCall(PetscFree2(intc, intn));
  PetscFunctionReturn(0);
}

/*@C
   SNESConvergedCorrectPressure - Convergence test that adds a vector in the nullspace to make the continuum integral of the pressure field equal to zero. This is normally used only to evaluate convergence rates for the pressure accurately. The convergence test itself just mimics SNESConvergedDefault().

   Logically Collective on SNES

   Input Parameters:
+  snes - the SNES context
.  it - the iteration (0 indicates before any Newton steps)
.  xnorm - 2-norm of current iterate
.  snorm - 2-norm of current step
.  fnorm - 2-norm of function at current iterate
-  ctx   - Optional user context

   Output Parameter:
.  reason  - SNES_CONVERGED_ITERATING, SNES_CONVERGED_ITS, or SNES_DIVERGED_FNORM_NAN

   Notes:
   In order to use this convergence test, you must set up several PETSc structures. First fields must be added to the DM, and a PetscDS must be created with discretizations of those fields. We currently assume that the pressure field has index 1. The pressure field must have a nullspace, likely created using the DMSetNullSpaceConstructor() interface. Last we must be able to integrate the pressure over the domain, so the DM attached to the SNES must be a Plex at this time.

   Options Database Keys:
.  -snes_convergence_test correct_pressure - see SNESSetFromOptions()

   Level: advanced

.seealso: `SNESConvergedDefault()`, `SNESSetConvergenceTest()`, `DMSetNullSpaceConstructor()`
@*/
PetscErrorCode SNESConvergedCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *ctx)
{
  PetscBool      monitorIntegral = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, ctx));
  if (monitorIntegral) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    const Vec   *nullvecs;
    PetscScalar  pintd;

    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(MatGetNullSpace(J, &nullspace));
    PetscCall(MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs));
    PetscCall(VecDot(nullvecs[0], u, &pintd));
    PetscCall(PetscInfo(snes, "SNES: Discrete integral of pressure: %g\n", (double) PetscRealPart(pintd)));
  }
  if (*reason > 0) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    PetscInt     pfield = 1;

    PetscCall(SNESGetSolution(snes, &u));
    PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    PetscCall(MatGetNullSpace(J, &nullspace));
    PetscCall(SNESCorrectDiscretePressure_Private(snes, pfield, nullspace, u, ctx));
  }
  PetscFunctionReturn(0);
}

/************************** Interpolation *******************************/

static PetscErrorCode DMSNESConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    PetscCall(PetscObjectReference((PetscObject) dm));
  } else {
    PetscCall(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      PetscCall(DMConvert(dm,DMPLEX,plex));
      PetscCall(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        PetscCall(DMCopyDMSNES(dm, *plex));
        PetscCall(DMCopyAuxiliaryVec(dm, *plex));
      }
    } else {
      PetscCall(PetscObjectReference((PetscObject) *plex));
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

.seealso: `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationDestroy()`
@*/
PetscErrorCode DMInterpolationCreate(MPI_Comm comm, DMInterpolationInfo *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx, 2);
  PetscCall(PetscNew(ctx));

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

.seealso: `DMInterpolationGetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ctx, PetscInt dim)
{
  PetscFunctionBegin;
  PetscCheck(!(dim < 1) && !(dim > 3),ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %" PetscInt_FMT, dim);
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

.seealso: `DMInterpolationSetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
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

.seealso: `DMInterpolationGetDof()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
@*/
PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ctx, PetscInt dof)
{
  PetscFunctionBegin;
  PetscCheck(dof >= 1,ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %" PetscInt_FMT, dof);
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

.seealso: `DMInterpolationSetDof()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`
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

.seealso: `DMInterpolationSetDim()`, `DMInterpolationEvaluate()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ctx, PetscInt n, PetscReal points[])
{
  PetscFunctionBegin;
  PetscCheck(ctx->dim >= 0,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  PetscCheck(!ctx->points,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  ctx->nInput = n;

  PetscCall(PetscMalloc1(n*ctx->dim, &ctx->points));
  PetscCall(PetscArraycpy(ctx->points, points, n*ctx->dim));
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationSetUp - Compute spatial indices for point location during interpolation

  Collective on ctx

  Input Parameters:
+ ctx - the context
. dm  - the DM for the function space used for interpolation
. redundantPoints - If PETSC_TRUE, all processes are passing in the same array of points. Otherwise, points need to be communicated among processes.
- ignoreOutsideDomain - If PETSC_TRUE, ignore points outside the domain, otherwise return an error

  Level: intermediate

.seealso: `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationSetUp(DMInterpolationInfo ctx, DM dm, PetscBool redundantPoints, PetscBool ignoreOutsideDomain)
{
  MPI_Comm          comm = ctx->comm;
  PetscScalar       *a;
  PetscInt          p, q, i;
  PetscMPIInt       rank, size;
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
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCheck(ctx->dim >= 0,comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    PetscCall(PetscLayoutCreate(comm, &layout));
    PetscCall(PetscLayoutSetBlockSize(layout, 1));
    PetscCall(PetscLayoutSetLocalSize(layout, n));
    PetscCall(PetscLayoutSetUp(layout));
    PetscCall(PetscLayoutGetSize(layout, &N));
    /* Communicate all points to all processes */
    PetscCall(PetscMalloc3(N*ctx->dim,&globalPoints,size,&counts,size,&displs));
    PetscCall(PetscLayoutGetRanges(layout, &ranges));
    for (p = 0; p < size; ++p) {
      counts[p] = (ranges[p+1] - ranges[p])*ctx->dim;
      displs[p] = ranges[p]*ctx->dim;
    }
    PetscCallMPI(MPI_Allgatherv(ctx->points, n*ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm));
  } else {
    N = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout = NULL;
  }
#if 0
  PetscCall(PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs));
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
#if defined(PETSC_USE_COMPLEX)
  PetscCall(PetscMalloc1(N*ctx->dim,&globalPointsScalar));
  for (i=0; i<N*ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
#else
  globalPointsScalar = globalPoints;
#endif
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N*ctx->dim, globalPointsScalar, &pointVec));
  PetscCall(PetscMalloc2(N,&foundProcs,N,&globalProcs));
  for (p = 0; p < N; ++p) {foundProcs[p] = size;}
  cellSF = NULL;
  PetscCall(DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF));
  PetscCall(PetscSFGetGraph(cellSF,NULL,&numFound,&foundPoints,&foundCells));
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  PetscCall(MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm));
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      PetscCheck(ignoreOutsideDomain,comm, PETSC_ERR_PLIB, "Point %" PetscInt_FMT ": %g %g %g not located in mesh", p, (double)globalPoints[p*ctx->dim+0], (double)(ctx->dim > 1 ? globalPoints[p*ctx->dim+1] : 0.0), (double)(ctx->dim > 2 ? globalPoints[p*ctx->dim+2] : 0.0));
      else if (rank == 0) ++ctx->n;
    } else if (globalProcs[p] == rank) ++ctx->n;
  }
  /* Create coordinates vector and array of owned cells */
  PetscCall(PetscMalloc1(ctx->n, &ctx->cells));
  PetscCall(VecCreate(comm, &ctx->coords));
  PetscCall(VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(ctx->coords, ctx->dim));
  PetscCall(VecSetType(ctx->coords,VECSTANDARD));
  PetscCall(VecGetArray(ctx->coords, &a));
  for (p = 0, q = 0, i = 0; p < N; ++p) {
    if (globalProcs[p] == rank) {
      PetscInt d;

      for (d = 0; d < ctx->dim; ++d, ++i) a[i] = globalPoints[p*ctx->dim+d];
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
  PetscCall(PetscFree2(foundProcs,globalProcs));
  PetscCall(PetscSFDestroy(&cellSF));
  PetscCall(VecDestroy(&pointVec));
#endif
  if ((void*)globalPointsScalar != (void*)globalPoints) PetscCall(PetscFree(globalPointsScalar));
  if (!redundantPoints) PetscCall(PetscFree3(globalPoints,counts,displs));
  PetscCall(PetscLayoutDestroy(&layout));
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

.seealso: `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationGetCoordinates(DMInterpolationInfo ctx, Vec *coordinates)
{
  PetscFunctionBegin;
  PetscValidPointer(coordinates, 2);
  PetscCheck(ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
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

.seealso: `DMInterpolationRestoreVector()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationGetVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  PetscCheck(ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  PetscCall(VecCreate(ctx->comm, v));
  PetscCall(VecSetSizes(*v, ctx->n*ctx->dof, PETSC_DECIDE));
  PetscCall(VecSetBlockSize(*v, ctx->dof));
  PetscCall(VecSetType(*v,VECSTANDARD));
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationRestoreVector - Returns a Vec which can hold all the interpolated field values

  Collective on ctx

  Input Parameters:
+ ctx - the context
- v  - a vector capable of holding the interpolated field values

  Level: intermediate

.seealso: `DMInterpolationGetVector()`, `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationRestoreVector(DMInterpolationInfo ctx, Vec *v)
{
  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  PetscCheck(ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  PetscCall(VecDestroy(v));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Segment_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal          v0, J, invJ, detJ;
  const PetscInt     dof = ctx->dof;
  const PetscScalar *coords;
  PetscScalar       *a;
  PetscInt           p;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt     c = ctx->cells[p];
    PetscScalar *x = NULL;
    PetscReal    xir[1];
    PetscInt     xSize, comp;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0, &J, &invJ, &detJ));
    PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double) detJ, c);
    xir[0] = invJ*PetscRealPart(coords[p] - v0);
    PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    if (2*dof == xSize) {
      for (comp = 0; comp < dof; ++comp) a[p*dof+comp] = x[0*dof+comp]*(1 - xir[0]) + x[1*dof+comp]*xir[0];
    } else if (dof == xSize) {
      for (comp = 0; comp < dof; ++comp) a[p*dof+comp] = x[0*dof+comp];
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input closure size %" PetscInt_FMT " must be either %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 2*dof, dof);
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Triangle_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ));
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt     c = ctx->cells[p];
    PetscScalar *x = NULL;
    PetscReal    xi[4];
    PetscInt     d, f, comp;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    PetscCheck(detJ > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double)detJ, c);
    PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[(d+1)*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  }
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Tetrahedron_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;

  PetscFunctionBegin;
  PetscCall(PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ));
  PetscCall(VecGetArrayRead(ctx->coords, &coords));
  PetscCall(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt       c = ctx->cells[p];
    const PetscInt order[3] = {2, 1, 3};
    PetscScalar   *x = NULL;
    PetscReal      xi[4];
    PetscInt       d, f, comp;

    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    PetscCheck(detJ > 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double)detJ, c);
    PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[order[d]*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  }
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
  PetscCall(PetscFree3(v0, J, invJ));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode QuadMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
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

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref,  &ref));
  PetscCall(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_01 * p0 * p1;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_01 * p0 * p1;
  }
  PetscCall(PetscLogFlops(28));
  PetscCall(VecRestoreArrayRead(Xref,  &ref));
  PetscCall(VecRestoreArray(Xreal, &real));
  PetscFunctionReturn(0);
}

#include <petsc/private/dmimpl.h>
static inline PetscErrorCode QuadJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
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

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref,  &ref));
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscInt    rows[2] = {0, 1};
    PetscScalar       values[4];

    values[0] = (x1 - x0 + f_01*y) * 0.5; values[1] = (x3 - x0 + f_01*x) * 0.5;
    values[2] = (y1 - y0 + g_01*y) * 0.5; values[3] = (y3 - y0 + g_01*x) * 0.5;
    PetscCall(MatSetValues(J, 2, rows, 2, rows, values, INSERT_VALUES));
  }
  PetscCall(PetscLogFlops(30));
  PetscCall(VecRestoreArrayRead(Xref,  &ref));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Quad_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  DM                 dmCoord;
  PetscFE            fem = NULL;
  SNES               snes;
  KSP                ksp;
  PC                 pc;
  Vec                coordsLocal, r, ref, real;
  Mat                J;
  PetscTabulation    T = NULL;
  const PetscScalar *coords;
  PetscScalar        *a;
  PetscReal          xir[2] = {0., 0.};
  PetscInt           Nf, p;
  const PetscInt     dof = ctx->dof;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(dm, &Nf));
  if (Nf) {
    PetscObject  obj;
    PetscClassId id;

    PetscCall(DMGetField(dm, 0, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {fem = (PetscFE) obj; PetscCall(PetscFECreateTabulation(fem, 1, 1, xir, 0, &T));}
  }
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateDM(dm, &dmCoord));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "quad_interp_"));
  PetscCall(VecCreate(PETSC_COMM_SELF, &r));
  PetscCall(VecSetSizes(r, 2, 2));
  PetscCall(VecSetType(r,dm->vectype));
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
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    PetscCall(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCheck(4*2 == coordSize,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %" PetscInt_FMT " should be %d", coordSize, 4*2);
    PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    PetscCall(SNESSetFunction(snes, NULL, NULL, vertices));
    PetscCall(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
    PetscCall(VecGetArray(real, &xi));
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    PetscCall(VecRestoreArray(real, &xi));
    PetscCall(SNESSolve(snes, real, ref));
    PetscCall(VecGetArray(ref, &xi));
    xir[0] = PetscRealPart(xi[0]);
    xir[1] = PetscRealPart(xi[1]);
    if (4*dof == xSize) {
      for (comp = 0; comp < dof; ++comp)
        a[p*dof+comp] = x[0*dof+comp]*(1 - xir[0])*(1 - xir[1]) + x[1*dof+comp]*xir[0]*(1 - xir[1]) + x[2*dof+comp]*xir[0]*xir[1] + x[3*dof+comp]*(1 - xir[0])*xir[1];
    } else if (dof == xSize) {
      for (comp = 0; comp < dof; ++comp) a[p*dof+comp] = x[0*dof+comp];
    } else {
      PetscInt d;

      PetscCheck(fem, ctx->comm, PETSC_ERR_ARG_WRONG, "Cannot have a higher order interpolant if the discretization is not PetscFE");
      xir[0] = 2.0*xir[0] - 1.0; xir[1] = 2.0*xir[1] - 1.0;
      PetscCall(PetscFEComputeTabulation(fem, 1, xir, 0, T));
      for (comp = 0; comp < dof; ++comp) {
        a[p*dof+comp] = 0.0;
        for (d = 0; d < xSize/dof; ++d) {
          a[p*dof+comp] += x[d*dof+comp]*T->T[0][d*dof+comp];
        }
      }
    }
    PetscCall(VecRestoreArray(ref, &xi));
    PetscCall(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  PetscCall(PetscTabulationDestroy(&T));
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&real));
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode HexMap_Private(SNES snes, Vec Xref, Vec Xreal, void *ctx)
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

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref,  &ref));
  PetscCall(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];
    const PetscScalar p2 = ref[2];

    real[0] = x0 + f_1*p0 + f_3*p1 + f_4*p2 + f_01*p0*p1 + f_12*p1*p2 + f_02*p0*p2 + f_012*p0*p1*p2;
    real[1] = y0 + g_1*p0 + g_3*p1 + g_4*p2 + g_01*p0*p1 + g_01*p0*p1 + g_12*p1*p2 + g_02*p0*p2 + g_012*p0*p1*p2;
    real[2] = z0 + h_1*p0 + h_3*p1 + h_4*p2 + h_01*p0*p1 + h_01*p0*p1 + h_12*p1*p2 + h_02*p0*p2 + h_012*p0*p1*p2;
  }
  PetscCall(PetscLogFlops(114));
  PetscCall(VecRestoreArrayRead(Xref,  &ref));
  PetscCall(VecRestoreArray(Xreal, &real));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode HexJacobian_Private(SNES snes, Vec Xref, Mat J, Mat M, void *ctx)
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

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(Xref,  &ref));
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

    PetscCall(MatSetValues(J, 3, rows, 3, rows, values, INSERT_VALUES));
  }
  PetscCall(PetscLogFlops(152));
  PetscCall(VecRestoreArrayRead(Xref,  &ref));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Hex_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
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

  PetscFunctionBegin;
  PetscCall(DMGetCoordinatesLocal(dm, &coordsLocal));
  PetscCall(DMGetCoordinateDM(dm, &dmCoord));
  PetscCall(SNESCreate(PETSC_COMM_SELF, &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "hex_interp_"));
  PetscCall(VecCreate(PETSC_COMM_SELF, &r));
  PetscCall(VecSetSizes(r, 3, 3));
  PetscCall(VecSetType(r,dm->vectype));
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
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscReal    xir[3];
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    PetscCall(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCheck(8*3 == coordSize,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid coordinate closure size %" PetscInt_FMT " should be %d", coordSize, 8*3);
    PetscCall(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    PetscCheck((8*ctx->dof == xSize) || (ctx->dof == xSize),ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input closure size %" PetscInt_FMT " should be %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 8*ctx->dof, ctx->dof);
    PetscCall(SNESSetFunction(snes, NULL, NULL, vertices));
    PetscCall(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
    PetscCall(VecGetArray(real, &xi));
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    xi[2]  = coords[p*ctx->dim+2];
    PetscCall(VecRestoreArray(real, &xi));
    PetscCall(SNESSolve(snes, real, ref));
    PetscCall(VecGetArray(ref, &xi));
    xir[0] = PetscRealPart(xi[0]);
    xir[1] = PetscRealPart(xi[1]);
    xir[2] = PetscRealPart(xi[2]);
    if (8*ctx->dof == xSize) {
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
    } else {
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];
    }
    PetscCall(VecRestoreArray(ref, &xi));
    PetscCall(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  PetscCall(VecRestoreArray(v, &a));
  PetscCall(VecRestoreArrayRead(ctx->coords, &coords));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&real));
  PetscCall(MatDestroy(&J));
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

.seealso: `DMInterpolationGetVector()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationEvaluate(DMInterpolationInfo ctx, DM dm, Vec x, Vec v)
{
  PetscDS        ds;
  PetscInt       n, p, Nf, field;
  PetscBool      useDS = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  PetscCall(VecGetLocalSize(v, &n));
  PetscCheck(n == ctx->n*ctx->dof,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %" PetscInt_FMT " should be %" PetscInt_FMT, n, ctx->n*ctx->dof);
  if (!n) PetscFunctionReturn(0);
  PetscCall(DMGetDS(dm, &ds));
  if (ds) {
    useDS = PETSC_TRUE;
    PetscCall(PetscDSGetNumFields(ds, &Nf));
    for (field = 0; field < Nf; ++field) {
      PetscObject  obj;
      PetscClassId id;

      PetscCall(PetscDSGetDiscretization(ds, field, &obj));
      PetscCall(PetscObjectGetClassId(obj, &id));
      if (id != PETSCFE_CLASSID) {useDS = PETSC_FALSE; break;}
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
      for (d = 0; d < cdim; ++d) pcoords[d] = PetscRealPart(coords[p*cdim+d]);
      PetscCall(DMPlexCoordinatesToReference(dm, ctx->cells[p], 1, pcoords, xi));
      PetscCall(DMPlexVecGetClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      for (field = 0; field < Nf; ++field) {
        PetscTabulation T;
        PetscFE         fe;

        PetscCall(PetscDSGetDiscretization(ds, field, (PetscObject *) &fe));
        PetscCall(PetscFECreateTabulation(fe, 1, 1, xi, 0, &T));
        {
          const PetscReal *basis = T->T[0];
          const PetscInt   Nb    = T->Nb;
          const PetscInt   Nc    = T->Nc;
          PetscInt         f, fc;

          for (fc = 0; fc < Nc; ++fc) {
            interpolant[p*ctx->dof+coff+fc] = 0.0;
            for (f = 0; f < Nb; ++f) {
              interpolant[p*ctx->dof+coff+fc] += xa[foff+f]*basis[(0*Nb + f)*Nc + fc];
            }
          }
          coff += Nc;
          foff += Nb;
        }
        PetscCall(PetscTabulationDestroy(&T));
      }
      PetscCall(DMPlexVecRestoreClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      PetscCheck(coff == ctx->dof,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total components %" PetscInt_FMT " != %" PetscInt_FMT " dof specified for interpolation", coff, ctx->dof);
      PetscCheck(foff == clSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total FE space size %" PetscInt_FMT " != %" PetscInt_FMT " closure size", foff, clSize);
    }
    PetscCall(VecRestoreArrayRead(ctx->coords, &coords));
    PetscCall(VecRestoreArrayWrite(v, &interpolant));
  } else {
    DMPolytopeType ct;

    /* TODO Check each cell individually */
    PetscCall(DMPlexGetCellType(dm, ctx->cells[0], &ct));
    switch (ct) {
      case DM_POLYTOPE_SEGMENT:       PetscCall(DMInterpolate_Segment_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_TRIANGLE:      PetscCall(DMInterpolate_Triangle_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_QUADRILATERAL: PetscCall(DMInterpolate_Quad_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_TETRAHEDRON:   PetscCall(DMInterpolate_Tetrahedron_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_HEXAHEDRON:    PetscCall(DMInterpolate_Hex_Private(ctx, dm, x, v));break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No support for cell type %s", DMPolytopeTypes[PetscMax(0, PetscMin(ct, DM_NUM_POLYTOPES))]);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMInterpolationDestroy - Destroys a DMInterpolationInfo context

  Collective on ctx

  Input Parameter:
. ctx - the context

  Level: beginner

.seealso: `DMInterpolationEvaluate()`, `DMInterpolationAddPoints()`, `DMInterpolationCreate()`
@*/
PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx, 1);
  PetscCall(VecDestroy(&(*ctx)->coords));
  PetscCall(PetscFree((*ctx)->points));
  PetscCall(PetscFree((*ctx)->cells));
  PetscCall(PetscFree(*ctx));
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

.seealso: `SNESMonitorSet()`, `SNESMonitorDefault()`
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  PetscCall(SNESGetFunction(snes, &res, NULL, NULL));
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &numFields));
  PetscCall(PetscSectionGetChart(s, &pStart, &pEnd));
  PetscCall(PetscCalloc2(numFields, &lnorms, numFields, &norms));
  PetscCall(VecGetArrayRead(res, &r));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < numFields; ++f) {
      PetscInt fdof, foff, d;

      PetscCall(PetscSectionGetFieldDof(s, p, f, &fdof));
      PetscCall(PetscSectionGetFieldOffset(s, p, f, &foff));
      for (d = 0; d < fdof; ++d) lnorms[f] += PetscRealPart(PetscSqr(r[foff+d]));
    }
  }
  PetscCall(VecRestoreArrayRead(res, &r));
  PetscCall(MPIU_Allreduce(lnorms, norms, numFields, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) dm)));
  PetscCall(PetscViewerPushFormat(viewer,vf->format));
  PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject) snes)->tablevel));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%3" PetscInt_FMT " SNES Function norm %14.12e [", its, (double) fgnorm));
  for (f = 0; f < numFields; ++f) {
    if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%14.12e", (double) PetscSqrtReal(norms[f])));
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "]\n"));
  PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject) snes)->tablevel));
  PetscCall(PetscViewerPopFormat(viewer));
  PetscCall(PetscFree2(lnorms, norms));
  PetscFunctionReturn(0);
}

/********************* Residual Computation **************************/

PetscErrorCode DMPlexGetAllCells_Internal(DM plex, IS *cellIS)
{
  PetscInt       depth;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepth(plex, &depth));
  PetscCall(DMGetStratumIS(plex, "dim", depth, cellIS));
  if (!*cellIS) PetscCall(DMGetStratumIS(plex, "depth", depth, cellIS));
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESComputeResidualFEM - Sums the local residual into vector F from the local input X using pointwise functions specified by the user

  Input Parameters:
+ dm - The mesh
. X  - Local solution
- user - The user context

  Output Parameter:
. F  - Local output vector

  Notes:
  The residual is summed into F; the caller is responsible for using VecZeroEntries() or otherwise ensuring that any data in F is intentional.

  Level: developer

.seealso: `DMPlexComputeJacobianAction()`
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(DMPlexComputeResidual_Internal(plex, key, cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESComputeResidual(DM dm, Vec X, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    PetscCall(DMGetRegionNumDS(dm, s, &label, NULL, &ds));
    {
      PetscWeakFormKind resmap[2] = {PETSC_WF_F0, PETSC_WF_F1};
      PetscWeakForm     wf;
      PetscInt          Nm = 2, m, Nk = 0, k, kp, off = 0;
      PetscFormKey *reskeys;

      /* Get unique residual keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        PetscCall(PetscHMapFormGetSize(ds->wf->form[resmap[m]], &Nkm));
        Nk  += Nkm;
      }
      PetscCall(PetscMalloc1(Nk, &reskeys));
      for (m = 0; m < Nm; ++m) {
        PetscCall(PetscHMapFormGetKeys(ds->wf->form[resmap[m]], &off, reskeys));
      }
      PetscCheck(off == Nk,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %" PetscInt_FMT " should be %" PetscInt_FMT, off, Nk);
      PetscCall(PetscFormKeySort(Nk, reskeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((reskeys[k].label != reskeys[kp].label) || (reskeys[k].value != reskeys[kp].value)) {
          ++k;
          if (kp != k) reskeys[k] = reskeys[kp];
        }
      }
      Nk = k;

      PetscCall(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = reskeys[k].label;
        PetscInt val   = reskeys[k].value;

        if (!label) {
          PetscCall(PetscObjectReference((PetscObject) allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          PetscCall(DMLabelGetStratumIS(label, val, &pointIS));
          PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          PetscCall(ISDestroy(&pointIS));
        }
        PetscCall(DMPlexComputeResidual_Internal(plex, reskeys[k], cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
        PetscCall(ISDestroy(&cellIS));
      }
      PetscCall(PetscFree(reskeys));
    }
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
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

.seealso: `DMPlexComputeJacobianAction()`
@*/
PetscErrorCode DMPlexSNESComputeBoundaryFEM(DM dm, Vec X, void *user)
{
  DM             plex;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm,&plex,PETSC_TRUE));
  PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, X, PETSC_MIN_REAL, NULL, NULL, NULL));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

/*@
  DMSNESComputeJacobianAction - Compute the action of the Jacobian J(X) on Y

  Input Parameters:
+ dm   - The DM
. X    - Local solution vector
. Y    - Local input vector
- user - The user context

  Output Parameter:
. F    - lcoal output vector

  Level: developer

  Notes:
  Users will typically use DMSNESCreateJacobianMF() followed by MatMult() instead of calling this routine directly.

.seealso: `DMSNESCreateJacobianMF()`, `DMPlexSNESComputeResidualFEM()`
@*/
PetscErrorCode DMSNESComputeJacobianAction(DM dm, Vec X, Vec Y, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    PetscCall(DMGetRegionNumDS(dm, s, &label, NULL, &ds));
    {
      PetscWeakFormKind jacmap[4] = {PETSC_WF_G0, PETSC_WF_G1, PETSC_WF_G2, PETSC_WF_G3};
      PetscWeakForm     wf;
      PetscInt          Nm = 4, m, Nk = 0, k, kp, off = 0;
      PetscFormKey *jackeys;

      /* Get unique Jacobian keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        PetscCall(PetscHMapFormGetSize(ds->wf->form[jacmap[m]], &Nkm));
        Nk  += Nkm;
      }
      PetscCall(PetscMalloc1(Nk, &jackeys));
      for (m = 0; m < Nm; ++m) {
        PetscCall(PetscHMapFormGetKeys(ds->wf->form[jacmap[m]], &off, jackeys));
      }
      PetscCheck(off == Nk,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %" PetscInt_FMT " should be %" PetscInt_FMT, off, Nk);
      PetscCall(PetscFormKeySort(Nk, jackeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((jackeys[k].label != jackeys[kp].label) || (jackeys[k].value != jackeys[kp].value)) {
          ++k;
          if (kp != k) jackeys[k] = jackeys[kp];
        }
      }
      Nk = k;

      PetscCall(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = jackeys[k].label;
        PetscInt val   = jackeys[k].value;

        if (!label) {
          PetscCall(PetscObjectReference((PetscObject) allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          PetscCall(DMLabelGetStratumIS(label, val, &pointIS));
          PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          PetscCall(ISDestroy(&pointIS));
        }
        PetscCall(DMPlexComputeJacobian_Action_Internal(plex, jackeys[k], cellIS, 0.0, 0.0, X, NULL, Y, F, user));
        PetscCall(ISDestroy(&cellIS));
      }
      PetscCall(PetscFree(jackeys));
    }
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
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

.seealso: `FormFunctionLocal()`
@*/
PetscErrorCode DMPlexSNESComputeJacobianFEM(DM dm, Vec X, Mat Jac, Mat JacP,void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscBool      hasJac, hasPrec;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  PetscCall(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  PetscCall(DMPlexGetAllCells_Internal(plex, &allcellIS));
  PetscCall(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    PetscCall(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      PetscCall(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      PetscCall(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      PetscCall(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      PetscCall(ISDestroy(&pointIS));
    }
    if (!s) {
      PetscCall(PetscDSHasJacobian(ds, &hasJac));
      PetscCall(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
      if (hasJac && hasPrec) PetscCall(MatZeroEntries(Jac));
      PetscCall(MatZeroEntries(JacP));
    }
    PetscCall(DMPlexComputeJacobian_Internal(plex, key, cellIS, 0.0, 0.0, X, NULL, Jac, JacP, user));
    PetscCall(ISDestroy(&cellIS));
  }
  PetscCall(ISDestroy(&allcellIS));
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

struct _DMSNESJacobianMFCtx
{
  DM    dm;
  Vec   X;
  void *ctx;
};

static PetscErrorCode DMSNESJacobianMF_Destroy_Private(Mat A)
{
  struct _DMSNESJacobianMFCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(MatShellSetContext(A, NULL));
  PetscCall(DMDestroy(&ctx->dm));
  PetscCall(VecDestroy(&ctx->X));
  PetscCall(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESJacobianMF_Mult_Private(Mat A, Vec Y, Vec Z)
{
  struct _DMSNESJacobianMFCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &ctx));
  PetscCall(DMSNESComputeJacobianAction(ctx->dm, ctx->X, Y, Z, ctx->ctx));
  PetscFunctionReturn(0);
}

/*@
  DMSNESCreateJacobianMF - Create a Mat which computes the action of the Jacobian matrix-free

  Collective on dm

  Input Parameters:
+ dm   - The DM
. X    - The evaluation point for the Jacobian
- user - A user context, or NULL

  Output Parameter:
. J    - The Mat

  Level: advanced

  Notes:
  Vec X is kept in Mat J, so updating X then updates the evaluation point.

.seealso: `DMSNESComputeJacobianAction()`
@*/
PetscErrorCode DMSNESCreateJacobianMF(DM dm, Vec X, void *user, Mat *J)
{
  struct _DMSNESJacobianMFCtx *ctx;
  PetscInt                     n, N;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject) dm), J));
  PetscCall(MatSetType(*J, MATSHELL));
  PetscCall(VecGetLocalSize(X, &n));
  PetscCall(VecGetSize(X, &N));
  PetscCall(MatSetSizes(*J, n, n, N, N));
  PetscCall(PetscObjectReference((PetscObject) dm));
  PetscCall(PetscObjectReference((PetscObject) X));
  PetscCall(PetscMalloc1(1, &ctx));
  ctx->dm  = dm;
  ctx->X   = X;
  ctx->ctx = user;
  PetscCall(MatShellSetContext(*J, ctx));
  PetscCall(MatShellSetOperation(*J, MATOP_DESTROY, (void (*)(void)) DMSNESJacobianMF_Destroy_Private));
  PetscCall(MatShellSetOperation(*J, MATOP_MULT,    (void (*)(void)) DMSNESJacobianMF_Mult_Private));
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

.seealso: `DMCreateNeumannOverlap()`, `MATIS`, `PCHPDDMSetAuxiliaryMat()`
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

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_MATIS",(PetscObject*)&pJ));
  PetscCheck(pJ,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing overlapping Mat");
  PetscCall(PetscObjectQuery((PetscObject)ovl,"_DM_Original_HPDDM",(PetscObject*)&origdm));
  PetscCheck(origdm,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing original DM");
  PetscCall(MatGetDM(pJ,&ovldm));
  PetscCall(DMSNESGetBoundaryLocal(origdm,&bfun,&bctx));
  PetscCall(DMSNESSetBoundaryLocal(ovldm,bfun,bctx));
  PetscCall(DMSNESGetJacobianLocal(origdm,&jfun,&jctx));
  PetscCall(DMSNESSetJacobianLocal(ovldm,jfun,jctx));
  PetscCall(PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject*)&snes));
  if (!snes) {
    PetscCall(SNESCreate(PetscObjectComm((PetscObject)ovl),&snes));
    PetscCall(SNESSetDM(snes,ovldm));
    PetscCall(PetscObjectCompose((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject)snes));
    PetscCall(PetscObjectDereference((PetscObject)snes));
  }
  PetscCall(DMGetDMSNES(ovldm,&sdm));
  PetscCall(VecLockReadPush(X));
  PetscStackPush("SNES user Jacobian function");
  PetscCall((*sdm->ops->computejacobian)(snes,X,pJ,pJ,sdm->jacobianctx));
  PetscStackPop;
  PetscCall(VecLockReadPop(X));
  /* this is a no-hop, just in case we decide to change the placeholder for the local Neumann matrix */
  {
    Mat locpJ;

    PetscCall(MatISGetLocalMat(pJ,&locpJ));
    PetscCall(MatCopy(locpJ,J,SAME_NONZERO_PATTERN));
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
  PetscFunctionBegin;
  PetscCall(DMSNESSetBoundaryLocal(dm,DMPlexSNESComputeBoundaryFEM,boundaryctx));
  PetscCall(DMSNESSetFunctionLocal(dm,DMPlexSNESComputeResidualFEM,residualctx));
  PetscCall(DMSNESSetJacobianLocal(dm,DMPlexSNESComputeJacobianFEM,jacobianctx));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"MatComputeNeumannOverlap_C",MatComputeNeumannOverlap_Plex));
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckDiscretization - Check the discretization error of the exact solution

  Input Parameters:
+ snes - the SNES object
. dm   - the DM
. t    - the time
. u    - a DM vector
- tol  - A tolerance for the check, or -1 to print the results instead

  Output Parameters:
. error - An array which holds the discretization error in each field, or NULL

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer

.seealso: `DNSNESCheckFromOptions()`, `DMSNESCheckResidual()`, `DMSNESCheckJacobian()`, `PetscDSSetExactSolution()`
@*/
PetscErrorCode DMSNESCheckDiscretization(SNES snes, DM dm, PetscReal t, Vec u, PetscReal tol, PetscReal error[])
{
  PetscErrorCode (**exacts)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx);
  void            **ectxs;
  PetscReal        *err;
  MPI_Comm          comm;
  PetscInt          Nf, f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  if (error) PetscValidRealPointer(error, 6);

  PetscCall(DMComputeExactSolution(dm, t, u, NULL));
  PetscCall(VecViewFromOptions(u, NULL, "-vec_view"));

  PetscCall(PetscObjectGetComm((PetscObject) snes, &comm));
  PetscCall(DMGetNumFields(dm, &Nf));
  PetscCall(PetscCalloc3(Nf, &exacts, Nf, &ectxs, PetscMax(1, Nf), &err));
  {
    PetscInt Nds, s;

    PetscCall(DMGetNumDS(dm, &Nds));
    for (s = 0; s < Nds; ++s) {
      PetscDS         ds;
      DMLabel         label;
      IS              fieldIS;
      const PetscInt *fields;
      PetscInt        dsNf, f;

      PetscCall(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds));
      PetscCall(PetscDSGetNumFields(ds, &dsNf));
      PetscCall(ISGetIndices(fieldIS, &fields));
      for (f = 0; f < dsNf; ++f) {
        const PetscInt field = fields[f];
        PetscCall(PetscDSGetExactSolution(ds, field, &exacts[field], &ectxs[field]));
      }
      PetscCall(ISRestoreIndices(fieldIS, &fields));
    }
  }
  if (Nf > 1) {
    PetscCall(DMComputeL2FieldDiff(dm, t, exacts, ectxs, u, err));
    if (tol >= 0.0) {
      for (f = 0; f < Nf; ++f) {
        PetscCheck(err[f] <= tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g for field %" PetscInt_FMT " exceeds tolerance %g", (double) err[f], f, (double) tol);
      }
    } else if (error) {
      for (f = 0; f < Nf; ++f) error[f] = err[f];
    } else {
      PetscCall(PetscPrintf(comm, "L_2 Error: ["));
      for (f = 0; f < Nf; ++f) {
        if (f) PetscCall(PetscPrintf(comm, ", "));
        PetscCall(PetscPrintf(comm, "%g", (double)err[f]));
      }
      PetscCall(PetscPrintf(comm, "]\n"));
    }
  } else {
    PetscCall(DMComputeL2Diff(dm, t, exacts, ectxs, u, &err[0]));
    if (tol >= 0.0) {
      PetscCheck(err[0] <= tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g exceeds tolerance %g", (double) err[0], (double) tol);
    } else if (error) {
      error[0] = err[0];
    } else {
      PetscCall(PetscPrintf(comm, "L_2 Error: %g\n", (double) err[0]));
    }
  }
  PetscCall(PetscFree3(exacts, ectxs, err));
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

.seealso: `DNSNESCheckFromOptions()`, `DMSNESCheckDiscretization()`, `DMSNESCheckJacobian()`
@*/
PetscErrorCode DMSNESCheckResidual(SNES snes, DM dm, Vec u, PetscReal tol, PetscReal *residual)
{
  MPI_Comm       comm;
  Vec            r;
  PetscReal      res;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (residual) PetscValidRealPointer(residual, 5);
  PetscCall(PetscObjectGetComm((PetscObject) snes, &comm));
  PetscCall(DMComputeExactSolution(dm, 0.0, u, NULL));
  PetscCall(VecDuplicate(u, &r));
  PetscCall(SNESComputeFunction(snes, u, r));
  PetscCall(VecNorm(r, NORM_2, &res));
  if (tol >= 0.0) {
    PetscCheck(res <= tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
  } else if (residual) {
    *residual = res;
  } else {
    PetscCall(PetscPrintf(comm, "L_2 Residual: %g\n", (double)res));
    PetscCall(VecChop(r, 1.0e-10));
    PetscCall(PetscObjectSetName((PetscObject) r, "Initial Residual"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)r,"res_"));
    PetscCall(VecViewFromOptions(r, NULL, "-vec_view"));
  }
  PetscCall(VecDestroy(&r));
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

.seealso: `DNSNESCheckFromOptions()`, `DMSNESCheckDiscretization()`, `DMSNESCheckResidual()`
@*/
PetscErrorCode DMSNESCheckJacobian(SNES snes, DM dm, Vec u, PetscReal tol, PetscBool *isLinear, PetscReal *convRate)
{
  MPI_Comm       comm;
  PetscDS        ds;
  Mat            J, M;
  MatNullSpace   nullspace;
  PetscReal      slope, intercept;
  PetscBool      hasJac, hasPrec, isLin = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (isLinear) PetscValidBoolPointer(isLinear, 5);
  if (convRate) PetscValidRealPointer(convRate, 6);
  PetscCall(PetscObjectGetComm((PetscObject) snes, &comm));
  PetscCall(DMComputeExactSolution(dm, 0.0, u, NULL));
  /* Create and view matrices */
  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSHasJacobian(ds, &hasJac));
  PetscCall(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
  if (hasJac && hasPrec) {
    PetscCall(DMCreateMatrix(dm, &M));
    PetscCall(SNESComputeJacobian(snes, u, J, M));
    PetscCall(PetscObjectSetName((PetscObject) M, "Preconditioning Matrix"));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_"));
    PetscCall(MatViewFromOptions(M, NULL, "-mat_view"));
    PetscCall(MatDestroy(&M));
  } else {
    PetscCall(SNESComputeJacobian(snes, u, J, J));
  }
  PetscCall(PetscObjectSetName((PetscObject) J, "Jacobian"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) J, "jac_"));
  PetscCall(MatViewFromOptions(J, NULL, "-mat_view"));
  /* Check nullspace */
  PetscCall(MatGetNullSpace(J, &nullspace));
  if (nullspace) {
    PetscBool isNull;
    PetscCall(MatNullSpaceTest(nullspace, J, &isNull));
    PetscCheck(isNull,comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  /* Taylor test */
  {
    PetscRandom rand;
    Vec         du, uhat, r, rhat, df;
    PetscReal   h;
    PetscReal  *es, *hs, *errors;
    PetscReal   hMax = 1.0, hMin = 1e-6, hMult = 0.1;
    PetscInt    Nv, v;

    /* Choose a perturbation direction */
    PetscCall(PetscRandomCreate(comm, &rand));
    PetscCall(VecDuplicate(u, &du));
    PetscCall(VecSetRandom(du, rand));
    PetscCall(PetscRandomDestroy(&rand));
    PetscCall(VecDuplicate(u, &df));
    PetscCall(MatMult(J, du, df));
    /* Evaluate residual at u, F(u), save in vector r */
    PetscCall(VecDuplicate(u, &r));
    PetscCall(SNESComputeFunction(snes, u, r));
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    PetscCall(PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors));
    PetscCall(VecDuplicate(u, &uhat));
    PetscCall(VecDuplicate(u, &rhat));
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      PetscCall(VecWAXPY(uhat, h, du, u));
      /* F(\hat u) \approx F(u) + J(u) (uhat - u) = F(u) + h * J(u) du */
      PetscCall(SNESComputeFunction(snes, uhat, rhat));
      PetscCall(VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df));
      PetscCall(VecNorm(rhat, NORM_2, &errors[Nv]));

      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    PetscCall(VecDestroy(&uhat));
    PetscCall(VecDestroy(&rhat));
    PetscCall(VecDestroy(&df));
    PetscCall(VecDestroy(&r));
    PetscCall(VecDestroy(&du));
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL)    break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    PetscCall(PetscLinearRegression(Nv, hs, es, &slope, &intercept));
    PetscCall(PetscFree3(es, hs, errors));
    /* Slope should be about 2 */
    if (tol >= 0) {
      PetscCheck(isLin || PetscAbsReal(2 - slope) <= tol,comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double) slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) PetscCall(PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double) slope));
      else        PetscCall(PetscPrintf(comm, "Function appears to be linear\n"));
    }
  }
  PetscCall(MatDestroy(&J));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESCheck_Internal(SNES snes, DM dm, Vec u)
{
  PetscFunctionBegin;
  PetscCall(DMSNESCheckDiscretization(snes, dm, 0.0, u, -1.0, NULL));
  PetscCall(DMSNESCheckResidual(snes, dm, u, -1.0, NULL));
  PetscCall(DMSNESCheckJacobian(snes, dm, u, -1.0, NULL, NULL));
  PetscFunctionReturn(0);
}

/*@C
  DMSNESCheckFromOptions - Check the residual and Jacobian functions using the exact solution by outputting some diagnostic information

  Input Parameters:
+ snes - the SNES object
- u    - representative SNES vector

  Note: The user must call PetscDSSetExactSolution() beforehand

  Level: developer
@*/
PetscErrorCode DMSNESCheckFromOptions(SNES snes, Vec u)
{
  DM             dm;
  Vec            sol;
  PetscBool      check;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHasName(((PetscObject)snes)->options,((PetscObject)snes)->prefix, "-dmsnes_check", &check));
  if (!check) PetscFunctionReturn(0);
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(VecDuplicate(u, &sol));
  PetscCall(SNESSetSolution(snes, sol));
  PetscCall(DMSNESCheck_Internal(snes, dm, sol));
  PetscCall(VecDestroy(&sol));
  PetscFunctionReturn(0);
}
