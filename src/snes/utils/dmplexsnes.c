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

.seealso: SNESConvergedCorrectPressure()
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
  CHKERRQ(PetscObjectGetComm((PetscObject) snes, &comm));
  CHKERRQ(SNESGetDM(snes, &dm));
  PetscCheckFalse(!dm,comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a SNES DM");
  PetscCheckFalse(!nullspace,comm, PETSC_ERR_ARG_WRONG, "Cannot compute test without a Jacobian nullspace");
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSSetObjective(ds, pfield, pressure_Private));
  CHKERRQ(MatNullSpaceGetVecs(nullspace, NULL, &Nv, &nullvecs));
  PetscCheckFalse(Nv != 1,comm, PETSC_ERR_ARG_OUTOFRANGE, "Can only handle a single null vector for pressure, not %D", Nv);
  CHKERRQ(VecDot(nullvecs[0], u, &pintd));
  PetscCheckFalse(PetscAbsScalar(pintd) > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Discrete integral of pressure: %g", (double) PetscRealPart(pintd));
  CHKERRQ(PetscDSGetNumFields(ds, &Nf));
  CHKERRQ(PetscMalloc2(Nf, &intc, Nf, &intn));
  CHKERRQ(DMPlexComputeIntegralFEM(dm, nullvecs[0], intn, ctx));
  CHKERRQ(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  CHKERRQ(VecAXPY(u, -intc[pfield]/intn[pfield], nullvecs[0]));
#if defined (PETSC_USE_DEBUG)
  CHKERRQ(DMPlexComputeIntegralFEM(dm, u, intc, ctx));
  PetscCheckFalse(PetscAbsScalar(intc[pfield]) > PETSC_SMALL,comm, PETSC_ERR_ARG_WRONG, "Continuum integral of pressure after correction: %g", (double) PetscRealPart(intc[pfield]));
#endif
  CHKERRQ(PetscFree2(intc, intn));
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
   In order to use this monitor, you must setup several PETSc structures. First fields must be added to the DM, and a PetscDS must be created with discretizations of those fields. We currently assume that the pressure field has index 1. The pressure field must have a nullspace, likely created using the DMSetNullSpaceConstructor() interface. Last we must be able to integrate the pressure over the domain, so the DM attached to the SNES must be a Plex at this time.

   Level: advanced

.seealso: SNESConvergedDefault(), SNESSetConvergenceTest(), DMSetNullSpaceConstructor()
@*/
PetscErrorCode SNESConvergedCorrectPressure(SNES snes, PetscInt it, PetscReal xnorm, PetscReal gnorm, PetscReal f, SNESConvergedReason *reason, void *ctx)
{
  PetscBool      monitorIntegral = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(SNESConvergedDefault(snes, it, xnorm, gnorm, f, reason, ctx));
  if (monitorIntegral) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    const Vec   *nullvecs;
    PetscScalar  pintd;

    CHKERRQ(SNESGetSolution(snes, &u));
    CHKERRQ(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    CHKERRQ(MatGetNullSpace(J, &nullspace));
    CHKERRQ(MatNullSpaceGetVecs(nullspace, NULL, NULL, &nullvecs));
    CHKERRQ(VecDot(nullvecs[0], u, &pintd));
    CHKERRQ(PetscInfo(snes, "SNES: Discrete integral of pressure: %g\n", (double) PetscRealPart(pintd)));
  }
  if (*reason > 0) {
    Mat          J;
    Vec          u;
    MatNullSpace nullspace;
    PetscInt     pfield = 1;

    CHKERRQ(SNESGetSolution(snes, &u));
    CHKERRQ(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
    CHKERRQ(MatGetNullSpace(J, &nullspace));
    CHKERRQ(SNESCorrectDiscretePressure_Private(snes, pfield, nullspace, u, ctx));
  }
  PetscFunctionReturn(0);
}

/************************** Interpolation *******************************/

static PetscErrorCode DMSNESConvertPlex(DM dm, DM *plex, PetscBool copy)
{
  PetscBool      isPlex;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject) dm, DMPLEX, &isPlex));
  if (isPlex) {
    *plex = dm;
    CHKERRQ(PetscObjectReference((PetscObject) dm));
  } else {
    CHKERRQ(PetscObjectQuery((PetscObject) dm, "dm_plex", (PetscObject *) plex));
    if (!*plex) {
      CHKERRQ(DMConvert(dm,DMPLEX,plex));
      CHKERRQ(PetscObjectCompose((PetscObject) dm, "dm_plex", (PetscObject) *plex));
      if (copy) {
        CHKERRQ(DMCopyDMSNES(dm, *plex));
        CHKERRQ(DMCopyAuxiliaryVec(dm, *plex));
      }
    } else {
      CHKERRQ(PetscObjectReference((PetscObject) *plex));
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
  PetscFunctionBegin;
  PetscValidPointer(ctx, 2);
  CHKERRQ(PetscNew(ctx));

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
  PetscCheckFalse((dim < 1) || (dim > 3),ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %D", dim);
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
  PetscCheckFalse(dof < 1,ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %D", dof);
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
  PetscFunctionBegin;
  PetscCheckFalse(ctx->dim < 0,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  PetscCheckFalse(ctx->points,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  ctx->nInput = n;

  CHKERRQ(PetscMalloc1(n*ctx->dim, &ctx->points));
  CHKERRQ(PetscArraycpy(ctx->points, points, n*ctx->dim));
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

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
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
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  PetscCheckFalse(ctx->dim < 0,comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  /* Locate points */
  n = ctx->nInput;
  if (!redundantPoints) {
    CHKERRQ(PetscLayoutCreate(comm, &layout));
    CHKERRQ(PetscLayoutSetBlockSize(layout, 1));
    CHKERRQ(PetscLayoutSetLocalSize(layout, n));
    CHKERRQ(PetscLayoutSetUp(layout));
    CHKERRQ(PetscLayoutGetSize(layout, &N));
    /* Communicate all points to all processes */
    CHKERRQ(PetscMalloc3(N*ctx->dim,&globalPoints,size,&counts,size,&displs));
    CHKERRQ(PetscLayoutGetRanges(layout, &ranges));
    for (p = 0; p < size; ++p) {
      counts[p] = (ranges[p+1] - ranges[p])*ctx->dim;
      displs[p] = ranges[p]*ctx->dim;
    }
    CHKERRMPI(MPI_Allgatherv(ctx->points, n*ctx->dim, MPIU_REAL, globalPoints, counts, displs, MPIU_REAL, comm));
  } else {
    N = n;
    globalPoints = ctx->points;
    counts = displs = NULL;
    layout = NULL;
  }
#if 0
  CHKERRQ(PetscMalloc3(N,&foundCells,N,&foundProcs,N,&globalProcs));
  /* foundCells[p] = m->locatePoint(&globalPoints[p*ctx->dim]); */
#else
#if defined(PETSC_USE_COMPLEX)
  CHKERRQ(PetscMalloc1(N*ctx->dim,&globalPointsScalar));
  for (i=0; i<N*ctx->dim; i++) globalPointsScalar[i] = globalPoints[i];
#else
  globalPointsScalar = globalPoints;
#endif
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF, ctx->dim, N*ctx->dim, globalPointsScalar, &pointVec));
  CHKERRQ(PetscMalloc2(N,&foundProcs,N,&globalProcs));
  for (p = 0; p < N; ++p) {foundProcs[p] = size;}
  cellSF = NULL;
  CHKERRQ(DMLocatePoints(dm, pointVec, DM_POINTLOCATION_REMOVE, &cellSF));
  CHKERRQ(PetscSFGetGraph(cellSF,NULL,&numFound,&foundPoints,&foundCells));
#endif
  for (p = 0; p < numFound; ++p) {
    if (foundCells[p].index >= 0) foundProcs[foundPoints ? foundPoints[p] : p] = rank;
  }
  /* Let the lowest rank process own each point */
  CHKERRMPI(MPIU_Allreduce(foundProcs, globalProcs, N, MPI_INT, MPI_MIN, comm));
  ctx->n = 0;
  for (p = 0; p < N; ++p) {
    if (globalProcs[p] == size) {
      PetscCheckFalse(!ignoreOutsideDomain,comm, PETSC_ERR_PLIB, "Point %d: %g %g %g not located in mesh", p, (double)globalPoints[p*ctx->dim+0], (double)(ctx->dim > 1 ? globalPoints[p*ctx->dim+1] : 0.0), (double)(ctx->dim > 2 ? globalPoints[p*ctx->dim+2] : 0.0));
      else if (rank == 0) ++ctx->n;
    } else if (globalProcs[p] == rank) ++ctx->n;
  }
  /* Create coordinates vector and array of owned cells */
  CHKERRQ(PetscMalloc1(ctx->n, &ctx->cells));
  CHKERRQ(VecCreate(comm, &ctx->coords));
  CHKERRQ(VecSetSizes(ctx->coords, ctx->n*ctx->dim, PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(ctx->coords, ctx->dim));
  CHKERRQ(VecSetType(ctx->coords,VECSTANDARD));
  CHKERRQ(VecGetArray(ctx->coords, &a));
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
  CHKERRQ(VecRestoreArray(ctx->coords, &a));
#if 0
  CHKERRQ(PetscFree3(foundCells,foundProcs,globalProcs));
#else
  CHKERRQ(PetscFree2(foundProcs,globalProcs));
  CHKERRQ(PetscSFDestroy(&cellSF));
  CHKERRQ(VecDestroy(&pointVec));
#endif
  if ((void*)globalPointsScalar != (void*)globalPoints) CHKERRQ(PetscFree(globalPointsScalar));
  if (!redundantPoints) CHKERRQ(PetscFree3(globalPoints,counts,displs));
  CHKERRQ(PetscLayoutDestroy(&layout));
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
  PetscCheckFalse(!ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
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
  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  PetscCheckFalse(!ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  CHKERRQ(VecCreate(ctx->comm, v));
  CHKERRQ(VecSetSizes(*v, ctx->n*ctx->dof, PETSC_DECIDE));
  CHKERRQ(VecSetBlockSize(*v, ctx->dof));
  CHKERRQ(VecSetType(*v,VECSTANDARD));
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
  PetscFunctionBegin;
  PetscValidPointer(v, 2);
  PetscCheckFalse(!ctx->coords,ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  CHKERRQ(VecDestroy(v));
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
  CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
  CHKERRQ(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt     c = ctx->cells[p];
    PetscScalar *x = NULL;
    PetscReal    xir[1];
    PetscInt     xSize, comp;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, &v0, &J, &invJ, &detJ));
    PetscCheck(detJ > 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %" PetscInt_FMT, (double) detJ, c);
    xir[0] = invJ*PetscRealPart(coords[p] - v0);
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    if (2*dof == xSize) {
      for (comp = 0; comp < dof; ++comp) a[p*dof+comp] = x[0*dof+comp]*(1 - xir[0]) + x[1*dof+comp]*xir[0];
    } else if (dof == xSize) {
      for (comp = 0; comp < dof; ++comp) a[p*dof+comp] = x[0*dof+comp];
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input closure size %" PetscInt_FMT " must be either %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 2*dof, dof);
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  CHKERRQ(VecRestoreArray(v, &a));
  CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Triangle_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ));
  CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
  CHKERRQ(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt     c = ctx->cells[p];
    PetscScalar *x = NULL;
    PetscReal    xi[4];
    PetscInt     d, f, comp;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    PetscCheckFalse(detJ <= 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D", (double)detJ, c);
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[(d+1)*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  }
  CHKERRQ(VecRestoreArray(v, &a));
  CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));
  CHKERRQ(PetscFree3(v0, J, invJ));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode DMInterpolate_Tetrahedron_Private(DMInterpolationInfo ctx, DM dm, Vec xLocal, Vec v)
{
  PetscReal      *v0, *J, *invJ, detJ;
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       p;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc3(ctx->dim,&v0,ctx->dim*ctx->dim,&J,ctx->dim*ctx->dim,&invJ));
  CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
  CHKERRQ(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscInt       c = ctx->cells[p];
    const PetscInt order[3] = {2, 1, 3};
    PetscScalar   *x = NULL;
    PetscReal      xi[4];
    PetscInt       d, f, comp;

    CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
    PetscCheckFalse(detJ <= 0.0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %D", (double)detJ, c);
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, xLocal, c, NULL, &x));
    for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] = x[0*ctx->dof+comp];

    for (d = 0; d < ctx->dim; ++d) {
      xi[d] = 0.0;
      for (f = 0; f < ctx->dim; ++f) xi[d] += invJ[d*ctx->dim+f]*0.5*PetscRealPart(coords[p*ctx->dim+f] - v0[f]);
      for (comp = 0; comp < ctx->dof; ++comp) a[p*ctx->dof+comp] += PetscRealPart(x[order[d]*ctx->dof+comp] - x[0*ctx->dof+comp])*xi[d];
    }
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, NULL, &x));
  }
  CHKERRQ(VecRestoreArray(v, &a));
  CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));
  CHKERRQ(PetscFree3(v0, J, invJ));
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
  CHKERRQ(VecGetArrayRead(Xref,  &ref));
  CHKERRQ(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];

    real[0] = x0 + f_1 * p0 + f_3 * p1 + f_01 * p0 * p1;
    real[1] = y0 + g_1 * p0 + g_3 * p1 + g_01 * p0 * p1;
  }
  CHKERRQ(PetscLogFlops(28));
  CHKERRQ(VecRestoreArrayRead(Xref,  &ref));
  CHKERRQ(VecRestoreArray(Xreal, &real));
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
  CHKERRQ(VecGetArrayRead(Xref,  &ref));
  {
    const PetscScalar x       = ref[0];
    const PetscScalar y       = ref[1];
    const PetscInt    rows[2] = {0, 1};
    PetscScalar       values[4];

    values[0] = (x1 - x0 + f_01*y) * 0.5; values[1] = (x3 - x0 + f_01*x) * 0.5;
    values[2] = (y1 - y0 + g_01*y) * 0.5; values[3] = (y3 - y0 + g_01*x) * 0.5;
    CHKERRQ(MatSetValues(J, 2, rows, 2, rows, values, INSERT_VALUES));
  }
  CHKERRQ(PetscLogFlops(30));
  CHKERRQ(VecRestoreArrayRead(Xref,  &ref));
  CHKERRQ(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
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
  CHKERRQ(DMGetNumFields(dm, &Nf));
  if (Nf) {
    PetscObject  obj;
    PetscClassId id;

    CHKERRQ(DMGetField(dm, 0, NULL, &obj));
    CHKERRQ(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {fem = (PetscFE) obj; CHKERRQ(PetscFECreateTabulation(fem, 1, 1, xir, 0, &T));}
  }
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordsLocal));
  CHKERRQ(DMGetCoordinateDM(dm, &dmCoord));
  CHKERRQ(SNESCreate(PETSC_COMM_SELF, &snes));
  CHKERRQ(SNESSetOptionsPrefix(snes, "quad_interp_"));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &r));
  CHKERRQ(VecSetSizes(r, 2, 2));
  CHKERRQ(VecSetType(r,dm->vectype));
  CHKERRQ(VecDuplicate(r, &ref));
  CHKERRQ(VecDuplicate(r, &real));
  CHKERRQ(MatCreate(PETSC_COMM_SELF, &J));
  CHKERRQ(MatSetSizes(J, 2, 2, 2, 2));
  CHKERRQ(MatSetType(J, MATSEQDENSE));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(SNESSetFunction(snes, r, QuadMap_Private, NULL));
  CHKERRQ(SNESSetJacobian(snes, J, J, QuadJacobian_Private, NULL));
  CHKERRQ(SNESGetKSP(snes, &ksp));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc, PCLU));
  CHKERRQ(SNESSetFromOptions(snes));

  CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
  CHKERRQ(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    CHKERRQ(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCheckFalse(4*2 != coordSize,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %D should be %d", coordSize, 4*2);
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    CHKERRQ(SNESSetFunction(snes, NULL, NULL, vertices));
    CHKERRQ(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
    CHKERRQ(VecGetArray(real, &xi));
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    CHKERRQ(VecRestoreArray(real, &xi));
    CHKERRQ(SNESSolve(snes, real, ref));
    CHKERRQ(VecGetArray(ref, &xi));
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
      CHKERRQ(PetscFEComputeTabulation(fem, 1, xir, 0, T));
      for (comp = 0; comp < dof; ++comp) {
        a[p*dof+comp] = 0.0;
        for (d = 0; d < xSize/dof; ++d) {
          a[p*dof+comp] += x[d*dof+comp]*T->T[0][d*dof+comp];
        }
      }
    }
    CHKERRQ(VecRestoreArray(ref, &xi));
    CHKERRQ(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  CHKERRQ(PetscTabulationDestroy(&T));
  CHKERRQ(VecRestoreArray(v, &a));
  CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));

  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&ref));
  CHKERRQ(VecDestroy(&real));
  CHKERRQ(MatDestroy(&J));
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
  CHKERRQ(VecGetArrayRead(Xref,  &ref));
  CHKERRQ(VecGetArray(Xreal, &real));
  {
    const PetscScalar p0 = ref[0];
    const PetscScalar p1 = ref[1];
    const PetscScalar p2 = ref[2];

    real[0] = x0 + f_1*p0 + f_3*p1 + f_4*p2 + f_01*p0*p1 + f_12*p1*p2 + f_02*p0*p2 + f_012*p0*p1*p2;
    real[1] = y0 + g_1*p0 + g_3*p1 + g_4*p2 + g_01*p0*p1 + g_01*p0*p1 + g_12*p1*p2 + g_02*p0*p2 + g_012*p0*p1*p2;
    real[2] = z0 + h_1*p0 + h_3*p1 + h_4*p2 + h_01*p0*p1 + h_01*p0*p1 + h_12*p1*p2 + h_02*p0*p2 + h_012*p0*p1*p2;
  }
  CHKERRQ(PetscLogFlops(114));
  CHKERRQ(VecRestoreArrayRead(Xref,  &ref));
  CHKERRQ(VecRestoreArray(Xreal, &real));
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
  CHKERRQ(VecGetArrayRead(Xref,  &ref));
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

    CHKERRQ(MatSetValues(J, 3, rows, 3, rows, values, INSERT_VALUES));
  }
  CHKERRQ(PetscLogFlops(152));
  CHKERRQ(VecRestoreArrayRead(Xref,  &ref));
  CHKERRQ(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
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
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordsLocal));
  CHKERRQ(DMGetCoordinateDM(dm, &dmCoord));
  CHKERRQ(SNESCreate(PETSC_COMM_SELF, &snes));
  CHKERRQ(SNESSetOptionsPrefix(snes, "hex_interp_"));
  CHKERRQ(VecCreate(PETSC_COMM_SELF, &r));
  CHKERRQ(VecSetSizes(r, 3, 3));
  CHKERRQ(VecSetType(r,dm->vectype));
  CHKERRQ(VecDuplicate(r, &ref));
  CHKERRQ(VecDuplicate(r, &real));
  CHKERRQ(MatCreate(PETSC_COMM_SELF, &J));
  CHKERRQ(MatSetSizes(J, 3, 3, 3, 3));
  CHKERRQ(MatSetType(J, MATSEQDENSE));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(SNESSetFunction(snes, r, HexMap_Private, NULL));
  CHKERRQ(SNESSetJacobian(snes, J, J, HexJacobian_Private, NULL));
  CHKERRQ(SNESGetKSP(snes, &ksp));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PCSetType(pc, PCLU));
  CHKERRQ(SNESSetFromOptions(snes));

  CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
  CHKERRQ(VecGetArray(v, &a));
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscReal    xir[3];
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    CHKERRQ(DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    PetscCheck(8*3 == coordSize,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid coordinate closure size %" PetscInt_FMT " should be %d", coordSize, 8*3);
    CHKERRQ(DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x));
    PetscCheck((8*ctx->dof == xSize) || (ctx->dof == xSize),ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input closure size %" PetscInt_FMT " should be %" PetscInt_FMT " or %" PetscInt_FMT, xSize, 8*ctx->dof, ctx->dof);
    CHKERRQ(SNESSetFunction(snes, NULL, NULL, vertices));
    CHKERRQ(SNESSetJacobian(snes, NULL, NULL, NULL, vertices));
    CHKERRQ(VecGetArray(real, &xi));
    xi[0]  = coords[p*ctx->dim+0];
    xi[1]  = coords[p*ctx->dim+1];
    xi[2]  = coords[p*ctx->dim+2];
    CHKERRQ(VecRestoreArray(real, &xi));
    CHKERRQ(SNESSolve(snes, real, ref));
    CHKERRQ(VecGetArray(ref, &xi));
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
    CHKERRQ(VecRestoreArray(ref, &xi));
    CHKERRQ(DMPlexVecRestoreClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices));
    CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, xLocal, c, &xSize, &x));
  }
  CHKERRQ(VecRestoreArray(v, &a));
  CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));

  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&ref));
  CHKERRQ(VecDestroy(&real));
  CHKERRQ(MatDestroy(&J));
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
  PetscDS        ds;
  PetscInt       n, p, Nf, field;
  PetscBool      useDS = PETSC_FALSE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  CHKERRQ(VecGetLocalSize(v, &n));
  PetscCheckFalse(n != ctx->n*ctx->dof,ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %D should be %D", n, ctx->n*ctx->dof);
  if (!n) PetscFunctionReturn(0);
  CHKERRQ(DMGetDS(dm, &ds));
  if (ds) {
    useDS = PETSC_TRUE;
    CHKERRQ(PetscDSGetNumFields(ds, &Nf));
    for (field = 0; field < Nf; ++field) {
      PetscObject  obj;
      PetscClassId id;

      CHKERRQ(PetscDSGetDiscretization(ds, field, &obj));
      CHKERRQ(PetscObjectGetClassId(obj, &id));
      if (id != PETSCFE_CLASSID) {useDS = PETSC_FALSE; break;}
    }
  }
  if (useDS) {
    const PetscScalar *coords;
    PetscScalar       *interpolant;
    PetscInt           cdim, d;

    CHKERRQ(DMGetCoordinateDim(dm, &cdim));
    CHKERRQ(VecGetArrayRead(ctx->coords, &coords));
    CHKERRQ(VecGetArrayWrite(v, &interpolant));
    for (p = 0; p < ctx->n; ++p) {
      PetscReal    pcoords[3], xi[3];
      PetscScalar *xa   = NULL;
      PetscInt     coff = 0, foff = 0, clSize;

      if (ctx->cells[p] < 0) continue;
      for (d = 0; d < cdim; ++d) pcoords[d] = PetscRealPart(coords[p*cdim+d]);
      CHKERRQ(DMPlexCoordinatesToReference(dm, ctx->cells[p], 1, pcoords, xi));
      CHKERRQ(DMPlexVecGetClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      for (field = 0; field < Nf; ++field) {
        PetscTabulation T;
        PetscFE         fe;

        CHKERRQ(PetscDSGetDiscretization(ds, field, (PetscObject *) &fe));
        CHKERRQ(PetscFECreateTabulation(fe, 1, 1, xi, 0, &T));
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
        CHKERRQ(PetscTabulationDestroy(&T));
      }
      CHKERRQ(DMPlexVecRestoreClosure(dm, NULL, x, ctx->cells[p], &clSize, &xa));
      PetscCheckFalse(coff != ctx->dof,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total components %D != %D dof specified for interpolation", coff, ctx->dof);
      PetscCheckFalse(foff != clSize,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Total FE space size %D != %D closure size", foff, clSize);
    }
    CHKERRQ(VecRestoreArrayRead(ctx->coords, &coords));
    CHKERRQ(VecRestoreArrayWrite(v, &interpolant));
  } else {
    DMPolytopeType ct;

    /* TODO Check each cell individually */
    CHKERRQ(DMPlexGetCellType(dm, ctx->cells[0], &ct));
    switch (ct) {
      case DM_POLYTOPE_SEGMENT:       CHKERRQ(DMInterpolate_Segment_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_TRIANGLE:      CHKERRQ(DMInterpolate_Triangle_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_QUADRILATERAL: CHKERRQ(DMInterpolate_Quad_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_TETRAHEDRON:   CHKERRQ(DMInterpolate_Tetrahedron_Private(ctx, dm, x, v));break;
      case DM_POLYTOPE_HEXAHEDRON:    CHKERRQ(DMInterpolate_Hex_Private(ctx, dm, x, v));break;
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

.seealso: DMInterpolationEvaluate(), DMInterpolationAddPoints(), DMInterpolationCreate()
@*/
PetscErrorCode DMInterpolationDestroy(DMInterpolationInfo *ctx)
{
  PetscFunctionBegin;
  PetscValidPointer(ctx, 1);
  CHKERRQ(VecDestroy(&(*ctx)->coords));
  CHKERRQ(PetscFree((*ctx)->points));
  CHKERRQ(PetscFree((*ctx)->cells));
  CHKERRQ(PetscFree(*ctx));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,4);
  CHKERRQ(SNESGetFunction(snes, &res, NULL, NULL));
  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(DMGetLocalSection(dm, &s));
  CHKERRQ(PetscSectionGetNumFields(s, &numFields));
  CHKERRQ(PetscSectionGetChart(s, &pStart, &pEnd));
  CHKERRQ(PetscCalloc2(numFields, &lnorms, numFields, &norms));
  CHKERRQ(VecGetArrayRead(res, &r));
  for (p = pStart; p < pEnd; ++p) {
    for (f = 0; f < numFields; ++f) {
      PetscInt fdof, foff, d;

      CHKERRQ(PetscSectionGetFieldDof(s, p, f, &fdof));
      CHKERRQ(PetscSectionGetFieldOffset(s, p, f, &foff));
      for (d = 0; d < fdof; ++d) lnorms[f] += PetscRealPart(PetscSqr(r[foff+d]));
    }
  }
  CHKERRQ(VecRestoreArrayRead(res, &r));
  CHKERRMPI(MPIU_Allreduce(lnorms, norms, numFields, MPIU_REAL, MPIU_SUM, PetscObjectComm((PetscObject) dm)));
  CHKERRQ(PetscViewerPushFormat(viewer,vf->format));
  CHKERRQ(PetscViewerASCIIAddTab(viewer, ((PetscObject) snes)->tablevel));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "%3D SNES Function norm %14.12e [", its, (double) fgnorm));
  for (f = 0; f < numFields; ++f) {
    if (f > 0) CHKERRQ(PetscViewerASCIIPrintf(viewer, ", "));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "%14.12e", (double) PetscSqrtReal(norms[f])));
  }
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "]\n"));
  CHKERRQ(PetscViewerASCIISubtractTab(viewer, ((PetscObject) snes)->tablevel));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscFree2(lnorms, norms));
  PetscFunctionReturn(0);
}

/********************* Residual Computation **************************/

PetscErrorCode DMPlexGetAllCells_Internal(DM plex, IS *cellIS)
{
  PetscInt       depth;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetDepth(plex, &depth));
  CHKERRQ(DMGetStratumIS(plex, "dim", depth, cellIS));
  if (!*cellIS) CHKERRQ(DMGetStratumIS(plex, "depth", depth, cellIS));
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

.seealso: DMPlexComputeJacobianAction()
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  CHKERRQ(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  CHKERRQ(DMPlexGetAllCells_Internal(plex, &allcellIS));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    CHKERRQ(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      CHKERRQ(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      CHKERRQ(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      CHKERRQ(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      CHKERRQ(ISDestroy(&pointIS));
    }
    CHKERRQ(DMPlexComputeResidual_Internal(plex, key, cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
    CHKERRQ(ISDestroy(&cellIS));
  }
  CHKERRQ(ISDestroy(&allcellIS));
  CHKERRQ(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESComputeResidual(DM dm, Vec X, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  CHKERRQ(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  CHKERRQ(DMPlexGetAllCells_Internal(plex, &allcellIS));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    CHKERRQ(DMGetRegionNumDS(dm, s, &label, NULL, &ds));
    {
      PetscWeakFormKind resmap[2] = {PETSC_WF_F0, PETSC_WF_F1};
      PetscWeakForm     wf;
      PetscInt          Nm = 2, m, Nk = 0, k, kp, off = 0;
      PetscFormKey *reskeys;

      /* Get unique residual keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        CHKERRQ(PetscHMapFormGetSize(ds->wf->form[resmap[m]], &Nkm));
        Nk  += Nkm;
      }
      CHKERRQ(PetscMalloc1(Nk, &reskeys));
      for (m = 0; m < Nm; ++m) {
        CHKERRQ(PetscHMapFormGetKeys(ds->wf->form[resmap[m]], &off, reskeys));
      }
      PetscCheckFalse(off != Nk,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %D should be %D", off, Nk);
      CHKERRQ(PetscFormKeySort(Nk, reskeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((reskeys[k].label != reskeys[kp].label) || (reskeys[k].value != reskeys[kp].value)) {
          ++k;
          if (kp != k) reskeys[k] = reskeys[kp];
        }
      }
      Nk = k;

      CHKERRQ(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = reskeys[k].label;
        PetscInt val   = reskeys[k].value;

        if (!label) {
          CHKERRQ(PetscObjectReference((PetscObject) allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          CHKERRQ(DMLabelGetStratumIS(label, val, &pointIS));
          CHKERRQ(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          CHKERRQ(ISDestroy(&pointIS));
        }
        CHKERRQ(DMPlexComputeResidual_Internal(plex, reskeys[k], cellIS, PETSC_MIN_REAL, X, NULL, 0.0, F, user));
        CHKERRQ(ISDestroy(&cellIS));
      }
      CHKERRQ(PetscFree(reskeys));
    }
  }
  CHKERRQ(ISDestroy(&allcellIS));
  CHKERRQ(DMDestroy(&plex));
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

  PetscFunctionBegin;
  CHKERRQ(DMSNESConvertPlex(dm,&plex,PETSC_TRUE));
  CHKERRQ(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, X, PETSC_MIN_REAL, NULL, NULL, NULL));
  CHKERRQ(DMDestroy(&plex));
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

.seealso: DMSNESCreateJacobianMF(), DMPlexSNESComputeResidualFEM()
@*/
PetscErrorCode DMSNESComputeJacobianAction(DM dm, Vec X, Vec Y, Vec F, void *user)
{
  DM             plex;
  IS             allcellIS;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  CHKERRQ(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  CHKERRQ(DMPlexGetAllCells_Internal(plex, &allcellIS));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS ds;
    DMLabel label;
    IS      cellIS;

    CHKERRQ(DMGetRegionNumDS(dm, s, &label, NULL, &ds));
    {
      PetscWeakFormKind jacmap[4] = {PETSC_WF_G0, PETSC_WF_G1, PETSC_WF_G2, PETSC_WF_G3};
      PetscWeakForm     wf;
      PetscInt          Nm = 4, m, Nk = 0, k, kp, off = 0;
      PetscFormKey *jackeys;

      /* Get unique Jacobian keys */
      for (m = 0; m < Nm; ++m) {
        PetscInt Nkm;
        CHKERRQ(PetscHMapFormGetSize(ds->wf->form[jacmap[m]], &Nkm));
        Nk  += Nkm;
      }
      CHKERRQ(PetscMalloc1(Nk, &jackeys));
      for (m = 0; m < Nm; ++m) {
        CHKERRQ(PetscHMapFormGetKeys(ds->wf->form[jacmap[m]], &off, jackeys));
      }
      PetscCheckFalse(off != Nk,PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Number of keys %D should be %D", off, Nk);
      CHKERRQ(PetscFormKeySort(Nk, jackeys));
      for (k = 0, kp = 1; kp < Nk; ++kp) {
        if ((jackeys[k].label != jackeys[kp].label) || (jackeys[k].value != jackeys[kp].value)) {
          ++k;
          if (kp != k) jackeys[k] = jackeys[kp];
        }
      }
      Nk = k;

      CHKERRQ(PetscDSGetWeakForm(ds, &wf));
      for (k = 0; k < Nk; ++k) {
        DMLabel  label = jackeys[k].label;
        PetscInt val   = jackeys[k].value;

        if (!label) {
          CHKERRQ(PetscObjectReference((PetscObject) allcellIS));
          cellIS = allcellIS;
        } else {
          IS pointIS;

          CHKERRQ(DMLabelGetStratumIS(label, val, &pointIS));
          CHKERRQ(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
          CHKERRQ(ISDestroy(&pointIS));
        }
        CHKERRQ(DMPlexComputeJacobian_Action_Internal(plex, jackeys[k], cellIS, 0.0, 0.0, X, NULL, Y, F, user));
        CHKERRQ(ISDestroy(&cellIS));
      }
      CHKERRQ(PetscFree(jackeys));
    }
  }
  CHKERRQ(ISDestroy(&allcellIS));
  CHKERRQ(DMDestroy(&plex));
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
  IS             allcellIS;
  PetscBool      hasJac, hasPrec;
  PetscInt       Nds, s;

  PetscFunctionBegin;
  CHKERRQ(DMSNESConvertPlex(dm, &plex, PETSC_TRUE));
  CHKERRQ(DMPlexGetAllCells_Internal(plex, &allcellIS));
  CHKERRQ(DMGetNumDS(dm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS          ds;
    IS               cellIS;
    PetscFormKey key;

    CHKERRQ(DMGetRegionNumDS(dm, s, &key.label, NULL, &ds));
    key.value = 0;
    key.field = 0;
    key.part  = 0;
    if (!key.label) {
      CHKERRQ(PetscObjectReference((PetscObject) allcellIS));
      cellIS = allcellIS;
    } else {
      IS pointIS;

      key.value = 1;
      CHKERRQ(DMLabelGetStratumIS(key.label, key.value, &pointIS));
      CHKERRQ(ISIntersect_Caching_Internal(allcellIS, pointIS, &cellIS));
      CHKERRQ(ISDestroy(&pointIS));
    }
    if (!s) {
      CHKERRQ(PetscDSHasJacobian(ds, &hasJac));
      CHKERRQ(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
      if (hasJac && hasPrec) CHKERRQ(MatZeroEntries(Jac));
      CHKERRQ(MatZeroEntries(JacP));
    }
    CHKERRQ(DMPlexComputeJacobian_Internal(plex, key, cellIS, 0.0, 0.0, X, NULL, Jac, JacP, user));
    CHKERRQ(ISDestroy(&cellIS));
  }
  CHKERRQ(ISDestroy(&allcellIS));
  CHKERRQ(DMDestroy(&plex));
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
  CHKERRQ(MatShellGetContext(A, &ctx));
  CHKERRQ(MatShellSetContext(A, NULL));
  CHKERRQ(DMDestroy(&ctx->dm));
  CHKERRQ(VecDestroy(&ctx->X));
  CHKERRQ(PetscFree(ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSNESJacobianMF_Mult_Private(Mat A, Vec Y, Vec Z)
{
  struct _DMSNESJacobianMFCtx *ctx;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(A, &ctx));
  CHKERRQ(DMSNESComputeJacobianAction(ctx->dm, ctx->X, Y, Z, ctx->ctx));
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

.seealso: DMSNESComputeJacobianAction()
@*/
PetscErrorCode DMSNESCreateJacobianMF(DM dm, Vec X, void *user, Mat *J)
{
  struct _DMSNESJacobianMFCtx *ctx;
  PetscInt                     n, N;

  PetscFunctionBegin;
  CHKERRQ(MatCreate(PetscObjectComm((PetscObject) dm), J));
  CHKERRQ(MatSetType(*J, MATSHELL));
  CHKERRQ(VecGetLocalSize(X, &n));
  CHKERRQ(VecGetSize(X, &N));
  CHKERRQ(MatSetSizes(*J, n, n, N, N));
  CHKERRQ(PetscObjectReference((PetscObject) dm));
  CHKERRQ(PetscObjectReference((PetscObject) X));
  CHKERRQ(PetscMalloc1(1, &ctx));
  ctx->dm  = dm;
  ctx->X   = X;
  ctx->ctx = user;
  CHKERRQ(MatShellSetContext(*J, ctx));
  CHKERRQ(MatShellSetOperation(*J, MATOP_DESTROY, (void (*)(void)) DMSNESJacobianMF_Destroy_Private));
  CHKERRQ(MatShellSetOperation(*J, MATOP_MULT,    (void (*)(void)) DMSNESJacobianMF_Mult_Private));
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

  PetscFunctionBegin;
  CHKERRQ(PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_MATIS",(PetscObject*)&pJ));
  PetscCheckFalse(!pJ,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing overlapping Mat");
  CHKERRQ(PetscObjectQuery((PetscObject)ovl,"_DM_Original_HPDDM",(PetscObject*)&origdm));
  PetscCheckFalse(!origdm,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing original DM");
  CHKERRQ(MatGetDM(pJ,&ovldm));
  CHKERRQ(DMSNESGetBoundaryLocal(origdm,&bfun,&bctx));
  CHKERRQ(DMSNESSetBoundaryLocal(ovldm,bfun,bctx));
  CHKERRQ(DMSNESGetJacobianLocal(origdm,&jfun,&jctx));
  CHKERRQ(DMSNESSetJacobianLocal(ovldm,jfun,jctx));
  CHKERRQ(PetscObjectQuery((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject*)&snes));
  if (!snes) {
    CHKERRQ(SNESCreate(PetscObjectComm((PetscObject)ovl),&snes));
    CHKERRQ(SNESSetDM(snes,ovldm));
    CHKERRQ(PetscObjectCompose((PetscObject)ovl,"_DM_Overlap_HPDDM_SNES",(PetscObject)snes));
    CHKERRQ(PetscObjectDereference((PetscObject)snes));
  }
  CHKERRQ(DMGetDMSNES(ovldm,&sdm));
  CHKERRQ(VecLockReadPush(X));
  PetscStackPush("SNES user Jacobian function");
  CHKERRQ((*sdm->ops->computejacobian)(snes,X,pJ,pJ,sdm->jacobianctx));
  PetscStackPop;
  CHKERRQ(VecLockReadPop(X));
  /* this is a no-hop, just in case we decide to change the placeholder for the local Neumann matrix */
  {
    Mat locpJ;

    CHKERRQ(MatISGetLocalMat(pJ,&locpJ));
    CHKERRQ(MatCopy(locpJ,J,SAME_NONZERO_PATTERN));
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
  CHKERRQ(DMSNESSetBoundaryLocal(dm,DMPlexSNESComputeBoundaryFEM,boundaryctx));
  CHKERRQ(DMSNESSetFunctionLocal(dm,DMPlexSNESComputeResidualFEM,residualctx));
  CHKERRQ(DMSNESSetJacobianLocal(dm,DMPlexSNESComputeJacobianFEM,jacobianctx));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)dm,"MatComputeNeumannOverlap_C",MatComputeNeumannOverlap_Plex));
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

.seealso: DNSNESCheckFromOptions(), DMSNESCheckResidual(), DMSNESCheckJacobian(), PetscDSSetExactSolution()
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

  CHKERRQ(DMComputeExactSolution(dm, t, u, NULL));
  CHKERRQ(VecViewFromOptions(u, NULL, "-vec_view"));

  CHKERRQ(PetscObjectGetComm((PetscObject) snes, &comm));
  CHKERRQ(DMGetNumFields(dm, &Nf));
  CHKERRQ(PetscCalloc3(Nf, &exacts, Nf, &ectxs, PetscMax(1, Nf), &err));
  {
    PetscInt Nds, s;

    CHKERRQ(DMGetNumDS(dm, &Nds));
    for (s = 0; s < Nds; ++s) {
      PetscDS         ds;
      DMLabel         label;
      IS              fieldIS;
      const PetscInt *fields;
      PetscInt        dsNf, f;

      CHKERRQ(DMGetRegionNumDS(dm, s, &label, &fieldIS, &ds));
      CHKERRQ(PetscDSGetNumFields(ds, &dsNf));
      CHKERRQ(ISGetIndices(fieldIS, &fields));
      for (f = 0; f < dsNf; ++f) {
        const PetscInt field = fields[f];
        CHKERRQ(PetscDSGetExactSolution(ds, field, &exacts[field], &ectxs[field]));
      }
      CHKERRQ(ISRestoreIndices(fieldIS, &fields));
    }
  }
  if (Nf > 1) {
    CHKERRQ(DMComputeL2FieldDiff(dm, t, exacts, ectxs, u, err));
    if (tol >= 0.0) {
      for (f = 0; f < Nf; ++f) {
        PetscCheckFalse(err[f] > tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g for field %D exceeds tolerance %g", (double) err[f], f, (double) tol);
      }
    } else if (error) {
      for (f = 0; f < Nf; ++f) error[f] = err[f];
    } else {
      CHKERRQ(PetscPrintf(comm, "L_2 Error: ["));
      for (f = 0; f < Nf; ++f) {
        if (f) CHKERRQ(PetscPrintf(comm, ", "));
        CHKERRQ(PetscPrintf(comm, "%g", (double)err[f]));
      }
      CHKERRQ(PetscPrintf(comm, "]\n"));
    }
  } else {
    CHKERRQ(DMComputeL2Diff(dm, t, exacts, ectxs, u, &err[0]));
    if (tol >= 0.0) {
      PetscCheckFalse(err[0] > tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Error %g exceeds tolerance %g", (double) err[0], (double) tol);
    } else if (error) {
      error[0] = err[0];
    } else {
      CHKERRQ(PetscPrintf(comm, "L_2 Error: %g\n", (double) err[0]));
    }
  }
  CHKERRQ(PetscFree3(exacts, ectxs, err));
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
  MPI_Comm       comm;
  Vec            r;
  PetscReal      res;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 3);
  if (residual) PetscValidRealPointer(residual, 5);
  CHKERRQ(PetscObjectGetComm((PetscObject) snes, &comm));
  CHKERRQ(DMComputeExactSolution(dm, 0.0, u, NULL));
  CHKERRQ(VecDuplicate(u, &r));
  CHKERRQ(SNESComputeFunction(snes, u, r));
  CHKERRQ(VecNorm(r, NORM_2, &res));
  if (tol >= 0.0) {
    PetscCheckFalse(res > tol,comm, PETSC_ERR_ARG_WRONG, "L_2 Residual %g exceeds tolerance %g", (double) res, (double) tol);
  } else if (residual) {
    *residual = res;
  } else {
    CHKERRQ(PetscPrintf(comm, "L_2 Residual: %g\n", (double)res));
    CHKERRQ(VecChop(r, 1.0e-10));
    CHKERRQ(PetscObjectSetName((PetscObject) r, "Initial Residual"));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)r,"res_"));
    CHKERRQ(VecViewFromOptions(r, NULL, "-vec_view"));
  }
  CHKERRQ(VecDestroy(&r));
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
  CHKERRQ(PetscObjectGetComm((PetscObject) snes, &comm));
  CHKERRQ(DMComputeExactSolution(dm, 0.0, u, NULL));
  /* Create and view matrices */
  CHKERRQ(DMCreateMatrix(dm, &J));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscDSHasJacobian(ds, &hasJac));
  CHKERRQ(PetscDSHasJacobianPreconditioner(ds, &hasPrec));
  if (hasJac && hasPrec) {
    CHKERRQ(DMCreateMatrix(dm, &M));
    CHKERRQ(SNESComputeJacobian(snes, u, J, M));
    CHKERRQ(PetscObjectSetName((PetscObject) M, "Preconditioning Matrix"));
    CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_"));
    CHKERRQ(MatViewFromOptions(M, NULL, "-mat_view"));
    CHKERRQ(MatDestroy(&M));
  } else {
    CHKERRQ(SNESComputeJacobian(snes, u, J, J));
  }
  CHKERRQ(PetscObjectSetName((PetscObject) J, "Jacobian"));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) J, "jac_"));
  CHKERRQ(MatViewFromOptions(J, NULL, "-mat_view"));
  /* Check nullspace */
  CHKERRQ(MatGetNullSpace(J, &nullspace));
  if (nullspace) {
    PetscBool isNull;
    CHKERRQ(MatNullSpaceTest(nullspace, J, &isNull));
    PetscCheckFalse(!isNull,comm, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
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
    CHKERRQ(PetscRandomCreate(comm, &rand));
    CHKERRQ(VecDuplicate(u, &du));
    CHKERRQ(VecSetRandom(du, rand));
    CHKERRQ(PetscRandomDestroy(&rand));
    CHKERRQ(VecDuplicate(u, &df));
    CHKERRQ(MatMult(J, du, df));
    /* Evaluate residual at u, F(u), save in vector r */
    CHKERRQ(VecDuplicate(u, &r));
    CHKERRQ(SNESComputeFunction(snes, u, r));
    /* Look at the convergence of our Taylor approximation as we approach u */
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv);
    CHKERRQ(PetscCalloc3(Nv, &es, Nv, &hs, Nv, &errors));
    CHKERRQ(VecDuplicate(u, &uhat));
    CHKERRQ(VecDuplicate(u, &rhat));
    for (h = hMax, Nv = 0; h >= hMin; h *= hMult, ++Nv) {
      CHKERRQ(VecWAXPY(uhat, h, du, u));
      /* F(\hat u) \approx F(u) + J(u) (uhat - u) = F(u) + h * J(u) du */
      CHKERRQ(SNESComputeFunction(snes, uhat, rhat));
      CHKERRQ(VecAXPBYPCZ(rhat, -1.0, -h, 1.0, r, df));
      CHKERRQ(VecNorm(rhat, NORM_2, &errors[Nv]));

      es[Nv] = PetscLog10Real(errors[Nv]);
      hs[Nv] = PetscLog10Real(h);
    }
    CHKERRQ(VecDestroy(&uhat));
    CHKERRQ(VecDestroy(&rhat));
    CHKERRQ(VecDestroy(&df));
    CHKERRQ(VecDestroy(&r));
    CHKERRQ(VecDestroy(&du));
    for (v = 0; v < Nv; ++v) {
      if ((tol >= 0) && (errors[v] > tol)) break;
      else if (errors[v] > PETSC_SMALL)    break;
    }
    if (v == Nv) isLin = PETSC_TRUE;
    CHKERRQ(PetscLinearRegression(Nv, hs, es, &slope, &intercept));
    CHKERRQ(PetscFree3(es, hs, errors));
    /* Slope should be about 2 */
    if (tol >= 0) {
      PetscCheckFalse(!isLin && PetscAbsReal(2 - slope) > tol,comm, PETSC_ERR_ARG_WRONG, "Taylor approximation convergence rate should be 2, not %0.2f", (double) slope);
    } else if (isLinear || convRate) {
      if (isLinear) *isLinear = isLin;
      if (convRate) *convRate = slope;
    } else {
      if (!isLin) CHKERRQ(PetscPrintf(comm, "Taylor approximation converging at order %3.2f\n", (double) slope));
      else        CHKERRQ(PetscPrintf(comm, "Function appears to be linear\n"));
    }
  }
  CHKERRQ(MatDestroy(&J));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESCheck_Internal(SNES snes, DM dm, Vec u)
{
  PetscFunctionBegin;
  CHKERRQ(DMSNESCheckDiscretization(snes, dm, 0.0, u, -1.0, NULL));
  CHKERRQ(DMSNESCheckResidual(snes, dm, u, -1.0, NULL));
  CHKERRQ(DMSNESCheckJacobian(snes, dm, u, -1.0, NULL, NULL));
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
  CHKERRQ(PetscOptionsHasName(((PetscObject)snes)->options,((PetscObject)snes)->prefix, "-dmsnes_check", &check));
  if (!check) PetscFunctionReturn(0);
  CHKERRQ(SNESGetDM(snes, &dm));
  CHKERRQ(VecDuplicate(u, &sol));
  CHKERRQ(SNESSetSolution(snes, sol));
  CHKERRQ(DMSNESCheck_Internal(snes, dm, sol));
  CHKERRQ(VecDestroy(&sol));
  PetscFunctionReturn(0);
}
