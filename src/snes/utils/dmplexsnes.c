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

PetscErrorCode DMInterpolationSetDim(DMInterpolationInfo ctx, PetscInt dim)
{
  PetscFunctionBegin;
  if ((dim < 1) || (dim > 3)) SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid dimension for points: %d", dim);
  ctx->dim = dim;
  PetscFunctionReturn(0);
}

PetscErrorCode DMInterpolationGetDim(DMInterpolationInfo ctx, PetscInt *dim)
{
  PetscFunctionBegin;
  PetscValidIntPointer(dim, 2);
  *dim = ctx->dim;
  PetscFunctionReturn(0);
}

PetscErrorCode DMInterpolationSetDof(DMInterpolationInfo ctx, PetscInt dof)
{
  PetscFunctionBegin;
  if (dof < 1) SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid number of components: %d", dof);
  ctx->dof = dof;
  PetscFunctionReturn(0);
}

PetscErrorCode DMInterpolationGetDof(DMInterpolationInfo ctx, PetscInt *dof)
{
  PetscFunctionBegin;
  PetscValidIntPointer(dof, 2);
  *dof = ctx->dof;
  PetscFunctionReturn(0);
}

PetscErrorCode DMInterpolationAddPoints(DMInterpolationInfo ctx, PetscInt n, PetscReal points[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (ctx->dim < 0) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The spatial dimension has not been set");
  if (ctx->points)  SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "Cannot add points multiple times yet");
  ctx->nInput = n;

  ierr = PetscMalloc1(n*ctx->dim, &ctx->points);CHKERRQ(ierr);
  ierr = PetscMemcpy(ctx->points, points, n*ctx->dim * sizeof(PetscReal));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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

PetscErrorCode DMInterpolationGetCoordinates(DMInterpolationInfo ctx, Vec *coordinates)
{
  PetscFunctionBegin;
  PetscValidPointer(coordinates, 2);
  if (!ctx->coords) SETERRQ(ctx->comm, PETSC_ERR_ARG_WRONGSTATE, "The interpolation context has not been setup.");
  *coordinates = ctx->coords;
  PetscFunctionReturn(0);
}

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
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", (double)detJ, c);
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
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for element %d", (double)detJ, c);
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
  const PetscScalar *coords;
  PetscScalar    *a;
  PetscInt       Nf, p;
  const PetscInt dof = ctx->dof;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
  if (Nf) {ierr = DMGetField(dm, 0, (PetscObject *) &fem);CHKERRQ(ierr);}
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
  for (p = 0; p < ctx->n; ++p) {
    PetscScalar *x = NULL, *vertices = NULL;
    PetscScalar *xi;
    PetscReal    xir[2];
    PetscInt     c = ctx->cells[p], comp, coordSize, xSize;

    /* Can make this do all points at once */
    ierr = DMPlexVecGetClosure(dmCoord, NULL, coordsLocal, c, &coordSize, &vertices);CHKERRQ(ierr);
    if (4*2 != coordSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %d should be %d", coordSize, 4*2);
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
      PetscReal *B;
      PetscInt   d;

      xir[0] = 2.0*xir[0] - 1.0; xir[1] = 2.0*xir[1] - 1.0;
      ierr = PetscFEGetTabulation(fem, 1, xir, &B, NULL, NULL);CHKERRQ(ierr);
      for (comp = 0; comp < dof; ++comp) {
        a[p*dof+comp] = 0.0;
        for (d = 0; d < xSize/dof; ++d) {
          a[p*dof+comp] += x[d*dof+comp]*B[d*dof+comp];
        }
      }
      ierr = PetscFERestoreTabulation(fem, 1, xir, &B, NULL, NULL);CHKERRQ(ierr);
    } else {
      for (comp = 0; comp < dof; ++comp)
        a[p*dof+comp] = x[0*dof+comp]*(1 - xir[0])*(1 - xir[1]) + x[1*dof+comp]*xir[0]*(1 - xir[1]) + x[2*dof+comp]*xir[0]*xir[1] + x[3*dof+comp]*(1 - xir[0])*xir[1];
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
    if (8*3 != coordSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %d should be %d", coordSize, 8*3);
    ierr = DMPlexVecGetClosure(dm, NULL, xLocal, c, &xSize, &x);CHKERRQ(ierr);
    if (8*ctx->dof != xSize) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid closure size %d should be %d", xSize, 8*ctx->dof);
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

/*
  Input Parameters:
+ ctx - The DMInterpolationInfo context
. dm  - The DM
- x   - The local vector containing the field to be interpolated

  Output Parameters:
. v   - The vector containing the interpolated values
*/
PetscErrorCode DMInterpolationEvaluate(DMInterpolationInfo ctx, DM dm, Vec x, Vec v)
{
  PetscInt       dim, coneSize, n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscValidHeaderSpecific(x, VEC_CLASSID, 3);
  PetscValidHeaderSpecific(v, VEC_CLASSID, 4);
  ierr = VecGetLocalSize(v, &n);CHKERRQ(ierr);
  if (n != ctx->n*ctx->dof) SETERRQ2(ctx->comm, PETSC_ERR_ARG_SIZ, "Invalid input vector size %d should be %d", n, ctx->n*ctx->dof);
  if (n) {
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm, ctx->cells[0], &coneSize);CHKERRQ(ierr);
    if (dim == 2) {
      if (coneSize == 3) {
        ierr = DMInterpolate_Triangle_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else if (coneSize == 4) {
        ierr = DMInterpolate_Quad_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for point interpolation", dim);
    } else if (dim == 3) {
      if (coneSize == 4) {
        ierr = DMInterpolate_Tetrahedron_Private(ctx, dm, x, v);CHKERRQ(ierr);
      } else {
        ierr = DMInterpolate_Hex_Private(ctx, dm, x, v);CHKERRQ(ierr);
      }
    } else SETERRQ1(ctx->comm, PETSC_ERR_ARG_OUTOFRANGE, "Unsupported dimension %d for point interpolation", dim);
  }
  PetscFunctionReturn(0);
}

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

.keywords: SNES, nonlinear, default, monitor, norm
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
  ierr = SNESGetFunction(snes, &res, 0, 0);CHKERRQ(ierr);
  ierr = SNESGetDM(snes, &dm);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &s);CHKERRQ(ierr);
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
  DMPlexSNESGetGeometryFEM - Return precomputed geometric data

  Input Parameter:
. dm - The DM

  Output Parameters:
. cellgeom - The values precomputed from cell geometry

  Level: developer

.seealso: DMPlexSNESSetFunctionLocal()
@*/
PetscErrorCode DMPlexSNESGetGeometryFEM(DM dm, Vec *cellgeom)
{
  DMSNES         dmsnes;
  PetscObject    obj;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMGetDMSNES(dm, &dmsnes);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dmsnes, "DMPlexSNES_cellgeom_fem", &obj);CHKERRQ(ierr);
  if (!obj) {
    Vec cellgeom;

    ierr = DMPlexComputeGeometryFEM(dm, &cellgeom);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject) dmsnes, "DMPlexSNES_cellgeom_fem", (PetscObject) cellgeom);CHKERRQ(ierr);
    ierr = VecDestroy(&cellgeom);CHKERRQ(ierr);
  }
  if (cellgeom) {PetscValidPointer(cellgeom, 3); ierr = PetscObjectQuery((PetscObject) dmsnes, "DMPlexSNES_cellgeom_fem", (PetscObject *) cellgeom);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

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

/*@C
  DMPlexGetCellFields - Retrieve the field values values for a chunk of cells

  Input Parameters:
+ dm     - The DM
. cStart - The first cell to include
. cEnd   - The first cell to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
- locA   - A local vector with auxiliary fields, or NULL

  Output Parameters:
+ u   - The field coefficients
. u_t - The fields derivative coefficients
- a   - The auxiliary field coefficients

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexGetCellFields(DM dm, PetscInt cStart, PetscInt cEnd, Vec locX, Vec locX_t, Vec locA, PetscScalar **u, PetscScalar **u_t, PetscScalar **a)
{
  DM             dmAux;
  PetscSection   section, sectionAux;
  PetscDS        prob;
  PetscInt       numCells = cEnd - cStart, totDim, totDimAux, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 5);}
  if (locA)   {PetscValidHeaderSpecific(locA, VEC_CLASSID, 6);}
  PetscValidPointer(u, 7);
  PetscValidPointer(u_t, 8);
  PetscValidPointer(a, 9);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  if (locA) {
    PetscDS probAux;

    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = DMGetWorkArray(dm, numCells*totDim, PETSC_SCALAR, u);CHKERRQ(ierr);
  if (locX_t) {ierr = DMGetWorkArray(dm, numCells*totDim, PETSC_SCALAR, u_t);CHKERRQ(ierr);} else {*u_t = NULL;}
  if (locA)   {ierr = DMGetWorkArray(dm, numCells*totDimAux, PETSC_SCALAR, a);CHKERRQ(ierr);} else {*a = NULL;}
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL, *x_t = NULL, *ul = *u, *ul_t = *u_t, *al = *a;
    PetscInt     i;

    ierr = DMPlexVecGetClosure(dm, section, locX, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) ul[(c-cStart)*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, locX, c, NULL, &x);CHKERRQ(ierr);
    if (locX_t) {
      ierr = DMPlexVecGetClosure(dm, section, locX_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) ul_t[(c-cStart)*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, locX_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (locA) {
      DM dmAuxPlex;

      ierr = DMSNESConvertPlex(dmAux, &dmAuxPlex, PETSC_FALSE);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dmAuxPlex, sectionAux, locA, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) al[(c-cStart)*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAuxPlex, sectionAux, locA, c, NULL, &x);CHKERRQ(ierr);
      ierr = DMDestroy(&dmAuxPlex);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreCellFields - Restore the field values values for a chunk of cells

  Input Parameters:
+ dm     - The DM
. cStart - The first cell to include
. cEnd   - The first cell to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
- locA   - A local vector with auxiliary fields, or NULL

  Output Parameters:
+ u   - The field coefficients
. u_t - The fields derivative coefficients
- a   - The auxiliary field coefficients

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreCellFields(DM dm, PetscInt cStart, PetscInt cEnd, Vec locX, Vec locX_t, Vec locA, PetscScalar **u, PetscScalar **u_t, PetscScalar **a)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, PETSC_SCALAR, u);CHKERRQ(ierr);
  if (*u_t) {ierr = DMRestoreWorkArray(dm, 0, PETSC_SCALAR, u_t);CHKERRQ(ierr);}
  if (*a)   {ierr = DMRestoreWorkArray(dm, 0, PETSC_SCALAR, a);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFaceFields - Retrieve the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
. faceGeometry - A local vector with face geometry
. cellGeometry - A local vector with cell geometry
- locaGrad - A local vector with field gradients, or NULL

  Output Parameters:
+ Nface - The number of faces with field values
. uL - The field values at the left side of the face
- uR - The field values at the right side of the face

  Level: developer

.seealso: DMPlexGetCellFields()
@*/
PetscErrorCode DMPlexGetFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
{
  DM                 dmFace, dmCell, dmGrad = NULL;
  PetscSection       section;
  PetscDS            prob;
  DMLabel            ghostLabel;
  const PetscScalar *facegeom, *cellgeom, *x, *lgrad;
  PetscBool         *isFE;
  PetscInt           dim, Nf, f, Nc, numFaces = fEnd - fStart, iface, face;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(locX, VEC_CLASSID, 4);
  if (locX_t) {PetscValidHeaderSpecific(locX_t, VEC_CLASSID, 5);}
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 6);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 7);
  if (locGrad) {PetscValidHeaderSpecific(locGrad, VEC_CLASSID, 8);}
  PetscValidPointer(uL, 9);
  PetscValidPointer(uR, 10);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalComponents(prob, &Nc);CHKERRQ(ierr);
  ierr = PetscMalloc1(Nf, &isFE);CHKERRQ(ierr);
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;

    ierr = PetscDSGetDiscretization(prob, f, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if (id == PETSCFE_CLASSID)      {isFE[f] = PETSC_TRUE;}
    else if (id == PETSCFV_CLASSID) {isFE[f] = PETSC_FALSE;}
    else                            {isFE[f] = PETSC_FALSE;}
  }
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = VecGetArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  if (locGrad) {
    ierr = VecGetDM(locGrad, &dmGrad);CHKERRQ(ierr);
    ierr = VecGetArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = DMGetWorkArray(dm, numFaces*Nc, PETSC_SCALAR, uL);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numFaces*Nc, PETSC_SCALAR, uR);CHKERRQ(ierr);
  /* Right now just eat the extra work for FE (could make a cell loop) */
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscScalar           *xL, *xR, *gL, *gR;
    PetscScalar           *uLl = *uL, *uRl = *uR;
    PetscInt               ghost, nsupp, nchild;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    for (f = 0; f < Nf; ++f) {
      PetscInt off;

      ierr = PetscDSGetComponentOffset(prob, f, &off);CHKERRQ(ierr);
      if (isFE[f]) {
        const PetscInt *cone;
        PetscInt        comp, coneSizeL, coneSizeR, faceLocL, faceLocR, ldof, rdof, d;

        xL = xR = NULL;
        ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRQ(ierr); 
        ierr = DMPlexVecGetClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL);CHKERRQ(ierr);
        ierr = DMPlexVecGetClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, cells[0], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, cells[0], &coneSizeL);CHKERRQ(ierr);
        for (faceLocL = 0; faceLocL < coneSizeL; ++faceLocL) if (cone[faceLocL] == face) break;
        ierr = DMPlexGetCone(dm, cells[1], &cone);CHKERRQ(ierr);
        ierr = DMPlexGetConeSize(dm, cells[1], &coneSizeR);CHKERRQ(ierr);
        for (faceLocR = 0; faceLocR < coneSizeR; ++faceLocR) if (cone[faceLocR] == face) break;
        if (faceLocL == coneSizeL && faceLocR == coneSizeR) SETERRQ3(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %d in cone of cell %d or cell %d", face, cells[0], cells[1]);
        /* Check that FEM field has values in the right cell (sometimes its an FV ghost cell) */
        /* TODO: this is a hack that might not be right for nonconforming */
        if (faceLocL < coneSizeL) {
          ierr = EvaluateFaceFields(prob, f, faceLocL, xL, &uLl[iface*Nc+off]);CHKERRQ(ierr);
          if (rdof == ldof && faceLocR < coneSizeR) {ierr = EvaluateFaceFields(prob, f, faceLocR, xR, &uRl[iface*Nc+off]);CHKERRQ(ierr);}
          else              {for(d = 0; d < comp; ++d) uRl[iface*Nc+off+d] = uLl[iface*Nc+off+d];}
        }
        else {
          ierr = EvaluateFaceFields(prob, f, faceLocR, xR, &uRl[iface*Nc+off]);CHKERRQ(ierr);
          ierr = PetscSectionGetFieldComponents(section, f, &comp);CHKERRQ(ierr);
          for(d = 0; d < comp; ++d) uLl[iface*Nc+off+d] = uRl[iface*Nc+off+d];
        }
        ierr = DMPlexVecRestoreClosure(dm, section, locX, cells[0], &ldof, (PetscScalar **) &xL);CHKERRQ(ierr);
        ierr = DMPlexVecRestoreClosure(dm, section, locX, cells[1], &rdof, (PetscScalar **) &xR);CHKERRQ(ierr);
      } else {
        PetscFV  fv;
        PetscInt numComp, c;

        ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fv);CHKERRQ(ierr);
        ierr = PetscFVGetNumComponents(fv, &numComp);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, cells[0], f, x, &xL);CHKERRQ(ierr);
        ierr = DMPlexPointLocalFieldRead(dm, cells[1], f, x, &xR);CHKERRQ(ierr);
        if (dmGrad) {
          PetscReal dxL[3], dxR[3];

          ierr = DMPlexPointLocalRead(dmGrad, cells[0], lgrad, &gL);CHKERRQ(ierr);
          ierr = DMPlexPointLocalRead(dmGrad, cells[1], lgrad, &gR);CHKERRQ(ierr);
          DMPlex_WaxpyD_Internal(dim, -1, cgL->centroid, fg->centroid, dxL);
          DMPlex_WaxpyD_Internal(dim, -1, cgR->centroid, fg->centroid, dxR);
          for (c = 0; c < numComp; ++c) {
            uLl[iface*Nc+off+c] = xL[c] + DMPlex_DotD_Internal(dim, &gL[c*dim], dxL);
            uRl[iface*Nc+off+c] = xR[c] + DMPlex_DotD_Internal(dim, &gR[c*dim], dxR);
          }
        } else {
          for (c = 0; c < numComp; ++c) {
            uLl[iface*Nc+off+c] = xL[c];
            uRl[iface*Nc+off+c] = xR[c];
          }
        }
      }
    }
    ++iface;
  }
  *Nface = iface;
  ierr = VecRestoreArrayRead(locX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  if (locGrad) {
    ierr = VecRestoreArrayRead(locGrad, &lgrad);CHKERRQ(ierr);
  }
  ierr = PetscFree(isFE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreFaceFields - Restore the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. locX   - A local vector with the solution fields
. locX_t - A local vector with solution field time derivatives, or NULL
. faceGeometry - A local vector with face geometry
. cellGeometry - A local vector with cell geometry
- locaGrad - A local vector with field gradients, or NULL

  Output Parameters:
+ Nface - The number of faces with field values
. uL - The field values at the left side of the face
- uR - The field values at the right side of the face

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreFaceFields(DM dm, PetscInt fStart, PetscInt fEnd, Vec locX, Vec locX_t, Vec faceGeometry, Vec cellGeometry, Vec locGrad, PetscInt *Nface, PetscScalar **uL, PetscScalar **uR)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMRestoreWorkArray(dm, 0, PETSC_SCALAR, uL);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_SCALAR, uR);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexGetFaceGeometry - Retrieve the geometric values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. faceGeometry - A local vector with face geometry
- cellGeometry - A local vector with cell geometry

  Output Parameters:
+ Nface - The number of faces with field values
. fgeom - The extract the face centroid and normal
- vol   - The cell volume

  Level: developer

.seealso: DMPlexGetCellFields()
@*/
PetscErrorCode DMPlexGetFaceGeometry(DM dm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, PetscInt *Nface, PetscFVFaceGeom **fgeom, PetscReal **vol)
{
  DM                 dmFace, dmCell;
  DMLabel            ghostLabel;
  const PetscScalar *facegeom, *cellgeom;
  PetscInt           dim, numFaces = fEnd - fStart, iface, face;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(faceGeometry, VEC_CLASSID, 4);
  PetscValidHeaderSpecific(cellGeometry, VEC_CLASSID, 5);
  PetscValidPointer(fgeom, 6);
  PetscValidPointer(vol, 7);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = VecGetDM(faceGeometry, &dmFace);CHKERRQ(ierr);
  ierr = VecGetArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecGetDM(cellGeometry, &dmCell);CHKERRQ(ierr);
  ierr = VecGetArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  ierr = PetscMalloc1(numFaces, fgeom);CHKERRQ(ierr);
  ierr = DMGetWorkArray(dm, numFaces*2, PETSC_SCALAR, vol);CHKERRQ(ierr);
  for (face = fStart, iface = 0; face < fEnd; ++face) {
    const PetscInt        *cells;
    PetscFVFaceGeom       *fg;
    PetscFVCellGeom       *cgL, *cgR;
    PetscFVFaceGeom       *fgeoml = *fgeom;
    PetscReal             *voll   = *vol;
    PetscInt               ghost, d, nchild, nsupp;

    ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
    ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
    ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
    if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
    ierr = DMPlexPointLocalRead(dmFace, face, facegeom, &fg);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[0], cellgeom, &cgL);CHKERRQ(ierr);
    ierr = DMPlexPointLocalRead(dmCell, cells[1], cellgeom, &cgR);CHKERRQ(ierr);
    for (d = 0; d < dim; ++d) {
      fgeoml[iface].centroid[d] = fg->centroid[d];
      fgeoml[iface].normal[d]   = fg->normal[d];
    }
    voll[iface*2+0] = cgL->volume;
    voll[iface*2+1] = cgR->volume;
    ++iface;
  }
  *Nface = iface;
  ierr = VecRestoreArrayRead(faceGeometry, &facegeom);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(cellGeometry, &cellgeom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexRestoreFaceGeometry - Restore the field values values for a chunk of faces

  Input Parameters:
+ dm     - The DM
. fStart - The first face to include
. fEnd   - The first face to exclude
. faceGeometry - A local vector with face geometry
- cellGeometry - A local vector with cell geometry

  Output Parameters:
+ Nface - The number of faces with field values
. fgeom - The extract the face centroid and normal
- vol   - The cell volume

  Level: developer

.seealso: DMPlexGetFaceFields()
@*/
PetscErrorCode DMPlexRestoreFaceGeometry(DM dm, PetscInt fStart, PetscInt fEnd, Vec faceGeometry, Vec cellGeometry, PetscInt *Nface, PetscFVFaceGeom **fgeom, PetscReal **vol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(*fgeom);CHKERRQ(ierr);
  ierr = DMRestoreWorkArray(dm, 0, PETSC_REAL, vol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeBdResidual_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DM               dmAux = NULL, plex = NULL;
  PetscSection     section, sectionAux = NULL;
  PetscDS          prob, probAux = NULL;
  DMLabel          depth;
  Vec              locA = NULL;
  PetscFEFaceGeom *fgeom;
  PetscScalar     *u = NULL, *u_t = NULL, *a = NULL, *elemVec = NULL;
  PetscInt         dim, totDim, totDimAux, numBd, bd;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  for (bd = 0; bd < numBd; ++bd) {
    DMBoundaryConditionType type;
    const char             *bdLabel;
    DMLabel                 label;
    IS                      pointIS;
    const PetscInt         *points;
    const PetscInt         *values;
    PetscInt                field, numValues, v, numPoints, p, dep, numFaces, face;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &type, NULL, &bdLabel, &field, NULL, NULL, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, field, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
      if (!pointIS) continue; /* No points with that id on this process */
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0, numFaces = 0; p < numPoints; ++p) {
        ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
        if (dep == dim-1) ++numFaces;
      }
      ierr = PetscMalloc4(numFaces*totDim,&u,locX_t ? numFaces*totDim : 0, &u_t, numFaces,&fgeom, numFaces*totDim,&elemVec);CHKERRQ(ierr);
      if (locA) {ierr = PetscMalloc1(numFaces*totDimAux,&a);CHKERRQ(ierr);}
      for (p = 0, face = 0; p < numPoints; ++p) {
        const PetscInt point = points[p], *support, *cone;
        PetscScalar   *x     = NULL;
        PetscReal      dummyJ[9], dummyDetJ;
        PetscInt       i, coneSize, faceLoc;

        ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
        if (dep != dim-1) continue;
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, support[0], NULL, NULL, dummyJ, fgeom[face].invJ[0], &dummyDetJ);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, point, NULL, fgeom[face].v0, fgeom[face].J, NULL, &fgeom[face].detJ);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFVM(dm, point, NULL, NULL, fgeom[face].n);CHKERRQ(ierr);
        if (fgeom[face].detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", (double)fgeom[face].detJ, point);
        ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
        for (faceLoc = 0; faceLoc < coneSize; ++faceLoc) if (cone[faceLoc] == point) break;
        if (faceLoc == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %d in cone of support[0] %d", point, support[0]);
        fgeom[face].face[0] = faceLoc;
        ierr = DMPlexVecGetClosure(dm, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dm, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        if (locX_t) {
          ierr = DMPlexVecGetClosure(dm, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
          for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
          ierr = DMPlexVecRestoreClosure(dm, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        }
        if (locA) {
          ierr = DMPlexVecGetClosure(plex, sectionAux, locA, support[0], NULL, &x);CHKERRQ(ierr);
          for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
          ierr = DMPlexVecRestoreClosure(plex, sectionAux, locA, support[0], NULL, &x);CHKERRQ(ierr);
        }
        ++face;
      }
      ierr = PetscMemzero(elemVec, numFaces* totDim * sizeof(PetscScalar));CHKERRQ(ierr);
      {
        PetscFE         fe;
        PetscQuadrature q;
        PetscInt        numQuadPoints, Nb;
        /* Conforming batches */
        PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
        /* Remainder */
        PetscInt        Nr, offset;

        ierr = PetscDSGetDiscretization(prob, field, (PetscObject *) &fe);CHKERRQ(ierr);
        ierr = PetscFEGetFaceQuadrature(fe, &q);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
        blockSize = Nb*numQuadPoints;
        batchSize = numBlocks * blockSize;
        ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numFaces / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numFaces % (numBatches*batchSize);
        offset    = numFaces - Nr;
        ierr = PetscFEIntegrateBdResidual(fe, prob, field, Ne, fgeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEIntegrateBdResidual(fe, prob, field, Nr, &fgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, &elemVec[offset*totDim]);CHKERRQ(ierr);
      }
      for (p = 0, face = 0; p < numPoints; ++p) {
        const PetscInt point = points[p], *support;

        ierr = DMLabelGetValue(depth, point, &dep);CHKERRQ(ierr);
        if (dep != dim-1) continue;
        if (mesh->printFEM > 1) {ierr = DMPrintCellVector(point, "BdResidual", totDim, &elemVec[face*totDim]);CHKERRQ(ierr);}
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexVecSetClosure(dm, NULL, locF, support[0], &elemVec[face*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
        ++face;
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      ierr = PetscFree4(u,u_t,fgeom,elemVec);CHKERRQ(ierr);
      if (locA) {ierr = PetscFree(a);CHKERRQ(ierr);}
    }
  }
  if (plex) {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeResidual_Internal(DM dm, PetscInt cStart, PetscInt cEnd, PetscReal time, Vec locX, Vec locX_t, PetscReal t, Vec locF, void *user)
{
  DM_Plex          *mesh       = (DM_Plex *) dm->data;
  const char       *name       = "Residual";
  DM                dmAux      = NULL;
  DM                dmGrad     = NULL;
  DMLabel           ghostLabel = NULL;
  PetscDS           prob       = NULL;
  PetscDS           probAux    = NULL;
  PetscSection      section    = NULL;
  PetscBool         useFEM     = PETSC_FALSE;
  PetscBool         useFVM     = PETSC_FALSE;
  PetscBool         isImplicit = (locX_t || time == PETSC_MIN_REAL) ? PETSC_TRUE : PETSC_FALSE;
  PetscFV           fvm        = NULL;
  PetscFECellGeom  *cgeomFEM   = NULL;
  PetscScalar      *cgeomScal;
  PetscFVCellGeom  *cgeomFVM   = NULL;
  PetscFVFaceGeom  *fgeomFVM   = NULL;
  Vec               locA, cellGeometryFEM = NULL, cellGeometryFVM = NULL, faceGeometryFVM = NULL, grad, locGrad = NULL;
  PetscScalar      *u = NULL, *u_t, *a, *uL, *uR;
  PetscInt          Nf, f, totDim, totDimAux, numChunks, cellChunkSize, faceChunkSize, chunk, fStart, fEnd;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_ResidualFEM,dm,0,0,0);CHKERRQ(ierr);
  /* TODO The places where we have to use isFE are probably the member functions for the PetscDisc class */
  /* TODO The FVM geometry is over-manipulated. Make the precalc functions return exactly what we need */
  /* FEM+FVM */
  /* 1: Get sizes from dm and dmAux */
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "ghost", &ghostLabel);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
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
    ierr = DMPlexSNESGetGeometryFEM(dm, &cellGeometryFEM);CHKERRQ(ierr);
    ierr = VecGetArray(cellGeometryFEM, &cgeomScal);CHKERRQ(ierr);
    if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
      DM dmCell;
      PetscInt c;

      ierr = VecGetDM(cellGeometryFEM,&dmCell);CHKERRQ(ierr);
      ierr = PetscMalloc1(cEnd-cStart,&cgeomFEM);CHKERRQ(ierr);
      for (c = 0; c < cEnd - cStart; c++) {
        PetscScalar *thisgeom;

        ierr = DMPlexPointLocalRef(dmCell, c + cStart, cgeomScal, &thisgeom);CHKERRQ(ierr);
        cgeomFEM[c] = *((PetscFECellGeom *) thisgeom);
      }
    }
    else {
      cgeomFEM = (PetscFECellGeom *) cgeomScal;
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
  ierr = DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd);CHKERRQ(ierr);
  numChunks     = 1;
  cellChunkSize = (cEnd - cStart)/numChunks;
  faceChunkSize = (fEnd - fStart)/numChunks;
  for (chunk = 0; chunk < numChunks; ++chunk) {
    PetscScalar     *elemVec, *fluxL, *fluxR;
    PetscReal       *vol;
    PetscFVFaceGeom *fgeom;
    PetscInt         cS = cStart+chunk*cellChunkSize, cE = PetscMin(cS+cellChunkSize, cEnd), numCells = cE - cS, cell;
    PetscInt         fS = fStart+chunk*faceChunkSize, fE = PetscMin(fS+faceChunkSize, fEnd), numFaces = 0, face;

    /* Extract field coefficients */
    if (useFEM) {
      ierr = DMPlexGetCellFields(dm, cS, cE, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numCells*totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
      ierr = PetscMemzero(elemVec, numCells*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexGetFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexGetFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, PETSC_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMGetWorkArray(dm, numFaces*totDim, PETSC_SCALAR, &fluxR);CHKERRQ(ierr);
      ierr = PetscMemzero(fluxL, numFaces*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
      ierr = PetscMemzero(fluxR, numFaces*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
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
        PetscQuadrature q;
        PetscInt        Nq, Nb;

        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);

        ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(q, NULL, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        blockSize = Nb*Nq;
        batchSize = numBlocks * blockSize;
        ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numCells / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numCells % (numBatches*batchSize);
        offset    = numCells - Nr;
        /* Integrate FE residual to get elemVec (need fields at quadrature points) */
        /*   For FV, I think we use a P0 basis and the cell coefficients (for subdivided cells, we can tweak the basis tabulation to be the indicator function) */
        ierr = PetscFEIntegrateResidual(fe, prob, f, Ne, cgeomFEM, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
        ierr = PetscFEIntegrateResidual(fe, prob, f, Nr, &cgeomFEM[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
      } else if (id == PETSCFV_CLASSID) {
        PetscFV fv = (PetscFV) obj;

        Ne = numFaces;
        /* Riemann solve over faces (need fields at face centroids) */
        /*   We need to evaluate FE fields at those coordinates */
        ierr = PetscFVIntegrateRHSFunction(fv, prob, f, Ne, fgeom, vol, uL, uR, fluxL, fluxR);CHKERRQ(ierr);
      } else SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %d", f);
    }
    if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
      ierr = PetscFree(cgeomFEM);CHKERRQ(ierr);
    }
    else {
      cgeomFEM = NULL;
    }
    if (cellGeometryFEM) {ierr = VecRestoreArray(cellGeometryFEM, &cgeomScal);CHKERRQ(ierr);}
    /* Loop over domain */
    if (useFEM) {
      /* Add elemVec to locX */
      for (cell = cS; cell < cE; ++cell) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellVector(cell, name, totDim, &elemVec[cell*totDim]);CHKERRQ(ierr);}
        if (ghostLabel) {
          PetscInt ghostVal;

          ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
          if (ghostVal > 0) continue;
        }
        ierr = DMPlexVecSetClosure(dm, section, locF, cell, &elemVec[cell*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
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
          const PetscInt *cells;
          PetscScalar    *fL = NULL, *fR = NULL;
          PetscInt        ghost, d, nsupp, nchild;

          ierr = DMLabelGetValue(ghostLabel, face, &ghost);CHKERRQ(ierr);
          ierr = DMPlexGetSupportSize(dm, face, &nsupp);CHKERRQ(ierr);
          ierr = DMPlexGetTreeChildren(dm, face, &nchild, NULL);CHKERRQ(ierr);
          if (ghost >= 0 || nsupp > 2 || nchild > 0) continue;
          ierr = DMPlexGetSupport(dm, face, &cells);CHKERRQ(ierr);
          ierr = DMLabelGetValue(ghostLabel,cells[0],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, cells[0], f, fa, &fL);CHKERRQ(ierr);}
          ierr = DMLabelGetValue(ghostLabel,cells[1],&ghost);CHKERRQ(ierr);
          if (ghost <= 0) {ierr = DMPlexPointLocalFieldRef(dm, cells[1], f, fa, &fR);CHKERRQ(ierr);}
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
        for (cell = cS; cell < cE; ++cell) {
          PetscScalar *u_t, *r;

          if (ghostLabel) {
            PetscInt ghostVal;

            ierr = DMLabelGetValue(ghostLabel,cell,&ghostVal);CHKERRQ(ierr);
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
      ierr = DMPlexRestoreCellFields(dm, cS, cE, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numCells*totDim, PETSC_SCALAR, &elemVec);CHKERRQ(ierr);
    }
    if (useFVM) {
      ierr = DMPlexRestoreFaceFields(dm, fS, fE, locX, locX_t, faceGeometryFVM, cellGeometryFVM, locGrad, &numFaces, &uL, &uR);CHKERRQ(ierr);
      ierr = DMPlexRestoreFaceGeometry(dm, fS, fE, faceGeometryFVM, cellGeometryFVM, &numFaces, &fgeom, &vol);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, PETSC_SCALAR, &fluxL);CHKERRQ(ierr);
      ierr = DMRestoreWorkArray(dm, numFaces*totDim, PETSC_SCALAR, &fluxR);CHKERRQ(ierr);
      if (dmGrad) {ierr = DMRestoreLocalVector(dmGrad, &locGrad);CHKERRQ(ierr);}
    }
  }

  if (useFEM) {ierr = DMPlexComputeBdResidual_Internal(dm, locX, locX_t, t, locF, user);CHKERRQ(ierr);}

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

static PetscErrorCode DMPlexComputeResidualFEM_Check_Internal(DM dm, Vec X, Vec X_t, PetscReal t, Vec F, void *user)
{
  DM                dmCh, dmAux;
  Vec               A, cellgeom;
  PetscDS           prob, probCh, probAux = NULL;
  PetscQuadrature   q;
  PetscSection      section, sectionAux;
  PetscFECellGeom  *cgeom = NULL;
  PetscScalar      *cgeomScal;
  PetscScalar      *elemVec, *elemVecCh, *u, *u_t, *a = NULL;
  PetscInt          dim, Nf, f, numCells, cStart, cEnd, c;
  PetscInt          totDim, totDimAux = 0, diffCell = 0;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmCh", (PetscObject *) &dmCh);CHKERRQ(ierr);
  ierr = DMGetDS(dmCh, &probCh);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMGetDefaultSection(dmAux, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = VecSet(F, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc3(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*totDim,&elemVec);CHKERRQ(ierr);
  ierr = PetscMalloc1(numCells*totDim,&elemVecCh);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMPlexSNESGetGeometryFEM(dm, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
    DM dmCell;

    ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart,&cgeom);CHKERRQ(ierr);
    for (c = 0; c < cEnd - cStart; c++) {
      PetscScalar *thisgeom;

      ierr = DMPlexPointLocalRef(dmCell, c + cStart, cgeomScal, &thisgeom);CHKERRQ(ierr);
      cgeom[c] = *((PetscFECellGeom *) thisgeom);
    }
  }
  else {
    cgeom = (PetscFECellGeom *) cgeomScal;
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL, *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[c*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[c*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      DM dmAuxPlex;

      ierr = DMSNESConvertPlex(dmAux,&dmAuxPlex, PETSC_FALSE);CHKERRQ(ierr);
      ierr = DMPlexVecGetClosure(dmAuxPlex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[c*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(dmAuxPlex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      ierr = DMDestroy(&dmAuxPlex);CHKERRQ(ierr);
    }
  }
  for (f = 0; f < Nf; ++f) {
    PetscFE  fe, feCh;
    PetscInt numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscDSGetDiscretization(prob, f, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(probCh, f, (PetscObject *) &feCh);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks * blockSize;
    ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    ierr = PetscFEIntegrateResidual(fe, prob, f, Ne, cgeom, u, u_t, probAux, a, t, elemVec);CHKERRQ(ierr);
    ierr = PetscFEIntegrateResidual(feCh, prob, f, Ne, cgeom, u, u_t, probAux, a, t, elemVecCh);CHKERRQ(ierr);
    ierr = PetscFEIntegrateResidual(fe, prob, f, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVec[offset*totDim]);CHKERRQ(ierr);
    ierr = PetscFEIntegrateResidual(feCh, prob, f, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, &elemVecCh[offset*totDim]);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscBool diff = PETSC_FALSE;
    PetscInt  d;

    for (d = 0; d < totDim; ++d) if (PetscAbsScalar(elemVec[c*totDim+d] - elemVecCh[c*totDim+d]) > 1.0e-7) {diff = PETSC_TRUE;break;}
    if (diff) {
      ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "Different cell %d\n", c);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Residual", totDim, &elemVec[c*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Check Residual", totDim, &elemVecCh[c*totDim]);CHKERRQ(ierr);
      ++diffCell;
    }
    if (diffCell > 9) break;
    ierr = DMPlexVecSetClosure(dm, section, F, c, &elemVec[c*totDim], ADD_ALL_VALUES);CHKERRQ(ierr);
  }
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
    ierr = PetscFree(cgeom);CHKERRQ(ierr);
  }
  else {
    cgeom = NULL;
  }
  ierr = VecRestoreArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  ierr = PetscFree3(u,u_t,elemVec);CHKERRQ(ierr);
  ierr = PetscFree(elemVecCh);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscFree(a);CHKERRQ(ierr);}
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

.seealso: DMPlexComputeJacobianActionFEM()
@*/
PetscErrorCode DMPlexSNESComputeResidualFEM(DM dm, Vec X, Vec F, void *user)
{
  PetscObject    check;
  PetscInt       cStart, cEnd, cEndInterior;
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(plex, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  /* The dmCh is used to check two mathematically equivalent discretizations for computational equivalence */
  ierr = PetscObjectQuery((PetscObject) plex, "dmCh", &check);CHKERRQ(ierr);
  if (check) {ierr = DMPlexComputeResidualFEM_Check_Internal(plex, X, NULL, 0.0, F, user);CHKERRQ(ierr);}
  else       {ierr = DMPlexComputeResidual_Internal(plex, cStart, cEnd, PETSC_MIN_REAL, X, NULL, 0.0, F, user);CHKERRQ(ierr);}
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

.seealso: DMPlexComputeJacobianActionFEM()
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

PetscErrorCode DMPlexComputeBdJacobian_Internal(DM dm, Vec locX, Vec locX_t, PetscReal t, PetscReal X_tShift, Mat Jac, Mat JacP, void *user)
{
  DM_Plex         *mesh = (DM_Plex *) dm->data;
  DM               dmAux = NULL, plex = NULL;
  PetscSection     section, globalSection, subSection, sectionAux = NULL;
  PetscDS          prob, probAux = NULL;
  DMLabel          depth;
  Vec              locA = NULL;
  PetscFEFaceGeom *fgeom;
  PetscScalar     *u = NULL, *u_t = NULL, *a = NULL, *elemMat = NULL;
  PetscInt         dim, totDim, totDimAux, numBd, bd, Nf;
  PetscBool        isMatISP;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {
    ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
  }
  ierr = DMPlexGetDepthLabel(dm, &depth);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &Nf);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSGetNumBoundary(prob, &numBd);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &locA);CHKERRQ(ierr);
  if (locA) {
    ierr = VecGetDM(locA, &dmAux);CHKERRQ(ierr);
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  for (bd = 0; bd < numBd; ++bd) {
    DMBoundaryConditionType type;
    const char             *bdLabel;
    DMLabel                 label;
    IS                      pointIS;
    const PetscInt         *points;
    const PetscInt         *values;
    PetscInt                fieldI, fieldJ, numValues, v, numPoints, p, dep, numFaces, face;
    PetscObject             obj;
    PetscClassId            id;

    ierr = PetscDSGetBoundary(prob, bd, &type, NULL, &bdLabel, &fieldI, NULL, NULL, NULL, &numValues, &values, NULL);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, fieldI, &obj);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId(obj, &id);CHKERRQ(ierr);
    if ((id != PETSCFE_CLASSID) || (type & DM_BC_ESSENTIAL)) continue;
    ierr = DMGetLabel(dm, bdLabel, &label);CHKERRQ(ierr);
    for (v = 0; v < numValues; ++v) {
      ierr = DMLabelGetStratumSize(label, values[v], &numPoints);CHKERRQ(ierr);
      ierr = DMLabelGetStratumIS(label, values[v], &pointIS);CHKERRQ(ierr);
      if (!pointIS) continue; /* No points with that id on this process */
      ierr = ISGetIndices(pointIS, &points);CHKERRQ(ierr);
      for (p = 0, numFaces = 0; p < numPoints; ++p) {
        ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
        if (dep == dim-1) ++numFaces;
      }
      ierr = PetscMalloc4(numFaces*totDim,&u,locX_t ? numFaces*totDim : 0,&u_t,numFaces,&fgeom,numFaces*totDim*totDim,&elemMat);CHKERRQ(ierr);
      if (locA) {ierr = PetscMalloc1(numFaces*totDimAux,&a);CHKERRQ(ierr);}
      for (p = 0, face = 0; p < numPoints; ++p) {
        const PetscInt point = points[p], *support, *cone;
        PetscScalar   *x     = NULL;
        PetscReal      dummyJ[9], dummyDetJ;
        PetscInt       i, coneSize, faceLoc;

        ierr = DMLabelGetValue(depth, points[p], &dep);CHKERRQ(ierr);
        if (dep != dim-1) continue;
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, support[0], NULL, NULL, dummyJ, fgeom[face].invJ[0], &dummyDetJ);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFEM(dm, point, NULL, fgeom[face].v0, fgeom[face].J, NULL, &fgeom[face].detJ);CHKERRQ(ierr);
        ierr = DMPlexComputeCellGeometryFVM(dm, point, NULL, NULL, fgeom[face].n);CHKERRQ(ierr);
        if (fgeom[face].detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for face %d", (double)fgeom[face].detJ, point);
        ierr = DMPlexGetConeSize(dm, support[0], &coneSize);CHKERRQ(ierr);
        ierr = DMPlexGetCone(dm, support[0], &cone);CHKERRQ(ierr);
        for (faceLoc = 0; faceLoc < coneSize; ++faceLoc) if (cone[faceLoc] == point) break;
        if (faceLoc == coneSize) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %d in cone of support[0] %d", point, support[0]);
        fgeom[face].face[0] = faceLoc;
        ierr = DMPlexVecGetClosure(dm, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        for (i = 0; i < totDim; ++i) u[face*totDim+i] = x[i];
        ierr = DMPlexVecRestoreClosure(dm, section, locX, support[0], NULL, &x);CHKERRQ(ierr);
        if (locX_t) {
          ierr = DMPlexVecGetClosure(dm, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
          for (i = 0; i < totDim; ++i) u_t[face*totDim+i] = x[i];
          ierr = DMPlexVecRestoreClosure(dm, section, locX_t, support[0], NULL, &x);CHKERRQ(ierr);
        }
        if (locA) {
          ierr = DMPlexVecGetClosure(plex, sectionAux, locA, support[0], NULL, &x);CHKERRQ(ierr);
          for (i = 0; i < totDimAux; ++i) a[face*totDimAux+i] = x[i];
          ierr = DMPlexVecRestoreClosure(plex, sectionAux, locA, support[0], NULL, &x);CHKERRQ(ierr);
        }
        ++face;
      }
      ierr = PetscMemzero(elemMat, numFaces*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
      {
        PetscFE         fe;
        PetscQuadrature q;
        PetscInt        numQuadPoints, Nb;
        /* Conforming batches */
        PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
        /* Remainder */
        PetscInt        Nr, offset;

        ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
        ierr = PetscFEGetFaceQuadrature(fe, &q);CHKERRQ(ierr);
        ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
        ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
        ierr = PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
        blockSize = Nb*numQuadPoints;
        batchSize = numBlocks * blockSize;
        ierr =  PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
        numChunks = numFaces / (numBatches*batchSize);
        Ne        = numChunks*numBatches*batchSize;
        Nr        = numFaces % (numBatches*batchSize);
        offset    = numFaces - Nr;
        for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
          ierr = PetscFEIntegrateBdJacobian(fe, prob, fieldI, fieldJ, Ne, fgeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
          ierr = PetscFEIntegrateBdJacobian(fe, prob, fieldI, fieldJ, Nr, &fgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, a ? &a[offset*totDimAux] : NULL, t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
        }
      }
      for (p = 0, face = 0; p < numPoints; ++p) {
        const PetscInt point = points[p], *support;

        ierr = DMLabelGetValue(depth, point, &dep);CHKERRQ(ierr);
        if (dep != dim-1) continue;
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(point, "BdJacobian", totDim, totDim, &elemMat[face*totDim*totDim]);CHKERRQ(ierr);}
        ierr = DMPlexGetSupport(dm, point, &support);CHKERRQ(ierr);
        if (!isMatISP) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, support[0], &elemMat[face*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
        ++face;
      }
      ierr = ISRestoreIndices(pointIS, &points);CHKERRQ(ierr);
      ierr = ISDestroy(&pointIS);CHKERRQ(ierr);
      ierr = PetscFree4(u,u_t,fgeom,elemMat);CHKERRQ(ierr);
      if (locA) {ierr = PetscFree(a);CHKERRQ(ierr);}
    }
  }
  if (plex) {ierr = DMDestroy(&plex);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexComputeJacobian_Internal(DM dm, PetscInt cStart, PetscInt cEnd, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Mat Jac, Mat JacP,void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  const char       *name  = "Jacobian";
  DM                dmAux, plex;
  Vec               A, cellgeom;
  PetscDS           prob, probAux = NULL;
  PetscSection      section, globalSection, subSection, sectionAux;
  PetscFECellGeom  *cgeom = NULL;
  PetscScalar      *cgeomScal;
  PetscScalar      *elemMat, *elemMatP, *elemMatD, *u, *u_t, *a = NULL;
  PetscInt          dim, Nf, fieldI, fieldJ, numCells, c;
  PetscInt          totDim, totDimAux;
  PetscBool         isMatIS, isMatISP, isShell, hasJac, hasPrec, hasDyn, hasFV = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatISP);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  if (isMatISP) {
    ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
  }
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  if (hasJac && hasPrec) {ierr = MatZeroEntries(Jac);CHKERRQ(ierr);}
  ierr = MatZeroEntries(JacP);CHKERRQ(ierr);
  ierr = PetscMalloc5(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,hasJac ? numCells*totDim*totDim : 0,&elemMat,hasPrec ? numCells*totDim*totDim : 0, &elemMatP,hasDyn ? numCells*totDim*totDim : 0, &elemMatD);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMPlexSNESGetGeometryFEM(dm, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
    DM dmCell;

    ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart,&cgeom);CHKERRQ(ierr);
    for (c = 0; c < cEnd - cStart; c++) {
      PetscScalar *thisgeom;

      ierr = DMPlexPointLocalRef(dmCell, c + cStart, cgeomScal, &thisgeom);CHKERRQ(ierr);
      cgeom[c] = *((PetscFECellGeom *) thisgeom);
    }
  } else {
    cgeom = (PetscFECellGeom *) cgeomScal;
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL,  *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[(c-cStart)*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[(c-cStart)*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      ierr = DMPlexVecGetClosure(plex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[(c-cStart)*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
  }
  if (hasJac)  {ierr = PetscMemzero(elemMat,  numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
  if (hasPrec) {ierr = PetscMemzero(elemMatP, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
  if (hasDyn)  {ierr = PetscMemzero(elemMatD, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscClassId    id;
    PetscFE         fe;
    PetscQuadrature q;
    PetscInt        numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt        numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt        Nr, offset;

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscObjectGetClassId((PetscObject) fe, &id);CHKERRQ(ierr);
    if (id == PETSCFV_CLASSID) {hasFV = PETSC_TRUE; continue;}
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(q, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      if (hasJac) {
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Ne, cgeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasPrec) {
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Ne, cgeom, u, u_t, probAux, a, t, X_tShift, elemMatP);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_PRE, fieldI, fieldJ, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatP[offset*totDim*totDim]);CHKERRQ(ierr);
      }
      if (hasDyn) {
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Ne, cgeom, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]);CHKERRQ(ierr);
      }
    }
  }
  if (hasDyn) {
    for (c = 0; c < (cEnd - cStart)*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];
  }
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
        const PetscInt eOffset = (c-cStart)*totDim*totDim;
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
  isMatIS = PETSC_FALSE;
  if (hasPrec && hasJac) {
    ierr = PetscObjectTypeCompare((PetscObject) JacP, MATIS, &isMatIS);CHKERRQ(ierr);
  }
  if (isMatIS && !subSection) {
    ierr = DMPlexGetSubdomainSection(dm, &subSection);CHKERRQ(ierr);
  }
  for (c = cStart; c < cEnd; ++c) {
    if (hasPrec) {
      if (hasJac) {
        if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);}
        if (!isMatIS) {
          ierr = DMPlexMatSetClosure(dm, section, globalSection, Jac, c, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        } else {
          Mat lJ;

          ierr = MatISGetLocalMat(Jac,&lJ);CHKERRQ(ierr);
          ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, c, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
        }
      }
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMatP[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, c, &elemMatP[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, c, &elemMatP[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    } else {
      if (mesh->printFEM > 1) {ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);}
      if (!isMatISP) {
        ierr = DMPlexMatSetClosure(dm, section, globalSection, JacP, c, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      } else {
        Mat lJ;

        ierr = MatISGetLocalMat(JacP,&lJ);CHKERRQ(ierr);
        ierr = DMPlexMatSetClosure(dm, section, subSection, lJ, c, &elemMat[(c-cStart)*totDim*totDim], ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  if (hasFV) {ierr = MatSetOption(JacP, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE);CHKERRQ(ierr);}
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
    ierr = PetscFree(cgeom);CHKERRQ(ierr);
  } else {
    cgeom = NULL;
  }
  ierr = VecRestoreArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  ierr = PetscFree5(u,u_t,elemMat,elemMatP,elemMatD);CHKERRQ(ierr);
  if (dmAux) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  ierr = DMPlexComputeBdJacobian_Internal(dm, X, X_t, t, X_tShift, Jac, JacP, user);CHKERRQ(ierr);
  if (hasJac && hasPrec) {
    ierr = MatAssemblyBegin(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(JacP, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s:\n", name);CHKERRQ(ierr);
    ierr = MatChop(JacP, 1.0e-10);CHKERRQ(ierr);
    ierr = MatView(JacP, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) Jac, MATSHELL, &isShell);CHKERRQ(ierr);
  if (isShell) {
    JacActionCtx *jctx;

    ierr = MatShellGetContext(Jac, &jctx);CHKERRQ(ierr);
    ierr = VecCopy(X, jctx->u);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


PetscErrorCode DMPlexComputeJacobianAction_Internal(DM dm, PetscInt cStart, PetscInt cEnd, PetscReal t, PetscReal X_tShift, Vec X, Vec X_t, Vec Y, Vec Z, void *user)
{
  DM_Plex          *mesh  = (DM_Plex *) dm->data;
  const char       *name  = "Jacobian";
  DM                dmAux, plex;
  Vec               A, cellgeom;
  PetscDS           prob, probAux = NULL;
  PetscQuadrature   quad;
  PetscSection      section, globalSection, sectionAux;
  PetscFECellGeom  *cgeom = NULL;
  PetscScalar      *cgeomScal;
  PetscScalar      *elemMat, *elemMatD, *u, *u_t, *a = NULL, *y, *z;
  PetscInt          dim, Nf, fieldI, fieldJ, numCells, c;
  PetscInt          totDim, totDimAux = 0;
  PetscBool         hasDyn;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(DMPLEX_JacobianFEM,dm,0,0,0);CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(dm, &section);CHKERRQ(ierr);
  ierr = DMGetDefaultGlobalSection(dm, &globalSection);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetTotalDimension(prob, &totDim);CHKERRQ(ierr);
  ierr = PetscDSHasDynamicJacobian(prob, &hasDyn);CHKERRQ(ierr);
  hasDyn = hasDyn && (X_tShift != 0.0) ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscSectionGetNumFields(section, &Nf);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  numCells = cEnd - cStart;
  ierr = PetscObjectQuery((PetscObject) dm, "dmAux", (PetscObject *) &dmAux);CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject) dm, "A", (PetscObject *) &A);CHKERRQ(ierr);
  if (dmAux) {
    ierr = DMConvert(dmAux, DMPLEX, &plex);CHKERRQ(ierr);
    ierr = DMGetDefaultSection(plex, &sectionAux);CHKERRQ(ierr);
    ierr = DMGetDS(dmAux, &probAux);CHKERRQ(ierr);
    ierr = PetscDSGetTotalDimension(probAux, &totDimAux);CHKERRQ(ierr);
  }
  ierr = VecSet(Z, 0.0);CHKERRQ(ierr);
  ierr = PetscMalloc6(numCells*totDim,&u,X_t ? numCells*totDim : 0,&u_t,numCells*totDim*totDim,&elemMat,hasDyn ? numCells*totDim*totDim : 0, &elemMatD,numCells*totDim,&y,totDim,&z);CHKERRQ(ierr);
  if (dmAux) {ierr = PetscMalloc1(numCells*totDimAux, &a);CHKERRQ(ierr);}
  ierr = DMPlexSNESGetGeometryFEM(dm, &cellgeom);CHKERRQ(ierr);
  ierr = VecGetArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {
    DM dmCell;

    ierr = VecGetDM(cellgeom,&dmCell);CHKERRQ(ierr);
    ierr = PetscMalloc1(cEnd-cStart,&cgeom);CHKERRQ(ierr);
    for (c = 0; c < cEnd - cStart; c++) {
      PetscScalar *thisgeom;

      ierr = DMPlexPointLocalRef(dmCell, c + cStart, cgeomScal, &thisgeom);CHKERRQ(ierr);
      cgeom[c] = *((PetscFECellGeom *) thisgeom);
    }
  } else {
    cgeom = (PetscFECellGeom *) cgeomScal;
  }
  for (c = cStart; c < cEnd; ++c) {
    PetscScalar *x = NULL,  *x_t = NULL;
    PetscInt     i;

    ierr = DMPlexVecGetClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) u[(c-cStart)*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, X, c, NULL, &x);CHKERRQ(ierr);
    if (X_t) {
      ierr = DMPlexVecGetClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
      for (i = 0; i < totDim; ++i) u_t[(c-cStart)*totDim+i] = x_t[i];
      ierr = DMPlexVecRestoreClosure(dm, section, X_t, c, NULL, &x_t);CHKERRQ(ierr);
    }
    if (dmAux) {
      ierr = DMPlexVecGetClosure(plex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
      for (i = 0; i < totDimAux; ++i) a[(c-cStart)*totDimAux+i] = x[i];
      ierr = DMPlexVecRestoreClosure(plex, sectionAux, A, c, NULL, &x);CHKERRQ(ierr);
    }
    ierr = DMPlexVecGetClosure(dm, section, Y, c, NULL, &x);CHKERRQ(ierr);
    for (i = 0; i < totDim; ++i) y[(c-cStart)*totDim+i] = x[i];
    ierr = DMPlexVecRestoreClosure(dm, section, Y, c, NULL, &x);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(elemMat, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);
  if (hasDyn)  {ierr = PetscMemzero(elemMatD, numCells*totDim*totDim * sizeof(PetscScalar));CHKERRQ(ierr);}
  for (fieldI = 0; fieldI < Nf; ++fieldI) {
    PetscFE  fe;
    PetscInt numQuadPoints, Nb;
    /* Conforming batches */
    PetscInt numChunks, numBatches, numBlocks, Ne, blockSize, batchSize;
    /* Remainder */
    PetscInt Nr, offset;

    ierr = PetscDSGetDiscretization(prob, fieldI, (PetscObject *) &fe);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
    ierr = PetscFEGetDimension(fe, &Nb);CHKERRQ(ierr);
    ierr = PetscFEGetTileSizes(fe, NULL, &numBlocks, NULL, &numBatches);CHKERRQ(ierr);
    ierr = PetscQuadratureGetData(quad, NULL, NULL, &numQuadPoints, NULL, NULL);CHKERRQ(ierr);
    blockSize = Nb*numQuadPoints;
    batchSize = numBlocks * blockSize;
    ierr = PetscFESetTileSizes(fe, blockSize, numBlocks, batchSize, numBatches);CHKERRQ(ierr);
    numChunks = numCells / (numBatches*batchSize);
    Ne        = numChunks*numBatches*batchSize;
    Nr        = numCells % (numBatches*batchSize);
    offset    = numCells - Nr;
    for (fieldJ = 0; fieldJ < Nf; ++fieldJ) {
      ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Ne, cgeom, u, u_t, probAux, a, t, X_tShift, elemMat);CHKERRQ(ierr);
      ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN, fieldI, fieldJ, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMat[offset*totDim*totDim]);CHKERRQ(ierr);
      if (hasDyn) {
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Ne, cgeom, u, u_t, probAux, a, t, X_tShift, elemMatD);CHKERRQ(ierr);
        ierr = PetscFEIntegrateJacobian(fe, prob, PETSCFE_JACOBIAN_DYN, fieldI, fieldJ, Nr, &cgeom[offset], &u[offset*totDim], u_t ? &u_t[offset*totDim] : NULL, probAux, &a[offset*totDimAux], t, X_tShift, &elemMatD[offset*totDim*totDim]);CHKERRQ(ierr);
      }
    }
  }
  if (hasDyn) {
    for (c = 0; c < (cEnd - cStart)*totDim*totDim; ++c) elemMat[c] += X_tShift*elemMatD[c];
  }
  for (c = cStart; c < cEnd; ++c) {
    const PetscBLASInt M = totDim, one = 1;
    const PetscScalar  a = 1.0, b = 0.0;

    PetscStackCallBLAS("BLASgemv", BLASgemv_("N", &M, &M, &a, &elemMat[(c-cStart)*totDim*totDim], &M, &y[(c-cStart)*totDim], &one, &b, z, &one));
    if (mesh->printFEM > 1) {
      ierr = DMPrintCellMatrix(c, name, totDim, totDim, &elemMat[(c-cStart)*totDim*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Y",  totDim, &y[(c-cStart)*totDim]);CHKERRQ(ierr);
      ierr = DMPrintCellVector(c, "Z",  totDim, z);CHKERRQ(ierr);
    }
    ierr = DMPlexVecSetClosure(dm, section, Z, c, z, ADD_VALUES);CHKERRQ(ierr);
  }
  if (sizeof(PetscFECellGeom) % sizeof(PetscScalar)) {ierr = PetscFree(cgeom);CHKERRQ(ierr);}
  else                                               {cgeom = NULL;}
  ierr = VecRestoreArray(cellgeom, &cgeomScal);CHKERRQ(ierr);
  ierr = PetscFree6(u,u_t,elemMat,elemMatD,y,z);CHKERRQ(ierr);
  if (dmAux) {
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = DMDestroy(&plex);CHKERRQ(ierr);
  }
  if (mesh->printFEM) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Z:\n");CHKERRQ(ierr);
    ierr = VecView(Z, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
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
  PetscInt       cStart, cEnd, cEndInterior;
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(plex, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMPlexComputeJacobian_Internal(plex, cStart, cEnd, 0.0, 0.0, X, NULL, Jac, JacP, user);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMPlexSNESComputeJacobianActionFEM - Form the local portion of the Jacobian action Z = J(X) Y at the local solution X using pointwise functions specified by the user.

  Input Parameters:
+ dm - The mesh
. X  - Local solution vector
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
PetscErrorCode DMPlexSNESComputeJacobianActionFEM(DM dm, Vec X, Vec Y, Vec Z, void *user)
{
  PetscInt       cStart, cEnd, cEndInterior;
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSNESConvertPlex(dm,&plex,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(plex, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMPlexComputeJacobianAction_Internal(plex, cStart, cEnd, 0.0, 0.0, X, NULL, Y, Z, user);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
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
  PetscFunctionReturn(0);
}

PetscErrorCode DMSNESCheckFromOptions_Internal(SNES snes, DM dm, Vec u, Vec sol, PetscErrorCode (**exactFuncs)(PetscInt, PetscReal, const PetscReal x[], PetscInt, PetscScalar *u, void *ctx), void **ctxs)
{
  PetscDS        prob;
  Mat            J, M;
  Vec            r, b;
  MatNullSpace   nullSpace;
  PetscReal     *error, res = 0.0;
  PetscInt       numFields;
  PetscBool      hasJac, hasPrec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &J);CHKERRQ(ierr);
  /* TODO Null space for J */
  /* Check discretization error */
  ierr = DMGetNumFields(dm, &numFields);CHKERRQ(ierr);
  ierr = PetscMalloc1(PetscMax(1, numFields), &error);CHKERRQ(ierr);
  if (numFields > 1) {
    PetscInt f;

    ierr = DMComputeL2FieldDiff(dm, 0.0, exactFuncs, ctxs, u, error);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: [");CHKERRQ(ierr);
    for (f = 0; f < numFields; ++f) {
      if (f) {ierr = PetscPrintf(PETSC_COMM_WORLD, ", ");CHKERRQ(ierr);}
      if (error[f] >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "%g", (double)error[f]);CHKERRQ(ierr);}
      else                     {ierr = PetscPrintf(PETSC_COMM_WORLD, "< 1.0e-11");CHKERRQ(ierr);}
    }
    ierr = PetscPrintf(PETSC_COMM_WORLD, "]\n");CHKERRQ(ierr);
  } else {
    ierr = DMComputeL2Diff(dm, 0.0, exactFuncs, ctxs, u, &error[0]);CHKERRQ(ierr);
    if (error[0] >= 1.0e-11) {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: %g\n", (double)error[0]);CHKERRQ(ierr);}
    else                     {ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Error: < 1.0e-11\n");CHKERRQ(ierr);}
  }
  ierr = PetscFree(error);CHKERRQ(ierr);
  /* Check residual */
  ierr = SNESComputeFunction(snes, u, r);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
  ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Initial Residual");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"res_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(r, NULL, "-vec_view");CHKERRQ(ierr);
  /* Check Jacobian */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSHasJacobian(prob, &hasJac);CHKERRQ(ierr);
  ierr = PetscDSHasJacobianPreconditioner(prob, &hasPrec);CHKERRQ(ierr);
  if (hasJac && hasPrec) {
    ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
    ierr = SNESComputeJacobian(snes, u, J, M);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) M, "jacpre_");CHKERRQ(ierr);
    ierr = MatViewFromOptions(M, NULL, "-mat_view");CHKERRQ(ierr);
    ierr = MatDestroy(&M);CHKERRQ(ierr);
  } else {
    ierr = SNESComputeJacobian(snes, u, J, J);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetOptionsPrefix((PetscObject) J, "jac_");CHKERRQ(ierr);
  ierr = MatViewFromOptions(J, NULL, "-mat_view");CHKERRQ(ierr);
  ierr = MatGetNullSpace(J, &nullSpace);CHKERRQ(ierr);
  if (nullSpace) {
    PetscBool isNull;
    ierr = MatNullSpaceTest(nullSpace, J, &isNull);CHKERRQ(ierr);
    if (!isNull) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "The null space calculated for the system operator is invalid.");
  }
  ierr = VecDuplicate(u, &b);CHKERRQ(ierr);
  ierr = VecSet(r, 0.0);CHKERRQ(ierr);
  ierr = SNESComputeFunction(snes, r, b);CHKERRQ(ierr);
  ierr = MatMult(J, u, r);CHKERRQ(ierr);
  ierr = VecAXPY(r, 1.0, b);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecNorm(r, NORM_2, &res);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Linear L_2 Residual: %g\n", (double)res);CHKERRQ(ierr);
  ierr = VecChop(r, 1.0e-10);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) r, "Au - b = Au + F(0)");CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)r,"linear_res_");CHKERRQ(ierr);
  ierr = VecViewFromOptions(r, NULL, "-vec_view");CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullSpace);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  ierr = DMSNESCheckFromOptions_Internal(snes, dm, u, sol, exactFuncs, ctxs);CHKERRQ(ierr);
  ierr = VecDestroy(&sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
