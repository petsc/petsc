
/*
   Code for manipulating distributed regular 1d arrays in parallel.
   This file was created by Peter Mell   6/30/95
*/

#include <petsc/private/dmdaimpl.h> /*I  "petscdmda.h"   I*/

#include <petscdraw.h>
static PetscErrorCode DMView_DA_1d(DM da, PetscViewer viewer)
{
  PetscMPIInt rank;
  PetscBool   iascii, isdraw, isglvis, isbinary;
  DM_DA      *dd = (DM_DA *)da->data;
#if defined(PETSC_HAVE_MATLAB)
  PetscBool ismatlab;
#endif

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)da), &rank));

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERDRAW, &isdraw));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERGLVIS, &isglvis));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERBINARY, &isbinary));
#if defined(PETSC_HAVE_MATLAB)
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERMATLAB, &ismatlab));
#endif
  if (iascii) {
    PetscViewerFormat format;

    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_LOAD_BALANCE) {
      PetscInt      i, nmax = 0, nmin = PETSC_MAX_INT, navg = 0, *nz, nzlocal;
      DMDALocalInfo info;
      PetscMPIInt   size;
      PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)da), &size));
      PetscCall(DMDAGetLocalInfo(da, &info));
      nzlocal = info.xm;
      PetscCall(PetscMalloc1(size, &nz));
      PetscCallMPI(MPI_Allgather(&nzlocal, 1, MPIU_INT, nz, 1, MPIU_INT, PetscObjectComm((PetscObject)da)));
      for (i = 0; i < (PetscInt)size; i++) {
        nmax = PetscMax(nmax, nz[i]);
        nmin = PetscMin(nmin, nz[i]);
        navg += nz[i];
      }
      PetscCall(PetscFree(nz));
      navg = navg / size;
      PetscCall(PetscViewerASCIIPrintf(viewer, "  Load Balance - Grid Points: Min %" PetscInt_FMT "  avg %" PetscInt_FMT "  max %" PetscInt_FMT "\n", nmin, navg, nmax));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    if (format != PETSC_VIEWER_ASCII_VTK_DEPRECATED && format != PETSC_VIEWER_ASCII_VTK_CELL_DEPRECATED && format != PETSC_VIEWER_ASCII_GLVIS) {
      DMDALocalInfo info;
      PetscCall(DMDAGetLocalInfo(da, &info));
      PetscCall(PetscViewerASCIIPushSynchronized(viewer));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "Processor [%d] M %" PetscInt_FMT " m %" PetscInt_FMT " w %" PetscInt_FMT " s %" PetscInt_FMT "\n", rank, dd->M, dd->m, dd->w, dd->s));
      PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "X range of indices: %" PetscInt_FMT " %" PetscInt_FMT "\n", info.xs, info.xs + info.xm));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerASCIIPopSynchronized(viewer));
    } else if (format == PETSC_VIEWER_ASCII_GLVIS) PetscCall(DMView_DA_GLVis(da, viewer));
    else PetscCall(DMView_DA_VTK(da, viewer));
  } else if (isdraw) {
    PetscDraw draw;
    double    ymin = -1, ymax = 1, xmin = -1, xmax = dd->M, x;
    PetscInt  base;
    char      node[10];
    PetscBool isnull;

    PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
    PetscCall(PetscDrawIsNull(draw, &isnull));
    if (isnull) PetscFunctionReturn(PETSC_SUCCESS);

    PetscCall(PetscDrawCheckResizedWindow(draw));
    PetscCall(PetscDrawClear(draw));
    PetscCall(PetscDrawSetCoordinates(draw, xmin, ymin, xmax, ymax));

    PetscDrawCollectiveBegin(draw);
    /* first processor draws all node lines */
    if (rank == 0) {
      PetscInt xmin_tmp;
      ymin = 0.0;
      ymax = 0.3;
      for (xmin_tmp = 0; xmin_tmp < dd->M; xmin_tmp++) PetscCall(PetscDrawLine(draw, (double)xmin_tmp, ymin, (double)xmin_tmp, ymax, PETSC_DRAW_BLACK));
      xmin = 0.0;
      xmax = dd->M - 1;
      PetscCall(PetscDrawLine(draw, xmin, ymin, xmax, ymin, PETSC_DRAW_BLACK));
      PetscCall(PetscDrawLine(draw, xmin, ymax, xmax, ymax, PETSC_DRAW_BLACK));
    }
    PetscDrawCollectiveEnd(draw);
    PetscCall(PetscDrawFlush(draw));
    PetscCall(PetscDrawPause(draw));

    PetscDrawCollectiveBegin(draw);
    /* draw my box */
    ymin = 0;
    ymax = 0.3;
    xmin = dd->xs / dd->w;
    xmax = (dd->xe / dd->w) - 1;
    PetscCall(PetscDrawLine(draw, xmin, ymin, xmax, ymin, PETSC_DRAW_RED));
    PetscCall(PetscDrawLine(draw, xmin, ymin, xmin, ymax, PETSC_DRAW_RED));
    PetscCall(PetscDrawLine(draw, xmin, ymax, xmax, ymax, PETSC_DRAW_RED));
    PetscCall(PetscDrawLine(draw, xmax, ymin, xmax, ymax, PETSC_DRAW_RED));
    /* Put in index numbers */
    base = dd->base / dd->w;
    for (x = xmin; x <= xmax; x++) {
      PetscCall(PetscSNPrintf(node, sizeof(node), "%d", (int)base++));
      PetscCall(PetscDrawString(draw, x, ymin, PETSC_DRAW_RED, node));
    }
    PetscDrawCollectiveEnd(draw);
    PetscCall(PetscDrawFlush(draw));
    PetscCall(PetscDrawPause(draw));
    PetscCall(PetscDrawSave(draw));
  } else if (isglvis) {
    PetscCall(DMView_DA_GLVis(da, viewer));
  } else if (isbinary) {
    PetscCall(DMView_DA_Binary(da, viewer));
#if defined(PETSC_HAVE_MATLAB)
  } else if (ismatlab) {
    PetscCall(DMView_DA_Matlab(da, viewer));
#endif
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSetUp_DA_1D(DM da)
{
  DM_DA          *dd    = (DM_DA *)da->data;
  const PetscInt  M     = dd->M;
  const PetscInt  dof   = dd->w;
  const PetscInt  s     = dd->s;
  const PetscInt  sDist = s; /* stencil distance in points */
  const PetscInt *lx    = dd->lx;
  DMBoundaryType  bx    = dd->bx;
  MPI_Comm        comm;
  Vec             local, global;
  VecScatter      gtol;
  IS              to, from;
  PetscBool       flg1 = PETSC_FALSE, flg2 = PETSC_FALSE;
  PetscMPIInt     rank, size;
  PetscInt        i, *idx, nn, left, xs, xe, x, Xs, Xe, start, m, IXs, IXe;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)da, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));

  dd->p = 1;
  dd->n = 1;
  dd->m = size;
  m     = dd->m;

  if (s > 0) {
    /* if not communicating data then should be ok to have nothing on some processes */
    PetscCheck(M >= m, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "More processes than data points! %" PetscInt_FMT " %" PetscInt_FMT, m, M);
    PetscCheck((M - 1) >= s || size <= 1, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Array is too small for stencil! %" PetscInt_FMT " %" PetscInt_FMT, M - 1, s);
  }

  /*
     Determine locally owned region
     xs is the first local node number, x is the number of local nodes
  */
  if (!lx) {
    PetscCall(PetscMalloc1(m, &dd->lx));
    PetscCall(PetscOptionsGetBool(((PetscObject)da)->options, ((PetscObject)da)->prefix, "-da_partition_blockcomm", &flg1, NULL));
    PetscCall(PetscOptionsGetBool(((PetscObject)da)->options, ((PetscObject)da)->prefix, "-da_partition_nodes_at_end", &flg2, NULL));
    if (flg1) { /* Block Comm type Distribution */
      xs = rank * M / m;
      x  = (rank + 1) * M / m - xs;
    } else if (flg2) { /* The odd nodes are evenly distributed across last nodes */
      x = (M + rank) / m;
      if (M / m == x) xs = rank * x;
      else xs = rank * (x - 1) + (M + rank) % (x * m);
    } else { /* The odd nodes are evenly distributed across the first k nodes */
      /* Regular PETSc Distribution */
      x = M / m + ((M % m) > rank);
      if (rank >= (M % m)) xs = (rank * (PetscInt)(M / m) + M % m);
      else xs = rank * (PetscInt)(M / m) + rank;
    }
    PetscCallMPI(MPI_Allgather(&xs, 1, MPIU_INT, dd->lx, 1, MPIU_INT, comm));
    for (i = 0; i < m - 1; i++) dd->lx[i] = dd->lx[i + 1] - dd->lx[i];
    dd->lx[m - 1] = M - dd->lx[m - 1];
  } else {
    x  = lx[rank];
    xs = 0;
    for (i = 0; i < rank; i++) xs += lx[i];
    /* verify that data user provided is consistent */
    left = xs;
    for (i = rank; i < size; i++) left += lx[i];
    PetscCheck(left == M, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Sum of lx across processors not equal to M %" PetscInt_FMT " %" PetscInt_FMT, left, M);
  }

  /*
   check if the scatter requires more than one process neighbor or wraps around
   the domain more than once
  */
  PetscCheck((x >= s) || ((M <= 1) && (bx != DM_BOUNDARY_PERIODIC)), PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Local x-width of domain x %" PetscInt_FMT " is smaller than stencil width s %" PetscInt_FMT, x, s);

  xe = xs + x;

  /* determine ghost region (Xs) and region scattered into (IXs)  */
  if (xs - sDist > 0) {
    Xs  = xs - sDist;
    IXs = xs - sDist;
  } else {
    if (bx) Xs = xs - sDist;
    else Xs = 0;
    IXs = 0;
  }
  if (xe + sDist <= M) {
    Xe  = xe + sDist;
    IXe = xe + sDist;
  } else {
    if (bx) Xe = xe + sDist;
    else Xe = M;
    IXe = M;
  }

  if (bx == DM_BOUNDARY_PERIODIC || bx == DM_BOUNDARY_MIRROR) {
    Xs  = xs - sDist;
    Xe  = xe + sDist;
    IXs = xs - sDist;
    IXe = xe + sDist;
  }

  /* allocate the base parallel and sequential vectors */
  dd->Nlocal = dof * x;
  PetscCall(VecCreateMPIWithArray(comm, dof, dd->Nlocal, PETSC_DECIDE, NULL, &global));
  dd->nlocal = dof * (Xe - Xs);
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, dof, dd->nlocal, NULL, &local));

  PetscCall(VecGetOwnershipRange(global, &start, NULL));

  /* Create Global to Local Vector Scatter Context */
  /* global to local must retrieve ghost points */
  PetscCall(ISCreateStride(comm, dof * (IXe - IXs), dof * (IXs - Xs), 1, &to));

  PetscCall(PetscMalloc1(x + 2 * sDist, &idx));

  for (i = 0; i < IXs - Xs; i++) idx[i] = -1; /* prepend with -1s if needed for ghosted case*/

  nn = IXs - Xs;
  if (bx == DM_BOUNDARY_PERIODIC) { /* Handle all cases with periodic first */
    for (i = 0; i < sDist; i++) {   /* Left ghost points */
      if ((xs - sDist + i) >= 0) idx[nn++] = xs - sDist + i;
      else idx[nn++] = M + (xs - sDist + i);
    }

    for (i = 0; i < x; i++) idx[nn++] = xs + i; /* Non-ghost points */

    for (i = 0; i < sDist; i++) { /* Right ghost points */
      if ((xe + i) < M) idx[nn++] = xe + i;
      else idx[nn++] = (xe + i) - M;
    }
  } else if (bx == DM_BOUNDARY_MIRROR) { /* Handle all cases with periodic first */
    for (i = 0; i < (sDist); i++) {      /* Left ghost points */
      if ((xs - sDist + i) >= 0) idx[nn++] = xs - sDist + i;
      else idx[nn++] = sDist - i;
    }

    for (i = 0; i < x; i++) idx[nn++] = xs + i; /* Non-ghost points */

    for (i = 0; i < (sDist); i++) { /* Right ghost points */
      if ((xe + i) < M) idx[nn++] = xe + i;
      else idx[nn++] = M - (i + 2);
    }
  } else { /* Now do all cases with no periodicity */
    if (0 <= xs - sDist) {
      for (i = 0; i < sDist; i++) idx[nn++] = xs - sDist + i;
    } else {
      for (i = 0; i < xs; i++) idx[nn++] = i;
    }

    for (i = 0; i < x; i++) idx[nn++] = xs + i;

    if ((xe + sDist) <= M) {
      for (i = 0; i < sDist; i++) idx[nn++] = xe + i;
    } else {
      for (i = xe; i < M; i++) idx[nn++] = i;
    }
  }

  PetscCall(ISCreateBlock(comm, dof, nn - IXs + Xs, &idx[IXs - Xs], PETSC_USE_POINTER, &from));
  PetscCall(VecScatterCreate(global, from, local, to, &gtol));
  PetscCall(ISDestroy(&to));
  PetscCall(ISDestroy(&from));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));

  dd->xs = dof * xs;
  dd->xe = dof * xe;
  dd->ys = 0;
  dd->ye = 1;
  dd->zs = 0;
  dd->ze = 1;
  dd->Xs = dof * Xs;
  dd->Xe = dof * Xe;
  dd->Ys = 0;
  dd->Ye = 1;
  dd->Zs = 0;
  dd->Ze = 1;

  dd->gtol      = gtol;
  dd->base      = dof * xs;
  da->ops->view = DMView_DA_1d;

  /*
     Set the local to global ordering in the global vector, this allows use
     of VecSetValuesLocal().
  */
  for (i = 0; i < Xe - IXe; i++) idx[nn++] = -1; /* pad with -1s if needed for ghosted case*/

  PetscCall(ISLocalToGlobalMappingCreate(comm, dof, nn, idx, PETSC_OWN_POINTER, &da->ltogmap));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   DMDACreate1d - Creates an object that will manage the communication of  one-dimensional
   regular array data that is distributed across some processors.

   Collective

   Input Parameters:
+  comm - MPI communicator
.  bx - type of ghost cells at the boundary the array should have, if any. Use
          `DM_BOUNDARY_NONE`, `DM_BOUNDARY_GHOSTED`, or `DM_BOUNDARY_PERIODIC`.
.  M - global dimension of the array (that is the number of grid points)
            from the command line with -da_grid_x <M>)
.  dof - number of degrees of freedom per node
.  s - stencil width
-  lx - array containing number of nodes in the X direction on each processor,
        or NULL. If non-null, must be of length as the number of processes in the MPI_Comm.
        The sum of these entries must equal M

   Output Parameter:
.  da - the resulting distributed array object

   Options Database Keys:
+  -dm_view - Calls `DMView()` at the conclusion of `DMDACreate1d()`
.  -da_grid_x <nx> - number of grid points in x direction
.  -da_refine_x <rx> - refinement factor
-  -da_refine <n> - refine the `DMDA` n times before creating it

   Level: beginner

   Notes:
   The array data itself is NOT stored in the `DMDA`, it is stored in `Vec` objects;
   The appropriate vector objects can be obtained with calls to `DMCreateGlobalVector()`
   and `DMCreateLocalVector()` and calls to `VecDuplicate()` if more are needed.

   You must call `DMSetUp()` after this call before using this `DM`.

   If you wish to use the options database to change values in the `DMDA` call `DMSetFromOptions()` after this call
   but before `DMSetUp()`.

.seealso: `DMDA`, `DM`, `DMDestroy()`, `DMView()`, `DMDACreate2d()`, `DMDACreate3d()`, `DMGlobalToLocalBegin()`, `DMDASetRefinementFactor()`,
          `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMLocalToLocalBegin()`, `DMLocalToLocalEnd()`, `DMDAGetRefinementFactor()`,
          `DMDAGetInfo()`, `DMCreateGlobalVector()`, `DMCreateLocalVector()`, `DMDACreateNaturalVector()`, `DMLoad()`, `DMDAGetOwnershipRanges()`,
          `DMStagCreate1d()`
@*/
PetscErrorCode DMDACreate1d(MPI_Comm comm, DMBoundaryType bx, PetscInt M, PetscInt dof, PetscInt s, const PetscInt lx[], DM *da)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(DMDACreate(comm, da));
  PetscCall(DMSetDimension(*da, 1));
  PetscCall(DMDASetSizes(*da, M, 1, 1));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMDASetNumProcs(*da, size, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(DMDASetBoundaryType(*da, bx, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetDof(*da, dof));
  PetscCall(DMDASetStencilWidth(*da, s));
  PetscCall(DMDASetOwnershipRanges(*da, lx, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
