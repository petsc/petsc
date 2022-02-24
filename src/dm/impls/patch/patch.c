#include <petsc/private/dmpatchimpl.h>   /*I      "petscdmpatch.h"   I*/
#include <petscdmda.h>
#include <petscsf.h>

/*
Solver loop to update \tau:

  DMZoom(dmc, &dmz)
  DMRefine(dmz, &dmf),
  Scatter Xcoarse -> Xzoom,
  Interpolate Xzoom -> Xfine (note that this may be on subcomms),
  Smooth Xfine using two-step smoother
    normal smoother plus Kaczmarz---moves back and forth from dmzoom to dmfine
  Compute residual Rfine
  Restrict Rfine to Rzoom_restricted
  Scatter Rzoom_restricted -> Rcoarse_restricted
  Compute global residual Rcoarse
  TauCoarse = Rcoarse - Rcoarse_restricted
*/

/*@C
  DMPatchZoom - Create patches of a DMDA on subsets of processes, indicated by commz

  Collective on dm

  Input Parameters:
  + dm - the DM
  . lower,upper - the upper right corner and the lower left corner of the requested patch
  - commz - the new communicator for the patch, MPI_COMM_NULL indicates that the given rank will not own a patch

  Output Parameters:
  + dmz  - the patch DM
  . sfz  - the PetscSF mapping the patch+halo to the zoomed version (optional)
  - sfzr - the PetscSF mapping the patch to the restricted zoomed version

  Level: intermediate

.seealso: DMPatchSolve(), DMDACreatePatchIS()
@*/
PetscErrorCode DMPatchZoom(DM dm, MatStencil lower, MatStencil upper, MPI_Comm commz, DM *dmz, PetscSF *sfz, PetscSF *sfzr)
{
  DMDAStencilType st;
  MatStencil      blower, bupper, loclower, locupper;
  IS              is;
  const PetscInt  *ranges, *indices;
  PetscInt        *localPoints  = NULL;
  PetscSFNode     *remotePoints = NULL;
  PetscInt        dim, dof;
  PetscInt        M, N, P, rM, rN, rP, halo = 1, sxb, syb, szb, sxr, syr, szr, exr, eyr, ezr, mxb, myb, mzb, i, j, k, l, q;
  PetscMPIInt     size;
  PetscBool       patchis_offproc = PETSC_TRUE;
  Vec             X;

  PetscFunctionBegin;
  if (!sfz) halo = 0;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  /* Create patch DM */
  CHKERRQ(DMDAGetInfo(dm, &dim, &M, &N, &P, NULL,NULL,NULL, &dof, NULL,NULL,NULL,NULL, &st));

  /* Get piece for rank r, expanded by halo */
  bupper.i = PetscMin(M, upper.i + halo); blower.i = PetscMax(lower.i - halo, 0);
  bupper.j = PetscMin(N, upper.j + halo); blower.j = PetscMax(lower.j - halo, 0);
  bupper.k = PetscMin(P, upper.k + halo); blower.k = PetscMax(lower.k - halo, 0);
  rM       = bupper.i - blower.i;
  rN       = bupper.j - blower.j;
  rP       = bupper.k - blower.k;

  if (commz != MPI_COMM_NULL) {
    CHKERRQ(DMDACreate(commz, dmz));
    CHKERRQ(DMSetDimension(*dmz, dim));
    CHKERRQ(DMDASetSizes(*dmz, rM, rN, rP));
    CHKERRQ(DMDASetNumProcs(*dmz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE));
    CHKERRQ(DMDASetBoundaryType(*dmz, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
    CHKERRQ(DMDASetDof(*dmz, dof));
    CHKERRQ(DMDASetStencilType(*dmz, st));
    CHKERRQ(DMDASetStencilWidth(*dmz, 0));
    CHKERRQ(DMDASetOwnershipRanges(*dmz, NULL, NULL, NULL));
    CHKERRQ(DMSetFromOptions(*dmz));
    CHKERRQ(DMSetUp(*dmz));
    CHKERRQ(DMDAGetCorners(*dmz, &sxb, &syb, &szb, &mxb, &myb, &mzb));
    sxr  = PetscMax(sxb,     lower.i - blower.i);
    syr  = PetscMax(syb,     lower.j - blower.j);
    szr  = PetscMax(szb,     lower.k - blower.k);
    exr  = PetscMin(sxb+mxb, upper.i - blower.i);
    eyr  = PetscMin(syb+myb, upper.j - blower.j);
    ezr  = PetscMin(szb+mzb, upper.k - blower.k);
    CHKERRQ(PetscMalloc2(dof*rM*rN*PetscMax(rP,1),&localPoints,dof*rM*rN*PetscMax(rP,1),&remotePoints));
  } else {
    sxr = syr = szr = exr = eyr = ezr = sxb = syb = szb = mxb = myb = mzb = 0;
  }

  /* Create SF for restricted map */
  CHKERRQ(DMCreateGlobalVector(dm,&X));
  CHKERRQ(VecGetOwnershipRanges(X,&ranges));

  loclower.i = blower.i + sxr; locupper.i = blower.i + exr;
  loclower.j = blower.j + syr; locupper.j = blower.j + eyr;
  loclower.k = blower.k + szr; locupper.k = blower.k + ezr;

  CHKERRQ(DMDACreatePatchIS(dm, &loclower, &locupper, &is, patchis_offproc));
  CHKERRQ(ISGetIndices(is, &indices));

  if (dim < 3) {mzb = 1; ezr = 1;}
  q = 0;
  for (k = szb; k < szb+mzb; ++k) {
    if ((k < szr) || (k >= ezr)) continue;
    for (j = syb; j < syb+myb; ++j) {
      if ((j < syr) || (j >= eyr)) continue;
      for (i = sxb; i < sxb+mxb; ++i) {
        for (l=0; l<dof; l++) {
          const PetscInt lp = l + dof*(((k-szb)*rN + (j-syb))*rM + i-sxb);
          PetscInt       r;

          if ((i < sxr) || (i >= exr)) continue;
          localPoints[q]        = lp;
          CHKERRQ(PetscFindInt(indices[q], size+1, ranges, &r));

          remotePoints[q].rank  = r < 0 ? -(r+1) - 1 : r;
          remotePoints[q].index = indices[q] - ranges[remotePoints[q].rank];
          ++q;
        }
      }
    }
  }
  CHKERRQ(ISRestoreIndices(is, &indices));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)dm), sfzr));
  CHKERRQ(PetscObjectSetName((PetscObject) *sfzr, "Restricted Map"));
  CHKERRQ(PetscSFSetGraph(*sfzr, dof*M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES));

  if (sfz) {
    /* Create SF for buffered map */
    loclower.i = blower.i + sxb; locupper.i = blower.i + sxb+mxb;
    loclower.j = blower.j + syb; locupper.j = blower.j + syb+myb;
    loclower.k = blower.k + szb; locupper.k = blower.k + szb+mzb;

    CHKERRQ(DMDACreatePatchIS(dm, &loclower, &locupper, &is, patchis_offproc));
    CHKERRQ(ISGetIndices(is, &indices));

    q = 0;
    for (k = szb; k < szb+mzb; ++k) {
      for (j = syb; j < syb+myb; ++j) {
        for (i = sxb; i < sxb+mxb; ++i, ++q) {
          PetscInt r;

          localPoints[q]        = q;
          CHKERRQ(PetscFindInt(indices[q], size+1, ranges, &r));
          remotePoints[q].rank  = r < 0 ? -(r+1) - 1 : r;
          remotePoints[q].index = indices[q] - ranges[remotePoints[q].rank];
        }
      }
    }
    CHKERRQ(ISRestoreIndices(is, &indices));
    CHKERRQ(ISDestroy(&is));
    CHKERRQ(PetscSFCreate(PetscObjectComm((PetscObject)dm), sfz));
    CHKERRQ(PetscObjectSetName((PetscObject) *sfz, "Buffered Map"));
    CHKERRQ(PetscSFSetGraph(*sfz, M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES));
  }

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(PetscFree2(localPoints, remotePoints));
  PetscFunctionReturn(0);
}

typedef enum {PATCH_COMM_TYPE_WORLD = 0, PATCH_COMM_TYPE_SELF = 1} PatchCommType;

PetscErrorCode DMPatchSolve(DM dm)
{
  MPI_Comm       comm;
  MPI_Comm       commz;
  DM             dmc;
  PetscSF        sfz, sfzr;
  Vec            XC;
  MatStencil     patchSize, commSize, gridRank, lower, upper;
  PetscInt       M, N, P, i, j, k, l, m, n, p = 0;
  PetscMPIInt    rank, size;
  PetscInt       debug = 0;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(DMPatchGetCoarse(dm, &dmc));
  CHKERRQ(DMPatchGetPatchSize(dm, &patchSize));
  CHKERRQ(DMPatchGetCommSize(dm, &commSize));
  CHKERRQ(DMPatchGetCommSize(dm, &commSize));
  CHKERRQ(DMGetGlobalVector(dmc, &XC));
  CHKERRQ(DMDAGetInfo(dmc, NULL, &M, &N, &P, &l, &m, &n, NULL,NULL,NULL,NULL,NULL,NULL));
  M    = PetscMax(M, 1); l = PetscMax(l, 1);
  N    = PetscMax(N, 1); m = PetscMax(m, 1);
  P    = PetscMax(P, 1); n = PetscMax(n, 1);

  gridRank.i = rank       % l;
  gridRank.j = rank/l     % m;
  gridRank.k = rank/(l*m) % n;

  if (commSize.i*commSize.j*commSize.k == size || commSize.i*commSize.j*commSize.k == 0) {
    commSize.i = l; commSize.j = m; commSize.k = n;
    commz      = comm;
  } else if (commSize.i*commSize.j*commSize.k == 1) {
    commz = PETSC_COMM_SELF;
  } else {
    const PetscMPIInt newComm = ((gridRank.k/commSize.k)*(m/commSize.j) + gridRank.j/commSize.j)*(l/commSize.i) + (gridRank.i/commSize.i);
    const PetscMPIInt newRank = ((gridRank.k%commSize.k)*commSize.j     + gridRank.j%commSize.j)*commSize.i     + (gridRank.i%commSize.i);

    CHKERRMPI(MPI_Comm_split(comm, newComm, newRank, &commz));
    if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Rank %d color %d key %d commz %p\n", rank, newComm, newRank, (void*)(MPI_Aint)commz));
  }
  /*
   Assumptions:
     - patchSize divides gridSize
     - commSize divides gridSize
     - commSize divides l,m,n
   Ignore multiple patches per rank for now

   Multiple ranks per patch:
     - l,m,n divides patchSize
     - commSize divides patchSize
   */
  for (k = 0; k < P; k += PetscMax(patchSize.k, 1)) {
    for (j = 0; j < N; j += PetscMax(patchSize.j, 1)) {
      for (i = 0; i < M; i += PetscMax(patchSize.i, 1), ++p) {
        MPI_Comm    commp = MPI_COMM_NULL;
        DM          dmz   = NULL;
#if 0
        DM          dmf     = NULL;
        Mat         interpz = NULL;
#endif
        Vec         XZ       = NULL;
        PetscScalar *xcarray = NULL;
        PetscScalar *xzarray = NULL;

        if ((gridRank.k/commSize.k == p/(l/commSize.i * m/commSize.j) % n/commSize.k) &&
            (gridRank.j/commSize.j == p/(l/commSize.i)                % m/commSize.j) &&
            (gridRank.i/commSize.i == p                               % l/commSize.i)) {
          if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Rank %d is accepting Patch %d\n", rank, p));
          commp = commz;
        }
        /* Zoom to coarse patch */
        lower.i = i; lower.j = j; lower.k = k;
        upper.i = i + patchSize.i; upper.j = j + patchSize.j; upper.k = k + patchSize.k;
        CHKERRQ(DMPatchZoom(dmc, lower, upper, commp, &dmz, &sfz, &sfzr));
        lower.c = 0; /* initialize member, otherwise compiler issues warnings */
        upper.c = 0; /* initialize member, otherwise compiler issues warnings */
        if (debug) CHKERRQ(PetscPrintf(comm, "Patch %d: (%d, %d, %d)--(%d, %d, %d)\n", p, lower.i, lower.j, lower.k, upper.i, upper.j, upper.k));
        if (dmz) CHKERRQ(DMView(dmz, PETSC_VIEWER_STDOUT_(commz)));
        CHKERRQ(PetscSFView(sfz,  PETSC_VIEWER_STDOUT_(comm)));
        CHKERRQ(PetscSFView(sfzr, PETSC_VIEWER_STDOUT_(comm)));
        /* Scatter Xcoarse -> Xzoom */
        if (dmz) CHKERRQ(DMGetGlobalVector(dmz, &XZ));
        if (XZ)  CHKERRQ(VecGetArray(XZ, &xzarray));
        CHKERRQ(VecGetArray(XC, &xcarray));
        CHKERRQ(PetscSFBcastBegin(sfz, MPIU_SCALAR, xcarray, xzarray,MPI_REPLACE));
        CHKERRQ(PetscSFBcastEnd(sfz, MPIU_SCALAR, xcarray, xzarray,MPI_REPLACE));
        CHKERRQ(VecRestoreArray(XC, &xcarray));
        if (XZ)  CHKERRQ(VecRestoreArray(XZ, &xzarray));
#if 0
        /* Interpolate Xzoom -> Xfine, note that this may be on subcomms */
        CHKERRQ(DMRefine(dmz, MPI_COMM_NULL, &dmf));
        CHKERRQ(DMCreateInterpolation(dmz, dmf, &interpz, NULL));
        CHKERRQ(DMInterpolate(dmz, interpz, dmf));
        /* Smooth Xfine using two-step smoother, normal smoother plus Kaczmarz---moves back and forth from dmzoom to dmfine */
        /* Compute residual Rfine */
        /* Restrict Rfine to Rzoom_restricted */
#endif
        /* Scatter Rzoom_restricted -> Rcoarse_restricted */
        if (XZ)  CHKERRQ(VecGetArray(XZ, &xzarray));
        CHKERRQ(VecGetArray(XC, &xcarray));
        CHKERRQ(PetscSFReduceBegin(sfzr, MPIU_SCALAR, xzarray, xcarray, MPIU_SUM));
        CHKERRQ(PetscSFReduceEnd(sfzr, MPIU_SCALAR, xzarray, xcarray, MPIU_SUM));
        CHKERRQ(VecRestoreArray(XC, &xcarray));
        if (XZ)  CHKERRQ(VecRestoreArray(XZ, &xzarray));
        if (dmz) CHKERRQ(DMRestoreGlobalVector(dmz, &XZ));
        /* Compute global residual Rcoarse */
        /* TauCoarse = Rcoarse - Rcoarse_restricted */

        CHKERRQ(PetscSFDestroy(&sfz));
        CHKERRQ(PetscSFDestroy(&sfzr));
        CHKERRQ(DMDestroy(&dmz));
      }
    }
  }
  CHKERRQ(DMRestoreGlobalVector(dmc, &XC));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchView_ASCII(DM dm, PetscViewer viewer)
{
  DM_Patch          *mesh = (DM_Patch*) dm->data;
  PetscViewerFormat format;
  const char        *name;

  PetscFunctionBegin;
  CHKERRQ(PetscViewerGetFormat(viewer, &format));
  /* if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) */
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Patch DM %s\n", name));
  CHKERRQ(PetscViewerASCIIPushTab(viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "Coarse DM\n"));
  CHKERRQ(DMView(mesh->dmCoarse, viewer));
  CHKERRQ(PetscViewerASCIIPopTab(viewer));
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Patch(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, isbinary;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary));
  if (iascii) {
    CHKERRQ(DMPatchView_ASCII(dm, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Patch(DM dm)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  CHKERRQ(DMDestroy(&mesh->dmCoarse));
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  CHKERRQ(PetscFree(mesh));
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUp_Patch(DM dm)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMSetUp(mesh->dmCoarse));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_Patch(DM dm, Vec *g)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMCreateGlobalVector(mesh->dmCoarse, g));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_Patch(DM dm, Vec *l)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  CHKERRQ(DMCreateLocalVector(mesh->dmCoarse, l));
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_Patch(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Tell me to code this");
}

PetscErrorCode DMPatchGetCoarse(DM dm, DM *dmCoarse)
{
  DM_Patch *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *dmCoarse = mesh->dmCoarse;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchGetPatchSize(DM dm, MatStencil *patchSize)
{
  DM_Patch *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(patchSize, 2);
  *patchSize = mesh->patchSize;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchSetPatchSize(DM dm, MatStencil patchSize)
{
  DM_Patch *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->patchSize = patchSize;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchGetCommSize(DM dm, MatStencil *commSize)
{
  DM_Patch *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(commSize, 2);
  *commSize = mesh->commSize;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchSetCommSize(DM dm, MatStencil commSize)
{
  DM_Patch *mesh = (DM_Patch*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->commSize = commSize;
  PetscFunctionReturn(0);
}
