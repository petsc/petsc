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

/*
  DMPatchZoom - Create a version of the coarse patch (identified by rank) with halo on communicator commz

  Collective on dm

  Input Parameters:
  + dm - the DM
  . rank - the rank which holds the given patch
  - commz - the new communicator for the patch

  Output Parameters:
  + dmz  - the patch DM
  . sfz  - the PetscSF mapping the patch+halo to the zoomed version
  . sfzr - the PetscSF mapping the patch to the restricted zoomed version

  Level: intermediate

  Note: All processes in commz should have the same rank (could autosplit comm)

.seealso: DMPatchSolve()
*/
PetscErrorCode DMPatchZoom(DM dm, Vec X, MatStencil lower, MatStencil upper, MPI_Comm commz, DM *dmz, PetscSF *sfz, PetscSF *sfzr)
{
  DMDAStencilType st;
  MatStencil      blower, bupper, loclower, locupper;
  IS              is;
  const PetscInt  *ranges, *indices;
  PetscInt        *localPoints  = NULL;
  PetscSFNode     *remotePoints = NULL;
  PetscInt        dim, dof;
  PetscInt        M, N, P, rM, rN, rP, halo = 1, sxb, syb, szb, sxr, syr, szr, exr, eyr, ezr, mxb, myb, mzb, i, j, k, q;
  PetscMPIInt     size;
  PetscBool       patchis_offproc = PETSC_TRUE;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size);CHKERRQ(ierr);
  /* Create patch DM */
  ierr = DMDAGetInfo(dm, &dim, &M, &N, &P, NULL,NULL,NULL, &dof, NULL,NULL,NULL,NULL, &st);CHKERRQ(ierr);

  /* Get piece for rank r, expanded by halo */
  bupper.i = PetscMin(M, upper.i + halo); blower.i = PetscMax(lower.i - halo, 0);
  bupper.j = PetscMin(N, upper.j + halo); blower.j = PetscMax(lower.j - halo, 0);
  bupper.k = PetscMin(P, upper.k + halo); blower.k = PetscMax(lower.k - halo, 0);
  rM       = bupper.i - blower.i;
  rN       = bupper.j - blower.j;
  rP       = bupper.k - blower.k;

  if (commz != MPI_COMM_NULL) {
    ierr = DMDACreate(commz, dmz);CHKERRQ(ierr);
    ierr = DMSetDimension(*dmz, dim);CHKERRQ(ierr);
    ierr = DMDASetSizes(*dmz, rM, rN, rP);CHKERRQ(ierr);
    ierr = DMDASetNumProcs(*dmz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(*dmz, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE);CHKERRQ(ierr);
    ierr = DMDASetDof(*dmz, dof);CHKERRQ(ierr);
    ierr = DMDASetStencilType(*dmz, st);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(*dmz, 0);CHKERRQ(ierr);
    ierr = DMDASetOwnershipRanges(*dmz, NULL, NULL, NULL);CHKERRQ(ierr);
    ierr = DMSetFromOptions(*dmz);CHKERRQ(ierr);
    ierr = DMSetUp(*dmz);CHKERRQ(ierr);
    ierr = DMDAGetCorners(*dmz, &sxb, &syb, &szb, &mxb, &myb, &mzb);CHKERRQ(ierr);
    sxr  = PetscMax(sxb,     lower.i - blower.i);
    syr  = PetscMax(syb,     lower.j - blower.j);
    szr  = PetscMax(szb,     lower.k - blower.k);
    exr  = PetscMin(sxb+mxb, upper.i - blower.i);
    eyr  = PetscMin(syb+myb, upper.j - blower.j);
    ezr  = PetscMin(szb+mzb, upper.k - blower.k);
    ierr = PetscMalloc2(rM*rN*rP,&localPoints,rM*rN*rP,&remotePoints);CHKERRQ(ierr);
  } else {
    sxr = syr = szr = exr = eyr = ezr = sxb = syb = szb = mxb = myb = mzb = 0;
  }

  /* Create SF for restricted map */
  ierr = VecGetOwnershipRanges(X,&ranges);CHKERRQ(ierr);

  loclower.i = blower.i + sxr; locupper.i = blower.i + exr;
  loclower.j = blower.j + syr; locupper.j = blower.j + eyr;
  loclower.k = blower.k + szr; locupper.k = blower.k + ezr;

  ierr = DMDACreatePatchIS(dm, &loclower, &locupper, &is, patchis_offproc);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &indices);CHKERRQ(ierr);

  q = 0;
  for (k = szb; k < szb+mzb; ++k) {
    if ((k < szr) || (k >= ezr)) continue;
    for (j = syb; j < syb+myb; ++j) {
      if ((j < syr) || (j >= eyr)) continue;
      for (i = sxb; i < sxb+mxb; ++i) {
        const PetscInt lp = ((k-szb)*rN + (j-syb))*rM + i-sxb;
        PetscInt       r;

        if ((i < sxr) || (i >= exr)) continue;
        localPoints[q]        = lp;
        ierr = PetscFindInt(indices[q], size+1, ranges, &r);CHKERRQ(ierr);

        remotePoints[q].rank  = r < 0 ? -(r+1) - 1 : r;
        remotePoints[q].index = indices[q] - ranges[remotePoints[q].rank];
        ++q;
      }
    }
  }
  ierr = ISRestoreIndices(is, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfzr);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sfzr, "Restricted Map");CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfzr, M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES);CHKERRQ(ierr);

  /* Create SF for buffered map */
  loclower.i = blower.i + sxb; locupper.i = blower.i + sxb+mxb;
  loclower.j = blower.j + syb; locupper.j = blower.j + syb+myb;
  loclower.k = blower.k + szb; locupper.k = blower.k + szb+mzb;

  ierr = DMDACreatePatchIS(dm, &loclower, &locupper, &is, patchis_offproc);CHKERRQ(ierr);
  ierr = ISGetIndices(is, &indices);CHKERRQ(ierr);

  q = 0;
  for (k = szb; k < szb+mzb; ++k) {
    for (j = syb; j < syb+myb; ++j) {
      for (i = sxb; i < sxb+mxb; ++i, ++q) {
        PetscInt r;

        localPoints[q]        = q;
        ierr = PetscFindInt(indices[q], size+1, ranges, &r);CHKERRQ(ierr);
        remotePoints[q].rank  = r < 0 ? -(r+1) - 1 : r;
        remotePoints[q].index = indices[q] - ranges[remotePoints[q].rank];
      }
    }
  }
  ierr = ISRestoreIndices(is, &indices);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm), sfz);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sfz, "Buffered Map");CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfz, M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES);CHKERRQ(ierr);

  ierr = PetscFree2(localPoints, remotePoints);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  ierr = DMPatchGetCoarse(dm, &dmc);CHKERRQ(ierr);
  ierr = DMPatchGetPatchSize(dm, &patchSize);CHKERRQ(ierr);
  ierr = DMPatchGetCommSize(dm, &commSize);CHKERRQ(ierr);
  ierr = DMPatchGetCommSize(dm, &commSize);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dmc, &XC);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dmc, NULL, &M, &N, &P, &l, &m, &n, NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
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

    ierr = MPI_Comm_split(comm, newComm, newRank, &commz);CHKERRQ(ierr);
    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Rank %d color %d key %d commz %d\n", rank, newComm, newRank, *((PetscMPIInt*) &commz));CHKERRQ(ierr);}
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
        MPI_Comm commp = MPI_COMM_NULL;
        DM       dmz   = NULL;
#if 0
        DM  dmf     = NULL;
        Mat interpz = NULL;
#endif
        Vec         XZ       = NULL;
        PetscScalar *xcarray = NULL;
        PetscScalar *xzarray = NULL;

        if ((gridRank.k/commSize.k == p/(l/commSize.i * m/commSize.j) % n/commSize.k) &&
            (gridRank.j/commSize.j == p/(l/commSize.i)                % m/commSize.j) &&
            (gridRank.i/commSize.i == p                               % l/commSize.i)) {
          if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "Rank %d is accepting Patch %d\n", rank, p);CHKERRQ(ierr);}
          commp = commz;
        }
        /* Zoom to coarse patch */
        lower.i = i; lower.j = j; lower.k = k;
        upper.i = i + patchSize.i; upper.j = j + patchSize.j; upper.k = k + patchSize.k;
        ierr    = DMPatchZoom(dmc, XC, lower, upper, commp, &dmz, &sfz, &sfzr);CHKERRQ(ierr);
        lower.c = 0; /* initialize member, otherwise compiler issues warnings */
        upper.c = 0; /* initialize member, otherwise compiler issues warnings */
        /* Debug */
        ierr = PetscPrintf(comm, "Patch %d: (%d, %d, %d)--(%d, %d, %d)\n", p, lower.i, lower.j, lower.k, upper.i, upper.j, upper.k);CHKERRQ(ierr);
        if (dmz) {ierr = DMView(dmz, PETSC_VIEWER_STDOUT_(commz));CHKERRQ(ierr);}
        ierr = PetscSFView(sfz,  PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
        ierr = PetscSFView(sfzr, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);
        /* Scatter Xcoarse -> Xzoom */
        if (dmz) {ierr = DMGetGlobalVector(dmz, &XZ);CHKERRQ(ierr);}
        if (XZ)  {ierr = VecGetArray(XZ, &xzarray);CHKERRQ(ierr);}
        ierr = VecGetArray(XC, &xcarray);CHKERRQ(ierr);
        ierr = PetscSFBcastBegin(sfz, MPIU_SCALAR, xcarray, xzarray);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(sfz, MPIU_SCALAR, xcarray, xzarray);CHKERRQ(ierr);
        ierr = VecRestoreArray(XC, &xcarray);CHKERRQ(ierr);
        if (XZ)  {ierr = VecRestoreArray(XZ, &xzarray);CHKERRQ(ierr);}
#if 0
        /* Interpolate Xzoom -> Xfine, note that this may be on subcomms */
        ierr = DMRefine(dmz, MPI_COMM_NULL, &dmf);CHKERRQ(ierr);
        ierr = DMCreateInterpolation(dmz, dmf, &interpz, NULL);CHKERRQ(ierr);
        ierr = DMInterpolate(dmz, interpz, dmf);CHKERRQ(ierr);
        /* Smooth Xfine using two-step smoother, normal smoother plus Kaczmarz---moves back and forth from dmzoom to dmfine */
        /* Compute residual Rfine */
        /* Restrict Rfine to Rzoom_restricted */
#endif
        /* Scatter Rzoom_restricted -> Rcoarse_restricted */
        if (XZ)  {ierr = VecGetArray(XZ, &xzarray);CHKERRQ(ierr);}
        ierr = VecGetArray(XC, &xcarray);CHKERRQ(ierr);
        ierr = PetscSFReduceBegin(sfzr, MPIU_SCALAR, xzarray, xcarray, MPIU_SUM);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(sfzr, MPIU_SCALAR, xzarray, xcarray, MPIU_SUM);CHKERRQ(ierr);
        ierr = VecRestoreArray(XC, &xcarray);CHKERRQ(ierr);
        if (XZ)  {ierr = VecRestoreArray(XZ, &xzarray);CHKERRQ(ierr);}
        if (dmz) {ierr = DMRestoreGlobalVector(dmz, &XZ);CHKERRQ(ierr);}
        /* Compute global residual Rcoarse */
        /* TauCoarse = Rcoarse - Rcoarse_restricted */

        ierr = PetscSFDestroy(&sfz);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&sfzr);CHKERRQ(ierr);
        ierr = DMDestroy(&dmz);CHKERRQ(ierr);
      }
    }
  }
  ierr = DMRestoreGlobalVector(dmc, &XC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Patch          *mesh = (DM_Patch*) dm->data;
  PetscViewerFormat format;
  const char        *name;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  /* if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) */
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Patch DM %s\n", name);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "Coarse DM\n");CHKERRQ(ierr);
  ierr = DMView(mesh->dmCoarse, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMView_Patch(DM dm, PetscViewer viewer)
{
  PetscBool      iascii, isbinary;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERBINARY, &isbinary);CHKERRQ(ierr);
  if (iascii) {
    ierr = DMPatchView_Ascii(dm, viewer);CHKERRQ(ierr);
#if 0
  } else if (isbinary) {
    ierr = DMPatchView_Binary(dm, viewer);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMDestroy_Patch(DM dm)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) PetscFunctionReturn(0);
  ierr = DMDestroy(&mesh->dmCoarse);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMSetUp_Patch(DM dm)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMSetUp(mesh->dmCoarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateGlobalVector_Patch(DM dm, Vec *g)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMCreateGlobalVector(mesh->dmCoarse, g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateLocalVector_Patch(DM dm, Vec *l)
{
  DM_Patch       *mesh = (DM_Patch*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMCreateLocalVector(mesh->dmCoarse, l);CHKERRQ(ierr);
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
