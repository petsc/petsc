#include <petsc-private/patchimpl.h>   /*I      "petscdmpatch.h"   I*/
#include <petscdmda.h>

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

#undef __FUNCT__
#define __FUNCT__ "DMPatchZoom"
/*
  DMPatchZoom - Create a version of the coarse patch (identified by rank) with halo on communicator commz

  Collective on DM

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
PetscErrorCode DMPatchZoom(DM dm, MatStencil lower, MatStencil upper, MPI_Comm commz, DM *dmz, PetscSF *sfz, PetscSF *sfzr)
{
  DMDAStencilType st;
  MatStencil      blower, bupper;
  PetscInt       *localPoints;
  PetscSFNode    *remotePoints;
  PetscInt        dim, dof;
  PetscInt        M, N, P, rM, rN, rP, halo = 1, sx, sy, sz, sxb, syb, szb, sxr, syr, szr, mxb, myb, mzb, mxr, myr, mzr, i, j, k, q;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (commz == MPI_COMM_NULL) {
    /* Split communicator */
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
  }
  /* Create patch DM */
  ierr = DMDAGetInfo(dm, &dim, &M, &N, &P, 0,0,0, &dof, 0,0,0,0, &st);CHKERRQ(ierr);

  /* Get piece for rank r, expanded by halo */
  bupper.i = PetscMin(M, upper.i + halo); blower.i = PetscMax(lower.i - halo, 0);
  bupper.j = PetscMin(N, upper.j + halo); blower.j = PetscMax(lower.j - halo, 0);
  bupper.k = PetscMin(P, upper.k + halo); blower.k = PetscMax(lower.k - halo, 0);
  rM = bupper.i - blower.i;
  rN = bupper.j - blower.j;
  rP = bupper.k - blower.k;

  ierr = DMDACreate(commz, dmz);CHKERRQ(ierr);
  ierr = DMDASetDim(*dmz, dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(*dmz, rM, rN, rP);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*dmz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*dmz, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*dmz, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*dmz, st);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*dmz, 0);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*dmz, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dmz);CHKERRQ(ierr);
  ierr = DMSetUp(*dmz);CHKERRQ(ierr);
  ierr = DMDAGetCorners(*dmz, &sx, &sy, &sz, &mxb, &myb, &mzb);CHKERRQ(ierr);

  ierr = PetscMalloc2(rM*rN*rP,PetscInt,&localPoints,rM*rN*rP,PetscSFNode,&remotePoints);CHKERRQ(ierr);

  /* Create SF for restricted map */
  q = 0;
  szr = lower.k + sz; syr = lower.j + sy; sxr = lower.i + sx;
  /* This is not quite right. Only works for serial patches */
  mzr = mzb - (bupper.k-upper.k) - (lower.k-blower.k);
  myr = myb - (bupper.j-upper.j) - (lower.j-blower.j);
  mxr = mxb - (bupper.i-upper.i) - (lower.i-blower.i);
  for(k = szr; k < szr+mzr; ++k) {
    for(j = syr; j < syr+myr; ++j) {
      for(i = sxr; i < sxr+mxr; ++i, ++q) {
        const PetscInt lp = ((k-szr)*rN + (j-syr))*rM + i-sxr;
        const PetscInt gp = (k*N + j)*M + i;

        localPoints[q]        = lp;
        /* Replace with a function that can figure this out */
        remotePoints[q].rank  = 0;
        remotePoints[q].index = gp;
      }
    }
  }
  ierr = PetscSFCreate(commz, sfzr);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfzr, M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES);CHKERRQ(ierr);

  /* Create SF for buffered map */
  q = 0;
  szb = blower.k + sz; syb = blower.j + sy; sxb = blower.i + sx;
  for(k = szb; k < szb+mzb; ++k) {
    for(j = syb; j < syb+myb; ++j) {
      for(i = sxb; i < sxb+mxb; ++i, ++q) {
        const PetscInt lp = ((k-szb)*rN + (j-syb))*rM + i-sxb;
        const PetscInt gp = (k*N + j)*M + i;

        localPoints[q]        = lp;
        /* Replace with a function that can figure this out */
        remotePoints[q].rank  = 0;
        remotePoints[q].index = gp;
      }
    }
  }
  ierr = PetscSFCreate(commz, sfz);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*sfz, M*N*P, q, localPoints, PETSC_COPY_VALUES, remotePoints, PETSC_COPY_VALUES);CHKERRQ(ierr);

  ierr = PetscFree2(localPoints, remotePoints);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchSolve"
PetscErrorCode DMPatchSolve(DM dm)
{
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  DM             cdm  = mesh->dmCoarse;
  DM             dmz;
  PetscSF        sfz, sfzr;
  MatStencil     patchSize, lower, upper;
  PetscInt       M, N, P, i, j, k, p = 0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPatchGetPatchSize(dm, &patchSize);CHKERRQ(ierr);
  ierr = DMDAGetInfo(cdm, 0, &M, &N, &P, 0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  M    = PetscMax(M, 1);
  N    = PetscMax(N, 1);
  P    = PetscMax(P, 1);
  for(k = 0; k < P; k += PetscMax(patchSize.k, 1)) {
    for(j = 0; j < N; j += PetscMax(patchSize.j, 1)) {
      for(i = 0; i < M; i += PetscMax(patchSize.i, 1), ++p) {
        lower.i = i; lower.j = j; lower.k = k;
        upper.i = i + patchSize.i; upper.j = j + patchSize.j; upper.k = k + patchSize.k;
        ierr = DMPatchZoom(cdm, lower, upper, comm, &dmz, &sfz, &sfzr);CHKERRQ(ierr);

        ierr = PetscPrintf(comm, "Patch %d: (%d, %d, %d)--(%d, %d, %d)\n", p, lower.i, lower.j, lower.k, upper.i, upper.j, upper.k);CHKERRQ(ierr);
        ierr = DMView(dmz, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscSFView(sfz,  PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = PetscSFView(sfzr, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#if 0
    DM  dmf;
    Mat interpz;
    /* Scatter Xcoarse -> Xzoom */
    ierr = PetscSFBcastBegin(sfz, MPIU_SCALAR, xcarray, xzarray);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfz, MPIU_SCALAR, xcarray, xzarray);CHKERRQ(ierr);
    /* Interpolate Xzoom -> Xfine, note that this may be on subcomms */
    ierr = DMRefine(dmz, MPI_COMM_NULL, &dmf);CHKERRQ(ierr);
    ierr = DMCreateInterpolation(dmz, dmf, &interpz, PETSC_NULL);CHKERRQ(ierr);
    ierr = DMInterpolate(dmz, interpz, dmf);CHKERRQ(ierr);
    /* Smooth Xfine using two-step smoother, normal smoother plus Kaczmarz---moves back and forth from dmzoom to dmfine */
    /* Compute residual Rfine */
    /* Restrict Rfine to Rzoom_restricted */
    /* Scatter Rzoom_restricted -> Rcoarse_restricted */
    ierr = PetscSFBcastBegin(sfzr, MPIU_SCALAR, xzarray, xcarray, MPI_SUM);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sfzr, MPIU_SCALAR, xzarray, xcarray, MPI_SUM);CHKERRQ(ierr);
    /* Compute global residual Rcoarse */
    /* TauCoarse = Rcoarse - Rcoarse_restricted */
#endif
        ierr = PetscSFDestroy(&sfz);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&sfzr);CHKERRQ(ierr);
        ierr = DMDestroy(&dmz);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchView_Ascii"
PetscErrorCode DMPatchView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Patch         *mesh = (DM_Patch *) dm->data;
  PetscViewerFormat format;
  const char       *name;
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

#undef __FUNCT__
#define __FUNCT__ "DMView_Patch"
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
  } else SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_SUP,"Viewer type %s not supported by this mesh object", ((PetscObject)viewer)->type_name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Patch"
PetscErrorCode DMDestroy_Patch(DM dm)
{
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) {PetscFunctionReturn(0);}
  ierr = DMDestroy(&mesh->dmCoarse);CHKERRQ(ierr);
  /* This was originally freed in DMDestroy(), but that prevents reference counting of backend objects */
  ierr = PetscFree(mesh);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetUp_Patch"
PetscErrorCode DMSetUp_Patch(DM dm)
{
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMSetUp(mesh->dmCoarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Patch"
PetscErrorCode DMCreateGlobalVector_Patch(DM dm, Vec *g)
{
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMCreateGlobalVector(mesh->dmCoarse, g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Patch"
PetscErrorCode DMCreateLocalVector_Patch(DM dm, Vec *l)
{
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMCreateLocalVector(mesh->dmCoarse, l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateSubDM_Patch"
PetscErrorCode DMCreateSubDM_Patch(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Tell me to code this");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchGetCoarse"
PetscErrorCode DMPatchGetCoarse(DM dm, DM *dmCoarse)
{
  DM_Patch *mesh = (DM_Patch *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *dmCoarse = mesh->dmCoarse;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchGetPatchSize"
PetscErrorCode DMPatchGetPatchSize(DM dm, MatStencil *patchSize)
{
  DM_Patch *mesh = (DM_Patch *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(patchSize, 2);
  *patchSize = mesh->patchSize;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchSetPatchSize"
PetscErrorCode DMPatchSetPatchSize(DM dm, MatStencil patchSize)
{
  DM_Patch *mesh = (DM_Patch *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh->patchSize = patchSize;
  PetscFunctionReturn(0);
}
