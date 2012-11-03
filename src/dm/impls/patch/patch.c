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
PetscErrorCode DMPatchZoom(DM dm, PetscInt rank, MPI_Comm commz, DM *dmz, PetscSF *sfz, PetscSF *sfzr)
{
#if 0
  PetscInt        dim, dof, sw;
  DMDAStencilType st;
  PetscErrorCode  ierr;
#endif 

  PetscFunctionBegin;
#if 0
  if (commz == MPI_COMM_NULL) {
    /* Split communicator */
    SETERRQ(((PetscObject) dm)->comm, PETSC_ERR_SUP, "Not implemented");
  }
  /* Create patch DM */
  ierr = DMDAGetDim(dm, &dim);CHKERRQ(ierr);
  ierr = DMDAGetDof(dm, &dof);CHKERRQ(ierr);
  ierr = DMDAGetStencilType(dm, &st);CHKERRQ(ierr);
  ierr = DMDAGetStencilWidth(dm, &sw);CHKERRQ(ierr);

  ierr = DMDACreate(commz, dmz);CHKERRQ(ierr);
  ierr = DMDASetDim(*dmz, dim);CHKERRQ(ierr);
  ierr = DMDASetSizes(*da, M, N, 1);CHKERRQ(ierr);
  ierr = DMDASetNumProcs(*dmz, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = DMDASetBoundaryType(*dmz, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
  ierr = DMDASetDof(*dmz, dof);CHKERRQ(ierr);
  ierr = DMDASetStencilType(*dmz, st);CHKERRQ(ierr);
  ierr = DMDASetStencilWidth(*dmz, sw);CHKERRQ(ierr);
  ierr = DMDASetOwnershipRanges(*dmz, PETSC_NULL, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dmz);CHKERRQ(ierr);
  ierr = DMSetUp(*dmz);CHKERRQ(ierr);
  /* Create SF for ghosted map */
  /* Create SF for restricted map */
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchSolve"
PetscErrorCode DMPatchSolve(DM dm)
{
#if 0
  MPI_Comm       comm = ((PetscObject) dm)->comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
#endif

  PetscFunctionBegin;
#if 0
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  for(r = 0; r < size; ++r) {
    DM  dmz, dmf;
    Mat interpz;

    ierr = DMPatchZoom(dm, r, comm, &dmz);CHKERRQ(ierr);
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
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchView_Ascii"
PetscErrorCode DMPatchView_Ascii(DM dm, PetscViewer viewer)
{
  DM_Patch         *mesh = (DM_Patch *) dm->data;
  PetscViewerFormat format;
  PetscInt          p;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  /* if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) */
  ierr = PetscViewerASCIIPrintf(viewer, "%D patches\n", mesh->numPatches);CHKERRQ(ierr);
  for(p = 0; p < mesh->numPatches; ++p) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Patch %D\n", p);CHKERRQ(ierr);
    ierr = DMView(mesh->patches[p], viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
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
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--mesh->refct > 0) {PetscFunctionReturn(0);}
  for(p = 0; p < mesh->numPatches; ++p) {
    ierr = DMDestroy(&mesh->patches[p]);CHKERRQ(ierr);
  }
  ierr = PetscFree(mesh->patches);CHKERRQ(ierr);
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
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  for(p = 0; p < mesh->numPatches; ++p) {
    ierr = DMSetUp(mesh->patches[p]);CHKERRQ(ierr);
  }
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
  ierr = DMCreateGlobalVector(mesh->patches[mesh->activePatch], g);CHKERRQ(ierr);
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
  ierr = DMCreateLocalVector(mesh->patches[mesh->activePatch], l);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchGetActivePatch"
PetscErrorCode DMPatchGetActivePatch(DM dm, PetscInt *patch)
{
  DM_Patch *mesh;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(patch, 2);
  mesh = (DM_Patch *) dm->data;
  *patch = mesh->activePatch;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchSetActivePatch"
PetscErrorCode DMPatchSetActivePatch(DM dm, PetscInt patch)
{
  DM_Patch *mesh;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  mesh = (DM_Patch *) dm->data;
  mesh->activePatch = patch;
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
