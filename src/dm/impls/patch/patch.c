#include <petsc-private/patchimpl.h>   /*I      "petscdmpatch.h"   I*/
#include <petscdmda.h>

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
