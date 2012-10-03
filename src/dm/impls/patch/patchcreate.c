#include <petsc-private/patchimpl.h>   /*I      "petscdmpatch.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Patch"
PetscErrorCode DMSetFromOptions_Patch(DM dm)
{
  DM_Patch      *mesh = (DM_Patch *) dm->data;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMPatch Options");CHKERRQ(ierr);
    /* Handle associated vectors */
    /* Handle viewing */
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  for (p = 0; p < mesh->numPatches; ++p) {
    ierr = DMSetFromOptions(mesh->patches[p]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMSetUp_Patch(DM dm);
extern PetscErrorCode DMView_Patch(DM dm, PetscViewer viewer);
extern PetscErrorCode DMCreateGlobalVector_Patch(DM dm, Vec *g);
extern PetscErrorCode DMCreateLocalVector_Patch(DM dm, Vec *l);
extern PetscErrorCode DMDestroy_Patch(DM dm);
extern PetscErrorCode DMCreateSubDM_Patch(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm);

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Patch"
PetscErrorCode DMInitialize_Patch(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(VECSTANDARD,(char**) &dm->vectype);CHKERRQ(ierr);
  dm->ops->view               = DMView_Patch;
  dm->ops->setfromoptions     = DMSetFromOptions_Patch;
  dm->ops->setup              = DMSetUp_Patch;
  dm->ops->createglobalvector = DMCreateGlobalVector_Patch;
  dm->ops->createlocalvector  = DMCreateLocalVector_Patch;
  dm->ops->createlocaltoglobalmapping      = PETSC_NULL;
  dm->ops->createlocaltoglobalmappingblock = PETSC_NULL;
  dm->ops->createfieldis      = PETSC_NULL;
  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix       = 0;
  dm->ops->createinterpolation= 0;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;
  dm->ops->refine             = 0;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;
  dm->ops->globaltolocalbegin = PETSC_NULL;
  dm->ops->globaltolocalend   = PETSC_NULL;
  dm->ops->localtoglobalbegin = PETSC_NULL;
  dm->ops->localtoglobalend   = PETSC_NULL;
  dm->ops->destroy            = DMDestroy_Patch;
  dm->ops->createsubdm        = DMCreateSubDM_Patch;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_Patch"
PetscErrorCode DMCreate_Patch(DM dm)
{
  DM_Patch      *mesh;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_Patch, &mesh);CHKERRQ(ierr);
  dm->data = mesh;

  mesh->refct       = 1;
  mesh->numPatches  = 0;
  mesh->patches     = PETSC_NULL;
  mesh->dmCoarse    = PETSC_NULL;
  mesh->activePatch = -1;

  ierr = DMInitialize_Patch(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMPatchCreate"
/*@
  DMPatchCreate - Creates a DMPatch object, which is a collections of DMs called patches.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMPatch object

  Output Parameter:
. mesh  - The DMPatch object

  Level: beginner

.keywords: DMPatch, create
@*/
PetscErrorCode DMPatchCreate(MPI_Comm comm, DM *mesh)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  ierr = DMCreate(comm, mesh);CHKERRQ(ierr);
  ierr = DMSetType(*mesh, DMPATCH);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPatchCreateGrid"
PetscErrorCode DMPatchCreateGrid(MPI_Comm comm, PetscInt dim, MatStencil patchSize, MatStencil gridSize, DM *dm)
{
  DM_Patch      *mesh;
  PetscInt       p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPatchCreate(comm, dm);CHKERRQ(ierr);
  mesh = (DM_Patch *) (*dm)->data;
  mesh->dim        = dim;
  if (dim < 2) {
    gridSize.j  = 1;
    patchSize.j = 1;
  }
  if (dim < 3) {
    gridSize.k  = 1;
    patchSize.k = 1;
  }
  mesh->numPatches = gridSize.i*gridSize.j*gridSize.k;
  ierr = PetscMalloc(mesh->numPatches * sizeof(DM), &mesh->patches);CHKERRQ(ierr);
  for (p = 0; p < mesh->numPatches; ++p) {
    /* Patches are overlapping: Right now we do not account for boundaries */
    DM da;
    PetscInt dof = 1, width = 1;

    ierr = DMCreate(comm, &da);
    ierr = DMSetType(da, DMDA);
    ierr = DMDASetDim(da, dim);CHKERRQ(ierr);
    ierr = DMDASetSizes(da, patchSize.i+width*2, patchSize.j+width*2, patchSize.k+width*2);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
    ierr = DMDASetDof(da, dof);CHKERRQ(ierr);
    ierr = DMDASetStencilType(da, DMDA_STENCIL_BOX);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(da, width);CHKERRQ(ierr);
    mesh->patches[p] = da;
  }
  {
    DM da;
    PetscInt dof = 1, width = 1;

    ierr = DMCreate(comm, &da);
    ierr = DMSetType(da, DMDA);
    ierr = DMDASetDim(da, dim);CHKERRQ(ierr);
    ierr = DMDASetSizes(da, gridSize.i, gridSize.j, gridSize.k);CHKERRQ(ierr);
    ierr = DMDASetBoundaryType(da, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE);CHKERRQ(ierr);
    ierr = DMDASetDof(da, dof);CHKERRQ(ierr);
    ierr = DMDASetStencilType(da, DMDA_STENCIL_BOX);CHKERRQ(ierr);
    ierr = DMDASetStencilWidth(da, width);CHKERRQ(ierr);
    mesh->dmCoarse = da;
  }
  PetscFunctionReturn(0);
}
