#include <petsc/private/dmpatchimpl.h>   /*I      "petscdmpatch.h"   I*/
#include <petscdmda.h>

PetscErrorCode DMSetFromOptions_Patch(PetscOptionItems *PetscOptionsObject,DM dm)
{
  /* DM_Patch      *mesh = (DM_Patch*) dm->data; */

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  PetscCall(PetscOptionsHead(PetscOptionsObject,"DMPatch Options"));
  /* Handle associated vectors */
  /* Handle viewing */
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMSetUp_Patch(DM dm);
extern PetscErrorCode DMView_Patch(DM dm, PetscViewer viewer);
extern PetscErrorCode DMCreateGlobalVector_Patch(DM dm, Vec *g);
extern PetscErrorCode DMCreateLocalVector_Patch(DM dm, Vec *l);
extern PetscErrorCode DMDestroy_Patch(DM dm);
extern PetscErrorCode DMCreateSubDM_Patch(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm);

PetscErrorCode DMInitialize_Patch(DM dm)
{
  PetscFunctionBegin;
  dm->ops->view                            = DMView_Patch;
  dm->ops->setfromoptions                  = DMSetFromOptions_Patch;
  dm->ops->setup                           = DMSetUp_Patch;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Patch;
  dm->ops->createlocalvector               = DMCreateLocalVector_Patch;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = NULL;
  dm->ops->createinterpolation             = NULL;
  dm->ops->createinjection                 = NULL;
  dm->ops->refine                          = NULL;
  dm->ops->coarsen                         = NULL;
  dm->ops->refinehierarchy                 = NULL;
  dm->ops->coarsenhierarchy                = NULL;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_Patch;
  dm->ops->createsubdm                     = DMCreateSubDM_Patch;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMCreate_Patch(DM dm)
{
  DM_Patch       *mesh;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(PetscNewLog(dm,&mesh));
  dm->data = mesh;

  mesh->refct       = 1;
  mesh->dmCoarse    = NULL;
  mesh->patchSize.i = 0;
  mesh->patchSize.j = 0;
  mesh->patchSize.k = 0;
  mesh->patchSize.c = 0;

  PetscCall(DMInitialize_Patch(dm));
  PetscFunctionReturn(0);
}

/*@
  DMPatchCreate - Creates a DMPatch object, which is a collections of DMs called patches.

  Collective

  Input Parameter:
. comm - The communicator for the DMPatch object

  Output Parameter:
. mesh  - The DMPatch object

  Notes:

  This code is incomplete and not used by other parts of PETSc.

  Level: beginner

.seealso: DMPatchZoom()

@*/
PetscErrorCode DMPatchCreate(MPI_Comm comm, DM *mesh)
{
  PetscFunctionBegin;
  PetscValidPointer(mesh,2);
  PetscCall(DMCreate(comm, mesh));
  PetscCall(DMSetType(*mesh, DMPATCH));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPatchCreateGrid(MPI_Comm comm, PetscInt dim, MatStencil patchSize, MatStencil commSize, MatStencil gridSize, DM *dm)
{
  DM_Patch       *mesh;
  DM             da;
  PetscInt       dof = 1, width = 1;

  PetscFunctionBegin;
  PetscCall(DMPatchCreate(comm, dm));
  mesh = (DM_Patch*) (*dm)->data;
  if (dim < 2) {
    gridSize.j  = 1;
    patchSize.j = 1;
  }
  if (dim < 3) {
    gridSize.k  = 1;
    patchSize.k = 1;
  }
  PetscCall(DMCreate(comm, &da));
  PetscCall(DMSetType(da, DMDA));
  PetscCall(DMSetDimension(da, dim));
  PetscCall(DMDASetSizes(da, gridSize.i, gridSize.j, gridSize.k));
  PetscCall(DMDASetBoundaryType(da, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE));
  PetscCall(DMDASetDof(da, dof));
  PetscCall(DMDASetStencilType(da, DMDA_STENCIL_BOX));
  PetscCall(DMDASetStencilWidth(da, width));

  mesh->dmCoarse = da;

  PetscCall(DMPatchSetPatchSize(*dm, patchSize));
  PetscCall(DMPatchSetCommSize(*dm, commSize));
  PetscFunctionReturn(0);
}
