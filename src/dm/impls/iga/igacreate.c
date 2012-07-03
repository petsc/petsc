#include <petsc-private/igaimpl.h>    /*I   "petscdmiga.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_IGA"
PetscErrorCode  DMSetFromOptions_IGA(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMIGA Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateGlobalVector_IGA(DM dm, Vec *gvec);
extern PetscErrorCode DMCreateLocalVector_IGA(DM dm, Vec *lvec);
extern PetscErrorCode DMCreateMatrix_IGA(DM dm, const MatType mtype, Mat *J);
extern PetscErrorCode DMGlobalToLocalBegin_IGA(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMGlobalToLocalEnd_IGA(DM dm, Vec g, InsertMode mode, Vec l);
extern PetscErrorCode DMLocalToGlobalBegin_IGA(DM dm, Vec l, InsertMode mode, Vec g);
extern PetscErrorCode DMLocalToGlobalEnd_IGA(DM dm, Vec l, InsertMode mode, Vec g);
#if 0
extern PetscErrorCode DMCreateLocalToGlobalMapping_IGA(DM dm);
extern PetscErrorCode DMCreateInterpolation_IGA(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling);
#endif
extern PetscErrorCode DMView_IGA(DM dm, PetscViewer viewer);
extern PetscErrorCode DMDestroy_IGA(DM dm);

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "DMCreate_IGA"
PetscErrorCode DMCreate_IGA(DM dm)
{
  DM_IGA        *iga;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscNewLog(dm, DM_IGA, &iga);CHKERRQ(ierr);
  dm->data = iga;

  /* TODO */
  iga->Ux  = 0; iga->Uy  = 0; iga->Uz  = 0;
  iga->bdX = 0; iga->bdY = 0; iga->bdZ = 0;

  ierr = PetscStrallocpy(VECSTANDARD, &dm->vectype);CHKERRQ(ierr);
  dm->ops->view               = DMView_IGA;
  dm->ops->setfromoptions     = DMSetFromOptions_IGA;
  dm->ops->setup              = 0;
  dm->ops->createglobalvector = DMCreateGlobalVector_IGA;
  dm->ops->createlocalvector  = DMCreateLocalVector_IGA;
  dm->ops->createlocaltoglobalmapping      = 0 /* DMCreateLocalToGlobalMapping_IGA */;
  dm->ops->createlocaltoglobalmappingblock = 0;

  dm->ops->getcoloring        = 0;
  dm->ops->creatematrix          = DMCreateMatrix_IGA;
  dm->ops->createinterpolation   = 0 /* DMCreateInterpolation_IGA */;
  dm->ops->getaggregates      = 0;
  dm->ops->getinjection       = 0;

  dm->ops->refine             = 0;
  dm->ops->coarsen            = 0;
  dm->ops->refinehierarchy    = 0;
  dm->ops->coarsenhierarchy   = 0;

  dm->ops->globaltolocalbegin = DMGlobalToLocalBegin_IGA;
  dm->ops->globaltolocalend   = DMGlobalToLocalEnd_IGA;
  dm->ops->localtoglobalbegin = DMLocalToGlobalBegin_IGA;
  dm->ops->localtoglobalend   = DMLocalToGlobalEnd_IGA;

  dm->ops->destroy            = DMDestroy_IGA;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "DMIGACreate"
/*@
  DMIGACreate - Creates a DMIGA object.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMIGA object

  Output Parameter:
. iga  - The DMIGA object

  Level: beginner

.keywords: DMIGA, create
@*/
PetscErrorCode  DMIGACreate(MPI_Comm comm, DM *iga)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(iga,2);
  ierr = DMCreate(comm, iga);CHKERRQ(ierr);
  ierr = DMSetType(*iga, DMIGA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
