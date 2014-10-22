#include <petsc-private/dmforestimpl.h>
#include <petsc-private/dmimpl.h>
#include <petscsf.h>                     /*I "petscsf.h" */

#undef __FUNCT__
#define __FUNCT__ "DMClone_Forest"
PETSC_EXTERN PetscErrorCode DMClone_Forest(DM dm, DM *newdm)
{
  DM_Forest        *forest = (DM_Forest *) dm->data;
  const char       *type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  forest->refct++;
  (*newdm)->data = forest;
  ierr = PetscObjectGetType((PetscObject) dm, &type);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, type);CHKERRQ(ierr);
  ierr = DMInitialize_Forest(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Forest"
PetscErrorCode DMDestroy_Forest(DM dm)
{
  DM_Forest     *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--forest->refct > 0) PetscFunctionReturn(0);
  ierr = PetscSFDestroy(&forest->cellSF);CHKERRQ(ierr);
  if (forest->adaptCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->adaptMarkers);CHKERRQ(ierr);
  }
  if (forest->cellWeightsCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->cellWeights);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "DMForestSetType"
PetscErrorCode DMForestSetType(DM dm, DMForestType type)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscBool      isSame;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!type && !forest->type) {
    isSame = PETSC_TRUE;
  }
  else if (!type || !forest->type) {
    isSame = PETSC_FALSE;
  }
  else {
    ierr = PetscStrcmp((char*)(forest->type),(char*)type,&isSame);CHKERRQ(ierr);
  }
  if (!isSame) {
    ierr = DMForestReset(dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Forest"
PetscErrorCode DMSetFromFromOptions_Forest(DM dm)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest->setFromOptions = PETSC_TRUE;
  ierr = PetscOptionsHead("DMForest Options");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-dm_forest_type","DMForest type","DMForestSetType",DMForestList,(char*)(forest->type ? forest->type : DMFORESTP4EST),forest->type,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetType(dm,forest->type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Forest"
PetscErrorCode DMInitialize_Forest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr                    = PetscMemzero(dm->ops, sizeof (*(dm->ops)));CHKERRQ(ierr);
  dm->ops->setfromoptions = DMSetFromOptions_Forest;
  dm->ops->destroy        = DMDestroy_Forest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Forest"
PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM dm)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr                        = PetscNewLog(dm,&forest);CHKERRQ(ierr);
  dm->dim                     = 0;
  dm->data                    = forest;
  forest->refct               = 1;
  forest->data                = NULL;
  forest->setup               = 0;
  forest->type                = NULL;
  forest->topology            = NULL;
  forest->base                = NULL;
  forest->coarse              = NULL;
  forest->fine                = NULL;
  forest->adjDim              = PETSC_DEFAULT;
  forest->overlap             = PETSC_DEFAULT;
  forest->minRefinement       = PETSC_DEFAULT;
  forest->maxRefinement       = PETSC_DEFAULT;
  forest->cStart              = 0;
  forest->cEnd                = 0;
  forest->cellSF              = 0;
  forest->adaptMarkers        = NULL;
  forest->adaptCopyMode       = PETSC_USE_POINTER;
  forest->adaptStrategy       = DMFORESTADAPTALL;
  forest->gradeFactor         = 2;
  forest->cellWeights         = NULL;
  forest->cellWeightsCopyMode = PETSC_USE_POINTER;
  forest->weightsFactor       = 1.;
  forest->weightCapacity      = 1.;
  ierr = DMInitialize_Forest(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

