
#include <petsc-private/daimpl.h>    /*I "petscdmda.h"  I*/

extern PetscErrorCode  DMSetUp_DA_1D(DM);
extern PetscErrorCode  DMSetUp_DA_2D(DM);
extern PetscErrorCode  DMSetUp_DA_3D(DM);

#undef __FUNCT__  
#define __FUNCT__ "DMSetUp_DA"
PetscErrorCode  DMSetUp_DA(DM da)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID,1);
  if (dd->dim == 1) {
    ierr = DMSetUp_DA_1D(da);CHKERRQ(ierr);
  } else if (dd->dim == 2) {
    ierr = DMSetUp_DA_2D(da);CHKERRQ(ierr);
  } else if (dd->dim == 3) {
    ierr = DMSetUp_DA_3D(da);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"DMs only supported for 1, 2, and 3d");
  PetscFunctionReturn(0);
}

