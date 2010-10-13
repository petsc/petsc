#define PETSCDM_DLL

#include "private/daimpl.h"    /*I "petscda.h"  I*/

extern PetscErrorCode PETSCDM_DLLEXPORT DASetUp_1D(DA);
extern PetscErrorCode PETSCDM_DLLEXPORT DASetUp_2D(DA);
extern PetscErrorCode PETSCDM_DLLEXPORT DASetUp_3D(DA);

#undef __FUNCT__  
#define __FUNCT__ "DASetUp"
/*@C
  DASetUp - Sets up the data structures for a DA

  Collective on DA

  Input Parameters:
. da     - The DA object

  Level: intermediate

.keywords: DA, set, type
.seealso: DACreate()
@*/
PetscErrorCode PETSCDM_DLLEXPORT DASetUp(DA da)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da, DM_CLASSID,1);
  if (dd->dim == 1) {
    ierr = DASetUp_1D(da);CHKERRQ(ierr);
  } else if (dd->dim == 2) {
    ierr = DASetUp_2D(da);CHKERRQ(ierr);
  } else if (dd->dim == 3) {
    ierr = DASetUp_3D(da);CHKERRQ(ierr);
  } else SETERRQ(((PetscObject)da)->comm,PETSC_ERR_SUP,"DAs only supported for 1, 2, and 3d");
  PetscFunctionReturn(0);
}

