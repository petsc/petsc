
#include "petscmat.h"

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatConvertTo_MPIAdj(Mat,MatType,Mat*);
EXTERN PetscErrorCode MatConvertTo_aij(Mat,MatType,Mat*);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatConvertRegisterAll"
/*@C
    MatConvertRegisterAll - Registers all standard matrix type routines to convert to

  Not Collective

  Level: developer

  Notes: To prevent registering all matrix types; copy this routine to 
         your source code and comment out the versions below that you do not need.

.seealso: MatRegister(), MatConvert()

@*/
PetscErrorCode MatConvertRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatConvertRegisterAllCalled = PETSC_TRUE;
  ierr = MatConvertRegisterDynamic(MATMPIADJ,path,"MatConvertTo_MPIAdj",MatConvertTo_MPIAdj);CHKERRQ(ierr);
  ierr = MatConvertRegisterDynamic(MATAIJ,path,"MatConvertTo_aij",MatConvert_aij);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  
