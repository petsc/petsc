
#include <petscmat.h>

EXTERN_C_BEGIN
extern PetscErrorCode  MatCoarsenCreate_MIS(MatCoarsen);
extern PetscErrorCode  MatCoarsenCreate_HEM(MatCoarsen);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCoarsenRegisterAll" 
/*@C
  MatCoarsenRegisterAll - Registers all of the matrix Coarsen routines in PETSc.

  Not Collective

  Level: developer

  Adding new methods:
  To add a new method to the registry. Copy this routine and 
  modify it to incorporate a call to MatCoarsenRegisterDynamic() for 
  the new method, after the current list.

  Restricting the choices: To prevent all of the methods from being
  registered and thus save memory, copy this routine and modify it to
  register a zero, instead of the function name, for those methods you
 do not wish to register.  Make sure that the replacement routine is
  linked before libpetscmat.a.

 .keywords: matrix, Coarsen, register, all
 
 .seealso: MatCoarsenRegisterDynamic(), MatCoarsenRegisterDestroy()
 @*/
PetscErrorCode  MatCoarsenRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCoarsenRegisterAllCalled = PETSC_TRUE;
  ierr = MatCoarsenRegisterDynamic(MATCOARSENMIS,path,"MatCoarsenCreate_MIS",MatCoarsenCreate_MIS);CHKERRQ(ierr);
  ierr = MatCoarsenRegisterDynamic(MATCOARSENHEM,path,"MatCoarsenCreate_HEM",MatCoarsenCreate_HEM);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

