#define PETSC_DLL

#include "petsc.h"         /*I  "petsc.h"  I*/
#include "petscsys.h"           /*I  "petscsys.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscGetArchType"
/*@C
     PetscGetArchType - Returns a standardized architecture type for the machine
     that is executing this routine. 

     Not Collective

     Input Parameter:
.    slen - length of string buffer

     Output Parameter:
.    str - string area to contain architecture name, should be at least 
           10 characters long. Name is truncated if string is not long enough.

     Level: developer

     Concepts: machine type
     Concepts: architecture

@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetArchType(char str[],size_t slen)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_ARCH_NAME)
  ierr = PetscStrncpy(str,PETSC_ARCH_NAME,slen-1);CHKERRQ(ierr);
  str[slen-1] = 0;
#else
#error "bmake/$PETSC_ARCH/petscconf.h is missing PETSC_ARCH_NAME"
#endif
  PetscFunctionReturn(0);
}

