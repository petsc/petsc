
#include <petscsys.h>         /*I  "petscsys.h"  I*/

/*@C
     PetscGetArchType - Returns the $PETSC_ARCH that was used for this configuration of PETSc

     Not Collective

     Input Parameter:
.    slen - length of string buffer

     Output Parameter:
.    str - string area to contain architecture name, should be at least
           10 characters long. Name is truncated if string is not long enough.

     Level: developer

   Fortran Version:
   In Fortran this routine has the format

$       character*(10) str
$       call PetscGetArchType(str,ierr)

   Notes:
    This name is arbitrary and need not correspond to the physical hardware or the software running on the system.

.seealso: PetscGetUserName(),PetscGetHostName()
@*/
PetscErrorCode  PetscGetArchType(char str[],size_t slen)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_ARCH)
  ierr = PetscStrncpy(str,PETSC_ARCH,slen-1);CHKERRQ(ierr);
  str[slen-1] = 0;
#else
#error "$PETSC_ARCH/include/petscconf.h is missing PETSC_ARCH"
#endif
  PetscFunctionReturn(0);
}

