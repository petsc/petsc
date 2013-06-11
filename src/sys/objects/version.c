#include <petscsys.h>
/*@C
    PetscGetVersion - Gets the PETSc version information in a string.

    Input Parameter:
.   len - length of the string

    Output Parameter:
.   version - version string

    Level: developer

    Fortran Note:
    This routine is not supported in Fortran.

    Developer Note: The version information is also listed in
$    src/docs/tex/manual/intro.tex,
$    src/docs/tex/manual/manual.tex.
$    src/docs/website/index.html.

.seealso: PetscGetProgramName()

@*/

#undef __FUNCT__
#define __FUNCT__ "PetscGetVersion"
PetscErrorCode PetscGetVersion(char version[], size_t len)
{
  PetscErrorCode ierr;
#if (PETSC_VERSION_RELEASE == 1)
  ierr = PetscSNPrintf(version,len,"Petsc Release Version %d.%d.%d, %s ",PETSC_VERSION_MAJOR,PETSC_VERSION_MINOR, PETSC_VERSION_SUBMINOR,PETSC_VERSION_DATE);CHKERRQ(ierr);
#else
  ierr = PetscSNPrintf(version,len,"Petsc Development GIT revision: %s  GIT Date: %s",PETSC_VERSION_GIT, PETSC_VERSION_DATE_GIT);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


