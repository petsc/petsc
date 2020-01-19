#include "petscsys.h"
#include "petscfix.h"
#include "petsc/private/fortranimpl.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscgetversion_ PETSCGETVERSION
#define petscgetversionnumber_ PETSCGETVERSIONNUMBER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscgetversion_ petscgetversion
#define petscgetversionnumber_ petscgetversionnumber
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif

PETSC_EXTERN void PETSC_STDCALL  petscgetversion_(char* version PETSC_MIXED_LEN(len1),int *ierr PETSC_END_LEN(len1))
{
  *ierr = PetscGetVersion(version,len1);
  FIXRETURNCHAR(PETSC_TRUE,version,len1);
}

PETSC_EXTERN void PETSC_STDCALL  petscgetversionnumber_(PetscInt *major,PetscInt *minor,PetscInt *subminor,PetscInt *release, int *ierr )
{
  CHKFORTRANNULLINTEGER(major);
  CHKFORTRANNULLINTEGER(minor);
  CHKFORTRANNULLINTEGER(subminor);
  CHKFORTRANNULLINTEGER(release);
  *ierr = PetscGetVersionNumber(major,minor,subminor,release);
}

#if defined(__cplusplus)
}
#endif
