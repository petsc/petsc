#include <petsc/private/fortranimpl.h>
#include <petsctime.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsctime_ PETSCTIME
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define petsctime_ petsctime
#endif

PETSC_EXTERN void petsctime_(PetscLogDouble *t, int *__ierr)
{
  *__ierr = PetscTime(t);
}
