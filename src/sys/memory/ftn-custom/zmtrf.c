#include <petsc/private/ftnimpl.h>
#include <petscsys.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscmallocdump_ PETSCMALLOCDUMP
  #define petscmallocview_ PETSCMALLOCVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscmallocdump_ petscmallocdump
  #define petscmallocview_ petscmallocview
#endif

PETSC_EXTERN void petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
PETSC_EXTERN void petscmallocview_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocView(stdout);
}
