#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsclogeventbegin_ PETSCLOGEVENTBEGIN
  #define petsclogeventend_   PETSCLOGEVENTEND
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsclogeventbegin_ petsclogeventbegin
  #define petsclogeventend_   petsclogeventend
#endif

PETSC_EXTERN void petsclogeventbegin_(PetscLogEvent *e, PetscErrorCode *ierr)
{
  *ierr = PetscLogEventBegin(*e, 0, 0, 0, 0);
}

PETSC_EXTERN void petsclogeventend_(PetscLogEvent *e, PetscErrorCode *ierr)
{
  *ierr = PetscLogEventEnd(*e, 0, 0, 0, 0);
}
