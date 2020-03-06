#include <petsc/private/fortranimpl.h>
#include <petscts.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define tssundialsgetiterations_             TSSUNDIALSGETITERATIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define tssundialsgetiterations_             tssundialsgetiterations
#endif

PETSC_EXTERN void tssundialsgetiterations_(TS *ts,PetscInt *nonlin,PetscInt *lin,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(nonlin);
  CHKFORTRANNULLINTEGER(lin);
  *ierr = TSSundialsGetIterations(*ts,nonlin,lin);
}

