#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscrandomsetseed_ PETSCRANDOMSETSEED
  #define petscrandomgetseed_ PETSCRANDOMGETSEED
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscrandomsetseed_ petscrandomsetseed
  #define petscrandomgetseed_ petscrandomgetseed
#endif

PETSC_EXTERN void petscrandomgetseed_(PetscRandom *r, unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomGetSeed(*r, seed);
}
PETSC_EXTERN void petscrandomsetseed_(PetscRandom *r, unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomSetSeed(*r, *seed);
}
