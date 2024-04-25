#include <petsc/private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscrandomsetseed_         PETSCRANDOMSETSEED
  #define petscrandomgetseed_         PETSCRANDOMGETSEED
  #define petscrandomviewfromoptions_ PETSCRANDOMVIEWFROMOPTIONS
  #define petscrandomdestroy_         PETSCRANDOMDESTROY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscrandomsetseed_         petscrandomsetseed
  #define petscrandomgetseed_         petscrandomgetseed
  #define petscrandomviewfromoptions_ petscrandomviewfromoptions
  #define petscrandomdestroy_         petscrandomdestroy
#endif

PETSC_EXTERN void petscrandomgetseed_(PetscRandom *r, unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomGetSeed(*r, seed);
}
PETSC_EXTERN void petscrandomsetseed_(PetscRandom *r, unsigned long *seed, PetscErrorCode *ierr)
{
  *ierr = PetscRandomSetSeed(*r, *seed);
}

PETSC_EXTERN void petscrandomviewfromoptions_(PetscRandom *ao, PetscObject obj, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type, len, t);
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscRandomViewFromOptions(*ao, obj, t);
  if (*ierr) return;
  FREECHAR(type, t);
}

PETSC_EXTERN void petscrandomdestroy_(PetscRandom *x, int *ierr)
{
  PETSC_FORTRAN_OBJECT_F_DESTROYED_TO_C_NULL(x);
  *ierr = PetscRandomDestroy(x);
  if (*ierr) return;
  PETSC_FORTRAN_OBJECT_C_NULL_TO_F_DESTROYED(x);
}
