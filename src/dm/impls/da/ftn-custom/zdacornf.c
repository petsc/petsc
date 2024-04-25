#include <petsc/private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmdagetcorners_       DMDAGETCORNERS
  #define dmdagetcorners000000_ DMDAGETCORNERS000000
  #define dmdagetcorners001001_ DMDAGETCORNERS001001
  #define dmdagetcorners011011_ DMDAGETCORNERS011011
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmdagetcorners_       dmdagetcorners
  #define dmdagetcorners000000_ dmdagetcorners000000
  #define dmdagetcorners001001_ dmdagetcorners001001
  #define dmdagetcorners011011_ dmdagetcorners011011
#endif

PETSC_EXTERN void dmdagetcorners_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);

  *ierr = DMDAGetCorners(*da, x, y, z, m, n, p);
}

PETSC_EXTERN void dmdagetcorners000000_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetcorners001001_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetcorners011011_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetcorners_(da, x, y, z, m, n, p, ierr);
}
