#include <petsc/private/fortranimpl.h>
#include <petsc/private/dmdaimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmdagetghostcorners_       DMDAGETGHOSTCORNERS
  #define dmdagetghostcorners000000_ DMDAGETGHOSTCORNERS000000
  #define dmdagetghostcorners001001_ DMDAGETGHOSTCORNERS001001
  #define dmdagetghostcorners011011_ DMDAGETGHOSTCORNERS011011
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmdagetghostcorners_       dmdagetghostcorners
  #define dmdagetghostcorners000000_ dmdagetghostcorners000000
  #define dmdagetghostcorners001001_ dmdagetghostcorners001001
  #define dmdagetghostcorners011011_ dmdagetghostcorners011011
#endif

PETSC_EXTERN void dmdagetghostcorners_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  CHKFORTRANNULLINTEGER(y);
  CHKFORTRANNULLINTEGER(z);
  CHKFORTRANNULLINTEGER(n);
  CHKFORTRANNULLINTEGER(p);

  *ierr = DMDAGetGhostCorners(*da, x, y, z, m, n, p);
}

PETSC_EXTERN void dmdagetghostcorners000000_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetghostcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetghostcorners001001_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetghostcorners_(da, x, y, z, m, n, p, ierr);
}

PETSC_EXTERN void dmdagetghostcorners011011_(DM *da, PetscInt *x, PetscInt *y, PetscInt *z, PetscInt *m, PetscInt *n, PetscInt *p, int *ierr)
{
  dmdagetghostcorners_(da, x, y, z, m, n, p, ierr);
}
