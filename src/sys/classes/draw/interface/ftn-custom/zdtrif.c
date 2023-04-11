#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscdrawtensorcontour_ PETSCDRAWTENSORCONTOUR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscdrawtensorcontour_ petscdrawtensorcontour
#endif

PETSC_EXTERN void petscdrawtensorcontour_(PetscDraw *win, int *m, int *n, PetscReal *x, PetscReal *y, PetscReal *V, PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(x);
  CHKFORTRANNULLREAL(y);
  *ierr = PetscDrawTensorContour(*win, *m, *n, x, y, V);
}
