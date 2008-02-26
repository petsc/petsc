#include "private/fortranimpl.h"
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscdrawtensorcontour_   PETSCDRAWTENSORCONTOUR
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscdrawtensorcontour_   petscdrawtensorcontour
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscdrawtensorcontour_(PetscDraw *win,int *m,int *n,PetscReal *x,PetscReal *y,PetscReal *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLDOUBLE(x);
  CHKFORTRANNULLDOUBLE(y);
  *ierr = PetscDrawTensorContour(*win,*m,*n,x,y,V);
}

EXTERN_C_END
