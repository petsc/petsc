#include <petsc/private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbarrier_ PETSCBARRIER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbarrier_ petscbarrier
#endif

PETSC_EXTERN void PETSC_STDCALL petscbarrier_(PetscObject *obj, int *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(obj);
  *ierr = PetscBarrier(*obj);
}
