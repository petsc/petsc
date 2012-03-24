#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbarrier_ PETSCBARRIER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbarrier_ petscbarrier
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL   petscbarrier_(PetscObject *obj, int *ierr ){
  CHKFORTRANNULLOBJECT(obj);
  *ierr = PetscBarrier(*obj);
}
EXTERN_C_END
