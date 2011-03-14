#include <private/fortranimpl.h>
#include <petscsnes.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define sneslinesearchgetparams_         SNESLINESEARCHGETPARAMS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define sneslinesearchgetparams_         sneslinesearchgetparams
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL sneslinesearchgetparams_(SNES *snes,PetscReal *alpha,PetscReal *maxstep,PetscReal *minlambda,PetscErrorCode *ierr)
{
  CHKFORTRANNULLREAL(alpha);
  CHKFORTRANNULLREAL(maxstep);
  CHKFORTRANNULLREAL(minlambda);
  *ierr = SNESLineSearchGetParams(*snes,alpha,maxstep,minlambda);
}


EXTERN_C_END
