
#include <petsc-private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmdagetscatter_                DMDAGETSCATTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmdagetscatter_                dmdagetscatter
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dmdagetscatter_(DM *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ltog);
  CHKFORTRANNULLOBJECT(gtol);
  CHKFORTRANNULLOBJECT(ltol);
  *ierr = DMDAGetScatter(*da,ltog,gtol,ltol);
}

EXTERN_C_END
