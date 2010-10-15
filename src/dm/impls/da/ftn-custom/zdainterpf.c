
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmgetinterpolation_          DMGETINTERPOLATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dmgetinterpolation_          dmgetinterpolation
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dmgetinterpolation_(DM *dac,DM *daf,Mat *A,Vec *scale,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(scale);
  *ierr = DMGetInterpolation(*dac,*daf,A,scale);
}

EXTERN_C_END
