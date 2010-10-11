
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetinterpolation_          DAGETINTERPOLATION
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetinterpolation_          dagetinterpolation
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL dagetinterpolation_(DA *dac,DA *daf,Mat *A,Vec *scale,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(scale);
  *ierr = DAGetInterpolation(*dac,*daf,A,scale);
}

EXTERN_C_END
