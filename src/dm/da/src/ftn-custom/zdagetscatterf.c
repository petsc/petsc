
#include "private/fortranimpl.h"
#include "petscda.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dagetscatter_                DAGETSCATTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define dagetscatter_                dagetscatter
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL dagetscatter_(DA *da,VecScatter *ltog,VecScatter *gtol,VecScatter *ltol,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(ltog);
  CHKFORTRANNULLOBJECT(gtol);
  CHKFORTRANNULLOBJECT(ltol);
  *ierr = DAGetScatter(*da,ltog,gtol,ltol);
}

EXTERN_C_END
