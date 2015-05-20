#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexinterpolate_         DMPLEXINTERPOLATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexinterpolate_         dmplexinterpolate
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexinterpolate_(DM *dm, DM *dmInt, int *ierr)
{
  *dmInt = NULL;
  *ierr = DMPlexInterpolate(*dm, dmInt);
}
