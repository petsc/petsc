
#include <petsc/private/fortranimpl.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmdagetscatter_ DMDAGETSCATTER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define dmdagetscatter_ dmdagetscatter
#endif

PETSC_EXTERN void dmdagetscatter_(DM *da, VecScatter *gtol, VecScatter *ltol, PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECT(gtol);
  CHKFORTRANNULLOBJECT(ltol);
  *ierr = DMDAGetScatter(*da, gtol, ltol);
}
