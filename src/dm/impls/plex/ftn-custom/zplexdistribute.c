#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexdistribute_ DMPLEXDISTRIBUTE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexdistribute_ dmplexdistribute
#endif

/* Definitions of Fortran Wrapper routines */
PETSC_EXTERN void dmplexdistribute_(DM *dm, PetscInt *overlap, PetscSF *sf, DM *dmParallel, int *ierr)
{
  CHKFORTRANNULLOBJECT(sf);
  *ierr = DMPlexDistribute(*dm, *overlap, sf, dmParallel);
}
