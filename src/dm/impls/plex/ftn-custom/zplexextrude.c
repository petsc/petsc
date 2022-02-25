#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexextrude_        DMPLEXEXTRUDE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexextrude_        dmplexextrude
#endif

/* Definitions of Fortran Wrapper routines */
PETSC_EXTERN void dmplexextrude_(DM *dm, PetscInt *layers, PetscReal *thickness, PetscBool *tensor, PetscBool *symmetric, PetscReal normal[], PetscReal thicknesses[], DM *edm, int *ierr)
{
  CHKFORTRANNULLREAL(normal);
  CHKFORTRANNULLREAL(thicknesses);
  *ierr = DMPlexExtrude(*dm,*layers,*thickness,*tensor,*symmetric,normal,thicknesses,edm);
}
