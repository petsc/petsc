#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexcreatesubmesh_          DMPLEXCREATESUBMESH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexcreatesubmesh_          dmplexcreatesubmesh
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexcreatesubmesh_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *value, DM *subdm, int *ierr PETSC_END_LEN(lenN))
{
  char *label;

  FIXCHAR(name, lenN, label);
  *ierr = DMPlexCreateSubmesh(*dm, label, *value, subdm);
  FREECHAR(name, label);
}
