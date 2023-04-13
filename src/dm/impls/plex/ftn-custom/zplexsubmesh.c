#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmplexconstructghostcells_ DMPLEXCONSTRUCTGHOSTCELLS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmplexconstructghostcells_ dmplexconstructghostcells
#endif

/* Definitions of Fortran Wrapper routines */
PETSC_EXTERN void dmplexconstructghostcells_(DM *dm, char *name, PetscInt *numGhostCells, DM *dmGhosted, int *ierr, PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *labelname;

  FIXCHAR(name, lenN, labelname);
  *ierr = DMPlexConstructGhostCells(*dm, labelname, numGhostCells, dmGhosted);
  if (*ierr) return;
  FREECHAR(name, labelname);
}
