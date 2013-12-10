#include <petsc-private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexdistribute_          DMPLEXDISTRIBUTE
#define dmplexcreatefromcelllist_  DMPLEXCREATEFROMCELLLIST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexdistribute_          dmplexdistribute
#define dmplexcreatefromcelllist_  dmplexcreatefromcelllist
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexdistribute_(DM *dm, CHAR name PETSC_MIXED_LEN(lenN), PetscInt *overlap, PetscSF *sf, DM *dmParallel, int *ierr PETSC_END_LEN(lenN))
{
  char *partitioner;

  FIXCHAR(name, lenN, partitioner);
  *ierr = DMPlexDistribute(*dm, partitioner, *overlap, sf, dmParallel);
  FREECHAR(name, partitioner);
}

PETSC_EXTERN void PETSC_STDCALL dmplexcreatefromcelllist_(MPI_Comm *comm, PetscInt *dim, PetscInt *numCells, PetscInt *numVertices, PetscInt *numCorners, PetscBool *interpolate, const int cells[], PetscInt *spaceDim, const double vertexCoords[], DM *dm, int *ierr)
{
  *ierr = DMPlexCreateFromCellList(*comm, *dim, *numCells, *numVertices, *numCorners, *interpolate, cells, *spaceDim, vertexCoords, dm);
}
