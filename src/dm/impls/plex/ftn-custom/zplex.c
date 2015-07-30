#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexcreatefromcelllist_  DMPLEXCREATEFROMCELLLIST
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexcreatefromcelllist_  dmplexcreatefromcelllist
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexcreatefromcelllist_(MPI_Fint *comm, PetscInt *dim, PetscInt *numCells, PetscInt *numVertices, PetscInt *numCorners, PetscBool *interpolate, const int cells[], PetscInt *spaceDim, const double vertexCoords[], DM *dm, int *ierr)
{
  *ierr = DMPlexCreateFromCellList(MPI_Comm_f2c(*(comm)), *dim, *numCells, *numVertices, *numCorners, *interpolate, cells, *spaceDim, vertexCoords, dm);
}
