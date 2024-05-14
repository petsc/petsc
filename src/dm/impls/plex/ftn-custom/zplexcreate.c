#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmplexcreateboxmesh_ DMPLEXCREATEBOXMESH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmplexcreateboxmesh_ dmplexcreateboxmesh
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmplexcreateboxmesh_(MPI_Fint *comm, PetscInt *dim, PetscBool *simplex, PetscInt faces[], PetscReal lower[], PetscReal upper[], DMBoundaryType periodicity[], PetscBool *interpolate, DM *dm, int *ierr)
{
  CHKFORTRANNULLINTEGER(faces);
  CHKFORTRANNULLREAL(lower);
  CHKFORTRANNULLREAL(upper);
  CHKFORTRANNULLINTEGER(periodicity);
  *ierr = DMPlexCreateBoxMesh(MPI_Comm_f2c(*(comm)), *dim, *simplex, faces, lower, upper, periodicity, *interpolate, dm);
}
