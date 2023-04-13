#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmplexcreateboxmesh_  DMPLEXCREATEBOXMESH
  #define dmplexcreatefromfile_ DMPLEXCREATEFROMFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmplexcreateboxmesh_  dmplexcreateboxmesh
  #define dmplexcreatefromfile_ dmplexcreatefromfile
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

PETSC_EXTERN void dmplexcreatefromfile_(MPI_Fint *comm, char *fname, char *pname, PetscBool *interpolate, DM *dm, int *ierr, PETSC_FORTRAN_CHARLEN_T lenfilename, PETSC_FORTRAN_CHARLEN_T lenplexname)
{
  char *filename;
  char *plexname;

  FIXCHAR(fname, lenfilename, filename);
  FIXCHAR(pname, lenplexname, plexname);
  *ierr = DMPlexCreateFromFile(MPI_Comm_f2c(*(comm)), filename, plexname, *interpolate, dm);
  if (*ierr) return;
  FREECHAR(fname, filename);
  FREECHAR(pname, plexname);
}
