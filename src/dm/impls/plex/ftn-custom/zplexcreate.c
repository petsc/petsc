#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexgenerateboxmesh_ DMPLEXGENERATEBOXMESH
#define dmplexcreatefromfile_  DMPLEXCREATEFROMFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexgenerateboxmesh_ dmplexgenerateboxmesh
#define dmplexcreatefromfile_  dmplexcreatefromfile
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void PETSC_STDCALL dmplexcreateboxmesh_(MPI_Fint *comm, PetscInt *dim, PetscInt faces[], PetscReal lower[], PetscReal upper[], PetscBool *interpolate, DM *dm, int *ierr)
{
  CHKFORTRANNULLINTEGER(faces);
  CHKFORTRANNULLREAL(lower);
  CHKFORTRANNULLREAL(upper);
  *ierr = DMPlexCreateBoxMesh(MPI_Comm_f2c(*(comm)),*dim,faces,lower,upper,*interpolate,dm);
}

PETSC_EXTERN void PETSC_STDCALL dmplexcreatefromfile_(MPI_Fint *comm, char* name PETSC_MIXED_LEN(lenN), PetscBool *interpolate, DM *dm, int *ierr PETSC_END_LEN(lenN))
{
  char *filename;

  FIXCHAR(name, lenN, filename);
  *ierr = DMPlexCreateFromFile(MPI_Comm_f2c(*(comm)), filename, *interpolate, dm);
  FREECHAR(name, filename);
}
