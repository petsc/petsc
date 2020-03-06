#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexcreategmshfromfile_  DMPLEXCREATEGMSHFROMFILE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexcreategmshfromfile_  dmplexcreategmshfromfile
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmplexcreategmshfromfile_(MPI_Fint *comm, char* name, PetscBool *interpolate, DM *dm, int *ierr,PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *filename;

  FIXCHAR(name, lenN, filename);
  *ierr = DMPlexCreateGmshFromFile(MPI_Comm_f2c(*(comm)), filename, *interpolate, dm);if (*ierr) return;
  FREECHAR(name, filename);
}
