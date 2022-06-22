#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>
#include <petsclandau.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define dmplexlandaucreatevelocityspace_ DMPLEXLANDAUCREATEVELOCITYSPACE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define dmplexlandaucreatevelocityspace_ dmplexlandaucreatevelocityspace
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmplexlandaucreatevelocityspace_(MPI_Fint * comm,PetscInt *dim,char* name,Vec *X,Mat *J, DM *dm, int *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *prefix;
  FIXCHAR(name, len, prefix);
  *ierr = DMPlexLandauCreateVelocitySpace(MPI_Comm_f2c(*(comm)),*dim,prefix,X,J,dm);
  FREECHAR(name, prefix);
}
