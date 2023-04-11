#include <petsc/private/fortranimpl.h>
#include <petscdmplex.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define dmplexcreateexodusfromfile_ DMPLEXCREATEEXODUSFROMFILE
  #define petscviewerexodusiiopen_    PETSCVIEWEREXODUSIIOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
  #define dmplexcreateexodusfromfile_ dmplexcreateexodusfromfile
  #define petscviewerexodusiiopen_    petscviewerexodusiiopen
#endif

/* Definitions of Fortran Wrapper routines */

PETSC_EXTERN void dmplexcreateexodusfromfile_(MPI_Fint *comm, char *name, PetscBool *interpolate, DM *dm, int *ierr, PETSC_FORTRAN_CHARLEN_T lenN)
{
  char *filename;

  FIXCHAR(name, lenN, filename);
  *ierr = DMPlexCreateExodusFromFile(MPI_Comm_f2c(*(comm)), filename, *interpolate, dm);
  if (*ierr) return;
  FREECHAR(name, filename);
}

PETSC_EXTERN void petscviewerexodusiiopen_(MPI_Comm *comm, char *name, PetscFileMode *type, PetscViewer *binv, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;

  FIXCHAR(name, len, c1);
  *ierr = PetscViewerExodusIIOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, *type, binv);
  if (*ierr) return;
  FREECHAR(name, c1);
}
