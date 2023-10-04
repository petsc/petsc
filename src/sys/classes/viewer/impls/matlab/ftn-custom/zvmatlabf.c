#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewermatlabopen_ PETSCVIEWERMATLABOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewermatlabopen_ petscviewermatlabopen
#endif

#if defined(PETSC_HAVE_MATLAB)
PETSC_EXTERN void petscviewermatlabopen_(MPI_Comm *comm, char *name, PetscFileMode *type, PetscViewer *binv, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;
  FIXCHAR(name, len, c1);
  *ierr = PetscViewerMatlabOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, *type, binv);
  if (*ierr) return;
  FREECHAR(name, c1);
}
#endif
