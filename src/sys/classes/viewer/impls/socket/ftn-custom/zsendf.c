#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewersocketopen_ PETSCVIEWERSOCKETOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewersocketopen_ petscviewersocketopen
#endif

PETSC_EXTERN void petscviewersocketopen_(MPI_Comm *comm, char *name, int *port, PetscViewer *lab, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;
  FIXCHAR(name, len, c1);
  *ierr = PetscViewerSocketOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, *port, lab);
  if (*ierr) return;
  FREECHAR(name, c1);
}
