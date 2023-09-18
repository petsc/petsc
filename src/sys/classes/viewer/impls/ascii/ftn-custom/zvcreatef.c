#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsc_viewer_stdout__ PETSC_VIEWER_STDOUT_BROKEN
  #define petscviewerasciiopen_ PETSCVIEWERASCIIOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerasciiopen_ petscviewerasciiopen
  #define petsc_viewer_stdout__ petsc_viewer_stdout_
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_stdout__ petsc_viewer_stdout___
#endif

PETSC_EXTERN void petscviewerasciiopen_(MPI_Comm *comm, char *name, PetscViewer *lab, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *c1;
  FIXCHAR(name, len, c1);
  *ierr = PetscViewerASCIIOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), c1, lab);
  if (*ierr) return;
  FREECHAR(name, c1);
}

PETSC_EXTERN PetscViewer petsc_viewer_stdout__(MPI_Comm *comm)
{
  return PETSC_VIEWER_STDOUT_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}
