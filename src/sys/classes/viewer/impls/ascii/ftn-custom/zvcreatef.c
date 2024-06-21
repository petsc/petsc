#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsc_viewer_stdout__ PETSC_VIEWER_STDOUT_BROKEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsc_viewer_stdout__ petsc_viewer_stdout_
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_stdout__ petsc_viewer_stdout___
#endif

PETSC_EXTERN PetscViewer petsc_viewer_stdout__(MPI_Comm *comm)
{
  return PETSC_VIEWER_STDOUT_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}
