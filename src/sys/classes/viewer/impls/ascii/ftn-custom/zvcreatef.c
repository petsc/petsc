#include <petsc/private/ftnimpl.h>
#include <petscviewer.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petsc_viewer_stdout_ PETSC_VIEWER_STDOUT
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petsc_viewer_stdout_ petsc_viewer_stdout
#endif

#if PetscDefined(HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_stdout_ petsc_viewer_stdout__
#endif

PETSC_EXTERN PetscViewer petsc_viewer_stdout_(MPI_Comm *comm)
{
  return PETSC_VIEWER_STDOUT_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}
