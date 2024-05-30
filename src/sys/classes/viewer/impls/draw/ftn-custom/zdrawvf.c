#include <petsc/private/fortranimpl.h>
#include <petscdraw.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petsc_viewer_draw__ PETSC_VIEWER_DRAW_BROKEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petsc_viewer_draw__ petsc_viewer_draw_
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
  #define petsc_viewer_draw__ petsc_viewer_draw___
#endif

PETSC_EXTERN PetscViewer petsc_viewer_draw__(MPI_Comm *comm)
{
  return PETSC_VIEWER_DRAW_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}
