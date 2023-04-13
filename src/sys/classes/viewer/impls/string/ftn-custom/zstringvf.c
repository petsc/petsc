#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscviewerstringopen_ PETSCVIEWERSTRINGOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscviewerstringopen_ petscviewerstringopen
#endif

PETSC_EXTERN void petscviewerstringopen_(MPI_Comm *comm, char *name, PetscInt *len, PetscViewer *str, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len1)
{
  *ierr = PetscViewerStringOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm), name, *len, str);
}
