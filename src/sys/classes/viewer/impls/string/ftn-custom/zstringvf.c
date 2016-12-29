#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewerstringopen_     PETSCVIEWERSTRINGOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerstringopen_     petscviewerstringopen
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewerstringopen_(MPI_Comm *comm,char* name PETSC_MIXED_LEN(len1),PetscInt *len,PetscViewer *str,
                                     PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  *ierr = PetscViewerStringOpen(MPI_Comm_f2c(*(MPI_Fint*)&*comm),name,*len,str);
}
