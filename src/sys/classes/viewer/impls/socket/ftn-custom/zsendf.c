#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersocketopen_     PETSCVIEWERSOCKETOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersocketopen_     petscviewersocketopen
#endif

PETSC_EXTERN void PETSC_STDCALL petscviewersocketopen_(MPI_Comm *comm,char* name PETSC_MIXED_LEN(len),int *port,PetscViewer *lab,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSocketOpen(MPI_Comm_f2c(*(MPI_Fint*)&*comm),c1,*port,lab);
  FREECHAR(name,c1);
}

