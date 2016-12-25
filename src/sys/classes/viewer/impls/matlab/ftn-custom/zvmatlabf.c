#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewermatlabopen_     PETSCVIEWERMATLABOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewermatlabopen_     petscviewermatlabopen
#endif

#if defined(PETSC_HAVE_MATLAB_ENGINE)
PETSC_EXTERN void PETSC_STDCALL petscviewermatlabopen_(MPI_Comm *comm,char* name PETSC_MIXED_LEN(len),PetscFileMode *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerMatlabOpen(MPI_Comm_f2c(*(MPI_Fint*)&*comm),c1,*type,binv);
  FREECHAR(name,c1);
}
#endif

