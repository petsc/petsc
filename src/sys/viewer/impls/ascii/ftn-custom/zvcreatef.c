#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsc_viewer_stdout__      PETSC_VIEWER_STDOUT_BROKEN
#define petscviewerasciiopen_      PETSCVIEWERASCIIOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerasciiopen_      petscviewerasciiopen
#define petsc_viewer_stdout__      petsc_viewer_stdout_
#endif

#if defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE)
#define petsc_viewer_stdout__      petsc_viewer_stdout___
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscviewerasciiopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewer *lab,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerASCIIOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm),c1,lab);
  FREECHAR(name,c1);
}

PetscViewer PETSC_STDCALL petsc_viewer_stdout__(MPI_Comm *comm)
{
  return PETSC_VIEWER_STDOUT_(MPI_Comm_f2c(*(MPI_Fint *)&*comm));
}

EXTERN_C_END

