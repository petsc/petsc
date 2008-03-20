#include "private/fortranimpl.h"
#include "petsc.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petsc_viewer_stdout_       PETSC_VIEWER_STDOUT
#define petscviewerasciiopen_      PETSCVIEWERASCIIOPEN
#elif defined(PETSC_HAVE_FORTRAN_UNDERSCORE_UNDERSCORE_AT_END)
#define petsc_viewer_stdout_      petsc_viewer_stdout__
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewerasciiopen_      petscviewerasciiopen
#define petsc_viewer_stdout        petsc_viewer_stdout
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscviewerasciiopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewer *lab,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerASCIIOpen((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
  FREECHAR(name,c1);
}

PetscViewer PETSC_STDCALL petsc_viewer_stdout_(MPI_Comm *comm)
{
  return PETSC_VIEWER_STDOUT_(*comm);
}

EXTERN_C_END

