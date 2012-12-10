#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewervtkopen_     PETSCVIEWERVTKOPEN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewervtkopen_     petscviewervtkopen
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscviewervtkopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscFileMode *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerVTKOpen(MPI_Comm_f2c(*(MPI_Fint *)&*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

EXTERN_C_END
