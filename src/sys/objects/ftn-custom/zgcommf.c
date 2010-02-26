#include "private/fortranimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscobjectgetcomm_        PETSCOBJECTGETCOMM
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscobjectgetcomm_        petscobjectgetcomm
#endif

EXTERN_C_BEGIN
void PETSC_STDCALL petscobjectgetcomm_(PetscObject *obj,int *comm,PetscErrorCode *ierr)
{
  MPI_Comm c;
  *ierr = PetscObjectGetComm(*obj,&c);
  *(int*)comm =  MPI_Comm_c2f(c);
}

EXTERN_C_END
