#include <petsc/private/ftnimpl.h>

#if PetscDefined(HAVE_FORTRAN_CAPS)
  #define petscobjectgetcomm_ PETSCOBJECTGETCOMM
#elif !PetscDefined(HAVE_FORTRAN_UNDERSCORE)
  #define petscobjectgetcomm_ petscobjectgetcomm
#endif

PETSC_EXTERN void petscobjectgetcomm_(PetscObject *obj, MPI_Fint *comm, PetscErrorCode *ierr)
{
  MPI_Comm c;
  *ierr        = PetscObjectGetComm(*obj, &c);
  *(int *)comm = MPI_Comm_c2f(c);
}
