
#include <petscvec.h>

PETSC_EXTERN void fillupvector(Vec *v,PetscErrorCode *ierr)
{
  *ierr = VecSet(*v,1.0);
  return;
}

