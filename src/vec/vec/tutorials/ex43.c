
#include <petscvec.h>

PETSC_INTERN void fillupvector(Vec *v, PetscErrorCode *ierr)
{
  *ierr = VecSet(*v, 1.0);
  return;
}
