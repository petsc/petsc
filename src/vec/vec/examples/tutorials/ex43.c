
#include <petscvec.h>

void fillupvector(Vec *v,PetscErrorCode *ierr)
{
  *ierr = VecSet(*v,1.0);
  return;
}

