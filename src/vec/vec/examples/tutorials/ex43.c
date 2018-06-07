
#include <petscvec.h>

void fillupvector(Vec *v,PetscErrorCode *ierr)
{
  *ierr = VecSet(*v,1.0);
  return;
}

/*TEST

   build:
     depends: ex43f.F
     requires: fortran

   test:

TEST*/
