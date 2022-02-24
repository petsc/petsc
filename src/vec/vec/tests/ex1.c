static char help[] = "Tests repeated VecSetType().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n   = 5;
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /* create vector */
  CHKERRQ(VecCreate(PETSC_COMM_SELF,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetType(x,"mpi"));
  CHKERRQ(VecSetType(x,"seq"));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecSetType(x,"mpi"));

  CHKERRQ(VecSet(x,one));
  CHKERRQ(VecSet(y,two));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       suffix: 1

     test:
       suffix: 2
       nsize: 2

TEST*/
