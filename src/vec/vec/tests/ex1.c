static char help[] = "Tests repeated VecSetType().\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n   = 5;
  PetscScalar    one = 1.0,two = 2.0;
  Vec            x,y;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* create vector */
  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(VecSetSizes(x,n,PETSC_DECIDE));
  PetscCall(VecSetType(x,"mpi"));
  PetscCall(VecSetType(x,"seq"));
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecSetType(x,"mpi"));

  PetscCall(VecSet(x,one));
  PetscCall(VecSet(y,two));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       suffix: 1

     test:
       suffix: 2
       nsize: 2

TEST*/
