
static char help[] = "Demonstrates VecStrideScatter() and VecStrideGather().\n\n";

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/

#include <petscvec.h>

int main(int argc,char **argv)
{
  Vec            v,s;               /* vectors */
  PetscInt       n   = 20;
  PetscScalar    one = 1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
      Create multi-component vector with 2 components
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecSetSizes(v,PETSC_DECIDE,n));
  PetscCall(VecSetBlockSize(v,2));
  PetscCall(VecSetFromOptions(v));

  /*
      Create single-component vector
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&s));
  PetscCall(VecSetSizes(s,PETSC_DECIDE,n/2));
  PetscCall(VecSetFromOptions(s));

  /*
     Set the vectors to entries to a constant value.
  */
  PetscCall(VecSet(v,one));

  /*
     Get the first component from the multi-component vector to the single vector
  */
  PetscCall(VecStrideGather(v,0,s,INSERT_VALUES));

  PetscCall(VecView(s,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Put the values back into the second component
  */
  PetscCall(VecStrideScatter(s,1,v,ADD_VALUES));

  PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&s));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
