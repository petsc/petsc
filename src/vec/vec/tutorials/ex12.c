
static char help[] = "Demonstrates VecStrideScatter() and VecStrideGather().\n\n";

/*T
   Concepts: vectors^sub-vectors;
   Processors: n
T*/

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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
      Create multi-component vector with 2 components
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
  CHKERRQ(VecSetSizes(v,PETSC_DECIDE,n));
  CHKERRQ(VecSetBlockSize(v,2));
  CHKERRQ(VecSetFromOptions(v));

  /*
      Create single-component vector
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&s));
  CHKERRQ(VecSetSizes(s,PETSC_DECIDE,n/2));
  CHKERRQ(VecSetFromOptions(s));

  /*
     Set the vectors to entries to a constant value.
  */
  CHKERRQ(VecSet(v,one));

  /*
     Get the first component from the multi-component vector to the single vector
  */
  CHKERRQ(VecStrideGather(v,0,s,INSERT_VALUES));

  CHKERRQ(VecView(s,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Put the values back into the second component
  */
  CHKERRQ(VecStrideScatter(s,1,v,ADD_VALUES));

  CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&s));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
