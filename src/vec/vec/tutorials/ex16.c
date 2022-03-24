
static char help[] = "Demonstrates VecStrideScatter() and VecStrideGather() with subvectors that are also strided.\n\n";

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
  Vec            v,s,r,vecs[2];               /* vectors */
  PetscInt       i,start,end,n = 20;
  PetscScalar    value;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
      Create multi-component vector with 2 components
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&v));
  CHKERRQ(VecSetSizes(v,PETSC_DECIDE,n));
  CHKERRQ(VecSetBlockSize(v,4));
  CHKERRQ(VecSetFromOptions(v));

  /*
      Create double-component vectors
  */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&s));
  CHKERRQ(VecSetSizes(s,PETSC_DECIDE,n/2));
  CHKERRQ(VecSetBlockSize(s,2));
  CHKERRQ(VecSetFromOptions(s));
  CHKERRQ(VecDuplicate(s,&r));

  vecs[0] = s;
  vecs[1] = r;
  /*
     Set the vector values
  */
  CHKERRQ(VecGetOwnershipRange(v,&start,&end));
  for (i=start; i<end; i++) {
    value = i;
    CHKERRQ(VecSetValues(v,1,&i,&value,INSERT_VALUES));
  }

  /*
     Get the components from the multi-component vector to the other vectors
  */
  CHKERRQ(VecStrideGatherAll(v,vecs,INSERT_VALUES));

  CHKERRQ(VecView(s,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(r,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecStrideScatterAll(vecs,v,ADD_VALUES));

  CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&s));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
