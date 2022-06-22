
static char help[] = "Demonstrates VecStrideScatter() and VecStrideGather() with subvectors that are also strided.\n\n";

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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
      Create multi-component vector with 2 components
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&v));
  PetscCall(VecSetSizes(v,PETSC_DECIDE,n));
  PetscCall(VecSetBlockSize(v,4));
  PetscCall(VecSetFromOptions(v));

  /*
      Create double-component vectors
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&s));
  PetscCall(VecSetSizes(s,PETSC_DECIDE,n/2));
  PetscCall(VecSetBlockSize(s,2));
  PetscCall(VecSetFromOptions(s));
  PetscCall(VecDuplicate(s,&r));

  vecs[0] = s;
  vecs[1] = r;
  /*
     Set the vector values
  */
  PetscCall(VecGetOwnershipRange(v,&start,&end));
  for (i=start; i<end; i++) {
    value = i;
    PetscCall(VecSetValues(v,1,&i,&value,INSERT_VALUES));
  }

  /*
     Get the components from the multi-component vector to the other vectors
  */
  PetscCall(VecStrideGatherAll(v,vecs,INSERT_VALUES));

  PetscCall(VecView(s,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(r,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecStrideScatterAll(vecs,v,ADD_VALUES));

  PetscCall(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&v));
  PetscCall(VecDestroy(&s));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
