
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
  PetscErrorCode ierr;
  PetscScalar    value;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /*
      Create multi-component vector with 2 components
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&v);CHKERRQ(ierr);
  ierr = VecSetSizes(v,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(v,4);CHKERRQ(ierr);
  ierr = VecSetFromOptions(v);CHKERRQ(ierr);

  /*
      Create double-component vectors
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&s);CHKERRQ(ierr);
  ierr = VecSetSizes(s,PETSC_DECIDE,n/2);CHKERRQ(ierr);
  ierr = VecSetBlockSize(s,2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(s);CHKERRQ(ierr);
  ierr = VecDuplicate(s,&r);CHKERRQ(ierr);

  vecs[0] = s;
  vecs[1] = r;
  /*
     Set the vector values
  */
  ierr = VecGetOwnershipRange(v,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = i;
    ierr  = VecSetValues(v,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Get the components from the multi-component vector to the other vectors
  */
  ierr = VecStrideGatherAll(v,vecs,INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecView(s,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecStrideScatterAll(vecs,v,ADD_VALUES);CHKERRQ(ierr);

  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&s);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

TEST*/
