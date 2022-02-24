
static char help[] = "Demonstrates VecStrideSubSetScatter() and VecStrideSubSetGather().\n\n";

/*T
   Concepts: vectors^sub-vectors;
   Processors: n

   Allows one to easily pull out some components of a multi-component vector and put them in another vector.

   Note that these are special cases of VecScatter
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
  Vec            v,s;
  PetscInt       i,start,end,n = 8;
  PetscErrorCode ierr;
  PetscScalar    value;
  const PetscInt vidx[] = {1,2},sidx[] = {1,0};
  PetscInt       miidx[2];
  PetscReal      mvidx[2];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /*
      Create multi-component vector with 4 components
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

  /*
     Set the vector values
  */
  CHKERRQ(VecGetOwnershipRange(v,&start,&end));
  for (i=start; i<end; i++) {
    value = i;
    CHKERRQ(VecSetValues(v,1,&i,&value,INSERT_VALUES));
  }

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  CHKERRQ(VecStrideSubSetGather(v,PETSC_DETERMINE,vidx,NULL,s,INSERT_VALUES));
  CHKERRQ(VecView(s,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecScale(s,100.0));

  CHKERRQ(VecStrideSubSetScatter(s,PETSC_DETERMINE,NULL,vidx,v,ADD_VALUES));
  CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  CHKERRQ(VecStrideSubSetGather(v,2,vidx,sidx,s,INSERT_VALUES));
  CHKERRQ(VecView(s,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecScale(s,100.0));

  CHKERRQ(VecStrideSubSetScatter(s,2,sidx,vidx,v,ADD_VALUES));
  CHKERRQ(VecView(v,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecStrideMax(v,1,&miidx[0],&mvidx[0]));
  CHKERRQ(VecStrideMin(v,1,&miidx[1],&mvidx[1]));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Min/Max: %" PetscInt_FMT " %g, %" PetscInt_FMT " %g\n",miidx[0],(double)mvidx[0],miidx[1],(double)mvidx[1]));
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(VecDestroy(&s));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      filter: grep -v type | grep -v "MPI processes" | grep -v Process
      diff_args: -j
      nsize: 2

   test:
      filter: grep -v type | grep -v "MPI processes" | grep -v Process
      output_file: output/ex45_1.out
      diff_args: -j
      suffix: 2
      nsize: 1
      args: -vec_type {{seq mpi}}

TEST*/
