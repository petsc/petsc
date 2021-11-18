
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /*
      Create multi-component vector with 4 components
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

  /*
     Set the vector values
  */
  ierr = VecGetOwnershipRange(v,&start,&end);CHKERRQ(ierr);
  for (i=start; i<end; i++) {
    value = i;
    ierr  = VecSetValues(v,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  ierr = VecStrideSubSetGather(v,PETSC_DETERMINE,vidx,NULL,s,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecView(s,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecScale(s,100.0);CHKERRQ(ierr);

  ierr = VecStrideSubSetScatter(s,PETSC_DETERMINE,NULL,vidx,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  ierr = VecStrideSubSetGather(v,2,vidx,sidx,s,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecView(s,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecScale(s,100.0);CHKERRQ(ierr);

  ierr = VecStrideSubSetScatter(s,2,sidx,vidx,v,ADD_VALUES);CHKERRQ(ierr);
  ierr = VecView(v,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecStrideMax(v,1,&miidx[0],&mvidx[0]);CHKERRQ(ierr);
  ierr = VecStrideMin(v,1,&miidx[1],&mvidx[1]);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Min/Max: %" PetscInt_FMT " %g, %" PetscInt_FMT " %g\n",miidx[0],(double)mvidx[0],miidx[1],(double)mvidx[1]);CHKERRQ(ierr);
  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&v);CHKERRQ(ierr);
  ierr = VecDestroy(&s);CHKERRQ(ierr);
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
