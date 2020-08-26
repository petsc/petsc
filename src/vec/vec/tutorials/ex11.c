
static char help[] = "Demonstrates VecStrideNorm().\n\n";

/*T
   Concepts: vectors^norms of sub-vectors;
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
  Vec            x;               /* vectors */
  PetscReal      norm;
  PetscInt       n = 20;
  PetscErrorCode ierr;
  PetscScalar    one = 1.0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /*
     Create a vector, specifying only its global dimension.
     When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
     the vector format (currently parallel,
     shared, or sequential) is determined at runtime.  Also, the parallel
     partitioning of the vector is determined by PETSc at runtime.

     Routines for creating particular vector types directly are:
        VecCreateSeq() - uniprocessor vector
        VecCreateMPI() - distributed vector, where the user can
                         determine the parallel partitioning
        VecCreateShared() - parallel vector that uses shared memory
                            (available only on the SGI); otherwise,
                            is the same as VecCreateMPI()

     With VecCreate(), VecSetSizes() and VecSetFromOptions() the option
     -vec_type mpi or -vec_type shared causes the
     particular type of vector to be formed.

  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /*
     Set the vectors to entries to a constant value.
  */
  ierr = VecSet(x,one);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of entire vector: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of entire vector: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecNorm(x,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of entire vector: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,0,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 0: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,0,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 0: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,0,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 0: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,1,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_2 Norm of sub-vector 1: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,1,NORM_1,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_1 Norm of sub-vector 1: %g\n",(double)norm);CHKERRQ(ierr);

  ierr = VecStrideNorm(x,1,NORM_INFINITY,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"L_inf Norm of sub-vector 1: %g\n",(double)norm);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

TEST*/
