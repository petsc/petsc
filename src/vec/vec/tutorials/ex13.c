
static char help[] = "Demonstrates VecStrideSum().\n\n";

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec         x; /* vectors */
  PetscScalar sum;
  PetscInt    n   = 20;
  PetscScalar one = 1.0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

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
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, n));
  PetscCall(VecSetBlockSize(x, 2));
  PetscCall(VecSetFromOptions(x));

  /*
     Set the subvector entries to a constant value.
  */
  PetscCall(VecSet(x, one));
  PetscCall(VecStrideScale(x, 1, -2));

  PetscCall(VecSum(x, &sum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum of entire vector: %g\n", (double)PetscRealPart(sum)));

  PetscCall(VecStrideSum(x, 0, &sum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum of sub-vector 0: %g\n", (double)PetscRealPart(sum)));

  PetscCall(VecStrideSum(x, 1, &sum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Sum of sub-vector 1: %g\n", (double)PetscRealPart(sum)));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     test:
       nsize: 2

TEST*/
