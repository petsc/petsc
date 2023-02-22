
static char help[] = "Demonstrates VecStrideSubSetScatter() and VecStrideSubSetGather().\n\n";

/*
   Allows one to easily pull out some components of a multi-component vector and put them in another vector.

   Note that these are special cases of VecScatter
*/

/*
  Include "petscvec.h" so that we can use vectors.  Note that this file
  automatically includes:
     petscsys.h       - base PETSc routines   petscis.h     - index sets
     petscviewer.h - viewers
*/

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec            v, s;
  PetscInt       i, start, end, n = 8;
  PetscScalar    value;
  const PetscInt vidx[] = {1, 2}, sidx[] = {1, 0};
  PetscInt       miidx[2];
  PetscReal      mvidx[2];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /*
      Create multi-component vector with 4 components
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, PETSC_DECIDE, n));
  PetscCall(VecSetBlockSize(v, 4));
  PetscCall(VecSetFromOptions(v));

  /*
      Create double-component vectors
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &s));
  PetscCall(VecSetSizes(s, PETSC_DECIDE, n / 2));
  PetscCall(VecSetBlockSize(s, 2));
  PetscCall(VecSetFromOptions(s));

  /*
     Set the vector values
  */
  PetscCall(VecGetOwnershipRange(v, &start, &end));
  for (i = start; i < end; i++) {
    value = i;
    PetscCall(VecSetValues(v, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  PetscCall(VecStrideSubSetGather(v, PETSC_DETERMINE, vidx, NULL, s, INSERT_VALUES));
  PetscCall(VecView(s, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecScale(s, 100.0));

  PetscCall(VecStrideSubSetScatter(s, PETSC_DETERMINE, NULL, vidx, v, ADD_VALUES));
  PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

  /*
     Get the components from the large multi-component vector to the small multi-component vector,
     scale the smaller vector and then move values back to the large vector
  */
  PetscCall(VecStrideSubSetGather(v, 2, vidx, sidx, s, INSERT_VALUES));
  PetscCall(VecView(s, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecScale(s, 100.0));

  PetscCall(VecStrideSubSetScatter(s, 2, sidx, vidx, v, ADD_VALUES));
  PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecStrideMax(v, 1, &miidx[0], &mvidx[0]));
  PetscCall(VecStrideMin(v, 1, &miidx[1], &mvidx[1]));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Min/Max: %" PetscInt_FMT " %g, %" PetscInt_FMT " %g\n", miidx[0], (double)mvidx[0], miidx[1], (double)mvidx[1]));
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
      filter: grep -v type | grep -v " MPI process" | grep -v Process
      diff_args: -j
      nsize: 2

   test:
      filter: grep -v type | grep -v " MPI process" | grep -v Process
      output_file: output/ex45_1.out
      diff_args: -j
      suffix: 2
      nsize: 1
      args: -vec_type {{seq mpi}}

TEST*/
