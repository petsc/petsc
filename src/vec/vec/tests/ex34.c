static char help[] = "Tests for norm caching\n";

#include <petscvec.h>
#include <petsc/private/petscimpl.h> /* to gain access to the private PetscObjectStateIncrease() */

int main(int argc, char **argv)
{
  Vec         V, W;
  MPI_Comm    comm;
  PetscScalar one = 1, e = 2.7181;
  PetscReal   nrm1, nrm2, nrm3, nrm4;
  PetscInt    ione = 1;

  PetscFunctionBegin;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  comm = MPI_COMM_SELF;

  PetscCall(VecCreate(comm, &V));
  PetscCall(VecSetSizes(V, 10, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(V));
  PetscCall(VecSetRandom(V, NULL));
  PetscCall(VecAssemblyBegin(V));
  PetscCall(VecAssemblyEnd(V));

  /*
   * Initial
   */
  /* display norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Original: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display cached norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "cached: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Alter an element
   */
  PetscCall(VecSetValues(V, 1, &ione, &one, INSERT_VALUES));

  /* display norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Altered: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display cached norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "recomputed: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Scale the vector a little
   */
  PetscCall(VecScale(V, e));

  /* display updated cached norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Scale: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display forced norm 1 & 2 */
  PetscCall(PetscObjectStateIncrease((PetscObject)V));
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "recompute: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Normalize the vector a little
   */
  PetscCall(VecNormalize(V, &nrm1));

  /* display updated cached norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Normalize: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display forced norm 1 & 2 */
  PetscCall(PetscObjectStateIncrease((PetscObject)V));
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "recompute: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Copy to another vector
   */
  PetscCall(VecDuplicate(V, &W));
  PetscCall(VecCopy(V, W));

  /* display norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Original: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display cached norm 1 & 2 */
  PetscCall(VecNorm(W, NORM_1, &nrm1));
  PetscCall(VecNorm(W, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "copied: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Copy while data is invalid
   */
  PetscCall(VecSetValues(V, 1, &ione, &one, INSERT_VALUES));
  PetscCall(VecCopy(V, W));

  /* display norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Invalidated: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display norm 1 & 2 */
  PetscCall(VecNorm(W, NORM_1, &nrm1));
  PetscCall(VecNorm(W, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "copied: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Constant vector
   */
  PetscCall(VecSet(V, e));

  /* display updated cached norm 1 & 2 */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "Constant: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /* display forced norm 1 & 2 */
  PetscCall(PetscObjectStateIncrease((PetscObject)V));
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(V, NORM_2, &nrm2));
  PetscCall(PetscPrintf(comm, "recomputed: norm1=%e,norm2=%e\n", (double)nrm1, (double)nrm2));

  /*
   * Swap vectors
   */
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(W, NORM_1, &nrm2));
  PetscCall(PetscPrintf(comm, "Orig: norm_V=%e,norm_W=%e\n", (double)nrm1, (double)nrm2));
  /* store inf norm */
  PetscCall(VecNorm(V, NORM_INFINITY, &nrm3));
  PetscCall(VecNorm(W, NORM_INFINITY, &nrm4));

  PetscCall(VecSwap(V, W));

  PetscCall(PetscObjectStateIncrease((PetscObject)V));
  PetscCall(PetscObjectStateIncrease((PetscObject)W));
  PetscCall(VecNorm(V, NORM_1, &nrm1));
  PetscCall(VecNorm(W, NORM_1, &nrm2));
  PetscCall(PetscPrintf(comm, "swapped: norm_V=%e,norm_W=%e\n", (double)nrm2, (double)nrm1));
  PetscCall(PetscPrintf(comm, "orig: F-norm_V=%e,F-norm_W=%e\n", (double)nrm3, (double)nrm4));
  PetscCall(VecNorm(V, NORM_INFINITY, &nrm3));
  PetscCall(VecNorm(W, NORM_INFINITY, &nrm4));
  PetscCall(PetscPrintf(comm, "swapped: F-norm_V=%e,F-norm_W=%e\n", (double)nrm4, (double)nrm3));

  PetscCall(VecDestroy(&V));
  PetscCall(VecDestroy(&W));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      output_file: output/ex34_1.out
      test:
        suffix: standard
      test:
        requires: cuda
        args: -vec_type cuda
        suffix: cuda
      test:
        requires: viennacl
        args: -vec_type viennacl
        suffix: viennacl
      test:
        requires: kokkos_kernels
        args: -vec_type kokkos
        suffix: kokkos
      test:
        requires: hip
        args: -vec_type hip
        suffix: hip

TEST*/
