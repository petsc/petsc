static char help[] = "Tests that MatSetValues_MPISELL() invalidates the device copy when it inserts a new nonzero.\n\n";

/*
  The matrix is assembled and multiplied once so that both local submatrices hold current device
  data, then new nonzeros are inserted into the diagonal block and into the off-diagonal block. If
  MatSetValues_MPISELL() fails to mark the host copy authoritative, MatSeqSELLCUDACopyToGPU() skips
  the transfer and the second MatMult() returns the pre-insertion result.

  The off-diagonal insertion deliberately reuses a global column that another local row already
  references. An unseen column would trigger MatDisAssemble_MPISELL(), which installs a fresh
  submatrix whose offload mask starts out unallocated, and the stale-data path would not be taken.
*/

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         A, B;
  PetscInt    i, rstart, rend, col;
  PetscMPIInt rank, size;
  PetscScalar value = 1.0;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_USER, "This test requires 2 processes");

  /* A is the matrix under test, B an MATMPIAIJ reference that receives the same values */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATSELL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPISELLSetPreallocation(A, 4, NULL, 4, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(B, MATAIJ));
  PetscCall(MatMPIAIJSetPreallocation(B, 4, NULL, 4, NULL));

  /* both matrices gain nonzeros after their first assembly */
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatSetValues(A, 1, &i, 1, &i, &value, INSERT_VALUES));
    PetscCall(MatSetValues(B, 1, &i, 1, &i, &value, INSERT_VALUES));
  }
  /* one off-diagonal entry, so that its global column enters garray */
  col = rank ? 0 : 4;
  PetscCall(MatSetValues(A, 1, &rstart, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatSetValues(B, 1, &rstart, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* pull both submatrices onto the device */
  PetscCall(MatMultEqual(A, B, 4, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() differs before insertion");

  /* new nonzero in the diagonal block */
  i     = rstart + 1;
  col   = rstart + 2;
  value = 2.0;
  PetscCall(MatSetValues(A, 1, &i, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatSetValues(B, 1, &i, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMultEqual(A, B, 4, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() differs after inserting into the diagonal block");

  /* new nonzero in the off-diagonal block, in a column garray already knows */
  i     = rstart + 1;
  col   = rank ? 0 : 4;
  value = 3.0;
  PetscCall(MatSetValues(A, 1, &i, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatSetValues(B, 1, &i, 1, &col, &value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatMultEqual(A, B, 4, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MatMult() differs after inserting into the off-diagonal block");

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -mat_type sell
      output_file: output/empty.out

   test:
      suffix: cuda
      nsize: 2
      requires: cuda !complex
      args: -mat_type sellcuda
      output_file: output/empty.out

   test:
      suffix: hip
      nsize: 2
      requires: hip !complex
      args: -mat_type sellhip
      output_file: output/empty.out

TEST*/
