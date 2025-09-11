static char help[] = "Tests resetting preallocation after filling the full sparsity pattern";

#include <petscmat.h>

PetscErrorCode Assemble(Mat mat)
{
  PetscInt    idx[4], i;
  PetscScalar vals[16];
  int         rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  for (i = 0; i < 16; ++i) vals[i] = 1;
  if (rank == 0) {
    // element 0
    idx[0] = 0;
    idx[1] = 1;
    idx[2] = 2;
    idx[3] = 3;
    PetscCall(MatSetValues(mat, 4, idx, 4, idx, vals, ADD_VALUES));
    // element 1
    idx[0] = 3;
    idx[1] = 2;
    idx[2] = 4;
    idx[3] = 5;
    PetscCall(MatSetValues(mat, 4, idx, 4, idx, vals, ADD_VALUES));
  } else {
    // element 2
    idx[0] = 6;
    idx[1] = 0;
    idx[2] = 3;
    idx[3] = 7;
    PetscCall(MatSetValues(mat, 4, idx, 4, idx, vals, ADD_VALUES));
    // element 3
    idx[0] = 7;
    idx[1] = 3;
    idx[2] = 5;
    idx[3] = 8;
    PetscCall(MatSetValues(mat, 4, idx, 4, idx, vals, ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat mat;
  int rank;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  if (rank == 0) PetscCall(MatSetSizes(mat, 6, 6, PETSC_DETERMINE, PETSC_DETERMINE));
  else PetscCall(MatSetSizes(mat, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(mat));
  if (rank == 0) {
    PetscInt ndz[6], noz[6];
    ndz[0] = 4;
    noz[0] = 2;
    ndz[1] = 4;
    noz[1] = 0;
    ndz[2] = 6;
    noz[2] = 0;
    ndz[3] = 6;
    noz[3] = 3;
    ndz[4] = 4;
    noz[4] = 0;
    ndz[5] = 4;
    noz[5] = 2;
    PetscCall(MatMPIAIJSetPreallocation(mat, 0, ndz, 0, noz));
  } else {
    PetscInt ndz[3], noz[3];
    ndz[0] = 2;
    noz[0] = 2;
    ndz[1] = 3;
    noz[1] = 3;
    ndz[2] = 2;
    noz[2] = 2;
    PetscCall(MatMPIAIJSetPreallocation(mat, 0, ndz, 0, noz));
  }
  PetscCall(MatSetUp(mat));
  PetscCall(Assemble(mat));
  PetscCall(MatView(mat, NULL));
  PetscCall(MatResetPreallocation(mat));
  PetscCall(Assemble(mat));
  PetscCall(MatView(mat, NULL));
  PetscCall(MatDestroy(&mat));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2

TEST*/
