static char help[] = "A placeholder for testing device matrices with over 2 billion nonzeros\n\n";

#include <petscmat.h>

int main(int argc, char **argv)
{
  Mat          A, B;
  Vec          x1, x2, y1, y2;
  PetscInt     m = 1 << 6;
  PetscInt     n = (1 << 6) + 2, nnz; // or m = 1 << 15, n = (1 << 16) + 2 to get > 2 billion nonzeros
  PetscInt    *i, *j;
  PetscScalar *a;
  PetscReal    r1, r2;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // Create a 'dense' SEQAIJ matrix for simplicity, and also to save the row pointer memory cost
  nnz = m * n;
  PetscCall(PetscMalloc3(m + 1, &i, nnz, &j, nnz, &a));

  i[0] = 0;
  for (PetscInt k = 0; k < m; k++) {
    i[k + 1] = i[k] + n;
    for (PetscInt l = 0; l < n; l++) {
      j[i[k] + l] = l;
      a[i[k] + l] = (PetscScalar)(i[k] + l);
    }
  }
  PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF, m, n, i, j, a, &A));

  PetscCall(MatCreateVecs(A, &x1, &y1));
  PetscCall(VecSetRandom(x1, NULL));
  PetscCall(MatMult(A, x1, y1));

  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree3(i, j, a));

  PetscCall(MatCreateVecs(B, &x2, &y2));
  PetscCall(VecCopy(x1, x2));
  PetscCall(MatMult(B, x2, y2));

  PetscCall(VecNorm(y1, NORM_INFINITY, &r1));
  PetscCall(VecAXPY(y2, -1.0, y1));
  PetscCall(VecNorm(y2, NORM_INFINITY, &r2));
  r2 /= r1;

  PetscCheck(r2 < PETSC_SQRT_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatMult wrong with indices beyond 32-bit: relative error %g", (double)r2);

  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&y1));
  PetscCall(VecDestroy(&y2));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  testset:
    nsize: 1
    output_file: output/empty.out

    test:
      requires: kokkos_kernels single defined(PETSC_USE_64BIT_INDICES)
      suffix: kokkos
      args: -mat_type aijkokkos

    test:
      requires: cuda single defined(PETSC_USE_64BIT_INDICES)
      suffix: cuda
      args: -mat_type aijcusparse

    test:
      requires: hip single defined(PETSC_USE_64BIT_INDICES)
      suffix: hip
      args: -mat_type aijhipsparse
TEST*/
