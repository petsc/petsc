static char help[] = "Example of inverting a block diagonal matrix.\n"
                     "\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A, A_inv;
  PetscMPIInt  rank, size;
  PetscInt     M, m, bs, rstart, rend, j, x, y;
  PetscInt    *dnnz;
  PetscScalar *v;
  Vec          X, Y;
  PetscReal    norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "ex184", "Mat");
  M = 8;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_size", &M, NULL));
  bs = 3;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_block_size", &bs, NULL));
  PetscOptionsEnd();

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, M * bs, M * bs));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(PetscMalloc1(m / bs, &dnnz));
  for (j = 0; j < m / bs; j++) dnnz[j] = 1;
  PetscCall(MatXAIJSetPreallocation(A, bs, dnnz, NULL, NULL, NULL));
  PetscCall(PetscFree(dnnz));

  PetscCall(PetscMalloc1(bs * bs, &v));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (j = rstart / bs; j < rend / bs; j++) {
    for (x = 0; x < bs; x++) {
      for (y = 0; y < bs; y++) {
        if (x == y) {
          v[y + bs * x] = 2 * bs;
        } else {
          v[y + bs * x] = -1 * (x < y) - 2 * (x > y);
        }
      }
    }
    PetscCall(MatSetValuesBlocked(A, 1, &j, 1, &j, v, INSERT_VALUES));
  }
  PetscCall(PetscFree(v));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* check that A  = inv(inv(A)) */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A_inv));
  PetscCall(MatSetFromOptions(A_inv));
  PetscCall(MatInvertBlockDiagonalMat(A, A_inv));

  /* Test A_inv * A on a random vector */
  PetscCall(MatCreateVecs(A, &X, &Y));
  PetscCall(VecSetRandom(X, NULL));
  PetscCall(MatMult(A, X, Y));
  PetscCall(VecScale(X, -1));
  PetscCall(MatMultAdd(A_inv, Y, X, X));
  PetscCall(VecNorm(X, NORM_MAX, &norm));
  if (norm > PETSC_SMALL) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error exceeds tolerance.\nInverse of block diagonal A\n"));
    PetscCall(MatView(A_inv, PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&A_inv));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&Y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  test:
    suffix: seqaij
    args: -mat_type seqaij -mat_size 12 -mat_block_size 3
    nsize: 1
  test:
    suffix: seqbaij
    args: -mat_type seqbaij -mat_size 12 -mat_block_size 3
    nsize: 1
  test:
    suffix: mpiaij
    args: -mat_type mpiaij -mat_size 12 -mat_block_size 3
    nsize: 2
  test:
    suffix: mpibaij
    args: -mat_type mpibaij -mat_size 12 -mat_block_size 3
    nsize: 2
TEST*/
