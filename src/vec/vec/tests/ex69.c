static const char help[] = "Test VecMAXPY() coefficient lifetime with short local vectors.\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  Vec          x, y, ref, ys[9];
  PetscScalar *alpha;
  struct {
    PetscReal   padding;
    PetscScalar values[9];
  }                 *host;
  const PetscScalar *a, *r;
  const PetscInt     counts[] = {1, 2, 3, 4, 7, 8, 9};
  PetscInt           n        = 1;
  PetscMPIInt        rank;
  PetscBool          empty = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-empty_rank", &empty, NULL));
  if (empty && rank) n = 0;
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(y, 2));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &ref));
  PetscCall(VecSetSizes(ref, n, PETSC_DECIDE));
  PetscCall(VecSetType(ref, VECSTANDARD));
  // A scalar array inside a C struct need not satisfy CUDA/HIP complex alignment.
  PetscCall(PetscMalloc1(1, &host));
  alpha = host->values;
  for (PetscInt j = 0; j < 9; ++j) ys[j] = y;
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(counts); ++k) {
    PetscScalar expected = 0;
    PetscInt    nv       = counts[k];

    PetscCall(VecSet(x, 0));
    for (PetscInt i = 0; i < 4; ++i) {
      for (PetscInt j = 0; j < nv; ++j) {
        alpha[j] = i + j + 1;
#if PetscDefined(USE_COMPLEX)
        alpha[j] += PETSC_i * (j + 1);
#endif
        expected += 2 * alpha[j];
      }
      PetscCall(VecMAXPY(x, nv, alpha, ys));
    }
    for (PetscInt j = 0; j < nv; ++j) alpha[j] = -100;
    PetscCall(VecSet(ref, expected));
    PetscCall(VecGetArrayRead(x, &a));
    PetscCall(VecGetArrayRead(ref, &r));
    for (PetscInt i = 0; i < n; ++i) PetscCheck(PetscAbsScalar(a[i] - r[i]) <= 10 * PETSC_MACHINE_EPSILON * PetscMax(PetscAbsScalar(r[i]), 1), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect VecMAXPY() result for %" PetscInt_FMT " coefficients", nv);
    PetscCall(VecRestoreArrayRead(ref, &r));
    PetscCall(VecRestoreArrayRead(x, &a));
  }
  PetscCall(PetscFree(host));
  PetscCall(VecDestroy(&ref));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out
    args: -empty_rank {{0 1}}
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip
      args: -vec_type hip

TEST*/
