static const char help[] = "Tests VecSum()\n\n";

#include <petscvec.h>

static PetscBool PetscIsCloseAtTolScalar(PetscScalar l, PetscScalar r, PetscReal atol, PetscReal rtol)
{
  return (PetscBool)(PetscIsCloseAtTol(PetscRealPart(l), PetscRealPart(r), atol, rtol) && PetscIsCloseAtTol(PetscImaginaryPart(l), PetscImaginaryPart(r), atol, rtol));
}

static PetscErrorCode CheckVecSumReturn(Vec v, PetscScalar expected)
{
  PetscScalar actual;

  PetscFunctionBegin;
  PetscCall(VecSum(v, &actual));
  PetscCheck(PetscIsCloseAtTolScalar(actual, expected, 1e-12, 0.0), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecSum() returned %g + %gi, expected %g + %gi", (double)PetscRealPart(actual), (double)PetscImaginaryPart(actual), (double)PetscRealPart(expected), (double)PetscImaginaryPart(expected));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscInt     n, lo, N = 10;
  PetscScalar  sum = 0.0;
  Vec          x;
  PetscScalar *array;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecZeroEntries(x));
  PetscCall(CheckVecSumReturn(x, 0.0));

  PetscCall(VecSet(x, 1.0));
  PetscCall(CheckVecSumReturn(x, 10.0));

  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetOwnershipRange(x, &lo, NULL));
  PetscCall(VecGetArrayWrite(x, &array));
  for (PetscInt i = 0; i < n; ++i) array[i] = (PetscScalar)(lo + i);
  PetscCall(VecRestoreArrayWrite(x, &array));

  PetscCall(VecGetSize(x, &N));
  for (PetscInt i = 0; i < N; ++i) sum += (PetscScalar)i;
  PetscCall(CheckVecSumReturn(x, sum));

  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
    test:
      requires: defined(PETSC_USE_SHARED_MEMORY)
      args: -vec_type shared
      suffix: shared
    test:
      requires: viennacl
      args: -vec_type viennacl
      suffix: viennacl
    test:
      requires: kokkos_kernels
      args: -vec_type kokkos
      suffix: kokkos
    test:
      requires: cuda
      args: -vec_type cuda
      suffix: cuda
    test:
      requires: hip
      args: -vec_type hip
      suffix: hip

TEST*/
