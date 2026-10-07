static const char help[] = "Compare LMVM updates, products, and solves on nonblocking contexts with host results.\n";

#include <petscksp.h>
#include <petscdevice.h>

static PetscErrorCode CheckResult(Vec actual, Vec expected)
{
  const PetscScalar *a, *e;
  PetscInt           n;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(actual, &n));
  PetscCall(VecGetArrayRead(actual, &a));
  PetscCall(VecGetArrayRead(expected, &e));
  for (PetscInt i = 0; i < n; ++i)
    PetscCheck(PetscAbsScalar(a[i] - e[i]) <= 500 * PETSC_MACHINE_EPSILON * PetscMax(1.0, PetscAbsScalar(e[i])), PETSC_COMM_SELF, PETSC_ERR_PLIB, "LMVM result differs at local entry %" PetscInt_FMT ": error %g", i, (double)PetscAbsScalar(a[i] - e[i]));
  PetscCall(VecRestoreArrayRead(expected, &e));
  PetscCall(VecRestoreArrayRead(actual, &a));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestLMVM(Vec x, PetscDeviceContext dctx)
{
  Mat          B, reference;
  Vec          f, result, hx, hf, expected;
  PetscInt     n, N, start;
  PetscScalar *xa, *fa;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &n));
  PetscCall(VecGetSize(x, &N));
  PetscCall(VecGetOwnershipRange(x, &start, NULL));
  PetscCall(VecDuplicate(x, &f));
  PetscCall(VecDuplicate(x, &result));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &hx));
  PetscCall(VecSetSizes(hx, n, N));
  PetscCall(VecSetType(hx, VECSTANDARD));
  PetscCall(VecDuplicate(hx, &hf));
  PetscCall(VecDuplicate(hx, &expected));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatSetSizes(B, n, n, N, N));
  PetscCall(MatSetType(B, MATLMVMDBFGS));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatLMVMSetHistorySize(B, 3));
  PetscCall(MatLMVMAllocate(B, x, f));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &reference));
  PetscCall(MatSetSizes(reference, n, n, N, N));
  PetscCall(MatSetType(reference, MATLMVMDBFGS));
  PetscCall(MatSetFromOptions(reference));
  PetscCall(MatLMVMSetHistorySize(reference, 3));
  PetscCall(MatLMVMAllocate(reference, hx, hf));
  PetscCall(VecSet(hx, 0));
  for (PetscInt update = 0; update < 8; ++update) {
    PetscCall(VecGetArray(hx, &xa));
    PetscCall(VecGetArray(hf, &fa));
    for (PetscInt i = 0; i < n; ++i) {
      xa[i] += PetscSinReal((start + i + 1) * (update + 1));
      fa[i] = (2.0 + (PetscReal)(start + i) / N) * xa[i];
    }
    PetscCall(VecRestoreArray(hf, &fa));
    PetscCall(VecRestoreArray(hx, &xa));
    PetscCall(VecCopy(hx, x));
    PetscCall(VecCopy(hf, f));
    PetscCall(MatLMVMUpdate(reference, hx, hf));
    PetscCall(PetscDeviceContextDelay(dctx, 0.02));
    PetscCall(MatLMVMUpdate(B, x, f));
    // Repeated applications exercise cached products as well as new history.
    for (PetscInt repeat = 0; repeat < 2; ++repeat) {
      for (PetscInt solve = 0; solve < 2; ++solve) {
        if (solve) PetscCall(MatSolve(reference, hf, expected));
        else PetscCall(MatMult(reference, hf, expected));
        PetscCall(PetscDeviceContextDelay(dctx, 0.02));
        if (solve) PetscCall(MatSolve(B, f, result));
        else PetscCall(MatMult(B, f, result));
        PetscCall(PetscDeviceContextSynchronize(dctx));
        PetscCall(CheckResult(result, expected));
      }
    }
  }
  PetscCall(MatDestroy(&reference));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&expected));
  PetscCall(VecDestroy(&hf));
  PetscCall(VecDestroy(&hx));
  PetscCall(VecDestroy(&result));
  PetscCall(VecDestroy(&f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec                   x;
  PetscDeviceContext    saved, dctx;
  const PetscStreamType streams[] = {PETSC_STREAM_NONBLOCKING, PETSC_STREAM_NONBLOCKING_WITH_BARRIER};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(KSPInitializePackage());
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 8));
  PetscCall(VecSetFromOptions(x));
  PetscCall(PetscDeviceContextGetCurrentContext(&saved));
  PetscCall(PetscDeviceContextSynchronize(saved));
  for (size_t k = 0; k < PETSC_STATIC_ARRAY_LENGTH(streams); ++k) {
    PetscCall(PetscDeviceContextDuplicate(saved, &dctx));
    PetscCall(PetscDeviceContextSetStreamType(dctx, streams[k]));
    PetscCall(PetscDeviceContextSetUp(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(dctx));
    PetscCall(TestLMVM(x, dctx));
    PetscCall(PetscDeviceContextSynchronize(dctx));
    PetscCall(PetscDeviceContextSetCurrentContext(saved));
    PetscCall(PetscDeviceContextDestroy(&dctx));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    nsize: {{1 2}}
    output_file: output/empty.out
    args: -mat_type {{lmvmdbfgs lmvmddfp lmvmdqn lmvmbfgs lmvmdfp}}
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip !complex
      args: -vec_type hip

TEST*/
