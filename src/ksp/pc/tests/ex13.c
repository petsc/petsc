static char help[] = "Tests setup-time weighted ASM scaling callbacks.\n";

#include <petscksp.h>

typedef struct {
  PetscInt  calls;
  PetscReal scale, normalizer;
} ScalingCtx;

static PetscErrorCode ComputeScaling(PC pc, PetscInt local, Vec scaling, PetscCtx ctx)
{
  ScalingCtx     *user = (ScalingCtx *)ctx;
  IS             *is;
  const PetscInt *indices;
  PetscInt        n;
  PetscScalar    *values;

  PetscFunctionBeginUser;
  PetscCall(PCASMGetLocalSubdomains(pc, NULL, &is, NULL));
  PetscCall(ISGetLocalSize(is[local], &n));
  PetscCall(ISGetIndices(is[local], &indices));
  PetscCall(VecGetArray(scaling, &values));
  for (PetscInt i = 0; i < n; ++i) values[i] = user->scale * (local + 1) * (indices[i] + 1) / user->normalizer;
  PetscCall(VecRestoreArray(scaling, &values));
  PetscCall(ISRestoreIndices(is[local], &indices));
  ++user->calls;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMatApply(PC, Vec, PetscReal, PetscInt, PetscScalar);

static PetscErrorCode CheckApply(PC pc, Vec x, Vec y, PetscReal scale, PetscInt nrhs)
{
  const PetscScalar *values;
  PetscInt           start, end;

  PetscFunctionBeginUser;
  PetscCall(PCApply(pc, x, y));
  PetscCall(VecGetOwnershipRange(y, &start, &end));
  PetscCall(VecGetArrayRead(y, &values));
  for (PetscInt i = start; i < end; ++i) PetscCheck(PetscAbsScalar(values[i - start] - scale * (i + 1)) < 100 * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect weighted correction");
  PetscCall(VecRestoreArrayRead(y, &values));
  if (nrhs) PetscCall(CheckMatApply(pc, x, scale, nrhs, 0.0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckMatApply(PC pc, Vec x, PetscReal scale, PetscInt nrhs, PetscScalar coupling)
{
  Mat                X, Y;
  PetscScalar       *values;
  const PetscScalar *rhs, *result;
  PetscInt           m, n, start, ldx, ldy;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(x, &m));
  PetscCall(VecGetSize(x, &n));
  PetscCall(VecGetOwnershipRange(x, &start, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, n, nrhs, NULL, &X));
  PetscCall(MatDuplicate(X, MAT_DO_NOT_COPY_VALUES, &Y));
  PetscCall(MatDenseGetLDA(X, &ldx));
  PetscCall(MatDenseGetLDA(Y, &ldy));
  PetscCall(MatDenseGetArray(X, &values));
  for (PetscInt j = 0; j < nrhs; ++j)
    for (PetscInt i = 0; i < m; ++i) values[i + j * ldx] = (start + i + 1) * (j + 1) - 2 * j;
  PetscCall(MatDenseRestoreArray(X, &values));
  PetscCall(MatDenseGetArrayRead(X, &rhs));
  for (PetscInt transpose = 0; transpose < 2; ++transpose) {
    if (transpose) PetscCall(PCMatApplyTranspose(pc, X, Y));
    else PetscCall(PCMatApply(pc, X, Y));
    PetscCall(MatDenseGetArrayRead(Y, &result));
    for (PetscInt j = 0; j < nrhs; ++j) {
      for (PetscInt i = 0; i < m; ++i) {
        PetscScalar expected = (start + i + 1) * rhs[i + j * ldx];

        // Blocks [1 coupling; 0 1] stay within each rank's four rows; weights are proportional to the global row + 1.
        if (transpose && i % 2) expected -= coupling * (start + i) * rhs[i - 1 + j * ldx];
        if (!transpose && !(i % 2)) expected -= coupling * (start + i + 1) * rhs[i + 1 + j * ldx];
        expected *= scale;
        PetscCheck(PetscAbsScalar(result[i + j * ldy] - expected) <= 100 * PETSC_MACHINE_EPSILON * PetscMax(1.0, PetscAbsScalar(expected)), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Incorrect weighted matrix correction");
      }
    }
    PetscCall(MatDenseRestoreArrayRead(Y, &result));
  }
  PetscCall(MatDenseRestoreArrayRead(X, &rhs));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&Y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PC          pc;
  Mat         A;
  Vec         x, y, *weights;
  IS         *is;
  PetscInt    nlocal, n, start, end, nrhs = 0;
  PetscMPIInt rank, size;
  ScalingCtx  ctx = {0, 2.0, 0.0};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCheck(nrhs >= 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Number of right-hand sides must be nonnegative");
  nlocal         = nrhs ? 1 : rank + 1;
  n              = 4 * size;
  ctx.normalizer = nrhs ? size : size * (size + 1.0) * (size + 2.0) / 6.0;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 4, n, n, nrhs ? 2 : 1, NULL, 0, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) PetscCall(MatSetValue(A, i, i, 1.0, INSERT_VALUES));
  if (nrhs)
    for (PetscInt i = start; i < end; i += 2) PetscCall(MatSetValue(A, i, i + 1, 0.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(x, 1.0));
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCASM));
  PetscCall(PCASMSetType(pc, PC_ASM_WEIGHTED));
  PetscCall(PCASMWeightedSetComputeScaling(pc, ComputeScaling, &ctx));
  for (PetscInt reset = 0; reset < 2; ++reset) {
    PetscCall(PCSetOperators(pc, A, A));
    PetscCall(PetscMalloc1(nlocal, &is));
    for (PetscInt i = 0; i < nlocal; ++i) PetscCall(ISCreateStride(PETSC_COMM_SELF, n, n - 1, -1, &is[i]));
    PetscCall(PCASMSetLocalSubdomains(pc, nlocal, is, NULL));
    for (PetscInt i = 0; i < nlocal; ++i) PetscCall(ISDestroy(&is[i]));
    PetscCall(PetscFree(is));
    ctx.calls = 0;
    ctx.scale = 2.0;
    PetscCall(CheckApply(pc, x, y, 2.0, nrhs)); // implicit setup, including after reset
    PetscCall(CheckApply(pc, x, y, 2.0, nrhs));
    PetscCheck(ctx.calls == nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Callback repeated without setup");
    ctx.scale = 6.0;
    PetscCall(MatScale(A, 2.0));
    PetscCall(CheckApply(pc, x, y, 3.0, nrhs)); // matrix change recomputes the weights
    PetscCheck(ctx.calls == 2 * nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Callback did not run on rebuild");
    PetscCall(PCASMWeightedGetScaling(pc, NULL, &weights));
    for (PetscInt i = 0; i < nlocal; ++i) PetscCall(VecSet(weights[i], 0.0));
    PetscCall(PCASMWeightedSetScaling(pc, nlocal, weights));
    PetscCall(CheckApply(pc, x, y, 0.0, nrhs)); // explicit weights apply until the next rebuild
    PetscCall(MatScale(A, 0.5));
    PetscCall(CheckApply(pc, x, y, 6.0, nrhs));
    PetscCheck(ctx.calls == 3 * nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Callback did not replace explicit weights");
    if (!reset) PetscCall(PCReset(pc));
  }
  PetscCall(PCASMWeightedSetComputeScaling(pc, NULL, NULL));
  PetscCall(MatScale(A, 2.0));
  PetscCall(CheckApply(pc, x, y, 3.0, nrhs)); // disabling preserves the last weights
  PetscCheck(ctx.calls == 3 * nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Disabled callback was called");
  if (nrhs) {
    PetscScalar coupling = 0.25;

#if PetscDefined(USE_COMPLEX)
    coupling += 0.125 * PETSC_i;
#endif
    for (PetscInt i = start; i < end; i += 2) PetscCall(MatSetValue(A, i, i + 1, 2.0 * coupling, INSERT_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(CheckMatApply(pc, x, 3.0, nrhs, coupling));
    PetscCheck(ctx.calls == 3 * nlocal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Disabled callback was called after operator change");
  }
  PetscCall(PCDestroy(&pc));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    nsize: {{1 2}}
    output_file: output/empty.out

  test:
    suffix: mat
    nsize: {{1 2}}
    args: -nrhs {{1 3}}
    output_file: output/empty.out

TEST*/
