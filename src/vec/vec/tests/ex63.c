static char help[] = "Tests VecExp() and VecSqrtAbs().\n\n";

#include <petscvec.h>

static PetscErrorCode CheckExp(Vec v, PetscInt n, PetscScalar *arr, PetscScalar value)
{
  const PetscReal    rtol = 1e-10, atol = PETSC_SMALL;
  const PetscScalar *varr;

  PetscFunctionBeginUser;
  PetscCall(VecSet(v, value));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));
  PetscCall(VecExp(v));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));

  for (PetscInt i = 0; i < n; ++i) arr[i] = PetscExpScalar(value);
  PetscCall(VecGetArrayRead(v, &varr));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscScalar lhs = varr[i];
    const PetscScalar rhs = arr[i];

    if (!PetscIsCloseAtTolScalar(lhs, rhs, rtol, atol)) {
      const PetscReal lhs_r = PetscRealPart(lhs);
      const PetscReal lhs_i = PetscImaginaryPart(lhs);
      const PetscReal rhs_r = PetscRealPart(rhs);
      const PetscReal rhs_i = PetscImaginaryPart(rhs);

      PetscCheck(PetscIsCloseAtTol(lhs_r, rhs_r, rtol, atol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Real component actual[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g", i, (double)lhs_r, i, (double)rhs_r);
      PetscCheck(PetscIsCloseAtTol(lhs_i, rhs_i, rtol, atol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Imaginary component actual[%" PetscInt_FMT "] %g != expected[%" PetscInt_FMT "] %g", i, (double)lhs_i, i, (double)rhs_i);
    }
  }
  PetscCall(VecRestoreArrayRead(v, &varr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Compare VecSqrtAbs() on v with VecSqrtAbs() on a host VECMPI vector holding the same entries,
   which are signed so the absolute value matters, and complex in complex builds
*/
static PetscErrorCode CheckSqrtAbs(Vec v)
{
  const PetscReal    rtol = 1e-10, atol = PETSC_SMALL;
  Vec                w;
  PetscInt           n, rstart;
  const PetscScalar *varr, *warr;

  PetscFunctionBeginUser;
  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(VecGetOwnershipRange(v, &rstart, NULL));
  PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)v), n, PETSC_DETERMINE, &w));
  for (PetscInt i = rstart; i < rstart + n; ++i) {
    PetscScalar value = (i % 2 ? -1.0 : 1.0) * (PetscReal)(3 * i + 2);

#if PetscDefined(USE_COMPLEX)
    value += PETSC_i * (PetscReal)(i - 4);
#endif
    PetscCall(VecSetValue(v, i, value, INSERT_VALUES));
    PetscCall(VecSetValue(w, i, value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(v));
  PetscCall(VecAssemblyEnd(v));
  PetscCall(VecAssemblyBegin(w));
  PetscCall(VecAssemblyEnd(w));
  PetscCall(VecSqrtAbs(v));
  PetscCall(VecSqrtAbs(w));
  PetscCall(VecViewFromOptions(v, NULL, "-vec_view"));

  PetscCall(VecGetArrayRead(v, &varr));
  PetscCall(VecGetArrayRead(w, &warr));
  for (PetscInt i = 0; i < n; ++i)
    PetscCheck(PetscIsCloseAtTolScalar(varr[i], warr[i], rtol, atol), PETSC_COMM_SELF, PETSC_ERR_PLIB, "VecSqrtAbs() actual[%" PetscInt_FMT "] %g != host %g", rstart + i, (double)PetscRealPart(varr[i]), (double)PetscRealPart(warr[i]));
  PetscCall(VecRestoreArrayRead(w, &warr));
  PetscCall(VecRestoreArrayRead(v, &varr));
  PetscCall(VecDestroy(&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec          v;
  PetscInt     n;
  PetscScalar *arr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &v));
  PetscCall(VecSetSizes(v, 10, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(v));

  PetscCall(VecGetLocalSize(v, &n));
  PetscCall(PetscMalloc1(n, &arr));

  PetscCall(CheckExp(v, n, arr, 0.0));
  PetscCall(CheckExp(v, n, arr, 1.0));
  PetscCall(CheckExp(v, n, arr, -1.0));
  PetscCall(CheckSqrtAbs(v));

  PetscCall(PetscFree(arr));
  PetscCall(VecDestroy(&v));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: output/empty.out
    nsize: {{1 2}}
    test:
      suffix: standard
      args: -vec_type standard
    test:
      suffix: viennacl
      requires: viennacl
      args: -vec_type viennacl
    test:
      suffix: cuda
      requires: cuda
      args: -vec_type cuda
    test:
      suffix: hip
      requires: hip
      args: -vec_type hip
    test:
      suffix: kokkos
      requires: kokkos kokkos_kernels
      args: -vec_type kokkos

TEST*/
