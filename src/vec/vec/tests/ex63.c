static char help[] = "Tests VecExp().\n\n";

#include <petscvec.h>

static PetscErrorCode CheckExp(Vec v, PetscInt n, PetscScalar *arr, PetscScalar value)
{
  const PetscReal    rtol = 1e-10, atol = PETSC_SMALL;
  const PetscScalar *varr;

  PetscFunctionBegin;
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

  PetscCall(PetscFree(arr));
  PetscCall(VecDestroy(&v));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
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
      requires: kokkos, kokkos_kernels
      args: -vec_type kokkos

TEST*/
