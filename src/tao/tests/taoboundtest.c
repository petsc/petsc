static char help[] = "Tests TAO utilities with one-sided variable bounds.\n";

#include <petsctao.h>

static PetscErrorCode SetSolution(Vec x)
{
  PetscScalar *values;
  PetscInt     i, rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(VecGetArray(x, &values));
  for (i = rstart; i < rend; ++i) values[i - rstart] = i == 0 ? -2.0 : (i == 3 ? 2.0 : 0.0);
  PetscCall(VecRestoreArray(x, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckSolution(Vec x, PetscBool lower)
{
  const PetscScalar *values;
  PetscInt           i, rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(VecGetOwnershipRange(x, &rstart, &rend));
  PetscCall(VecGetArrayRead(x, &values));
  for (i = rstart; i < rend; ++i) {
    PetscScalar expected = i == 0 ? (lower ? -1.0 : -2.0) : (i == 3 ? (lower ? 2.0 : 1.0) : 0.0);

    PetscCheck(values[i - rstart] == expected, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected value at index %" PetscInt_FMT, i);
  }
  PetscCall(VecRestoreArrayRead(x, &values));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestActiveBounds(Vec x, Vec bound, Vec g, Vec s, Vec work, PetscBool lower)
{
  IS        active_lower = NULL, active_upper = NULL, active_fixed = NULL, active = NULL, inactive = NULL;
  IS        active_bound;
  PetscReal bound_tol = 0.0;
  PetscInt  n;

  PetscFunctionBeginUser;
  PetscCall(SetSolution(x));
  PetscCall(VecSet(g, lower ? 1.0 : -1.0));
  PetscCall(VecSet(s, 0.0));
  PetscCall(TaoEstimateActiveBounds(x, lower ? bound : NULL, lower ? NULL : bound, g, s, work, 1.0, &bound_tol, &active_lower, &active_upper, &active_fixed, &active, &inactive));
  active_bound = lower ? active_lower : active_upper;
  PetscCheck(active_bound, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected an active bound");
  PetscCall(ISGetSize(active_bound, &n));
  PetscCheck(n == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected one active bound, got %" PetscInt_FMT, n);
  PetscCheck(!(lower ? active_upper : active_lower), PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unexpected active bound on the unbounded side");
  PetscCall(ISDestroy(&active_lower));
  PetscCall(ISDestroy(&active_upper));
  PetscCall(ISDestroy(&active_fixed));
  PetscCall(ISDestroy(&active));
  PetscCall(ISDestroy(&inactive));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Vec      x, xl, xu, g, s, work;
  PetscInt nDiff;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 4));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(VecDuplicate(x, &s));
  PetscCall(VecDuplicate(x, &work));
  PetscCall(VecSet(xl, -1.0));
  PetscCall(VecSet(xu, 1.0));

  PetscCall(SetSolution(x));
  PetscCall(TaoBoundSolution(x, xl, NULL, 0.0, &nDiff, x));
  PetscCheck(nDiff == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected one lower-bound correction, got %" PetscInt_FMT, nDiff);
  PetscCall(CheckSolution(x, PETSC_TRUE));

  PetscCall(SetSolution(x));
  PetscCall(TaoBoundSolution(x, NULL, xu, 0.0, &nDiff, x));
  PetscCheck(nDiff == 1, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected one upper-bound correction, got %" PetscInt_FMT, nDiff);
  PetscCall(CheckSolution(x, PETSC_FALSE));

  PetscCall(TestActiveBounds(x, xl, g, s, work, PETSC_TRUE));
  PetscCall(TestActiveBounds(x, xu, g, s, work, PETSC_FALSE));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&s));
  PetscCall(VecDestroy(&work));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex

  test:
    output_file: output/empty.out

  test:
    suffix: 2
    nsize: 2
    output_file: output/empty.out

TEST*/
