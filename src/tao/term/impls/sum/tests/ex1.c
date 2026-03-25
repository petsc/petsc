const char help[] = "Coverage tests for TAOTERMSUM";

#include <petsctaoterm.h>

int main(int argc, char **argv)
{
  TaoTerm               sub_a, sub_b, sub_c, sum;
  PetscInt              n_a = 10, n_b = 11, n_c = 12;
  PetscInt              k_a = 9, k_c = 8;
  PetscInt              N, K;
  Mat                   map_b, map_c;
  MPI_Comm              comm;
  Vec                   sol, params;
  TaoTermParametersMode mode;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;

  PetscCall(TaoTermCreate(comm, &sum));
  PetscCall(TaoTermSetType(sum, TAOTERMSUM));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_a));
  PetscCall(TaoTermSetParametersMode(sub_a, TAOTERM_PARAMETERS_OPTIONAL));
  PetscCall(TaoTermSetSolutionSizes(sub_a, PETSC_DECIDE, n_a, 1));
  PetscCall(TaoTermSetParametersSizes(sub_a, PETSC_DECIDE, k_a, 1));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 2.0, sub_a, NULL, NULL));
  PetscCall(TaoTermDestroy(&sub_a));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_b));
  PetscCall(TaoTermSetParametersMode(sub_b, TAOTERM_PARAMETERS_NONE));
  PetscCall(TaoTermSetSolutionSizes(sub_b, PETSC_DECIDE, n_b, 1));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, n_b, n_a, NULL, &map_b));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 3.0, sub_b, map_b, NULL));
  PetscCall(MatDestroy(&map_b));
  PetscCall(TaoTermDestroy(&sub_b));

  PetscCall(TaoTermCreateShell(comm, NULL, NULL, &sub_c));
  PetscCall(TaoTermSetParametersMode(sub_c, TAOTERM_PARAMETERS_REQUIRED));
  PetscCall(TaoTermSetSolutionSizes(sub_c, PETSC_DECIDE, n_c, 1));
  PetscCall(TaoTermSetParametersSizes(sub_c, PETSC_DECIDE, k_c, 1));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, n_c, n_a, NULL, &map_c));
  PetscCall(TaoTermSumAddTerm(sum, NULL, 4.0, sub_c, map_c, NULL));
  PetscCall(MatDestroy(&map_c));
  PetscCall(TaoTermDestroy(&sub_c));

  PetscCall(TaoTermSetUp(sum));
  PetscCall(TaoTermGetParametersMode(sum, &mode));
  PetscCheck(mode == TAOTERM_PARAMETERS_REQUIRED, comm, PETSC_ERR_PLIB, "wrong parameters mode");

  PetscCall(TaoTermCreateSolutionVec(sum, &sol));
  PetscCall(TaoTermCreateParametersVec(sum, &params));
  PetscCall(VecGetSize(sol, &N));
  PetscCall(VecGetSize(params, &K));

  PetscCheck(N == n_a, comm, PETSC_ERR_PLIB, "wrong solution size");
  PetscCheck(K == k_a + k_c, comm, PETSC_ERR_PLIB, "wrong parameters size");

  PetscCall(VecDestroy(&params));
  PetscCall(VecDestroy(&sol));
  PetscCall(TaoTermDestroy(&sum));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out

TEST*/
