const char help[] = "Coverage tests for TAOTERMSHELL";

#include <petsctaoterm.h>

static PetscErrorCode TaoTermCreateSolutionVec_Test(TaoTerm term, Vec *solution)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &A));
  PetscCall(MatCreateVecs(A, NULL, solution));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermCreateParametersVec_Test(TaoTerm term, Vec *params)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &A));
  PetscCall(MatCreateVecs(A, params, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermView_Test(TaoTerm term, PetscViewer viewer)
{
  PetscFunctionBeginUser;
  PetscCall(PetscViewerASCIIPrintf(viewer, "TaoTermView_Test()\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjective_Test(TaoTerm term, Vec x, Vec params, PetscReal *value)
{
  Mat A;
  Vec r;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &A));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(MatMult(A, params, r));
  PetscCall(VecDotRealPart(x, r, value));
  PetscCall(VecDestroy(&r));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeGradient_Test(TaoTerm term, Vec x, Vec params, Vec g)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &A));
  PetscCall(MatMult(A, params, g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoTermComputeObjectiveAndGradient_Test(TaoTerm term, Vec x, Vec params, PetscReal *value, Vec g)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(TaoTermShellGetContext(term, &A));
  PetscCall(MatMult(A, params, g));
  PetscCall(VecDotRealPart(x, g, value));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testShell(MPI_Comm comm, PetscBool separate)
{
  PetscRandom rand;
  Mat         A;
  TaoTerm     term;
  PetscInt    m = 23, n = 11;
  Vec         x, params, g;
  PetscInt    test_m, test_n;
  PetscReal   value, g_norm;

  PetscFunctionBeginUser;
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, m, n, NULL, &A));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetRandom(A, rand));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(TaoTermCreateShell(comm, A, NULL, &term));
  PetscCall(TaoTermSetParametersMode(term, TAOTERM_PARAMETERS_REQUIRED));

  if (separate) {
    PetscCall(MatCreateVecs(A, &params, &x));
    PetscCall(TaoTermSetSolutionTemplate(term, x));
    PetscCall(TaoTermSetParametersTemplate(term, params));
    PetscCall(VecDestroy(&params));
    PetscCall(VecDestroy(&x));
  } else {
    PetscCall(TaoTermShellSetCreateSolutionVec(term, TaoTermCreateSolutionVec_Test));
    PetscCall(TaoTermShellSetCreateParametersVec(term, TaoTermCreateParametersVec_Test));
  }

  PetscCall(TaoTermSetUp(term));

  PetscCall(TaoTermGetSolutionSizes(term, NULL, &test_m, NULL));
  PetscCall(TaoTermGetParametersSizes(term, NULL, &test_n, NULL));
  PetscCheck(test_m == m, comm, PETSC_ERR_PLIB, "Inconsistent solution size");
  PetscCheck(test_n == n, comm, PETSC_ERR_PLIB, "Inconsistent parameters size");

  if (separate) {
    PetscCall(TaoTermShellSetObjective(term, TaoTermComputeObjective_Test));
    PetscCall(TaoTermShellSetGradient(term, TaoTermComputeGradient_Test));
  } else {
    PetscCall(TaoTermShellSetObjectiveAndGradient(term, TaoTermComputeObjectiveAndGradient_Test));
    PetscCall(TaoTermShellSetView(term, TaoTermView_Test));
  }

  PetscCall(TaoTermView(term, PETSC_VIEWER_STDOUT_(comm)));

  PetscCall(TaoTermCreateSolutionVec(term, &x));
  PetscCall(TaoTermCreateParametersVec(term, &params));

  PetscCall(VecSetRandom(x, rand));
  PetscCall(VecSetRandom(params, rand));
  PetscCall(VecDuplicate(x, &g));
  PetscCall(TaoTermComputeObjectiveAndGradient(term, x, params, &value, g));
  PetscCall(VecNorm(g, NORM_2, &g_norm));
  PetscCall(PetscPrintf(comm, "objective: %g, gradient norm %g\n", (double)value, (double)g_norm));

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&params));
  PetscCall(VecDestroy(&x));
  PetscCall(TaoTermDestroy(&term));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rand));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(testShell(PETSC_COMM_WORLD, PETSC_TRUE));
  PetscCall(testShell(PETSC_COMM_WORLD, PETSC_FALSE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0

TEST*/
