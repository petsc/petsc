static char help[] = "Tests basic creation and destruction of PetscDA objects, and a simple ETKF analysis step.\n\n";
#include <petscda.h>

int main(int argc, char **argv)
{
  PetscDA     da;
  Mat         H;
  Vec         x_true, y_obs, obs_error_var;
  Vec         x_mean_forecast, x_mean_analysis;
  PetscInt    state_size = 10, obs_size = 10, ensemble_size = 20;
  PetscRandom rng;
  PetscInt    i;
  PetscReal   norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Create the DA object */
  PetscCall(PetscDACreate(PETSC_COMM_WORLD, &da));
  PetscCall(PetscDASetType(da, PETSCDAETKF));
  PetscCall(PetscDASetSizes(da, state_size, obs_size));
  PetscCall(PetscDAEnsembleSetSize(da, ensemble_size));
  PetscCall(PetscDASetFromOptions(da));
  PetscCall(PetscDASetUp(da));

  /* Initialize random number generator */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rng));
  PetscCall(PetscRandomSetFromOptions(rng));

  /* Create identity observation matrix H (obs_size x state_size) */
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, obs_size, state_size, 1, NULL, 0, NULL, &H));
  PetscCall(MatSetFromOptions(H));
  PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  PetscCall(MatShift(H, 1.0));

  /* Create vectors using MatCreateVecs from H */
  PetscCall(MatCreateVecs(H, &x_true, &y_obs));
  PetscCall(VecSet(x_true, 1.0)); /* True state is all 1s */

  PetscCall(VecDuplicate(y_obs, &obs_error_var));
  PetscCall(VecSet(obs_error_var, 0.1)); /* Observation error variance */
  PetscCall(PetscDASetObsErrorVariance(da, obs_error_var));

  /* Create synthetic observation: y = x_true (identity observation, no noise for this simple test) */
  PetscCall(VecCopy(x_true, y_obs));

  /* Initialize ensemble with some spread around 0 (far from truth 1.0) */
  for (i = 0; i < ensemble_size; i++) {
    Vec member;
    PetscCall(VecDuplicate(x_true, &member));
    PetscCall(VecSetRandom(member, rng)); /* Uniform random [0, 1] */
    PetscCall(PetscDAEnsembleSetMember(da, i, member));
    PetscCall(VecDestroy(&member));
  }

  /* Compute forecast mean before analysis */
  PetscCall(VecDuplicate(x_true, &x_mean_forecast));
  PetscCall(PetscDAEnsembleComputeMean(da, x_mean_forecast));

  /* Check forecast error */
  PetscCall(VecAXPY(x_mean_forecast, -1.0, x_true));
  PetscCall(VecNorm(x_mean_forecast, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Forecast error norm: %g\n", (double)norm));

  /* Perform Analysis Step */
  PetscCall(PetscDAEnsembleAnalysis(da, y_obs, H));

  /* Compute analysis mean */
  PetscCall(VecDuplicate(x_true, &x_mean_analysis));
  PetscCall(PetscDAEnsembleComputeMean(da, x_mean_analysis));

  /* Check analysis error */
  PetscCall(VecAXPY(x_mean_analysis, -1.0, x_true));
  PetscCall(VecNorm(x_mean_analysis, NORM_2, &norm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Analysis error norm: %g\n", (double)norm));

  /* The analysis should move the ensemble closer to the observation (truth) */
  /* Since observation error is small (0.1) and prior spread is ~0.08, it should pull towards observation */

  PetscCall(PetscDAViewFromOptions(da, NULL, "-petscda_view"));

  /* Cleanup */
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&x_true));
  PetscCall(VecDestroy(&y_obs));
  PetscCall(VecDestroy(&obs_error_var));
  PetscCall(VecDestroy(&x_mean_forecast));
  PetscCall(VecDestroy(&x_mean_analysis));
  PetscCall(PetscRandomDestroy(&rng));
  PetscCall(PetscDADestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    requires: !complex
    args: -petscda_view

  test:
    suffix: chol
    requires: !complex
    args: -petscda_view -petscda_ensemble_sqrt_type cholesky

TEST*/
