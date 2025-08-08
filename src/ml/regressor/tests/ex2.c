static char help[] = "Tests basic creation and destruction of PetscRegressor objects.\n\n";

/*
    Uses PetscRegressor to train a linear model (that is, linear in its coefficients)
    for a quadratic polynomial data-fitting problem. This is example 3.2 in the first (1996) edition of Michael
    T. Heath's "Scientific Computing: An Introductory Survey" textbook.
    This example and ex1.c are essentially the same, except the input arrays are mean-centered in ex1.c
    and are not in ex2.c. (The data in ex2.c correspond to the data as presented in Heath's example.)
*/

#include <petscregressor.h>

int main(int argc, char **args)
{
  PetscRegressor regressor;
  PetscMPIInt    rank;
  Mat            X;
  Vec            y, y_predicted, coefficients;
  PetscScalar    intercept;
  /* y_array[] and X_array[] are NOT mean-centered; in ex1.c they are! */
  PetscScalar y_array[5]  = {1.0, 0.5, 0, 0.5, 2};
  PetscScalar X_array[10] = {-1.00000, 1.00000, -0.50000, 0.25000, 0.00000, 0.00000, 0.50000, 0.25000, 1.00000, 1.00000};
  PetscInt    rows_ix[5]  = {0, 1, 2, 3, 4};
  PetscInt    cols_ix[2]  = {0, 1};

  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, PETSC_DECIDE, 5));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecDuplicate(y, &y_predicted));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, PETSC_DECIDE, PETSC_DECIDE, 5, 2));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetUp(X));

  if (!rank) {
    PetscCall(VecSetValues(y, 5, rows_ix, y_array, INSERT_VALUES));
    PetscCall(MatSetValues(X, 5, rows_ix, 2, cols_ix, X_array, ADD_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscRegressorCreate(PETSC_COMM_WORLD, &regressor));
  PetscCall(PetscRegressorSetType(regressor, PETSCREGRESSORLINEAR));
  PetscCall(PetscRegressorSetFromOptions(regressor));
  PetscCall(PetscRegressorFit(regressor, X, y));
  PetscCall(PetscRegressorPredict(regressor, X, y_predicted));
  PetscCall(PetscRegressorLinearGetIntercept(regressor, &intercept));
  PetscCall(PetscRegressorLinearGetCoefficients(regressor, &coefficients));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Intercept is %lf\n", (double)intercept));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Coefficients are\n"));
  PetscCall(VecView(coefficients, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Predicted values are\n"));
  PetscCall(VecView(y_predicted, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&y_predicted));
  PetscCall(PetscRegressorDestroy(&regressor));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

  test:
    suffix: prefix_tao
    args: -regressor_view

  test:
    suffix: prefix_ksp
    args: -regressor_view -regressor_linear_use_ksp -regressor_linear_ksp_lsqr_monitor

  test:
    requires: suitesparse
    suffix: prefix_ksp_qr
    args: -regressor_view -regressor_linear_use_ksp -regressor_linear_ksp_lsqr_monitor -regressor_linear_pc_type qr regressor_linear_pc_factor_mat_solver_type spqr
    TODO: Matrix of type composite does not support checking for transpose

TEST*/
