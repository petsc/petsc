static char help[] = "Compare assembled and sampled shell CUTEst Hessians.\n\
  -cutest_lib filename  Shared library containing the decoded problem\n\
  -cutest_data filename Decoded data file (default OUTSDIF.d)\n\
  -cutest_check_type (matrix|mult)  Compare full matrices or forward and transpose products (default matrix)\n\
  -cutest_check_vectors n          Number of products for each multiplication check (default 8)\n";

#include <petsc/private/taocutestimpl.h>

static PetscErrorCode CheckHessian(Mat H, Mat reference, PetscBool mult, PetscInt nvec, PetscInt point, const char stage[])
{
  Mat       computed, difference;
  PetscReal norm, error;
  PetscBool match;

  PetscFunctionBeginUser;
  for (PetscInt transpose = 0; transpose < 2; ++transpose) {
    if (mult) {
      if (transpose) PetscCall(MatMultTransposeEqual(reference, H, nvec, &match));
      else PetscCall(MatMultEqual(reference, H, nvec, &match));
      PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hessian%s mismatch at point %" PetscInt_FMT " (%s); rerun with -cutest_check_type matrix to display the matrices", transpose ? " transpose" : "", point, stage);
      continue;
    }
    if (transpose) PetscCall(MatComputeOperatorTranspose(H, MATDENSE, &computed));
    else PetscCall(MatComputeOperator(H, MATDENSE, &computed));
    PetscCall(MatDuplicate(computed, MAT_COPY_VALUES, &difference));
    PetscCall(MatNorm(computed, NORM_FROBENIUS, &norm));
    PetscCall(MatAXPY(difference, -1.0, reference, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatNorm(difference, NORM_FROBENIUS, &error));
    match = (PetscBool)(!PetscIsInfOrNanReal(norm) && !PetscIsInfOrNanReal(error) && error <= 100.0 * PETSC_MACHINE_EPSILON * PetscMax(norm, 1.0));
    if (!match) {
      PetscViewer viewer = PETSC_VIEWER_STDOUT_SELF;

      PetscCall(PetscViewerASCIIPrintf(viewer, "Hessian%s mismatch at point %" PetscInt_FMT " (%s)\nComputed Hessian:\n", transpose ? " transpose" : "", point, stage));
      PetscCall(MatView(computed, viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Reference Hessian:\n"));
      PetscCall(MatView(reference, viewer));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Difference (computed - reference):\n"));
      PetscCall(MatView(difference, viewer));
      PetscCall(PetscViewerFlush(viewer));
    }
    PetscCheck(match, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hessian%s mismatch at point %" PetscInt_FMT " (%s): error %g, computed norm %g", transpose ? " transpose" : "", point, stage, (double)error, (double)norm);
    PetscCall(PetscInfo(H, "Point %" PetscInt_FMT " (%s): relative Hessian%s difference %g\n", point, stage, transpose ? " transpose" : "", (double)(error / PetscMax(norm, 1.0))));
    PetscCall(MatDestroy(&computed));
    PetscCall(MatDestroy(&difference));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const char    *comparisons[]               = {"matrix", "mult"};
  char           library[PETSC_MAX_PATH_LEN] = "", data[PETSC_MAX_PATH_LEN] = "OUTSDIF.d";
  PetscDLLibrary dll       = NULL;
  CUTEstCtx      assembled = {0}, shell = {0};
  Vec            X, initial, trial, G;
  Mat            H, S;
  PetscInt       comparison = 0, nvec = 8;
  PetscMPIInt    size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This CUTEst test runs on one MPI process");
  PetscCall(PetscOptionsGetString(NULL, NULL, "-cutest_lib", library, sizeof(library), NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-cutest_data", data, sizeof(data), NULL));
  PetscOptionsBegin(PETSC_COMM_SELF, NULL, "CUTEst Hessian comparison", NULL);
  PetscCall(PetscOptionsEList("-cutest_check_type", "Hessian comparison method", NULL, comparisons, 2, comparisons[comparison], &comparison, NULL));
  PetscCall(PetscOptionsInt("-cutest_check_vectors", "Number of random products", "MatMultEqual", nvec, &nvec, NULL));
  PetscOptionsEnd();
  PetscCheck(nvec > 0, PETSC_COMM_SELF, PETSC_ERR_USER_INPUT, "The number of random products must be positive");
  PetscCall(CUTEstLoadProblem(library, data, &assembled, &dll, &X));
  shell.n = assembled.n;
  PetscCall(CUTEstCreateHessian(X, PETSC_FALSE, &assembled, &H));
  PetscCall(CUTEstCreateHessian(X, PETSC_TRUE, &shell, &S));
  PetscCall(VecDuplicate(X, &initial));
  PetscCall(VecCopy(X, initial));
  PetscCall(VecDuplicate(X, &trial));
  PetscCall(VecDuplicate(X, &G));
  for (PetscInt point = 0; point < 5; ++point) {
    Mat          reference;
    PetscScalar *x;
    PetscReal    f;

    if (!point || point == 4) PetscCall(VecCopy(initial, X));
    else {
      PetscCall(VecGetArrayWrite(X, &x));
      for (PetscInt i = 0; i < assembled.n; ++i) x[i] = point == 1 ? 0.0 : (point == 2 ? 0.125 : -0.25) * (i + 1);
      PetscCall(VecRestoreArrayWrite(X, &x));
    }
    PetscCall(CUTEstFormHessian(NULL, X, H, H, &assembled));
    PetscCall(CUTEstFormHessian(NULL, X, S, S, &shell));
    /* Keep an immutable reference across trial evaluations and solver shifts. */
    if (comparison) PetscCall(MatDuplicate(H, MAT_COPY_VALUES, &reference));
    else PetscCall(MatComputeOperator(S, MATDENSE, &reference));
    PetscCall(CheckHessian(H, reference, (PetscBool)comparison, nvec, point, "assembled"));
    PetscCall(CheckHessian(S, reference, (PetscBool)comparison, nvec, point, "cached shell"));

    PetscCall(VecCopy(X, trial));
    PetscCall(VecShift(trial, 0.375));
    PetscCall(CUTEstFormObjective(NULL, trial, &f, &shell));
    PetscCall(CheckHessian(S, reference, (PetscBool)comparison, nvec, point, "objective trial"));
    PetscCall(CUTEstFormGradient(NULL, trial, G, &shell));
    PetscCall(CheckHessian(S, reference, (PetscBool)comparison, nvec, point, "gradient trial"));
    PetscCall(CUTEstFormObjectiveGradient(NULL, trial, &f, G, &shell));
    PetscCall(CheckHessian(S, reference, (PetscBool)comparison, nvec, point, "objective/gradient trial"));

    PetscCall(MatShift(H, 1.0));
    PetscCall(MatShift(S, 2.0));
    PetscCall(CUTEstFormHessian(NULL, X, H, H, &assembled));
    PetscCall(CUTEstFormHessian(NULL, X, S, S, &shell));
    PetscCall(CheckHessian(H, reference, (PetscBool)comparison, nvec, point, "assembled after shift"));
    PetscCall(CheckHessian(S, reference, (PetscBool)comparison, nvec, point, "shell after shift"));
    PetscCall(MatDestroy(&reference));
  }
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Assembled and shell Hessians and their transposes agree at all 5 points\n"));
  PetscCall(MatDestroy(&H));
  PetscCall(MatDestroy(&S));
  PetscCall(VecDestroy(&shell.hessian_x));
  PetscCall(PetscFree3(assembled.rows, assembled.cols, assembled.values));
  PetscCall(VecDestroy(&initial));
  PetscCall(VecDestroy(&trial));
  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&X));
  PetscCall(CUTEstUnloadProblem(dll));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: cutest double !complex defined(PETSC_HAVE_DYNAMIC_LIBRARIES)

  testset:
    requires: sifdecode
    args: -cutest_check_type {{matrix mult}}
    command: @PYTHON@ ${wPETSC_DIR}/share/petsc/cutest/run.py --petsc-dir ${wPETSC_DIR} --petsc-arch=${petsc_arch} --driver ../cutest --sif ${wPETSC_DIR}/share/petsc/datafiles/cutest/${CUTEST_SIF}.SIF -- ${args} @SUBARGS@
    test:
      suffix: rosenbrock
      output_file: output/cutest_rosenbrock.out
      env: CUTEST_SIF=ROSENBR
    test:
      suffix: polynomial
      output_file: output/cutest_polynomial.out
      env: CUTEST_SIF=COOHESS

TEST*/
