static char help[] = "Solve a decoded unconstrained CUTEst problem with TAO.\n\
  -cutest_lib filename          Shared library containing the decoded problem\n\
  -cutest_data filename         Decoded data file (default OUTSDIF.d)\n\
  -cutest_hessian (sparse|shell) Sparse Hessian or exact Hessian-vector products\n\
  -cutest_solution_view         View the final solution\n\
  -cutest_result filename       Write solve statistics as CSV instead of a screen summary\n\
See the TAO users manual for configuration, decoding, and running examples.\n";

#include <petsc/private/taocutestimpl.h>

typedef struct {
  PetscErrorCode (*mult)(Mat, Vec, Vec);
  PetscErrorCode (*multtranspose)(Mat, Vec, Vec);
  PetscCount products;
} HessianCounter;

static PetscErrorCode CountHessianMult(Mat H, Vec X, Vec Y)
{
  HessianCounter *counter;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectContainerQuery((PetscObject)H, "CUTEstHessianProducts", &counter));
  PetscCall((*counter->mult)(H, X, Y));
  ++counter->products;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CountHessianMultTranspose(Mat H, Vec X, Vec Y)
{
  HessianCounter *counter;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectContainerQuery((PetscObject)H, "CUTEstHessianProducts", &counter));
  PetscCall((*counter->multtranspose)(H, X, Y));
  ++counter->products;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode WriteResult(Tao tao, Vec X, CUTEstCtx *user, const char filename[], PetscReal initial_f, PetscReal initial_gnorm, const PetscReal initial_calls[], PetscLogDouble elapsed, PetscCount products)
{
  PetscViewer         viewer;
  TaoConvergedReason  reason;
  SNESConvergedReason snes_reason = SNES_CONVERGED_ITERATING;
  TaoType             type;
  const char         *solver, *snes_reason_name = "";
  char                name[256];
  Vec                 G;
  PetscReal           f, gnorm, calls[4], time[4];
  PetscInt            its, rejected = -1;
  PetscBool           is_snes, is_python;

  PetscFunctionBeginUser;
  /* Exclude the final verification evaluation from CUTEst's solve counters. */
  PetscCallCUTEst(CUTEST_ureport, calls, time);
  PetscCheck(!user->hessian_x || (PetscReal)products == calls[3] - initial_calls[3], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Hessian multiplication count disagrees with CUTEst");
  PetscCall(VecDuplicate(X, &G));
  PetscCall(CUTEstFormObjectiveGradient(tao, X, &f, G, user));
  PetscCall(VecNorm(G, NORM_2, &gnorm));
  PetscCall(VecDestroy(&G));
  PetscCall(TaoGetSolutionStatus(tao, &its, NULL, NULL, NULL, NULL, &reason));
  PetscCall(TaoGetType(tao, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOSNES, &is_snes));
  PetscCall(PetscObjectTypeCompare((PetscObject)tao, TAOPYTHON, &is_python));
  solver = name;
  if (is_snes) {
    SNES      snes;
    SNESType  snes_type;
    PetscBool newton;

    PetscCall(TaoSNESGetSNES(tao, &snes));
    PetscCall(SNESGetType(snes, &snes_type));
    PetscCall(SNESGetConvergedReason(snes, &snes_reason));
    PetscCall(SNESGetNonlinearStepFailures(snes, &rejected));
    snes_reason_name = SNESConvergedReasons[snes_reason];
    PetscCall(PetscObjectTypeCompare((PetscObject)snes, SNESPYTHON, &is_python));
    if (is_python) PetscCall(SNESPythonGetType(snes, &solver));
    else {
      PetscCall(PetscStrncmp(snes_type, "newton", 6, &newton));
      PetscCall(PetscSNPrintf(name, sizeof(name), "snes%s", newton ? snes_type + 6 : snes_type));
    }
  } else if (is_python) PetscCall(TaoPythonGetType(tao, &solver));
  else PetscCall(PetscSNPrintf(name, sizeof(name), "tao%s", type));
  if (filename[0]) {
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer, "n,solver,reason_code,reason,snes_reason_code,snes_reason,iterations,initial_objective,initial_gradient_norm,objective,gradient_norm,objective_evaluations,gradient_evaluations,hessian_"
                                             "evaluations,hessian_products,solve_seconds,rejected_steps,cutest_hessian_products\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%d,%s,%d,%s,%d,%s,%" PetscInt_FMT ",%.17g,%.17g,%.17g,%.17g,%.0f,%.0f,%.0f,%" PetscCount_FMT ",%.17g,%" PetscInt_FMT ",%.0f\n", user->n, solver, (int)reason, TaoConvergedReasons[reason], (int)snes_reason, snes_reason_name, its, (double)initial_f, (double)initial_gnorm, (double)f, (double)gnorm, (double)(calls[0] - initial_calls[0]), (double)(calls[1] - initial_calls[1]), (double)(calls[2] - initial_calls[2]), products, (double)elapsed, rejected, (double)(calls[3] - initial_calls[3])));
    PetscCall(PetscViewerDestroy(&viewer));
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\nCUTEst solve summary\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Solver:                    %s\n", solver));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Variables:                 %d\n", user->n));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Termination reason:        %s\n", TaoConvergedReasons[reason]));
    if (is_snes) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  SNES termination reason:   %s\n", snes_reason_name));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Iterations:                %" PetscInt_FMT "\n", its));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n  %-26s %-16s %s\n", "", "Initial", "Final"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %-26s %-16.8e %.8e\n", "Objective:", (double)initial_f, (double)f));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %-26s %-16.8e %.8e\n", "Gradient norm (2-norm):", (double)initial_gnorm, (double)gnorm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n  Objective evaluations:     %.0f\n", (double)(calls[0] - initial_calls[0])));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Gradient evaluations:      %.0f\n", (double)(calls[1] - initial_calls[1])));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Hessian evaluations:       %.0f\n", (double)(calls[2] - initial_calls[2])));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Hessian-vector products:   %" PetscCount_FMT "\n", products));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  CUTEst Hessian products:   %.0f\n", (double)(calls[3] - initial_calls[3])));
    if (rejected >= 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Rejected steps:            %" PetscInt_FMT "\n", rejected));
    else PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Rejected steps:            unavailable\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Solve time (seconds):      %.6g\n", (double)elapsed));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  const char    *hessians[]                  = {"sparse", "shell"};
  char           library[PETSC_MAX_PATH_LEN] = "", data[PETSC_MAX_PATH_LEN] = "OUTSDIF.d", result[PETSC_MAX_PATH_LEN] = "";
  PetscDLLibrary dll     = NULL;
  CUTEstCtx      user    = {0};
  HessianCounter counter = {0};
  Tao            tao;
  Vec            X;
  Mat            H;
  PetscInt       hessian = 0;
  PetscMPIInt    size;
  PetscReal      initial_f = 0.0, initial_gnorm = 0.0, initial_calls[4] = {0.0};
  PetscLogDouble start, end;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This CUTEst driver runs on one MPI process");
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "CUTEst problem options", "Tao");
  PetscCall(PetscOptionsString("-cutest_lib", "Decoded problem library", NULL, library, library, sizeof(library), NULL));
  PetscCall(PetscOptionsString("-cutest_data", "Decoded problem data", NULL, data, data, sizeof(data), NULL));
  PetscCall(PetscOptionsEList("-cutest_hessian", "Hessian representation", NULL, hessians, 2, hessians[hessian], &hessian, NULL));
  PetscCall(PetscOptionsString("-cutest_result", "CSV solve statistics", NULL, result, result, sizeof(result), NULL));
  PetscOptionsEnd();
  PetscCall(CUTEstLoadProblem(library, data, &user, &dll, &X));

  PetscCall(CUTEstCreateHessian(X, (PetscBool)hessian, &user, &H));
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
  PetscCall(TaoSetType(tao, TAOLMVM));
  PetscCall(TaoSetSolution(tao, X));
  PetscCall(TaoSetObjective(tao, CUTEstFormObjective, &user));
  PetscCall(TaoSetGradient(tao, NULL, CUTEstFormGradient, &user));
  PetscCall(TaoSetObjectiveAndGradient(tao, NULL, CUTEstFormObjectiveGradient, &user));
  PetscCall(PetscObjectContainerCompose((PetscObject)H, "CUTEstHessianProducts", &counter, NULL));
  PetscCall(MatGetOperation(H, MATOP_MULT, (PetscErrorCodeFn **)&counter.mult));
  PetscCall(MatSetOperation(H, MATOP_MULT, (PetscErrorCodeFn *)CountHessianMult));
  PetscCall(MatGetOperation(H, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn **)&counter.multtranspose));
  PetscCall(MatSetOperation(H, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)CountHessianMultTranspose));
  PetscCall(TaoSetHessian(tao, H, H, CUTEstFormHessian, &user));
  PetscCall(TaoSetFromOptions(tao));
  {
    Vec       G;
    PetscReal time[4];

    PetscCall(VecDuplicate(X, &G));
    PetscCall(CUTEstFormObjectiveGradient(tao, X, &initial_f, G, &user));
    PetscCall(VecNorm(G, NORM_2, &initial_gnorm));
    PetscCall(VecDestroy(&G));
    PetscCallCUTEst(CUTEST_ureport, initial_calls, time);
  }
  counter.products = 0;
  PetscCall(PetscTime(&start));
  PetscCall(TaoSolve(tao));
  PetscCall(PetscTime(&end));
  PetscCall(WriteResult(tao, X, &user, result, initial_f, initial_gnorm, initial_calls, end - start, counter.products));
  PetscCall(VecViewFromOptions(X, NULL, "-cutest_solution_view"));
  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&user.hessian_x));
  PetscCall(PetscFree3(user.rows, user.cols, user.values));
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
    command: @PYTHON@ ${wPETSC_DIR}/share/petsc/cutest/run.py --petsc-dir ${wPETSC_DIR} --petsc-arch=${petsc_arch} --driver ../cutest --sif ${wPETSC_DIR}/share/petsc/datafiles/cutest/ROSENBR.SIF -- ${args} @SUBARGS@
    args: -tao_gatol 1.e-6 -tao_grtol 0 -tao_gttol 0 -tao_converged_reason -cutest_result result.csv
    temporaries: result.csv
    filter: sed -E "s/ iterations [0-9]+//"
    output_file: output/cutest_1.out
    test:
      suffix: 1
      args: -tao_type {{lmvm nls ntr}} -cutest_hessian {{sparse shell}}
    test:
      suffix: snes
      args: -tao_type snes -snes_type newtontr -snes_atol 1.e-6 -ksp_type cg -pc_type none -cutest_hessian {{sparse shell}}
      output_file: output/cutest_snes.out

  testset:
    requires: sifdecode
    command: @PYTHON@ ${wPETSC_DIR}/share/petsc/cutest/run.py --petsc-dir ${wPETSC_DIR} --petsc-arch=${petsc_arch} --driver ../cutest --sif ${wPETSC_DIR}/share/petsc/datafiles/cutest/ROSENBR.SIF -- ${args} @SUBARGS@
    args: -tao_gatol 1.e-6 -tao_grtol 0 -tao_gttol 0
    filter: sed -E "s,Solve time [(]seconds[)]:.*,Solve time (seconds):,"
    test:
      suffix: summary_ntr
      args: -tao_type ntr
    test:
      suffix: summary_snes_sparse
      args: -tao_type snes -snes_type newtonls -snes_atol 1.e-6 -ksp_type bicg -pc_type none
    test:
      suffix: summary_snes_shell
      args: -tao_type snes -snes_type newtonls -snes_atol 1.e-6 -ksp_type bicg -pc_type none -cutest_hessian shell

TEST*/
