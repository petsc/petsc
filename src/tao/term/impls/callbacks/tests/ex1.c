const char help[] = "TAOTERMCALLBACKS coverage tests";

#include <petsctao.h>

typedef struct {
  PetscInt obj_count;
  PetscInt grad_count;
  PetscInt obj_and_grad_count;
  PetscInt hess_count;
} AppCtx;

static PetscErrorCode objective(Tao tao, Vec x, PetscReal *value, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  *value = 0.0;
  app->obj_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode gradient(Tao tao, Vec x, Vec g, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(g));
  app->grad_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode objective_and_gradient(Tao tao, Vec x, PetscReal *value, Vec g, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  *value = 0.0;
  PetscCall(VecZeroEntries(g));
  app->obj_and_grad_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode hessian(Tao tao, Vec x, Mat H, Mat Hpre, void *ctx)
{
  AppCtx *app = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  if (H) {
    PetscCall(MatZeroEntries(H));
    PetscCall(MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY));
  }
  if (Hpre && Hpre != H) {
    PetscCall(MatZeroEntries(Hpre));
    PetscCall(MatAssemblyBegin(Hpre, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre, MAT_FINAL_ASSEMBLY));
  }
  app->hess_count++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode testCallbacks(PetscBool separate)
{
  MPI_Comm    comm = PETSC_COMM_WORLD;
  Tao         tao;
  TaoTerm     term;
  TaoTermType type;
  PetscBool   same;
  PetscErrorCode (*_hessian)(Tao, Vec, Mat, Mat, void *);
  AppCtx    app;
  Vec       sol, grad;
  Mat       H, Hpre;
  PetscInt  N = 10;
  PetscReal value;

  PetscFunctionBeginUser;
  app.obj_count          = 0;
  app.grad_count         = 0;
  app.obj_and_grad_count = 0;
  app.hess_count         = 0;
  PetscCall(VecCreateMPI(comm, PETSC_DECIDE, N, &sol));
  PetscCall(VecZeroEntries(sol));
  PetscCall(VecDuplicate(sol, &grad));
  PetscCall(MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, N, N, 1, NULL, 0, NULL, &H));
  PetscCall(MatDuplicate(H, MAT_DO_NOT_COPY_VALUES, &Hpre));
  PetscCall(TaoCreate(comm, &tao));
  PetscCall(TaoSetSolution(tao, sol));
  PetscCall(TaoGetTerm(tao, NULL, &term, NULL, NULL));
  PetscCall(TaoTermGetType(term, &type));
  PetscCall(PetscStrcmp(type, TAOTERMCALLBACKS, &same));
  PetscCheck(same, comm, PETSC_ERR_PLIB, "wrong TaoTermType");

  if (separate) {
    PetscCall(TaoSetObjective(tao, objective, (void *)&app));
    PetscCall(TaoSetGradient(tao, grad, gradient, (void *)&app));
  } else {
    PetscCall(TaoSetObjectiveAndGradient(tao, grad, objective_and_gradient, (void *)&app));
  }
  PetscCall(TaoSetHessian(tao, H, Hpre, hessian, (void *)&app));

  {
    PetscBool is_defined;

    PetscCall(TaoTermIsHessianDefined(term, &is_defined));
    PetscCheck(is_defined == PETSC_TRUE, comm, PETSC_ERR_PLIB, "Hessian should be defined after setting it");
  }

  if (separate) {
    PetscErrorCode (*_objective)(Tao, Vec, PetscReal *, void *);
    PetscErrorCode (*_gradient)(Tao, Vec, Vec, void *);

    PetscCall(TaoGetObjective(tao, &_objective, NULL));
    PetscCall(TaoGetGradient(tao, NULL, &_gradient, NULL));
    PetscCheck(_objective == objective, comm, PETSC_ERR_PLIB, "wrong objective callback");
    PetscCheck(_gradient == gradient, comm, PETSC_ERR_PLIB, "wrong gradient callback");
  } else {
    PetscErrorCode (*_objective_and_gradient)(Tao, Vec, PetscReal *, Vec, void *);

    PetscCall(TaoGetObjectiveAndGradient(tao, NULL, &_objective_and_gradient, NULL));
    PetscCheck(_objective_and_gradient == objective_and_gradient, comm, PETSC_ERR_PLIB, "wrong objective and gradient callback");
  }
  PetscCall(TaoGetHessian(tao, NULL, NULL, &_hessian, NULL));
  PetscCheck(_hessian == hessian, comm, PETSC_ERR_PLIB, "wrong hessian callback");

  PetscCall(TaoComputeObjective(tao, sol, &value));
  (void)value;
  PetscCall(TaoComputeGradient(tao, sol, grad));
  PetscCall(TaoComputeObjectiveAndGradient(tao, sol, &value, grad));
  (void)value;
  PetscCall(TaoComputeHessian(tao, sol, H, Hpre));

  if (separate) {
    PetscCheck(app.obj_count == 2, comm, PETSC_ERR_PLIB, "Incorrect number of objective evaluations");
    PetscCheck(app.grad_count == 2, comm, PETSC_ERR_PLIB, "Incorrect number of gradient evaluations");
    PetscCheck(app.obj_and_grad_count == 0, comm, PETSC_ERR_PLIB, "Incorrect number of objective+gradient evaluations");
  } else {
    PetscCheck(app.obj_count == 0, comm, PETSC_ERR_PLIB, "Incorrect number of objective evaluations");
    PetscCheck(app.grad_count == 0, comm, PETSC_ERR_PLIB, "Incorrect number of gradient evaluations");
    PetscCheck(app.obj_and_grad_count == 3, comm, PETSC_ERR_PLIB, "Incorrect number of objective+gradient evaluations");
  }
  PetscCheck(app.hess_count == 1, comm, PETSC_ERR_PLIB, "Incorrect number of hessian evaluations");

  PetscCall(TaoDestroy(&tao));
  PetscCall(MatDestroy(&Hpre));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&grad));
  PetscCall(VecDestroy(&sol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(testCallbacks(PETSC_FALSE));
  PetscCall(testCallbacks(PETSC_TRUE));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out

TEST*/
