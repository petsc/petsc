static char help[] = "TAO/Pounders MATLAB Testing on the More'-Wild Benchmark Problems\n\
The interface calls:\n\
    TestingInitialize.m to initialize the problem set\n\
    ProblemInitialize.m to initialize each instance\n\
    ProblemFinalize.m to store the performance data for the instance solved\n\
    TestingFinalize.m to store the entire set of performance data\n\
\n\
TestingPlot.m is called outside of TAO/Pounders to produce a performance profile\n\
of the results compared to the MATLAB fminsearch algorithm.\n";

#include <petsctao.h>
#include <petscmatlab.h>

typedef struct {
  PetscMatlabEngine mengine;

  double delta; /* Initial trust region radius */

  int n;     /* Number of inputs */
  int m;     /* Number of outputs */
  int nfmax; /* Maximum function evaluations */
  int npmax; /* Maximum interpolation points */
} AppCtx;

static PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  PetscCall(PetscMatlabEnginePut(user->mengine, (PetscObject)X));
  PetscCall(PetscMatlabEngineEvaluate(user->mengine, "F = func(X);"));
  PetscCall(PetscObjectSetName((PetscObject)F, "F"));
  PetscCall(PetscMatlabEngineGet(user->mengine, (PetscObject)F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat JPre, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  PetscCall(PetscObjectSetName((PetscObject)X, "X"));
  PetscCall(PetscMatlabEnginePut(user->mengine, (PetscObject)X));
  PetscCall(PetscMatlabEngineEvaluate(user->mengine, "J = jac(X);"));
  PetscCall(PetscObjectSetName((PetscObject)J, "J"));
  PetscCall(PetscMatlabEngineGet(user->mengine, (PetscObject)J));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TaoPounders(AppCtx *user)
{
  Tao  tao;
  Vec  X, F;
  Mat  J;
  char buf[1024];

  PetscFunctionBegin;

  /* Set the values for the algorithm options we want to use */
  PetscCall(PetscSNPrintf(buf, PETSC_STATIC_ARRAY_LENGTH(buf), "%d", user->npmax));
  PetscCall(PetscOptionsSetValue(NULL, "-tao_pounders_npmax", buf));
  PetscCall(PetscSNPrintf(buf, PETSC_STATIC_ARRAY_LENGTH(buf), "%5.4e", user->delta));
  PetscCall(PetscOptionsSetValue(NULL, "-tao_pounders_delta", buf));

  /* Create the TAO objects and set the type */
  PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));

  /* Create starting point and initialize */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->n, &X));
  PetscCall(PetscObjectSetName((PetscObject)X, "X0"));
  PetscCall(PetscMatlabEngineGet(user->mengine, (PetscObject)X));
  PetscCall(TaoSetSolution(tao, X));

  /* Create residuals vector and set residual function */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->m, &F));
  PetscCall(PetscObjectSetName((PetscObject)F, "F"));
  PetscCall(TaoSetResidualRoutine(tao, F, EvaluateResidual, (void *)user));

  /* Create Jacobian matrix and set residual Jacobian routine */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_WORLD, user->m, user->n, user->n, NULL, &J));
  PetscCall(PetscObjectSetName((PetscObject)J, "J"));
  PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, EvaluateJacobian, (void *)user));

  /* Solve the problem */
  PetscCall(TaoSetType(tao, TAOPOUNDERS));
  PetscCall(TaoSetMaximumFunctionEvaluations(tao, user->nfmax));
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoSolve(tao));

  /* Finish the problem */
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&F));
  PetscCall(TaoDestroy(&tao));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  AppCtx      user;
  PetscScalar tmp;
  PetscInt    prob_id = 0;
  PetscBool   flg, testall = PETSC_FALSE;
  int         i, i0, imax;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_all", &testall, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-prob_id", &prob_id, &flg));
  if (!testall) {
    if (!flg) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Problem number must be specified with -prob_id");
    } else if ((prob_id < 1) || (prob_id > 53)) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Problem number must be between 1 and 53!");
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Running problem %d\n", prob_id));
    }
  } else {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Running all problems\n"));
  }

  PetscCall(PetscMatlabEngineCreate(PETSC_COMM_SELF, NULL, &user.mengine));
  PetscCall(PetscMatlabEngineEvaluate(user.mengine, "TestingInitialize"));

  if (testall) {
    i0   = 1;
    imax = 53;
  } else {
    i0   = (int)prob_id;
    imax = (int)prob_id;
  }

  for (i = i0; i <= imax; ++i) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%d\n", i));
    PetscCall(PetscMatlabEngineEvaluate(user.mengine, "np = %d; ProblemInitialize", i));
    PetscCall(PetscMatlabEngineGetArray(user.mengine, 1, 1, &tmp, "n"));
    user.n = (int)tmp;
    PetscCall(PetscMatlabEngineGetArray(user.mengine, 1, 1, &tmp, "m"));
    user.m = (int)tmp;
    PetscCall(PetscMatlabEngineGetArray(user.mengine, 1, 1, &tmp, "nfmax"));
    user.nfmax = (int)tmp;
    PetscCall(PetscMatlabEngineGetArray(user.mengine, 1, 1, &tmp, "npmax"));
    user.npmax = (int)tmp;
    PetscCall(PetscMatlabEngineGetArray(user.mengine, 1, 1, &tmp, "delta"));
    user.delta = (double)tmp;

    /* Ignore return code for now -- do not stop testing on inf or nan errors */
    PetscCall(TaoPounders(&user));

    PetscCall(PetscMatlabEngineEvaluate(user.mengine, "ProblemFinalize"));
  }

  PetscCall(PetscMatlabEngineEvaluate(user.mengine, "TestingFinalize"));
  PetscCall(PetscMatlabEngineDestroy(&user.mengine));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: matlab

   test:
      localrunfiles: more_wild_probs TestingInitialize.m TestingFinalize.m ProblemInitialize.m ProblemFinalize.m
      args: -tao_smonitor -prob_id 5

TEST*/
