static char help[] = "TAO/Pounders Matlab Testing on the More'-Wild Benchmark Problems\n\
The interface calls:\n\
    TestingInitialize.m to initialize the problem set\n\
    ProblemInitialize.m to initialize each instance\n\
    ProblemFinalize.m to store the performance data for the instance solved\n\
    TestingFinalize.m to store the entire set of performance data\n\
\n\
TestingPlot.m is called outside of TAO/Pounders to produce a performance profile\n\
of the results compared to the Matlab fminsearch algorithm.\n";

#include <petsctao.h>
#include <petscmatlab.h>

typedef struct {
  PetscMatlabEngine mengine;

  double delta;           /* Initial trust region radius */

  int n;                  /* Number of inputs */
  int m;                  /* Number of outputs */
  int nfmax;              /* Maximum function evaluations */
  int npmax;              /* Maximum interpolation points */
} AppCtx;

static PetscErrorCode EvaluateResidual(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(PetscMatlabEnginePut(user->mengine,(PetscObject)X));
  CHKERRQ(PetscMatlabEngineEvaluate(user->mengine,"F = func(X);"));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  CHKERRQ(PetscMatlabEngineGet(user->mengine,(PetscObject)F));
  PetscFunctionReturn(0);
}

static PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat JPre, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X"));
  CHKERRQ(PetscMatlabEnginePut(user->mengine,(PetscObject)X));
  CHKERRQ(PetscMatlabEngineEvaluate(user->mengine,"J = jac(X);"));
  CHKERRQ(PetscObjectSetName((PetscObject)J,"J"));
  CHKERRQ(PetscMatlabEngineGet(user->mengine,(PetscObject)J));
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoPounders(AppCtx *user)
{
  Tao            tao;
  Vec            X, F;
  Mat            J;
  char           buf[1024];

  PetscFunctionBegin;

  /* Set the values for the algorithm options we want to use */
  sprintf(buf,"%d",user->npmax);
  CHKERRQ(PetscOptionsSetValue(NULL,"-tao_pounders_npmax",buf));
  sprintf(buf,"%5.4e",user->delta);
  CHKERRQ(PetscOptionsSetValue(NULL,"-tao_pounders_delta",buf));

  /* Create the TAO objects and set the type */
  CHKERRQ(TaoCreate(PETSC_COMM_SELF,&tao));

  /* Create starting point and initialize */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,user->n,&X));
  CHKERRQ(PetscObjectSetName((PetscObject)X,"X0"));
  CHKERRQ(PetscMatlabEngineGet(user->mengine,(PetscObject)X));
  CHKERRQ(TaoSetSolution(tao,X));

  /* Create residuals vector and set residual function */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,user->m,&F));
  CHKERRQ(PetscObjectSetName((PetscObject)F,"F"));
  CHKERRQ(TaoSetResidualRoutine(tao,F,EvaluateResidual,(void*)user));

  /* Create Jacobian matrix and set residual Jacobian routine */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,user->m,user->n,user->n,NULL,&J));
  CHKERRQ(PetscObjectSetName((PetscObject)J,"J"));
  CHKERRQ(TaoSetJacobianResidualRoutine(tao,J,J,EvaluateJacobian,(void*)user));

  /* Solve the problem */
  CHKERRQ(TaoSetType(tao,TAOPOUNDERS));
  CHKERRQ(TaoSetMaximumFunctionEvaluations(tao,user->nfmax));
  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoSolve(tao));

  /* Finish the problem */
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(TaoDestroy(&tao));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  AppCtx         user;
  PetscErrorCode ierr;
  PetscScalar    tmp;
  PetscInt       prob_id = 0;
  PetscBool      flg, testall = PETSC_FALSE;
  int            i, i0, imax;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_all",&testall,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-prob_id",&prob_id,&flg));
  if (!testall) {
    if (!flg) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_NULL, "Problem number must be specified with -prob_id");
    } else if ((prob_id < 1) || (prob_id > 53)) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Problem number must be between 1 and 53!");
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Running problem %d\n",prob_id));
    }
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Running all problems\n"));
  }

  CHKERRQ(PetscMatlabEngineCreate(PETSC_COMM_SELF,NULL,&user.mengine));
  CHKERRQ(PetscMatlabEngineEvaluate(user.mengine,"TestingInitialize"));

  if (testall) {
    i0 = 1;
    imax = 53;
  } else {
    i0 = (int)prob_id;
    imax = (int)prob_id;
  }

  for (i = i0; i <= imax; ++i) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%d\n",i));
      CHKERRQ(PetscMatlabEngineEvaluate(user.mengine,"np = %d; ProblemInitialize",i));
      CHKERRQ(PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"n"));
      user.n = (int)tmp;
      CHKERRQ(PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"m"));
      user.m = (int)tmp;
      CHKERRQ(PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"nfmax"));
      user.nfmax = (int)tmp;
      CHKERRQ(PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"npmax"));
      user.npmax = (int)tmp;
      CHKERRQ(PetscMatlabEngineGetArray(user.mengine,1,1,&tmp,"delta"));
      user.delta = (double)tmp;

      /* Ignore return code for now -- do not stop testing on inf or nan errors */
      CHKERRQ(TaoPounders(&user));

      CHKERRQ(PetscMatlabEngineEvaluate(user.mengine,"ProblemFinalize"));
    }

  CHKERRQ(PetscMatlabEngineEvaluate(user.mengine,"TestingFinalize"));
  CHKERRQ(PetscMatlabEngineDestroy(&user.mengine));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: matlab_engine

   test:
      localrunfiles: more_wild_probs TestingInitialize.m TestingFinalize.m ProblemInitialize.m ProblemFinalize.m
      args: -tao_smonitor -prob_id 5

TEST*/
