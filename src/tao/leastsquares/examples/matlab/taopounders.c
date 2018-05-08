#include <petsctao.h>
#include <stdio.h>

#include "mex.h"

typedef struct {
  mxArray *mat_handle;    /* Matlab function handle to call */
  double  *x;             /* Matlab array of the inputs */

  double delta;           /* Initial trust region radius */

  int n;                  /* Number of inputs */
  int m;                  /* Number of outputs */
  int nfmax;              /* Maximum function evaluations */
  int npmax;              /* Maximum interpolation points */
} AppCtx;

static PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscScalar    *x, *f;
  PetscErrorCode  ierr;
  PetscInt        i;

  mxArray        *lhs[1];
  mxArray        *rhs[2];
  double         *xp, *op;
  static int      cnt = 0;

  PetscFunctionBegin;

  mexPrintf("Function start %d.\n", cnt++);
  rhs[0] = user->mat_handle;
  rhs[1] = mxCreateDoubleMatrix(user->n,1,mxREAL);

  xp = mxGetPr(rhs[1]);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    xp[i] = x[i];
    /* printf("x[%d] = %5.4e\n", i, xp[i]); */
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  mexCallMATLAB(1,lhs,2,rhs,"feval");

  op = mxGetPr(lhs[0]);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (i = 0; i < user->m; ++i) {
    f[i] = op[i];
    /* printf("f[%d] = %5.4e\n", i, op[i]); */
  }
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  mxDestroyArray(rhs[1]);
  mxDestroyArray(lhs[0]);
  mexPrintf("Function end.\n");
  PetscFunctionReturn(0);
}

static PetscErrorCode TaoPounders(AppCtx *user)
{
  PetscErrorCode  ierr;
  Tao             tao;
  Vec             X, F;
  PetscScalar    *x;
  PetscInt        i;

  int argc = 8;
  char **argv;

  argv = (char **)mxMalloc(argc*sizeof(char *));
  for (i = 0; i < argc; ++i) {
    argv[i] = (char *)mxMalloc(256*sizeof(char));
  }
  strcpy(argv[0], "taopounders");
  strcpy(argv[1], "-tao_max_funcs");
  sprintf(argv[2], "%5d", user->nfmax);
  strcpy(argv[3], "-tao_pounders_npmax");
  sprintf(argv[4], "%5d", user->npmax);
  strcpy(argv[5], "-tao_pounders_delta");
  sprintf(argv[6], "%5.4e", user->delta);
  strcpy(argv[7],"-tao_monitor");
  /*
  strcpy(argv[8],"-tao_view");
  strcpy(argv[9],"-pounders_subsolver_tao_monitor");
  strcpy(argv[10],"-pounders_subsolver_ksp_monitor");
  strcpy(argv[11],"-pounders_subsolver_tao_view");
  */
  mexPrintf("n: %d m: %d\n", user->n, user->m);

  ierr = PetscInitialize(&argc,&argv,NULL,"taopounders help");CHKERRQ(ierr);

  for (i = 0; i < argc; ++i) {
    mxFree(argv[i]);
  }
  mxFree(argv);
  return 0;

  PetscFunctionBegin;
  /* Initialize PETSc and create TAO pounders instance */
  /* ierr = PetscPopSignalHandler();CHKERRQ(ierr); */
  /* ierr = PetscFinalize();CHKERRQ(ierr);*/
  PetscFunctionReturn(0);

  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* Create starting point and initialize */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->n,&X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    x[i] = user->x[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,X);CHKERRQ(ierr);

  /* Create residuals vector and set objective function */  
  ierr = VecCreateSeq(PETSC_COMM_SELF,user->m,&F);CHKERRQ(ierr);
  ierr = TaoSetSeparableObjectiveRoutine(tao,F,EvaluateFunction,(void*)user);CHKERRQ(ierr);

  /* Solve the problem */

  /* ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  /*ierr = TaoSolve(tao);CHKERRQ(ierr);*/

  /* Save the solution */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    user->x[i] = (double) x[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  /* Finish the problem */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  int i;
  int argc = 1;
  char **argv;

  argv = (char **)mxMalloc((argc+1)*sizeof(char *));
  for (i = 0; i <= argc; ++i) {
    argv[i] = (char *)mxMalloc(256*sizeof(char));
  }
  strcpy(argv[0], "taopounders");
  strcpy(argv[1], "");

  /* PetscInitialize(&argc,&argv,NULL,"taopounders help"); */
  PetscInitializeNoArguments();
  /* PetscPopSignalHandler(); */
  PetscFinalize();

  for (i = 0; i < argc; ++i) {
    mxFree(argv[i]);
  }
  mxFree(argv);
  return;
}

#if 0
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  /* 
     The number of input arguments (nrhs) should be 4
     0 - dimension of the inputs problem (n)
     1 - starting point of length n
     2 - dimension of the residuals (m)
     3 - matlab function handle to evaluate the residuals
     4 - maximum number of function evaluations
     5 - maximum number of interpolation points
     6 - initial trust region radius

     The number of output arguments (nlhs) should be 1
     0 - solution (n)
  */

  AppCtx  user;
  double *xp, *op;
  int     i;

  if (nrhs < 7) {
    mexErrMsgTxt ("Not enough input arguments.");
  }

  if (nlhs < 1) {
    mexErrMsgTxt ("Not enough output arguments.");
  }

  user.n = (int) mxGetScalar(prhs[0]);
  user.m = (int) mxGetScalar(prhs[2]);
  if (!mxIsClass(prhs[3], "function_handle")) {
    mexErrMsgTxt("Argument 4 must be function handle.\n");
    return;
  }
  user.mat_handle = (mxArray *)prhs[3];
  user.nfmax = (int) mxGetScalar(prhs[4]);
  user.npmax = (int) mxGetScalar(prhs[5]);
  user.delta = (double) mxGetScalar(prhs[6]);

  plhs[0] = mxCreateDoubleMatrix(user.n,1,mxREAL);
  user.x = mxGetPr(plhs[0]);

  /* Store the intial starting point in the output */
  xp = mxGetPr(prhs[1]);
  op = mxGetPr(plhs[0]);
  for (i = 0; i < user.n; ++i) {
    op[i] = xp[i];
  }

  TaoPounders(&user);
  return;
}
#endif

