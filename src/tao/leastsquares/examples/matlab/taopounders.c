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

  PetscFunctionBegin;

  rhs[0] = user->mat_handle;
  rhs[1] = mxCreateDoubleMatrix(user->n,1,mxREAL);

  xp = mxGetPr(rhs[1]);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    xp[i] = x[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  mexCallMATLAB(1,lhs,2,rhs,"feval");

  op = mxGetPr(lhs[0]);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (i = 0; i < user->m; ++i) {
    f[i] = op[i];
  }
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  mxDestroyArray(rhs[1]);
  mxDestroyArray(lhs[0]);
  PetscFunctionReturn(0);
}

static void TaoPoundersExit()
{
  PetscFinalize();
  return;
}

static PetscErrorCode TaoPounders(AppCtx *user)
{
  PetscErrorCode  ierr;
  Tao             tao;
  Vec             X, F;
  PetscScalar    *x;
  PetscInt        i;
  char            buf[256];

  PetscFunctionBegin;

  /* Set the values for the algorithm options we want to use */
  sprintf(buf,"%d",user->nfmax);
  ierr = PetscOptionsSetValue(NULL,"-tao_max_funcs",buf);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-tao_monitor",NULL);CHKERRQ(ierr);
  sprintf(buf,"%d",user->npmax);
  ierr = PetscOptionsSetValue(NULL,"-tao_pounders_npmax",buf);CHKERRQ(ierr);
  sprintf(buf,"%5.4e",user->delta);
  ierr = PetscOptionsSetValue(NULL,"-tao_pounders_delta",buf);CHKERRQ(ierr);

  ierr = PetscOptionsSetValue(NULL,"-pounders_subsolver_tao_monitor",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-tao_view", NULL);CHKERRQ(ierr);
  /*ierr = PetscOptionsSetValue(NULL,"-info", NULL);CHKERRQ(ierr);*/

  ierr = PetscInitializeNoArguments();CHKERRQ(ierr);
  mexAtExit(TaoPoundersExit);

  /* Create the TAO objects and set the type */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);

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
  ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  /* Save the solution */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    user->x[i] = (double) x[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  /* Finish the problem */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
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

int main(int argc, char **argv)
{
  return 0;
}

