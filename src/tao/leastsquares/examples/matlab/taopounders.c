#include <petsctao.h>

#include "mex.h"

typedef struct {
  mxArray *mat_handle;    /* Matlab function handle to call */
  mxArray *mat_x;         /* Matlab array of the inputs */

  PetscInt n;             /* Number of inputs */
  PetscInt m;             /* Number of outputs */
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
  rhs[1] = user->mat_x;

  xp = mxGetPr(rhs[1]);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (i = 0; i < user->n; ++i) {
    xp[i] = (double) x[i];
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  mexCallMATLAB(1,lhs,2,rhs,"feval");

  op = mxGetPr(lhs[0]);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  for (i = 0; i < user->m; ++i) {
    f[i] = (PetscReal) op[i];
  }
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  mxDestroyArray(lhs[0]);
  PetscFunctionReturn(0);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  PetscErrorCode  ierr;
  Tao             tao;
  Vec             X, F;
  PetscScalar    *x;
  PetscInt        n, m;
  PetscInt        i;
  AppCtx          user;
 
  double         *xp, *op;

  /* 
     The number of input arguments (nrhs) should be 4
     0 - dimension of the inputs problem (n)
     1 - starting point of length n
     2 - dimension of the residuals (m)
     3 - character string containing name of the function to evaluate the residuals

     The number of output arguments (nlhs) should be 1
     0 - solution (n)
  */

  if (nrhs < 4) {
    mexErrMsgTxt ("Not enough input arguments.");
  }

  if (nlhs < 1) {
    mexErrMsgTxt ("Not enough output arguments.");
  }

  user.n = (PetscInt) mxGetScalar(prhs[0]);
  user.m = (PetscInt) mxGetScalar(prhs[2]);

  if (!mxIsClass(prhs[3], "function_handle")) {
    mexErrMsgTxt("Argument 4 must be function handle.\n");
    return;
  }
  user.mat_handle = (mxArray *)prhs[3];

  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);
  user.mat_x = plhs[0];
  
  /* Initialize PETSc and create TAO pounders instance */
  ierr = PetscInitialize(NULL,NULL,(char *)0,"taopounders");/*CHKERRQ(ierr);*/
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);/*CHKERRQ(ierr);*/
  ierr = TaoSetType(tao,TAOPOUNDERS);/*CHKERRQ(ierr);*/

  /* Create starting point and initialize */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&X);/*CHKERRQ(ierr);*/
  xp = mxGetPr(prhs[1]);
  ierr = VecGetArray(X,&x);/*CHKERRQ(ierr);*/
  for (i = 0; i < user.n; ++i) {
    x[i] = (PetscReal) xp[i];
  }
  ierr = VecRestoreArray(X,&x);/*CHKERRQ(ierr);*/
  ierr = TaoSetInitialVector(tao,X);/*CHKERRQ(ierr);*/

  /* Create residuals vector and set objective function */  
  ierr = VecCreateSeq(MPI_COMM_SELF,m,&F);/*CHKERRQ(ierr);*/
  ierr = TaoSetSeparableObjectiveRoutine(tao,F,EvaluateFunction,(void*)&user);/*CHKERRQ(ierr);*/

  /* Solve the problem */

  /* ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);*/
  ierr = TaoSetFromOptions(tao);/*CHKERRQ(ierr);*/
  ierr = TaoSolve(tao);/*CHKERRQ(ierr);*/

  /* Save the solution */
  op = mxGetPr(plhs[0]);
  ierr = VecGetArray(X,&x);/*CHKERRQ(ierr);*/
  for (i = 0; i < user.n; ++i) {
    op[i] = (double) x[i];
  }
  ierr = VecRestoreArray(X,&x);/*CHKERRQ(ierr);*/

  /* Finish the problem */
  ierr = TaoDestroy(&tao);/*CHKERRQ(ierr);*/
  ierr = VecDestroy(&X);/*CHKERRQ(ierr);*/
  ierr = VecDestroy(&F);/*CHKERRQ(ierr);*/
  ierr = PetscFinalize();
  return;
}

