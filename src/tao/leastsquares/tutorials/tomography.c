/* XH:
    Todo: add cs1f.F90 and adjust makefile.
    Todo: maybe provide code template to generate 1D/2D/3D gradient, DCT transform matrix for D etc.
*/
/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/

#include <petsctao.h>

/*
Description:   BRGN tomography reconstruction example .
               0.5*||Ax-b||^2 + lambda*g(x)
Reference:     None
*/

static char help[] = "Finds the least-squares solution to the under constraint linear model Ax = b, with regularizer. \n\
            A is a M*N real matrix (M<N), x is sparse. A good regularizer is an L1 regularizer. \n\
            We find the sparse solution by solving 0.5*||Ax-b||^2 + lambda*||D*x||_1, where lambda (by default 1e-4) is a user specified weight.\n\
            D is the K*N transform matrix so that D*x is sparse. By default D is identity matrix, so that D*x = x.\n";
/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetSolution();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/

/* User-defined application context */
typedef struct {
  /* Working space. linear least square:  res(x) = A*x - b */
  PetscInt  M,N,K;            /* Problem dimension: A is M*N Matrix, D is K*N Matrix */
  Mat       A,D;              /* Coefficients, Dictionary Transform of size M*N and K*N respectively. For linear least square, Jacobian Matrix J = A. For nonlinear least square, it is different from A */
  Vec       b,xGT,xlb,xub;    /* observation b, ground truth xGT, the lower bound and upper bound of x*/
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeUserData(AppCtx *);
PetscErrorCode FormStartingPoint(Vec,AppCtx *);
PetscErrorCode EvaluateResidual(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao,Vec,PetscReal *,Vec,void*);
PetscErrorCode EvaluateRegularizerHessian(Tao,Vec,Mat,void*);
PetscErrorCode EvaluateRegularizerHessianProd(Mat,Vec,Vec);

/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  Vec            x,res;              /* solution, function res(x) = A*x-b */
  Mat            Hreg;               /* regularizer Hessian matrix for user specified regularizer*/
  Tao            tao;                /* Tao solver context */
  PetscReal      hist[100],resid[100],v1,v2;
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */
  PetscViewer    fd;   /* used to save result to file */
  char           resultFile[] = "tomographyResult_x";  /* Debug: change from "tomographyResult_x" to "cs1Result_x" */

  PetscCall(PetscInitialize(&argc,&argv,(char *)0,help));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_SELF,&tao));
  PetscCall(TaoSetType(tao,TAOBRGN));

  /* User set application context: A, D matrice, and b vector. */
  PetscCall(InitializeUserData(&user));

  /* Allocate solution vector x,  and function vectors Ax-b, */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,user.N,&x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,user.M,&res));

  /* Set initial guess */
  PetscCall(FormStartingPoint(x,&user));

  /* Bind x to tao->solution. */
  PetscCall(TaoSetSolution(tao,x));
  /* Sets the upper and lower bounds of x */
  PetscCall(TaoSetVariableBounds(tao,user.xlb,user.xub));

  /* Bind user.D to tao->data->D */
  PetscCall(TaoBRGNSetDictionaryMatrix(tao,user.D));

  /* Set the residual function and Jacobian routines for least squares. */
  PetscCall(TaoSetResidualRoutine(tao,res,EvaluateResidual,(void*)&user));
  /* Jacobian matrix fixed as user.A for Linear least square problem. */
  PetscCall(TaoSetJacobianResidualRoutine(tao,user.A,user.A,EvaluateJacobian,(void*)&user));

  /* User set the regularizer objective, gradient, and hessian. Set it the same as using l2prox choice, for testing purpose.  */
  PetscCall(TaoBRGNSetRegularizerObjectiveAndGradientRoutine(tao,EvaluateRegularizerObjectiveAndGradient,(void*)&user));
  /* User defined regularizer Hessian setup, here is identiy shell matrix */
  PetscCall(MatCreate(PETSC_COMM_SELF,&Hreg));
  PetscCall(MatSetSizes(Hreg,PETSC_DECIDE,PETSC_DECIDE,user.N,user.N));
  PetscCall(MatSetType(Hreg,MATSHELL));
  PetscCall(MatSetUp(Hreg));
  PetscCall(MatShellSetOperation(Hreg,MATOP_MULT,(void (*)(void))EvaluateRegularizerHessianProd));
  PetscCall(TaoBRGNSetRegularizerHessianRoutine(tao,Hreg,EvaluateRegularizerHessian,(void*)&user));

  /* Check for any TAO command line arguments */
  PetscCall(TaoSetFromOptions(tao));

  PetscCall(TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE));

  /* Perform the Solve */
  PetscCall(TaoSolve(tao));

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_SELF,resultFile,FILE_MODE_WRITE,&fd));
  PetscCall(VecView(x,fd));
  PetscCall(PetscViewerDestroy(&fd));

  /* compute the error */
  PetscCall(VecAXPY(x,-1,user.xGT));
  PetscCall(VecNorm(x,NORM_2,&v1));
  PetscCall(VecNorm(user.xGT,NORM_2,&v2));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2)));

  /* Free TAO data structures */
  PetscCall(TaoDestroy(&tao));

   /* Free PETSc data structures */
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&res));
  PetscCall(MatDestroy(&Hreg));
  /* Free user data structures */
  PetscCall(MatDestroy(&user.A));
  PetscCall(MatDestroy(&user.D));
  PetscCall(VecDestroy(&user.b));
  PetscCall(VecDestroy(&user.xGT));
  PetscCall(VecDestroy(&user.xlb));
  PetscCall(VecDestroy(&user.xub));
  PetscCall(PetscFinalize());
  return 0;
}

/*--------------------------------------------------------------------*/
/* Evaluate residual function A(x)-b in least square problem ||A(x)-b||^2 */
PetscErrorCode EvaluateResidual(Tao tao,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;

  PetscFunctionBegin;
  /* Compute Ax - b */
  PetscCall(MatMult(user->A,X,F));
  PetscCall(VecAXPY(F,-1,user->b));
  PetscLogFlops(2.0*user->M*user->N);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
PetscErrorCode EvaluateJacobian(Tao tao,Vec X,Mat J,Mat Jpre,void *ptr)
{
  /* Jacobian is not changing here, so use a empty dummy function here.  J[m][n] = df[m]/dx[n] = A[m][n] for linear least square */
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerObjectiveAndGradient(Tao tao,Vec X,PetscReal *f_reg,Vec G_reg,void *ptr)
{
  PetscFunctionBegin;
  /* compute regularizer objective = 0.5*x'*x */
  PetscCall(VecDot(X,X,f_reg));
  *f_reg *= 0.5;
  /* compute regularizer gradient = x */
  PetscCall(VecCopy(X,G_reg));
  PetscFunctionReturn(0);
}

PetscErrorCode EvaluateRegularizerHessianProd(Mat Hreg,Vec in,Vec out)
{
  PetscFunctionBegin;
  PetscCall(VecCopy(in,out));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode EvaluateRegularizerHessian(Tao tao,Vec X,Mat Hreg,void *ptr)
{
  /* Hessian for regularizer objective = 0.5*x'*x is identity matrix, and is not changing*/
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X,AppCtx *user)
{
  PetscFunctionBegin;
  PetscCall(VecSet(X,0.0));
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeUserData(AppCtx *user)
{
  PetscInt       k,n; /* indices for row and columns of D. */
  char           dataFile[] = "tomographyData_A_b_xGT";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. Debug: change from "tomographyData_A_b_xGT" to "cs1Data_A_b_xGT". */
  PetscInt       dictChoice = 1; /* choose from 0:identity, 1:gradient1D, 2:gradient2D, 3:DCT etc */
  PetscViewer    fd;   /* used to load data from file */
  PetscReal      v;

  PetscFunctionBegin;

  /*
  Matrix Vector read and write refer to:
  https://petsc.org/release/src/mat/tutorials/ex10.c
  https://petsc.org/release/src/mat/tutorials/ex12.c
 */
  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user->A));
  PetscCall(MatSetType(user->A,MATSEQAIJ));
  PetscCall(MatLoad(user->A,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->b));
  PetscCall(VecLoad(user->b,fd));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&user->xGT));
  PetscCall(VecLoad(user->xGT,fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(VecDuplicate(user->xGT,&(user->xlb)));
  PetscCall(VecSet(user->xlb,0.0));
  PetscCall(VecDuplicate(user->xGT,&(user->xub)));
  PetscCall(VecSet(user->xub,PETSC_INFINITY));

  /* Specify the size */
  PetscCall(MatGetSize(user->A,&user->M,&user->N));

  /* shortcut, when D is identity matrix, we may just specify it as NULL, and brgn will treat D*x as x without actually computing D*x.
  if (dictChoice == 0) {
    user->D = NULL;
    PetscFunctionReturn(0);
  }
  */

  /* Speficy D */
  /* (1) Specify D Size */
  switch (dictChoice) {
    case 0: /* 0:identity */
      user->K = user->N;
      break;
    case 1: /* 1:gradient1D */
      user->K = user->N-1;
      break;
  }

  PetscCall(MatCreate(PETSC_COMM_SELF,&user->D));
  PetscCall(MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->K,user->N));
  PetscCall(MatSetFromOptions(user->D));
  PetscCall(MatSetUp(user->D));

  /* (2) Specify D Content */
  switch (dictChoice) {
    case 0: /* 0:identity */
      for (k=0; k<user->K; k++) {
        v = 1.0;
        PetscCall(MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES));
      }
      break;
    case 1: /* 1:gradient1D.  [-1, 1, 0,...; 0, -1, 1, 0, ...] */
      for (k=0; k<user->K; k++) {
        v = 1.0;
        n = k+1;
        PetscCall(MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES));
        v = -1.0;
        PetscCall(MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES));
      }
      break;
  }
  PetscCall(MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY));

  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex !single !__float128 !defined(PETSC_USE_64BIT_INDICES)

   test:
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_max_it 1000 -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-8 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-8

   test:
      suffix: 2
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type l2prox -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

   test:
      suffix: 3
      localrunfiles: tomographyData_A_b_xGT
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_regularization_type user -tao_brgn_regularizer_weight 1e-8 -tao_gatol 1.e-6

TEST*/
