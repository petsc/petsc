/* XH: 
    Todo: add cs1f.F90 and adjust makefile. 
    Todo: maybe provide code template to generate 1D/2D/3D gradient, DCT tranform matrix for D etc.
*/
/*
    XH: refactored example chwirut1 so that user contest contains not arrays but matrix and vectors
*/
/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/


#include <petsctao.h>

/*
Description:   Compressive sensing test example 1.
               0.5*||Ax-b||^2 + lambda*||D*x||_1                
               Xiang Huang: Nov 19, 2018

Reference:     None
               
*/

static char help[] = "Finds the least-squares solution to the underconstraint linear model Ax = b, with L1-norm regularizer. \n\
            A is a M*N real flat matrix (M<N), x is sparse. \n\
            We find the sparse solution by solving 0.5*||Ax-b||^2 + lambda*||D*x||_1, where lambda (by default 1e-4) is a user specified weight.\n\
            D is the K*N transform matrix so that D*x is sparse. By default D is identity matrix, so that D*x = x.\n";
/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/




/* User-defined application context */
typedef struct {
  /* Working space. linear least square:  f(x) = A*x - b */
  /*PetscReal A[M][N]; */    /* array of coefficients */
  PetscInt  M,N,K;      /* Problem dimension: A is M*N Matrix, D is K*N Matrix */
  Mat       A,D;      /* Coefficients, Dictionary Transform of size M*N and K*N respectively. For linear least square, Jacobian Matrix J = A. For nonlinear least square, it is different from A */  
  Vec       b,xGT,xlb,xub; /* observation b, ground truth xGT, the lower bound and upper bound of x */
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeUserData(AppCtx *);
PetscErrorCode FormStartingPoint(Vec,AppCtx *);
PetscErrorCode EvaluateFunction(Tao,Vec,Vec,void *);
PetscErrorCode EvaluateJacobian(Tao,Vec,Mat,Mat,void *);


/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;               /* used to check for functions returning nonzeros */
  Vec            x,f;               /* solution, function f(x) = A*x-b */
  Mat            J;               /* Jacobian matrix */
  Tao            tao;                /* Tao solver context */  
  PetscReal      hist[100],resid[100],v1,v2;
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */
  PetscViewer    fd;   /* used to save result to file */
  char           resultFile[] = "tomographyResult_x";  /* Debug: change from "tomographyResult_x" to "cs1Result_x" */  

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBRGN);CHKERRQ(ierr);

  /* User set application context: A, D matrice, and b vector. */   
  ierr = InitializeUserData(&user);CHKERRQ(ierr);

  /* Allocate solution vector x,  and function vectors Ax-b, */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.N,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.M,&f);CHKERRQ(ierr);

  /* Allocate Jacobian matrix. */
  ierr = MatConvert(user.A,MATSAME,MAT_INITIAL_MATRIX,&J);

  /* Set initial guess */
  ierr = FormStartingPoint(x,&user);CHKERRQ(ierr);
  
  /* Bind x to tao->solution. */
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  /* Sets the upper and lower bounds of x */
  ierr = TaoSetVariableBounds(tao,user.xlb,user.xub);CHKERRQ(ierr);
  /* Bind user.D to tao->data->D */
  ierr = TaoBRGNSetDictionaryMatrix(tao,user.D);CHKERRQ(ierr);

  /* Set the function and Jacobian routines. */
  ierr = TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianResidualRoutine(tao,J,J,EvaluateJacobian,(void*)&user);CHKERRQ(ierr);

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);
  
  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);


  /* XH: Debug: Do we really need to assembly the vector? should be called after completing all calls to VecSetValues()
     Assemble vector, using the 2-step process: VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition by placing code between these two statements.
  */
  /* Is it neccssary to use TaoGetSolutionVector(tao, &x); */
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /* Save x (reconstruction of object) vector to a binary file, which maybe read from Matlab and convert to a 2D image for comparison. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,resultFile,FILE_MODE_WRITE,&fd);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);

  /* compute the error */
  ierr = VecAXPY(x,-1,user.xGT);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&v1);CHKERRQ(ierr);
  ierr = VecNorm(user.xGT,NORM_2,&v2);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "relative reconstruction error: ||x-xGT||/||xGT|| = %6.4e.\n", (double)(v1/v2));CHKERRQ(ierr);

  /* XH: Debug: View the result, function and Jacobian.  */      
#if 0
  PetscPrintf(PETSC_COMM_SELF,"-------- result x, residual f=A*x-b, Jacobian=A, and D Matrix. -------- \n");  
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(J,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(user.A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatView(user.D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
#endif

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  /* Free user data structures: maybe not necessary for user.A and uuser.D since they are binded to J and tao->data->D, but just destroy to make sure? */  
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = MatDestroy(&user.D);CHKERRQ(ierr);
  ierr = VecDestroy(&user.b);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xGT);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xlb);CHKERRQ(ierr);
  ierr = VecDestroy(&user.xub);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*--------------------------------------------------------------------*/
PetscErrorCode EvaluateFunction(Tao tao,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Compute Ax - b */  
  ierr = MatMult(user->A,X,F);CHKERRQ(ierr);
  ierr = VecAXPY(F,-1,user->b);CHKERRQ(ierr);
  PetscLogFlops(user->M*user->N*2);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/* J[m][n] = df[m]/dx[n] = A[m][n] for linear least square */
PetscErrorCode EvaluateJacobian(Tao tao,Vec X,Mat J,Mat Jpre,void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  if (user->A) {
    /* PetscValidHeaderSpecific(user->A,MAT_CLASSID,2); */ /* Todo: XH do not understand what is this valiidation for and it cause compile error, so commented out. originally copied from TaoBRGNSetL1RegularizerWeight*/
    /*PetscCheckSameComm(J,1,user->A,2);*/
    /* TODO: how to just bind J to user->A instead of copy the content to it? Do we really need to pass Mat *J instead of Mat J?
    ierr = PetscObjectReference((PetscObject)user->A);CHKERRQ(ierr);
    ierr = MatDestroy(&J);CHKERRQ(ierr);
    J = user->A;  
    */
    ierr = MatCopy(user->A,J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }  
  PetscLogFlops(0);  /* 0 for linear least square, >0 for nonlinear least square */
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X,AppCtx *user) 
{
  PetscReal      *x;
  PetscErrorCode ierr;
  PetscInt       n;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  for (n=0; n<user->N; n++) x[0] = 0.0;  /* XH: i.e. VecSet(x, 0); */
  VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeUserData(AppCtx *user)
{  
  PetscInt       k,n; /* indices for row and columns of D. */  
  char           dataFile[] = "tomographyData_A_b_xGT";   /* Matrix A and vectors b, xGT(ground truth) binary files generated by Matlab. Debug: change from "tomographyData_A_b_xGT" to "cs1Data_A_b_xGT". */  
  PetscInt       dictChoice = 1; /* choose from 0:identity, 1:gradient1D, 2:gradient2D, 3:DCT etc */
  PetscViewer    fd;   /* used to load data from file */
  PetscErrorCode ierr;
  PetscReal      v;

  PetscFunctionBegin;

  /* 
  Matrix Vector read and write refer to: 
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex10.c 
  https://www.mcs.anl.gov/petsc/petsc-current/src/mat/examples/tutorials/ex12.c
 */
  /* Load the A matrix, b vector, and xGT vector from a binary file. */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,dataFile,FILE_MODE_READ,&fd);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&user->A);CHKERRQ(ierr);
  ierr = MatSetType(user->A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatLoad(user->A,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->b);CHKERRQ(ierr);
  ierr = VecLoad(user->b,fd);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->xGT);CHKERRQ(ierr);
  ierr = VecLoad(user->xGT,fd);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&fd);CHKERRQ(ierr);
  ierr = VecDuplicate(user->xGT,&(user->xlb));CHKERRQ(ierr);
  ierr = VecSet(user->xlb,0.0);CHKERRQ(ierr);
  ierr = VecDuplicate(user->xGT,&(user->xub));CHKERRQ(ierr);
  ierr = VecSet(user->xub,PETSC_INFINITY);CHKERRQ(ierr);

  /* Specify the size */
  ierr = MatGetSize(user->A,&user->M,&user->N);CHKERRQ(ierr);

  /* Specify D */
  /* Todo: let user specify dictChoice from input */ 
  switch (dictChoice) {
    case 0: /* 0:identity */ 
      user->K = user->N;
      break;
    case 1: /* 1:gradient1D */
      user->K = user->N-1;
      break;
  }

  ierr = MatCreate(PETSC_COMM_SELF,&user->D);CHKERRQ(ierr);
  ierr = MatSetSizes(user->D,PETSC_DECIDE,PETSC_DECIDE,user->K,user->N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user->D);CHKERRQ(ierr);
  ierr = MatSetUp(user->D);CHKERRQ(ierr);

  switch (dictChoice) {
    case 0: /* 0:identity */ 
      for (k=0; k<user->K; k++) {
        v = 1.0;    
        ierr = MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      break;
    case 1: /* 1:gradient1D.  [-1, 1, 0,...; 0, -1, 1, 0, ...] */
      for (k=0; k<user->K; k++) {
        v = 1.0;
        n = k+1; 
        ierr = MatSetValues(user->D,1,&k,1,&n,&v,INSERT_VALUES);CHKERRQ(ierr);
        v = -1.0;        
        ierr = MatSetValues(user->D,1,&k,1,&k,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
      break;
  }
  ierr = MatAssemblyBegin(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(user->D,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

   /* Verify the norms for A, D, b, xGT, only used in the debug?*/
/* #if 0 */
  ierr = MatNorm(user->A,NORM_FROBENIUS,&v);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "A has size %dx%d and Frobenius-norm: %6.4e.\n", user->M, user->N, (double)v);CHKERRQ(ierr);

  ierr = MatNorm(user->D,NORM_FROBENIUS,&v);CHKERRQ(ierr);
  ierr = MatGetSize(user->D,&k,&n);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "D has size %dx%d and Frobenius-norm: %6.4e.\n", k, n, (double)v);CHKERRQ(ierr);

  ierr = VecNorm(user->b,NORM_2,&v);CHKERRQ(ierr);
  ierr = VecGetSize(user->b,&user->M);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "b has size %d and 2-norm: %6.4e.\n", user->M, (double)v);CHKERRQ(ierr);

  ierr = VecNorm(user->xGT,NORM_2,&v);CHKERRQ(ierr);
  ierr = VecGetSize(user->xGT,&user->N);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "xGT has size %d and 2-norm: %6.4e.\n", user->N, (double)v);CHKERRQ(ierr);
/* #endif */

  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex  XH: template from chwirut1.c, why it requires complex?

   test:
      args: -tao_monitor -tao_max_it 1000 -tao_brgn_lambda 1e-8 -tao_brgn_epsilon 1e-6 -tao_gatol 1.e-8
      requires: !single
      

TEST*/