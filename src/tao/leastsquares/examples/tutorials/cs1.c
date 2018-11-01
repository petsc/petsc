/* XH: todo add cs1f.F90 and asjust makefile */
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
               0.5*||Ax-b||^2 + lambda*||x||_1                
               Xiang Huang: Oct 31, 2018

Reference:     None
               
*/



static char help[]="Finds the least-squares solution to the underconstraint linear model Ax = b. \n\
            A is a M*N real flat matrix (M<N), x is sparse. \n\
            We find the sparse solution by solving 0.5*||Ax-b||^2 + lambda*||x||_1, where lambda (by default 1e-4) is a user specified weight.\n";
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



#define M 3
#define N 5

/* User-defined application context */
/* XH: t -> A,  y--> b, j = j, but j = A */
typedef struct {
  /* Working space */
  PetscReal A[M][N];    /* array of independent variables of observation */
  PetscReal b[M];       /* array of dependent variables */
  PetscReal j[M][N];    /* dense jacobian matrix array. For linear least square, j = A. For nonlinear least square, it is different from A */
  PetscInt  idm[M];     /* Matrix indices for jacobian */
  PetscInt  idn[N];
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeData(AppCtx *user);
PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao, Vec, Vec, void *);
PetscErrorCode EvaluateJacobian(Tao, Vec, Mat, Mat, void *);


/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;               /* used to check for functions returning nonzeros */
  Vec            x, f;               /* solution, function */
  Mat            J;                  /* Jacobian matrix */
  Tao            tao;                /* Tao solver context */
  PetscInt       i;                  /* iteration information */
  PetscReal      hist[100],resid[100];
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);
  /* Allocate vectors */
  ierr = VecCreateSeq(MPI_COMM_SELF,N,&x);CHKERRQ(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,M,&f);CHKERRQ(ierr);

  /* Create the Jacobian matrix. */
  ierr = MatCreateSeqDense(MPI_COMM_SELF,M,N,NULL,&J);CHKERRQ(ierr);

  for (i=0;i<M;i++) user.idm[i] = i;

  for (i=0;i<N;i++) user.idn[i] = i;

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);

 /* Set the function and Jacobian routines. */
  ierr = InitializeData(&user);CHKERRQ(ierr);
  ierr = FormStartingPoint(x);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
  ierr = TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianResidualRoutine(tao, J, J, EvaluateJacobian, (void*)&user);CHKERRQ(ierr);

  /* Check for any TAO command line arguments */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE);CHKERRQ(ierr);
  /* Perform the Solve */
  ierr = TaoSolve(tao);CHKERRQ(ierr);


  /*
     Assemble vector, using the 2-step process: VecAssemblyBegin(), VecAssemblyEnd()
     Computations can be done while messages are in transition by placing code between these two statements.
  */
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  /* View the vector.  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(f,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

   /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&f);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*--------------------------------------------------------------------*/
PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscInt       m,n;
  const PetscReal *x;
  PetscReal      *b=user->b,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  
  /* Even for linear least square, we do not use matrix operation f = b-A*x now, just for future modification and compatability for nonlinear least square */
  for (m=0;m<M;m++) {
    f[m] = b[m];
    for (n=0;n<N;n++) {
      f[m] -= user->A[m][n]*x[n];
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscLogFlops(6*M);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/* J[m][n] = df[m]/dt[n] */
PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscInt       m,n;
  const PetscReal *x;  
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr); /* not used for linear least square, but keep for future nonlinear least square) */
  /* For linear least square, we can just set j=A, but for nonlinear least square, we require x to compute j, keep codes here for future nonlinear least square*/
  for (m=0;m<M;m++) {
    for (n=0;n<N;n++) {  
      user->j[m][n] = user->A[m][n];
    }
  }

  /* Assemble the matrix */
  ierr = MatSetValues(J,M,user->idm, N, user->idn,(PetscReal *)user->j,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);  /* not used for linear least square, but keep for future nonlinear least square) */
  PetscLogFlops(M * 13);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X)
{
  PetscReal      *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 0;
  x[1] = 0;
  x[2] = 0;
  x[3] = 0;
  x[4] = 0;
  VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeData(AppCtx *user)
{
  PetscReal *b=user->b; /* **A=user->A, but we don't kown the dimension of A in this way */
  PetscInt  m=0, n=0;

  PetscFunctionBegin;
  /* b = A*x while x = [0;0;1;0;0] here*/
  b[m++] = 0.28;
  b[m++] = 0.55;
  b[m++] = 0.96;
  
  /* matlab generated random matrix, uniformly distributed in [0,1] with 2 digits accuracy. rng(0); A = rand(M, N); A = round(A*100)/100;
  A = [0.81  0.91  0.28  0.96  0.96
       0.91  0.63  0.55  0.16  0.49
       0.13  0.10  0.96  0.97  0.80]       
  */
  m=0; n=0; user->A[m][n++] = 0.81; user->A[m][n++] = 0.91; user->A[m][n++] = 0.28; user->A[m][n++] = 0.96; user->A[m][n++] = 0.96; 
  ++m; n=0; user->A[m][n++] = 0.91; user->A[m][n++] = 0.63; user->A[m][n++] = 0.55; user->A[m][n++] = 0.16; user->A[m][n++] = 0.49;
  ++m; n=0; user->A[m][n++] = 0.13; user->A[m][n++] = 0.10; user->A[m][n++] = 0.96; user->A[m][n++] = 0.97; user->A[m][n++] = 0.80;
  
  PetscFunctionReturn(0);
}


/*TEST

   build:
      requires: !complex

   test:
      args: -tao_smonitor -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5
      requires: !single
      
   test:
      suffix: 2
      args: -tao_smonitor -tao_max_it 100 -tao_type brgn -tao_gatol 1.e-5
      requires: !single

TEST*/

/* XH: Done InitializeData() and FormStartingPoint()*/