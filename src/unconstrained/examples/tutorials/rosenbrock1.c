/* Program usage: mpirun -np 1 rosenbrock1 [-help] [all TAO options] */

/*  Include "tao.h" so we can use TAO solvers.  */
#include "tao.h"

static  char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 ) \n";

/*T 
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoInitialize(); TaoFinalize();
   Routines: TaoCreate();
   Routines: TaoSetType(); TaoSetObjectiveAndGradientRoutine();
   Routines: TaoSetHessianRoutine();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoGetTerminationReason(); TaoDestroy(); 
   Processors: 1
T*/ 


/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/
typedef struct {
  PetscInt n;          /* dimension */
  PetscReal alpha;   /* condition parameter */
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(TaoSolver,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(TaoSolver,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;                  /* used to check for functions returning nonzeros */
  PetscReal zero=0.0;
  Vec        x;                     /* solution vector */
  Mat        H;
  TaoSolver  tao;                   /* TaoSolver solver context */
  PetscBool  flg;
  int        size,rank;                  /* number of processes running */
  TaoSolverTerminationReason reason;
  AppCtx     user;                  /* user-defined application context */

  /* Initialize TAO and PETSc */
  TaoInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(ierr);

  if (size >1) {
      if (rank == 0) {
	  PetscPrintf(PETSC_COMM_SELF,"This example is intended for single processor use!\n"); 
	  SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");
      }
  }


  /* Initialize problem parameters */
  user.n = 2; user.alpha = 99.0;
  /* Check for command line arguments to override defaults */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&user.n,&flg); CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-alpha",&user.alpha,&flg); CHKERRQ(ierr);

  /* Allocate vectors for the solution and gradient */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(ierr);
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,PETSC_NULL,&H); CHKERRQ(ierr);

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao); CHKERRQ(ierr);
  ierr = TaoSetType(tao,"tao_lmvm"); CHKERRQ(ierr);

  /* Set solution vec and an initial guess */
  ierr = VecSet(x, zero); CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,x); CHKERRQ(ierr); 

  /* Set routines for function, gradient, hessian evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(ierr);
  ierr = TaoSetHessianRoutine(tao,H,H,FormHessian,&user); CHKERRQ(ierr);
    
  /* Check for TAO command line options */
  ierr = TaoSetFromOptions(tao); CHKERRQ(ierr);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Get termination information */
  ierr = TaoGetTerminationReason(tao,&reason); CHKERRQ(ierr);
  if (reason <= 0)
    PetscPrintf(MPI_COMM_WORLD,"Try a different TAO type, adjust some parameters, or check the function evaluation routines\n");


  /* Free TAO data structures */
  ierr = TaoDestroy(&tao); CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = MatDestroy(&H); CHKERRQ(ierr);

  TaoFinalize();

  return 0;
}

/* -------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*  
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X). 

    Input Parameters:
.   tao  - the TaoSolver context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoSetFunctionGradient()
    
    Output Parameters:
.   G - vector containing the newly evaluated gradient
.   f - function value

    Note:
    Some optimization methods ask for the function and the gradient evaluation
    at the same time.  Evaluating both at once may be more efficient that
    evaluating each separately. 
*/
PetscErrorCode FormFunctionGradient(TaoSolver tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;  
  PetscInt    i,nn=user->n/2;
  PetscErrorCode ierr;
  PetscReal ff=0,t1,t2,alpha=user->alpha;
  PetscReal *x,*g;

  /* Get pointers to vector data */
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);
  ierr = VecGetArray(G,&g); CHKERRQ(ierr);

  /* Compute G(X) */
  for (i=0; i<nn; i++){
    t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
    ff += alpha*t1*t1 + t2*t2;
    g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
    g[2*i+1] = 2*alpha*t1;
  }

  /* Restore vectors */
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g); CHKERRQ(ierr);
  *f=ff;

  ierr = PetscLogFlops(nn*15); CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao   - the TaoSolver context
.  x     - input vector
.  ptr   - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix

   Note:  Providing the Hessian may not be necessary.  Only some solvers
   require this matrix.
*/
PetscErrorCode FormHessian(TaoSolver tao,Vec X,Mat *HH, Mat *Hpre, MatStructure *flag,void *ptr)
{
  AppCtx  *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt     i, nn=user->n/2, ind[2];
  PetscReal  alpha=user->alpha;
  PetscReal  v[2][2],*x;
  Mat H=*HH;
  PetscBool assembled;

  /* Zero existing matrix entries */
  ierr = MatAssembled(H,&assembled); CHKERRQ(ierr);
  if (assembled){ierr = MatZeroEntries(H);  CHKERRQ(ierr);}


  /* Get a pointer to vector data */
  ierr = VecGetArray(X,&x); CHKERRQ(ierr);

  /* Compute H(X) entries */
  for (i=0; i<user->n/2; i++){
    v[1][1] = 2*alpha;
    v[0][0] = -4*alpha*(x[2*i+1]-3*x[2*i]*x[2*i]) + 2;
    v[1][0] = v[0][1] = -4.0*alpha*x[2*i];
    ind[0]=2*i; ind[1]=2*i+1;
    ierr = MatSetValues(H,2,ind,2,ind,v[0],INSERT_VALUES); CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(X,&x); CHKERRQ(ierr);

  /* Assemble matrix */
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  *flag=SAME_NONZERO_PATTERN;
  
  ierr = PetscLogFlops(nn*9); CHKERRQ(ierr);
  return 0;
}
