/* Program usage: mpirun -np 1 rosenbrock1 [-help] [all TAO options] */

/*  Include "tao.h" so we can use TAO solvers.  */
#include "tao.h"

static  char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} ( alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2 ) \n";

/*T 
   Concepts: TAO - Solving an unconstrained minimization problem
   Routines: TaoSolverCreate(); TaoSolverDestroy(); 
   Routines: TaoSolverSetType();TaoSolverSetObjectiveAndGradient();
   Routines: TaoSolverSetFromOptions();
   Routines: TaoSolverSetInitialVector();
   Routines: TaoSolverSolve();
   Routines: TaoSolverGetTerminationReason;
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
  int        info;                  /* used to check for functions returning nonzeros */
  PetscReal zero=0.0;
  Vec        x;                     /* solution vector */
  Mat        H;
  TaoSolver  tao;                   /* TAO_SOLVER solver context */
  PetscBool  flg;
  int        size,rank;                  /* number of processes running */
  TaoSolverTerminationReason reason;
  AppCtx     user;                  /* user-defined application context */

  /* Initialize TAO and PETSc */
  //PetscInitialize(&argc,&argv,(char *)0,help);
  TaoInitialize(&argc,&argv,(char*)0,help);
  info = MPI_Comm_size(PETSC_COMM_WORLD,&size); CHKERRQ(info);
  info = MPI_Comm_rank(PETSC_COMM_WORLD,&rank); CHKERRQ(info);

  if (size >1) {
      if (rank == 0) {
	  PetscPrintf(PETSC_COMM_SELF,"This example is intended for single processor use!\n"); 
	  SETERRQ(PETSC_COMM_SELF,1,"Incorrect number of processors");
      }
  }


  /* Initialize problem parameters */
  user.n = 2; user.alpha = 99.0;

  /* Check for command line arguments to override defaults */
  info = PetscOptionsGetInt(PETSC_NULL,"-n",&user.n,&flg); CHKERRQ(info);
  info = PetscOptionsGetReal(PETSC_NULL,"-alpha",&user.alpha,&flg); CHKERRQ(info);

  /* Allocate vectors for the solution and gradient */
  info = VecCreateSeq(PETSC_COMM_SELF,user.n,&x); CHKERRQ(info);
  info = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,PETSC_NULL,&H); CHKERRQ(info);

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  info = TaoSolverCreate(PETSC_COMM_SELF,&tao); CHKERRQ(info);
  info = TaoSolverSetType(tao,"tao_lmvm"); CHKERRQ(info);

  /* Set solution vec and an initial guess */
  info = VecSet(x, zero); CHKERRQ(info);
  info = TaoSolverSetInitialVector(tao,x); CHKERRQ(info); 

  /* Set routines for function, gradient, hessian evaluation */
  info = TaoSolverSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,&user); CHKERRQ(info);
  info = TaoSolverSetHessianRoutine(tao,H,H,FormHessian,&user); CHKERRQ(info);
    
  /* Check for TAO command line options */
  info = TaoSolverSetFromOptions(tao); CHKERRQ(info);

  /* SOLVE THE APPLICATION */
  info = TaoSolverSolve(tao); CHKERRQ(info);

  /* Get termination information */
  info = TaoSolverGetTerminationReason(tao,&reason); CHKERRQ(info);
  // info = TaoSolverView(tao,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(info);
  if (reason <= 0)
    PetscPrintf(MPI_COMM_WORLD,"Try a different TAO type, adjust some parameters, or check the function evaluation routines\n");


  /* Free TAO data structures */
  info = TaoSolverDestroy(&tao); CHKERRQ(info);

  /* Free PETSc data structures */
  info = VecDestroy(&x); CHKERRQ(info);
  info = MatDestroy(&H); CHKERRQ(info);

  TaoFinalize();
//  PetscFinalize();

  return 0;
}

/* -------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
/*  
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X). 

    Input Parameters:
.   taoapp  - the TAO_APPLICATION context
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
  int    i,info,nn=user->n/2;
  PetscReal ff=0,t1,t2,alpha=user->alpha;
  PetscReal *x,*g;

  /* Get pointers to vector data */
  info = VecGetArray(X,&x); CHKERRQ(info);
  info = VecGetArray(G,&g); CHKERRQ(info);

  /* Compute G(X) */
  for (i=0; i<nn; i++){
    t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
    ff += alpha*t1*t1 + t2*t2;
    g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
    g[2*i+1] = 2*alpha*t1;
  }

  /* Restore vectors */
  info = VecRestoreArray(X,&x); CHKERRQ(info);
  info = VecRestoreArray(G,&g); CHKERRQ(info);
  *f=ff;

  info = PetscLogFlops(nn*15); CHKERRQ(info);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormHessian"
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  taoapp   - the TAO_APPLICATION context
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
  PetscErrorCode info;
  PetscInt     i, nn=user->n/2, ind[2];
  PetscReal  alpha=user->alpha;
  PetscReal  v[2][2],*x;
  Mat H=*HH;
  PetscBool assembled;

  /* Zero existing matrix entries */
  info = MatAssembled(H,&assembled); CHKERRQ(info);
  if (assembled){info = MatZeroEntries(H);  CHKERRQ(info);}


  /* Get a pointer to vector data */
  info = VecGetArray(X,&x); CHKERRQ(info);

  /* Compute H(X) entries */
  for (i=0; i<user->n/2; i++){
    v[1][1] = 2*alpha;
    v[0][0] = -4*alpha*(x[2*i+1]-3*x[2*i]*x[2*i]) + 2;
    v[1][0] = v[0][1] = -4.0*alpha*x[2*i];
    ind[0]=2*i; ind[1]=2*i+1;
    info = MatSetValues(H,2,ind,2,ind,v[0],INSERT_VALUES); CHKERRQ(info);
  }
  info = VecRestoreArray(X,&x); CHKERRQ(info);

  /* Assemble matrix */
  info = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  *flag=SAME_NONZERO_PATTERN;
  
  info = PetscLogFlops(nn*9); CHKERRQ(info);
  return 0;
}
