/* Program usage: mpiexec -n 1 rosenbrock2 [-help] [all TAO options] */

/*  Include "petsctao.h" so we can use TAO solvers.  */
#include <petsctao.h>

static  char help[] = "This example demonstrates use of the TAO package to \n\
solve an unconstrained minimization problem on a single processor.  We \n\
minimize the extended Rosenbrock function: \n\
   sum_{i=0}^{n/2-1} (alpha*(x_{2i+1}-x_{2i}^2)^2 + (1-x_{2i})^2) \n\
or the chained Rosenbrock function:\n\
   sum_{i=0}^{n-1} alpha*(x_{i+1} - x_i^2)^2 + (1 - x_i)^2\n";

/*T
   Concepts: TAO^Solving an unconstrained minimization problem
   Routines: TaoCreate();
   Routines: TaoSetType(); TaoSetObjectiveAndGradient();
   Routines: TaoSetHessian();
   Routines: TaoSetSolution();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: 1
T*/

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines that evaluate the function,
   gradient, and hessian.
*/
typedef struct {
  PetscInt  n;          /* dimension */
  PetscReal alpha;   /* condition parameter */
  PetscBool chained;
} AppCtx;

/* -------------- User-defined routines ---------- */
PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  PetscErrorCode     ierr;                  /* used to check for functions returning nonzeros */
  PetscReal          zero=0.0;
  Vec                x;                     /* solution vector */
  Mat                H;
  Tao                tao;                   /* Tao solver context */
  PetscBool          flg, test_lmvm = PETSC_FALSE;
  PetscMPIInt        size;                  /* number of processes running */
  AppCtx             user;                  /* user-defined application context */
  TaoConvergedReason reason;
  PetscInt           its, recycled_its=0, oneshot_its=0;

  /* Initialize TAO and PETSc */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheckFalse(size >1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Incorrect number of processors");

  /* Initialize problem parameters */
  user.n = 2; user.alpha = 99.0; user.chained = PETSC_FALSE;
  /* Check for command line arguments to override defaults */
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&user.n,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-alpha",&user.alpha,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-chained",&user.chained,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_lmvm",&test_lmvm,&flg);CHKERRQ(ierr);

  /* Allocate vectors for the solution and gradient */
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.n,&x);CHKERRQ(ierr);
  ierr = MatCreateSeqBAIJ(PETSC_COMM_SELF,2,user.n,user.n,1,NULL,&H);CHKERRQ(ierr);

  /* The TAO code begins here */

  /* Create TAO solver with desired solution method */
  ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOLMVM);CHKERRQ(ierr);

  /* Set solution vec and an initial guess */
  ierr = VecSet(x, zero);CHKERRQ(ierr);
  ierr = TaoSetSolution(tao,x);CHKERRQ(ierr);

  /* Set routines for function, gradient, hessian evaluation */
  ierr = TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,&user);CHKERRQ(ierr);
  ierr = TaoSetHessian(tao,H,H,FormHessian,&user);CHKERRQ(ierr);

  /* Check for TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Solve the problem */
  ierr = TaoSetTolerances(tao, 1.e-5, 0.0, 0.0);CHKERRQ(ierr);
  ierr = TaoSetMaximumIterations(tao, 5);CHKERRQ(ierr);
  ierr = TaoLMVMRecycle(tao, PETSC_TRUE);CHKERRQ(ierr);
  reason = TAO_CONTINUE_ITERATING;
  while (reason != TAO_CONVERGED_GATOL) {
    ierr = TaoSolve(tao);CHKERRQ(ierr);
    ierr = TaoGetConvergedReason(tao, &reason);CHKERRQ(ierr);
    ierr = TaoGetIterationNumber(tao, &its);CHKERRQ(ierr);
    recycled_its += its;
    ierr = PetscPrintf(PETSC_COMM_SELF, "-----------------------\n");CHKERRQ(ierr);
  }

  /* Disable recycling and solve again! */
  ierr = TaoSetMaximumIterations(tao, 100);CHKERRQ(ierr);
  ierr = TaoLMVMRecycle(tao, PETSC_FALSE);CHKERRQ(ierr);
  ierr = VecSet(x, zero);CHKERRQ(ierr);
  ierr = TaoSolve(tao);CHKERRQ(ierr);
  ierr = TaoGetConvergedReason(tao, &reason);CHKERRQ(ierr);
  PetscCheckFalse(reason != TAO_CONVERGED_GATOL,PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "Solution failed to converge!");
  ierr = TaoGetIterationNumber(tao, &oneshot_its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "-----------------------\n");CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF, "recycled its: %D | oneshot its: %D\n", recycled_its, oneshot_its);CHKERRQ(ierr);
  PetscCheckFalse(recycled_its != oneshot_its,PETSC_COMM_SELF, PETSC_ERR_NOT_CONVERGED, "LMVM recycling does not work!");

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = MatDestroy(&H);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* -------------------------------------------------------------------- */
/*
    FormFunctionGradient - Evaluates the function, f(X), and gradient, G(X).

    Input Parameters:
.   tao  - the Tao context
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
PetscErrorCode FormFunctionGradient(Tao tao,Vec X,PetscReal *f, Vec G,void *ptr)
{
  AppCtx            *user = (AppCtx *) ptr;
  PetscInt          i,nn=user->n/2;
  PetscErrorCode    ierr;
  PetscReal         ff=0,t1,t2,alpha=user->alpha;
  PetscScalar       *g;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  /* Get pointers to vector data */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);

  /* Compute G(X) */
  if (user->chained) {
    g[0] = 0;
    for (i=0; i<user->n-1; i++) {
      t1 = x[i+1] - x[i]*x[i];
      ff += PetscSqr(1 - x[i]) + alpha*t1*t1;
      g[i] += -2*(1 - x[i]) + 2*alpha*t1*(-2*x[i]);
      g[i+1] = 2*alpha*t1;
    }
  } else {
    for (i=0; i<nn; i++) {
      t1 = x[2*i+1]-x[2*i]*x[2*i]; t2= 1-x[2*i];
      ff += alpha*t1*t1 + t2*t2;
      g[2*i] = -4*alpha*t1*x[2*i]-2.0*t2;
      g[2*i+1] = 2*alpha*t1;
    }
  }

  /* Restore vectors */
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  *f   = ff;

  ierr = PetscLogFlops(15.0*nn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormHessian - Evaluates Hessian matrix.

   Input Parameters:
.  tao   - the Tao context
.  x     - input vector
.  ptr   - optional user-defined context, as set by TaoSetHessian()

   Output Parameters:
.  H     - Hessian matrix

   Note:  Providing the Hessian may not be necessary.  Only some solvers
   require this matrix.
*/
PetscErrorCode FormHessian(Tao tao,Vec X,Mat H, Mat Hpre, void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscErrorCode    ierr;
  PetscInt          i, ind[2];
  PetscReal         alpha=user->alpha;
  PetscReal         v[2][2];
  const PetscScalar *x;
  PetscBool         assembled;

  PetscFunctionBeginUser;
  /* Zero existing matrix entries */
  ierr = MatAssembled(H,&assembled);CHKERRQ(ierr);
  if (assembled) {ierr = MatZeroEntries(H);CHKERRQ(ierr);}

  /* Get a pointer to vector data */
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);

  /* Compute H(X) entries */
  if (user->chained) {
    ierr = MatZeroEntries(H);CHKERRQ(ierr);
    for (i=0; i<user->n-1; i++) {
      PetscScalar t1 = x[i+1] - x[i]*x[i];
      v[0][0] = 2 + 2*alpha*(t1*(-2) - 2*x[i]);
      v[0][1] = 2*alpha*(-2*x[i]);
      v[1][0] = 2*alpha*(-2*x[i]);
      v[1][1] = 2*alpha*t1;
      ind[0] = i; ind[1] = i+1;
      ierr = MatSetValues(H,2,ind,2,ind,v[0],ADD_VALUES);CHKERRQ(ierr);
    }
  } else {
    for (i=0; i<user->n/2; i++) {
      v[1][1] = 2*alpha;
      v[0][0] = -4*alpha*(x[2*i+1]-3*x[2*i]*x[2*i]) + 2;
      v[1][0] = v[0][1] = -4.0*alpha*x[2*i];
      ind[0]=2*i; ind[1]=2*i+1;
      ierr = MatSetValues(H,2,ind,2,ind,v[0],INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);

  /* Assemble matrix */
  ierr = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(9.0*user->n/2.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_type lmvm -tao_monitor
      requires: !single

TEST*/
