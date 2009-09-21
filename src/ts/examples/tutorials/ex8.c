
/* Program usage:  ./ex8 [-help] [all PETSc options] */

static char help[] = "Nonlinear DAE.\n";


/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include "petscts.h"

typedef struct _Problem *Problem;
struct _Problem {
  PetscErrorCode (*create)(MPI_Comm,void**);
  PetscErrorCode (*destroy)(void*);
  TSIFunction function;
  TSIJacobian jacobian;
  PetscErrorCode (*solution)(PetscReal,Vec,void*);
  PetscReal final_time;
  PetscInt n;
};

/*
*  User-defined routines
*/

/*
*  Stiff 3-variable system from chemical reactions, due to Robertson (1966), problem ROBER in Hairer&Wanner, ODE 2, 1996
*/
#undef __FUNCT__
#define __FUNCT__ "RoberFunction"
static PetscErrorCode RoberFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar *x,*xdot,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + 0.04*x[0] - 1e4*x[1]*x[2];
  f[1] = xdot[1] - 0.04*x[0] + 1e4*x[1]*x[2] + 3e7*PetscSqr(x[1]);
  f[2] = xdot[2] - 3e7*PetscSqr(x[1]);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RoberJacobian"
static PetscErrorCode RoberJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscErrorCode ierr;
  PetscInt rowcol[] = {0,1,2};
  PetscScalar *x,*xdot,J[3][3];

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + 0.04;     J[0][1] = -1e4*x[2];                   J[0][2] = -1e4*x[1];
  J[1][0] = -0.04;        J[1][1] = a + 1e4*x[2] + 3e7*2*x[1];   J[1][2] = 1e4*x[1];
  J[2][0] = 0;            J[2][1] = -3e7*2*x[1];                 J[2][2] = a;
  ierr = MatSetValues(*B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RoberSolution"
static PetscErrorCode RoberSolution(PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar *x;

  PetscFunctionBegin;
  if (t != 0) SETERRQ(1,"not implemented");
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct _Problem problem_rober = {0,0,&RoberFunction,&RoberJacobian,&RoberSolution,1e11,3};


typedef struct {
  PetscReal lambda;
} CECtx;


/*
* Stiff scalar valued problem with an exact solution
*/
#undef __FUNCT__
#define __FUNCT__ "CECreate"
static PetscErrorCode CECreate(MPI_Comm comm,void **ctx)
{
  PetscErrorCode ierr;
  CECtx         *ce;

  PetscFunctionBegin;
  ierr = PetscMalloc(sizeof(CECtx),&ce);CHKERRQ(ierr);
  *ctx = (void*)ce;

  ce->lambda = 10;
  ierr = PetscOptionsBegin(comm,PETSC_NULL,"CE options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-problem_ce_lambda","Parameter controlling stiffness: xdot + lambda*(x - cos(t))","",ce->lambda,&ce->lambda,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CEDestroy"
static PetscErrorCode CEDestroy(void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CEFunction"
static PetscErrorCode CEFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  PetscReal l = ((CECtx*)ctx)->lambda;
  PetscScalar *x,*xdot,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + l*(x[0] - cos(t));
#if 0
  ierr = PetscPrintf(PETSC_COMM_WORLD," f(t=%G,x=%G,xdot=%G) = %G\n",t,x[0],xdot[0],f[0]);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CEJacobian"
static PetscErrorCode CEJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
  PetscReal l = ((CECtx*)ctx)->lambda;
  PetscErrorCode ierr;
  PetscInt rowcol[] = {0};
  PetscScalar *x,*xdot,J[1][1];

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + l;
  ierr = MatSetValues(*B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "CESolution"
static PetscErrorCode CESolution(PetscReal t,Vec X,void *ctx)
{
  PetscReal l = ((CECtx*)ctx)->lambda;
  PetscErrorCode ierr;
  PetscScalar *x;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = l/(l*l+1)*(l*cos(t)+sin(t)) - l*l/(l*l+1)*exp(-l*t);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

struct _Problem problem_ce = {&CECreate,&CEDestroy,&CEFunction,&CEJacobian,&CESolution,2,1};


/*
*  User-defined monitor for comparing to exact solutions when possible
*/
typedef struct {
  MPI_Comm  comm;
  Problem   problem;
  void     *user;
  Vec       x;
} MonitorCtx;

#undef __FUNCT__
#define __FUNCT__ "MonitorError"
static PetscErrorCode MonitorError(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx)
{
  PetscErrorCode ierr;
  MonitorCtx *mon = (MonitorCtx*)ctx;
  PetscReal nrm_x,nrm_exact,nrm_diff;

  PetscFunctionBegin;
  if (!mon->problem->solution) PetscFunctionReturn(0);
  ierr = (*mon->problem->solution)(t,mon->x,mon->user);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&nrm_x);CHKERRQ(ierr);
  ierr = VecNorm(mon->x,NORM_2,&nrm_exact);CHKERRQ(ierr);
  ierr = VecAYPX(mon->x,-1,x);CHKERRQ(ierr);
  ierr = VecNorm(mon->x,NORM_2,&nrm_diff);CHKERRQ(ierr);
  ierr = PetscPrintf(mon->comm,"step %4D t=%12G  |x|=%9.2e  |x_e|=%9.2e  |x-x_e|=%9.2e\n",step,t,nrm_x,nrm_exact,nrm_diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS              ts;           /* nonlinear solver */
  Vec             x,r;          /* solution, residual vectors */
  Mat             A;            /* Jacobian matrix */
  Problem         problem;
  PetscInt        steps,maxsteps = 100;
  PetscReal       ftime;
  void           *user;
  MonitorCtx      mon;
  PetscErrorCode  ierr;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char *)0,help);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Choose which problem to solve
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  problem = &problem_ce;

  if (problem->create) {ierr = (*problem->create)(MPI_COMM_WORLD,&user);CHKERRQ(ierr);}

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,problem->n,problem->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);

  ierr = MatGetVecs(A,&x,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  mon.comm = PETSC_COMM_WORLD;
  mon.problem = problem;
  mon.user = user;
  ierr = VecDuplicate(x,&mon.x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,problem->function,user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,problem->jacobian,user);CHKERRQ(ierr);

  ierr = TSSetDuration(ts,maxsteps,problem->final_time);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = (*problem->solution)(0,x,user);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Timestepping benchmark options","");CHKERRQ(ierr);
  {
    PetscTruth flg;

    flg = PETSC_FALSE;
    ierr = PetscOptionsTruth("-monitor_error","Display errors relative to exact solutions","",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = TSMonitorSet(ts,&MonitorError,&mon,PETSC_NULL);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSStep(ts,&steps,&ftime);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"steps %D, ftime %G\n",steps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(r);CHKERRQ(ierr);
  ierr = VecDestroy(mon.x);CHKERRQ(ierr);
  ierr = TSDestroy(ts);CHKERRQ(ierr);
  if (problem->destroy) {
    ierr = (*problem->destroy)(user);CHKERRQ(ierr);
  }

  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
