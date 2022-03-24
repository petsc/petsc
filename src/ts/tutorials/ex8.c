
static char help[] = "Nonlinear DAE benchmark problems.\n";

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscts.h>

typedef struct _Problem* Problem;
struct _Problem {
  PetscErrorCode (*destroy)(Problem);
  TSIFunction    function;
  TSIJacobian    jacobian;
  PetscErrorCode (*solution)(PetscReal,Vec,void*);
  MPI_Comm       comm;
  PetscReal      final_time;
  PetscInt       n;
  PetscBool      hasexact;
  void           *data;
};

/*
      Stiff 3-variable system from chemical reactions, due to Robertson (1966), problem ROBER in Hairer&Wanner, ODE 2, 1996
*/
static PetscErrorCode RoberFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = xdot[0] + 0.04*x[0] - 1e4*x[1]*x[2];
  f[1] = xdot[1] - 0.04*x[0] + 1e4*x[1]*x[2] + 3e7*PetscSqr(x[1]);
  f[2] = xdot[2] - 3e7*PetscSqr(x[1]);
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  J[0][0] = a + 0.04;     J[0][1] = -1e4*x[2];                   J[0][2] = -1e4*x[1];
  J[1][0] = -0.04;        J[1][1] = a + 1e4*x[2] + 3e7*2*x[1];   J[1][2] = 1e4*x[1];
  J[2][0] = 0;            J[2][1] = -3e7*2*x[1];                 J[2][2] = a;
  CHKERRQ(MatSetValues(B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberSolution(PetscReal t,Vec X,void *ctx)
{
  PetscScalar    *x;

  PetscFunctionBeginUser;
  PetscCheck(t == 0,PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented");
  CHKERRQ(VecGetArray(X,&x));
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  CHKERRQ(VecRestoreArray(X,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberCreate(Problem p)
{
  PetscFunctionBeginUser;
  p->destroy    = 0;
  p->function   = &RoberFunction;
  p->jacobian   = &RoberJacobian;
  p->solution   = &RoberSolution;
  p->final_time = 1e11;
  p->n          = 3;
  PetscFunctionReturn(0);
}

/*
     Stiff scalar valued problem
*/

typedef struct {
  PetscReal lambda;
} CECtx;

static PetscErrorCode CEDestroy(Problem p)
{
  PetscFunctionBeginUser;
  CHKERRQ(PetscFree(p->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode CEFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscReal         l = ((CECtx*)ctx)->lambda;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = xdot[0] + l*(x[0] - PetscCosReal(t));
#if 0
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," f(t=%g,x=%g,xdot=%g) = %g\n",(double)t,(double)x[0],(double)xdot[0],(double)f[0]));
#endif
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode CEJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscReal         l = ((CECtx*)ctx)->lambda;
  PetscInt          rowcol[] = {0};
  PetscScalar       J[1][1];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  J[0][0] = a + l;
  CHKERRQ(MatSetValues(B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CESolution(PetscReal t,Vec X,void *ctx)
{
  PetscReal      l = ((CECtx*)ctx)->lambda;
  PetscScalar    *x;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArray(X,&x));
  x[0] = l/(l*l+1)*(l*PetscCosReal(t)+PetscSinReal(t)) - l*l/(l*l+1)*PetscExpReal(-l*t);
  CHKERRQ(VecRestoreArray(X,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode CECreate(Problem p)
{
  PetscErrorCode ierr;
  CECtx          *ce;

  PetscFunctionBeginUser;
  CHKERRQ(PetscMalloc(sizeof(CECtx),&ce));
  p->data = (void*)ce;

  p->destroy    = &CEDestroy;
  p->function   = &CEFunction;
  p->jacobian   = &CEJacobian;
  p->solution   = &CESolution;
  p->final_time = 10;
  p->n          = 1;
  p->hasexact   = PETSC_TRUE;

  ce->lambda = 10;
  ierr       = PetscOptionsBegin(p->comm,NULL,"CE options","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsReal("-problem_ce_lambda","Parameter controlling stiffness: xdot + lambda*(x - cos(t))","",ce->lambda,&ce->lambda,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Stiff 3-variable oscillatory system from chemical reactions. problem OREGO in Hairer&Wanner
*/
static PetscErrorCode OregoFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = xdot[0] - 77.27*(x[1] + x[0]*(1. - 8.375e-6*x[0] - x[1]));
  f[1] = xdot[1] - 1/77.27*(x[2] - (1. + x[0])*x[1]);
  f[2] = xdot[2] - 0.161*(x[0] - x[2]);
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  J[0][0] = a - 77.27*((1. - 8.375e-6*x[0] - x[1]) - 8.375e-6*x[0]);
  J[0][1] = -77.27*(1. - x[0]);
  J[0][2] = 0;
  J[1][0] = 1./77.27*x[1];
  J[1][1] = a + 1./77.27*(1. + x[0]);
  J[1][2] = -1./77.27;
  J[2][0] = -0.161;
  J[2][1] = 0;
  J[2][2] = a + 0.161;
  CHKERRQ(MatSetValues(B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoSolution(PetscReal t,Vec X,void *ctx)
{
  PetscScalar    *x;

  PetscFunctionBeginUser;
  PetscCheck(t == 0,PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented");
  CHKERRQ(VecGetArray(X,&x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  CHKERRQ(VecRestoreArray(X,&x));
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoCreate(Problem p)
{
  PetscFunctionBeginUser;
  p->destroy    = 0;
  p->function   = &OregoFunction;
  p->jacobian   = &OregoJacobian;
  p->solution   = &OregoSolution;
  p->final_time = 360;
  p->n          = 3;
  PetscFunctionReturn(0);
}

/*
   User-defined monitor for comparing to exact solutions when possible
*/
typedef struct {
  MPI_Comm comm;
  Problem  problem;
  Vec      x;
} MonitorCtx;

static PetscErrorCode MonitorError(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx)
{
  MonitorCtx     *mon = (MonitorCtx*)ctx;
  PetscReal      h,nrm_x,nrm_exact,nrm_diff;

  PetscFunctionBeginUser;
  if (!mon->problem->solution) PetscFunctionReturn(0);
  CHKERRQ((*mon->problem->solution)(t,mon->x,mon->problem->data));
  CHKERRQ(VecNorm(x,NORM_2,&nrm_x));
  CHKERRQ(VecNorm(mon->x,NORM_2,&nrm_exact));
  CHKERRQ(VecAYPX(mon->x,-1,x));
  CHKERRQ(VecNorm(mon->x,NORM_2,&nrm_diff));
  CHKERRQ(TSGetTimeStep(ts,&h));
  if (step < 0) {
    CHKERRQ(PetscPrintf(mon->comm,"Interpolated final solution "));
  }
  CHKERRQ(PetscPrintf(mon->comm,"step %4D t=%12.8e h=% 8.2e  |x|=%9.2e  |x_e|=%9.2e  |x-x_e|=%9.2e\n",step,(double)t,(double)h,(double)nrm_x,(double)nrm_exact,(double)nrm_diff));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscFunctionList plist = NULL;
  char              pname[256];
  TS                ts;            /* nonlinear solver */
  Vec               x,r;           /* solution, residual vectors */
  Mat               A;             /* Jacobian matrix */
  Problem           problem;
  PetscBool         use_monitor = PETSC_FALSE;
  PetscBool         use_result = PETSC_FALSE;
  PetscInt          steps,nonlinits,linits,snesfails,rejects;
  PetscReal         ftime;
  MonitorCtx        mon;
  PetscErrorCode    ierr;
  PetscMPIInt       size;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* Register the available problems */
  CHKERRQ(PetscFunctionListAdd(&plist,"rober",&RoberCreate));
  CHKERRQ(PetscFunctionListAdd(&plist,"ce",&CECreate));
  CHKERRQ(PetscFunctionListAdd(&plist,"orego",&OregoCreate));
  CHKERRQ(PetscStrcpy(pname,"ce"));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Timestepping benchmark options","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsFList("-problem_type","Name of problem to run","",plist,pname,pname,sizeof(pname),NULL));
    use_monitor = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-monitor_error","Display errors relative to exact solutions","",use_monitor,&use_monitor,NULL));
    CHKERRQ(PetscOptionsBool("-monitor_result","Display result","",use_result,&use_result,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create the new problem */
  CHKERRQ(PetscNew(&problem));
  problem->comm = MPI_COMM_WORLD;
  {
    PetscErrorCode (*pcreate)(Problem);

    CHKERRQ(PetscFunctionListFind(plist,pname,&pcreate));
    PetscCheck(pcreate,PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"No problem '%s'",pname);
    CHKERRQ((*pcreate)(problem));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,problem->n,problem->n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&x,NULL));
  CHKERRQ(VecDuplicate(x,&r));

  mon.comm    = PETSC_COMM_WORLD;
  mon.problem = problem;
  CHKERRQ(VecDuplicate(x,&mon.x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW)); /* Rosenbrock-W */
  CHKERRQ(TSSetIFunction(ts,NULL,problem->function,problem->data));
  CHKERRQ(TSSetIJacobian(ts,A,A,problem->jacobian,problem->data));
  CHKERRQ(TSSetMaxTime(ts,problem->final_time));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetMaxStepRejections(ts,10));
  CHKERRQ(TSSetMaxSNESFailures(ts,-1)); /* unlimited */
  if (use_monitor) {
    CHKERRQ(TSMonitorSet(ts,&MonitorError,&mon,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ((*problem->solution)(0,x,problem->data));
  CHKERRQ(TSSetTimeStep(ts,.001));
  CHKERRQ(TSSetSolution(ts,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(TSGetSNESFailures(ts,&snesfails));
  CHKERRQ(TSGetStepRejections(ts,&rejects));
  CHKERRQ(TSGetSNESIterations(ts,&nonlinits));
  CHKERRQ(TSGetKSPIterations(ts,&linits));
  if (use_result) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits %D, linits %D\n",steps,rejects,snesfails,(double)ftime,nonlinits,linits));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(VecDestroy(&mon.x));
  CHKERRQ(TSDestroy(&ts));
  if (problem->destroy) {
    CHKERRQ((*problem->destroy)(problem));
  }
  CHKERRQ(PetscFree(problem));
  CHKERRQ(PetscFunctionListDestroy(&plist));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: !complex
      args:  -monitor_result -monitor_error -ts_atol 1e-2 -ts_rtol 1e-2 -ts_exact_final_time interpolate -ts_type arkimex

    test:
      suffix: 2
      requires: !single !complex
      args: -monitor_result -ts_atol 1e-2 -ts_rtol 1e-2 -ts_max_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 0 -ts_adapt_time_step_increase_delay 4

    test:
      suffix: 3
      requires: !single !complex
      args: -monitor_result -ts_atol 1e-2 -ts_rtol 1e-2 -ts_max_time 15 -ts_type arkimex -ts_arkimex_type 2e -problem_type orego -ts_arkimex_initial_guess_extrapolate 1

    test:
      suffix: 4

    test:
      suffix: 5
      args: -snes_lag_jacobian 20 -snes_lag_jacobian_persists

TEST*/
