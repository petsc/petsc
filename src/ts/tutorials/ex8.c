
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
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + 0.04*x[0] - 1e4*x[1]*x[2];
  f[1] = xdot[1] - 0.04*x[0] + 1e4*x[1]*x[2] + 3e7*PetscSqr(x[1]);
  f[2] = xdot[2] - 3e7*PetscSqr(x[1]);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + 0.04;     J[0][1] = -1e4*x[2];                   J[0][2] = -1e4*x[1];
  J[1][0] = -0.04;        J[1][1] = a + 1e4*x[2] + 3e7*2*x[1];   J[1][2] = 1e4*x[1];
  J[2][0] = 0;            J[2][1] = -3e7*2*x[1];                 J[2][2] = a;
  ierr    = MatSetValues(B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode RoberSolution(PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *x;

  PetscFunctionBeginUser;
  if (t != 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented");
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1;
  x[1] = 0;
  x[2] = 0;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscFree(p->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CEFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  PetscReal         l = ((CECtx*)ctx)->lambda;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] + l*(x[0] - PetscCosReal(t));
#if 0
  ierr = PetscPrintf(PETSC_COMM_WORLD," f(t=%g,x=%g,xdot=%g) = %g\n",(double)t,(double)x[0],(double)xdot[0],(double)f[0]);CHKERRQ(ierr);
#endif
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CEJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscReal         l = ((CECtx*)ctx)->lambda;
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0};
  PetscScalar       J[1][1];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + l;
  ierr    = MatSetValues(B,1,rowcol,1,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CESolution(PetscReal t,Vec X,void *ctx)
{
  PetscReal      l = ((CECtx*)ctx)->lambda;
  PetscErrorCode ierr;
  PetscScalar    *x;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = l/(l*l+1)*(l*PetscCosReal(t)+PetscSinReal(t)) - l*l/(l*l+1)*PetscExpReal(-l*t);
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CECreate(Problem p)
{
  PetscErrorCode ierr;
  CECtx          *ce;

  PetscFunctionBeginUser;
  ierr    = PetscMalloc(sizeof(CECtx),&ce);CHKERRQ(ierr);
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
    ierr = PetscOptionsReal("-problem_ce_lambda","Parameter controlling stiffness: xdot + lambda*(x - cos(t))","",ce->lambda,&ce->lambda,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
*  Stiff 3-variable oscillatory system from chemical reactions. problem OREGO in Hairer&Wanner
*/
static PetscErrorCode OregoFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - 77.27*(x[1] + x[0]*(1. - 8.375e-6*x[0] - x[1]));
  f[1] = xdot[1] - 1/77.27*(x[2] - (1. + x[0])*x[1]);
  f[2] = xdot[2] - 0.161*(x[0] - x[2]);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1,2};
  PetscScalar       J[3][3];
  const PetscScalar *x,*xdot;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a - 77.27*((1. - 8.375e-6*x[0] - x[1]) - 8.375e-6*x[0]);
  J[0][1] = -77.27*(1. - x[0]);
  J[0][2] = 0;
  J[1][0] = 1./77.27*x[1];
  J[1][1] = a + 1./77.27*(1. + x[0]);
  J[1][2] = -1./77.27;
  J[2][0] = -0.161;
  J[2][1] = 0;
  J[2][2] = a + 0.161;
  ierr    = MatSetValues(B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode OregoSolution(PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *x;

  PetscFunctionBeginUser;
  if (t != 0) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"not implemented");
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
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
*  User-defined monitor for comparing to exact solutions when possible
*/
typedef struct {
  MPI_Comm comm;
  Problem  problem;
  Vec      x;
} MonitorCtx;

static PetscErrorCode MonitorError(TS ts,PetscInt step,PetscReal t,Vec x,void *ctx)
{
  PetscErrorCode ierr;
  MonitorCtx     *mon = (MonitorCtx*)ctx;
  PetscReal      h,nrm_x,nrm_exact,nrm_diff;

  PetscFunctionBeginUser;
  if (!mon->problem->solution) PetscFunctionReturn(0);
  ierr = (*mon->problem->solution)(t,mon->x,mon->problem->data);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&nrm_x);CHKERRQ(ierr);
  ierr = VecNorm(mon->x,NORM_2,&nrm_exact);CHKERRQ(ierr);
  ierr = VecAYPX(mon->x,-1,x);CHKERRQ(ierr);
  ierr = VecNorm(mon->x,NORM_2,&nrm_diff);CHKERRQ(ierr);
  ierr = TSGetTimeStep(ts,&h);CHKERRQ(ierr);
  if (step < 0) {
    ierr = PetscPrintf(mon->comm,"Interpolated final solution ");CHKERRQ(ierr);
  }
  ierr = PetscPrintf(mon->comm,"step %4D t=%12.8e h=% 8.2e  |x|=%9.2e  |x_e|=%9.2e  |x-x_e|=%9.2e\n",step,(double)t,(double)h,(double)nrm_x,(double)nrm_exact,(double)nrm_diff);CHKERRQ(ierr);
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
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* Register the available problems */
  ierr = PetscFunctionListAdd(&plist,"rober",&RoberCreate);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&plist,"ce",&CECreate);CHKERRQ(ierr);
  ierr = PetscFunctionListAdd(&plist,"orego",&OregoCreate);CHKERRQ(ierr);
  ierr = PetscStrcpy(pname,"ce");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Timestepping benchmark options","");CHKERRQ(ierr);
  {
    ierr        = PetscOptionsFList("-problem_type","Name of problem to run","",plist,pname,pname,sizeof(pname),NULL);CHKERRQ(ierr);
    use_monitor = PETSC_FALSE;
    ierr        = PetscOptionsBool("-monitor_error","Display errors relative to exact solutions","",use_monitor,&use_monitor,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsBool("-monitor_result","Display result","",use_result,&use_result,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create the new problem */
  ierr          = PetscNew(&problem);CHKERRQ(ierr);
  problem->comm = MPI_COMM_WORLD;
  {
    PetscErrorCode (*pcreate)(Problem);

    ierr = PetscFunctionListFind(plist,pname,&pcreate);CHKERRQ(ierr);
    if (!pcreate) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"No problem '%s'",pname);
    ierr = (*pcreate)(problem);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,problem->n,problem->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&x,NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  mon.comm    = PETSC_COMM_WORLD;
  mon.problem = problem;
  ierr        = VecDuplicate(x,&mon.x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr); /* Rosenbrock-W */
  ierr = TSSetIFunction(ts,NULL,problem->function,problem->data);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,problem->jacobian,problem->data);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,problem->final_time);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetMaxStepRejections(ts,10);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr); /* unlimited */
  if (use_monitor) {
    ierr = TSMonitorSet(ts,&MonitorError,&mon,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = (*problem->solution)(0,x,problem->data);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.001);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetSNESFailures(ts,&snesfails);CHKERRQ(ierr);
  ierr = TSGetStepRejections(ts,&rejects);CHKERRQ(ierr);
  ierr = TSGetSNESIterations(ts,&nonlinits);CHKERRQ(ierr);
  ierr = TSGetKSPIterations(ts,&linits);CHKERRQ(ierr);
  if (use_result) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"steps %D (%D rejected, %D SNES fails), ftime %g, nonlinits %D, linits %D\n",steps,rejects,snesfails,(double)ftime,nonlinits,linits);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&mon.x);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  if (problem->destroy) {
    ierr = (*problem->destroy)(problem);CHKERRQ(ierr);
  }
  ierr = PetscFree(problem);CHKERRQ(ierr);
  ierr = PetscFunctionListDestroy(&plist);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
