static char help[] ="Solves the time dependent Allen-Cahn equation with IMEX methods";

/*
 * allen_cahn.c
 *
 *  Created on: Jun 8, 2012
 *      Author: Hong Zhang
 */

#include <petscts.h>

/*
 * application context
 */
typedef struct {
  PetscReal   param;        /* parameter */
  PetscReal   xleft,xright;  /* range in x-direction */
  PetscInt    mx;           /* Discretization in x-direction */
}AppCtx;

static PetscErrorCode RHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormIFunction(TS,PetscReal,Vec,Vec,Vec,void*);
static PetscErrorCode FormIJacobian(TS,PetscReal,Vec,Vec,PetscReal,Mat,Mat,void *ctx);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

int main(int argc, char **argv)
{
  TS                ts;
  Vec               x; /*solution vector*/
  Mat               A; /*Jacobian*/
  PetscInt          steps,mx;
  PetscErrorCode    ierr;
  PetscReal         ftime;
  AppCtx            user;       /* user-defined work context */

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  /* Initialize user application context */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Allen-Cahn equation","");CHKERRQ(ierr);
  user.param = 9e-4;
  user.xleft = -1.;
  user.xright = 2.;
  user.mx = 400;
  CHKERRQ(PetscOptionsReal("-eps","parameter","",user.param,&user.param,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
   * CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
   */
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create necessary matrix and vectors, solve same ODE on every process
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,user.mx,user.mx));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatCreateVecs(A,&x,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create time stepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSEIMEX));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,&user));
  CHKERRQ(TSSetIFunction(ts,NULL,FormIFunction,&user));
  CHKERRQ(TSSetIJacobian(ts,A,A,FormIJacobian,&user));
  ftime = 22;
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(ts,x,&user));
  CHKERRQ(TSSetSolution(ts,x));
  CHKERRQ(VecGetSize(x,&mx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,x));
  CHKERRQ(TSGetTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"eps %g, steps %D, ftime %g\n",(double)user.param,steps,(double)ftime));
  /*   CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));*/

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(TSDestroy(&ts));
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscScalar       *f;
  const PetscScalar *x;
  PetscInt          i,mx;
  PetscReal         hx,eps;

  PetscFunctionBegin;
  mx = user->mx;
  eps = user->param;
  hx = (user->xright-user->xleft)/(mx-1);
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = 2.*eps*(x[1]-x[0])/(hx*hx); /*boundary*/
  for (i=1;i<mx-1;i++) {
    f[i]= eps*(x[i+1]-2.*x[i]+x[i-1])/(hx*hx);
  }
  f[mx-1] = 2.*eps*(x[mx-2]- x[mx-1])/(hx*hx); /*boundary*/
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ptr)
{
  AppCtx            *user = (AppCtx*)ptr;
  PetscScalar       *f;
  const PetscScalar *x,*xdot;
  PetscInt          i,mx;

  PetscFunctionBegin;
  mx = user->mx;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArrayRead(Xdot,&xdot));
  CHKERRQ(VecGetArray(F,&f));

  for (i=0;i<mx;i++) {
    f[i]= xdot[i] - x[i]*(1.-x[i]*x[i]);
  }

  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArrayRead(Xdot,&xdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormIJacobian(TS ts,PetscReal t,Vec U, Vec Udot, PetscReal a, Mat J,Mat Jpre,void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  PetscScalar       v;
  const PetscScalar *x;
  PetscInt          i,col;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&x));
  for (i=0; i < user->mx; i++) {
    v = a - 1. + 3.*x[i]*x[i];
    col = i;
    CHKERRQ(MatSetValues(J,1,&i,1,&col,&v,INSERT_VALUES));
  }
  CHKERRQ(VecRestoreArrayRead(U,&x));

  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  if (J != Jpre) {
    CHKERRQ(MatAssemblyBegin(Jpre,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Jpre,MAT_FINAL_ASSEMBLY));
  }
  /*  MatView(J,PETSC_VIEWER_STDOUT_WORLD);*/
  PetscFunctionReturn(0);
}

static PetscErrorCode FormInitialSolution(TS ts,Vec U,void *ctx)
{
  AppCtx *user = (AppCtx*)ctx;
  PetscInt       i;
  PetscScalar    *x;
  PetscReal      hx,x_map;

  PetscFunctionBegin;
  hx = (user->xright-user->xleft)/(PetscReal)(user->mx-1);
  CHKERRQ(VecGetArray(U,&x));
  for (i=0;i<user->mx;i++) {
    x_map = user->xleft + i*hx;
    if (x_map >= 0.7065) {
      x[i] = PetscTanhReal((x_map-0.8)/(2.*PetscSqrtReal(user->param)));
    } else if (x_map >= 0.4865) {
      x[i] = PetscTanhReal((0.613-x_map)/(2.*PetscSqrtReal(user->param)));
    } else if (x_map >= 0.28) {
      x[i] = PetscTanhReal((x_map-0.36)/(2.*PetscSqrtReal(user->param)));
    } else if (x_map >= -0.7) {
      x[i] = PetscTanhReal((0.2-x_map)/(2.*PetscSqrtReal(user->param)));
    } else if (x_map >= -1) {
      x[i] = PetscTanhReal((x_map+0.9)/(2.*PetscSqrtReal(user->param)));
    }
  }
  CHKERRQ(VecRestoreArray(U,&x));
  PetscFunctionReturn(0);
}

/*TEST

     test:
       args:  -ts_rtol 1e-04 -ts_dt 0.025 -pc_type lu -ksp_error_if_not_converged TRUE  -ts_type eimex -ts_adapt_type none -ts_eimex_order_adapt -ts_eimex_max_rows 7 -ts_monitor_draw_solution
       requires: x
       timeoutfactor: 3

TEST*/
