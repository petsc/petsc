static char help[] = "Serial bouncing ball example to test TS event feature.\n";

/*
  The dynamics of the bouncing ball is described by the ODE
                  u1_t = u2
                  u2_t = -9.8

  There are two events set in this example. The first one checks for the ball hitting the
  ground (u1 = 0). Every time the ball hits the ground, its velocity u2 is attenuated by
  a factor of 0.9. The second event sets a limit on the number of ball bounces.
*/

#include <petscts.h>

typedef struct {
  PetscInt maxbounces;
  PetscInt nbounces;
} AppCtx;

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *app=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  const PetscScalar *u;

  PetscFunctionBegin;
  /* Event for ball height */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  fvalue[0] = u[0];
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->nbounces;
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscErrorCode ierr;
  PetscScalar    *u;

  PetscFunctionBegin;
  if (event_list[0] == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ball hit the ground at t = %5.2f seconds\n",(double)t);CHKERRQ(ierr);
    /* Set new initial conditions with .9 attenuation */
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    u[0] =  0.0;
    u[1] = -0.9*u[1];
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  } else if (event_list[0] == 1) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ball bounced %D times\n",app->nbounces);CHKERRQ(ierr);
  }
  app->nbounces++;
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver in explicit form: U_t = F(U)
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = u[1];
  f[1] = - 9.8;

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetRHSJacobian() for the meaning of the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  J[0][0] = 0.0;     J[0][1] = 1.0;
  J[1][0] = 0.0;     J[1][1] = 0.0;
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver in implicit form: F(U_t,U) = 0
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = udot[0] - u[1];
  f[1] = udot[1] + 9.8;

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  J[0][0] = a;      J[0][1] = -1.0;
  J[1][0] = 0.0;    J[1][1] = a;
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  PetscScalar    *u;
  AppCtx         app;
  PetscInt       direction[2];
  PetscBool      terminate[2];
  PetscBool      rhs_form=PETSC_FALSE,hist=PETSC_TRUE;
  TSAdapt        adapt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  app.nbounces = 0;
  app.maxbounces = 10;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex40 options","");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-maxbounces","","",app.maxbounces,&app.maxbounces,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-test_adapthistory","","",hist,&hist,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set ODE routines
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  /* Users are advised against the following branching and code duplication.
     For problems without a mass matrix like the one at hand, the RHSFunction
     (and companion RHSJacobian) interface is enough to support both explicit
     and implicit timesteppers. This tutorial example also deals with the
     IFunction/IJacobian interface for demonstration and testing purposes. */
  ierr = PetscOptionsGetBool(NULL,NULL,"-rhs-form",&rhs_form,NULL);CHKERRQ(ierr);
  if (rhs_form) {
    ierr = TSSetRHSFunction(ts,NULL,RHSFunction,NULL);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,NULL);CHKERRQ(ierr);
  } else {
    Mat A; /* Jacobian matrix */
    ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
    ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
    ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = TSSetIFunction(ts,NULL,IFunction,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,A,A,IJacobian,NULL);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,n,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(U);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = 0.0;
  u[1] = 20.0;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(ts,30.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  /* The adapative time step controller could take very large timesteps resulting in
     the same event occuring multiple times in the same interval. A maximum step size
     limit is enforced here to avoid this issue. */
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,0.0,0.5);CHKERRQ(ierr);

  /* Set directions and terminate flags for the two events */
  direction[0] = -1;            direction[1] = -1;
  terminate[0] = PETSC_FALSE;   terminate[1] = PETSC_TRUE;
  ierr = TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)&app);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  if (hist) { /* replay following history */
    TSTrajectory tj;
    PetscReal    tf,t0,dt;

    app.nbounces = 0;
    ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
    ierr = TSRestartStep(ts);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
    ierr = TSAdaptSetType(adapt,TSADAPTHISTORY);CHKERRQ(ierr);
    ierr = TSGetTrajectory(ts,&tj);CHKERRQ(ierr);
    ierr = TSAdaptHistorySetTrajectory(adapt,tj,PETSC_FALSE);CHKERRQ(ierr);
    ierr = TSAdaptHistoryGetStep(adapt,0,&t0,&dt);CHKERRQ(ierr);
    /* this example fails with single (or smaller) precision */
#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL__FP16)
    ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);
    ierr = TSAdaptSetStepLimits(adapt,0.0,0.5);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
#endif
    ierr = TSSetTime(ts,t0);CHKERRQ(ierr);
    ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
    ierr = TSResetTrajectory(ts);CHKERRQ(ierr);
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    u[0] = 0.0;
    u[1] = 20.0;
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
    ierr = TSSolve(ts,U);CHKERRQ(ierr);
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      suffix: a
      args: -snes_stol 1e-4 -ts_trajectory_dirname ex40_a_dir
      output_file: output/ex40.out

    test:
      suffix: b
      args: -ts_type arkimex -snes_stol 1e-4 -ts_trajectory_dirname ex40_b_dir
      output_file: output/ex40.out

    test:
      suffix: c
      args: -ts_type theta -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4 -ts_trajectory_dirname ex40_c_dir
      output_file: output/ex40.out

    test:
      suffix: d
      args: -ts_type alpha -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4 -ts_trajectory_dirname ex40_d_dir
      output_file: output/ex40.out

    test:
      suffix: e
      args:  -ts_type bdf -ts_adapt_dt_max 0.025 -ts_max_steps 1500 -ts_trajectory_dirname ex40_e_dir
      output_file: output/ex40.out

    test:
      suffix: f
      args: -rhs-form -ts_type rk -ts_rk_type 3bs -ts_trajectory_dirname ex40_f_dir
      output_file: output/ex40.out

    test:
      suffix: g
      args: -rhs-form -ts_type rk -ts_rk_type 5bs -ts_trajectory_dirname ex40_g_dir
      output_file: output/ex40.out

    test:
      suffix: h
      args: -rhs-form -ts_type rk -ts_rk_type 6vr -ts_trajectory_dirname ex40_h_dir
      output_file: output/ex40.out

    test:
      suffix: i
      args: -rhs-form -ts_type rk -ts_rk_type 7vr -ts_trajectory_dirname ex40_i_dir
      output_file: output/ex40.out

    test:
      suffix: j
      args: -rhs-form -ts_type rk -ts_rk_type 8vr -ts_trajectory_dirname ex40_j_dir
      output_file: output/ex40.out

TEST*/
