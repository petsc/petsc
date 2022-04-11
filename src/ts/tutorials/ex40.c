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
  const PetscScalar *u;

  PetscFunctionBegin;
  /* Event for ball height */
  PetscCall(VecGetArrayRead(U,&u));
  fvalue[0] = u[0];
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->nbounces;
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *app=(AppCtx*)ctx;
  PetscScalar    *u;

  PetscFunctionBegin;
  if (event_list[0] == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Ball hit the ground at t = %5.2f seconds\n",(double)t));
    /* Set new initial conditions with .9 attenuation */
    PetscCall(VecGetArray(U,&u));
    u[0] =  0.0;
    u[1] = -0.9*u[1];
    PetscCall(VecRestoreArray(U,&u));
  } else if (event_list[0] == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Ball bounced %D times\n",app->nbounces));
  }
  app->nbounces++;
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver in explicit form: U_t = F(U)
*/
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));

  f[0] = u[1];
  f[1] = - 9.8;

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetRHSJacobian() for the meaning of the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));

  J[0][0] = 0.0;     J[0][1] = 1.0;
  J[1][0] = 0.0;     J[1][1] = 0.0;
  PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));

  PetscCall(VecRestoreArrayRead(U,&u));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver in implicit form: F(U_t,U) = 0
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  f[0] = udot[0] - u[1];
  f[1] = udot[1] + 9.8;

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));

  J[0][0] = a;      J[0][1] = -1.0;
  J[1][0] = 0.0;    J[1][1] = a;
  PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  app.nbounces = 0;
  app.maxbounces = 10;
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex40 options","");
  PetscCall(PetscOptionsInt("-maxbounces","","",app.maxbounces,&app.maxbounces,NULL));
  PetscCall(PetscOptionsBool("-test_adapthistory","","",hist,&hist,NULL));
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSROSW));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set ODE routines
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  /* Users are advised against the following branching and code duplication.
     For problems without a mass matrix like the one at hand, the RHSFunction
     (and companion RHSJacobian) interface is enough to support both explicit
     and implicit timesteppers. This tutorial example also deals with the
     IFunction/IJacobian interface for demonstration and testing purposes. */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-rhs-form",&rhs_form,NULL));
  if (rhs_form) {
    PetscCall(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));
    PetscCall(TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,NULL));
  } else {
    Mat A; /* Jacobian matrix */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
    PetscCall(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
    PetscCall(MatSetType(A,MATDENSE));
    PetscCall(MatSetFromOptions(A));
    PetscCall(MatSetUp(A));
    PetscCall(TSSetIFunction(ts,NULL,IFunction,NULL));
    PetscCall(TSSetIJacobian(ts,A,A,IJacobian,NULL));
    PetscCall(MatDestroy(&A));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&U));
  PetscCall(VecSetSizes(U,n,PETSC_DETERMINE));
  PetscCall(VecSetUp(U));
  PetscCall(VecGetArray(U,&u));
  u[0] = 0.0;
  u[1] = 20.0;
  PetscCall(VecRestoreArray(U,&u));
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (hist) PetscCall(TSSetSaveTrajectory(ts));
  PetscCall(TSSetMaxTime(ts,30.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSSetTimeStep(ts,0.1));
  /* The adaptive time step controller could take very large timesteps resulting in
     the same event occurring multiple times in the same interval. A maximum step size
     limit is enforced here to avoid this issue. */
  PetscCall(TSGetAdapt(ts,&adapt));
  PetscCall(TSAdaptSetStepLimits(adapt,0.0,0.5));

  /* Set directions and terminate flags for the two events */
  direction[0] = -1;            direction[1] = -1;
  terminate[0] = PETSC_FALSE;   terminate[1] = PETSC_TRUE;
  PetscCall(TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)&app));

  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  if (hist) { /* replay following history */
    TSTrajectory tj;
    PetscReal    tf,t0,dt;

    app.nbounces = 0;
    PetscCall(TSGetTime(ts,&tf));
    PetscCall(TSSetMaxTime(ts,tf));
    PetscCall(TSSetStepNumber(ts,0));
    PetscCall(TSRestartStep(ts));
    PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
    PetscCall(TSSetFromOptions(ts));
    PetscCall(TSGetAdapt(ts,&adapt));
    PetscCall(TSAdaptSetType(adapt,TSADAPTHISTORY));
    PetscCall(TSGetTrajectory(ts,&tj));
    PetscCall(TSAdaptHistorySetTrajectory(adapt,tj,PETSC_FALSE));
    PetscCall(TSAdaptHistoryGetStep(adapt,0,&t0,&dt));
    /* this example fails with single (or smaller) precision */
#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL__FP16)
    PetscCall(TSAdaptSetType(adapt,TSADAPTBASIC));
    PetscCall(TSAdaptSetStepLimits(adapt,0.0,0.5));
    PetscCall(TSSetFromOptions(ts));
#endif
    PetscCall(TSSetTime(ts,t0));
    PetscCall(TSSetTimeStep(ts,dt));
    PetscCall(TSResetTrajectory(ts));
    PetscCall(VecGetArray(U,&u));
    u[0] = 0.0;
    u[1] = 20.0;
    PetscCall(VecRestoreArray(U,&u));
    PetscCall(TSSolve(ts,U));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
  return 0;
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
      args: -ts_type bdf -ts_adapt_dt_max 0.025 -ts_max_steps 1500 -ts_trajectory_dirname ex40_e_dir
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

    test:
      suffix: k
      args: -ts_type theta -ts_adapt_type dsp -ts_trajectory_dirname ex40_k_dir
      output_file: output/ex40.out

    test:
      suffix: l
      args: -rhs-form -ts_type rk -ts_rk_type 2a -ts_trajectory_dirname ex40_l_dir
      args: -ts_adapt_type dsp -ts_adapt_always_accept {{false true}} -ts_adapt_dt_min 0.01
      output_file: output/ex40.out

    test:
      suffix: m
      args: -ts_type alpha -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4 -test_adapthistory false
      args: -ts_max_time 10 -ts_exact_final_time {{STEPOVER MATCHSTEP INTERPOLATE}}

    test:
      requires: !single
      suffix: n
      args: -test_adapthistory false
      args: -ts_type alpha -ts_alpha_radius 1.0 -ts_view
      args: -ts_dt 0.25 -ts_adapt_type basic -ts_adapt_wnormtype INFINITY -ts_adapt_monitor
      args: -ts_max_steps 1 -ts_max_reject {{0 1 2}separate_output} -ts_error_if_step_fails false

    test:
      requires: !single
      suffix: o
      args: -rhs-form -ts_type rk -ts_rk_type 2b -ts_trajectory_dirname ex40_o_dir
      output_file: output/ex40.out
TEST*/
