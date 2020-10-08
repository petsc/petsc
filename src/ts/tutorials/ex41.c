static char help[] = "Parallel bouncing ball example to test TS event feature.\n";

/*
  The dynamics of the bouncing ball is described by the ODE
                  u1_t = u2
                  u2_t = -9.8

  Each processor is assigned one ball.

  The event function routine checks for the ball hitting the
  ground (u1 = 0). Every time the ball hits the ground, its velocity u2 is attenuated by
  a factor of 0.9 and its height set to 1.0*rank.
*/

#include <petscts.h>

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;

  PetscFunctionBegin;
  /* Event for ball height */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  fvalue[0] = u[0];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *u;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  if (nevents) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Ball hit the ground at t = %5.2f seconds -> Processor[%d]\n",(double)t,rank);CHKERRQ(ierr);
    /* Set new initial conditions with .9 attenuation */
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    u[0] =  1.0*rank;
    u[1] = -0.9*u[1];
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  }
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
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetRHSJacobian() for the meaning the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[2],rstart;
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
  rowcol[0] = rstart; rowcol[1] = rstart+1;

  J[0][0] = 0.0;      J[0][1] = 1.0;
  J[1][0] = 0.0;      J[1][1] = 0.0;
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscInt          rowcol[2],rstart;
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(B,&rstart,NULL);CHKERRQ(ierr);
  rowcol[0] = rstart; rowcol[1] = rstart+1;

  J[0][0] = a;        J[0][1] = -1.0;
  J[1][0] = 0.0;      J[1][1] = a;
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscInt       n = 2;
  PetscScalar    *u;
  PetscInt       direction=-1;
  PetscBool      terminate=PETSC_FALSE;
  PetscBool      rhs_form=PETSC_FALSE,hist=PETSC_TRUE;
  TSAdapt        adapt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

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
    ierr = TSSetIFunction(ts,NULL,IFunction,NULL);CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,NULL,NULL,IJacobian,NULL);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecCreate(PETSC_COMM_WORLD,&U);CHKERRQ(ierr);
  ierr = VecSetSizes(U,n,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetUp(U);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = 1.0*rank;
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
  ierr = TSAdaptSetType(adapt,TSADAPTBASIC);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,0.0,0.5);CHKERRQ(ierr);

  /* Set direction and terminate flag for the event */
  ierr = TSSetEventHandler(ts,1,&direction,&terminate,EventFunction,PostEventFunction,NULL);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  if (hist) { /* replay following history */
    TSTrajectory tj;
    PetscReal    tf,t0,dt;

    ierr = TSGetTime(ts,&tf);CHKERRQ(ierr);
    ierr = TSSetMaxTime(ts,tf);CHKERRQ(ierr);
    ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
    ierr = TSRestartStep(ts);CHKERRQ(ierr);
    ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
    ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
    ierr = TSSetEventHandler(ts,1,&direction,&terminate,EventFunction,PostEventFunction,NULL);CHKERRQ(ierr);
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
    u[0] = 1.0*rank;
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
      nsize: 2
      args: -ts_trajectory_type memory -snes_stol 1e-4
      filter: sort -b

   test:
      suffix: b
      nsize: 2
      args: -ts_trajectory_type memory -ts_type arkimex -snes_stol 1e-4
      filter: sort -b

   test:
      suffix: c
      nsize: 2
      args: -ts_trajectory_type memory -ts_type theta -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4
      filter: sort -b

   test:
      suffix: d
      nsize: 2
      args: -ts_trajectory_type memory -ts_type alpha -ts_adapt_type basic -ts_atol 1e-1 -snes_stol 1e-4
      filter: sort -b

   test:
      suffix: e
      nsize: 2
      args: -ts_trajectory_type memory -ts_type bdf -ts_adapt_dt_max 0.015 -ts_max_steps 3000
      filter: sort -b

   test:
      suffix: f
      nsize: 2
      args: -ts_trajectory_type memory -rhs-form -ts_type rk -ts_rk_type 3bs
      filter: sort -b

   test:
      suffix: g
      nsize: 2
      args: -ts_trajectory_type memory -rhs-form -ts_type rk -ts_rk_type 5bs
      filter: sort -b

   test:
      suffix: h
      nsize: 2
      args: -ts_trajectory_type memory -rhs-form -ts_type rk -ts_rk_type 6vr
      filter: sort -b
      output_file: output/ex41_g.out

   test:
      suffix: i
      nsize: 2
      args: -ts_trajectory_type memory -rhs-form -ts_type rk -ts_rk_type 7vr
      filter: sort -b
      output_file: output/ex41_g.out

   test:
      suffix: j
      nsize: 2
      args: -ts_trajectory_type memory -rhs-form -ts_type rk -ts_rk_type 8vr
      filter: sort -b
      output_file: output/ex41_g.out

TEST*/
