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
  const PetscScalar *u;

  PetscFunctionBegin;
  /* Event for ball height */
  CHKERRQ(VecGetArrayRead(U,&u));
  fvalue[0] = u[0];
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  PetscScalar    *u;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (nevents) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Ball hit the ground at t = %5.2f seconds -> Processor[%d]\n",(double)t,rank));
    /* Set new initial conditions with .9 attenuation */
    CHKERRQ(VecGetArray(U,&u));
    u[0] =  1.0*rank;
    u[1] = -0.9*u[1];
    CHKERRQ(VecRestoreArray(U,&u));
  }
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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = u[1];
  f[1] = - 9.8;

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetRHSJacobian() for the meaning the Jacobian.
*/
static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[2],rstart;
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));

  CHKERRQ(MatGetOwnershipRange(B,&rstart,NULL));
  rowcol[0] = rstart; rowcol[1] = rstart+1;

  J[0][0] = 0.0;      J[0][1] = 1.0;
  J[1][0] = 0.0;      J[1][1] = 0.0;
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = udot[0] - u[1];
  f[1] = udot[1] + 9.8;

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscInt          rowcol[2],rstart;
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));

  CHKERRQ(MatGetOwnershipRange(B,&rstart,NULL));
  rowcol[0] = rstart; rowcol[1] = rstart+1;

  J[0][0] = a;        J[0][1] = -1.0;
  J[1][0] = 0.0;      J[1][1] = a;
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));

  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
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
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSROSW));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set ODE routines
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  /* Users are advised against the following branching and code duplication.
     For problems without a mass matrix like the one at hand, the RHSFunction
     (and companion RHSJacobian) interface is enough to support both explicit
     and implicit timesteppers. This tutorial example also deals with the
     IFunction/IJacobian interface for demonstration and testing purposes. */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-rhs-form",&rhs_form,NULL));
  if (rhs_form) {
    CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));
    CHKERRQ(TSSetRHSJacobian(ts,NULL,NULL,RHSJacobian,NULL));
  } else {
    CHKERRQ(TSSetIFunction(ts,NULL,IFunction,NULL));
    CHKERRQ(TSSetIJacobian(ts,NULL,NULL,IJacobian,NULL));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(VecSetSizes(U,n,PETSC_DETERMINE));
  CHKERRQ(VecSetUp(U));
  CHKERRQ(VecGetArray(U,&u));
  u[0] = 1.0*rank;
  u[1] = 20.0;
  CHKERRQ(VecRestoreArray(U,&u));
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSSetMaxTime(ts,30.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetTimeStep(ts,0.1));
  /* The adaptive time step controller could take very large timesteps resulting in
     the same event occurring multiple times in the same interval. A maximum step size
     limit is enforced here to avoid this issue. */
  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetType(adapt,TSADAPTBASIC));
  CHKERRQ(TSAdaptSetStepLimits(adapt,0.0,0.5));

  /* Set direction and terminate flag for the event */
  CHKERRQ(TSSetEventHandler(ts,1,&direction,&terminate,EventFunction,PostEventFunction,NULL));

  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  if (hist) { /* replay following history */
    TSTrajectory tj;
    PetscReal    tf,t0,dt;

    CHKERRQ(TSGetTime(ts,&tf));
    CHKERRQ(TSSetMaxTime(ts,tf));
    CHKERRQ(TSSetStepNumber(ts,0));
    CHKERRQ(TSRestartStep(ts));
    CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
    CHKERRQ(TSSetFromOptions(ts));
    CHKERRQ(TSSetEventHandler(ts,1,&direction,&terminate,EventFunction,PostEventFunction,NULL));
    CHKERRQ(TSGetAdapt(ts,&adapt));
    CHKERRQ(TSAdaptSetType(adapt,TSADAPTHISTORY));
    CHKERRQ(TSGetTrajectory(ts,&tj));
    CHKERRQ(TSAdaptHistorySetTrajectory(adapt,tj,PETSC_FALSE));
    CHKERRQ(TSAdaptHistoryGetStep(adapt,0,&t0,&dt));
    /* this example fails with single (or smaller) precision */
#if defined(PETSC_USE_REAL_SINGLE) || defined(PETSC_USE_REAL__FP16)
    CHKERRQ(TSAdaptSetType(adapt,TSADAPTBASIC));
    CHKERRQ(TSAdaptSetStepLimits(adapt,0.0,0.5));
    CHKERRQ(TSSetFromOptions(ts));
#endif
    CHKERRQ(TSSetTime(ts,t0));
    CHKERRQ(TSSetTimeStep(ts,dt));
    CHKERRQ(TSResetTrajectory(ts));
    CHKERRQ(VecGetArray(U,&u));
    u[0] = 1.0*rank;
    u[1] = 20.0;
    CHKERRQ(VecRestoreArray(U,&u));
    CHKERRQ(TSSolve(ts,U));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
  return 0;
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
