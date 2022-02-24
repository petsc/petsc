static char help[] = "Small ODE to test TS accuracy.\n";

/*
  The ODE
                  u1_t = cos(t),
                  u2_t = sin(u2)
  with analytical solution
                  u1(t) = sin(t),
                  u2(t) = 2 * atan(exp(t) * tan(0.5))
  is used to test the accuracy of TS schemes.
*/

#include <petscts.h>

/*
     Defines the ODE passed to the ODE solver in explicit form: U_t = F(U)
*/
static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec F, void *s)
{
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = PetscCosReal(t);
  f[1] = PetscSinReal(u[1]);

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the exact solution.
*/
static PetscErrorCode ExactSolution(PetscReal t, Vec U)
{
  PetscScalar       *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(U,&u));

  u[0] = PetscSinReal(t);
  u[1] = 2 * PetscAtanReal(PetscExpReal(t) * PetscTanReal(0.5));

  CHKERRQ(VecRestoreArray(U,&u));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* numerical solution will be stored here */
  Vec            Uex;           /* analytical (exact) solution will be stored here */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  PetscScalar    *u;
  PetscReal      t, final_time = 1.0, dt = 0.25;
  PetscReal      error;
  TSAdapt        adapt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSROSW));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set ODE routines
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetRHSFunction(ts,NULL,RHSFunction,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&U));
  CHKERRQ(VecSetSizes(U,n,PETSC_DETERMINE));
  CHKERRQ(VecSetUp(U));
  CHKERRQ(VecGetArray(U,&u));
  u[0] = 0.0;
  u[1] = 1.0;
  CHKERRQ(VecRestoreArray(U,&u));
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSaveTrajectory(ts));
  CHKERRQ(TSSetMaxTime(ts,final_time));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSSetTimeStep(ts,dt));
  /* The adaptive time step controller is forced to take constant time steps. */
  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetType(adapt,TSADAPTNONE));

  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver and compute error
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));
  CHKERRQ(TSGetTime(ts,&t));

  if (PetscAbsReal(t-final_time)>100*PETSC_MACHINE_EPSILON) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Note: There is a difference of %g between the prescribed final time %g and the actual final time.\n",(double)(final_time-t),(double)final_time));
  }
  CHKERRQ(VecDuplicate(U,&Uex));
  CHKERRQ(ExactSolution(t,Uex));

  CHKERRQ(VecAYPX(Uex,-1.0,U));
  CHKERRQ(VecNorm(Uex,NORM_2,&error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error at final time: %.2E\n",(double)error));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(VecDestroy(&Uex));
  CHKERRQ(TSDestroy(&ts));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      suffix: 3bs
      args: -ts_type rk -ts_rk_type 3bs
      requires: !single

    test:
      suffix: 5bs
      args: -ts_type rk -ts_rk_type 5bs
      requires: !single

    test:
      suffix: 5dp
      args: -ts_type rk -ts_rk_type 5dp
      requires: !single

    test:
      suffix: 6vr
      args: -ts_type rk -ts_rk_type 6vr
      requires: !single

    test:
      suffix: 7vr
      args: -ts_type rk -ts_rk_type 7vr
      requires: !single

    test:
      suffix: 8vr
      args: -ts_type rk -ts_rk_type 8vr
      requires: !single

TEST*/
