static char help[] = "Parallel bouncing ball example formulated as a second-order system to test TS event feature.\n";

/*
  The dynamics of the bouncing ball with drag coefficient Cd is described by the ODE

      u_tt = -9.8 - 1/2 Cd (u_t)^2 sign(u_t)

  There are two events set in this example. The first one checks for the ball hitting the
  ground (u = 0). Every time the ball hits the ground, its velocity u_t is attenuated by
  a restitution coefficient Cr. The second event sets a limit on the number of ball bounces.
*/

#include <petscts.h>

typedef struct {
  PetscReal Cd;      /* drag coefficient */
  PetscReal Cr;      /* restitution coefficient */
  PetscInt  bounces;
  PetscInt  maxbounces;
} AppCtx;

static PetscErrorCode Event(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  Vec               V;
  const PetscScalar *u,*v;

  PetscFunctionBegin;
  /* Event for ball height */
  CHKERRQ(TS2GetSolution(ts,&U,&V));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(V,&v));
  fvalue[0] = u[0];
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->bounces;
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(V,&v));
  PetscFunctionReturn(0);
}

static PetscErrorCode PostEvent(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *app = (AppCtx*)ctx;
  Vec            V;
  PetscScalar    *u,*v;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!nevents) PetscFunctionReturn(0);
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (event_list[0] == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball hit the ground at t = %5.2f seconds\n",rank,(double)t));
    /* Set new initial conditions with .9 attenuation */
    CHKERRQ(TS2GetSolution(ts,&U,&V));
    CHKERRQ(VecGetArray(U,&u));
    CHKERRQ(VecGetArray(V,&v));
    u[0] = 0.0; v[0] = -app->Cr*v[0];
    CHKERRQ(VecRestoreArray(U,&u));
    CHKERRQ(VecRestoreArray(V,&v));
    app->bounces++;
  } else if (event_list[0] == 1) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball bounced %D times\n",rank,app->bounces));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Function(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscScalar       Res,*f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(V,&v));
  CHKERRQ(VecGetArrayRead(A,&a));
  Res = a[0] + 9.8 + 0.5 * app->Cd * v[0]*v[0] * PetscSignReal(PetscRealPart(v[0]));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(V,&v));
  CHKERRQ(VecRestoreArrayRead(A,&a));

  CHKERRQ(VecGetArray(F,&f));
  f[0] = Res;
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Jacobian(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscInt          i;
  PetscScalar       Jac;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(V,&v));
  CHKERRQ(VecGetArrayRead(A,&a));
  Jac  = shiftA + shiftV * app->Cd * v[0];
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(V,&v));
  CHKERRQ(VecRestoreArrayRead(A,&a));

  CHKERRQ(MatGetOwnershipRange(P,&i,NULL));
  CHKERRQ(MatSetValue(P,i,i,Jac,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  if (J != P) {
    CHKERRQ(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U,V;           /* solution will be stored here */
  Vec            F;             /* residual vector */
  Mat            J;             /* Jacobian matrix */
  PetscMPIInt    rank;
  PetscScalar    *u,*v;
  AppCtx         app;
  PetscInt       direction[2];
  PetscBool      terminate[2];
  TSAdapt        adapt;
  PetscErrorCode ierr;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  app.Cd = 0.0;
  app.Cr = 0.9;
  app.bounces = 0;
  app.maxbounces = 10;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex44 options","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsReal("-Cd","Drag coefficient","",app.Cd,&app.Cd,NULL));
  CHKERRQ(PetscOptionsReal("-Cr","Restitution coefficient","",app.Cr,&app.Cr,NULL));
  CHKERRQ(PetscOptionsInt("-maxbounces","Maximum number of bounces","",app.maxbounces,&app.maxbounces,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  /*CHKERRQ(TSSetSaveTrajectory(ts));*/
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSALPHA2));

  CHKERRQ(TSSetMaxTime(ts,PETSC_INFINITY));
  CHKERRQ(TSSetTimeStep(ts,0.1));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetStepLimits(adapt,0.0,0.5));

  direction[0] = -1; terminate[0] = PETSC_FALSE;
  direction[1] = -1; terminate[1] = PETSC_TRUE;
  CHKERRQ(TSSetEventHandler(ts,2,direction,terminate,Event,PostEvent,&app));

  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,1,1,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&J));
  CHKERRQ(MatSetFromOptions(J));
  CHKERRQ(MatSetUp(J));
  CHKERRQ(MatCreateVecs(J,NULL,&F));
  CHKERRQ(TSSetI2Function(ts,F,I2Function,&app));
  CHKERRQ(TSSetI2Jacobian(ts,J,J,I2Jacobian,&app));
  CHKERRQ(VecDestroy(&F));
  CHKERRQ(MatDestroy(&J));

  CHKERRQ(TSGetI2Jacobian(ts,&J,NULL,NULL,NULL));
  CHKERRQ(MatCreateVecs(J,&U,NULL));
  CHKERRQ(MatCreateVecs(J,&V,NULL));
  CHKERRQ(VecGetArray(U,&u));
  CHKERRQ(VecGetArray(V,&v));
  u[0] = 5.0*rank; v[0] = 20.0;
  CHKERRQ(VecRestoreArray(U,&u));
  CHKERRQ(VecRestoreArray(V,&v));

  CHKERRQ(TS2SetSolution(ts,U,V));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSolve(ts,NULL));

  CHKERRQ(VecDestroy(&U));
  CHKERRQ(VecDestroy(&V));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

    test:
      suffix: a
      args: -ts_alpha_radius {{1.0 0.5}}
      output_file: output/ex44.out

    test:
      suffix: b
      args: -ts_rtol 0 -ts_atol 1e-1 -ts_adapt_type basic
      output_file: output/ex44.out

    test:
      suffix: 2
      nsize: 2
      args: -ts_rtol 0 -ts_atol 1e-1 -ts_adapt_type basic
      output_file: output/ex44_2.out
      filter: sort -b
      filter_output: sort -b

    test:
      requires: !single
      args: -ts_dt 0.25 -ts_adapt_type basic -ts_adapt_wnormtype INFINITY -ts_adapt_monitor
      args: -ts_max_steps 1 -ts_max_reject {{0 1 2}separate_output} -ts_error_if_step_fails false

TEST*/
