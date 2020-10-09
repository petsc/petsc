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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* Event for ball height */
  ierr = TS2GetSolution(ts,&U,&V);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  fvalue[0] = u[0];
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->bounces;
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PostEvent(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *app = (AppCtx*)ctx;
  Vec            V;
  PetscScalar    *u,*v;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!nevents) PetscFunctionReturn(0);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (event_list[0] == 0) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball hit the ground at t = %5.2f seconds\n",rank,(double)t);CHKERRQ(ierr);
    /* Set new initial conditions with .9 attenuation */
    ierr = TS2GetSolution(ts,&U,&V);CHKERRQ(ierr);
    ierr = VecGetArray(U,&u);CHKERRQ(ierr);
    ierr = VecGetArray(V,&v);CHKERRQ(ierr);
    u[0] = 0.0; v[0] = -app->Cr*v[0];
    ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
    ierr = VecRestoreArray(V,&v);CHKERRQ(ierr);
    app->bounces++;
  } else if (event_list[0] == 1) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball bounced %D times\n",rank,app->bounces);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Function(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscScalar       Res,*f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(A,&a);CHKERRQ(ierr);
  Res = a[0] + 9.8 + 0.5 * app->Cd * v[0]*v[0] * PetscSignReal(PetscRealPart(v[0]));
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(A,&a);CHKERRQ(ierr);

  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = Res;
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Jacobian(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscInt          i;
  PetscScalar       Jac;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(A,&a);CHKERRQ(ierr);
  Jac  = shiftA + shiftV * app->Cd * v[0];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(A,&a);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(P,&i,NULL);CHKERRQ(ierr);
  ierr = MatSetValue(P,i,i,Jac,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (J != P) {
    ierr = MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  app.Cd = 0.0;
  app.Cr = 0.9;
  app.bounces = 0;
  app.maxbounces = 10;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex44 options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Cd","Drag coefficient","",app.Cd,&app.Cd,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-Cr","Restitution coefficient","",app.Cr,&app.Cr,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-maxbounces","Maximum number of bounces","",app.maxbounces,&app.maxbounces,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  /*ierr = TSSetSaveTrajectory(ts);CHKERRQ(ierr);*/
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSALPHA2);CHKERRQ(ierr);

  ierr = TSSetMaxTime(ts,PETSC_INFINITY);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.1);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,0.0,0.5);CHKERRQ(ierr);

  direction[0] = -1; terminate[0] = PETSC_FALSE;
  direction[1] = -1; terminate[1] = PETSC_TRUE;
  ierr = TSSetEventHandler(ts,2,direction,terminate,Event,PostEvent,&app);CHKERRQ(ierr);

  ierr = MatCreateAIJ(PETSC_COMM_WORLD,1,1,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&J);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);
  ierr = MatCreateVecs(J,NULL,&F);CHKERRQ(ierr);
  ierr = TSSetI2Function(ts,F,I2Function,&app);CHKERRQ(ierr);
  ierr = TSSetI2Jacobian(ts,J,J,I2Jacobian,&app);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);

  ierr = TSGetI2Jacobian(ts,&J,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(J,&U,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(J,&V,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(V,&v);CHKERRQ(ierr);
  u[0] = 5.0*rank; v[0] = 20.0;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(V,&v);CHKERRQ(ierr);

  ierr = TS2SetSolution(ts,U,V);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,NULL);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
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
