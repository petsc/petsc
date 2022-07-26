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
  PetscCall(TS2GetSolution(ts,&U,&V));
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(V,&v));
  fvalue[0] = u[0];
  /* Event for number of bounces */
  fvalue[1] = app->maxbounces - app->bounces;
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(V,&v));
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
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  if (event_list[0] == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball hit the ground at t = %5.2f seconds\n",rank,(double)t));
    /* Set new initial conditions with .9 attenuation */
    PetscCall(TS2GetSolution(ts,&U,&V));
    PetscCall(VecGetArray(U,&u));
    PetscCall(VecGetArray(V,&v));
    u[0] = 0.0; v[0] = -app->Cr*v[0];
    PetscCall(VecRestoreArray(U,&u));
    PetscCall(VecRestoreArray(V,&v));
    app->bounces++;
  } else if (event_list[0] == 1) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Processor [%d]: Ball bounced %" PetscInt_FMT " times\n",rank,app->bounces));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Function(TS ts,PetscReal t,Vec U,Vec V,Vec A,Vec F,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscScalar       Res,*f;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(V,&v));
  PetscCall(VecGetArrayRead(A,&a));
  Res = a[0] + 9.8 + 0.5 * app->Cd * v[0]*v[0] * PetscSignReal(PetscRealPart(v[0]));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(V,&v));
  PetscCall(VecRestoreArrayRead(A,&a));

  PetscCall(VecGetArray(F,&f));
  f[0] = Res;
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode I2Jacobian(TS ts,PetscReal t,Vec U,Vec V,Vec A,PetscReal shiftV,PetscReal shiftA,Mat J,Mat P,void *ctx)
{
  AppCtx            *app = (AppCtx*)ctx;
  const PetscScalar *u,*v,*a;
  PetscInt          i;
  PetscScalar       Jac;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(V,&v));
  PetscCall(VecGetArrayRead(A,&a));
  Jac  = shiftA + shiftV * app->Cd * v[0];
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(V,&v));
  PetscCall(VecRestoreArrayRead(A,&a));

  PetscCall(MatGetOwnershipRange(P,&i,NULL));
  PetscCall(MatSetValue(P,i,i,Jac,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));
  if (J != P) {
    PetscCall(MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY));
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  app.Cd = 0.0;
  app.Cr = 0.9;
  app.bounces = 0;
  app.maxbounces = 10;
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex44 options","");
  PetscCall(PetscOptionsReal("-Cd","Drag coefficient","",app.Cd,&app.Cd,NULL));
  PetscCall(PetscOptionsReal("-Cr","Restitution coefficient","",app.Cr,&app.Cr,NULL));
  PetscCall(PetscOptionsInt("-maxbounces","Maximum number of bounces","",app.maxbounces,&app.maxbounces,NULL));
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  /*PetscCall(TSSetSaveTrajectory(ts));*/
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSALPHA2));

  PetscCall(TSSetMaxTime(ts,PETSC_INFINITY));
  PetscCall(TSSetTimeStep(ts,0.1));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));
  PetscCall(TSGetAdapt(ts,&adapt));
  PetscCall(TSAdaptSetStepLimits(adapt,0.0,0.5));

  direction[0] = -1; terminate[0] = PETSC_FALSE;
  direction[1] = -1; terminate[1] = PETSC_TRUE;
  PetscCall(TSSetEventHandler(ts,2,direction,terminate,Event,PostEvent,&app));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,1,1,PETSC_DECIDE,PETSC_DECIDE,1,NULL,0,NULL,&J));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));
  PetscCall(MatCreateVecs(J,NULL,&F));
  PetscCall(TSSetI2Function(ts,F,I2Function,&app));
  PetscCall(TSSetI2Jacobian(ts,J,J,I2Jacobian,&app));
  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&J));

  PetscCall(TSGetI2Jacobian(ts,&J,NULL,NULL,NULL));
  PetscCall(MatCreateVecs(J,&U,NULL));
  PetscCall(MatCreateVecs(J,&V,NULL));
  PetscCall(VecGetArray(U,&u));
  PetscCall(VecGetArray(V,&v));
  u[0] = 5.0*rank; v[0] = 20.0;
  PetscCall(VecRestoreArray(U,&u));
  PetscCall(VecRestoreArray(V,&v));

  PetscCall(TS2SetSolution(ts,U,V));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts,NULL));

  PetscCall(VecDestroy(&U));
  PetscCall(VecDestroy(&V));
  PetscCall(TSDestroy(&ts));

  PetscCall(PetscFinalize());
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
