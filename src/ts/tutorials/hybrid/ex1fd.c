static char help[] = "An example of hybrid system using TS event.\n";

/*
  The dynamics is described by the ODE
                  u_t = A_i u

  where A_1 = [ 1  -100
                10  1  ],
        A_2 = [ 1    10
               -100  1 ].
  The index i changes from 1 to 2 when u[1]=2.75u[0] and from 2 to 1 when u[1]=0.36u[0].
  Initially u=[0 1]^T and i=1.

  Reference:
  I. A. Hiskens, M.A. Pai, Trajectory Sensitivity Analysis of Hybrid Systems, IEEE Transactions on Circuits and Systems, Vol 47, No 2, February 2000
*/

#include <petscts.h>

typedef struct {
  PetscReal lambda1;
  PetscReal lambda2;
  PetscInt  mode;  /* mode flag*/
} AppCtx;

PetscErrorCode FWDRun(TS, Vec, void *);

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  if (actx->mode == 1) {
    fvalue[0] = u[1]-actx->lambda1*u[0];
  } else if (actx->mode == 2) {
    fvalue[0] = u[1]-actx->lambda2*u[0];
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode ShiftGradients(TS ts,Vec U,AppCtx *actx)
{
  Vec               *lambda,*mu;
  PetscScalar       *x,*y;
  const PetscScalar *u;
  PetscScalar       tmp[2],A1[2][2],A2[2],denorm1,denorm2;
  PetscInt          numcost;

  PetscFunctionBegin;
  CHKERRQ(TSGetCostGradients(ts,&numcost,&lambda,&mu));
  CHKERRQ(VecGetArrayRead(U,&u));

  if (actx->mode==2) {
    denorm1 = -actx->lambda1*(u[0]-100.*u[1])+1.*(10.*u[0]+u[1]);
    denorm2 = -actx->lambda1*(u[0]+10.*u[1])+1.*(-100.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(-actx->lambda1)/denorm1+1.;
    A1[0][1] = -110.*u[0]*(-actx->lambda1)/denorm1;
    A1[1][0] = 110.*u[1]*1./denorm1;
    A1[1][1] = -110.*u[0]*1./denorm1+1.;

    A2[0] = 110.*u[1]*(-u[0])/denorm2;
    A2[1] = -110.*u[0]*(-u[0])/denorm2;
  } else {
    denorm2 = -actx->lambda2*(u[0]+10.*u[1])+1.*(-100.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(-actx->lambda1)/denorm2+1;
    A1[0][1] = -110.*u[0]*(-actx->lambda1)/denorm2;
    A1[1][0] = 110.*u[1]*1./denorm2;
    A1[1][1] = -110.*u[0]*1./denorm2+1.;

    A2[0] = 0;
    A2[1] = 0;
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));

  CHKERRQ(VecGetArray(lambda[0],&x));
  CHKERRQ(VecGetArray(mu[0],&y));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  y[0]   = y[0] + A2[0]*x[0]+A2[1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  CHKERRQ(VecRestoreArray(mu[0],&y));
  CHKERRQ(VecRestoreArray(lambda[0],&x));

  CHKERRQ(VecGetArray(lambda[1],&x));
  CHKERRQ(VecGetArray(mu[1],&y));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  y[0]   = y[0] + A2[0]*x[0]+A2[1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  CHKERRQ(VecRestoreArray(mu[1],&y));
  CHKERRQ(VecRestoreArray(lambda[1],&x));
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *actx=(AppCtx*)ctx;

  PetscFunctionBegin;
  if (!forwardsolve) {
    CHKERRQ(ShiftGradients(ts,U,actx));
  }
  if (actx->mode == 1) {
    actx->mode = 2;
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Change from mode 1 to 2 at t = %f \n",(double)t));
  } else if (actx->mode == 2) {
    actx->mode = 1;
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Change from mode 2 to 1 at t = %f \n",(double)t));
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  if (actx->mode == 1) {
    f[0] = udot[0]-u[0]+100*u[1];
    f[1] = udot[1]-10*u[0]-u[1];
  } else if (actx->mode == 2) {
    f[0] = udot[0]-u[0]-10*u[1];
    f[1] = udot[1]+100*u[0]-u[1];
  }

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
  AppCtx            *actx=(AppCtx*)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));

  if (actx->mode == 1) {
    J[0][0] = a-1;                       J[0][1] = 100;
    J[1][0] = -10;                       J[1][1] = a-1;
  } else if (actx->mode == 2) {
    J[0][0] = a-1;                       J[0][1] = -10;
    J[1][0] = 100;                       J[1][1] = a-1;
  }
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  Mat            Ap;            /* dfdp */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  PetscScalar    *u;
  AppCtx         app;
  PetscInt       direction[1];
  PetscBool      terminate[1];
  PetscReal      delta,tmp[2],sensi[2];

  delta = 1e-8;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");
  app.mode = 1;
  app.lambda1 = 2.75;
  app.lambda2 = 0.36;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex1 options","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsReal("-lambda1","","",app.lambda1,&app.lambda1,NULL));
    CHKERRQ(PetscOptionsReal("-lambda2","","",app.lambda2,&app.lambda2,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Ap));
  CHKERRQ(MatSetSizes(Ap,n,1,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(Ap,MATDENSE));
  CHKERRQ(MatSetFromOptions(Ap));
  CHKERRQ(MatSetUp(Ap));
  CHKERRQ(MatZeroEntries(Ap)); /* initialize to zeros */

  CHKERRQ(VecGetArray(U,&u));
  u[0] = 0;
  u[1] = 1;
  CHKERRQ(VecRestoreArray(U,&u));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSCN));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction)IFunction,&app));
  CHKERRQ(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,0.125));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ts,1./256.));
  CHKERRQ(TSSetFromOptions(ts));

  /* Set directions and terminate flags for the two events */
  direction[0] = 0;
  terminate[0] = PETSC_FALSE;
  CHKERRQ(TSSetEventHandler(ts,1,direction,terminate,EventFunction,PostEventFunction,(void*)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(VecGetArray(U,&u));
  tmp[0] = u[0];
  tmp[1] = u[1];

  u[0] = 0+delta;
  u[1] = 1;
  CHKERRQ(VecRestoreArray(U,&u));

  CHKERRQ(FWDRun(ts,U,(void*)&app));

  CHKERRQ(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"d x1(tf) /d x1(t0) = %f d x2(tf) / d x1(t0) = %f \n",sensi[0],sensi[1]));
  u[0] = 0;
  u[1] = 1+delta;
  CHKERRQ(VecRestoreArray(U,&u));

  CHKERRQ(FWDRun(ts,U,(void*)&app));

  CHKERRQ(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"d x1(tf) /d x2(t0) = %f d x2(tf) / d x2(t0) = %f \n",sensi[0],sensi[1]));
  u[0] = 0;
  u[1] = 1;
  app.lambda1 = app.lambda1+delta;
  CHKERRQ(VecRestoreArray(U,&u));

  CHKERRQ(FWDRun(ts,U,(void*)&app));

  CHKERRQ(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Final gradients: d x1(tf) /d p = %f d x2(tf) / d p = %f \n",sensi[0],sensi[1]));
  CHKERRQ(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(MatDestroy(&Ap));
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FWDRun(TS ts, Vec U0, void *ctx0)
{
  Vec            U;             /* solution will be stored here */
  AppCtx         *ctx=(AppCtx*)ctx0;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetSolution(ts,&U));
  CHKERRQ(VecCopy(U0,U));

  ctx->mode = 1;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetTime(ts, 0.0));

  CHKERRQ(TSSolve(ts,U));

  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !defined(PETSC_USE_CXXCOMPLEX)

   test:
      args: -ts_event_tol 1e-9
      timeoutfactor: 18
      requires: !single

TEST*/
