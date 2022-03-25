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
  PetscCall(VecGetArrayRead(U,&u));
  if (actx->mode == 1) {
    fvalue[0] = u[1]-actx->lambda1*u[0];
  } else if (actx->mode == 2) {
    fvalue[0] = u[1]-actx->lambda2*u[0];
  }
  PetscCall(VecRestoreArrayRead(U,&u));
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
  PetscCall(TSGetCostGradients(ts,&numcost,&lambda,&mu));
  PetscCall(VecGetArrayRead(U,&u));

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

  PetscCall(VecRestoreArrayRead(U,&u));

  PetscCall(VecGetArray(lambda[0],&x));
  PetscCall(VecGetArray(mu[0],&y));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  y[0]   = y[0] + A2[0]*x[0]+A2[1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  PetscCall(VecRestoreArray(mu[0],&y));
  PetscCall(VecRestoreArray(lambda[0],&x));

  PetscCall(VecGetArray(lambda[1],&x));
  PetscCall(VecGetArray(mu[1],&y));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  y[0]   = y[0] + A2[0]*x[0]+A2[1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  PetscCall(VecRestoreArray(mu[1],&y));
  PetscCall(VecRestoreArray(lambda[1],&x));
  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *actx=(AppCtx*)ctx;

  PetscFunctionBegin;
  if (!forwardsolve) {
    PetscCall(ShiftGradients(ts,U,actx));
  }
  if (actx->mode == 1) {
    actx->mode = 2;
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Change from mode 1 to 2 at t = %f \n",(double)t));
  } else if (actx->mode == 2) {
    actx->mode = 1;
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Change from mode 2 to 1 at t = %f \n",(double)t));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  if (actx->mode == 1) {
    f[0] = udot[0]-u[0]+100*u[1];
    f[1] = udot[1]-10*u[0]-u[1];
  } else if (actx->mode == 2) {
    f[0] = udot[0]-u[0]-10*u[1];
    f[1] = udot[1]+100*u[0]-u[1];
  }

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
  AppCtx            *actx=(AppCtx*)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));

  if (actx->mode == 1) {
    J[0][0] = a-1;                       J[0][1] = 100;
    J[1][0] = -10;                       J[1][1] = a-1;
  } else if (actx->mode == 2) {
    J[0][0] = a-1;                       J[0][1] = -10;
    J[1][0] = 100;                       J[1][1] = a-1;
  }
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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");
  app.mode = 1;
  app.lambda1 = 2.75;
  app.lambda2 = 0.36;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex1 options","");PetscCall(ierr);
  {
    PetscCall(PetscOptionsReal("-lambda1","","",app.lambda1,&app.lambda1,NULL));
    PetscCall(PetscOptionsReal("-lambda2","","",app.lambda2,&app.lambda2,NULL));
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&U,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&Ap));
  PetscCall(MatSetSizes(Ap,n,1,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(Ap,MATDENSE));
  PetscCall(MatSetFromOptions(Ap));
  PetscCall(MatSetUp(Ap));
  PetscCall(MatZeroEntries(Ap)); /* initialize to zeros */

  PetscCall(VecGetArray(U,&u));
  u[0] = 0;
  u[1] = 1;
  PetscCall(VecRestoreArray(U,&u));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSCN));
  PetscCall(TSSetIFunction(ts,NULL,(TSIFunction)IFunction,&app));
  PetscCall(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts,0.125));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts,1./256.));
  PetscCall(TSSetFromOptions(ts));

  /* Set directions and terminate flags for the two events */
  direction[0] = 0;
  terminate[0] = PETSC_FALSE;
  PetscCall(TSSetEventHandler(ts,1,direction,terminate,EventFunction,PostEventFunction,(void*)&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  PetscCall(VecGetArray(U,&u));
  tmp[0] = u[0];
  tmp[1] = u[1];

  u[0] = 0+delta;
  u[1] = 1;
  PetscCall(VecRestoreArray(U,&u));

  PetscCall(FWDRun(ts,U,(void*)&app));

  PetscCall(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"d x1(tf) /d x1(t0) = %f d x2(tf) / d x1(t0) = %f \n",sensi[0],sensi[1]));
  u[0] = 0;
  u[1] = 1+delta;
  PetscCall(VecRestoreArray(U,&u));

  PetscCall(FWDRun(ts,U,(void*)&app));

  PetscCall(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"d x1(tf) /d x2(t0) = %f d x2(tf) / d x2(t0) = %f \n",sensi[0],sensi[1]));
  u[0] = 0;
  u[1] = 1;
  app.lambda1 = app.lambda1+delta;
  PetscCall(VecRestoreArray(U,&u));

  PetscCall(FWDRun(ts,U,(void*)&app));

  PetscCall(VecGetArray(U,&u));
  sensi[0] = (u[0]-tmp[0])/delta;
  sensi[1] = (u[1]-tmp[1])/delta;
  PetscCall(PetscPrintf(PETSC_COMM_SELF,"Final gradients: d x1(tf) /d p = %f d x2(tf) / d p = %f \n",sensi[0],sensi[1]));
  PetscCall(VecRestoreArray(U,&u));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(MatDestroy(&Ap));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FWDRun(TS ts, Vec U0, void *ctx0)
{
  Vec            U;             /* solution will be stored here */
  AppCtx         *ctx=(AppCtx*)ctx0;

  PetscFunctionBeginUser;
  PetscCall(TSGetSolution(ts,&U));
  PetscCall(VecCopy(U0,U));

  ctx->mode = 1;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetTime(ts, 0.0));

  PetscCall(TSSolve(ts,U));

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
