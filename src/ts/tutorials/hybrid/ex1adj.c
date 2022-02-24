static char help[] = "Adjoint sensitivity of a hybrid system with state-dependent switchings.\n";

/*
  The dynamics is described by the ODE
                  u_t = A_i u

  where A_1 = [ 1  -100
                10  1  ],
        A_2 = [ 1    10
               -100  1 ].
  The index i changes from 1 to 2 when u[1]=2.75u[0] and from 2 to 1 when u[1]=0.36u[0].
  Initially u=[0 1]^T and i=1.

  References:
+ * - H. Zhang, S. Abhyankar, E. Constantinescu, M. Mihai, Discrete Adjoint Sensitivity Analysis of Hybrid Dynamical Systems With Switching, IEEE Transactions on Circuits and Systems I: Regular Papers, 64(5), May 2017
- * - I. A. Hiskens, M.A. Pai, Trajectory Sensitivity Analysis of Hybrid Systems, IEEE Transactions on Circuits and Systems, Vol 47, No 2, February 2000
*/

#include <petscts.h>

typedef struct {
  PetscScalar lambda1;
  PetscScalar lambda2;
  PetscInt    mode;  /* mode flag*/
} AppCtx;

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  if (actx->mode == 1) {
    fvalue[0] = u[1]-actx->lambda1*u[0];
  }else if (actx->mode == 2) {
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
  PetscScalar       tmp[2],A1[2][2],A2[2],denorm;
  PetscInt          numcost;

  PetscFunctionBegin;
  CHKERRQ(TSGetCostGradients(ts,&numcost,&lambda,&mu));
  CHKERRQ(VecGetArrayRead(U,&u));

  if (actx->mode==2) {
    denorm = -actx->lambda1*(u[0]-100.*u[1])+1.*(10.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(-actx->lambda1)/denorm+1.;
    A1[0][1] = -110.*u[0]*(-actx->lambda1)/denorm;
    A1[1][0] = 110.*u[1]*1./denorm;
    A1[1][1] = -110.*u[0]*1./denorm+1.;

    A2[0] = 110.*u[1]*(-u[0])/denorm;
    A2[1] = -110.*u[0]*(-u[0])/denorm;
  } else {
    denorm = -actx->lambda2*(u[0]+10.*u[1])+1.*(-100.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(actx->lambda2)/denorm+1;
    A1[0][1] = -110.*u[0]*(actx->lambda2)/denorm;
    A1[1][0] = -110.*u[1]*1./denorm;
    A1[1][1] = 110.*u[0]*1./denorm+1.;

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
  /* CHKERRQ(VecView(U,PETSC_VIEWER_STDOUT_WORLD)); */
  if (!forwardsolve) {
    CHKERRQ(ShiftGradients(ts,U,actx));
  }
  if (actx->mode == 1) {
    actx->mode = 2;
    /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Change from mode 1 to 2 at t = %f \n",(double)t)); */
  } else if (actx->mode == 2) {
    actx->mode = 1;
    /* CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Change from mode 2 to 1 at t = %f \n",(double)t)); */
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

/* Matrix JacobianP is constant so that it only needs to be evaluated once */
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat A, void *ctx)
{
  PetscFunctionBeginUser;
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
  PetscScalar    *u,*v;
  AppCtx         app;
  PetscInt       direction[1];
  PetscBool      terminate[1];
  Vec            lambda[2],mu[2];
  PetscReal      tend;

  FILE           *f;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");
  app.mode = 1;
  app.lambda1 = 2.75;
  app.lambda2 = 0.36;
  tend = 0.125;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex1adj options","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsReal("-lambda1","","",app.lambda1,&app.lambda1,NULL));
    CHKERRQ(PetscOptionsReal("-lambda2","","",app.lambda2,&app.lambda2,NULL));
    CHKERRQ(PetscOptionsReal("-tend","","",tend,&tend,NULL));
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
  CHKERRQ(TSSetRHSJacobianP(ts,Ap,RHSJacobianP,&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSaveTrajectory(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,tend));
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreateVecs(A,&lambda[0],NULL));
  CHKERRQ(MatCreateVecs(A,&lambda[1],NULL));
  /*   Set initial conditions for the adjoint integration */
  CHKERRQ(VecZeroEntries(lambda[0]));
  CHKERRQ(VecZeroEntries(lambda[1]));
  CHKERRQ(VecGetArray(lambda[0],&u));
  u[0] = 1.;
  CHKERRQ(VecRestoreArray(lambda[0],&u));
  CHKERRQ(VecGetArray(lambda[1],&u));
  u[1] = 1.;
  CHKERRQ(VecRestoreArray(lambda[1],&u));

  CHKERRQ(MatCreateVecs(Ap,&mu[0],NULL));
  CHKERRQ(MatCreateVecs(Ap,&mu[1],NULL));
  CHKERRQ(VecZeroEntries(mu[0]));
  CHKERRQ(VecZeroEntries(mu[1]));
  CHKERRQ(TSSetCostGradients(ts,2,lambda,mu));

  CHKERRQ(TSAdjointSolve(ts));

  /*
  CHKERRQ(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(lambda[1],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(mu[1],PETSC_VIEWER_STDOUT_WORLD));
  */
  CHKERRQ(VecGetArray(mu[0],&u));
  CHKERRQ(VecGetArray(mu[1],&v));
  f = fopen("adj_mu.out", "a");
  CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,f,"%20.15lf %20.15lf %20.15lf\n",tend,u[0],v[0]));
  CHKERRQ(VecRestoreArray(mu[0],&u));
  CHKERRQ(VecRestoreArray(mu[1],&v));
  fclose(f);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));

  CHKERRQ(MatDestroy(&Ap));
  CHKERRQ(VecDestroy(&lambda[0]));
  CHKERRQ(VecDestroy(&lambda[1]));
  CHKERRQ(VecDestroy(&mu[0]));
  CHKERRQ(VecDestroy(&mu[1]));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -ts_monitor -ts_adjoint_monitor

TEST*/
