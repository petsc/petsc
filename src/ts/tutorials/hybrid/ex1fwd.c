static char help[] = "Trajectory sensitivity of a hybrid system with state-dependent switchings.\n";

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
  PetscReal   print_time;
} AppCtx;

PetscErrorCode MyMonitor(TS ts,PetscInt stepnum,PetscReal time,Vec U,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  Mat               sp;
  PetscScalar       *u;
  PetscInt          nump;
  FILE              *f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (time >= actx->print_time) {
    actx->print_time += 1./256.;
    ierr = TSForwardGetSensitivities(ts,&nump,&sp);CHKERRQ(ierr);
    ierr = MatDenseGetColumn(sp,2,&u);CHKERRQ(ierr);
    f = fopen("fwd_sp.out", "a");
    ierr = PetscFPrintf(PETSC_COMM_WORLD,f,"%20.15lf %20.15lf %20.15lf\n",time,u[0],u[1]);CHKERRQ(ierr);
    ierr = MatDenseRestoreColumn(sp,&u);CHKERRQ(ierr);
    fclose(f);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  const PetscScalar *u;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  if (actx->mode == 1) {
    fvalue[0] = u[1]-actx->lambda1*u[0];
  }else if (actx->mode == 2) {
    fvalue[0] = u[1]-actx->lambda2*u[0];
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ShiftGradients(TS ts,Vec U,AppCtx *actx)
{
  Mat               sp;
  PetscScalar       *x;
  PetscScalar       *u;
  PetscErrorCode    ierr;
  PetscScalar       tmp[2],A1[2][2],A2[2],denorm;
  PetscInt          nump;

  PetscFunctionBegin;
  ierr = TSForwardGetSensitivities(ts,&nump,&sp);CHKERRQ(ierr);
  ierr = VecGetArray(U,&u);CHKERRQ(ierr);

  if (actx->mode==1) {
    denorm = -actx->lambda1*(u[0]-100.*u[1])+1.*(10.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(-actx->lambda1)/denorm+1.;
    A1[1][0] = -110.*u[0]*(-actx->lambda1)/denorm;
    A1[0][1] = 110.*u[1]*1./denorm;
    A1[1][1] = -110.*u[0]*1./denorm+1.;

    A2[0] = 110.*u[1]*(-u[0])/denorm;
    A2[1] = -110.*u[0]*(-u[0])/denorm;
  }else {
    denorm = -actx->lambda2*(u[0]+10.*u[1])+1.*(-100.*u[0]+u[1]);
    A1[0][0] = 110.*u[1]*(actx->lambda2)/denorm+1;
    A1[1][0] = -110.*u[0]*(actx->lambda2)/denorm;
    A1[0][1] = -110.*u[1]*1./denorm;
    A1[1][1] = 110.*u[0]*1./denorm+1.;

    A2[0] = 0;
    A2[1] = 0;
  }

  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  ierr   = MatDenseGetColumn(sp,0,&x);CHKERRQ(ierr);
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  ierr   = MatDenseRestoreColumn(sp,&x);CHKERRQ(ierr);

  ierr   = MatDenseGetColumn(sp,1,&x);CHKERRQ(ierr);
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  ierr   = MatDenseRestoreColumn(sp,&x);CHKERRQ(ierr);

  ierr   = MatDenseGetColumn(sp,2,&x);CHKERRQ(ierr);
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0]+A2[0];
  x[1]   = tmp[1]+A2[1];
  ierr   = MatDenseRestoreColumn(sp,&x);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *actx=(AppCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  ierr = ShiftGradients(ts,U,actx);CHKERRQ(ierr);
  if (actx->mode == 1) {
    actx->mode = 2;
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"Change from mode 1 to 2 at t = %f \n",(double)t);CHKERRQ(ierr); */
  } else if (actx->mode == 2) {
    actx->mode = 1;
    /* ierr = PetscPrintf(PETSC_COMM_SELF,"Change from mode 2 to 1 at t = %f \n",(double)t);CHKERRQ(ierr); */
  }
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  if (actx->mode == 1) {
    f[0] = udot[0]-u[0]+100*u[1];
    f[1] = udot[1]-10*u[0]-u[1];
  } else if (actx->mode == 2) {
    f[0] = udot[0]-u[0]-10*u[1];
    f[1] = udot[1]+100*u[0]-u[1];
  }

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
  AppCtx            *actx=(AppCtx*)ctx;
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);

  if (actx->mode == 1) {
    J[0][0] = a-1;                       J[0][1] = 100;
    J[1][0] = -10;                       J[1][1] = a-1;
  } else if (actx->mode == 2) {
    J[0][0] = a-1;                       J[0][1] = -10;
    J[1][0] = 100;                       J[1][1] = a-1;
  }
  ierr = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Matrix JacobianP is constant so that it only needs to be evaluated once */
static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec X,Mat Ap,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  Mat            Ap;            /* Jacobian dfdp */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  PetscScalar    *u;
  AppCtx         app;
  PetscInt       direction[1];
  PetscBool      terminate[1];
  Mat            sp;
  PetscReal      tend;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");
  app.mode = 1;
  app.lambda1 = 2.75;
  app.lambda2 = 0.36;
  app.print_time = 1./256.;
  tend = 0.125;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex1fwd options","");CHKERRQ(ierr);
  {
    ierr = PetscOptionsReal("-lambda1","","",app.lambda1,&app.lambda1,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-lambda2","","",app.lambda2,&app.lambda2,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-tend","","",tend,&tend,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&Ap);CHKERRQ(ierr);
  ierr = MatSetSizes(Ap,n,3,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(Ap,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(Ap);CHKERRQ(ierr);
  ierr = MatSetUp(Ap);CHKERRQ(ierr);
  ierr = MatZeroEntries(Ap);CHKERRQ(ierr);

  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,3,NULL,&sp);CHKERRQ(ierr);
  ierr = MatZeroEntries(sp);CHKERRQ(ierr);
  ierr = MatShift(sp,1.0);CHKERRQ(ierr);

  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = 0;
  u[1] = 1;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction)IFunction,&app);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&app);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  ierr = TSForwardSetSensitivities(ts,3,sp);CHKERRQ(ierr);
  /*   Set RHS JacobianP */
  ierr = TSSetRHSJacobianP(ts,Ap,RHSJacobianP,&app);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1./256.);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MyMonitor,&app,PETSC_NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* Set directions and terminate flags for the two events */
  direction[0] = 0;
  terminate[0] = PETSC_FALSE;
  ierr = TSSetEventHandler(ts,1,direction,terminate,EventFunction,PostEventFunction,(void*)&app);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = MatDestroy(&Ap);CHKERRQ(ierr);
  ierr = MatDestroy(&sp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return(ierr);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -ts_monitor

TEST*/
