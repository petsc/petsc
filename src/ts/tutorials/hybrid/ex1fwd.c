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

  PetscFunctionBegin;
  if (time >= actx->print_time) {
    actx->print_time += 1./256.;
    PetscCall(TSForwardGetSensitivities(ts,&nump,&sp));
    PetscCall(MatDenseGetColumn(sp,2,&u));
    f = fopen("fwd_sp.out", "a");
    PetscCall(PetscFPrintf(PETSC_COMM_WORLD,f,"%20.15lf %20.15lf %20.15lf\n",(double)time,(double)PetscRealPart(u[0]),(double)PetscRealPart(u[1])));
    PetscCall(MatDenseRestoreColumn(sp,&u));
    fclose(f);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode EventFunction(TS ts,PetscReal t,Vec U,PetscScalar *fvalue,void *ctx)
{
  AppCtx            *actx=(AppCtx*)ctx;
  const PetscScalar *u;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(U,&u));
  if (actx->mode == 1) {
    fvalue[0] = u[1]-actx->lambda1*u[0];
  }else if (actx->mode == 2) {
    fvalue[0] = u[1]-actx->lambda2*u[0];
  }
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

PetscErrorCode ShiftGradients(TS ts,Vec U,AppCtx *actx)
{
  Mat               sp;
  PetscScalar       *x;
  PetscScalar       *u;
  PetscScalar       tmp[2],A1[2][2],A2[2],denorm;
  PetscInt          nump;

  PetscFunctionBegin;
  PetscCall(TSForwardGetSensitivities(ts,&nump,&sp));
  PetscCall(VecGetArray(U,&u));

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

  PetscCall(VecRestoreArray(U,&u));

  PetscCall(MatDenseGetColumn(sp,0,&x));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  PetscCall(MatDenseRestoreColumn(sp,&x));

  PetscCall(MatDenseGetColumn(sp,1,&x));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0];
  x[1]   = tmp[1];
  PetscCall(MatDenseRestoreColumn(sp,&x));

  PetscCall(MatDenseGetColumn(sp,2,&x));
  tmp[0] = A1[0][0]*x[0]+A1[0][1]*x[1];
  tmp[1] = A1[1][0]*x[0]+A1[1][1]*x[1];
  x[0]   = tmp[0]+A2[0];
  x[1]   = tmp[1]+A2[1];
  PetscCall(MatDenseRestoreColumn(sp,&x));

  PetscFunctionReturn(0);
}

PetscErrorCode PostEventFunction(TS ts,PetscInt nevents,PetscInt event_list[],PetscReal t,Vec U,PetscBool forwardsolve,void* ctx)
{
  AppCtx         *actx=(AppCtx*)ctx;

  PetscFunctionBegin;
  /* PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD)); */
  PetscCall(ShiftGradients(ts,U,actx));
  if (actx->mode == 1) {
    actx->mode = 2;
    /* PetscCall(PetscPrintf(PETSC_COMM_SELF,"Change from mode 1 to 2 at t = %f \n",(double)t)); */
  } else if (actx->mode == 2) {
    actx->mode = 1;
    /* PetscCall(PetscPrintf(PETSC_COMM_SELF,"Change from mode 2 to 1 at t = %f \n",(double)t)); */
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
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");
  app.mode = 1;
  app.lambda1 = 2.75;
  app.lambda2 = 0.36;
  app.print_time = 1./256.;
  tend = 0.125;
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"ex1fwd options","");
  {
    PetscCall(PetscOptionsReal("-lambda1","","",app.lambda1,&app.lambda1,NULL));
    PetscCall(PetscOptionsReal("-lambda2","","",app.lambda2,&app.lambda2,NULL));
    PetscCall(PetscOptionsReal("-tend","","",tend,&tend,NULL));
  }
  PetscOptionsEnd();

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
  PetscCall(MatSetSizes(Ap,n,3,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(Ap,MATDENSE));
  PetscCall(MatSetFromOptions(Ap));
  PetscCall(MatSetUp(Ap));
  PetscCall(MatZeroEntries(Ap));

  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,3,NULL,&sp));
  PetscCall(MatZeroEntries(sp));
  PetscCall(MatShift(sp,1.0));

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

  PetscCall(TSForwardSetSensitivities(ts,3,sp));
  /*   Set RHS JacobianP */
  PetscCall(TSSetRHSJacobianP(ts,Ap,RHSJacobianP,&app));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts,tend));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts,1./256.));
  PetscCall(TSMonitorSet(ts,MyMonitor,&app,PETSC_NULL));
  PetscCall(TSSetFromOptions(ts));

  /* Set directions and terminate flags for the two events */
  direction[0] = 0;
  terminate[0] = PETSC_FALSE;
  PetscCall(TSSetEventHandler(ts,1,direction,terminate,EventFunction,PostEventFunction,(void*)&app));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Run timestepping solver
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));

  PetscCall(MatDestroy(&Ap));
  PetscCall(MatDestroy(&sp));
  PetscCall(PetscFinalize());
  return(0);
}

/*TEST

   build:
      requires: !complex

   test:
      args: -ts_monitor

TEST*/
