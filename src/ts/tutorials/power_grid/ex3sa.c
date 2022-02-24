
static char help[] = "Adjoint and tangent linear sensitivity analysis of the basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  This code demonstrate the sensitivity analysis interface to a system of ordinary differential equations with discontinuities.
  It computes the sensitivities of an integral cost function
  \int c*max(0,\theta(t)-u_s)^beta dt
  w.r.t. initial conditions and the parameter P_m.
  Backward Euler method is used for time integration.
  The discontinuities are detected with TSEvent.
 */

#include <petscts.h>
#include "ex3.h"

int main(int argc,char **argv)
{
  TS             ts,quadts;     /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  AppCtx         ctx;
  PetscScalar    *u;
  PetscReal      du[2] = {0.0,0.0};
  PetscBool      ensemble = PETSC_FALSE,flg1,flg2;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *x_ptr,*y_ptr,*s_ptr;
  Vec            lambda[1],q,mu[1];
  PetscInt       direction[2];
  PetscBool      terminate[2];
  Mat            qgrad;
  Mat            sp;            /* Forward sensitivity matrix */
  SAMethod       sa;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ctx.Jac));
  CHKERRQ(MatSetSizes(ctx.Jac,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(ctx.Jac,MATDENSE));
  CHKERRQ(MatSetFromOptions(ctx.Jac));
  CHKERRQ(MatSetUp(ctx.Jac));
  CHKERRQ(MatCreateVecs(ctx.Jac,&U,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ctx.Jacp));
  CHKERRQ(MatSetSizes(ctx.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetFromOptions(ctx.Jacp));
  CHKERRQ(MatSetUp(ctx.Jacp));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&ctx.DRDP));
  CHKERRQ(MatSetUp(ctx.DRDP));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&ctx.DRDU));
  CHKERRQ(MatSetUp(ctx.DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Swing equation options","");CHKERRQ(ierr);
  {
    ctx.beta    = 2;
    ctx.c       = 10000.0;
    ctx.u_s     = 1.0;
    ctx.omega_s = 1.0;
    ctx.omega_b = 120.0*PETSC_PI;
    ctx.H       = 5.0;
    CHKERRQ(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = 5.0;
    CHKERRQ(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    CHKERRQ(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = 1.1;
    CHKERRQ(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    CHKERRQ(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    CHKERRQ(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));
    CHKERRQ(PetscOptionsBool("-ensemble","Run ensemble of different initial conditions","",ensemble,&ensemble,NULL));
    if (ensemble) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }

    CHKERRQ(VecGetArray(U,&u));
    u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
    u[1] = 1.0;
    CHKERRQ(PetscOptionsRealArray("-u","Initial solution","",u,&n,&flg1));
    n    = 2;
    CHKERRQ(PetscOptionsRealArray("-du","Perturbation in initial solution","",du,&n,&flg2));
    u[0] += du[0];
    u[1] += du[1];
    CHKERRQ(VecRestoreArray(U,&u));
    if (flg1 || flg2) {
      ctx.tf      = -1;
      ctx.tcl     = -1;
    }
    sa = SA_ADJ;
    CHKERRQ(PetscOptionsEnum("-sa_method","Sensitivity analysis method (adj or tlm)","",SAMethods,(PetscEnum)sa,(PetscEnum*)&sa,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSSetRHSFunction(ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  CHKERRQ(TSSetRHSJacobian(ts,ctx.Jac,ctx.Jac,(TSRHSJacobian)RHSJacobian,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /*   Set RHS JacobianP */
  CHKERRQ(TSSetRHSJacobianP(ts,ctx.Jacp,RHSJacobianP,&ctx));

  CHKERRQ(TSCreateQuadratureTS(ts,PETSC_FALSE,&quadts));
  CHKERRQ(TSSetRHSFunction(quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
  CHKERRQ(TSSetRHSJacobian(quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
  CHKERRQ(TSSetRHSJacobianP(quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx));
  if (sa == SA_ADJ) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Save trajectory of solution so that TSAdjointSolve() may be used
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    CHKERRQ(TSSetSaveTrajectory(ts));
    CHKERRQ(MatCreateVecs(ctx.Jac,&lambda[0],NULL));
    CHKERRQ(MatCreateVecs(ctx.Jacp,&mu[0],NULL));
    CHKERRQ(TSSetCostGradients(ts,1,lambda,mu));
  }

  if (sa == SA_TLM) {
    PetscScalar val[2];
    PetscInt    row[]={0,1},col[]={0};

    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&qgrad));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&sp));
    CHKERRQ(TSForwardSetSensitivities(ts,1,sp));
    CHKERRQ(TSForwardSetSensitivities(quadts,1,qgrad));
    val[0] = 1./PetscSqrtScalar(1.-(ctx.Pm/ctx.Pmax)*(ctx.Pm/ctx.Pmax))/ctx.Pmax;
    val[1] = 0.0;
    CHKERRQ(MatSetValues(sp,2,row,1,col,val,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(sp,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(sp,MAT_FINAL_ASSEMBLY));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,1.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ts,0.03125));
  CHKERRQ(TSSetFromOptions(ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;

  CHKERRQ(TSSetEventHandler(ts,2,direction,terminate,EventFunction,PostEventFunction,(void*)&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (ensemble) {
    for (du[1] = -2.5; du[1] <= .01; du[1] += .1) {
      CHKERRQ(VecGetArray(U,&u));
      u[0] = PetscAsinScalar(ctx.Pm/ctx.Pmax);
      u[1] = ctx.omega_s;
      u[0] += du[0];
      u[1] += du[1];
      CHKERRQ(VecRestoreArray(U,&u));
      CHKERRQ(TSSetTimeStep(ts,0.03125));
      CHKERRQ(TSSolve(ts,U));
    }
  } else {
    CHKERRQ(TSSolve(ts,U));
  }
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));

  if (sa == SA_ADJ) {
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Adjoint model starts here
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /*   Set initial conditions for the adjoint integration */
    CHKERRQ(VecGetArray(lambda[0],&y_ptr));
    y_ptr[0] = 0.0; y_ptr[1] = 0.0;
    CHKERRQ(VecRestoreArray(lambda[0],&y_ptr));

    CHKERRQ(VecGetArray(mu[0],&x_ptr));
    x_ptr[0] = 0.0;
    CHKERRQ(VecRestoreArray(mu[0],&x_ptr));

    CHKERRQ(TSAdjointSolve(ts));

    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n lambda: d[Psi(tf)]/d[phi0]  d[Psi(tf)]/d[omega0]\n"));
    CHKERRQ(VecView(lambda[0],PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n mu: d[Psi(tf)]/d[pm]\n"));
    CHKERRQ(VecView(mu[0],PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(TSGetCostIntegral(ts,&q));
    CHKERRQ(VecGetArray(q,&x_ptr));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n cost function=%g\n",(double)(x_ptr[0]-ctx.Pm)));
    CHKERRQ(VecRestoreArray(q,&x_ptr));
    CHKERRQ(ComputeSensiP(lambda[0],mu[0],&ctx));
    CHKERRQ(VecGetArray(mu[0],&x_ptr));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n gradient=%g\n",(double)x_ptr[0]));
    CHKERRQ(VecRestoreArray(mu[0],&x_ptr));
    CHKERRQ(VecDestroy(&lambda[0]));
    CHKERRQ(VecDestroy(&mu[0]));
  }
  if (sa == SA_TLM) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n trajectory sensitivity: d[phi(tf)]/d[pm]  d[omega(tf)]/d[pm]\n"));
    CHKERRQ(MatView(sp,PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(TSGetCostIntegral(ts,&q));
    CHKERRQ(VecGetArray(q,&s_ptr));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n cost function=%g\n",(double)(s_ptr[0]-ctx.Pm)));
    CHKERRQ(VecRestoreArray(q,&s_ptr));
    CHKERRQ(MatDenseGetArray(qgrad,&s_ptr));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n gradient=%g\n",(double)s_ptr[0]));
    CHKERRQ(MatDenseRestoreArray(qgrad,&s_ptr));
    CHKERRQ(MatDestroy(&qgrad));
    CHKERRQ(MatDestroy(&sp));
  }
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&ctx.Jac));
  CHKERRQ(MatDestroy(&ctx.Jacp));
  CHKERRQ(MatDestroy(&ctx.DRDU));
  CHKERRQ(MatDestroy(&ctx.DRDP));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -sa_method adj -viewer_binary_skip_info -ts_type cn -pc_type lu

   test:
      suffix: 2
      args: -sa_method tlm -ts_type cn -pc_type lu

   test:
      suffix: 3
      args: -sa_method adj -ts_type rk -ts_rk_type 2a -ts_adapt_type dsp

TEST*/
