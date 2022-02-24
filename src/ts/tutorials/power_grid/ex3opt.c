
static char help[] = "Finds optimal parameter P_m for the generator system while maintaining generator stability.\n";

/*F

\begin{eqnarray}
                 \frac{d \theta}{dt} = \omega_b (\omega - \omega_s)
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - P_max \sin(\theta) -D(\omega - \omega_s)\\
\end{eqnarray}

F*/

/*
  This code demonstrates how to solve a ODE-constrained optimization problem with TAO, TSEvent, TSAdjoint and TS.
  The problem features discontinuities and a cost function in integral form.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex3adj.c for details.
*/

#include <petsctao.h>
#include <petscts.h>
#include "ex3.h"

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

PetscErrorCode monitor(Tao tao,AppCtx *ctx)
{
  FILE               *fp;
  PetscInt           iterate;
  PetscReal          f,gnorm,cnorm,xdiff;
  TaoConvergedReason reason;

  PetscFunctionBeginUser;
  CHKERRQ(TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason));

  fp = fopen("ex3opt_conv.out","a");
  CHKERRQ(PetscFPrintf(PETSC_COMM_WORLD,fp,"%D %g\n",iterate,(double)gnorm));
  fclose(fp);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec                p;
  PetscScalar        *x_ptr;
  PetscErrorCode     ierr;
  PetscMPIInt        size;
  AppCtx             ctx;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  Vec                lambda[1],mu[1],lowerb,upperb;
  PetscBool          printtofile;
  PetscInt           direction[2];
  PetscBool          terminate[2];
  Mat                qgrad;         /* Forward sesivitiy */
  Mat                sp;            /* Forward sensitivity matrix */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  PetscFunctionBeginUser;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

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
    ctx.Pm      = 1.06;
    CHKERRQ(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    CHKERRQ(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    CHKERRQ(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));
    printtofile = PETSC_FALSE;
    CHKERRQ(PetscOptionsBool("-printtofile","Print convergence results to file","",printtofile,&printtofile,NULL));
    ctx.sa      = SA_ADJ;
    CHKERRQ(PetscOptionsEnum("-sa_method","Sensitivity analysis method (adj or tlm)","",SAMethods,(PetscEnum)ctx.sa,(PetscEnum*)&ctx.sa,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ctx.Jac));
  CHKERRQ(MatSetSizes(ctx.Jac,2,2,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetType(ctx.Jac,MATDENSE));
  CHKERRQ(MatSetFromOptions(ctx.Jac));
  CHKERRQ(MatSetUp(ctx.Jac));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&ctx.Jacp));
  CHKERRQ(MatSetSizes(ctx.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetFromOptions(ctx.Jacp));
  CHKERRQ(MatSetUp(ctx.Jacp));
  CHKERRQ(MatCreateVecs(ctx.Jac,&ctx.U,NULL));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&ctx.DRDP));
  CHKERRQ(MatSetUp(ctx.DRDP));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&ctx.DRDU));
  CHKERRQ(MatSetUp(ctx.DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ctx.ts));
  CHKERRQ(TSSetProblemType(ctx.ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ctx.ts,TSCN));
  CHKERRQ(TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  CHKERRQ(TSSetRHSJacobian(ctx.ts,ctx.Jac,ctx.Jac,(TSRHSJacobian)RHSJacobian,&ctx));
  CHKERRQ(TSSetRHSJacobianP(ctx.ts,ctx.Jacp,RHSJacobianP,&ctx));

  if (ctx.sa == SA_ADJ) {
    CHKERRQ(MatCreateVecs(ctx.Jac,&lambda[0],NULL));
    CHKERRQ(MatCreateVecs(ctx.Jacp,&mu[0],NULL));
    CHKERRQ(TSSetSaveTrajectory(ctx.ts));
    CHKERRQ(TSSetCostGradients(ctx.ts,1,lambda,mu));
    CHKERRQ(TSCreateQuadratureTS(ctx.ts,PETSC_FALSE,&ctx.quadts));
    CHKERRQ(TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
    CHKERRQ(TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
    CHKERRQ(TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx));
  }
  if (ctx.sa == SA_TLM) {
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&qgrad));
    CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&sp));
    CHKERRQ(TSForwardSetSensitivities(ctx.ts,1,sp));
    CHKERRQ(TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&ctx.quadts));
    CHKERRQ(TSForwardSetSensitivities(ctx.quadts,1,qgrad));
    CHKERRQ(TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
    CHKERRQ(TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
    CHKERRQ(TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ctx.ts,1.0));
  CHKERRQ(TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ctx.ts,0.03125));
  CHKERRQ(TSSetFromOptions(ctx.ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;
  CHKERRQ(TSSetEventHandler(ctx.ts,2,direction,terminate,EventFunction,PostEventFunction,&ctx));

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOBLMVM));
  if (printtofile) {
    CHKERRQ(TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL));
  }
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  CHKERRQ(VecCreateSeq(PETSC_COMM_WORLD,1,&p));
  CHKERRQ(VecGetArray(p,&x_ptr));
  x_ptr[0] = ctx.Pm;
  CHKERRQ(VecRestoreArray(p,&x_ptr));

  CHKERRQ(TaoSetSolution(tao,p));
  /* Set routine for function and gradient evaluation */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&ctx));

  /* Set bounds for the optimization */
  CHKERRQ(VecDuplicate(p,&lowerb));
  CHKERRQ(VecDuplicate(p,&upperb));
  CHKERRQ(VecGetArray(lowerb,&x_ptr));
  x_ptr[0] = 0.;
  CHKERRQ(VecRestoreArray(lowerb,&x_ptr));
  CHKERRQ(VecGetArray(upperb,&x_ptr));
  x_ptr[0] = 1.1;
  CHKERRQ(VecRestoreArray(upperb,&x_ptr));
  CHKERRQ(TaoSetVariableBounds(tao,lowerb,upperb));

  /* Check for any TAO command line options */
  CHKERRQ(TaoSetFromOptions(tao));
  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
  }

  /* SOLVE THE APPLICATION */
  CHKERRQ(TaoSolve(tao));

  CHKERRQ(VecView(p,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&ctx.Jac));
  CHKERRQ(MatDestroy(&ctx.Jacp));
  CHKERRQ(MatDestroy(&ctx.DRDU));
  CHKERRQ(MatDestroy(&ctx.DRDP));
  CHKERRQ(VecDestroy(&ctx.U));
  if (ctx.sa == SA_ADJ) {
    CHKERRQ(VecDestroy(&lambda[0]));
    CHKERRQ(VecDestroy(&mu[0]));
  }
  if (ctx.sa == SA_TLM) {
    CHKERRQ(MatDestroy(&qgrad));
    CHKERRQ(MatDestroy(&sp));
  }
  CHKERRQ(TSDestroy(&ctx.ts));
  CHKERRQ(VecDestroy(&p));
  CHKERRQ(VecDestroy(&lowerb));
  CHKERRQ(VecDestroy(&upperb));
  CHKERRQ(TaoDestroy(&tao));
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx0)
{
  AppCtx         *ctx = (AppCtx*)ctx0;
  PetscInt       nadj;
  PetscReal      ftime;
  PetscInt       steps;
  PetscScalar    *u;
  PetscScalar    *x_ptr,*y_ptr;
  Vec            q;
  Mat            qgrad;

  CHKERRQ(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reinitialize the solution vector */
  CHKERRQ(VecGetArray(ctx->U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  CHKERRQ(VecRestoreArray(ctx->U,&u));
  CHKERRQ(TSSetSolution(ctx->ts,ctx->U));

  /* reset time */
  CHKERRQ(TSSetTime(ctx->ts,0.0));

  /* reset step counter, this is critical for adjoint solver */
  CHKERRQ(TSSetStepNumber(ctx->ts,0));

  /* reset step size, the step size becomes negative after TSAdjointSolve */
  CHKERRQ(TSSetTimeStep(ctx->ts,0.03125));

  /* reinitialize the integral value */
  CHKERRQ(TSGetQuadratureTS(ctx->ts,NULL,&ctx->quadts));
  CHKERRQ(TSGetSolution(ctx->quadts,&q));
  CHKERRQ(VecSet(q,0.0));

  if (ctx->sa == SA_TLM) { /* reset the forward sensitivities */
    TS             quadts;
    Mat            sp;
    PetscScalar    val[2];
    const PetscInt row[]={0,1},col[]={0};

    CHKERRQ(TSGetQuadratureTS(ctx->ts,NULL,&quadts));
    CHKERRQ(TSForwardGetSensitivities(quadts,NULL,&qgrad));
    CHKERRQ(MatZeroEntries(qgrad));
    CHKERRQ(TSForwardGetSensitivities(ctx->ts,NULL,&sp));
    val[0] = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax;
    val[1] = 0.0;
    CHKERRQ(MatSetValues(sp,2,row,1,col,val,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(sp,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(sp,MAT_FINAL_ASSEMBLY));
  }

  /* solve the ODE */
  CHKERRQ(TSSolve(ctx->ts,ctx->U));
  CHKERRQ(TSGetSolveTime(ctx->ts,&ftime));
  CHKERRQ(TSGetStepNumber(ctx->ts,&steps));

  if (ctx->sa == SA_ADJ) {
    Vec *lambda,*mu;
    /* reset the terminal condition for adjoint */
    CHKERRQ(TSGetCostGradients(ctx->ts,&nadj,&lambda,&mu));
    CHKERRQ(VecGetArray(lambda[0],&y_ptr));
    y_ptr[0] = 0.0; y_ptr[1] = 0.0;
    CHKERRQ(VecRestoreArray(lambda[0],&y_ptr));
    CHKERRQ(VecGetArray(mu[0],&x_ptr));
    x_ptr[0] = -1.0;
    CHKERRQ(VecRestoreArray(mu[0],&x_ptr));

    /* solve the adjont */
    CHKERRQ(TSAdjointSolve(ctx->ts));

    CHKERRQ(ComputeSensiP(lambda[0],mu[0],ctx));
    CHKERRQ(VecCopy(mu[0],G));
  }

  if (ctx->sa == SA_TLM) {
    CHKERRQ(VecGetArray(G,&x_ptr));
    CHKERRQ(MatDenseGetArray(qgrad,&y_ptr));
    x_ptr[0] = y_ptr[0]-1.;
    CHKERRQ(MatDenseRestoreArray(qgrad,&y_ptr));
    CHKERRQ(VecRestoreArray(G,&x_ptr));
  }

  CHKERRQ(TSGetSolution(ctx->quadts,&q));
  CHKERRQ(VecGetArray(q,&x_ptr));
  *f   = -ctx->Pm + x_ptr[0];
  CHKERRQ(VecRestoreArray(q,&x_ptr));
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      args: -viewer_binary_skip_info -ts_type cn -pc_type lu -tao_monitor

   test:
      suffix: 2
      output_file: output/ex3opt_1.out
      args: -sa_method tlm -ts_type cn -pc_type lu -tao_monitor
TEST*/
