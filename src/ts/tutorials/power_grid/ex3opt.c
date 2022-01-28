
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
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason);CHKERRQ(ierr);

  fp = fopen("ex3opt_conv.out","a");
  ierr = PetscFPrintf(PETSC_COMM_WORLD,fp,"%D %g\n",iterate,(double)gnorm);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

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
    ierr        = PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL);CHKERRQ(ierr);
    ctx.D       = 5.0;
    ierr        = PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL);CHKERRQ(ierr);
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    ierr        = PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL);CHKERRQ(ierr);
    ctx.Pm      = 1.06;
    ierr        = PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL);CHKERRQ(ierr);
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    ierr        = PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL);CHKERRQ(ierr);
    ierr        = PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL);CHKERRQ(ierr);
    printtofile = PETSC_FALSE;
    ierr        = PetscOptionsBool("-printtofile","Print convergence results to file","",printtofile,&printtofile,NULL);CHKERRQ(ierr);
    ctx.sa      = SA_ADJ;
    ierr        = PetscOptionsEnum("-sa_method","Sensitivity analysis method (adj or tlm)","",SAMethods,(PetscEnum)ctx.sa,(PetscEnum*)&ctx.sa,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx.Jac);CHKERRQ(ierr);
  ierr = MatSetSizes(ctx.Jac,2,2,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(ctx.Jac,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx.Jac);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.Jac);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&ctx.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(ctx.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(ctx.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.Jacp);CHKERRQ(ierr);
  ierr = MatCreateVecs(ctx.Jac,&ctx.U,NULL);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&ctx.DRDP);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.DRDP);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&ctx.DRDU);CHKERRQ(ierr);
  ierr = MatSetUp(ctx.DRDU);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ctx.ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ctx.ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ctx.ts,TSCN);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ctx.ts,ctx.Jac,ctx.Jac,(TSRHSJacobian)RHSJacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(ctx.ts,ctx.Jacp,RHSJacobianP,&ctx);CHKERRQ(ierr);

  if (ctx.sa == SA_ADJ) {
    ierr = MatCreateVecs(ctx.Jac,&lambda[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(ctx.Jacp,&mu[0],NULL);CHKERRQ(ierr);
    ierr = TSSetSaveTrajectory(ctx.ts);CHKERRQ(ierr);
    ierr = TSSetCostGradients(ctx.ts,1,lambda,mu);CHKERRQ(ierr);
    ierr = TSCreateQuadratureTS(ctx.ts,PETSC_FALSE,&ctx.quadts);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx);CHKERRQ(ierr);
  }
  if (ctx.sa == SA_TLM) {
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&qgrad);CHKERRQ(ierr);
    ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&sp);CHKERRQ(ierr);
    ierr = TSForwardSetSensitivities(ctx.ts,1,sp);CHKERRQ(ierr);
    ierr = TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&ctx.quadts);CHKERRQ(ierr);
    ierr = TSForwardSetSensitivities(ctx.quadts,1,qgrad);CHKERRQ(ierr);
    ierr = TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx);CHKERRQ(ierr);
    ierr = TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ctx.ts,1.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ctx.ts,0.03125);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ctx.ts);CHKERRQ(ierr);

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;
  ierr = TSSetEventHandler(ctx.ts,2,direction,terminate,EventFunction,PostEventFunction,&ctx);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  if (printtofile) {
    ierr = TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL);CHKERRQ(ierr);
  }
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = VecCreateSeq(PETSC_COMM_WORLD,1,&p);CHKERRQ(ierr);
  ierr = VecGetArray(p,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = ctx.Pm;
  ierr = VecRestoreArray(p,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,p);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&ctx);CHKERRQ(ierr);

  /* Set bounds for the optimization */
  ierr = VecDuplicate(p,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(p,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.1;
  ierr = VecRestoreArray(upperb,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,lowerb,upperb);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = VecView(p,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&ctx.Jac);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx.Jacp);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx.DRDU);CHKERRQ(ierr);
  ierr = MatDestroy(&ctx.DRDP);CHKERRQ(ierr);
  ierr = VecDestroy(&ctx.U);CHKERRQ(ierr);
  if (ctx.sa == SA_ADJ) {
    ierr = VecDestroy(&lambda[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&mu[0]);CHKERRQ(ierr);
  }
  if (ctx.sa == SA_TLM) {
    ierr = MatDestroy(&qgrad);CHKERRQ(ierr);
    ierr = MatDestroy(&sp);CHKERRQ(ierr);
  }
  ierr = TSDestroy(&ctx.ts);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

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
  PetscErrorCode ierr;

  ierr = VecGetArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);
  ctx->Pm = x_ptr[0];
  ierr = VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr);CHKERRQ(ierr);

  /* reinitialize the solution vector */
  ierr = VecGetArray(ctx->U,&u);CHKERRQ(ierr);
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  ierr = VecRestoreArray(ctx->U,&u);CHKERRQ(ierr);
  ierr = TSSetSolution(ctx->ts,ctx->U);CHKERRQ(ierr);

  /* reset time */
  ierr = TSSetTime(ctx->ts,0.0);CHKERRQ(ierr);

  /* reset step counter, this is critical for adjoint solver */
  ierr = TSSetStepNumber(ctx->ts,0);CHKERRQ(ierr);

  /* reset step size, the step size becomes negative after TSAdjointSolve */
  ierr = TSSetTimeStep(ctx->ts,0.03125);CHKERRQ(ierr);

  /* reinitialize the integral value */
  ierr = TSGetQuadratureTS(ctx->ts,NULL,&ctx->quadts);CHKERRQ(ierr);
  ierr = TSGetSolution(ctx->quadts,&q);CHKERRQ(ierr);
  ierr = VecSet(q,0.0);CHKERRQ(ierr);

  if (ctx->sa == SA_TLM) { /* reset the forward sensitivities */
    TS             quadts;
    Mat            sp;
    PetscScalar    val[2];
    const PetscInt row[]={0,1},col[]={0};

    ierr = TSGetQuadratureTS(ctx->ts,NULL,&quadts);CHKERRQ(ierr);
    ierr = TSForwardGetSensitivities(quadts,NULL,&qgrad);CHKERRQ(ierr);
    ierr = MatZeroEntries(qgrad);CHKERRQ(ierr);
    ierr = TSForwardGetSensitivities(ctx->ts,NULL,&sp);CHKERRQ(ierr);
    val[0] = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax;
    val[1] = 0.0;
    ierr = MatSetValues(sp,2,row,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* solve the ODE */
  ierr = TSSolve(ctx->ts,ctx->U);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ctx->ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ctx->ts,&steps);CHKERRQ(ierr);

  if (ctx->sa == SA_ADJ) {
    Vec *lambda,*mu;
    /* reset the terminal condition for adjoint */
    ierr = TSGetCostGradients(ctx->ts,&nadj,&lambda,&mu);CHKERRQ(ierr);
    ierr = VecGetArray(lambda[0],&y_ptr);CHKERRQ(ierr);
    y_ptr[0] = 0.0; y_ptr[1] = 0.0;
    ierr = VecRestoreArray(lambda[0],&y_ptr);CHKERRQ(ierr);
    ierr = VecGetArray(mu[0],&x_ptr);CHKERRQ(ierr);
    x_ptr[0] = -1.0;
    ierr = VecRestoreArray(mu[0],&x_ptr);CHKERRQ(ierr);

    /* solve the adjont */
    ierr = TSAdjointSolve(ctx->ts);CHKERRQ(ierr);

    ierr = ComputeSensiP(lambda[0],mu[0],ctx);CHKERRQ(ierr);
    ierr = VecCopy(mu[0],G);CHKERRQ(ierr);
  }

  if (ctx->sa == SA_TLM) {
    ierr = VecGetArray(G,&x_ptr);CHKERRQ(ierr);
    ierr = MatDenseGetArray(qgrad,&y_ptr);CHKERRQ(ierr);
    x_ptr[0] = y_ptr[0]-1.;
    ierr = MatDenseRestoreArray(qgrad,&y_ptr);CHKERRQ(ierr);
    ierr = VecRestoreArray(G,&x_ptr);CHKERRQ(ierr);
  }

  ierr = TSGetSolution(ctx->quadts,&q);CHKERRQ(ierr);
  ierr = VecGetArray(q,&x_ptr);CHKERRQ(ierr);
  *f   = -ctx->Pm + x_ptr[0];
  ierr = VecRestoreArray(q,&x_ptr);CHKERRQ(ierr);
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
