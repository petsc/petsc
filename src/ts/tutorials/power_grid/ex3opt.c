
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
  PetscCall(TaoGetSolutionStatus(tao,&iterate,&f,&gnorm,&cnorm,&xdiff,&reason));

  fp = fopen("ex3opt_conv.out","a");
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD,fp,"%" PetscInt_FMT " %g\n",iterate,(double)gnorm));
  fclose(fp);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec                p;
  PetscScalar        *x_ptr;
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
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscFunctionBeginUser;
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Swing equation options","");
  {
    ctx.beta    = 2;
    ctx.c       = 10000.0;
    ctx.u_s     = 1.0;
    ctx.omega_s = 1.0;
    ctx.omega_b = 120.0*PETSC_PI;
    ctx.H       = 5.0;
    PetscCall(PetscOptionsScalar("-Inertia","","",ctx.H,&ctx.H,NULL));
    ctx.D       = 5.0;
    PetscCall(PetscOptionsScalar("-D","","",ctx.D,&ctx.D,NULL));
    ctx.E       = 1.1378;
    ctx.V       = 1.0;
    ctx.X       = 0.545;
    ctx.Pmax    = ctx.E*ctx.V/ctx.X;
    ctx.Pmax_ini = ctx.Pmax;
    PetscCall(PetscOptionsScalar("-Pmax","","",ctx.Pmax,&ctx.Pmax,NULL));
    ctx.Pm      = 1.06;
    PetscCall(PetscOptionsScalar("-Pm","","",ctx.Pm,&ctx.Pm,NULL));
    ctx.tf      = 0.1;
    ctx.tcl     = 0.2;
    PetscCall(PetscOptionsReal("-tf","Time to start fault","",ctx.tf,&ctx.tf,NULL));
    PetscCall(PetscOptionsReal("-tcl","Time to end fault","",ctx.tcl,&ctx.tcl,NULL));
    printtofile = PETSC_FALSE;
    PetscCall(PetscOptionsBool("-printtofile","Print convergence results to file","",printtofile,&printtofile,NULL));
    ctx.sa      = SA_ADJ;
    PetscCall(PetscOptionsEnum("-sa_method","Sensitivity analysis method (adj or tlm)","",SAMethods,(PetscEnum)ctx.sa,(PetscEnum*)&ctx.sa,NULL));
  }
  PetscOptionsEnd();

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&ctx.Jac));
  PetscCall(MatSetSizes(ctx.Jac,2,2,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetType(ctx.Jac,MATDENSE));
  PetscCall(MatSetFromOptions(ctx.Jac));
  PetscCall(MatSetUp(ctx.Jac));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&ctx.Jacp));
  PetscCall(MatSetSizes(ctx.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  PetscCall(MatSetFromOptions(ctx.Jacp));
  PetscCall(MatSetUp(ctx.Jacp));
  PetscCall(MatCreateVecs(ctx.Jac,&ctx.U,NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&ctx.DRDP));
  PetscCall(MatSetUp(ctx.DRDP));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&ctx.DRDU));
  PetscCall(MatSetUp(ctx.DRDU));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ctx.ts));
  PetscCall(TSSetProblemType(ctx.ts,TS_NONLINEAR));
  PetscCall(TSSetType(ctx.ts,TSCN));
  PetscCall(TSSetRHSFunction(ctx.ts,NULL,(TSRHSFunction)RHSFunction,&ctx));
  PetscCall(TSSetRHSJacobian(ctx.ts,ctx.Jac,ctx.Jac,(TSRHSJacobian)RHSJacobian,&ctx));
  PetscCall(TSSetRHSJacobianP(ctx.ts,ctx.Jacp,RHSJacobianP,&ctx));

  if (ctx.sa == SA_ADJ) {
    PetscCall(MatCreateVecs(ctx.Jac,&lambda[0],NULL));
    PetscCall(MatCreateVecs(ctx.Jacp,&mu[0],NULL));
    PetscCall(TSSetSaveTrajectory(ctx.ts));
    PetscCall(TSSetCostGradients(ctx.ts,1,lambda,mu));
    PetscCall(TSCreateQuadratureTS(ctx.ts,PETSC_FALSE,&ctx.quadts));
    PetscCall(TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
    PetscCall(TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
    PetscCall(TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx));
  }
  if (ctx.sa == SA_TLM) {
    PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,&qgrad));
    PetscCall(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&sp));
    PetscCall(TSForwardSetSensitivities(ctx.ts,1,sp));
    PetscCall(TSCreateQuadratureTS(ctx.ts,PETSC_TRUE,&ctx.quadts));
    PetscCall(TSForwardSetSensitivities(ctx.quadts,1,qgrad));
    PetscCall(TSSetRHSFunction(ctx.quadts,NULL,(TSRHSFunction)CostIntegrand,&ctx));
    PetscCall(TSSetRHSJacobian(ctx.quadts,ctx.DRDU,ctx.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&ctx));
    PetscCall(TSSetRHSJacobianP(ctx.quadts,ctx.DRDP,DRDPJacobianTranspose,&ctx));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ctx.ts,1.0));
  PetscCall(TSSetExactFinalTime(ctx.ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ctx.ts,0.03125));
  PetscCall(TSSetFromOptions(ctx.ts));

  direction[0] = direction[1] = 1;
  terminate[0] = terminate[1] = PETSC_FALSE;
  PetscCall(TSSetEventHandler(ctx.ts,2,direction,terminate,EventFunction,PostEventFunction,&ctx));

  /* Create TAO solver and set desired solution method */
  PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
  PetscCall(TaoSetType(tao,TAOBLMVM));
  if (printtofile) {
    PetscCall(TaoSetMonitor(tao,(PetscErrorCode (*)(Tao, void*))monitor,(void *)&ctx,PETSC_NULL));
  }
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  PetscCall(VecCreateSeq(PETSC_COMM_WORLD,1,&p));
  PetscCall(VecGetArray(p,&x_ptr));
  x_ptr[0] = ctx.Pm;
  PetscCall(VecRestoreArray(p,&x_ptr));

  PetscCall(TaoSetSolution(tao,p));
  /* Set routine for function and gradient evaluation */
  PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&ctx));

  /* Set bounds for the optimization */
  PetscCall(VecDuplicate(p,&lowerb));
  PetscCall(VecDuplicate(p,&upperb));
  PetscCall(VecGetArray(lowerb,&x_ptr));
  x_ptr[0] = 0.;
  PetscCall(VecRestoreArray(lowerb,&x_ptr));
  PetscCall(VecGetArray(upperb,&x_ptr));
  x_ptr[0] = 1.1;
  PetscCall(VecRestoreArray(upperb,&x_ptr));
  PetscCall(TaoSetVariableBounds(tao,lowerb,upperb));

  /* Check for any TAO command line options */
  PetscCall(TaoSetFromOptions(tao));
  PetscCall(TaoGetKSP(tao,&ksp));
  if (ksp) {
    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCNONE));
  }

  /* SOLVE THE APPLICATION */
  PetscCall(TaoSolve(tao));

  PetscCall(VecView(p,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&ctx.Jac));
  PetscCall(MatDestroy(&ctx.Jacp));
  PetscCall(MatDestroy(&ctx.DRDU));
  PetscCall(MatDestroy(&ctx.DRDP));
  PetscCall(VecDestroy(&ctx.U));
  if (ctx.sa == SA_ADJ) {
    PetscCall(VecDestroy(&lambda[0]));
    PetscCall(VecDestroy(&mu[0]));
  }
  if (ctx.sa == SA_TLM) {
    PetscCall(MatDestroy(&qgrad));
    PetscCall(MatDestroy(&sp));
  }
  PetscCall(TSDestroy(&ctx.ts));
  PetscCall(VecDestroy(&p));
  PetscCall(VecDestroy(&lowerb));
  PetscCall(VecDestroy(&upperb));
  PetscCall(TaoDestroy(&tao));
  PetscCall(PetscFinalize());
  return 0;
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

  PetscCall(VecGetArrayRead(P,(const PetscScalar**)&x_ptr));
  ctx->Pm = x_ptr[0];
  PetscCall(VecRestoreArrayRead(P,(const PetscScalar**)&x_ptr));

  /* reinitialize the solution vector */
  PetscCall(VecGetArray(ctx->U,&u));
  u[0] = PetscAsinScalar(ctx->Pm/ctx->Pmax);
  u[1] = 1.0;
  PetscCall(VecRestoreArray(ctx->U,&u));
  PetscCall(TSSetSolution(ctx->ts,ctx->U));

  /* reset time */
  PetscCall(TSSetTime(ctx->ts,0.0));

  /* reset step counter, this is critical for adjoint solver */
  PetscCall(TSSetStepNumber(ctx->ts,0));

  /* reset step size, the step size becomes negative after TSAdjointSolve */
  PetscCall(TSSetTimeStep(ctx->ts,0.03125));

  /* reinitialize the integral value */
  PetscCall(TSGetQuadratureTS(ctx->ts,NULL,&ctx->quadts));
  PetscCall(TSGetSolution(ctx->quadts,&q));
  PetscCall(VecSet(q,0.0));

  if (ctx->sa == SA_TLM) { /* reset the forward sensitivities */
    TS             quadts;
    Mat            sp;
    PetscScalar    val[2];
    const PetscInt row[]={0,1},col[]={0};

    PetscCall(TSGetQuadratureTS(ctx->ts,NULL,&quadts));
    PetscCall(TSForwardGetSensitivities(quadts,NULL,&qgrad));
    PetscCall(MatZeroEntries(qgrad));
    PetscCall(TSForwardGetSensitivities(ctx->ts,NULL,&sp));
    val[0] = 1./PetscSqrtScalar(1.-(ctx->Pm/ctx->Pmax)*(ctx->Pm/ctx->Pmax))/ctx->Pmax;
    val[1] = 0.0;
    PetscCall(MatSetValues(sp,2,row,1,col,val,INSERT_VALUES));
    PetscCall(MatAssemblyBegin(sp,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(sp,MAT_FINAL_ASSEMBLY));
  }

  /* solve the ODE */
  PetscCall(TSSolve(ctx->ts,ctx->U));
  PetscCall(TSGetSolveTime(ctx->ts,&ftime));
  PetscCall(TSGetStepNumber(ctx->ts,&steps));

  if (ctx->sa == SA_ADJ) {
    Vec *lambda,*mu;
    /* reset the terminal condition for adjoint */
    PetscCall(TSGetCostGradients(ctx->ts,&nadj,&lambda,&mu));
    PetscCall(VecGetArray(lambda[0],&y_ptr));
    y_ptr[0] = 0.0; y_ptr[1] = 0.0;
    PetscCall(VecRestoreArray(lambda[0],&y_ptr));
    PetscCall(VecGetArray(mu[0],&x_ptr));
    x_ptr[0] = -1.0;
    PetscCall(VecRestoreArray(mu[0],&x_ptr));

    /* solve the adjont */
    PetscCall(TSAdjointSolve(ctx->ts));

    PetscCall(ComputeSensiP(lambda[0],mu[0],ctx));
    PetscCall(VecCopy(mu[0],G));
  }

  if (ctx->sa == SA_TLM) {
    PetscCall(VecGetArray(G,&x_ptr));
    PetscCall(MatDenseGetArray(qgrad,&y_ptr));
    x_ptr[0] = y_ptr[0]-1.;
    PetscCall(MatDenseRestoreArray(qgrad,&y_ptr));
    PetscCall(VecRestoreArray(G,&x_ptr));
  }

  PetscCall(TSGetSolution(ctx->quadts,&q));
  PetscCall(VecGetArray(q,&x_ptr));
  *f   = -ctx->Pm + x_ptr[0];
  PetscCall(VecRestoreArray(q,&x_ptr));
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
