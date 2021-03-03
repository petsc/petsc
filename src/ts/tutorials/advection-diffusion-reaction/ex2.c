
static char help[] = "Reaction Equation from Chemistry\n";

/*

     Page 6, An example from Atomospheric Chemistry

                 u_1_t =
                 u_2_t =
                 u_3_t =
                 u_4_t =

  -ts_monitor_lg_error -ts_monitor_lg_solution  -ts_view -ts_max_time 2.e4

*/


/*
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscts.h>

typedef struct {
  PetscScalar k1,k2,k3;
  PetscScalar sigma2;
  Vec         initialsolution;
} AppCtx;

PetscScalar k1(AppCtx *ctx,PetscReal t)
{
  PetscReal th    = t/3600.0;
  PetscReal barth = th - 24.0*PetscFloorReal(th/24.0);
  if (((((PetscInt)th) % 24) < 4) || ((((PetscInt)th) % 24) >= 20)) return(1.0e-40);
  else return(ctx->k1*PetscExpReal(7.0*PetscPowReal(PetscSinReal(.0625*PETSC_PI*(barth - 4.0)),.2)));
}

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(F,&f);CHKERRQ(ierr);
  f[0] = udot[0] - k1(ctx,t)*u[2] + ctx->k2*u[0];
  f[1] = udot[1] - k1(ctx,t)*u[2] + ctx->k3*u[1]*u[3] - ctx->sigma2;
  f[2] = udot[2] - ctx->k3*u[1]*u[3] + k1(ctx,t)*u[2];
  f[3] = udot[3] - ctx->k2*u[0] + ctx->k3*u[1]*u[3];
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          rowcol[] = {0,1,2,3};
  PetscScalar       J[4][4];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  ierr    = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  J[0][0] = a + ctx->k2;   J[0][1] = 0.0;                J[0][2] = -k1(ctx,t);       J[0][3] = 0.0;
  J[1][0] = 0.0;           J[1][1] = a + ctx->k3*u[3];   J[1][2] = -k1(ctx,t);       J[1][3] = ctx->k3*u[1];
  J[2][0] = 0.0;           J[2][1] = -ctx->k3*u[3];      J[2][2] = a + k1(ctx,t);    J[2][3] =  -ctx->k3*u[1];
  J[3][0] =  -ctx->k2;     J[3][1] = ctx->k3*u[3];       J[3][2] = 0.0;              J[3][3] = a + ctx->k3*u[1];
  ierr    = MatSetValues(B,4,rowcol,4,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode Solution(TS ts,PetscReal t,Vec U,AppCtx *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ctx->initialsolution,U);CHKERRQ(ierr);
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Solution not given");
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 4;
  AppCtx         ctx;
  PetscScalar    *u;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  ctx.k1     = 1.0e-5;
  ctx.k2     = 1.0e5;
  ctx.k3     = 1.0e-16;
  ctx.sigma2 = 1.0e6;

  ierr = VecDuplicate(U,&ctx.initialsolution);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(ctx.initialsolution,&u);CHKERRQ(ierr);
  u[0] = 0.0;
  u[1] = 1.3e8;
  u[2] = 5.0e11;
  u[3] = 8.0e11;
  ierr = VecRestoreArrayWrite(ctx.initialsolution,&u);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Solution(ts,0,U,&ctx);CHKERRQ(ierr);
  ierr = TSSetTime(ts,4.0*3600);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,1.0);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,518400.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);
  ierr = TSSetMaxStepRejections(ts,100);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr); /* unlimited */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&ctx.initialsolution);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
     args: -ts_view -ts_max_time 2.e4
     timeoutfactor: 15
     requires: !single

TEST*/
