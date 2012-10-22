
static char help[] = "Reaction Equation from Chemistry\n";

/*

     Page 6, An example from Atomospheric Chemistry

                 u_1_t =
                 u_2_t =
                 u_3_t =
                 u_4_t =
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
  Vec         initialdata;
} AppCtx;

PetscScalar k1(AppCtx *ctx,PetscReal t)
{
  PetscReal th = t/3600.0;
  PetscReal barth = th - 24.0*floor(th/24.0);
  if (((((PetscInt)th) % 24) < 4)               || ((((PetscInt)th) % 24) >= 20)) return(1.0e-40); 
  else return(ctx->k1*PetscExpScalar(7.0*PetscPowScalar(PetscSinScalar(.0625*PETSC_PI*(barth - 4.0)),.2)));
}

#undef __FUNCT__
#define __FUNCT__ "Function"
static PetscErrorCode Function(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,AppCtx *ctx)
{
  PetscErrorCode ierr;
  PetscScalar    *x,*xdot,*f;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - k1(ctx,t)*x[2] + ctx->k2*x[0];
  f[1] = xdot[1] - k1(ctx,t)*x[2] + ctx->k3*x[1]*x[3] - ctx->sigma2;
  f[2] = xdot[2] - ctx->k3*x[1]*x[3] + k1(ctx,t)*x[2];
  f[3] = xdot[3] - ctx->k2*x[0] + ctx->k3*x[1]*x[3];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Jacobian"
static PetscErrorCode Jacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat *A,Mat *B,MatStructure *flag,AppCtx *ctx)
{
  PetscErrorCode ierr;
  PetscInt       rowcol[] = {0,1,2,3};
  PetscScalar    *x,*xdot,J[4][4];

  PetscFunctionBegin;
  ierr    = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + ctx->k2;   J[0][1] = 0.0;                J[0][2] = -k1(ctx,t);       J[0][3] = 0.0;
  J[1][0] = 0.0;           J[1][1] = a + ctx->k3*x[3];   J[1][2] = -k1(ctx,t);       J[1][3] = ctx->k3*x[1];
  J[2][0] = 0.0;           J[2][1] = - ctx->k3*x[3];     J[2][2] = a + k1(ctx,t);    J[2][3] =  - ctx->k3*x[1];
  J[3][0] =  - ctx->k2;    J[3][1] = ctx->k3*x[3];       J[3][2] = 0.0;              J[3][3] = a + ctx->k3*x[1];
  ierr    = MatSetValues(*B,4,rowcol,4,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr    = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*A != *B) {
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Solution"
static PetscErrorCode Solution(TS ts,PetscReal t,Vec X,AppCtx *ctx)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecCopy(ctx->initialdata,X);
  if (t > 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Solution not given");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x,r;           /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  PetscInt       steps,maxsteps = 1000000000,nonlinits,linits,snesfails,rejects;
  PetscReal      ftime;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 4;
  AppCtx         ctx;
  PetscScalar    *xx;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatGetVecs(A,&x,PETSC_NULL);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  ctx.k1       = 1.0e-5;
  ctx.k2       = 1.0e5;
  ctx.k3       = 1.0e-16;
  ctx.sigma2   = 1.0e6;

  ierr = VecDuplicate(x,&ctx.initialdata);CHKERRQ(ierr);
  ierr = VecGetArray(ctx.initialdata,&xx);CHKERRQ(ierr);
  xx[0] = 0.0;
  xx[1] = 1.3e8;
  xx[2] = 5.0e11;
  xx[3] = 8.0e11;
  ierr = VecRestoreArray(ctx.initialdata,&xx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr); 
  ierr = TSSetIFunction(ts,PETSC_NULL,(TSIFunction) Function,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)Jacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,518400.0);CHKERRQ(ierr);
  ierr = TSSetMaxStepRejections(ts,100);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr); /* unlimited */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Solution(ts,0,x,&ctx);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,4.0*3600,1.0);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetSNESFailures(ts,&snesfails);CHKERRQ(ierr);
  ierr = TSGetStepRejections(ts,&rejects);CHKERRQ(ierr);
  ierr = TSGetSNESIterations(ts,&nonlinits);CHKERRQ(ierr);
  ierr = TSGetKSPIterations(ts,&linits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"steps %D (%D rejected, %D SNES fails), ftime %G, nonlinits %D, linits %D\n",steps,rejects,snesfails,ftime,nonlinits,linits);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&ctx.initialdata);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
