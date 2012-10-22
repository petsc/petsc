
static char help[] = "Nonlinear Reaction Problems from Chemistry.\n";

/*

     This directory contains examples based on the PDES/ODES given in the book
      Numerical Solution of Time-Dependent Advection-Diffusion-Reaction Equations by
      W. Hundsdorf and J.G. Verwer

     Page 3, Section 1.1 Nonlinear Reaction Problems from Chemistry
                 u_1_t =  -k u_1 u_2
                 u_2_t =  -k u_1 u_2
                 u_3_t =   k u_1 u_2
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
  PetscScalar k;
  Vec         initialdata;
} AppCtx;

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
  f[0] = xdot[0] + ctx->k*x[0]*x[1];
  f[1] = xdot[1] + ctx->k*x[0]*x[1];
  f[2] = xdot[2] - ctx->k*x[0]*x[1];
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
  PetscInt       rowcol[] = {0,1,2};
  PetscScalar    *x,*xdot,J[3][3];

  PetscFunctionBegin;
  ierr    = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr    = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  J[0][0] = a + ctx->k*x[1];   J[0][1] = ctx->k*x[0];       J[0][2] = 0.0;
  J[1][0] = ctx->k*x[1];       J[1][1] = a + ctx->k*x[0];   J[1][2] = 0.0;
  J[2][0] = ctx->k*x[1];       J[2][1] = ctx->k*x[0];       J[2][2] = a;
  ierr    = MatSetValues(*B,3,rowcol,3,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
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
  const PetscScalar *xinit;
  PetscScalar       *x,d0,q;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(ctx->initialdata,&xinit);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  d0 = xinit[0] - xinit[1];
  if (d0 == 0.0) q = ctx->k*t;
  else q = (1.0 - PetscExpScalar(-ctx->k*t*d0))/d0;
  x[0] = xinit[0]/(1.0 + xinit[1]*q);
  x[1] = x[0] - d0;
  x[2] = xinit[1] + xinit[2] - x[1];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->initialdata,&xinit);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x,r;           /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  PetscInt       steps,maxsteps = 1000,nonlinits,linits,snesfails,rejects;
  PetscReal      ftime;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 3;
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Reaction options","");CHKERRQ(ierr);
  {
    ctx.k       = .9;
    ierr        = PetscOptionsReal("-k","Reaction coefficient","",ctx.k,&ctx.k,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(x,&ctx.initialdata);CHKERRQ(ierr);
    ierr = VecGetArray(ctx.initialdata,&xx);CHKERRQ(ierr);
    xx[0] = 1;
    xx[1] = 1;
    xx[2] = 0;
    ierr = VecRestoreArray(ctx.initialdata,&xx);CHKERRQ(ierr);
    ierr = PetscOptionsVec("-initial","Initial values","",ctx.initialdata,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSROSW);CHKERRQ(ierr); 
  ierr = TSSetIFunction(ts,PETSC_NULL,(TSIFunction) Function,&ctx);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)Jacobian,&ctx);CHKERRQ(ierr);
  ierr = TSSetSolutionFunction(ts,(TSSolutionFunction)Solution,&ctx);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxsteps,20.0);CHKERRQ(ierr);
  ierr = TSSetMaxStepRejections(ts,10);CHKERRQ(ierr);
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr); /* unlimited */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Solution(ts,0,x,&ctx);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
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
