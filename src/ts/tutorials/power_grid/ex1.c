
static char help[] = "Basic equation for generator stability analysis.\n";

/*F

\begin{eqnarray}
                 \frac{2 H}{\omega_s}\frac{d \omega}{dt} & = & P_m - \frac{EV}{X} \sin(\theta) \\
                 \frac{d \theta}{dt} = \omega - \omega_s
\end{eqnarray}

F*/

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
  PetscScalar H,omega_s,E,V,X;
  PetscRandom rand;
} AppCtx;

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *ctx)
{
  PetscScalar        *f,r;
  const PetscScalar  *u,*udot;
  static PetscScalar R = .4;

  PetscFunctionBegin;
  CHKERRQ(PetscRandomGetValue(ctx->rand,&r));
  if (r > .9) R = .5;
  if (r < .1) R = .4;
  R = .4;
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = 2.0*ctx->H*udot[0]/ctx->omega_s + ctx->E*ctx->V*PetscSinScalar(u[1])/ctx->X - R;
  f[1] = udot[1] - u[0] + ctx->omega_s;

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

/*
     Defines the Jacobian of the ODE passed to the ODE solver. See TSSetIJacobian() for the meaning of a and the Jacobian.
*/
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,AppCtx *ctx)
{
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u,*udot;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  J[0][0] = 2.0*ctx->H*a/ctx->omega_s;   J[0][1] = -ctx->E*ctx->V*PetscCosScalar(u[1])/ctx->X;
  J[1][0] = -1.0;                        J[1][1] = a;
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

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 2;
  AppCtx         ctx;
  PetscScalar    *u;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Reaction options","");CHKERRQ(ierr);
  {
    ctx.omega_s = 1.0;
    CHKERRQ(PetscOptionsScalar("-omega_s","","",ctx.omega_s,&ctx.omega_s,NULL));
    ctx.H       = 1.0;
    CHKERRQ(PetscOptionsScalar("-H","","",ctx.H,&ctx.H,NULL));
    ctx.E       = 1.0;
    CHKERRQ(PetscOptionsScalar("-E","","",ctx.E,&ctx.E,NULL));
    ctx.V       = 1.0;
    CHKERRQ(PetscOptionsScalar("-V","","",ctx.V,&ctx.V,NULL));
    ctx.X       = 1.0;
    CHKERRQ(PetscOptionsScalar("-X","","",ctx.X,&ctx.X,NULL));

    CHKERRQ(VecGetArray(U,&u));
    u[0] = 1;
    u[1] = .7;
    CHKERRQ(VecRestoreArray(U,&u));
    CHKERRQ(PetscOptionsGetVec(NULL,NULL,"-initial",U,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&ctx.rand));
  CHKERRQ(PetscRandomSetFromOptions(ctx.rand));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSROSW));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&ctx));
  CHKERRQ(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,2000.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ts,.001));
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscRandomDestroy(&ctx.rand));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
     requires: !complex !single

   test:
      args: -ksp_gmres_cgs_refinement_type refine_always -snes_type newtonls -ts_max_steps 10

   test:
      suffix: 2
      args: -ts_max_steps 10

TEST*/
