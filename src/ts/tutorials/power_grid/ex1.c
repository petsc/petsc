
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
  PetscCall(PetscRandomGetValue(ctx->rand,&r));
  if (r > .9) R = .5;
  if (r < .1) R = .4;
  R = .4;
  /*  The next three lines allow us to access the entries of the vectors directly */
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));
  f[0] = 2.0*ctx->H*udot[0]/ctx->omega_s + ctx->E*ctx->V*PetscSinScalar(u[1])/ctx->X - R;
  f[1] = udot[1] - u[0] + ctx->omega_s;

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
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
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  J[0][0] = 2.0*ctx->H*a/ctx->omega_s;   J[0][1] = -ctx->E*ctx->V*PetscCosScalar(u[1])/ctx->X;
  J[1][0] = -1.0;                        J[1][1] = a;
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

int main(int argc,char **argv)
{
  TS             ts;            /* ODE integrator */
  Vec            U;             /* solution will be stored here */
  Mat            A;             /* Jacobian matrix */
  PetscMPIInt    size;
  PetscInt       n = 2;
  AppCtx         ctx;
  PetscScalar    *u;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));

  PetscCall(MatCreateVecs(A,&U,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Reaction options","");
  {
    ctx.omega_s = 1.0;
    PetscCall(PetscOptionsScalar("-omega_s","","",ctx.omega_s,&ctx.omega_s,NULL));
    ctx.H       = 1.0;
    PetscCall(PetscOptionsScalar("-H","","",ctx.H,&ctx.H,NULL));
    ctx.E       = 1.0;
    PetscCall(PetscOptionsScalar("-E","","",ctx.E,&ctx.E,NULL));
    ctx.V       = 1.0;
    PetscCall(PetscOptionsScalar("-V","","",ctx.V,&ctx.V,NULL));
    ctx.X       = 1.0;
    PetscCall(PetscOptionsScalar("-X","","",ctx.X,&ctx.X,NULL));

    PetscCall(VecGetArray(U,&u));
    u[0] = 1;
    u[1] = .7;
    PetscCall(VecRestoreArray(U,&u));
    PetscCall(PetscOptionsGetVec(NULL,NULL,"-initial",U,NULL));
  }
  PetscOptionsEnd();

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&ctx.rand));
  PetscCall(PetscRandomSetFromOptions(ctx.rand));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetProblemType(ts,TS_NONLINEAR));
  PetscCall(TSSetType(ts,TSROSW));
  PetscCall(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&ctx));
  PetscCall(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&ctx));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetSolution(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetMaxTime(ts,2000.0));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSSetTimeStep(ts,.001));
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,U));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscRandomDestroy(&ctx.rand));
  PetscCall(PetscFinalize());
  return 0;
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
