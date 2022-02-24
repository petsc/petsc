
static char help[] = "Newton methods to solve u''  = f in parallel with periodic boundary conditions.\n\n";

/*T
   Concepts: SNES^basic parallel example
   Concepts: periodic boundary conditions
   Processors: n
T*/

/*
   Compare this example to ex3.c that handles Dirichlet boundary conditions

   Though this is a linear problem it is treated as a nonlinear problem in this example to demonstrate
   handling periodic boundary conditions for nonlinear problems.

   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  SNES           snes;                 /* SNES context */
  Mat            J;                    /* Jacobian matrix */
  DM             da;
  Vec            x,r;              /* vectors */
  PetscErrorCode ierr;
  PetscInt       N = 5;
  MatNullSpace   constants;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,N,1,1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(VecDuplicate(x,&r));

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  CHKERRQ(SNESSetFunction(snes,r,FormFunction,da));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(DMCreateMatrix(da,&J));
  CHKERRQ(MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&constants));
  CHKERRQ(MatSetNullSpace(J,constants));
  CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian,da));

  CHKERRQ(SNESSetFromOptions(snes));
  CHKERRQ(SNESSolve(snes,NULL,x));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(MatNullSpaceDestroy(&constants));
  CHKERRQ(SNESDestroy(&snes));
  CHKERRQ(DMDestroy(&da));
  ierr = PetscFinalize();
  return ierr;
}

/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific
   data needed for the function evaluation.
*/
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  DM             da    = (DM) ctx;
  PetscScalar    *xx,*ff;
  PetscReal      h;
  PetscInt       i,M,xs,xm;
  Vec            xlocal;

  PetscFunctionBeginUser;
  /* Get local work vector */
  CHKERRQ(DMGetLocalVector(da,&xlocal));

  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  CHKERRQ(DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal));
  CHKERRQ(DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal));

  /*
     Get pointers to vector data.
       - The vector xlocal includes ghost point; the vectors x and f do
         NOT include ghost points.
       - Using DMDAVecGetArray() allows accessing the values using global ordering
  */
  CHKERRQ(DMDAVecGetArray(da,xlocal,&xx));
  CHKERRQ(DMDAVecGetArray(da,f,&ff));

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));
  CHKERRQ(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  /*
     Compute function over locally owned part of the grid
     Note the [i-1] and [i+1] will automatically access the ghost points from other processes or the periodic points.
  */
  h = 1.0/M;
  for (i=xs; i<xs+xm; i++) ff[i] = (xx[i-1] - 2.0*xx[i] + xx[i+1])/(h*h)  - PetscSinReal(2.0*PETSC_PI*i*h);

  /*
     Restore vectors
  */
  CHKERRQ(DMDAVecRestoreArray(da,xlocal,&xx));
  CHKERRQ(DMDAVecRestoreArray(da,f,&ff));
  CHKERRQ(DMRestoreLocalVector(da,&xlocal));
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx)
{
  PetscScalar    *xx,A[3];
  PetscInt       i,M,xs,xm;
  DM             da = (DM) ctx;
  MatStencil     row,cols[3];
  PetscReal      h;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  CHKERRQ(DMDAVecGetArrayRead(da,x,&xx));
  CHKERRQ(DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL));

  /*
    Get range of locally owned matrix
  */
  CHKERRQ(DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL));

  CHKERRQ(MatZeroEntries(jac));
  h = 1.0/M;
  /* because of periodic boundary conditions we can simply loop over all local nodes and access to the left and right */
  for (i=xs; i<xs+xm; i++) {
    row.i = i;
    cols[0].i = i - 1;
    cols[1].i = i;
    cols[2].i = i + 1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    CHKERRQ(MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES));
  }

  CHKERRQ(DMDAVecRestoreArrayRead(da,x,&xx));
  CHKERRQ(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -snes_monitor_short -ksp_monitor_short -pc_type sor -snes_converged_reason -da_refine 3
      requires: !single

   test:
      suffix: 2
      args: -snes_monitor_short -ksp_monitor_short -pc_type sor -snes_converged_reason -da_refine 3 -snes_type newtontrdc
      requires: !single

   test:
      suffix: 3
      args: -snes_monitor_short -ksp_monitor_short -pc_type sor -snes_converged_reason -da_refine 3 -snes_type newtontrdc -snes_trdc_use_cauchy false
      requires: !single

TEST*/
