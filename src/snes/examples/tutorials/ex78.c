
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
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,N,1,1,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,da);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateMatrix(da,&J);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&constants);CHKERRQ(ierr);
  ierr = MatSetNullSpace(J,constants);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,da);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&constants);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  Vec            xlocal;

  PetscFunctionBeginUser;
  /* Get local work vector */
  ierr = DMGetLocalVector(da,&xlocal);CHKERRQ(ierr);

  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  /*
     Get pointers to vector data.
       - The vector xlocal includes ghost point; the vectors x and f do
         NOT include ghost points.
       - Using DMDAVecGetArray() allows accessing the values using global ordering
  */
  ierr = DMDAVecGetArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,f,&ff);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Compute function over locally owned part of the grid
     Note the [i-1] and [i+1] will automatically access the ghost points from other processes or the periodic points.
  */
  h = 1.0/M;
  for (i=xs; i<xs+xm; i++) ff[i] = (xx[i-1] - 2.0*xx[i] + xx[i+1])/(h*h)  - PetscSinReal(2.0*PETSC_PI*i*h);

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,f,&ff);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  DM             da = (DM) ctx;
  MatStencil     row,cols[3];
  PetscReal      h;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  ierr = MatZeroEntries(jac);CHKERRQ(ierr);
  h = 1.0/M;
  /* because of periodic boundary conditions we can simply loop over all local nodes and access to the left and right */
  for (i=xs; i<xs+xm; i++) {
    row.i = i;
    cols[0].i = i - 1;
    cols[1].i = i;
    cols[2].i = i + 1;
    A[0] = A[2] = 1.0/(h*h); A[1] = -2.0/(h*h);
    ierr = MatSetValuesStencil(jac,1,&row,3,cols,A,ADD_VALUES);CHKERRQ(ierr);
  }

  ierr = DMDAVecRestoreArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*TEST

   test:
      args: -snes_monitor_short -ksp_monitor_short -pc_type sor -snes_converged_reason -da_refine 3
      requires: !single

TEST*/
