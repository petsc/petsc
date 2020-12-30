static const char help[] = "Attempts to solve for root of a function with multiple local minima.\n\
With the proper initial guess, a backtracking line-search fails because Newton's method gets\n\
stuck in a local minimum. However, a critical point line-search or Newton's method without a\n\
line search succeeds.\n";

/* Solve 1D problem f(x) = 8 * exp(-4 * (x - 2)^2) * (x - 2) + 2 * x = 0

This problem is based on the example given here: https://scicomp.stackexchange.com/a/2446/24756
Originally an optimization problem to find the minimum of the function

g(x) = x^2 - exp(-4 * (x - 2)^2)

it has been reformulated to solve dg(x)/dx = f(x) = 0. The reformulated problem has several local
minima that can cause problems for some global Newton root-finding methods. In this particular
example, an initial guess of x0 = 2.5 generates an initial search direction (-df/dx is positive)
away from the root and towards a local minimum in which a back-tracking line search gets trapped.
However, omitting a line-search or using a critical point line search, the solve is successful.

The test outputs the final result for x and f(x).

Example usage:

Get help:
  ./ex99 -help

Monitor run (with default back-tracking line search; solve fails):
  ./ex99 -snes_converged_reason -snes_monitor -snes_linesearch_monitor -ksp_converged_reason -ksp_monitor

Run without a line search; solve succeeds:
  ./ex99 -snes_linesearch_type basic

Run with a critical point line search; solve succeeds:
  ./ex99 -snes_linesearch_type cp
*/

#include <math.h>
#include <petscsnes.h>

extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;          /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Example is only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,1);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  /*
     Create Jacobian matrix data structure
  */
  ierr = MatCreate(PETSC_COMM_WORLD,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSetUp(J);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,NULL);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set linear solver defaults for this problem. By extracting the
     KSP and PC contexts from the SNES context, we can then
     directly call any KSP and PC routines to set various options.
  */
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20);CHKERRQ(ierr);

  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecSet(x,2.5);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Output x and f(x)
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(r,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(&x);CHKERRQ(ierr); ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr); ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *xx;
  PetscScalar       *ff;

  /*
   Get pointers to vector data.
      - For default PETSc vectors, VecGetArray() returns a pointer to
        the data array.  Otherwise, the routine is implementation dependent.
      - You MUST call VecRestoreArray() when you no longer need access to
        the array.
   */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);

  /* Compute function */
  ff[0] = 8. * PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)) * (xx[0] - 2.) + 2. * xx[0];

  /* Restore vectors */
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[1];
  PetscErrorCode    ierr;
  PetscInt          idx[1] = {0};

  /*
     Get pointer to vector data
  */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0]  = 8. * ((xx[0] - 2.) * (PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)) * -8. * (xx[0] - 2.))
                + PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)))
          + 2.;

  ierr  = MatSetValues(B,1,idx,1,idx,A,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Restore vector
  */
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (jac != B) {
    ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -snes_linesearch_type cp
   test:
      suffix: 2
      args: -snes_linesearch_type basic
   test:
      suffix: 3

TEST*/
