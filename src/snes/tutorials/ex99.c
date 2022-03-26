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
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size > 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Example is only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,1));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&r));

  /*
     Create Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,1,1));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESSetFunction(snes,r,FormFunction,NULL));
  PetscCall(SNESSetJacobian(snes,J,J,FormJacobian,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set linear solver defaults for this problem. By extracting the
     KSP and PC contexts from the SNES context, we can then
     directly call any KSP and PC routines to set various options.
  */
  PetscCall(SNESGetKSP(snes,&ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCNONE));
  PetscCall(KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20));

  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecSet(x,2.5));

  PetscCall(SNESSolve(snes,NULL,x));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Output x and f(x)
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(r,PETSC_VIEWER_STDOUT_WORLD));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  /*
   Get pointers to vector data.
      - For default PETSc vectors, VecGetArray() returns a pointer to
        the data array.  Otherwise, the routine is implementation dependent.
      - You MUST call VecRestoreArray() when you no longer need access to
        the array.
   */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(f,&ff));

  /* Compute function */
  ff[0] = 8. * PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)) * (xx[0] - 2.) + 2. * xx[0];

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  return 0;
}

PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[1];
  PetscInt          idx[1] = {0};

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0]  = 8. * ((xx[0] - 2.) * (PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)) * -8. * (xx[0] - 2.))
                + PetscExpScalar(-4. * (xx[0] - 2.) * (xx[0] - 2.)))
          + 2.;

  PetscCall(MatSetValues(B,1,idx,1,idx,A,INSERT_VALUES));

  /*
     Restore vector
  */
  PetscCall(VecRestoreArrayRead(x,&xx));

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
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
   test:
      suffix: 4
      args: -snes_type newtontrdc
   test:
      suffix: 5
      args: -snes_type newtontrdc -snes_trdc_use_cauchy false

TEST*/
