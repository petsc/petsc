
static char help[] = "Tests SNESLinesearch handling of Inf/Nan.\n\n";

/*
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
/*F
This examples solves
\begin{equation}
  F\genfrac{(}{)}{0pt}{}{x_0}{x_1} = \genfrac{(}{)}{0pt}{}{\sin(3 x_0) + x_0}{x_1}
\end{equation}
F*/
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormJacobian2(SNES, Vec, Mat, Mat, void *);
extern PetscErrorCode FormFunction2(SNES, Vec, Vec, void *);
extern PetscErrorCode FormObjective(SNES, Vec, PetscReal *, void *);

/*
     This is a very hacking way to trigger the objective function generating an infinity at a particular count to the call FormObjective().
     Different line searches evaluate the full step at different counts. For l2 it is the third call (infatcount == 2) while for bt it is the second call.
*/
PetscInt infatcount = 0;

int main(int argc, char **argv)
{
  SNES         snes; /* nonlinear solver context */
  KSP          ksp;  /* linear solver context */
  PC           pc;   /* preconditioner context */
  Vec          x, r; /* solution, residual vectors */
  Mat          J;    /* Jacobian matrix */
  PetscInt     its;
  PetscMPIInt  size;
  PetscScalar *xx;
  PetscBool    flg;
  char         type[256];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-snes_linesearch_type", type, sizeof(type), &flg));
  if (flg) {
    PetscCall(PetscStrcmp(type, SNESLINESEARCHBT, &flg));
    if (flg) infatcount = 1;
    PetscCall(PetscStrcmp(type, SNESLINESEARCHL2, &flg));
    if (flg) infatcount = 2;
  }

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Example is only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Create vectors for solution and nonlinear function
  */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 2));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));

  /*
     Create Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 2, 2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESSetFunction(snes, r, FormFunction2, NULL));
  PetscCall(SNESSetObjective(snes, FormObjective, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian2, NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Set linear solver defaults for this problem. By extracting the
     KSP and PC contexts from the SNES context, we can then
     directly call any KSP and PC routines to set various options.
  */
  PetscCall(SNESGetKSP(snes, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetTolerances(ksp, 1.e-4, PETSC_DEFAULT, PETSC_DEFAULT, 20));

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
  PetscCall(VecGetArray(x, &xx));
  xx[0] = 2.0;
  xx[1] = 3.0;
  PetscCall(VecRestoreArray(x, &xx));

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  PetscCall(SNESSolve(snes, NULL, x));
  PetscCall(SNESGetIterationNumber(snes, &its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Number of SNES iterations = %" PetscInt_FMT "\n", its));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

PetscErrorCode FormObjective(SNES snes, Vec x, PetscReal *f, void *dummy)
{
  Vec             F;
  static PetscInt cnt = 0;
  const PetscReal one = 1.0, zero = 0.0, inf = one / zero;

  PetscFunctionBeginUser;
  if (cnt++ == infatcount) *f = inf;
  else {
    PetscCall(VecDuplicate(x, &F));
    PetscCall(FormFunction2(snes, x, F, dummy));
    PetscCall(VecNorm(F, NORM_2, f));
    PetscCall(VecDestroy(&F));
    *f = (*f) * (*f);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormFunction2(SNES snes, Vec x, Vec f, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));

  /*
     Compute function
  */
  ff[0] = PetscSinScalar(3.0 * xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
     Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian2(SNES snes, Vec x, Mat jac, Mat B, void *dummy)
{
  const PetscScalar *xx;
  PetscScalar        A[4];
  PetscInt           idx[2] = {0, 1};

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x, &xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0] = 3.0 * PetscCosScalar(3.0 * xx[0]) + 1.0;
  A[1] = 0.0;
  A[2] = 0.0;
  A[3] = 1.0;
  PetscCall(MatSetValues(B, 2, idx, 2, idx, A, INSERT_VALUES));

  /*
     Restore vector
  */
  PetscCall(VecRestoreArrayRead(x, &xx));

  /*
     Assemble matrix
  */
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*TEST

   test:
      args: -snes_converged_reason -snes_linesearch_monitor -snes_linesearch_type l2
      filter: grep Inf

TEST*/
