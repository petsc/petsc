
static char help[] = "Newton's method for a two-variable system, sequential.\n\n";

/*T
   Concepts: SNES^basic example
T*/

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
This examples solves either
\begin{equation}
  F\genfrac{(}{)}{0pt}{}{x_0}{x_1} = \genfrac{(}{)}{0pt}{}{x^2_0 + x_0 x_1 - 3}{x_0 x_1 + x^2_1 - 6}
\end{equation}
or if the {\tt -hard} options is given
\begin{equation}
  F\genfrac{(}{)}{0pt}{}{x_0}{x_1} = \genfrac{(}{)}{0pt}{}{\sin(3 x_0) + x_0}{x_1}
\end{equation}
F*/
#include <petscsnes.h>

/*
   User-defined routines
*/
extern PetscErrorCode FormJacobian1(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian2(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormFunction2(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  KSP            ksp;          /* linear solver context */
  PC             pc;           /* preconditioner context */
  Vec            x,r;          /* solution, residual vectors */
  Mat            J;            /* Jacobian matrix */
  PetscMPIInt    size;
  PetscScalar    pfive = .5,*xx;
  PetscBool      flg;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Example is only for sequential runs");

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
  PetscCall(VecSetSizes(x,PETSC_DECIDE,2));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&r));

  /*
     Create Jacobian matrix data structure
  */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-hard",&flg));
  if (!flg) {
    /*
     Set function evaluation routine and vector.
    */
    PetscCall(SNESSetFunction(snes,r,FormFunction1,NULL));

    /*
     Set Jacobian matrix data structure and Jacobian evaluation routine
    */
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian1,NULL));
  } else {
    PetscCall(SNESSetFunction(snes,r,FormFunction2,NULL));
    PetscCall(SNESSetJacobian(snes,J,J,FormJacobian2,NULL));
  }

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
  if (!flg) {
    PetscCall(VecSet(x,pfive));
  } else {
    PetscCall(VecGetArray(x,&xx));
    xx[0] = 2.0; xx[1] = 3.0;
    PetscCall(VecRestoreArray(x,&xx));
  }
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */

  PetscCall(SNESSolve(snes,NULL,x));
  if (flg) {
    Vec f;
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(SNESGetFunction(snes,&f,0,0));
    PetscCall(VecView(r,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(VecDestroy(&x)); PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J)); PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction1 - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x    - input vector
.  ctx  - optional user-defined context

   Output Parameter:
.  f - function vector
 */
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *ctx)
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
  ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
  ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;

  /* Restore vectors */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormJacobian1 - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0]  = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2]  = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  PetscCall(MatSetValues(B,2,idx,2,idx,A,INSERT_VALUES));

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

/* ------------------------------------------------------------------- */
PetscErrorCode FormFunction2(SNES snes,Vec x,Vec f,void *dummy)
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

  /*
     Compute function
  */
  ff[0] = PetscSinScalar(3.0*xx[0]) + xx[0];
  ff[1] = xx[1];

  /*
     Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  return 0;
}
/* ------------------------------------------------------------------- */
PetscErrorCode FormJacobian2(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[4];
  PetscInt          idx[2] = {0,1};

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0]  = 3.0*PetscCosScalar(3.0*xx[0]) + 1.0; A[1] = 0.0;
  A[2]  = 0.0;                     A[3] = 1.0;
  PetscCall(MatSetValues(B,2,idx,2,idx,A,INSERT_VALUES));

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
      args: -ksp_gmres_cgs_refinement_type refine_always -snes_monitor_short
      requires: !single

   test:
      suffix: 2
      requires: !single
      args:  -snes_monitor_short
      output_file: output/ex1_1.out

   test:
      suffix: 3
      args: -ksp_view_solution ascii:ex1_2_sol.tmp:ascii_matlab  -snes_monitor_short
      requires: !single
      output_file: output/ex1_1.out

   test:
      suffix: 4
      args: -ksp_view_solution ascii:ex1_2_sol.tmp::append  -snes_monitor_short
      requires: !single
      output_file: output/ex1_1.out

   test:
      suffix: 5
      args: -ksp_view_solution ascii:ex1_2_sol.tmp:ascii_matlab:append  -snes_monitor_short
      requires: !single
      output_file: output/ex1_1.out

   test:
      suffix: 6
      args: -ksp_view_solution ascii:ex1_2_sol.tmp:default:append  -snes_monitor_short
      requires: !single
      output_file: output/ex1_1.out

   test:
      suffix: X
      args: -ksp_monitor_short -ksp_type gmres -ksp_gmres_krylov_monitor -snes_monitor_short -snes_rtol 1.e-4
      requires: !single x

TEST*/
