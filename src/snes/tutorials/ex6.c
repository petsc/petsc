
static char help[] = "Newton method to solve u'' + u^{2} = f, sequentially.\n\
This example employs a user-defined reasonview routine.\n\n";

/*T
   Concepts: SNES^basic uniprocessor example
   Concepts: SNES^setting a user-defined reasonview routine
   Processors: 1
T*/

#include <petscsnes.h>

/*
   User-defined routines
*/
PETSC_EXTERN PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
PETSC_EXTERN PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PETSC_EXTERN PetscErrorCode FormInitialGuess(Vec);
PETSC_EXTERN PetscErrorCode MySNESConvergedReasonView(SNES,void*);
PETSC_EXTERN PetscErrorCode MyKSPConvergedReasonView(KSP,void*);

/*
   User-defined context for monitoring
*/
typedef struct {
  PetscViewer viewer;
} ReasonViewCtx;

int main(int argc,char **argv)
{
  SNES           snes;                /* SNES context */
  KSP            ksp;                 /* KSP context */
  Vec            x,r,F,U;             /* vectors */
  Mat            J;                   /* Jacobian matrix */
  ReasonViewCtx     monP;             /* monitoring context */
  PetscErrorCode ierr;
  PetscInt       its,n = 5,i;
  PetscMPIInt    size;
  PetscScalar    h,xp,v;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  h    = 1.0/(n-1);
  comm = PETSC_COMM_WORLD;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note that we form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(comm,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&U);CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(comm,&J);CHKERRQ(ierr);
  ierr = MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(J,3,NULL);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_fd : default finite differencing approximation of Jacobian
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner)
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method
  */

  ierr = SNESSetJacobian(snes,J,J,FormJacobian,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set an optional user-defined reasonview routine
  */
  ierr = PetscViewerASCIIGetStdout(comm,&monP.viewer);CHKERRQ(ierr);
  /* Just make sure we can not repeat addding the same function
   * PETSc will be able to igore the repeated function
   */
  for (i=0; i<4; i++){
    ierr = SNESConvergedReasonViewSet(snes,MySNESConvergedReasonView,&monP,0);CHKERRQ(ierr);
  }
  ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
  /* Just make sure we can not repeat addding the same function
   * PETSc will be able to igore the repeated function
   */
  for (i=0; i<4; i++){
    ierr = KSPConvergedReasonViewSet(ksp,MyKSPConvergedReasonView,&monP,0);CHKERRQ(ierr);
  }
  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  xp = 0.0;
  for (i=0; i<n; i++) {
    v    = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v    = xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    xp  += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  /*ierr = PetscViewerDestroy(&monP.viewer);CHKERRQ(ierr);*/
  ierr = PetscFinalize();
  return ierr;
}
/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode FormInitialGuess(Vec x)
{
  PetscErrorCode ierr;
  PetscScalar    pfive = .50;
  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific data
   needed for the function evaluation (such as various parameters, work
   vectors, and grid information).  In this program the context is just
   a vector containing the right-hand-side of the discretized PDE.
 */

PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  Vec               g = (Vec)ctx;
  const PetscScalar *xx,*gg;
  PetscScalar       *ff,d;
  PetscErrorCode    ierr;
  PetscInt          i,n;

  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArrayRead(g,&gg);CHKERRQ(ierr);

  /*
     Compute function
  */
  ierr  = VecGetSize(x,&n);CHKERRQ(ierr);
  d     = (PetscReal)(n - 1); d = d*d;
  ff[0] = xx[0];
  for (i=1; i<n-1; i++) ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - gg[i];
  ff[n-1] = xx[n-1] - 1.0;

  /*
     Restore vectors
  */
  ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(g,&gg);CHKERRQ(ierr);
  return 0;
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

*/

PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *dummy)
{
  const PetscScalar *xx;
  PetscScalar       A[3],d;
  PetscErrorCode    ierr;
  PetscInt          i,n,j[3];

  /*
     Get pointer to vector data
  */
  ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Note that in this case we set all elements for a particular
        row at once.
  */
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d    = (PetscReal)(n - 1); d = d*d;

  /*
     Interior grid points
  */
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1;
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(B,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  i = 0;   A[0] = 1.0;

  ierr = MatSetValues(B,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);

  i = n-1; A[0] = 1.0;

  ierr = MatSetValues(B,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);

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

PetscErrorCode MySNESConvergedReasonView(SNES snes,void *ctx)
{
  PetscErrorCode        ierr;
  ReasonViewCtx         *monP = (ReasonViewCtx*) ctx;
  PetscViewer           viewer = monP->viewer;
  SNESConvergedReason   reason;
  const char            *strreason;

  ierr = SNESGetConvergedReason(snes,&reason);CHKERRQ(ierr);
  ierr = SNESGetConvergedReasonString(snes,&strreason);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Customized SNES converged reason view\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,1);CHKERRQ(ierr);
  if (reason > 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Converged due to %s\n",strreason);CHKERRQ(ierr);
  } else if (reason <= 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Did not converge due to %s\n",strreason);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,1);CHKERRQ(ierr);
  return 0;
}

PetscErrorCode MyKSPConvergedReasonView(KSP ksp,void *ctx)
{
  PetscErrorCode        ierr;
  ReasonViewCtx         *monP = (ReasonViewCtx*) ctx;
  PetscViewer           viewer = monP->viewer;
  KSPConvergedReason    reason;
  const char            *reasonstr;

  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  ierr = KSPGetConvergedReasonString(ksp,&reasonstr);CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,2);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"Customized KSP converged reason view\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIAddTab(viewer,1);CHKERRQ(ierr);
  if (reason > 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Converged due to %s\n",reasonstr);CHKERRQ(ierr);
  } else if (reason <= 0) {
    ierr = PetscViewerASCIIPrintf(viewer,"Did not converge due to %s\n",reasonstr);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIISubtractTab(viewer,3);CHKERRQ(ierr);
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1

   test:
      suffix: 2
      nsize: 1
      args: -ksp_converged_reason_view_cancel

   test:
      suffix: 3
      nsize: 1
      args: -ksp_converged_reason_view_cancel -ksp_converged_reason

   test:
      suffix: 4
      nsize: 1
      args: -snes_converged_reason_view_cancel

   test:
      suffix: 5
      nsize: 1
      args: -snes_converged_reason_view_cancel -snes_converged_reason

TEST*/
