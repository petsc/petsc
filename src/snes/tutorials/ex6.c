
static char help[] = "Newton method to solve u'' + u^{2} = f, sequentially.\n\
This example employs a user-defined reasonview routine.\n\n";

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
  PetscInt       its,n = 5,i;
  PetscMPIInt    size;
  PetscScalar    h,xp,v;
  MPI_Comm       comm;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  h    = 1.0/(n-1);
  comm = PETSC_COMM_WORLD;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(SNESCreate(comm,&snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note that we form 1 vector from scratch and then duplicate as needed.
  */
  PetscCall(VecCreate(comm,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x,&r));
  PetscCall(VecDuplicate(x,&F));
  PetscCall(VecDuplicate(x,&U));

  /*
     Set function evaluation routine and vector
  */
  PetscCall(SNESSetFunction(snes,r,FormFunction,(void*)F));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(MatCreate(comm,&J));
  PetscCall(MatSetSizes(J,PETSC_DECIDE,PETSC_DECIDE,n,n));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSeqAIJSetPreallocation(J,3,NULL));

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

  PetscCall(SNESSetJacobian(snes,J,J,FormJacobian,NULL));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set an optional user-defined reasonview routine
  */
  PetscCall(PetscViewerASCIIGetStdout(comm,&monP.viewer));
  /* Just make sure we can not repeat addding the same function
   * PETSc will be able to igore the repeated function
   */
  for (i=0; i<4; i++) {
    PetscCall(SNESConvergedReasonViewSet(snes,MySNESConvergedReasonView,&monP,0));
  }
  PetscCall(SNESGetKSP(snes,&ksp));
  /* Just make sure we can not repeat addding the same function
   * PETSc will be able to igore the repeated function
   */
  for (i=0; i<4; i++) {
    PetscCall(KSPConvergedReasonViewSet(ksp,MyKSPConvergedReasonView,&monP,0));
  }
  /*
     Set SNES/KSP/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  PetscCall(SNESSetFromOptions(snes));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  xp = 0.0;
  for (i=0; i<n; i++) {
    v    = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    PetscCall(VecSetValues(F,1,&i,&v,INSERT_VALUES));
    v    = xp*xp*xp;
    PetscCall(VecSetValues(U,1,&i,&v,INSERT_VALUES));
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
  PetscCall(FormInitialGuess(x));
  PetscCall(SNESSolve(snes,NULL,x));
  PetscCall(SNESGetIterationNumber(snes,&its));

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  PetscCall(VecDestroy(&x));  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&U));  PetscCall(VecDestroy(&F));
  PetscCall(MatDestroy(&J));  PetscCall(SNESDestroy(&snes));
  /*PetscCall(PetscViewerDestroy(&monP.viewer));*/
  PetscCall(PetscFinalize());
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode FormInitialGuess(Vec x)
{
  PetscScalar    pfive = .50;
  PetscCall(VecSet(x,pfive));
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
  PetscInt          i,n;

  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  PetscCall(VecGetArrayRead(x,&xx));
  PetscCall(VecGetArray(f,&ff));
  PetscCall(VecGetArrayRead(g,&gg));

  /*
     Compute function
  */
  PetscCall(VecGetSize(x,&n));
  d     = (PetscReal)(n - 1); d = d*d;
  ff[0] = xx[0];
  for (i=1; i<n-1; i++) ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - gg[i];
  ff[n-1] = xx[n-1] - 1.0;

  /*
     Restore vectors
  */
  PetscCall(VecRestoreArrayRead(x,&xx));
  PetscCall(VecRestoreArray(f,&ff));
  PetscCall(VecRestoreArrayRead(g,&gg));
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
  PetscInt          i,n,j[3];

  /*
     Get pointer to vector data
  */
  PetscCall(VecGetArrayRead(x,&xx));

  /*
     Compute Jacobian entries and insert into matrix.
      - Note that in this case we set all elements for a particular
        row at once.
  */
  PetscCall(VecGetSize(x,&n));
  d    = (PetscReal)(n - 1); d = d*d;

  /*
     Interior grid points
  */
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1;
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];
    PetscCall(MatSetValues(B,1,&i,3,j,A,INSERT_VALUES));
  }

  /*
     Boundary points
  */
  i = 0;   A[0] = 1.0;

  PetscCall(MatSetValues(B,1,&i,1,&i,A,INSERT_VALUES));

  i = n-1; A[0] = 1.0;

  PetscCall(MatSetValues(B,1,&i,1,&i,A,INSERT_VALUES));

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

PetscErrorCode MySNESConvergedReasonView(SNES snes,void *ctx)
{
  ReasonViewCtx         *monP = (ReasonViewCtx*) ctx;
  PetscViewer           viewer = monP->viewer;
  SNESConvergedReason   reason;
  const char            *strreason;

  PetscCall(SNESGetConvergedReason(snes,&reason));
  PetscCall(SNESGetConvergedReasonString(snes,&strreason));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Customized SNES converged reason view\n"));
  PetscCall(PetscViewerASCIIAddTab(viewer,1));
  if (reason > 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Converged due to %s\n",strreason));
  } else if (reason <= 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Did not converge due to %s\n",strreason));
  }
  PetscCall(PetscViewerASCIISubtractTab(viewer,1));
  return 0;
}

PetscErrorCode MyKSPConvergedReasonView(KSP ksp,void *ctx)
{
  ReasonViewCtx         *monP = (ReasonViewCtx*) ctx;
  PetscViewer           viewer = monP->viewer;
  KSPConvergedReason    reason;
  const char            *reasonstr;

  PetscCall(KSPGetConvergedReason(ksp,&reason));
  PetscCall(KSPGetConvergedReasonString(ksp,&reasonstr));
  PetscCall(PetscViewerASCIIAddTab(viewer,2));
  PetscCall(PetscViewerASCIIPrintf(viewer,"Customized KSP converged reason view\n"));
  PetscCall(PetscViewerASCIIAddTab(viewer,1));
  if (reason > 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Converged due to %s\n",reasonstr));
  } else if (reason <= 0) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Did not converge due to %s\n",reasonstr));
  }
  PetscCall(PetscViewerASCIISubtractTab(viewer,3));
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      filter: sed -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g"

   test:
      suffix: 2
      nsize: 1
      args: -ksp_converged_reason_view_cancel
      filter: sed -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g"

   test:
      suffix: 3
      nsize: 1
      args: -ksp_converged_reason_view_cancel -ksp_converged_reason
      filter: sed -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g"

   test:
      suffix: 4
      nsize: 1
      args: -snes_converged_reason_view_cancel
      filter: sed -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g"

   test:
      suffix: 5
      nsize: 1
      args: -snes_converged_reason_view_cancel -snes_converged_reason
      filter: sed -e "s/CONVERGED_ATOL/CONVERGED_RTOL/g"

TEST*/
