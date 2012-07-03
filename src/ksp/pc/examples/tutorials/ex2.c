/*
 * Test file for the PCFactorSetShiftType() routine or -pc_factor_shift_type POSITIVE_DEFINITE option.
 * The test matrix is the example from Kershaw's paper [J.Comp.Phys 1978]
 * of a positive definite matrix for which ILU(0) will give a negative pivot.
 * This means that the CG method will break down; the Manteuffel shift
 * [Math. Comp. 1980] repairs this.
 *
 * Run the executable twice:
 * 1/ without options: the iterative method diverges because of an
 *    indefinite preconditioner
 * 2/ with -pc_factor_shift_positive_definite option (or comment in the PCFactorSetShiftType() line below):
 *    the method will now successfully converge.
 */

#include <stdlib.h>
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  KSP                ksp;
  PC                 pc;
  Mat                A,M;
  Vec                X,B,D;
  MPI_Comm           comm;
  PetscScalar        v; 
  KSPConvergedReason reason;
  PetscInt           i,j,its;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc,&argv,0,0);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue("-options_left",PETSC_NULL);CHKERRQ(ierr);
  comm = MPI_COMM_SELF;
  
  /*
   * Construct the Kershaw matrix
   * and a suitable rhs / initial guess
   */
  ierr = MatCreateSeqAIJ(comm,4,4,4,0,&A);CHKERRQ(ierr);
  ierr = VecCreateSeq(comm,4,&B);CHKERRQ(ierr);
  ierr = VecDuplicate(B,&X);CHKERRQ(ierr);
  for (i=0; i<4; i++) {
    v=3;
    ierr = MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    v=1;
    ierr = VecSetValues(B,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecSetValues(X,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }

  i=0; v=0;
  ierr = VecSetValues(X,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);

  for (i=0; i<3; i++) {
    v=-2; j=i+1;
    ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  }
  i=0; j=3; v=2;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(B);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(B);CHKERRQ(ierr);
  printf("\nThe Kershaw matrix:\n\n"); MatView(A,0);

  /*
   * A Conjugate Gradient method
   * with ILU(0) preconditioning
   */
  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);

  ierr = KSPSetType(ksp,KSPCG);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);CHKERRQ(ierr);

  /*
   * ILU preconditioner;
   * The iterative method will break down unless you comment in the SetShift
   * line below, or use the -pc_factor_shift_positive_definite option.
   * Run the code twice: once as given to see the negative pivot and the
   * divergence behaviour, then comment in the Shift line, or add the 
   * command line option, and see that the pivots are all positive and
   * the method converges.
   */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCICC);CHKERRQ(ierr);
  /* ierr = PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE);CHKERRQ(ierr); */

  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  /*
   * Now that the factorisation is done, show the pivots;
   * note that the last one is negative. This in itself is not an error,
   * but it will make the iterative method diverge.
   */
  ierr = PCFactorGetMatrix(pc,&M);CHKERRQ(ierr);
  ierr = VecDuplicate(B,&D);CHKERRQ(ierr);
  ierr = MatGetDiagonal(M,D);CHKERRQ(ierr);
  printf("\nPivots:\n\n"); VecView(D,0);

  /*
   * Solve the system;
   * without the shift this will diverge with
   * an indefinite preconditioner
   */
  ierr = KSPSolve(ksp,B,X);CHKERRQ(ierr);
  ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
  if (reason==KSP_DIVERGED_INDEFINITE_PC) {
    printf("\nDivergence because of indefinite preconditioner;\n");
    printf("Run the executable again but with -pc_factor_shift_positive_definite option.\n");
  } else if (reason<0) {
    printf("\nOther kind of divergence: this should not happen.\n");
  } else {
    ierr = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
    printf("\nConvergence in %d iterations.\n",(int)its);
  }
  printf("\n");

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&B);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&D);CHKERRQ(ierr);
  PetscFinalize();
  PetscFunctionReturn(0);
}
