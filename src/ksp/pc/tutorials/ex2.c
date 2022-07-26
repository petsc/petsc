static char help[] = "Test file for the PCFactorSetShiftType()\n";
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

#include <petscksp.h>

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

  PetscFunctionBegin;
  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  comm = MPI_COMM_SELF;

  /*
   * Construct the Kershaw matrix
   * and a suitable rhs / initial guess
   */
  PetscCall(MatCreateSeqAIJ(comm,4,4,4,0,&A));
  PetscCall(VecCreateSeq(comm,4,&B));
  PetscCall(VecDuplicate(B,&X));
  for (i=0; i<4; i++) {
    v    = 3;
    PetscCall(MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES));
    v    = 1;
    PetscCall(VecSetValues(B,1,&i,&v,INSERT_VALUES));
    PetscCall(VecSetValues(X,1,&i,&v,INSERT_VALUES));
  }

  i    =0; v=0;
  PetscCall(VecSetValues(X,1,&i,&v,INSERT_VALUES));

  for (i=0; i<3; i++) {
    v    = -2; j=i+1;
    PetscCall(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
    PetscCall(MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES));
  }
  i=0; j=3; v=2;

  PetscCall(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  PetscCall(MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(VecAssemblyBegin(B));
  PetscCall(VecAssemblyEnd(B));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nThe Kershaw matrix:\n\n"));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  /*
   * A Conjugate Gradient method
   * with ILU(0) preconditioning
   */
  PetscCall(KSPCreate(comm,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));

  PetscCall(KSPSetType(ksp,KSPCG));
  PetscCall(KSPSetInitialGuessNonzero(ksp,PETSC_TRUE));

  /*
   * ILU preconditioner;
   * The iterative method will break down unless you comment in the SetShift
   * line below, or use the -pc_factor_shift_positive_definite option.
   * Run the code twice: once as given to see the negative pivot and the
   * divergence behaviour, then comment in the Shift line, or add the
   * command line option, and see that the pivots are all positive and
   * the method converges.
   */
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCSetType(pc,PCICC));
  /* PetscCall(PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE)); */

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSetUp(ksp));

  /*
   * Now that the factorisation is done, show the pivots;
   * note that the last one is negative. This in itself is not an error,
   * but it will make the iterative method diverge.
   */
  PetscCall(PCFactorGetMatrix(pc,&M));
  PetscCall(VecDuplicate(B,&D));
  PetscCall(MatGetDiagonal(M,D));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nPivots:\n\n"));
  PetscCall(VecView(D,0));

  /*
   * Solve the system;
   * without the shift this will diverge with
   * an indefinite preconditioner
   */
  PetscCall(KSPSolve(ksp,B,X));
  PetscCall(KSPGetConvergedReason(ksp,&reason));
  if (reason==KSP_DIVERGED_INDEFINITE_PC) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nDivergence because of indefinite preconditioner;\n"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with -pc_factor_shift_positive_definite option.\n"));
  } else if (reason<0) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nOther kind of divergence: this should not happen.\n"));
  } else {
    PetscCall(KSPGetIterationNumber(ksp,&its));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nConvergence in %d iterations.\n",(int)its));
  }
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n"));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&B));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&D));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     filter:  sed -e "s/in 5 iterations/in 4 iterations/g"

TEST*/
