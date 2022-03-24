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
 * 2/ with -pc_factor_shift_type POSITIVE_DEFINITE option (or comment in the PCFactorSetShiftType() line below):
 *    the method will now successfully converge.
 *
 * Contributed by Victor Eijkhout 2003.
 */

#include <petscksp.h>

int main(int argc,char **argv)
{
  KSP                solver;
  PC                 prec;
  Mat                A,M;
  Vec                X,B,D;
  MPI_Comm           comm;
  PetscScalar        v;
  KSPConvergedReason reason;
  PetscInt           i,j,its;

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  comm = MPI_COMM_SELF;

  /*
   * Construct the Kershaw matrix
   * and a suitable rhs / initial guess
   */
  CHKERRQ(MatCreateSeqAIJ(comm,4,4,4,0,&A));
  CHKERRQ(VecCreateSeq(comm,4,&B));
  CHKERRQ(VecDuplicate(B,&X));
  for (i=0; i<4; i++) {
    v    = 3;
    CHKERRQ(MatSetValues(A,1,&i,1,&i,&v,INSERT_VALUES));
    v    = 1;
    CHKERRQ(VecSetValues(B,1,&i,&v,INSERT_VALUES));
    CHKERRQ(VecSetValues(X,1,&i,&v,INSERT_VALUES));
  }

  i=0; v=0;
  CHKERRQ(VecSetValues(X,1,&i,&v,INSERT_VALUES));

  for (i=0; i<3; i++) {
    v    = -2; j=i+1;
    CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
    CHKERRQ(MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES));
  }
  i=0; j=3; v=2;

  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  CHKERRQ(MatSetValues(A,1,&j,1,&i,&v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(VecAssemblyBegin(B));
  CHKERRQ(VecAssemblyEnd(B));

  /*
   * A Conjugate Gradient method
   * with ILU(0) preconditioning
   */
  CHKERRQ(KSPCreate(comm,&solver));
  CHKERRQ(KSPSetOperators(solver,A,A));

  CHKERRQ(KSPSetType(solver,KSPCG));
  CHKERRQ(KSPSetInitialGuessNonzero(solver,PETSC_TRUE));

  /*
   * ILU preconditioner;
   * this will break down unless you add the Shift line,
   * or use the -pc_factor_shift_positive_definite option */
  CHKERRQ(KSPGetPC(solver,&prec));
  CHKERRQ(PCSetType(prec,PCILU));
  /* CHKERRQ(PCFactorSetShiftType(prec,MAT_SHIFT_POSITIVE_DEFINITE)); */

  CHKERRQ(KSPSetFromOptions(solver));
  CHKERRQ(KSPSetUp(solver));

  /*
   * Now that the factorisation is done, show the pivots;
   * note that the last one is negative. This in itself is not an error,
   * but it will make the iterative method diverge.
   */
  CHKERRQ(PCFactorGetMatrix(prec,&M));
  CHKERRQ(VecDuplicate(B,&D));
  CHKERRQ(MatGetDiagonal(M,D));

  /*
   * Solve the system;
   * without the shift this will diverge with
   * an indefinite preconditioner
   */
  CHKERRQ(KSPSolve(solver,B,X));
  CHKERRQ(KSPGetConvergedReason(solver,&reason));
  if (reason==KSP_DIVERGED_INDEFINITE_PC) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nDivergence because of indefinite preconditioner;\n"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Run the executable again but with '-pc_factor_shift_type POSITIVE_DEFINITE' option.\n"));
  } else if (reason<0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nOther kind of divergence: this should not happen.\n"));
  } else {
    CHKERRQ(KSPGetIterationNumber(solver,&its));
  }

  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&B));
  CHKERRQ(VecDestroy(&D));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&solver));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -pc_factor_shift_type positive_definite

TEST*/
