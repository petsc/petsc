static const char help[] = "Example demonstrating PCCOMPOSITE where one of the inner PCs uses a different operator\n\
\n";

#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscInt    n = 10, i, col[3];
  Vec         x, b;
  Mat         A, B;
  KSP         ksp;
  PC          pc, subpc;
  PetscScalar value[3];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* Create a diagonal matrix with a given distribution of diagonal elements */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  /*
     Assemble matrix
  */
  value[0] = -1.0;
  value[1] = 2.0;
  value[2] = -1.0;
  for (i = 1; i < n - 1; i++) {
    col[0] = i - 1;
    col[1] = i;
    col[2] = i + 1;
    PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
  }
  i      = n - 1;
  col[0] = n - 2;
  col[1] = n - 1;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  i        = 0;
  col[0]   = 0;
  col[1]   = 1;
  value[0] = 2.0;
  value[1] = -1.0;
  PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));

  /* Create a KSP object */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));

  /* Set up a composite preconditioner */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCCOMPOSITE)); /* default composite with single Identity PC */
  PetscCall(PCCompositeSetType(pc, PC_COMPOSITE_ADDITIVE));
  PetscCall(PCCompositeAddPCType(pc, PCLU));
  PetscCall(PCCompositeGetPC(pc, 0, &subpc));
  /*  B is set to the diagonal of A; this demonstrates that setting the operator for a subpc changes the preconditioning */
  PetscCall(MatDuplicate(A, MAT_DO_NOT_COPY_VALUES, &B));
  PetscCall(MatGetDiagonal(A, b));
  PetscCall(MatDiagonalSet(B, b, ADD_VALUES));
  PetscCall(PCSetOperators(subpc, B, B));
  PetscCall(PCCompositeAddPCType(pc, PCNONE));

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     args: -ksp_monitor -pc_composite_type multiplicative

TEST*/
