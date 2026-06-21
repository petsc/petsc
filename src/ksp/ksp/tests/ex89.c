static char help[] = "Test interface functions of KSPIDR with a nonsymmetric matrix.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, xe; /* approx solution, RHS, exact solution */
  Mat         A;        /* linear system matrix */
  KSP         ksp;      /* linear solver context */
  PC          pc;       /* preconditioner */
  PetscInt    i, Istart, Iend, n = 40, s = 0;
  PetscReal   nrm;
  PetscRandom rctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  /* Create bidiagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  for (i = Istart; i < Iend; i++) {
    PetscCall(MatSetValue(A, i, i, 2.0, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(A, i, i + 1, 1.0 / (i + 1), INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecDuplicate(x, &xe));
  PetscCall(VecSet(xe, 1.0));
  PetscCall(MatMult(A, xe, b));

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPIDR));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));

  /* Pass specific IDR options, including user-provided PetscRandom */
  PetscCall(KSPIDRSetS(ksp, 6));
  PetscCall(KSPIDRSetCosine(ksp, 0.8));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(PetscRandomSetFromOptions(rctx));
  PetscCall(KSPIDRSetRandom(ksp, rctx));
  PetscCall(PetscRandomDestroy(&rctx));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecAXPY(x, -1.0, xe));
  PetscCall(VecNorm(x, NORM_2, &nrm));
  PetscCall(KSPIDRGetS(ksp, &s));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Absolute error (s=%" PetscInt_FMT ") = %1.6f\n", s, (double)nrm));

  /* Solve again with different s */
  PetscCall(KSPIDRSetS(ksp, 4));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecAXPY(x, -1.0, xe));
  PetscCall(VecNorm(x, NORM_2, &nrm));
  PetscCall(KSPIDRGetS(ksp, &s));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Absolute error (s=%" PetscInt_FMT ") = %1.6f\n", s, (double)nrm));

  /* Free work space */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&xe));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

   test:
      args: -pc_type jacobi -ksp_pc_side right
      suffix: 2

TEST*/
