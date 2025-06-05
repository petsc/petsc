static char help[] = "Tests solving linear system on 0 by 0 matrix, and KSPLSQR and user convergence test handling.\n\n";

#include <petscksp.h>

typedef struct {
  PetscBool converged;
} ConvergedCtx;

static PetscErrorCode TestConvergence(KSP ksp, PetscInt it, PetscReal rnorm, KSPConvergedReason *reason, void *ctx)
{
  ConvergedCtx *user = (ConvergedCtx *)ctx;

  PetscFunctionBeginUser;
  if (user->converged) *reason = KSP_CONVERGED_USER;
  else *reason = KSP_DIVERGED_USER;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         C;
  PetscInt    n = 1, m0, m1, i;
  Vec         b, x;
  KSP         ksp;
  PetscScalar one = 1;
  PetscRandom rctx;

  ConvergedCtx user;
  user.converged = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  /* create stiffness matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatGetOwnershipRange(C, &m0, &m1));
  for (i = m0; i < m1; i++) PetscCall(MatSetValues(C, 1, &i, 1, &i, &one, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-mark_converged", &user.converged, NULL));
  /* create right-hand side and solution */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecSet(x, 0.0));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rctx));
  PetscCall(VecSetRandom(b, rctx));
  PetscCall(PetscRandomDestroy(&rctx));

  /* solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(KSPSetConvergenceTest(ksp, TestConvergence, &user, NULL));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPConvergedReasonView(ksp, NULL));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args:

    test:
      suffix: 2
      args: -mark_converged 0

TEST*/
