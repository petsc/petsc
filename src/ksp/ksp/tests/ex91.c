static char help[] = "Two SuperLU_DIST factorizations alive at once on one communicator.\n"
                     "Exercises the 'communicator attribute already in use' path of\n"
                     "MatLUFactorSymbolic_SuperLU_DIST and the later attribute delete callback.\n\n";

#include <petscksp.h>

static PetscErrorCode MakeMat(PetscInt n, Mat *A)
{
  PetscInt rs, re;

  PetscFunctionBeginUser;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 3, NULL, 2, NULL, A));
  PetscCall(MatGetOwnershipRange(*A, &rs, &re));
  for (PetscInt i = rs; i < re; i++) {
    PetscScalar d = 4.0, o = -1.0;
    PetscInt    im = i - 1, ip = i + 1;

    PetscCall(MatSetValues(*A, 1, &i, 1, &i, &d, INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValues(*A, 1, &i, 1, &im, &o, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValues(*A, 1, &i, 1, &ip, &o, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SolveOnce(Mat A, KSP *ksp)
{
  Vec x, b;

  PetscFunctionBeginUser;
  PetscCall(KSPCreate(PETSC_COMM_WORLD, ksp));
  PetscCall(KSPSetFromOptions(*ksp));
  PetscCall(KSPSetOperators(*ksp, A, A));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(b, 1.0));
  PetscCall(KSPSolve(*ksp, b, x));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat A1, A2, B;
  KSP k1, k2, k3, kb;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  /* first factorization: grid stored in the communicator attribute */
  PetscCall(MakeMat(18, &B));
  PetscCall(SolveOnce(B, &kb));
  PetscCall(KSPDestroy(&kb));
  PetscCall(MatDestroy(&B));

  /* three factorizations kept alive at the same time */
  PetscCall(MakeMat(18, &A1));
  PetscCall(MakeMat(18, &A2));
  PetscCall(SolveOnce(A1, &k1));
  PetscCall(SolveOnce(A2, &k2));
  PetscCall(SolveOnce(A1, &k3));

  /* a fourth one created and destroyed while the others are alive */
  PetscCall(MakeMat(18, &B));
  PetscCall(SolveOnce(B, &kb));
  PetscCall(KSPDestroy(&kb));
  PetscCall(MatDestroy(&B));

  /* destroying the survivors frees the inner communicator and runs the attribute delete callback */
  PetscCall(KSPDestroy(&k1));
  PetscCall(KSPDestroy(&k2));
  PetscCall(KSPDestroy(&k3));
  PetscCall(MatDestroy(&A1));
  PetscCall(MatDestroy(&A2));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "done\n"));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: superlu_dist

  test:
    args: -ksp_type richardson -ksp_max_it 1 -pc_type lu -pc_factor_mat_solver_type superlu_dist

  test:
    suffix: 2
    nsize: 2
    args: -ksp_type richardson -ksp_max_it 1 -pc_type lu -pc_factor_mat_solver_type superlu_dist
    output_file: output/ex91_1.out

TEST*/
