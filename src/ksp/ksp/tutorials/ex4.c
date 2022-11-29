
static char help[] = "Solves a linear system in parallel with KSP and HMG.\n\
Input parameters include:\n\
  -view_exact_sol    : write exact solution vector to stdout\n\
  -m  <mesh_x>       : number of mesh points in x-direction\n\
  -n  <mesh_y>       : number of mesh points in y-direction\n\
  -bs                : number of variables on each mesh vertex \n\n";

/*
  Simple example is used to test PCHMG
*/
#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b, u; /* approx solution, RHS, exact solution */
  Mat         A;       /* linear system matrix */
  KSP         ksp;     /* linear solver context */
  PetscReal   norm;    /* norm of solution error */
  PetscInt    i, j, Ii, J, Istart, Iend, m = 8, n = 7, its, bs = 1, II, JJ, jj;
  PetscBool   flg, test = PETSC_FALSE, reuse = PETSC_FALSE, viewexpl = PETSC_FALSE;
  PetscScalar v;
  PC          pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_hmg_interface", &test, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_reuse_interpolation", &reuse, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_explicit_mat", &viewexpl, NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m * n * bs, m * n * bs));
  PetscCall(MatSetBlockSize(A, bs));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 5, NULL, 5, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 5, NULL));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(MatHYPRESetPreallocation(A, 5, NULL, 5, NULL));
#endif

  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  for (Ii = Istart / bs; Ii < Iend / bs; Ii++) {
    v = -1.0;
    i = Ii / n;
    j = Ii - i * n;
    if (i > 0) {
      J = Ii - n;
      for (jj = 0; jj < bs; jj++) {
        II = Ii * bs + jj;
        JJ = J * bs + jj;
        PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
      }
    }
    if (i < m - 1) {
      J = Ii + n;
      for (jj = 0; jj < bs; jj++) {
        II = Ii * bs + jj;
        JJ = J * bs + jj;
        PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
      }
    }
    if (j > 0) {
      J = Ii - 1;
      for (jj = 0; jj < bs; jj++) {
        II = Ii * bs + jj;
        JJ = J * bs + jj;
        PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
      }
    }
    if (j < n - 1) {
      J = Ii + 1;
      for (jj = 0; jj < bs; jj++) {
        II = Ii * bs + jj;
        JJ = J * bs + jj;
        PetscCall(MatSetValues(A, 1, &II, 1, &JJ, &v, ADD_VALUES));
      }
    }
    v = 4.0;
    for (jj = 0; jj < bs; jj++) {
      II = Ii * bs + jj;
      PetscCall(MatSetValues(A, 1, &II, 1, &II, &v, ADD_VALUES));
    }
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  if (viewexpl) {
    Mat E;
    PetscCall(MatComputeOperator(A, MATAIJ, &E));
    PetscCall(MatView(E, NULL));
    PetscCall(MatDestroy(&E));
  }

  PetscCall(MatCreateVecs(A, &u, NULL));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(b, &x));

  PetscCall(VecSet(u, 1.0));
  PetscCall(MatMult(A, u, b));

  flg = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-view_exact_sol", &flg, NULL));
  if (flg) PetscCall(VecView(u, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetTolerances(ksp, 1.e-2 / ((m + 1) * (n + 1)), 1.e-50, PETSC_DEFAULT, PETSC_DEFAULT));

  if (test) {
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCSetType(pc, PCHMG));
    PetscCall(PCHMGSetInnerPCType(pc, PCGAMG));
    PetscCall(PCHMGSetReuseInterpolation(pc, PETSC_TRUE));
    PetscCall(PCHMGSetUseSubspaceCoarsening(pc, PETSC_TRUE));
    PetscCall(PCHMGUseMatMAIJ(pc, PETSC_FALSE));
    PetscCall(PCHMGSetCoarseningComponent(pc, 0));
  }

  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  if (reuse) {
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSolve(ksp, b, x));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSolve(ksp, b, x));
    /* Make sparsity pattern different and reuse interpolation */
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    PetscCall(MatSetOption(A, MAT_IGNORE_ZERO_ENTRIES, PETSC_FALSE));
    PetscCall(MatGetSize(A, &m, NULL));
    n = 0;
    v = 0;
    m--;
    /* Connect the last element to the first element */
    PetscCall(MatSetValue(A, m, n, v, ADD_VALUES));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(KSPSolve(ksp, b, x));
  }

  PetscCall(VecAXPY(x, -1.0, u));
  PetscCall(VecNorm(x, NORM_2, &norm));
  PetscCall(KSPGetIterationNumber(ksp, &its));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of error %g iterations %" PetscInt_FMT "\n", (double)norm, its));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: !complex !single

   test:
      suffix: hypre
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre

   test:
      suffix: hypre_bs4
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -pc_hmg_use_subspace_coarsening 1

   test:
      suffix: hypre_asm
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -pc_hmg_use_subspace_coarsening 1 -mg_levels_3_pc_type asm

   test:
      suffix: hypre_fieldsplit
      nsize: 2
      requires: hypre !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type hypre -bs 4 -mg_levels_4_pc_type fieldsplit

   test:
      suffix: gamg
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg

   test:
      suffix: gamg_bs4
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -pc_hmg_use_subspace_coarsening 1

   test:
      suffix: gamg_asm
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -pc_hmg_use_subspace_coarsening 1 -mg_levels_1_pc_type asm

   test:
      suffix: gamg_fieldsplit
      nsize: 2
      args: -ksp_monitor -pc_type hmg -ksp_rtol 1e-6 -hmg_inner_pc_type gamg -bs 4 -mg_levels_1_pc_type fieldsplit

   test:
      suffix: interface
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6 -test_hmg_interface 1 -bs 4

   test:
      suffix: reuse
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6   -pc_type hmg -pc_hmg_reuse_interpolation 1 -test_reuse_interpolation 1 -hmg_inner_pc_type gamg

   test:
      suffix: component
      nsize: 2
      args: -ksp_monitor -ksp_rtol 1e-6 -pc_type hmg -pc_hmg_coarsening_component 2  -pc_hmg_use_subspace_coarsening 1 -bs 4 -hmg_inner_pc_type gamg

   testset:
      output_file: output/ex4_expl.out
      nsize: {{1 2}}
      filter: grep -v " MPI process" | grep -v " type:" | grep -v "Mat Object"
      args: -ksp_converged_reason -view_explicit_mat -pc_type none -ksp_type {{cg gmres}}
      test:
        suffix: expl_aij
        args: -mat_type aij
      test:
        suffix: expl_hypre
        requires: hypre
        args: -mat_type hypre

   test:
      suffix: hypre_device
      nsize: {{1 2}}
      requires: hypre defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -mat_type hypre -ksp_converged_reason -pc_type hypre -m 13 -n 17

   test:
      suffix: hypre_device_cusparse
      output_file: output/ex4_hypre_device.out
      nsize: {{1 2}}
      requires: hypre cuda defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -mat_type {{aij aijcusparse}} -vec_type cuda -ksp_converged_reason -pc_type hypre -m 13 -n 17

TEST*/
