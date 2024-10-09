static char help[] = "Block-structured Nest matrix involving a HermitianTranspose block.\n\n"
                     "The command line options are:\n"
                     "  -n <n>, where <n> = dimension of the blocks.\n\n";

#include <petscksp.h>

/*
   Solves a linear system with coefficient matrix

        H = [  R    C
              -C^H -R^T ],

   where R is Hermitian and C is complex symmetric. In particular, R and C have the
   following Toeplitz structure:

        R = pentadiag{a,b,c,conj(b),conj(a)}
        C = tridiag{b,d,b}

   where a,b,d are complex scalars, and c is real.
*/

int main(int argc, char **argv)
{
  Mat         H, R, C, block[4];
  Vec         rhs, x, r;
  KSP         ksp;
  PC          pc;
  PCType      type;
  PetscReal   norm[2], tol = 100.0 * PETSC_MACHINE_EPSILON;
  PetscScalar a, b, c, d;
  PetscInt    n = 24, Istart, Iend, i;
  PetscBool   flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nBlock-structured matrix, n=%" PetscInt_FMT "\n\n", n));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
               Compute the blocks R and C, and the Nest H
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

#if defined(PETSC_USE_COMPLEX)
  a = PetscCMPLX(-0.1, 0.2);
  b = PetscCMPLX(1.0, 0.5);
  d = PetscCMPLX(2.0, 0.2);
#else
  a = -0.1;
  b = 1.0;
  d = 2.0;
#endif
  c = 4.5;

  PetscCall(MatCreate(PETSC_COMM_WORLD, &R));
  PetscCall(MatSetSizes(R, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(R));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(C));

  PetscCall(MatGetOwnershipRange(R, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (i > 1) PetscCall(MatSetValue(R, i, i - 2, a, INSERT_VALUES));
    if (i > 0) PetscCall(MatSetValue(R, i, i - 1, b, INSERT_VALUES));
    PetscCall(MatSetValue(R, i, i, c, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(R, i, i + 1, PetscConj(b), INSERT_VALUES));
    if (i < n - 2) PetscCall(MatSetValue(R, i, i + 2, PetscConj(a), INSERT_VALUES));
  }

  PetscCall(MatGetOwnershipRange(C, &Istart, &Iend));
  for (i = Istart; i < Iend; i++) {
    if (i > 0) PetscCall(MatSetValue(C, i, i - 1, b, INSERT_VALUES));
    PetscCall(MatSetValue(C, i, i, d, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(C, i, i + 1, b, INSERT_VALUES));
  }

  PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  /* top block row */
  block[0] = R;
  block[1] = C;

  /* bottom block row */
  PetscCall(MatCreateHermitianTranspose(C, &block[2]));
  PetscCall(MatScale(block[2], -1.0));
  PetscCall(MatCreateTranspose(R, &block[3]));
  PetscCall(MatScale(block[3], -1.0));

  /* create nest matrix */
  PetscCall(MatCreateNest(PetscObjectComm((PetscObject)R), 2, NULL, 2, NULL, block, &H));
  PetscCall(MatDestroy(&block[2]));
  PetscCall(MatDestroy(&block[3]));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
                Create linear system and solve
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, H, H));
  PetscCall(KSPSetTolerances(ksp, tol, PETSC_CURRENT, PETSC_CURRENT, PETSC_CURRENT));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(MatCreateVecs(H, &x, &rhs));
  PetscCall(VecSet(rhs, 1.0));
  PetscCall(KSPSolve(ksp, rhs, x));

  /* check final residual */
  PetscCall(VecDuplicate(rhs, &r));
  PetscCall(MatMult(H, x, r));
  PetscCall(VecAXPY(r, -1.0, rhs));
  PetscCall(VecNorm(r, NORM_2, norm));
  PetscCall(VecNorm(rhs, NORM_2, norm + 1));
  PetscCheck(norm[0] / norm[1] < 10.0 * PETSC_SMALL, PetscObjectComm((PetscObject)H), PETSC_ERR_PLIB, "Error ||H x-rhs||_2 / ||rhs||_2: %1.6e", (double)(norm[0] / norm[1]));

  /* check PetscMemType */
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCGetType(pc, &type));
  PetscCall(PetscStrcmp(type, PCFIELDSPLIT, &flg));
  if (flg) {
    KSP               *subksp;
    Mat                pmat;
    const PetscScalar *array;
    PetscInt           n;
    PetscMemType       type[2];

    PetscCall(PCFieldSplitGetSubKSP(pc, &n, &subksp));
    PetscCheck(n == 2, PetscObjectComm((PetscObject)pc), PETSC_ERR_PLIB, "Wrong number of fields");
    PetscCall(KSPGetOperators(subksp[1], NULL, &pmat));
    PetscCall(PetscFree(subksp));
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)pmat, &flg, MATSEQDENSE, MATMPIDENSE, ""));
    if (flg) {
      PetscCall(VecGetArrayReadAndMemType(x, &array, type));
      PetscCall(VecRestoreArrayReadAndMemType(x, &array));
      PetscCall(MatDenseGetArrayReadAndMemType(pmat, &array, type + 1));
      PetscCall(MatDenseRestoreArrayReadAndMemType(pmat, &array));
      PetscCheck(type[0] == type[1], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Failed PetscMemType comparison");
    }
  }

  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&H));
  PetscCall(VecDestroy(&rhs));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      requires: !complex
      output_file: output/ex87.out
      test:
         suffix: real
         args: -ksp_pc_side right
      test:
         suffix: real_fieldsplit
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type cholesky -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full
      test:
         requires: cuda
         nsize: {{1 4}}
         suffix: real_fieldsplit_cuda
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_pc_type redundant -fieldsplit_redundant_ksp_type preonly -fieldsplit_redundant_pc_type lu -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full -mat_type aijcusparse
      test:
         requires: hip
         nsize: 4 # this is broken with a single process, see https://gitlab.com/petsc/petsc/-/issues/1529
         suffix: real_fieldsplit_hip
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_pc_type redundant -fieldsplit_redundant_ksp_type preonly -fieldsplit_redundant_pc_type lu -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full -mat_type aijhipsparse

   testset:
      requires: complex
      output_file: output/ex87.out
      test:
         suffix: complex
         args: -ksp_pc_side right
      test:
         requires: elemental
         nsize: 4
         suffix: complex_fieldsplit_elemental
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_0_pc_type redundant -fieldsplit_0_redundant_pc_type lu -fieldsplit_1_pc_type lu -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full -fieldsplit_1_explicit_operator_mat_type elemental
      test:
         requires: scalapack
         nsize: 4
         suffix: complex_fieldsplit_scalapack
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_pc_type lu -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full -mat_type scalapack -fieldsplit_1_explicit_operator_mat_type scalapack
      test:
         suffix: complex_fieldsplit
         args: -ksp_type preonly -pc_type fieldsplit -fieldsplit_ksp_type preonly -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type cholesky -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition full -fieldsplit_1_explicit_operator_mat_hermitian

TEST*/
