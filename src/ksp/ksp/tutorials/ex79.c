#include <petsc.h>

#if PetscDefined(HAVE_HYPRE_DEVICE)
  #include <petsc/private/petschypre.h>
#endif

static char help[] = "Solves a linear system with a block of right-hand sides, apply a preconditioner to the same block.\n\n";

PetscErrorCode MatApply(PC pc, Mat X, Mat Y)
{
  PetscFunctionBeginUser;
  PetscCall(MatCopy(X, Y, SAME_NONZERO_PATTERN));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat                A, X, B; /* computed solutions and RHS */
  KSP                ksp;     /* linear solver context */
  PC                 pc;      /* preconditioner context */
  PetscInt           m = 10;
  PetscBool          flg, transpose = PETSC_FALSE, hypre_mat_on_device = PETSC_FALSE;
  PetscLogEvent      event;
  PetscEventPerfInfo info;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, m, m, PETSC_DECIDE, PETSC_DECIDE, m, NULL, m, NULL, &A));
  PetscCall(MatSetRandom(A, NULL));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-transpose", &transpose, NULL));
  PetscCall(MatShift(A, 10.0));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, PETSC_DECIDE, m, NULL, &B));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, PETSC_DECIDE, m, NULL, &X));
  PetscCall(MatSetRandom(B, NULL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, ""));
  if (flg) {
    PetscCall(MatConvert(B, MATDENSECUDA, MAT_INPLACE_MATRIX, &B));
    PetscCall(MatConvert(X, MATDENSECUDA, MAT_INPLACE_MATRIX, &X));
  } else {
    PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJHIPSPARSE, MATMPIAIJHIPSPARSE, ""));
    if (flg) {
      PetscCall(MatConvert(B, MATDENSEHIP, MAT_INPLACE_MATRIX, &B));
      PetscCall(MatConvert(X, MATDENSEHIP, MAT_INPLACE_MATRIX, &X));
    }
  }
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
#if defined(PETSC_HAVE_HYPRE_DEVICE)
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCHYPRE, &flg));
  if (flg) {
    HYPRE_MemoryLocation hmem;
    PetscCallExternal(HYPRE_GetMemoryLocation, &hmem);
    if (hmem == HYPRE_MEMORY_DEVICE) hypre_mat_on_device = PETSC_TRUE;
  }
#endif
  if (!hypre_mat_on_device) {
    PetscCall(PCShellSetMatApply(pc, MatApply));
    PetscCall(PCShellSetMatApplyTranspose(pc, MatApply));
  }
  PetscCall(KSPMatSolve(ksp, B, X));
  PetscCall(PCMatApply(pc, B, X));
  if (transpose) {
    PetscCall(KSPMatSolveTranspose(ksp, B, X));
    PetscCall(PCMatApply(pc, B, X));
    PetscCall(KSPMatSolve(ksp, B, X));
  }
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscLogEventRegister("PCApply", PC_CLASSID, &event)); // there is no PCApplyTranspose event
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info));
  PetscCheck(!PetscDefined(USE_LOG) || m <= 1 || !info.count, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "PCApply() called %d times", info.count);
  PetscCall(PetscLogEventRegister("PCMatApply", PC_CLASSID, &event)); // there is no PCMatApplyTranspose event
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info));
  PetscCheck(!PetscDefined(USE_LOG) || m <= 1 || info.count, PetscObjectComm((PetscObject)ksp), PETSC_ERR_PLIB, "PCMatApply() never called");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   # KSPHPDDM does either pseudo-blocking or "true" blocking, all tests should succeed with other -ksp_hpddm_type
   testset:
      nsize: 1
      args: -pc_type {{mat none shell gasm}shared output}
      test:
         suffix: 1
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 1_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      args: -pc_type ksp -transpose
      test:
         suffix: 2
         output_file: output/empty.out
         args: -ksp_ksp_type preonly -ksp_type preonly
      test:
         suffix: 2_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_ksp_type hpddm -ksp_type hpddm -ksp_hpddm_type preonly -ksp_ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: h2opus
      args: -pc_type h2opus -pc_h2opus_init_mat_h2opus_leafsize 10
      test:
         suffix: 3
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 3_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: spai
      args: -pc_type spai
      test:
         suffix: 4
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 4_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly
   # special code path in PCMatApply() for PCBJACOBI when a block is shared by multiple processes
   testset:
      nsize: 2
      args: -pc_type bjacobi -pc_bjacobi_blocks 1 -sub_pc_type none
      test:
         suffix: 5
         output_file: output/empty.out
         args: -ksp_type preonly -sub_ksp_type preonly
      test:
         suffix: 5_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly -sub_ksp_type hpddm
   # special code path in PCMatApply() for PCGASM when a block is shared by multiple processes
   testset:
      nsize: 2
      args: -pc_type gasm -pc_gasm_total_subdomains 1 -sub_pc_type none
      test:
         suffix: 6
         output_file: output/empty.out
         args: -ksp_type preonly -sub_ksp_type preonly
      test:
         suffix: 6_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly -sub_ksp_type hpddm

   testset:
      nsize: 1
      requires: suitesparse
      args: -pc_type qr
      test:
         suffix: 7
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 7_hpddm
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: hpddm cuda
      args: -mat_type aijcusparse -ksp_type hpddm
      test:
         suffix: 8_hpddm
         output_file: output/empty.out
      test:
         suffix: 8_hpddm_transpose
         output_file: output/empty.out
         args: -pc_type icc -transpose

   testset:
      nsize: 1
      args: -pc_type {{bjacobi lu ilu cholesky icc none shell}shared output} -transpose
      test:
         suffix: 1_transpose
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 1_hpddm_transpose
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 2
      args: -pc_type asm -transpose
      test:
         suffix: 2_transpose
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 2_hpddm_transpose
         output_file: output/empty.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      requires: hypre !complex
      args: -pc_type hypre -pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi -pc_hypre_boomeramg_no_CF
      test:
         suffix: 9
         output_file: output/empty.out
         args: -ksp_type preonly
      test:
         suffix: 9_hpddm
         output_file: output/empty.out
         requires: hpddm !hip
         args: -ksp_type hpddm -ksp_max_it 15 -ksp_error_if_not_converged
      test:
         suffix: 9_cuda
         output_file: output/empty.out
         requires: cuda
         args: -mat_type aijcusparse -ksp_type preonly

TEST*/
