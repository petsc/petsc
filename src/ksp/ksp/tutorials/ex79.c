#include <petsc.h>

static char help[] = "Solves a linear system with a block of right-hand sides, apply a preconditioner to the same block.\n\n";

PetscErrorCode MatApply(PC pc, Mat X, Mat Y)
{
  PetscFunctionBeginUser;
  PetscCall(MatCopy(X,Y,SAME_NONZERO_PATTERN));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat                X,B;         /* computed solutions and RHS */
  Mat                A;           /* linear system matrix */
  KSP                ksp;         /* linear solver context */
  PC                 pc;          /* preconditioner context */
  PetscInt           m = 10;
#if defined(PETSC_USE_LOG)
  PetscLogEvent      event;
#endif
  PetscEventPerfInfo info;

  PetscCall(PetscInitialize(&argc,&args,NULL,help));
  PetscCall(PetscLogDefaultBegin());
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD,m,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,m,NULL,&A));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&B));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&X));
  PetscCall(MatSetRandom(A,NULL));
  PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
  PetscCall(MatShift(A,10.0));
  PetscCall(MatSetRandom(B,NULL));
  PetscCall(KSPCreate(PETSC_COMM_WORLD,&ksp));
  PetscCall(KSPSetOperators(ksp,A,A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp,&pc));
  PetscCall(PCShellSetMatApply(pc,MatApply));
  PetscCall(KSPMatSolve(ksp,B,X));
  PetscCall(PCMatApply(pc,B,X));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscLogEventRegister("PCApply",PC_CLASSID,&event));
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info));
  PetscCheckFalse(PetscDefined(USE_LOG) && m > 1 && info.count,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"PCApply() called %d times",info.count);
  PetscCall(PetscLogEventRegister("PCMatApply",PC_CLASSID,&event));
  PetscCall(PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info));
  PetscCheckFalse(PetscDefined(USE_LOG) && m > 1 && !info.count,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"PCMatApply() never called");
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   # KSPHPDDM does either pseudo-blocking or "true" blocking, all tests should succeed with other -ksp_hpddm_type
   testset:
      nsize: 1
      args: -pc_type {{bjacobi lu ilu mat cholesky icc none shell asm gasm}shared output}
      test:
         suffix: 1
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly
      test:
         suffix: 1_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      args: -pc_type ksp
      test:
         suffix: 2
         output_file: output/ex77_preonly.out
         args: -ksp_ksp_type preonly -ksp_type preonly
      test:
         suffix: 2_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_ksp_type hpddm -ksp_type hpddm -ksp_hpddm_type preonly -ksp_ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: h2opus
      args: -pc_type h2opus -pc_h2opus_init_mat_h2opus_leafsize 10
      test:
         suffix: 3
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly
      test:
         suffix: 3_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: spai
      args: -pc_type spai
      test:
         suffix: 4
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly
      test:
         suffix: 4_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly
   # special code path in PCMatApply() for PCBJACOBI when a block is shared by multiple processes
   testset:
      nsize: 2
      args: -pc_type bjacobi -pc_bjacobi_blocks 1 -sub_pc_type none
      test:
         suffix: 5
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly -sub_ksp_type preonly
      test:
         suffix: 5_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly -sub_ksp_type hpddm
   # special code path in PCMatApply() for PCGASM when a block is shared by multiple processes
   testset:
      nsize: 2
      args: -pc_type gasm -pc_gasm_total_subdomains 1 -sub_pc_type none
      test:
         suffix: 6
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly -sub_ksp_type preonly
      test:
         suffix: 6_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly -sub_ksp_type hpddm

   testset:
      nsize: 1
      requires: suitesparse
      args: -pc_type qr
      test:
         suffix: 7
         output_file: output/ex77_preonly.out
         args: -ksp_type preonly
      test:
         suffix: 7_hpddm
         output_file: output/ex77_preonly.out
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_type preonly

TEST*/
