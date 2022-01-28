#include <petsc.h>

static char help[] = "Solves a linear system with a block of right-hand sides, apply a preconditioner to the same block.\n\n";

PetscErrorCode MatApply(PC pc, Mat X, Mat Y)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatCopy(X,Y,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscLogDefaultBegin();CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,m,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,m,NULL,&A);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&B);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,m,NULL,&X);CHKERRQ(ierr);
  ierr = MatSetRandom(A,NULL);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatShift(A,10.0);CHKERRQ(ierr);
  ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCShellSetMatApply(pc,MatApply);CHKERRQ(ierr);
  ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
  ierr = PCMatApply(pc,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscLogEventRegister("PCApply",PC_CLASSID,&event);CHKERRQ(ierr);
  ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info);CHKERRQ(ierr);
  if (PetscDefined(USE_LOG) && m > 1 && info.count) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"PCApply() called %d times",info.count);
  ierr = PetscLogEventRegister("PCMatApply",PC_CLASSID,&event);CHKERRQ(ierr);
  ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE,event,&info);CHKERRQ(ierr);
  if (PetscDefined(USE_LOG) && m > 1 && !info.count) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"PCMatApply() never called");
  ierr = PetscFinalize();
  return ierr;
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
