#include <petsc.h>

static char help[] = "Solves a series of linear systems using KSPHPDDM.\n\n";

int main(int argc,char **args)
{
  Vec            x,b;        /* computed solution and RHS */
  Mat            A;          /* linear system matrix */
  KSP            ksp;        /* linear solver context */
#if defined(PETSC_HAVE_HPDDM)
  Mat            U;          /* deflation space */
#endif
  PetscInt       i,j,nmat = 10;
  PetscViewer    viewer;
  char           dir[PETSC_MAX_PATH_LEN],name[256];
  PetscBool      flg,reset = PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,NULL,help));
  CHKERRQ(PetscStrcpy(dir,"."));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nmat",&nmat,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-reset",&reset,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  for (i=0; i<nmat; i++) {
    j = i+400;
    CHKERRQ(PetscSNPrintf(name,sizeof(name),"%s/A_%d.dat",dir,j));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
    CHKERRQ(MatLoad(A,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    if (i == 0) {
      CHKERRQ(MatCreateVecs(A,&x,&b));
    }
    CHKERRQ(PetscSNPrintf(name,sizeof(name),"%s/rhs_%d.dat",dir,j));
    CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
    CHKERRQ(VecLoad(b,viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(KSPSetFromOptions(ksp));
    CHKERRQ(KSPSolve(ksp,b,x));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)ksp,KSPHPDDM,&flg));
#if defined(PETSC_HAVE_HPDDM)
    if (flg && reset) {
      CHKERRQ(KSPHPDDMGetDeflationSpace(ksp,&U));
      CHKERRQ(KSPReset(ksp));
      CHKERRQ(KSPSetOperators(ksp,A,A));
      CHKERRQ(KSPSetFromOptions(ksp));
      CHKERRQ(KSPSetUp(ksp));
      if (U) {
        CHKERRQ(KSPHPDDMSetDeflationSpace(ksp,U));
        CHKERRQ(MatDestroy(&U));
      }
    }
#endif
  }
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -nmat 1 -pc_type none -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -ksp_hpddm_type {{gmres bgmres}shared output} -options_left no -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   test:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: 1_icc
      nsize: 1
      args: -nmat 1 -pc_type icc -ksp_converged_reason -ksp_type {{gmres hpddm}shared output} -ksp_max_it 1000 -ksp_gmres_restart 1000 -ksp_rtol 1e-10 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR

   testset:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -nmat 3 -pc_type none -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type {{gcrodr bgcrodr}shared output} -ksp_hpddm_recycle 20 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
      test:
        nsize: 1
        suffix: 2_seq
        output_file: output/ex75_2.out
      test:
        nsize: 2
        suffix: 2_par
        output_file: output/ex75_2.out

   testset:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      nsize: 1
      args: -nmat 3 -pc_type icc -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_rtol 1e-10 -ksp_hpddm_type {{gcrodr bgcrodr}shared output} -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR
      test:
        suffix: 2_icc
        args:
      test:
        suffix: 2_icc_atol
        args: -ksp_atol 1e-12

   test:
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      nsize: 2
      suffix: symmetric
      args: -nmat 3 -pc_type jacobi -ksp_converged_reason -ksp_type hpddm -ksp_max_it 1000 -ksp_gmres_restart 40 -ksp_atol 1e-11 -ksp_hpddm_type bgcrodr -ksp_hpddm_recycle 20 -reset {{false true}shared output} -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -ksp_hpddm_recycle_symmetric true

TEST*/
