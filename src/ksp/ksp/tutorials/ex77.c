#include <petsc.h>

static char help[] = "Solves a linear system with a block of right-hand sides using KSPHPDDM.\n\n";

int main(int argc,char **args)
{
  Mat                X,B;         /* computed solutions and RHS */
  Vec                cx,cb;       /* columns of X and B */
  Mat                A,KA = NULL; /* linear system matrix */
  KSP                ksp;         /* linear solver context */
  PC                 pc;          /* preconditioner context */
  Mat                F;           /* factored matrix from the preconditioner context */
  PetscScalar        *x,*S = NULL,*T = NULL;
  PetscReal          norm,deflation = -1.0;
  PetscInt           m,M,N = 5,i;
  PetscMPIInt        rank,size;
  const char         *deft = MATAIJ;
  PetscViewer        viewer;
  char               name[PETSC_MAX_PATH_LEN],type[256];
  PetscBool          breakdown = PETSC_FALSE,flg;
  KSPConvergedReason reason;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",name,sizeof(name),&flg));
  PetscCheckFalse(!flg,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must provide a binary file for the matrix");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-breakdown",&breakdown,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-ksp_hpddm_deflation_tol",&deflation,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    CHKERRQ(PetscStrcmp(type,MATKAIJ,&flg));
    if (!flg) {
      CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE));
      CHKERRQ(MatConvert(A,type,MAT_INPLACE_MATRIX,&A));
    } else {
      if (size > 2) {
        CHKERRQ(MatGetSize(A,&M,NULL));
        CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
        if (rank > 1) {
          CHKERRQ(MatSetSizes(B,0,0,M,M));
        } else {
          CHKERRQ(MatSetSizes(B,rank?M-M/2:M/2,rank?M-M/2:M/2,M,M));
        }
        CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
        CHKERRQ(MatLoad(B,viewer));
        CHKERRQ(PetscViewerDestroy(&viewer));
        CHKERRQ(MatHeaderReplace(A,&B));
      }
      CHKERRQ(PetscCalloc2(N*N,&S,N*N,&T));
      for (i=0; i<N; i++) { /* really easy problem used for testing */
        S[i*(N+1)] = 1e+6;
        T[i*(N+1)] = 1e-2;
      }
      CHKERRQ(MatCreateKAIJ(A,N,N,S,T,&KA));
    }
  }
  if (!flg) {
    if (size > 4) {
      Mat B;
      CHKERRQ(MatGetSize(A,&M,NULL));
      CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
      if (rank > 3) {
        CHKERRQ(MatSetSizes(B,0,0,M,M));
      } else {
        CHKERRQ(MatSetSizes(B,rank == 0?M-3*(M/4):M/4,rank == 0?M-3*(M/4):M/4,M,M));
      }
      CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer));
      CHKERRQ(MatLoad(B,viewer));
      CHKERRQ(PetscViewerDestroy(&viewer));
      CHKERRQ(MatHeaderReplace(A,&B));
    }
  }
  CHKERRQ(MatGetLocalSize(A,&m,NULL));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&B));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&X));
  if (!breakdown) {
    CHKERRQ(MatSetRandom(B,NULL));
  }
  CHKERRQ(KSPSetFromOptions(ksp));
  if (!flg) {
    if (!breakdown) {
      CHKERRQ(KSPMatSolve(ksp,B,X));
      CHKERRQ(KSPGetMatSolveBatchSize(ksp,&M));
      if (M != PETSC_DECIDE) {
        CHKERRQ(KSPSetMatSolveBatchSize(ksp,PETSC_DECIDE));
        CHKERRQ(MatZeroEntries(X));
        CHKERRQ(KSPMatSolve(ksp,B,X));
      }
      CHKERRQ(KSPGetPC(ksp,&pc));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)pc,PCLU,&flg));
      if (flg) {
        CHKERRQ(PCFactorGetMatrix(pc,&F));
        CHKERRQ(MatMatSolve(F,B,B));
        CHKERRQ(MatAYPX(B,-1.0,X,SAME_NONZERO_PATTERN));
        CHKERRQ(MatNorm(B,NORM_INFINITY,&norm));
        PetscCheckFalse(norm > 100*PETSC_MACHINE_EPSILON,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPMatSolve() and MatMatSolve() difference has nonzero norm %g",(double)norm);
      }
    } else {
      CHKERRQ(MatZeroEntries(B));
      CHKERRQ(KSPMatSolve(ksp,B,X));
      CHKERRQ(KSPGetConvergedReason(ksp,&reason));
      PetscCheckFalse(reason != KSP_CONVERGED_HAPPY_BREAKDOWN,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPConvergedReason() %s != KSP_CONVERGED_HAPPY_BREAKDOWN",KSPConvergedReasons[reason]);
      CHKERRQ(MatDenseGetArrayWrite(B,&x));
      for (i=0; i<m*N; ++i) x[i] = 1.0;
      CHKERRQ(MatDenseRestoreArrayWrite(B,&x));
      CHKERRQ(KSPMatSolve(ksp,B,X));
      CHKERRQ(KSPGetConvergedReason(ksp,&reason));
      PetscCheckFalse(reason != KSP_DIVERGED_BREAKDOWN && deflation < 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPConvergedReason() %s != KSP_DIVERGED_BREAKDOWN",KSPConvergedReasons[reason]);
    }
  } else {
    CHKERRQ(KSPSetOperators(ksp,KA,KA));
    CHKERRQ(MatGetSize(KA,&M,NULL));
    CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,m*N,M,&cb));
    CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,m*N,M,&cx));
    CHKERRQ(VecSetRandom(cb,NULL));
    /* solving with MatKAIJ is equivalent to block solving with row-major RHS and solutions */
    /* only applies if MatKAIJGetScaledIdentity() returns true                              */
    CHKERRQ(KSPSolve(ksp,cb,cx));
    CHKERRQ(VecDestroy(&cx));
    CHKERRQ(VecDestroy(&cb));
  }
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(PetscFree2(S,T));
  CHKERRQ(MatDestroy(&KA));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(KSPDestroy(&ksp));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -mat_type {{aij sbaij}shared output}
      test:
         suffix: 1
         args:
      test:
         suffix: 2
         requires: hpddm
         args: -ksp_type hpddm -pc_type asm -ksp_hpddm_type {{gmres bgmres}separate output}
      test:
         suffix: 3
         requires: hpddm
         args: -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type {{gcrodr bgcrodr}separate output}
      test:
         nsize: 4
         suffix: 4
         requires: hpddm
         args: -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5

   test:
      nsize: 1
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: preonly
      args: -N 6 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -pc_type lu -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -N 3 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_type hpddm -breakdown
      test:
         suffix: breakdown_wo_deflation
         output_file: output/ex77_preonly.out
         args: -pc_type none -ksp_hpddm_type {{bcg bgmres bgcrodr bfbcg}shared output}
      test:
         suffix: breakdown_w_deflation
         output_file: output/ex77_deflation.out
         filter: sed "s/BGCRODR/BGMRES/g"
         args: -pc_type lu -ksp_hpddm_type {{bgmres bgcrodr}shared output} -ksp_hpddm_deflation_tol 1e-1 -info :ksp

   test:
      nsize: 2
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type {{gmres bgmres}separate output}

   test:
      nsize: 3
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: kaij_zero
      output_file: output/ex77_ksp_hpddm_type-bgmres.out
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type bgmres

   test:
      nsize: 4
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      suffix: 4_slepc
      output_file: output/ex77_4.out
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type {{aij dense}shared output} -ksp_hpddm_recycle_eps_converged_reason -ksp_hpddm_recycle_st_pc_type redundant

   testset:
      nsize: 4
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES) slepc defined(PETSC_HAVE_DYNAMIC_LIBRARIES) defined(PETSC_USE_SHARED_LIBRARIES)
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_eps_converged_reason
      test:
         requires: elemental
         suffix: 4_elemental
         output_file: output/ex77_4.out
         args: -ksp_hpddm_recycle_mat_type elemental
      test:
         requires: elemental
         suffix: 4_elemental_matmat
         output_file: output/ex77_4.out
         args: -ksp_hpddm_recycle_mat_type elemental -ksp_hpddm_recycle_eps_type subspace -ksp_hpddm_recycle_eps_target 0 -ksp_hpddm_recycle_eps_target_magnitude -ksp_hpddm_recycle_st_type sinvert -ksp_hpddm_recycle_bv_type mat -ksp_hpddm_recycle_bv_orthog_block svqb
      test:
         requires: scalapack
         suffix: 4_scalapack
         output_file: output/ex77_4.out
         args: -ksp_hpddm_recycle_mat_type scalapack

   test:
      nsize: 5
      requires: hpddm datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: 4_zero
      output_file: output/ex77_4.out
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_batch_size 5

TEST*/
