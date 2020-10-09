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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-f",name,sizeof(name),&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Must provide a binary file for the matrix");
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-breakdown",&breakdown,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-ksp_hpddm_deflation_tol",&deflation,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","","");CHKERRQ(ierr);
  ierr = PetscOptionsFList("-mat_type","Matrix type","MatSetType",MatList,deft,type,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  if (flg) {
    ierr = PetscStrcmp(type,MATKAIJ,&flg);CHKERRQ(ierr);
    if (!flg) {
      ierr = MatSetOption(A,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = MatConvert(A,type,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
    } else {
      if (size > 2) {
        ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
        ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
        if (rank > 1) {
          ierr = MatSetSizes(B,0,0,M,M);CHKERRQ(ierr);
        } else {
          ierr = MatSetSizes(B,rank?M-M/2:M/2,rank?M-M/2:M/2,M,M);CHKERRQ(ierr);
        }
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
        ierr = MatLoad(B,viewer);CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
        ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
      }
      ierr = PetscCalloc2(N*N,&S,N*N,&T);CHKERRQ(ierr);
      for (i=0; i<N; i++) { /* really easy problem used for testing */
        S[i*(N+1)] = 1e+6;
        T[i*(N+1)] = 1e-2;
      }
      ierr = MatCreateKAIJ(A,N,N,S,T,&KA);CHKERRQ(ierr);
    }
  }
  if (!flg) {
    if (size > 4) {
      Mat B;
      ierr = MatGetSize(A,&M,NULL);CHKERRQ(ierr);
      ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
      if (rank > 3) {
        ierr = MatSetSizes(B,0,0,M,M);CHKERRQ(ierr);
      } else {
        ierr = MatSetSizes(B,!rank?M-3*(M/4):M/4,!rank?M-3*(M/4):M/4,M,M);CHKERRQ(ierr);
      }
      ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,name,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
      ierr = MatLoad(B,viewer);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
      ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
    }
  }
  ierr = MatGetLocalSize(A,&m,NULL);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&B);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,m,PETSC_DECIDE,PETSC_DECIDE,N,NULL,&X);CHKERRQ(ierr);
  if (!breakdown) {
    ierr = MatSetRandom(B,NULL);CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  if (!flg) {
    if (!breakdown) {
      ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
      ierr = KSPGetMatSolveBlockSize(ksp,&M);CHKERRQ(ierr);
      if (M != PETSC_DECIDE) {
        ierr = KSPSetMatSolveBlockSize(ksp,PETSC_DECIDE);CHKERRQ(ierr);
        ierr = MatZeroEntries(X);CHKERRQ(ierr);
        ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
      }
      ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)pc,PCLU,&flg);CHKERRQ(ierr);
      if (flg) {
        ierr = PCFactorGetMatrix(pc,&F);
        ierr = MatMatSolve(F,B,B);CHKERRQ(ierr);
        ierr = MatAYPX(B,-1.0,X,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatNorm(B,NORM_INFINITY,&norm);CHKERRQ(ierr);
        if (norm > 100*PETSC_MACHINE_EPSILON) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPMatSolve() and MatMatSolve() difference has nonzero norm %g",(double)norm);
      }
    } else {
      ierr = MatZeroEntries(B);CHKERRQ(ierr);
      ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
      if (reason != KSP_CONVERGED_HAPPY_BREAKDOWN) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPConvergedReason() %s != KSP_CONVERGED_HAPPY_BREAKDOWN",KSPConvergedReasons[reason]);
      ierr = MatDenseGetArrayWrite(B,&x);CHKERRQ(ierr);
      for (i=0; i<m*N; ++i) x[i] = 1.0;
      ierr = MatDenseRestoreArrayWrite(B,&x);CHKERRQ(ierr);
      ierr = KSPMatSolve(ksp,B,X);CHKERRQ(ierr);
      ierr = KSPGetConvergedReason(ksp,&reason);CHKERRQ(ierr);
      if (reason != KSP_DIVERGED_BREAKDOWN && deflation < 0.0) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPConvergedReason() %s != KSP_DIVERGED_BREAKDOWN",KSPConvergedReasons[reason]);
    }
  } else {
    ierr = KSPSetOperators(ksp,KA,KA);CHKERRQ(ierr);
    ierr = MatGetSize(KA,&M,NULL);CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,m*N,M,&cb);CHKERRQ(ierr);
    ierr = VecCreateMPI(PETSC_COMM_WORLD,m*N,M,&cx);CHKERRQ(ierr);
    ierr = VecSetRandom(cb,NULL);CHKERRQ(ierr);
    /* solving with MatKAIJ is equivalent to block solving with row-major RHS and solutions */
    /* only applies if MatKAIJGetScaledIdentity() returns true                              */
    ierr = KSPSolve(ksp,cb,cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cb);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = PetscFree2(S,T);CHKERRQ(ierr);
  ierr = MatDestroy(&KA);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   testset:
      nsize: 2
      requires: datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
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
         args: -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5

   test:
      nsize: 1
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: preonly
      args: -N 6 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -pc_type lu -ksp_type hpddm -ksp_hpddm_type preonly

   testset:
      nsize: 1
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
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
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type {{gmres bgmres}separate output}

   test:
      nsize: 3
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: kaij_zero
      output_file: output/ex77_ksp_hpddm_type-bgmres.out
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type bgmres

   test:
      nsize: 4
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) slepc
      suffix: 4_slepc
      output_file: output/ex77_4.out
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type {{aij dense}shared output} -ksp_hpddm_recycle_eps_converged_reason -ksp_hpddm_recycle_st_pc_type redundant

   testset:
      nsize: 4
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) slepc elemental
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type elemental -ksp_hpddm_recycle_eps_converged_reason
      test:
         suffix: 4_elemental
         output_file: output/ex77_4.out
      test:
         suffix: 4_elemental_matmat
         output_file: output/ex77_4.out
         args: -ksp_hpddm_recycle_eps_type subspace -ksp_hpddm_recycle_eps_target 0 -ksp_hpddm_recycle_eps_target_magnitude -ksp_hpddm_recycle_st_type sinvert -ksp_hpddm_recycle_bv_type mat -ksp_hpddm_recycle_bv_orthog_block svqb

   test:
      nsize: 5
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: 4_zero
      output_file: output/ex77_4.out
      args: -ksp_converged_reason -ksp_max_it 500 -f ${DATAFILESPATH}/matrices/hpddm/GCRODR/A_400.dat -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5

TEST*/
