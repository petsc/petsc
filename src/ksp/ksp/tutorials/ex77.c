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
  const PetscScalar  *b;
  PetscScalar        *x,*S = NULL,*T = NULL;
  PetscReal          norm;
  PetscInt           m,M,N = 5,i,j;
  const char         *deft = MATAIJ;
  PetscViewer        viewer;
  char               dir[PETSC_MAX_PATH_LEN],name[256],type[256];
  PetscBool          breakdown = PETSC_FALSE,flg;
  KSPConvergedReason reason;
  PetscErrorCode     ierr;

  ierr = PetscInitialize(&argc,&args,NULL,help);if (ierr) return ierr;
  ierr = PetscStrcpy(dir,".");CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-load_dir",dir,sizeof(dir),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-breakdown",&breakdown,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = PetscSNPrintf(name,sizeof(name),"%s/A_400.dat",dir);CHKERRQ(ierr);
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
    }
    else {
      ierr = PetscCalloc2(N*N,&S,N*N,&T);CHKERRQ(ierr);
      for (i=0; i<N; i++) {
        for (j=0; j<N; j++) {
          S[i*(N+1)] = 1e+6; /* really easy problem used for testing */
          T[i*(N+1)] = 1e-2;
        }
      }
      ierr = MatCreateKAIJ(A,N,N,S,T,&KA);CHKERRQ(ierr);
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
      if (reason != KSP_DIVERGED_BREAKDOWN) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"KSPConvergedReason() %s != KSP_DIVERGED_BREAKDOWN",KSPConvergedReasons[reason]);
    }
  } else {
    ierr = KSPSetOperators(ksp,KA,KA);CHKERRQ(ierr);
    ierr = MatGetSize(KA,&M,NULL);CHKERRQ(ierr);
    /* from column- to row-major to be consistent with MatKAIJ format */
    ierr = MatTranspose(B,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    ierr = MatDenseGetArrayRead(B,&b);CHKERRQ(ierr);
    ierr = MatDenseGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m*N,M,b,&cb);CHKERRQ(ierr);
    ierr = VecCreateMPIWithArray(PETSC_COMM_WORLD,1,m*N,M,x,&cx);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,cb,cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cx);CHKERRQ(ierr);
    ierr = VecDestroy(&cb);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = MatDenseRestoreArrayRead(B,&b);CHKERRQ(ierr);
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
      args: -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -mat_type {{aij sbaij}shared output}
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
      args: -N 6 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -pc_type lu -ksp_type hpddm -ksp_hpddm_type preonly

   # to avoid breakdown failures, use -ksp_hpddm_deflation_tol, cf. KSPHPDDM documentation
   test:
      nsize: 1
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      suffix: breakdown
      output_file: output/ex77_preonly.out
      args: -N 3 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -pc_type none -ksp_type hpddm -ksp_hpddm_type {{bcg bgmres bgcrodr bfbcg}shared output} -breakdown

   test:
      nsize: 2
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES)
      args: -N 12 -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -mat_type kaij -pc_type pbjacobi -ksp_type hpddm -ksp_hpddm_type {{gmres bgmres}separate output}

   test:
      nsize: 4
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) slepc
      suffix: 4_slepc
      output_file: output/ex77_4.out
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type {{aij dense}shared output} -ksp_hpddm_recycle_eps_converged_reason -ksp_hpddm_recycle_st_pc_type redundant

   test:
      nsize: 4
      requires: hpddm datafilespath double !complex !define(PETSC_USE_64BIT_INDICES) slepc elemental
      suffix: 4_elemental
      output_file: output/ex77_4.out
      filter: sed "/^ksp_hpddm_recycle_ Linear eigensolve converged/d"
      args: -ksp_converged_reason -ksp_max_it 500 -load_dir ${DATAFILESPATH}/matrices/hpddm/GCRODR -ksp_rtol 1e-4 -ksp_type hpddm -ksp_hpddm_recycle 5 -ksp_hpddm_type bgcrodr -ksp_view_final_residual -N 12 -ksp_matsolve_block_size 5 -ksp_hpddm_recycle_redistribute 2 -ksp_hpddm_recycle_mat_type elemental -ksp_hpddm_recycle_eps_converged_reason

TEST*/
