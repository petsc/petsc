static char help[] = "Reads a PETSc matrix and vector from a file and solves a linear system.\n\
Input arguments are:\n\
  -f <input_file> : file to load. For example see $PETSC_DIR/share/petsc/datafiles/matrices\n\n";

#include <petscksp.h>
#include <petsclog.h>

static PetscErrorCode KSPTestResidualMonitor(KSP ksp, PetscInt i, PetscReal r, void* ctx)
{
  Vec            *t,*v;
  PetscReal      err;

  PetscFunctionBeginUser;
  CHKERRQ(KSPCreateVecs(ksp,2,&t,2,&v));
  CHKERRQ(KSPBuildResidualDefault(ksp,t[0],v[0],&v[0]));
  CHKERRQ(KSPBuildResidual(ksp,t[1],v[1],&v[1]));
  CHKERRQ(VecAXPY(v[1],-1.0,v[0]));
  CHKERRQ(VecNorm(v[1],NORM_INFINITY,&err));
  PetscCheckFalse(err > PETSC_SMALL,PetscObjectComm((PetscObject)ksp),PETSC_ERR_PLIB,"Inconsistent residual computed at step %D: %g (KSP %g)",i,(double)err,(double)r);
  CHKERRQ(VecDestroyVecs(2,&t));
  CHKERRQ(VecDestroyVecs(2,&v));
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscInt       its;
#if defined(PETSC_USE_LOG)
  PetscLogStage  stage1,stage2;
#endif
  PetscReal      norm;
  Vec            x,b,u;
  Mat            A;
  char           file[PETSC_MAX_PATH_LEN];
  PetscViewer    fd;
  PetscBool      table = PETSC_FALSE,flg,test_residual = PETSC_FALSE,b_in_f = PETSC_TRUE;
  KSP            ksp;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-table",&table,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_residual",&test_residual,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-b_in_f",&b_in_f,NULL));

  /* Read matrix and RHS */
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-f",file,sizeof(file),&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_USER_INPUT,"Must indicate binary file with the -f option");
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,FILE_MODE_READ,&fd));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatLoad(A,fd));
  if (b_in_f) {
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
    CHKERRQ(VecLoad(b,fd));
  } else {
    CHKERRQ(MatCreateVecs(A,NULL,&b));
    CHKERRQ(VecSetRandom(b,NULL));
  }
  CHKERRQ(PetscViewerDestroy(&fd));

  /*
   If the load matrix is larger then the vector, due to being padded
   to match the blocksize then create a new padded vector
  */
  {
    PetscInt    m,n,j,mvec,start,end,indx;
    Vec         tmp;
    PetscScalar *bold;

    CHKERRQ(MatGetLocalSize(A,&m,&n));
    CHKERRQ(VecCreate(PETSC_COMM_WORLD,&tmp));
    CHKERRQ(VecSetSizes(tmp,m,PETSC_DECIDE));
    CHKERRQ(VecSetFromOptions(tmp));
    CHKERRQ(VecGetOwnershipRange(b,&start,&end));
    CHKERRQ(VecGetLocalSize(b,&mvec));
    CHKERRQ(VecGetArray(b,&bold));
    for (j=0; j<mvec; j++) {
      indx = start+j;
      CHKERRQ(VecSetValues(tmp,1,&indx,bold+j,INSERT_VALUES));
    }
    CHKERRQ(VecRestoreArray(b,&bold));
    CHKERRQ(VecDestroy(&b));
    CHKERRQ(VecAssemblyBegin(tmp));
    CHKERRQ(VecAssemblyEnd(tmp));
    b    = tmp;
  }
  CHKERRQ(VecDuplicate(b,&x));
  CHKERRQ(VecDuplicate(b,&u));

  CHKERRQ(VecSet(x,0.0));
  CHKERRQ(PetscBarrier((PetscObject)A));

  CHKERRQ(PetscLogStageRegister("mystage 1",&stage1));
  CHKERRQ(PetscLogStagePush(stage1));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  if (test_residual) {
    CHKERRQ(KSPMonitorSet(ksp,KSPTestResidualMonitor,NULL,NULL));
  }
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPSetUpOnBlocks(ksp));
  CHKERRQ(PetscLogStagePop());
  CHKERRQ(PetscBarrier((PetscObject)A));

  CHKERRQ(PetscLogStageRegister("mystage 2",&stage2));
  CHKERRQ(PetscLogStagePush(stage2));
  CHKERRQ(KSPSolve(ksp,b,x));
  CHKERRQ(PetscLogStagePop());

  /* Show result */
  CHKERRQ(MatMult(A,x,u));
  CHKERRQ(VecAXPY(u,-1.0,b));
  CHKERRQ(VecNorm(u,NORM_2,&norm));
  CHKERRQ(KSPGetIterationNumber(ksp,&its));
  /*  matrix PC   KSP   Options       its    residual  */
  if (table) {
    char        *matrixname,kspinfo[120];
    PetscViewer viewer;
    CHKERRQ(PetscViewerStringOpen(PETSC_COMM_WORLD,kspinfo,sizeof(kspinfo),&viewer));
    CHKERRQ(KSPView(ksp,viewer));
    CHKERRQ(PetscStrrchr(file,'/',&matrixname));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%-8.8s %3D %2.0e %s \n",matrixname,its,norm,kspinfo));
    CHKERRQ(PetscViewerDestroy(&viewer));
  } else {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Number of iterations = %3D\n",its));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Residual norm = %g\n",(double)norm));
  }

  /* Cleanup */
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -ksp_type preonly  -pc_type lu -options_left no  -f ${DATAFILESPATH}/matrices/arco1
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 2
      args: -sub_pc_type ilu -options_left no  -f ${DATAFILESPATH}/matrices/arco1 -ksp_gmres_restart 100 -ksp_gmres_cgs_refinement_type refine_always -sub_ksp_type preonly -pc_type bjacobi -pc_bjacobi_blocks 8 -sub_pc_factor_in_place -ksp_monitor_short
      requires: datafilespath double  !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      suffix: 7
      args: -ksp_gmres_cgs_refinement_type refine_always -pc_type asm -pc_asm_blocks 6 -f ${DATAFILESPATH}/matrices/small -matload_block_size 6  -ksp_monitor_short
      requires: datafilespath double  !complex !defined(PETSC_USE_64BIT_INDICES)

    test:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: 3
      filter: sed -e "s/CONVERGED_RTOL/CONVERGED_ATOL/g"
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_type {{cg groppcg pipecg pipecgrr pipelcg pipeprcg cgne nash stcg gltr fcg pipefcg gmres pipefgmres fgmres lgmres dgmres pgmres tcqmr bcgs ibcgs qmrcgs fbcgs fbcgsr bcgsl pipebcgs cgs tfqmr cr pipecr lsqr qcg bicg minres symmlq lcd gcr pipegcr cgls}} -ksp_max_it 20 -ksp_error_if_not_converged -ksp_converged_reason -test_residual

    test:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      suffix: 3_maxits
      output_file: output/ex6_maxits.out
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_type {{chebyshev cg groppcg pipecg pipecgrr pipelcg pipeprcg cgne nash stcg gltr fcg pipefcg gmres pipefgmres fgmres lgmres dgmres pgmres tcqmr bcgs ibcgs qmrcgs fbcgs fbcgsr bcgsl pipebcgs cgs tfqmr cr pipecr qcg bicg minres symmlq lcd gcr pipegcr cgls richardson}} -ksp_max_it 4 -ksp_error_if_not_converged -ksp_converged_maxits -ksp_converged_reason -test_residual -ksp_norm_type none

    testset:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES)
      output_file: output/ex6_skip.out
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_max_it 8 -ksp_error_if_not_converged -ksp_convergence_test skip -ksp_converged_reason -test_residual
      #SYMMLQ converges in 4 iterations and then generate nans
      test:
        suffix: 3_skip
        args: -ksp_type {{chebyshev cg groppcg pipecg pipecgrr pipelcg pipeprcg cgne nash stcg gltr fcg pipefcg gmres fgmres lgmres dgmres pgmres tcqmr bcgs ibcgs qmrcgs fbcgs fbcgsr bcgsl pipebcgs cgs tfqmr cr pipecr qcg bicg minres lcd gcr cgls richardson}}
      test:
        requires: !pgf90_compiler
        suffix: 3_skip_pipefgmres
        args: -ksp_type pipefgmres
      #PIPEGCR generates nans on linux-knl
      test:
        requires: !defined(PETSC_USE_AVX512_KERNELS)
        suffix: 3_skip_pipegcr
        args: -ksp_type pipegcr
      test:
        requires: hpddm
        suffix: 3_skip_hpddm
        args: -ksp_type hpddm -ksp_hpddm_type {{cg gmres bgmres bcg bfbcg gcrodr bgcrodr}}

    test:
      requires: double !complex !defined(PETSC_USE_64BIT_INDICES) hpddm
      suffix: 3_hpddm
      output_file: output/ex6_3.out
      filter: sed -e "s/CONVERGED_RTOL/CONVERGED_ATOL/g"
      args: -f ${wPETSC_DIR}/share/petsc/datafiles/matrices/spd-real-int32-float64 -pc_type none -ksp_type hpddm -ksp_hpddm_type {{cg gmres bgmres bcg bfbcg gcrodr bgcrodr}} -ksp_max_it 20 -ksp_error_if_not_converged -ksp_converged_reason -test_residual

    # test CG shortcut for residual access
    test:
      suffix: 4
      args: -ksp_converged_reason -ksp_max_it 20 -ksp_converged_maxits -ksp_type {{cg pipecg groppcg}} -ksp_norm_type {{preconditioned unpreconditioned natural}separate output} -pc_type {{bjacobi none}separate output} -f ${DATAFILESPATH}/matrices/poisson_2d13p -b_in_f 0 -test_residual
      requires: datafilespath double !complex !defined(PETSC_USE_64BIT_INDICES)

TEST*/
