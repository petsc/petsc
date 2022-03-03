#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>

/* E + small_solve */
static PetscErrorCode PCBDDCNullSpaceCorrPreSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  Mat                     K;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0));
  CHKERRQ(MatMultTranspose(corr_ctx->basis_mat,y,corr_ctx->sw[0]));
  if (corr_ctx->symm) {
    CHKERRQ(MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]));
  } else {
    CHKERRQ(MatMultTranspose(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]));
  }
  CHKERRQ(VecScale(corr_ctx->sw[1],-1.0));
  CHKERRQ(MatMult(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0]));
  CHKERRQ(VecScale(corr_ctx->sw[1],-1.0));
  CHKERRQ(KSPGetOperators(ksp,&K,NULL));
  CHKERRQ(MatMultAdd(K,corr_ctx->fw[0],y,y));
  CHKERRQ(PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0));
  PetscFunctionReturn(0);
}

/* E^t + small */
static PetscErrorCode PCBDDCNullSpaceCorrPostSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  Mat                     K;

  PetscFunctionBegin;
  CHKERRQ(PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0));
  CHKERRQ(KSPGetOperators(ksp,&K,NULL));
  if (corr_ctx->symm) {
    CHKERRQ(MatMult(K,x,corr_ctx->fw[0]));
  } else {
    CHKERRQ(MatMultTranspose(K,x,corr_ctx->fw[0]));
  }
  CHKERRQ(MatMultTranspose(corr_ctx->basis_mat,corr_ctx->fw[0],corr_ctx->sw[0]));
  CHKERRQ(VecScale(corr_ctx->sw[0],-1.0));
  CHKERRQ(MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[2]));
  CHKERRQ(MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[2],x,corr_ctx->fw[0]));
  CHKERRQ(VecScale(corr_ctx->fw[0],corr_ctx->scale));
  /* Sum contributions from approximate solver and projected system */
  CHKERRQ(MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0],x));
  CHKERRQ(PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCNullSpaceCorrDestroy(void * ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;

  PetscFunctionBegin;
  CHKERRQ(VecDestroyVecs(3,&corr_ctx->sw));
  CHKERRQ(VecDestroyVecs(1,&corr_ctx->fw));
  CHKERRQ(MatDestroy(&corr_ctx->basis_mat));
  CHKERRQ(MatDestroy(&corr_ctx->inv_smat));
  CHKERRQ(PetscFree(corr_ctx));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC pc, PetscBool isdir, PetscBool needscaling)
{
  PC_BDDC                  *pcbddc = (PC_BDDC*)pc->data;
  MatNullSpace             NullSpace = NULL;
  KSP                      local_ksp;
  NullSpaceCorrection_ctx  shell_ctx;
  Mat                      local_mat,local_pmat,dmat,Kbasis_mat;
  Vec                      v;
  PetscContainer           c;
  PetscInt                 basis_size;
  IS                       zerorows;
  PetscBool                iscusp;

  PetscFunctionBegin;
  if (isdir) local_ksp = pcbddc->ksp_D; /* Dirichlet solver */
  else local_ksp = pcbddc->ksp_R; /* Neumann solver */
  CHKERRQ(KSPGetOperators(local_ksp,&local_mat,&local_pmat));
  CHKERRQ(MatGetNearNullSpace(local_pmat,&NullSpace));
  if (!NullSpace) {
    if (pcbddc->dbg_flag) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d doesn't have local (near) nullspace: no need for correction in %s solver \n",PetscGlobalRank,isdir ? "Dirichlet" : "Neumann"));
    }
    PetscFunctionReturn(0);
  }
  CHKERRQ(PetscObjectQuery((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject*)&dmat));
  PetscCheck(dmat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing dense matrix");
  CHKERRQ(PetscLogEventBegin(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0));

  CHKERRQ(PetscNew(&shell_ctx));
  shell_ctx->scale = 1.0;
  CHKERRQ(PetscObjectReference((PetscObject)dmat));
  shell_ctx->basis_mat = dmat;
  CHKERRQ(MatGetSize(dmat,NULL,&basis_size));
  shell_ctx->evapply = PC_BDDC_ApproxApply[pcbddc->current_level];

  CHKERRQ(MatGetOption(local_mat,MAT_SYMMETRIC,&shell_ctx->symm));

  /* explicit construct (Phi^T K Phi)^-1 */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQAIJCUSPARSE,&iscusp));
  if (iscusp) {
    CHKERRQ(MatConvert(shell_ctx->basis_mat,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&shell_ctx->basis_mat));
  }
  CHKERRQ(MatMatMult(local_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Kbasis_mat));
  CHKERRQ(MatTransposeMatMult(Kbasis_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&shell_ctx->inv_smat));
  CHKERRQ(MatDestroy(&Kbasis_mat));
  CHKERRQ(MatBindToCPU(shell_ctx->inv_smat,PETSC_TRUE));
  CHKERRQ(MatFindZeroRows(shell_ctx->inv_smat,&zerorows));
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    CHKERRQ(ISGetLocalSize(zerorows,&nz));
    CHKERRQ(ISGetIndices(zerorows,&idxs));
    for (i=0;i<nz;i++) {
      CHKERRQ(MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],1.0,INSERT_VALUES));
    }
    CHKERRQ(ISRestoreIndices(zerorows,&idxs));
    CHKERRQ(MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(MatLUFactor(shell_ctx->inv_smat,NULL,NULL,NULL));
  CHKERRQ(MatSeqDenseInvertFactors_Private(shell_ctx->inv_smat));
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    CHKERRQ(ISGetLocalSize(zerorows,&nz));
    CHKERRQ(ISGetIndices(zerorows,&idxs));
    for (i=0;i<nz;i++) {
      CHKERRQ(MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],0.0,INSERT_VALUES));
    }
    CHKERRQ(ISRestoreIndices(zerorows,&idxs));
    CHKERRQ(MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(ISDestroy(&zerorows));

  /* Create work vectors in shell context */
  CHKERRQ(MatCreateVecs(shell_ctx->inv_smat,&v,NULL));
  CHKERRQ(KSPCreateVecs(local_ksp,1,&shell_ctx->fw,0,NULL));
  CHKERRQ(VecDuplicateVecs(v,3,&shell_ctx->sw));
  CHKERRQ(VecDestroy(&v));

  /* add special pre/post solve to KSP (see [1], eq. 48) */
  CHKERRQ(KSPSetPreSolve(local_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx));
  CHKERRQ(KSPSetPostSolve(local_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx));
  CHKERRQ(PetscContainerCreate(PetscObjectComm((PetscObject)local_ksp),&c));
  CHKERRQ(PetscContainerSetPointer(c,shell_ctx));
  CHKERRQ(PetscContainerSetUserDestroy(c,PCBDDCNullSpaceCorrDestroy));
  CHKERRQ(PetscObjectCompose((PetscObject)local_ksp,"_PCBDDC_Null_PrePost_ctx",(PetscObject)c));
  CHKERRQ(PetscContainerDestroy(&c));

  /* Create ksp object suitable for extreme eigenvalues' estimation */
  if (needscaling || pcbddc->dbg_flag) {
    KSP         check_ksp;
    PC          local_pc;
    Vec         work1,work2;
    const char* prefix;
    PetscReal   test_err,lambda_min,lambda_max;
    PetscInt    k,maxit;

    CHKERRQ(VecDuplicate(shell_ctx->fw[0],&work1));
    CHKERRQ(VecDuplicate(shell_ctx->fw[0],&work2));
    CHKERRQ(KSPCreate(PETSC_COMM_SELF,&check_ksp));
    if (local_mat->spd) {
      CHKERRQ(KSPSetType(check_ksp,KSPCG));
    }
    CHKERRQ(PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)local_ksp,0));
    CHKERRQ(KSPGetOptionsPrefix(local_ksp,&prefix));
    CHKERRQ(KSPSetOptionsPrefix(check_ksp,prefix));
    CHKERRQ(KSPAppendOptionsPrefix(check_ksp,"approximate_scale_"));
    CHKERRQ(KSPSetErrorIfNotConverged(check_ksp,PETSC_FALSE));
    CHKERRQ(KSPSetOperators(check_ksp,local_mat,local_pmat));
    CHKERRQ(KSPSetComputeSingularValues(check_ksp,PETSC_TRUE));
    CHKERRQ(KSPSetPreSolve(check_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx));
    CHKERRQ(KSPSetPostSolve(check_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx));
    CHKERRQ(KSPSetTolerances(check_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT));
    CHKERRQ(KSPSetFromOptions(check_ksp));
    /* setup with default maxit, then set maxit to min(10,any_set_from_command_line) (bug in computing eigenvalues when chaning the number of iterations */
    CHKERRQ(KSPSetUp(check_ksp));
    CHKERRQ(KSPGetPC(local_ksp,&local_pc));
    CHKERRQ(KSPSetPC(check_ksp,local_pc));
    CHKERRQ(KSPGetTolerances(check_ksp,NULL,NULL,NULL,&maxit));
    CHKERRQ(KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PetscMin(10,maxit)));
    CHKERRQ(VecSetRandom(work2,NULL));
    CHKERRQ(MatMult(local_mat,work2,work1));
    CHKERRQ(KSPSolve(check_ksp,work1,work1));
    CHKERRQ(KSPCheckSolve(check_ksp,pc,work1));
    CHKERRQ(VecAXPY(work1,-1.,work2));
    CHKERRQ(VecNorm(work1,NORM_INFINITY,&test_err));
    CHKERRQ(KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min));
    CHKERRQ(KSPGetIterationNumber(check_ksp,&k));
    if (pcbddc->dbg_flag) {
      if (isdir) {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max));
      } else {
        CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max));
      }
    }
    if (needscaling) shell_ctx->scale = 1.0/lambda_max;

    if (needscaling && pcbddc->dbg_flag) { /* test for scaling factor */
      PC new_pc;

      CHKERRQ(VecSetRandom(work2,NULL));
      CHKERRQ(MatMult(local_mat,work2,work1));
      CHKERRQ(PCCreate(PetscObjectComm((PetscObject)check_ksp),&new_pc));
      CHKERRQ(PCSetType(new_pc,PCKSP));
      CHKERRQ(PCSetOperators(new_pc,local_mat,local_pmat));
      CHKERRQ(PCKSPSetKSP(new_pc,local_ksp));
      CHKERRQ(KSPSetPC(check_ksp,new_pc));
      CHKERRQ(PCDestroy(&new_pc));
      CHKERRQ(KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,maxit));
      CHKERRQ(KSPSetPreSolve(check_ksp,NULL,NULL));
      CHKERRQ(KSPSetPostSolve(check_ksp,NULL,NULL));
      CHKERRQ(KSPSolve(check_ksp,work1,work1));
      CHKERRQ(KSPCheckSolve(check_ksp,pc,work1));
      CHKERRQ(VecAXPY(work1,-1.,work2));
      CHKERRQ(VecNorm(work1,NORM_INFINITY,&test_err));
      CHKERRQ(KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min));
      CHKERRQ(KSPGetIterationNumber(check_ksp,&k));
      if (pcbddc->dbg_flag) {
        if (isdir) {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max));
        } else {
          CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max));
        }
      }
    }
    CHKERRQ(KSPDestroy(&check_ksp));
    CHKERRQ(VecDestroy(&work1));
    CHKERRQ(VecDestroy(&work2));
  }
  CHKERRQ(PetscLogEventEnd(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0));

  if (pcbddc->dbg_flag) {
    Vec       work1,work2,work3;
    PetscReal test_err;

    /* check nullspace basis is solved exactly */
    CHKERRQ(VecDuplicate(shell_ctx->fw[0],&work1));
    CHKERRQ(VecDuplicate(shell_ctx->fw[0],&work2));
    CHKERRQ(VecDuplicate(shell_ctx->fw[0],&work3));
    CHKERRQ(VecSetRandom(shell_ctx->sw[0],NULL));
    CHKERRQ(MatMult(shell_ctx->basis_mat,shell_ctx->sw[0],work1));
    CHKERRQ(VecCopy(work1,work2));
    CHKERRQ(MatMult(local_mat,work1,work3));
    CHKERRQ(KSPSolve(local_ksp,work3,work1));
    CHKERRQ(VecAXPY(work1,-1.,work2));
    CHKERRQ(VecNorm(work1,NORM_INFINITY,&test_err));
    if (isdir) {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err));
    } else {
      CHKERRQ(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err));
    }
    CHKERRQ(VecDestroy(&work1));
    CHKERRQ(VecDestroy(&work2));
    CHKERRQ(VecDestroy(&work3));
  }
  PetscFunctionReturn(0);
}
