#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>

/* E + small_solve */
static PetscErrorCode PCBDDCNullSpaceCorrPreSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  Mat                     K;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0));
  PetscCall(MatMultTranspose(corr_ctx->basis_mat,y,corr_ctx->sw[0]));
  if (corr_ctx->symm) {
    PetscCall(MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]));
  } else {
    PetscCall(MatMultTranspose(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]));
  }
  PetscCall(VecScale(corr_ctx->sw[1],-1.0));
  PetscCall(MatMult(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0]));
  PetscCall(VecScale(corr_ctx->sw[1],-1.0));
  PetscCall(KSPGetOperators(ksp,&K,NULL));
  PetscCall(MatMultAdd(K,corr_ctx->fw[0],y,y));
  PetscCall(PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0));
  PetscFunctionReturn(0);
}

/* E^t + small */
static PetscErrorCode PCBDDCNullSpaceCorrPostSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  Mat                     K;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0));
  PetscCall(KSPGetOperators(ksp,&K,NULL));
  if (corr_ctx->symm) {
    PetscCall(MatMult(K,x,corr_ctx->fw[0]));
  } else {
    PetscCall(MatMultTranspose(K,x,corr_ctx->fw[0]));
  }
  PetscCall(MatMultTranspose(corr_ctx->basis_mat,corr_ctx->fw[0],corr_ctx->sw[0]));
  PetscCall(VecScale(corr_ctx->sw[0],-1.0));
  PetscCall(MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[2]));
  PetscCall(MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[2],x,corr_ctx->fw[0]));
  PetscCall(VecScale(corr_ctx->fw[0],corr_ctx->scale));
  /* Sum contributions from approximate solver and projected system */
  PetscCall(MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0],x));
  PetscCall(PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCNullSpaceCorrDestroy(void * ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;

  PetscFunctionBegin;
  PetscCall(VecDestroyVecs(3,&corr_ctx->sw));
  PetscCall(VecDestroyVecs(1,&corr_ctx->fw));
  PetscCall(MatDestroy(&corr_ctx->basis_mat));
  PetscCall(MatDestroy(&corr_ctx->inv_smat));
  PetscCall(PetscFree(corr_ctx));
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
  PetscCall(KSPGetOperators(local_ksp,&local_mat,&local_pmat));
  PetscCall(MatGetNearNullSpace(local_pmat,&NullSpace));
  if (!NullSpace) {
    if (pcbddc->dbg_flag) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d doesn't have local (near) nullspace: no need for correction in %s solver \n",PetscGlobalRank,isdir ? "Dirichlet" : "Neumann"));
    }
    PetscFunctionReturn(0);
  }
  PetscCall(PetscObjectQuery((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject*)&dmat));
  PetscCheck(dmat,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing dense matrix");
  PetscCall(PetscLogEventBegin(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0));

  PetscCall(PetscNew(&shell_ctx));
  shell_ctx->scale = 1.0;
  PetscCall(PetscObjectReference((PetscObject)dmat));
  shell_ctx->basis_mat = dmat;
  PetscCall(MatGetSize(dmat,NULL,&basis_size));
  shell_ctx->evapply = PC_BDDC_ApproxApply[pcbddc->current_level];

  PetscCall(MatGetOption(local_mat,MAT_SYMMETRIC,&shell_ctx->symm));

  /* explicit construct (Phi^T K Phi)^-1 */
  PetscCall(PetscObjectTypeCompare((PetscObject)local_mat,MATSEQAIJCUSPARSE,&iscusp));
  if (iscusp) {
    PetscCall(MatConvert(shell_ctx->basis_mat,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&shell_ctx->basis_mat));
  }
  PetscCall(MatMatMult(local_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Kbasis_mat));
  PetscCall(MatTransposeMatMult(Kbasis_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&shell_ctx->inv_smat));
  PetscCall(MatDestroy(&Kbasis_mat));
  PetscCall(MatBindToCPU(shell_ctx->inv_smat,PETSC_TRUE));
  PetscCall(MatFindZeroRows(shell_ctx->inv_smat,&zerorows));
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    PetscCall(ISGetLocalSize(zerorows,&nz));
    PetscCall(ISGetIndices(zerorows,&idxs));
    for (i=0;i<nz;i++) {
      PetscCall(MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],1.0,INSERT_VALUES));
    }
    PetscCall(ISRestoreIndices(zerorows,&idxs));
    PetscCall(MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(MatLUFactor(shell_ctx->inv_smat,NULL,NULL,NULL));
  PetscCall(MatSeqDenseInvertFactors_Private(shell_ctx->inv_smat));
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    PetscCall(ISGetLocalSize(zerorows,&nz));
    PetscCall(ISGetIndices(zerorows,&idxs));
    for (i=0;i<nz;i++) {
      PetscCall(MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],0.0,INSERT_VALUES));
    }
    PetscCall(ISRestoreIndices(zerorows,&idxs));
    PetscCall(MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(ISDestroy(&zerorows));

  /* Create work vectors in shell context */
  PetscCall(MatCreateVecs(shell_ctx->inv_smat,&v,NULL));
  PetscCall(KSPCreateVecs(local_ksp,1,&shell_ctx->fw,0,NULL));
  PetscCall(VecDuplicateVecs(v,3,&shell_ctx->sw));
  PetscCall(VecDestroy(&v));

  /* add special pre/post solve to KSP (see [1], eq. 48) */
  PetscCall(KSPSetPreSolve(local_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx));
  PetscCall(KSPSetPostSolve(local_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx));
  PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)local_ksp),&c));
  PetscCall(PetscContainerSetPointer(c,shell_ctx));
  PetscCall(PetscContainerSetUserDestroy(c,PCBDDCNullSpaceCorrDestroy));
  PetscCall(PetscObjectCompose((PetscObject)local_ksp,"_PCBDDC_Null_PrePost_ctx",(PetscObject)c));
  PetscCall(PetscContainerDestroy(&c));

  /* Create ksp object suitable for extreme eigenvalues' estimation */
  if (needscaling || pcbddc->dbg_flag) {
    KSP         check_ksp;
    PC          local_pc;
    Vec         work1,work2;
    const char* prefix;
    PetscReal   test_err,lambda_min,lambda_max;
    PetscInt    k,maxit;

    PetscCall(VecDuplicate(shell_ctx->fw[0],&work1));
    PetscCall(VecDuplicate(shell_ctx->fw[0],&work2));
    PetscCall(KSPCreate(PETSC_COMM_SELF,&check_ksp));
    if (local_mat->spd) {
      PetscCall(KSPSetType(check_ksp,KSPCG));
    }
    PetscCall(PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)local_ksp,0));
    PetscCall(KSPGetOptionsPrefix(local_ksp,&prefix));
    PetscCall(KSPSetOptionsPrefix(check_ksp,prefix));
    PetscCall(KSPAppendOptionsPrefix(check_ksp,"approximate_scale_"));
    PetscCall(KSPSetErrorIfNotConverged(check_ksp,PETSC_FALSE));
    PetscCall(KSPSetOperators(check_ksp,local_mat,local_pmat));
    PetscCall(KSPSetComputeSingularValues(check_ksp,PETSC_TRUE));
    PetscCall(KSPSetPreSolve(check_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx));
    PetscCall(KSPSetPostSolve(check_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx));
    PetscCall(KSPSetTolerances(check_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT));
    PetscCall(KSPSetFromOptions(check_ksp));
    /* setup with default maxit, then set maxit to min(10,any_set_from_command_line) (bug in computing eigenvalues when chaning the number of iterations */
    PetscCall(KSPSetUp(check_ksp));
    PetscCall(KSPGetPC(local_ksp,&local_pc));
    PetscCall(KSPSetPC(check_ksp,local_pc));
    PetscCall(KSPGetTolerances(check_ksp,NULL,NULL,NULL,&maxit));
    PetscCall(KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PetscMin(10,maxit)));
    PetscCall(VecSetRandom(work2,NULL));
    PetscCall(MatMult(local_mat,work2,work1));
    PetscCall(KSPSolve(check_ksp,work1,work1));
    PetscCall(KSPCheckSolve(check_ksp,pc,work1));
    PetscCall(VecAXPY(work1,-1.,work2));
    PetscCall(VecNorm(work1,NORM_INFINITY,&test_err));
    PetscCall(KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min));
    PetscCall(KSPGetIterationNumber(check_ksp,&k));
    if (pcbddc->dbg_flag) {
      if (isdir) {
        PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max));
      } else {
        PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max));
      }
    }
    if (needscaling) shell_ctx->scale = 1.0/lambda_max;

    if (needscaling && pcbddc->dbg_flag) { /* test for scaling factor */
      PC new_pc;

      PetscCall(VecSetRandom(work2,NULL));
      PetscCall(MatMult(local_mat,work2,work1));
      PetscCall(PCCreate(PetscObjectComm((PetscObject)check_ksp),&new_pc));
      PetscCall(PCSetType(new_pc,PCKSP));
      PetscCall(PCSetOperators(new_pc,local_mat,local_pmat));
      PetscCall(PCKSPSetKSP(new_pc,local_ksp));
      PetscCall(KSPSetPC(check_ksp,new_pc));
      PetscCall(PCDestroy(&new_pc));
      PetscCall(KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,maxit));
      PetscCall(KSPSetPreSolve(check_ksp,NULL,NULL));
      PetscCall(KSPSetPostSolve(check_ksp,NULL,NULL));
      PetscCall(KSPSolve(check_ksp,work1,work1));
      PetscCall(KSPCheckSolve(check_ksp,pc,work1));
      PetscCall(VecAXPY(work1,-1.,work2));
      PetscCall(VecNorm(work1,NORM_INFINITY,&test_err));
      PetscCall(KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min));
      PetscCall(KSPGetIterationNumber(check_ksp,&k));
      if (pcbddc->dbg_flag) {
        if (isdir) {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max));
        } else {
          PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max));
        }
      }
    }
    PetscCall(KSPDestroy(&check_ksp));
    PetscCall(VecDestroy(&work1));
    PetscCall(VecDestroy(&work2));
  }
  PetscCall(PetscLogEventEnd(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0));

  if (pcbddc->dbg_flag) {
    Vec       work1,work2,work3;
    PetscReal test_err;

    /* check nullspace basis is solved exactly */
    PetscCall(VecDuplicate(shell_ctx->fw[0],&work1));
    PetscCall(VecDuplicate(shell_ctx->fw[0],&work2));
    PetscCall(VecDuplicate(shell_ctx->fw[0],&work3));
    PetscCall(VecSetRandom(shell_ctx->sw[0],NULL));
    PetscCall(MatMult(shell_ctx->basis_mat,shell_ctx->sw[0],work1));
    PetscCall(VecCopy(work1,work2));
    PetscCall(MatMult(local_mat,work1,work3));
    PetscCall(KSPSolve(local_ksp,work3,work1));
    PetscCall(VecAXPY(work1,-1.,work2));
    PetscCall(VecNorm(work1,NORM_INFINITY,&test_err));
    if (isdir) {
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err));
    } else {
      PetscCall(PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err));
    }
    PetscCall(VecDestroy(&work1));
    PetscCall(VecDestroy(&work2));
    PetscCall(VecDestroy(&work3));
  }
  PetscFunctionReturn(0);
}
