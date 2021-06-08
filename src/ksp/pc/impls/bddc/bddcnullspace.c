#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>

/* E + small_solve */
static PetscErrorCode PCBDDCNullSpaceCorrPreSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  Mat                     K;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0);CHKERRQ(ierr);
  ierr = MatMultTranspose(corr_ctx->basis_mat,y,corr_ctx->sw[0]);CHKERRQ(ierr);
  if (corr_ctx->symm) {
    ierr = MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[1]);CHKERRQ(ierr);
  }
  ierr = VecScale(corr_ctx->sw[1],-1.0);CHKERRQ(ierr);
  ierr = MatMult(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0]);CHKERRQ(ierr);
  ierr = VecScale(corr_ctx->sw[1],-1.0);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&K,NULL);CHKERRQ(ierr);
  ierr = MatMultAdd(K,corr_ctx->fw[0],y,y);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* E^t + small */
static PetscErrorCode PCBDDCNullSpaceCorrPostSolve(KSP ksp,Vec y,Vec x, void* ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  PetscErrorCode          ierr;
  Mat                     K;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(corr_ctx->evapply,ksp,0,0,0);CHKERRQ(ierr);
  ierr = KSPGetOperators(ksp,&K,NULL);CHKERRQ(ierr);
  if (corr_ctx->symm) {
    ierr = MatMult(K,x,corr_ctx->fw[0]);CHKERRQ(ierr);
  } else {
    ierr = MatMultTranspose(K,x,corr_ctx->fw[0]);CHKERRQ(ierr);
  }
  ierr = MatMultTranspose(corr_ctx->basis_mat,corr_ctx->fw[0],corr_ctx->sw[0]);CHKERRQ(ierr);
  ierr = VecScale(corr_ctx->sw[0],-1.0);CHKERRQ(ierr);
  ierr = MatMult(corr_ctx->inv_smat,corr_ctx->sw[0],corr_ctx->sw[2]);CHKERRQ(ierr);
  ierr = MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[2],x,corr_ctx->fw[0]);CHKERRQ(ierr);
  ierr = VecScale(corr_ctx->fw[0],corr_ctx->scale);CHKERRQ(ierr);
  /* Sum contributions from approximate solver and projected system */
  ierr = MatMultAdd(corr_ctx->basis_mat,corr_ctx->sw[1],corr_ctx->fw[0],x);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(corr_ctx->evapply,ksp,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCNullSpaceCorrDestroy(void * ctx)
{
  NullSpaceCorrection_ctx corr_ctx = (NullSpaceCorrection_ctx)ctx;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = VecDestroyVecs(3,&corr_ctx->sw);CHKERRQ(ierr);
  ierr = VecDestroyVecs(1,&corr_ctx->fw);CHKERRQ(ierr);
  ierr = MatDestroy(&corr_ctx->basis_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&corr_ctx->inv_smat);CHKERRQ(ierr);
  ierr = PetscFree(corr_ctx);CHKERRQ(ierr);
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
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  if (isdir) local_ksp = pcbddc->ksp_D; /* Dirichlet solver */
  else local_ksp = pcbddc->ksp_R; /* Neumann solver */
  ierr = KSPGetOperators(local_ksp,&local_mat,&local_pmat);CHKERRQ(ierr);
  ierr = MatGetNearNullSpace(local_pmat,&NullSpace);CHKERRQ(ierr);
  if (!NullSpace) {
    if (pcbddc->dbg_flag) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d doesn't have local (near) nullspace: no need for correction in %s solver \n",PetscGlobalRank,isdir ? "Dirichlet" : "Neumann");CHKERRQ(ierr);
    }
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectQuery((PetscObject)NullSpace,"_PBDDC_Null_dmat",(PetscObject*)&dmat);CHKERRQ(ierr);
  if (!dmat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing dense matrix");
  ierr = PetscLogEventBegin(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  ierr = PetscNew(&shell_ctx);CHKERRQ(ierr);
  shell_ctx->scale = 1.0;
  ierr = PetscObjectReference((PetscObject)dmat);CHKERRQ(ierr);
  shell_ctx->basis_mat = dmat;
  ierr = MatGetSize(dmat,NULL,&basis_size);CHKERRQ(ierr);
  shell_ctx->evapply = PC_BDDC_ApproxApply[pcbddc->current_level];

  ierr = MatGetOption(local_mat,MAT_SYMMETRIC,&shell_ctx->symm);CHKERRQ(ierr);

  /* explicit construct (Phi^T K Phi)^-1 */
  ierr = PetscObjectTypeCompare((PetscObject)local_mat,MATSEQAIJCUSPARSE,&iscusp);CHKERRQ(ierr);
  if (iscusp) {
    ierr = MatConvert(shell_ctx->basis_mat,MATSEQDENSECUDA,MAT_INPLACE_MATRIX,&shell_ctx->basis_mat);CHKERRQ(ierr);
  }
  ierr = MatMatMult(local_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Kbasis_mat);CHKERRQ(ierr);
  ierr = MatTransposeMatMult(Kbasis_mat,shell_ctx->basis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&shell_ctx->inv_smat);CHKERRQ(ierr);
  ierr = MatDestroy(&Kbasis_mat);CHKERRQ(ierr);
  ierr = MatBindToCPU(shell_ctx->inv_smat,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatFindZeroRows(shell_ctx->inv_smat,&zerorows);CHKERRQ(ierr);
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    ierr = ISGetLocalSize(zerorows,&nz);CHKERRQ(ierr);
    ierr = ISGetIndices(zerorows,&idxs);CHKERRQ(ierr);
    for (i=0;i<nz;i++) {
      ierr = MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(zerorows,&idxs);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = MatLUFactor(shell_ctx->inv_smat,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = MatSeqDenseInvertFactors_Private(shell_ctx->inv_smat);CHKERRQ(ierr);
  if (zerorows) { /* linearly dependent basis */
    const PetscInt *idxs;
    PetscInt       i,nz;

    ierr = ISGetLocalSize(zerorows,&nz);CHKERRQ(ierr);
    ierr = ISGetIndices(zerorows,&idxs);CHKERRQ(ierr);
    for (i=0;i<nz;i++) {
      ierr = MatSetValue(shell_ctx->inv_smat,idxs[i],idxs[i],0.0,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(zerorows,&idxs);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(shell_ctx->inv_smat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&zerorows);CHKERRQ(ierr);

  /* Create work vectors in shell context */
  ierr = MatCreateVecs(shell_ctx->inv_smat,&v,NULL);CHKERRQ(ierr);
  ierr = KSPCreateVecs(local_ksp,1,&shell_ctx->fw,0,NULL);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(v,3,&shell_ctx->sw);CHKERRQ(ierr);
  ierr = VecDestroy(&v);CHKERRQ(ierr);

  /* add special pre/post solve to KSP (see [1], eq. 48) */
  ierr = KSPSetPreSolve(local_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx);CHKERRQ(ierr);
  ierr = KSPSetPostSolve(local_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx);CHKERRQ(ierr);
  ierr = PetscContainerCreate(PetscObjectComm((PetscObject)local_ksp),&c);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(c,shell_ctx);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(c,PCBDDCNullSpaceCorrDestroy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)local_ksp,"_PCBDDC_Null_PrePost_ctx",(PetscObject)c);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(&c);CHKERRQ(ierr);

  /* Create ksp object suitable for extreme eigenvalues' estimation */
  if (needscaling || pcbddc->dbg_flag) {
    KSP         check_ksp;
    PC          local_pc;
    Vec         work1,work2;
    const char* prefix;
    PetscReal   test_err,lambda_min,lambda_max;
    PetscInt    k,maxit;

    ierr = VecDuplicate(shell_ctx->fw[0],&work1);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->fw[0],&work2);CHKERRQ(ierr);
    ierr = KSPCreate(PETSC_COMM_SELF,&check_ksp);CHKERRQ(ierr);
    if (local_mat->spd) {
      ierr = KSPSetType(check_ksp,KSPCG);CHKERRQ(ierr);
    }
    ierr = PetscObjectIncrementTabLevel((PetscObject)check_ksp,(PetscObject)local_ksp,0);CHKERRQ(ierr);
    ierr = KSPGetOptionsPrefix(local_ksp,&prefix);CHKERRQ(ierr);
    ierr = KSPSetOptionsPrefix(check_ksp,prefix);CHKERRQ(ierr);
    ierr = KSPAppendOptionsPrefix(check_ksp,"approximate_scale_");CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(check_ksp,PETSC_FALSE);CHKERRQ(ierr);
    ierr = KSPSetOperators(check_ksp,local_mat,local_pmat);CHKERRQ(ierr);
    ierr = KSPSetComputeSingularValues(check_ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = KSPSetPreSolve(check_ksp,PCBDDCNullSpaceCorrPreSolve,shell_ctx);CHKERRQ(ierr);
    ierr = KSPSetPostSolve(check_ksp,PCBDDCNullSpaceCorrPostSolve,shell_ctx);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,PETSC_SMALL,PETSC_SMALL,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(check_ksp);CHKERRQ(ierr);
    /* setup with default maxit, then set maxit to min(10,any_set_from_command_line) (bug in computing eigenvalues when chaning the number of iterations */
    ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(local_ksp,&local_pc);CHKERRQ(ierr);
    ierr = KSPSetPC(check_ksp,local_pc);CHKERRQ(ierr);
    ierr = KSPGetTolerances(check_ksp,NULL,NULL,NULL,&maxit);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,PetscMin(10,maxit));CHKERRQ(ierr);
    ierr = VecSetRandom(work2,NULL);CHKERRQ(ierr);
    ierr = MatMult(local_mat,work2,work1);CHKERRQ(ierr);
    ierr = KSPSolve(check_ksp,work1,work1);CHKERRQ(ierr);
    ierr = KSPCheckSolve(check_ksp,pc,work1);CHKERRQ(ierr);
    ierr = VecAXPY(work1,-1.,work2);CHKERRQ(ierr);
    ierr = VecNorm(work1,NORM_INFINITY,&test_err);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
    if (pcbddc->dbg_flag) {
      if (isdir) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (no scale) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
      }
    }
    if (needscaling) shell_ctx->scale = 1.0/lambda_max;

    if (needscaling && pcbddc->dbg_flag) { /* test for scaling factor */
      PC new_pc;

      ierr = VecSetRandom(work2,NULL);CHKERRQ(ierr);
      ierr = MatMult(local_mat,work2,work1);CHKERRQ(ierr);
      ierr = PCCreate(PetscObjectComm((PetscObject)check_ksp),&new_pc);CHKERRQ(ierr);
      ierr = PCSetType(new_pc,PCKSP);CHKERRQ(ierr);
      ierr = PCSetOperators(new_pc,local_mat,local_pmat);CHKERRQ(ierr);
      ierr = PCKSPSetKSP(new_pc,local_ksp);CHKERRQ(ierr);
      ierr = KSPSetPC(check_ksp,new_pc);CHKERRQ(ierr);
      ierr = PCDestroy(&new_pc);CHKERRQ(ierr);
      ierr = KSPSetTolerances(check_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,maxit);CHKERRQ(ierr);
      ierr = KSPSetPreSolve(check_ksp,NULL,NULL);CHKERRQ(ierr);
      ierr = KSPSetPostSolve(check_ksp,NULL,NULL);CHKERRQ(ierr);
      ierr = KSPSolve(check_ksp,work1,work1);CHKERRQ(ierr);
      ierr = KSPCheckSolve(check_ksp,pc,work1);CHKERRQ(ierr);
      ierr = VecAXPY(work1,-1.,work2);CHKERRQ(ierr);
      ierr = VecNorm(work1,NORM_INFINITY,&test_err);CHKERRQ(ierr);
      ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
      if (pcbddc->dbg_flag) {
        if (isdir) {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (scale %g) %1.14e (it %D, eigs %1.6e %1.6e)\n",PetscGlobalRank,(double)PetscRealPart(shell_ctx->scale),test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
        }
      }
    }
    ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&work1);CHKERRQ(ierr);
    ierr = VecDestroy(&work2);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(PC_BDDC_ApproxSetUp[pcbddc->current_level],pc,0,0,0);CHKERRQ(ierr);

  if (pcbddc->dbg_flag) {
    Vec       work1,work2,work3;
    PetscReal test_err;

    /* check nullspace basis is solved exactly */
    ierr = VecDuplicate(shell_ctx->fw[0],&work1);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->fw[0],&work2);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->fw[0],&work3);CHKERRQ(ierr);
    ierr = VecSetRandom(shell_ctx->sw[0],NULL);CHKERRQ(ierr);
    ierr = MatMult(shell_ctx->basis_mat,shell_ctx->sw[0],work1);CHKERRQ(ierr);
    ierr = VecCopy(work1,work2);CHKERRQ(ierr);
    ierr = MatMult(local_mat,work1,work3);CHKERRQ(ierr);
    ierr = KSPSolve(local_ksp,work3,work1);CHKERRQ(ierr);
    ierr = VecAXPY(work1,-1.,work2);CHKERRQ(ierr);
    ierr = VecNorm(work1,NORM_INFINITY,&test_err);CHKERRQ(ierr);
    if (isdir) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&work1);CHKERRQ(ierr);
    ierr = VecDestroy(&work2);CHKERRQ(ierr);
    ierr = VecDestroy(&work3);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
