#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

static PetscErrorCode PCBDDCApplyNullSpaceCorrectionPC(PC pc,Vec x,Vec y)
{
  NullSpaceCorrection_ctx pc_ctx;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pc_ctx);CHKERRQ(ierr);
  /* E */
  ierr = MatMultTranspose(pc_ctx->Lbasis_mat,x,pc_ctx->work_small_2);CHKERRQ(ierr);
  ierr = MatMultAdd(pc_ctx->Kbasis_mat,pc_ctx->work_small_2,x,pc_ctx->work_full_1);CHKERRQ(ierr);
  /* P^-1 */
  ierr = PCApply(pc_ctx->local_pc,pc_ctx->work_full_1,pc_ctx->work_full_2);CHKERRQ(ierr);
  /* E^T */
  ierr = MatMultTranspose(pc_ctx->Kbasis_mat,pc_ctx->work_full_2,pc_ctx->work_small_1);CHKERRQ(ierr);
  ierr = VecScale(pc_ctx->work_small_1,-1.0);CHKERRQ(ierr);
  ierr = MatMultAdd(pc_ctx->Lbasis_mat,pc_ctx->work_small_1,pc_ctx->work_full_2,pc_ctx->work_full_1);CHKERRQ(ierr);
  /* Sum contributions */
  ierr = MatMultAdd(pc_ctx->basis_mat,pc_ctx->work_small_2,pc_ctx->work_full_1,y);CHKERRQ(ierr);
  if (pc_ctx->apply_scaling) {
    ierr = VecScale(y,pc_ctx->scale);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCDestroyNullSpaceCorrectionPC(PC pc)
{
  NullSpaceCorrection_ctx pc_ctx;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void**)&pc_ctx);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->work_small_1);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->work_small_2);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->work_full_1);CHKERRQ(ierr);
  ierr = VecDestroy(&pc_ctx->work_full_2);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->basis_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->Lbasis_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&pc_ctx->Kbasis_mat);CHKERRQ(ierr);
  ierr = PCDestroy(&pc_ctx->local_pc);CHKERRQ(ierr);
  ierr = PetscFree(pc_ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC pc, PetscBool isdir, PetscBool needscaling)
{
  PC_BDDC                  *pcbddc = (PC_BDDC*)pc->data;
  PC_IS                    *pcis = (PC_IS*)pc->data;
  Mat_IS                   *matis = (Mat_IS*)pc->pmat->data;
  MatNullSpace             NullSpace = NULL;
  KSP                      local_ksp;
  PC                       newpc;
  NullSpaceCorrection_ctx  shell_ctx;
  Mat                      local_mat,local_pmat,small_mat,inv_small_mat;
  const Vec                *nullvecs;
  VecScatter               scatter_ctx;
  IS                       is_aux,local_dofs;
  MatFactorInfo            matinfo;
  PetscScalar              *basis_mat,*Kbasis_mat,*array,*array_mat;
  PetscScalar              one = 1.0,zero = 0.0, m_one = -1.0;
  PetscInt                 basis_dofs,basis_size,nnsp_size,i,k;
  PetscBool                nnsp_has_cnst;
  PetscReal                test_err,lambda_min,lambda_max;
  PetscBool                setsym,issym=PETSC_FALSE;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = MatGetNullSpace(matis->A,&NullSpace);CHKERRQ(ierr);
  if (!NullSpace) PetscFunctionReturn(0);
  /* Infer the local solver */
  if (isdir) {
    /* Dirichlet solver */
    local_ksp = pcbddc->ksp_D;
    local_dofs = pcis->is_I_local;
  } else {
    /* Neumann solver */
    local_ksp = pcbddc->ksp_R;
    local_dofs = pcbddc->is_R_local;
  }
  ierr = ISGetSize(local_dofs,&basis_dofs);CHKERRQ(ierr);
  ierr = KSPGetOperators(local_ksp,&local_mat,&local_pmat);CHKERRQ(ierr);

  /* Get null space vecs */
  ierr = MatNullSpaceGetVecs(NullSpace,&nnsp_has_cnst,&nnsp_size,&nullvecs);CHKERRQ(ierr);
  basis_size = nnsp_size;
  if (nnsp_has_cnst) basis_size++;

   /* Create shell ctx */
  ierr = PetscNew(&shell_ctx);CHKERRQ(ierr);
  shell_ctx->apply_scaling = needscaling;

  /* Create work vectors in shell context */
  ierr = VecCreate(PETSC_COMM_SELF,&shell_ctx->work_small_1);CHKERRQ(ierr);
  ierr = VecSetSizes(shell_ctx->work_small_1,basis_size,basis_size);CHKERRQ(ierr);
  ierr = VecSetType(shell_ctx->work_small_1,VECSEQ);CHKERRQ(ierr);
  ierr = VecDuplicate(shell_ctx->work_small_1,&shell_ctx->work_small_2);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&shell_ctx->work_full_1);CHKERRQ(ierr);
  ierr = VecSetSizes(shell_ctx->work_full_1,basis_dofs,basis_dofs);CHKERRQ(ierr);
  ierr = VecSetType(shell_ctx->work_full_1,VECSEQ);CHKERRQ(ierr);
  ierr = VecDuplicate(shell_ctx->work_full_1,&shell_ctx->work_full_2);CHKERRQ(ierr);

  /* Allocate workspace */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,basis_dofs,basis_size,NULL,&shell_ctx->basis_mat);CHKERRQ(ierr);
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,basis_dofs,basis_size,NULL,&shell_ctx->Kbasis_mat);CHKERRQ(ierr);
  ierr = MatDenseGetArray(shell_ctx->basis_mat,&basis_mat);CHKERRQ(ierr);
  ierr = MatDenseGetArray(shell_ctx->Kbasis_mat,&Kbasis_mat);CHKERRQ(ierr);

  /* Restrict local null space on selected dofs
     and compute matrices N and K*N */
  ierr = VecScatterCreate(pcis->vec1_N,local_dofs,shell_ctx->work_full_1,(IS)0,&scatter_ctx);CHKERRQ(ierr);
  for (k=0;k<nnsp_size;k++) {
    ierr = VecPlaceArray(shell_ctx->work_full_1,(const PetscScalar*)&basis_mat[k*basis_dofs]);CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter_ctx,nullvecs[k],shell_ctx->work_full_1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter_ctx,nullvecs[k],shell_ctx->work_full_1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecPlaceArray(shell_ctx->work_full_2,(const PetscScalar*)&Kbasis_mat[k*basis_dofs]);CHKERRQ(ierr);
    ierr = MatMult(local_mat,shell_ctx->work_full_1,shell_ctx->work_full_2);CHKERRQ(ierr);
    ierr = VecResetArray(shell_ctx->work_full_1);CHKERRQ(ierr);
    ierr = VecResetArray(shell_ctx->work_full_2);CHKERRQ(ierr);
  }
  if (nnsp_has_cnst) {
    ierr = VecPlaceArray(shell_ctx->work_full_1,(const PetscScalar*)&basis_mat[k*basis_dofs]);CHKERRQ(ierr);
    ierr = VecSet(shell_ctx->work_full_1,one);CHKERRQ(ierr);
    ierr = VecPlaceArray(shell_ctx->work_full_2,(const PetscScalar*)&Kbasis_mat[k*basis_dofs]);CHKERRQ(ierr);
    ierr = MatMult(local_mat,shell_ctx->work_full_1,shell_ctx->work_full_2);CHKERRQ(ierr);
    ierr = VecResetArray(shell_ctx->work_full_1);CHKERRQ(ierr);
    ierr = VecResetArray(shell_ctx->work_full_2);CHKERRQ(ierr);
  }
  ierr = VecScatterDestroy(&scatter_ctx);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(shell_ctx->basis_mat,&basis_mat);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(shell_ctx->Kbasis_mat,&Kbasis_mat);CHKERRQ(ierr);

  /* Assemble another Mat object in shell context */
  ierr = MatTransposeMatMult(shell_ctx->basis_mat,shell_ctx->Kbasis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&small_mat);CHKERRQ(ierr);
  ierr = MatFactorInfoInitialize(&matinfo);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,basis_size,0,1,&is_aux);CHKERRQ(ierr);
  ierr = MatLUFactor(small_mat,is_aux,is_aux,&matinfo);CHKERRQ(ierr);
  ierr = ISDestroy(&is_aux);CHKERRQ(ierr);
  ierr = PetscMalloc1(basis_size*basis_size,&array_mat);CHKERRQ(ierr);
  for (k=0;k<basis_size;k++) {
    ierr = VecSet(shell_ctx->work_small_1,zero);CHKERRQ(ierr);
    ierr = VecSetValue(shell_ctx->work_small_1,k,one,INSERT_VALUES);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(shell_ctx->work_small_1);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(shell_ctx->work_small_1);CHKERRQ(ierr);
    ierr = MatSolve(small_mat,shell_ctx->work_small_1,shell_ctx->work_small_2);CHKERRQ(ierr);
    ierr = VecGetArrayRead(shell_ctx->work_small_2,(const PetscScalar**)&array);CHKERRQ(ierr);
    for (i=0;i<basis_size;i++) {
      array_mat[i*basis_size+k]=array[i];
    }
    ierr = VecRestoreArrayRead(shell_ctx->work_small_2,(const PetscScalar**)&array);CHKERRQ(ierr);
  }
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,basis_size,basis_size,array_mat,&inv_small_mat);CHKERRQ(ierr);
  ierr = MatMatMult(shell_ctx->basis_mat,inv_small_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&shell_ctx->Lbasis_mat);CHKERRQ(ierr);
  ierr = PetscFree(array_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&inv_small_mat);CHKERRQ(ierr);
  ierr = MatDestroy(&small_mat);CHKERRQ(ierr);
  ierr = MatScale(shell_ctx->Kbasis_mat,m_one);CHKERRQ(ierr);

  /* Rebuild local PC */
  ierr = KSPGetPC(local_ksp,&shell_ctx->local_pc);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)shell_ctx->local_pc);CHKERRQ(ierr);
  ierr = PCCreate(PETSC_COMM_SELF,&newpc);CHKERRQ(ierr);
  ierr = PCSetOperators(newpc,local_mat,local_mat);CHKERRQ(ierr);
  ierr = PCSetType(newpc,PCSHELL);CHKERRQ(ierr);
  ierr = PCShellSetContext(newpc,shell_ctx);CHKERRQ(ierr);
  ierr = PCShellSetApply(newpc,PCBDDCApplyNullSpaceCorrectionPC);CHKERRQ(ierr);
  ierr = PCShellSetDestroy(newpc,PCBDDCDestroyNullSpaceCorrectionPC);CHKERRQ(ierr);
  ierr = PCSetUp(newpc);CHKERRQ(ierr);
  ierr = KSPSetPC(local_ksp,newpc);CHKERRQ(ierr);
  ierr = KSPSetUp(local_ksp);CHKERRQ(ierr);

  /* Create ksp object suitable for extreme eigenvalues' estimation */
  if (needscaling) {
    KSP check_ksp;
    Vec *workv;

    ierr = KSPCreate(PETSC_COMM_SELF,&check_ksp);CHKERRQ(ierr);
    ierr = KSPSetErrorIfNotConverged(check_ksp,pc->erroriffailure);CHKERRQ(ierr);
    ierr = KSPSetOperators(check_ksp,local_mat,local_mat);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,1.e-4,1.e-12,PETSC_DEFAULT,basis_dofs);CHKERRQ(ierr);
    ierr = KSPSetComputeSingularValues(check_ksp,PETSC_TRUE);CHKERRQ(ierr);
    ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);
    if (issym) {
      ierr = KSPSetType(check_ksp,KSPCG);CHKERRQ(ierr);
    }
    ierr = VecDuplicateVecs(shell_ctx->work_full_1,2,&workv);CHKERRQ(ierr);
    ierr = KSPSetPC(check_ksp,newpc);CHKERRQ(ierr);
    ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
    ierr = VecSetRandom(workv[1],NULL);CHKERRQ(ierr);
    ierr = MatMult(local_mat,workv[1],workv[0]);CHKERRQ(ierr);
    ierr = KSPSolve(check_ksp,workv[0],workv[0]);CHKERRQ(ierr);
    ierr = VecAXPY(workv[0],-1.,workv[1]);CHKERRQ(ierr);
    ierr = VecNorm(workv[0],NORM_INFINITY,&test_err);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
    if (pcbddc->dbg_flag) {
      if (isdir) {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted solver (no scale) %1.14e (it %d, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted solver (no scale) %1.14e (it %d, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
      }
    }
    shell_ctx->scale = 1./lambda_max;
    ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
    ierr = VecDestroyVecs(2,&workv);CHKERRQ(ierr);
  }
  ierr = PCDestroy(&newpc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCNullSpaceCheckCorrection(PC pc, PetscBool isdir)
{
  PC_BDDC                  *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS                   *matis = (Mat_IS*)pc->pmat->data;
  KSP                      check_ksp,local_ksp;
  MatNullSpace             NullSpace = NULL;
  NullSpaceCorrection_ctx  shell_ctx;
  PC                       check_pc;
  Mat                      test_mat,local_mat;
  PetscReal                test_err,lambda_min,lambda_max;
  PetscBool                setsym,issym=PETSC_FALSE;
  Vec                      work1,work2,work3;
  PetscInt                 k;
  PetscErrorCode           ierr;

  PetscFunctionBegin;
  ierr = MatGetNullSpace(matis->A,&NullSpace);CHKERRQ(ierr);
  if (!NullSpace) PetscFunctionReturn(0);
  if (!pcbddc->dbg_flag) PetscFunctionReturn(0);
  if (isdir) local_ksp = pcbddc->ksp_D;
  else local_ksp = pcbddc->ksp_R;
  ierr = KSPGetOperators(local_ksp,&local_mat,NULL);CHKERRQ(ierr);
  ierr = KSPGetPC(local_ksp,&check_pc);CHKERRQ(ierr);
  ierr = PCShellGetContext(check_pc,(void**)&shell_ctx);CHKERRQ(ierr);
  ierr = VecDuplicate(shell_ctx->work_full_1,&work1);CHKERRQ(ierr);
  ierr = VecDuplicate(shell_ctx->work_full_1,&work2);CHKERRQ(ierr);
  ierr = VecDuplicate(shell_ctx->work_full_1,&work3);CHKERRQ(ierr);

  ierr = VecSetRandom(shell_ctx->work_small_1,NULL);CHKERRQ(ierr);
  ierr = MatMult(shell_ctx->basis_mat,shell_ctx->work_small_1,work1);CHKERRQ(ierr);
  ierr = VecCopy(work1,work2);CHKERRQ(ierr);
  ierr = MatMult(local_mat,work1,work3);CHKERRQ(ierr);
  ierr = PCApply(check_pc,work3,work1);CHKERRQ(ierr);
  ierr = VecAXPY(work1,-1.,work2);CHKERRQ(ierr);
  ierr = VecNorm(work1,NORM_INFINITY,&test_err);CHKERRQ(ierr);
  if (isdir) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann nullspace correction solver: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
  }

  ierr = MatTransposeMatMult(shell_ctx->Lbasis_mat,shell_ctx->Kbasis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&test_mat);CHKERRQ(ierr);
  ierr = MatShift(test_mat,1.);CHKERRQ(ierr);
  ierr = MatNorm(test_mat,NORM_INFINITY,&test_err);CHKERRQ(ierr);
  ierr = MatDestroy(&test_mat);CHKERRQ(ierr);
  if (isdir) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet nullspace matrices: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann nullspace matrices: %1.14e\n",PetscGlobalRank,test_err);CHKERRQ(ierr);
  }

  /* Create ksp object suitable for extreme eigenvalues' estimation */
  ierr = KSPCreate(PETSC_COMM_SELF,&check_ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(check_ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr = KSPSetOperators(check_ksp,local_mat,local_mat);CHKERRQ(ierr);
  ierr = KSPSetTolerances(check_ksp,1.e-10,1.e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(check_ksp,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatIsSymmetricKnown(pc->pmat,&setsym,&issym);CHKERRQ(ierr);
  if (issym) {
    ierr = KSPSetType(check_ksp,KSPCG);CHKERRQ(ierr);
  }
  ierr = KSPSetPC(check_ksp,check_pc);CHKERRQ(ierr);
  ierr = KSPSetUp(check_ksp);CHKERRQ(ierr);
  ierr = VecSetRandom(work1,NULL);CHKERRQ(ierr);
  ierr = MatMult(local_mat,work1,work2);CHKERRQ(ierr);
  ierr = KSPSolve(check_ksp,work2,work2);CHKERRQ(ierr);
  ierr = VecAXPY(work2,-1.,work1);CHKERRQ(ierr);
  ierr = VecNorm(work2,NORM_INFINITY,&test_err);CHKERRQ(ierr);
  ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
  ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
  if (isdir) {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Dirichlet adapted KSP (scale %d) %1.14e (it %d, eigs %1.6e %1.6e)\n",PetscGlobalRank,shell_ctx->apply_scaling,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d infinity error for Neumann adapted KSP (scale %d) %1.14e (it %d, eigs %1.6e %1.6e)\n",PetscGlobalRank,shell_ctx->apply_scaling,test_err,k,lambda_min,lambda_max);CHKERRQ(ierr);
  }
  ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&work1);CHKERRQ(ierr);
  ierr = VecDestroy(&work2);CHKERRQ(ierr);
  ierr = VecDestroy(&work3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
