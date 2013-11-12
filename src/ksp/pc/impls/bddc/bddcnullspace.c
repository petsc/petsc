#include "bddc.h"
#include "bddcprivate.h"

#undef __FUNCT__
#define __FUNCT__ "PCBDDCNullSpaceAssembleCoarse"
PetscErrorCode PCBDDCNullSpaceAssembleCoarse(PC pc, Mat coarse_mat, MatNullSpace* CoarseNullSpace)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  Mat_IS         *matis = (Mat_IS*)pc->pmat->data;
  MatNullSpace   tempCoarseNullSpace;
  const Vec      *nsp_vecs;
  Vec            *coarse_nsp_vecs,local_vec,local_primal_vec;
  PetscInt       nsp_size,coarse_nsp_size,i;
  PetscBool      nsp_has_cnst;
  PetscReal      test_null;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  tempCoarseNullSpace = 0;
  coarse_nsp_size = 0;
  coarse_nsp_vecs = 0;
  ierr = MatNullSpaceGetVecs(pcbddc->NullSpace,&nsp_has_cnst,&nsp_size,&nsp_vecs);CHKERRQ(ierr);
  if (coarse_mat) {
    ierr = PetscMalloc((nsp_size+1)*sizeof(Vec),&coarse_nsp_vecs);CHKERRQ(ierr);
    for (i=0;i<nsp_size+1;i++) {
      ierr = VecDuplicate(pcbddc->coarse_vec,&coarse_nsp_vecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = MatGetVecs(pcbddc->ConstraintMatrix,&local_vec,&local_primal_vec);CHKERRQ(ierr);
  if (nsp_has_cnst) {
    ierr = VecSet(local_vec,1.0);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->ConstraintMatrix,local_vec,local_primal_vec);CHKERRQ(ierr);
    ierr = PCBDDCScatterCoarseDataBegin(pc,local_primal_vec,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PCBDDCScatterCoarseDataEnd(pc,local_primal_vec,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (coarse_mat) {
      if (pcbddc->dbg_flag) {
        ierr = MatMult(coarse_mat,pcbddc->coarse_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->coarse_rhs,NORM_INFINITY,&test_null);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Constant coarse null space error % 1.14e\n",test_null);CHKERRQ(ierr);
      }
      ierr = VecCopy(pcbddc->coarse_vec,coarse_nsp_vecs[coarse_nsp_size]);CHKERRQ(ierr);
      coarse_nsp_size++;
    }
  }
  for (i=0;i<nsp_size;i++)  {
    ierr = VecScatterBegin(matis->ctx,nsp_vecs[i],local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,nsp_vecs[i],local_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatMult(pcbddc->ConstraintMatrix,local_vec,local_primal_vec);CHKERRQ(ierr);
    ierr = PCBDDCScatterCoarseDataBegin(pc,local_primal_vec,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = PCBDDCScatterCoarseDataEnd(pc,local_primal_vec,pcbddc->coarse_vec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (coarse_mat) {
      if (pcbddc->dbg_flag) {
        ierr = MatMult(coarse_mat,pcbddc->coarse_vec,pcbddc->coarse_rhs);CHKERRQ(ierr);
        ierr = VecNorm(pcbddc->coarse_rhs,NORM_2,&test_null);CHKERRQ(ierr);
        ierr = PetscViewerASCIIPrintf(pcbddc->dbg_viewer,"Vec %d coarse null space error % 1.14e\n",i,test_null);CHKERRQ(ierr);
      }
      ierr = VecCopy(pcbddc->coarse_vec,coarse_nsp_vecs[coarse_nsp_size]);CHKERRQ(ierr);
      coarse_nsp_size++;
    }
  }
  if (coarse_nsp_size > 0) {
    ierr = PCBDDCOrthonormalizeVecs(coarse_nsp_size,coarse_nsp_vecs);CHKERRQ(ierr);
    ierr = MatNullSpaceCreate(PetscObjectComm((PetscObject)coarse_mat),PETSC_FALSE,coarse_nsp_size,coarse_nsp_vecs,&tempCoarseNullSpace);CHKERRQ(ierr);
    for (i=0;i<nsp_size+1;i++) {
      ierr = VecDestroy(&coarse_nsp_vecs[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(coarse_nsp_vecs);CHKERRQ(ierr);
  ierr = VecDestroy(&local_vec);CHKERRQ(ierr);
  ierr = VecDestroy(&local_primal_vec);CHKERRQ(ierr);
  *CoarseNullSpace = tempCoarseNullSpace;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PCBDDCApplyNullSpaceCorrectionPC"
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCDestroyNullSpaceCorrectionPC"
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

/*
PETSC_EXTERN PetscErrorCode PCBDDCApplyNullSpaceCorrectionPC(PC,Vec,Vec);
PETSC_EXTERN PetscErrorCode PCBDDCDestroyNullSpaceCorrectionPC(PC);
*/

#undef __FUNCT__
#define __FUNCT__ "PCBDDCNullSpaceAssembleCorrection"
PetscErrorCode PCBDDCNullSpaceAssembleCorrection(PC pc,IS local_dofs)
{
  PC_BDDC        *pcbddc = (PC_BDDC*)pc->data;
  PC_IS          *pcis = (PC_IS*)pc->data;
  Mat_IS*        matis = (Mat_IS*)pc->pmat->data;
  KSP            *local_ksp;
  PC             newpc;
  NullSpaceCorrection_ctx  shell_ctx;
  Mat            local_mat,local_pmat,small_mat,inv_small_mat;
  MatStructure   local_mat_struct;
  Vec            work1,work2;
  const Vec      *nullvecs;
  VecScatter     scatter_ctx;
  IS             is_aux;
  MatFactorInfo  matinfo;
  PetscScalar    *basis_mat,*Kbasis_mat,*array,*array_mat;
  PetscScalar    one = 1.0,zero = 0.0, m_one = -1.0;
  PetscInt       basis_dofs,basis_size,nnsp_size,i,k,n_I,n_R;
  PetscBool      nnsp_has_cnst;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Infer the local solver */
  ierr = ISGetSize(local_dofs,&basis_dofs);CHKERRQ(ierr);
  ierr = VecGetSize(pcis->vec1_D,&n_I);CHKERRQ(ierr);
  ierr = VecGetSize(pcbddc->vec1_R,&n_R);CHKERRQ(ierr);
  if (basis_dofs == n_I) {
    /* Dirichlet solver */
    local_ksp = &pcbddc->ksp_D;
  } else if (basis_dofs == n_R) {
    /* Neumann solver */
    local_ksp = &pcbddc->ksp_R;
  } else {
    SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in %s: unknown local IS size %d. n_I=%d, n_R=%d)\n",__FUNCT__,basis_dofs,n_I,n_R);
  }
  ierr = KSPGetOperators(*local_ksp,&local_mat,&local_pmat,&local_mat_struct);CHKERRQ(ierr);

  /* Get null space vecs */
  ierr = MatNullSpaceGetVecs(pcbddc->NullSpace,&nnsp_has_cnst,&nnsp_size,&nullvecs);CHKERRQ(ierr);
  basis_size = nnsp_size;
  if (nnsp_has_cnst) {
    basis_size++;
  }

  if (basis_dofs) {
     /* Create shell ctx */
     ierr = PetscMalloc(sizeof(*shell_ctx),&shell_ctx);CHKERRQ(ierr);

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
     ierr = MatCreateSeqDense(PETSC_COMM_SELF,basis_dofs,basis_size,NULL,&shell_ctx->basis_mat );CHKERRQ(ierr);
     ierr = MatCreateSeqDense(PETSC_COMM_SELF,basis_dofs,basis_size,NULL,&shell_ctx->Kbasis_mat);CHKERRQ(ierr);
     ierr = MatDenseGetArray(shell_ctx->basis_mat,&basis_mat);CHKERRQ(ierr);
     ierr = MatDenseGetArray(shell_ctx->Kbasis_mat,&Kbasis_mat);CHKERRQ(ierr);

     /* Restrict local null space on selected dofs (Dirichlet or Neumann)
        and compute matrices N and K*N */
     ierr = VecDuplicate(shell_ctx->work_full_1,&work1);CHKERRQ(ierr);
     ierr = VecDuplicate(shell_ctx->work_full_1,&work2);CHKERRQ(ierr);
     ierr = VecScatterCreate(pcis->vec1_N,local_dofs,work1,(IS)0,&scatter_ctx);CHKERRQ(ierr);
  }

  for (k=0;k<nnsp_size;k++) {
    ierr = VecScatterBegin(matis->ctx,nullvecs[k],pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(matis->ctx,nullvecs[k],pcis->vec1_N,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    if (basis_dofs) {
      ierr = VecPlaceArray(work1,(const PetscScalar*)&basis_mat[k*basis_dofs]);CHKERRQ(ierr);
      ierr = VecScatterBegin(scatter_ctx,pcis->vec1_N,work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(scatter_ctx,pcis->vec1_N,work1,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecPlaceArray(work2,(const PetscScalar*)&Kbasis_mat[k*basis_dofs]);CHKERRQ(ierr);
      ierr = MatMult(local_mat,work1,work2);CHKERRQ(ierr);
      ierr = VecResetArray(work1);CHKERRQ(ierr);
      ierr = VecResetArray(work2);CHKERRQ(ierr);
    }
  }

  if (basis_dofs) {
    if (nnsp_has_cnst) {
      ierr = VecPlaceArray(work1,(const PetscScalar*)&basis_mat[k*basis_dofs]);CHKERRQ(ierr);
      ierr = VecSet(work1,one);CHKERRQ(ierr);
      ierr = VecPlaceArray(work2,(const PetscScalar*)&Kbasis_mat[k*basis_dofs]);CHKERRQ(ierr);
      ierr = MatMult(local_mat,work1,work2);CHKERRQ(ierr);
      ierr = VecResetArray(work1);CHKERRQ(ierr);
      ierr = VecResetArray(work2);CHKERRQ(ierr);
    }
    ierr = VecDestroy(&work1);CHKERRQ(ierr);
    ierr = VecDestroy(&work2);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&scatter_ctx);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(shell_ctx->basis_mat,&basis_mat);CHKERRQ(ierr);
    ierr = MatDenseRestoreArray(shell_ctx->Kbasis_mat,&Kbasis_mat);CHKERRQ(ierr);

    /* Assemble another Mat object in shell context */
    ierr = MatTransposeMatMult(shell_ctx->basis_mat,shell_ctx->Kbasis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&small_mat);CHKERRQ(ierr);
    ierr = MatFactorInfoInitialize(&matinfo);CHKERRQ(ierr);
    ierr = ISCreateStride(PETSC_COMM_SELF,basis_size,0,1,&is_aux);CHKERRQ(ierr);
    ierr = MatLUFactor(small_mat,is_aux,is_aux,&matinfo);CHKERRQ(ierr);
    ierr = ISDestroy(&is_aux);CHKERRQ(ierr);
    ierr = PetscMalloc(basis_size*basis_size*sizeof(PetscScalar),&array_mat);CHKERRQ(ierr);
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
    ierr = KSPGetPC(*local_ksp,&shell_ctx->local_pc);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)shell_ctx->local_pc);CHKERRQ(ierr);
    ierr = PCCreate(PETSC_COMM_SELF,&newpc);CHKERRQ(ierr);
    ierr = PCSetOperators(newpc,local_mat,local_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = PCSetType(newpc,PCSHELL);CHKERRQ(ierr);
    ierr = PCShellSetContext(newpc,shell_ctx);CHKERRQ(ierr);
    ierr = PCShellSetApply(newpc,PCBDDCApplyNullSpaceCorrectionPC);CHKERRQ(ierr);
    ierr = PCShellSetDestroy(newpc,PCBDDCDestroyNullSpaceCorrectionPC);CHKERRQ(ierr);
    ierr = PCSetUp(newpc);CHKERRQ(ierr);
    ierr = KSPSetPC(*local_ksp,newpc);CHKERRQ(ierr);
    ierr = PCDestroy(&newpc);CHKERRQ(ierr);
    ierr = KSPSetUp(*local_ksp);CHKERRQ(ierr);
  }
  /* test */
  if (pcbddc->dbg_flag && basis_dofs) {
    KSP         check_ksp;
    PC          check_pc;
    Mat         test_mat;
    Vec         work3;
    PetscReal   test_err,lambda_min,lambda_max;
    PetscBool   setsym,issym=PETSC_FALSE;
    PetscInt    tabs;

    ierr = PetscViewerASCIIGetTab(pcbddc->dbg_viewer,&tabs);CHKERRQ(ierr);
    ierr = KSPGetPC(*local_ksp,&check_pc);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->work_full_1,&work1);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->work_full_1,&work2);CHKERRQ(ierr);
    ierr = VecDuplicate(shell_ctx->work_full_1,&work3);CHKERRQ(ierr);
    ierr = VecSetRandom(shell_ctx->work_small_1,NULL);CHKERRQ(ierr);
    ierr = MatMult(shell_ctx->basis_mat,shell_ctx->work_small_1,work1);CHKERRQ(ierr);
    ierr = VecCopy(work1,work2);CHKERRQ(ierr);
    ierr = MatMult(local_mat,work1,work3);CHKERRQ(ierr);
    ierr = PCApply(check_pc,work3,work1);CHKERRQ(ierr);
    ierr = VecAXPY(work1,m_one,work2);CHKERRQ(ierr);
    ierr = VecNorm(work1,NORM_INFINITY,&test_err);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d error for nullspace correction for ",PetscGlobalRank);
    ierr = PetscViewerASCIIUseTabs(pcbddc->dbg_viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (basis_dofs == n_I) {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Dirichlet ");
    } else {
      ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Neumann ");
    }
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"solver is :%1.14e\n",test_err);
    ierr = PetscViewerASCIISetTab(pcbddc->dbg_viewer,tabs);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(pcbddc->dbg_viewer,PETSC_TRUE);CHKERRQ(ierr);

    ierr = MatTransposeMatMult(shell_ctx->Lbasis_mat,shell_ctx->Kbasis_mat,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&test_mat);CHKERRQ(ierr);
    ierr = MatShift(test_mat,one);CHKERRQ(ierr);
    ierr = MatNorm(test_mat,NORM_INFINITY,&test_err);CHKERRQ(ierr);
    ierr = MatDestroy(&test_mat);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d error for nullspace matrices is :%1.14e\n",PetscGlobalRank,test_err);

    /* Create ksp object suitable for extreme eigenvalues' estimation */
    ierr = KSPCreate(PETSC_COMM_SELF,&check_ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(check_ksp,local_mat,local_mat,SAME_PRECONDITIONER);CHKERRQ(ierr);
    ierr = KSPSetTolerances(check_ksp,1.e-8,1.e-8,PETSC_DEFAULT,basis_dofs);CHKERRQ(ierr);
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
    ierr = VecAXPY(work2,m_one,work1);CHKERRQ(ierr);
    ierr = VecNorm(work2,NORM_INFINITY,&test_err);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(check_ksp,&lambda_max,&lambda_min);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(check_ksp,&k);CHKERRQ(ierr);
    ierr = PetscViewerASCIISynchronizedPrintf(pcbddc->dbg_viewer,"Subdomain %04d error for adapted KSP %1.14e (it %d, eigs %1.6e %1.6e)\n",PetscGlobalRank,test_err,k,lambda_min,lambda_max);
    ierr = KSPDestroy(&check_ksp);CHKERRQ(ierr);
    ierr = VecDestroy(&work1);CHKERRQ(ierr);
    ierr = VecDestroy(&work2);CHKERRQ(ierr);
    ierr = VecDestroy(&work3);CHKERRQ(ierr);
  }
  /* all processes shoud call this, even the void ones */
  if (pcbddc->dbg_flag) {
    ierr = PetscViewerFlush(pcbddc->dbg_viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCNullSpaceAdaptGlobal"
PetscErrorCode PCBDDCNullSpaceAdaptGlobal(PC pc)
{
  PC_IS*         pcis = (PC_IS*)  (pc->data);
  PC_BDDC*       pcbddc = (PC_BDDC*)(pc->data);
  KSP            inv_change;
  PC             pc_change;
  const Vec      *nsp_vecs;
  Vec            *new_nsp_vecs;
  PetscInt       i,nsp_size,new_nsp_size,start_new;
  PetscBool      nsp_has_cnst;
  MatNullSpace   new_nsp;
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  /* create KSP for change of basis */
  ierr = KSPCreate(PETSC_COMM_SELF,&inv_change);CHKERRQ(ierr);
  ierr = KSPSetOperators(inv_change,pcbddc->ChangeOfBasisMatrix,pcbddc->ChangeOfBasisMatrix,SAME_PRECONDITIONER);CHKERRQ(ierr);
  ierr = KSPSetType(inv_change,KSPPREONLY);CHKERRQ(ierr);
  ierr = KSPGetPC(inv_change,&pc_change);CHKERRQ(ierr);
  ierr = PCSetType(pc_change,PCLU);CHKERRQ(ierr);
  ierr = KSPSetUp(inv_change);CHKERRQ(ierr);
  /* get nullspace and transform it */
  ierr = MatNullSpaceGetVecs(pcbddc->NullSpace,&nsp_has_cnst,&nsp_size,&nsp_vecs);CHKERRQ(ierr);
  new_nsp_size = nsp_size;
  if (nsp_has_cnst) {
    new_nsp_size++;
  }
  ierr = PetscMalloc(new_nsp_size*sizeof(Vec),&new_nsp_vecs);CHKERRQ(ierr);
  for (i=0;i<new_nsp_size;i++) {
    ierr = VecDuplicate(pcis->vec1_global,&new_nsp_vecs[i]);CHKERRQ(ierr);
  }
  start_new = 0;
  if (nsp_has_cnst) {
    start_new = 1;
    ierr = VecSet(new_nsp_vecs[0],1.0);CHKERRQ(ierr);
    ierr = VecSet(pcis->vec1_B,1.0);CHKERRQ(ierr);
    ierr = KSPSolve(inv_change,pcis->vec1_B,pcis->vec1_B);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[0],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[0],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  for (i=0;i<nsp_size;i++) {
    ierr = VecCopy(nsp_vecs[i],new_nsp_vecs[i+start_new]);CHKERRQ(ierr);
    ierr = VecScatterBegin(pcis->global_to_B,nsp_vecs[i],pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,nsp_vecs[i],pcis->vec1_B,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = KSPSolve(inv_change,pcis->vec1_B,pcis->vec1_B);
    ierr = VecScatterBegin(pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[i+start_new],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd  (pcis->global_to_B,pcis->vec1_B,new_nsp_vecs[i+start_new],INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  }
  ierr = PCBDDCOrthonormalizeVecs(new_nsp_size,new_nsp_vecs);CHKERRQ(ierr);
#if 0
  PetscBool nsp_t=PETSC_FALSE;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("Original Null Space test: %d\n",nsp_t);
  Mat temp_mat;
  Mat_IS* matis = (Mat_IS*)pc->pmat->data;
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("Original Null Space, mat changed test: %d\n",nsp_t);
  {
    PetscReal test_norm;
    for (i=0;i<new_nsp_size;i++) {
      ierr = MatMult(pc->pmat,new_nsp_vecs[i],pcis->vec1_global);CHKERRQ(ierr);
      ierr = VecNorm(pcis->vec1_global,NORM_2,&test_norm);CHKERRQ(ierr);
      if (test_norm > 1.e-12) {
        printf("------------ERROR VEC %d------------------\n",i);
        ierr = VecView(pcis->vec1_global,PETSC_VIEWER_STDOUT_WORLD);
        printf("------------------------------------------\n");
      }
    }
  }
#endif

  ierr = KSPDestroy(&inv_change);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)pc,&comm);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm,PETSC_FALSE,new_nsp_size,new_nsp_vecs,&new_nsp);CHKERRQ(ierr);
  ierr = PCBDDCSetNullSpace(pc,new_nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&new_nsp);CHKERRQ(ierr);
#if 0
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("New Null Space, mat changed: %d\n",nsp_t);
    temp_mat = matis->A;
    matis->A = pcbddc->local_mat;
    pcbddc->local_mat = temp_mat;
  ierr = MatNullSpaceTest(pcbddc->NullSpace,pc->pmat,&nsp_t);CHKERRQ(ierr);
  printf("New Null Space, mat original: %d\n",nsp_t);
#endif

  for (i=0;i<new_nsp_size;i++) {
    ierr = VecDestroy(&new_nsp_vecs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(new_nsp_vecs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
