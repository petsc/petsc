#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

static PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursCreate"
PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs *sub_schurs)
{
  PCBDDCSubSchurs schurs_ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&schurs_ctx);CHKERRQ(ierr);
  schurs_ctx->n_subs = 0;
  *sub_schurs = schurs_ctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursDestroy"
PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs *sub_schurs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCSubSchursReset(*sub_schurs);CHKERRQ(ierr);
  ierr = PetscFree(*sub_schurs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursReset"
PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs sub_schurs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDestroy(&sub_schurs->is_AEj_I[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&sub_schurs->work1[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&sub_schurs->work2[i]);CHKERRQ(ierr);
  }
  if (sub_schurs->n_subs) {
    ierr = PetscFree5(sub_schurs->is_AEj_I,sub_schurs->is_AEj_B,sub_schurs->S_Ej,sub_schurs->work1,sub_schurs->work2);CHKERRQ(ierr);
  }
  sub_schurs->n_subs = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCAdjGetNextLayer_Private"
PETSC_STATIC_INLINE PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt* queue_tip,PetscInt n_prev,PetscBT touched,PetscInt* xadj,PetscInt* adjncy,PetscInt* n_added)
{
  PetscInt       i,j,n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  n = 0;
  for (i=-n_prev;i<0;i++) {
    PetscInt start_dof = queue_tip[i];
    for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
      PetscInt dof = adjncy[j];
      if (!PetscBTLookup(touched,dof)) {
        ierr = PetscBTSet(touched,dof);CHKERRQ(ierr);
        queue_tip[n] = dof;
        n++;
      }
    }
  }
  *n_added = n;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursSetUp"
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs sub_schurs, Mat S, IS is_A_I, IS is_A_B, PetscInt ncc, IS is_cc[], PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers)
{
  Mat                    A_II,A_IB,A_BI,A_BB;
  Mat                    AE_II,*AE_IE,*AE_EI,*AE_EE;
  IS                     is_I,*is_subset_B;
  ISLocalToGlobalMapping BtoNmap;
  PetscInt               i;
  PetscBool              is_sorted;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = ISSorted(is_A_I,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) {
    SETERRQ(PetscObjectComm((PetscObject)is_A_I),PETSC_ERR_PLIB,"IS for I dofs should be shorted");
  }
  ierr = ISSorted(is_A_B,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) {
    SETERRQ(PetscObjectComm((PetscObject)is_A_B),PETSC_ERR_PLIB,"IS for B dofs should be shorted");
  }

  /* get Schur complement matrices */
  ierr = MatSchurComplementGetSubMatrices(S,&A_II,NULL,&A_IB,&A_BI,&A_BB);CHKERRQ(ierr);

  /* allocate space for schur complements */
  ierr = PetscMalloc5(ncc,&sub_schurs->is_AEj_I,ncc,&sub_schurs->is_AEj_B,ncc,&sub_schurs->S_Ej,ncc,&sub_schurs->work1,ncc,&sub_schurs->work2);CHKERRQ(ierr);
  ierr = PetscMalloc4(ncc,&is_subset_B,ncc,&AE_IE,ncc,&AE_EI,ncc,&AE_EE);CHKERRQ(ierr);
  sub_schurs->n_subs = ncc;

  /* maps */
  if (sub_schurs->n_subs && nlayers >= 0 && xadj != NULL && adjncy != NULL) { /* Interior problems can be different from the original one */
    ISLocalToGlobalMapping ItoNmap;
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    /* get sizes */
    ierr = ISGetLocalSize(is_A_I,&n_I);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is_A_B,&n_B);CHKERRQ(ierr);

    ierr = ISLocalToGlobalMappingCreateIS(is_A_I,&ItoNmap);CHKERRQ(ierr);
    ierr = PetscMalloc1(n_I+n_B,&local_numbering);CHKERRQ(ierr);
    ierr = PetscBTCreate(n_I+n_B,&touched);CHKERRQ(ierr);
    ierr = PetscBTMemzero(n_I+n_B,touched);CHKERRQ(ierr);

    /* all boundary dofs must be skipped when adding layers */
    ierr = ISGetIndices(is_A_B,&idx_B);CHKERRQ(ierr);
    for (j=0;j<n_B;j++) {
      ierr = PetscBTSet(touched,idx_B[j]);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(local_numbering,idx_B,n_B*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(is_A_B,&idx_B);CHKERRQ(ierr);

    /* add next layers of dofs */
    n_local_dofs = n_B;
    n_prev_added = n_B;
    for (layer=0;layer<nlayers;layer++) {
      PetscInt n_added;
      if (n_local_dofs == n_I+n_B) break;
      if (n_local_dofs > n_I+n_B) {
        SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error querying layer %d. Out of bound access (%d > %d)",layer,n_local_dofs,n_I+n_B);
      }
      ierr = PCBDDCAdjGetNextLayer_Private(local_numbering+n_local_dofs,n_prev_added,touched,xadj,adjncy,&n_added);CHKERRQ(ierr);
      n_prev_added = n_added;
      n_local_dofs += n_added;
      if (!n_added) break;
    }
    ierr = PetscBTDestroy(&touched);CHKERRQ(ierr);

    /* IS for I dofs in original numbering and in I numbering */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ItoNmap),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&sub_schurs->is_AEj_I[0]);CHKERRQ(ierr);
    ierr = PetscFree(local_numbering);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_AEj_I[0]);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,sub_schurs->is_AEj_I[0],&is_I);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ItoNmap);CHKERRQ(ierr);

    /* II block */
    ierr = MatGetSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II);CHKERRQ(ierr);
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    ierr = PetscObjectReference((PetscObject)is_A_I);CHKERRQ(ierr);
    sub_schurs->is_AEj_I[0] = is_A_I;

    /* IS for I dofs in I numbering (strided 1) */
    ierr = ISGetSize(is_A_I,&n_I);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)is_A_I),n_I,0,1,&is_I);CHKERRQ(ierr);

    /* II block is the same */
    ierr = PetscObjectReference((PetscObject)A_II);CHKERRQ(ierr);
    AE_II = A_II;
  }

  /* TODO: just for compatibility with the previous version, needs to be fixed */
  for (i=1;i<sub_schurs->n_subs;i++) {
    ierr = PetscObjectReference((PetscObject)sub_schurs->is_AEj_I[0]);CHKERRQ(ierr);
    sub_schurs->is_AEj_I[i] = sub_schurs->is_AEj_I[0];
  }

  /* subsets in original and boundary numbering */
  ierr = ISLocalToGlobalMappingCreateIS(is_A_B,&BtoNmap);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDuplicate(is_cc[i],&sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyIS(BtoNmap,IS_GTOLM_DROP,sub_schurs->is_AEj_B[i],&is_subset_B[i]);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&BtoNmap);CHKERRQ(ierr);

  /* EE block */
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = MatGetSubMatrix(A_BB,is_subset_B[i],is_subset_B[i],MAT_INITIAL_MATRIX,&AE_EE[i]);CHKERRQ(ierr);
  }
  /* IE block */
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = MatGetSubMatrix(A_IB,is_I,is_subset_B[i],MAT_INITIAL_MATRIX,&AE_IE[i]);CHKERRQ(ierr);
  }
  /* EI block */
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = MatGetSubMatrix(A_BI,is_subset_B[i],is_I,MAT_INITIAL_MATRIX,&AE_EI[i]);CHKERRQ(ierr);
  }

  /* setup Schur complements on subset */
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = MatCreateSchurComplement(AE_II,AE_II,AE_IE[i],AE_EI[i],AE_EE[i],&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
    ierr = MatCreateVecs(sub_schurs->S_Ej[i],&sub_schurs->work1[i],&sub_schurs->work2[i]);CHKERRQ(ierr);
    if (AE_II == A_II) { /* we can reuse the same ksp */
      KSP ksp;
      ierr = MatSchurComplementGetKSP(S,&ksp);CHKERRQ(ierr);
      ierr = MatSchurComplementSetKSP(sub_schurs->S_Ej[i],ksp);CHKERRQ(ierr);
    } else { /* build new ksp object which inherits ksp and pc types from the original one */
      KSP      origksp,schurksp;
      PC       origpc,schurpc;
      KSPType  ksp_type;
      PCType   pc_type;
      PetscInt n_internal;

      ierr = MatSchurComplementGetKSP(S,&origksp);CHKERRQ(ierr);
      ierr = MatSchurComplementGetKSP(sub_schurs->S_Ej[i],&schurksp);CHKERRQ(ierr);
      ierr = KSPGetType(origksp,&ksp_type);CHKERRQ(ierr);
      ierr = KSPSetType(schurksp,ksp_type);CHKERRQ(ierr);
      ierr = KSPGetPC(schurksp,&schurpc);CHKERRQ(ierr);
      ierr = KSPGetPC(origksp,&origpc);CHKERRQ(ierr);
      ierr = PCGetType(origpc,&pc_type);CHKERRQ(ierr);
      ierr = PCSetType(schurpc,pc_type);CHKERRQ(ierr);
      ierr = ISGetSize(is_I,&n_internal);CHKERRQ(ierr);
      if (n_internal) { /* UMFPACK gives error with 0 sized problems */
        MatSolverPackage solver=NULL;
        ierr = PCFactorGetMatSolverPackage(origpc,(const MatSolverPackage*)&solver);CHKERRQ(ierr);
        if (solver) {
          ierr = PCFactorSetMatSolverPackage(schurpc,solver);CHKERRQ(ierr);
        }
      }
      ierr = KSPSetUp(schurksp);CHKERRQ(ierr);
    }
  }
  /* free */
  ierr = ISDestroy(&is_I);CHKERRQ(ierr);
  ierr = MatDestroy(&AE_II);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = MatDestroy(&AE_EE[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&AE_IE[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&AE_EI[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&is_subset_B[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree4(is_subset_B,AE_IE,AE_EI,AE_EE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
