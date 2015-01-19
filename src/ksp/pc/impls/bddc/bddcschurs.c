#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

static PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursSetUp"
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs sub_schurs, PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers)
{
  Mat                    A_II,A_IB,A_BI,A_BB;
  Mat                    AE_II,*AE_IE,*AE_EI,*AE_EE;
  Mat                    S_all,global_schur_subsets,work_mat;
  ISLocalToGlobalMapping l2gmap_subsets;
  IS                     is_I,*is_subset_B,temp_is;
  PetscInt               *nnz,*all_local_idx_G,*all_local_idx_B,*all_local_idx_N;
  PetscInt               i,subset_size,max_subset_size;
  PetscInt               extra,local_size,global_size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;

  /* allocate space for schur complements */
  ierr = PetscMalloc4(sub_schurs->n_subs,&sub_schurs->is_AEj_B,
                      sub_schurs->n_subs,&sub_schurs->S_Ej,
                      sub_schurs->n_subs,&sub_schurs->work1,
                      sub_schurs->n_subs,&sub_schurs->work2);CHKERRQ(ierr);

  /* get Schur complement matrices */
  if (!sub_schurs->use_mumps) {
    ierr = MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,&A_IB,&A_BI,&A_BB);CHKERRQ(ierr);
    ierr = PetscMalloc4(sub_schurs->n_subs,&is_subset_B,
                        sub_schurs->n_subs,&AE_IE,
                        sub_schurs->n_subs,&AE_EI,
                        sub_schurs->n_subs,&AE_EE);CHKERRQ(ierr);
  }

  /* determine interior problems */
  if (nlayers >= 0 && xadj != NULL && adjncy != NULL) { /* Interior problems can be different from the original one */
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    /* get sizes */
    ierr = ISGetLocalSize(sub_schurs->is_I,&n_I);CHKERRQ(ierr);
    ierr = ISGetLocalSize(sub_schurs->is_B,&n_B);CHKERRQ(ierr);

    ierr = PetscMalloc1(n_I+n_B,&local_numbering);CHKERRQ(ierr);
    ierr = PetscBTCreate(n_I+n_B,&touched);CHKERRQ(ierr);
    ierr = PetscBTMemzero(n_I+n_B,touched);CHKERRQ(ierr);

    /* all boundary dofs must be skipped when adding layers */
    ierr = ISGetIndices(sub_schurs->is_B,&idx_B);CHKERRQ(ierr);
    for (j=0;j<n_B;j++) {
      ierr = PetscBTSet(touched,idx_B[j]);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(local_numbering,idx_B,n_B*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_B,&idx_B);CHKERRQ(ierr);

    /* add prescribed number of layers of dofs */
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

    /* IS for I layer dofs in original numbering */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->is_I),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&sub_schurs->is_I_layer);CHKERRQ(ierr);
    ierr = PetscFree(local_numbering);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_I_layer);CHKERRQ(ierr);
    /* IS for I layer dofs in I numbering */
    if (!sub_schurs->use_mumps) {
      ISLocalToGlobalMapping ItoNmap;
      ierr = ISLocalToGlobalMappingCreateIS(sub_schurs->is_I,&ItoNmap);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,sub_schurs->is_I_layer,&is_I);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ItoNmap);CHKERRQ(ierr);

      /* II block */
      ierr = MatGetSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II);CHKERRQ(ierr);
    }
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    ierr = PetscObjectReference((PetscObject)sub_schurs->is_I);CHKERRQ(ierr);
    sub_schurs->is_I_layer = sub_schurs->is_I;

    /* IS for I dofs in I numbering (strided 1) */
    if (!sub_schurs->use_mumps) {
      ierr = ISGetSize(sub_schurs->is_I,&n_I);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)sub_schurs->is_I),n_I,0,1,&is_I);CHKERRQ(ierr);

      /* II block is the same */
      ierr = PetscObjectReference((PetscObject)A_II);CHKERRQ(ierr);
      AE_II = A_II;
    }
  }

  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDuplicate(sub_schurs->is_subs[i],&sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
  }

  /* Get info on subset sizes and sum of all subsets sizes */
  max_subset_size = 0;
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs_seq;i++) {
    PetscInt j = sub_schurs->index_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[j],&subset_size);CHKERRQ(ierr);
    max_subset_size = PetscMax(subset_size,max_subset_size);
    local_size += subset_size;
  }

  /* Work arrays for local indices */
  ierr = PetscMalloc1(local_size,&all_local_idx_B);CHKERRQ(ierr);
  extra = 0;
  if (sub_schurs->use_mumps) {
    ierr = ISGetLocalSize(sub_schurs->is_I_layer,&extra);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(local_size+extra,&all_local_idx_N);CHKERRQ(ierr);
  if (extra) {
    const PetscInt *idxs;
    ierr = ISGetIndices(sub_schurs->is_I_layer,&idxs);CHKERRQ(ierr);
    ierr = PetscMemcpy(all_local_idx_N,idxs,extra*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_I_layer,&idxs);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(local_size,&nnz);CHKERRQ(ierr);

  /* Get local indices in local numbering */
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs_seq;i++) {
    PetscInt j;
    const    PetscInt *idxs;

    PetscInt local_problem_index = sub_schurs->index_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_AEj_B[local_problem_index],&idxs);CHKERRQ(ierr);
    /* subset indices in local numbering */
    ierr = PetscMemcpy(all_local_idx_N+local_size+extra,idxs,subset_size*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_AEj_B[local_problem_index],&idxs);CHKERRQ(ierr);
    for (j=0;j<subset_size;j++) nnz[local_size+j] = subset_size;
    local_size += subset_size;
  }

  S_all = NULL;
  if (!sub_schurs->use_mumps) {
    /* subsets in original and boundary numbering */
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_AEj_B[i],&is_subset_B[i]);CHKERRQ(ierr);
    }

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
      if (AE_II == A_II) { /* we can reuse the same ksp */
        KSP ksp;
        ierr = MatSchurComplementGetKSP(sub_schurs->S,&ksp);CHKERRQ(ierr);
        ierr = MatSchurComplementSetKSP(sub_schurs->S_Ej[i],ksp);CHKERRQ(ierr);
      } else { /* build new ksp object which inherits ksp and pc types from the original one */
        KSP      origksp,schurksp;
        PC       origpc,schurpc;
        KSPType  ksp_type;
        PCType   pc_type;
        PetscInt n_internal;

        ierr = MatSchurComplementGetKSP(sub_schurs->S,&origksp);CHKERRQ(ierr);
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
#if defined(PETSC_HAVE_MUMPS)
  } else {
    Mat           A,F;
    IS            is_A_all;
    PetscBool     is_symmetric;
    PetscInt      *idxs_schur,n_I;

    /* get working mat */
    ierr = ISGetLocalSize(sub_schurs->is_I_layer,&n_I);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size+n_I,all_local_idx_N,PETSC_COPY_VALUES,&is_A_all);CHKERRQ(ierr);
    ierr = MatGetSubMatrixUnsorted(sub_schurs->A,is_A_all,is_A_all,&A);CHKERRQ(ierr);
    ierr = ISDestroy(&is_A_all);CHKERRQ(ierr);

    ierr = MatIsSymmetric(sub_schurs->A,0.0,&is_symmetric);CHKERRQ(ierr);
    if (is_symmetric) {
      ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
    } else {
      ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    }

    /* subsets ordered last */
    ierr = PetscMalloc1(local_size,&idxs_schur);CHKERRQ(ierr);
    for (i=0;i<local_size;i++) {
      idxs_schur[i] = n_I+i+1;
    }
    ierr = MatMumpsSetSchurIndices(F,local_size,idxs_schur);CHKERRQ(ierr);
    ierr = PetscFree(idxs_schur);CHKERRQ(ierr);

    /* factorization step */
    if (is_symmetric) {
      ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
      ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    } else {
      ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    }

    /* get explicit Schur Complement computed during numeric factorization */
    ierr = MatMumpsGetSchurComplement(F,&S_all);CHKERRQ(ierr);

    /* free workspace */
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);

    /* unused */
    for (i=0;i<sub_schurs->n_subs;i++) {
       sub_schurs->S_Ej[i] = 0;
    }
#endif
  }

  /* TODO: just for compatibility with the previous version, needs to be fixed */
  for (i=0;i<sub_schurs->n_subs_par;i++) {
    PetscInt j = sub_schurs->index_parallel[i];
    ierr = MatCreateVecs(sub_schurs->S_Ej[j],&sub_schurs->work1[j],&sub_schurs->work2[j]);CHKERRQ(ierr);
  }
  for (i=0;i<sub_schurs->n_subs_seq;i++) {
     sub_schurs->work1[sub_schurs->index_sequential[i]] = 0;
     sub_schurs->work2[sub_schurs->index_sequential[i]] = 0;
  }

  if (!sub_schurs->n_subs_seq_g) {
    sub_schurs->S_Ej_all = 0;
    sub_schurs->sum_S_Ej_all = 0;
    sub_schurs->is_Ej_all = 0;
    PetscFunctionReturn(0);
  }

  /* subset indices in local boundary numbering */
  ierr = ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B);CHKERRQ(ierr);
  if (subset_size != local_size) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %d != %d\n",subset_size,local_size);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all);CHKERRQ(ierr);

  /* Local matrix of all local Schur on subsets */
  ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatSetSizes(sub_schurs->S_Ej_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
  ierr = MatSetType(sub_schurs->S_Ej_all,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(sub_schurs->S_Ej_all,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  if (!sub_schurs->use_mumps) {
    PetscScalar *fill_vals;
    PetscInt    *dummy_idx;

    /* Work arrays */
    ierr = PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&fill_vals);CHKERRQ(ierr);

    /* Loop on local problems to compute Schur complements explicitly (TODO; optimize)*/
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs_seq;i++) {
      Vec work1,work2;
      PetscInt j,local_problem_index = sub_schurs->index_sequential[i];

      ierr = MatCreateVecs(sub_schurs->S_Ej[local_problem_index],&work1,&work2);CHKERRQ(ierr);
      ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
      /* local Schur */
      for (j=0;j<subset_size;j++) {
        ierr = VecSet(work1,0.0);CHKERRQ(ierr);
        ierr = VecSetValue(work1,j,1.0,INSERT_VALUES);CHKERRQ(ierr);
        ierr = VecPlaceArray(work2,&fill_vals[j*subset_size]);CHKERRQ(ierr);
        ierr = MatMult(sub_schurs->S_Ej[local_problem_index],work1,work2);CHKERRQ(ierr);
        ierr = VecResetArray(work2);CHKERRQ(ierr);
      }
      for (j=0;j<subset_size;j++) {
        dummy_idx[j]=local_size+j;
      }
      ierr = MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,fill_vals,INSERT_VALUES);CHKERRQ(ierr);
      local_size += subset_size;
      ierr = VecDestroy(&work1);CHKERRQ(ierr);
      ierr = VecDestroy(&work2);CHKERRQ(ierr);
    }
    ierr = PetscFree2(dummy_idx,fill_vals);CHKERRQ(ierr);
  } else {
    PetscInt    *dummy_idx,n_all;
    PetscScalar *vals,*fill_vals;

    /* Work arrays */
    ierr = PetscMalloc1(max_subset_size,&dummy_idx);CHKERRQ(ierr);
    ierr = MatGetSize(S_all,&n_all,NULL);CHKERRQ(ierr);
    ierr = MatDenseGetArray(S_all,&vals);CHKERRQ(ierr);
    local_size = 0;
    subset_size = 0;
    fill_vals = vals;
    for (i=0;i<sub_schurs->n_subs_seq;i++) {
      PetscInt j,lpi = sub_schurs->index_sequential[i];

      fill_vals += subset_size;
      ierr = ISGetLocalSize(sub_schurs->is_AEj_B[lpi],&subset_size);CHKERRQ(ierr);
      for (j=0;j<subset_size;j++) {
        dummy_idx[j]=local_size+j;
      }
      for (j=0;j<subset_size;j++) {
        ierr = MatSetValues(sub_schurs->S_Ej_all,1,dummy_idx+j,subset_size,dummy_idx,fill_vals,INSERT_VALUES);CHKERRQ(ierr);
        fill_vals += n_all;
      }
      local_size += subset_size;
    }
    ierr = MatDenseRestoreArray(S_all,&vals);CHKERRQ(ierr);
    ierr = PetscFree(dummy_idx);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatDestroy(&S_all);CHKERRQ(ierr);

  /* Global matrix of all assembled Schur on subsets */
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->l2gmap,local_size,all_local_idx_N+extra,PETSC_NULL,&global_size,&all_local_idx_G);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,local_size,all_local_idx_G,PETSC_COPY_VALUES,&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size,l2gmap_subsets,&work_mat);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(work_mat,sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(work_mat,MAT_INITIAL_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&work_mat);CHKERRQ(ierr);

  /* Get local part of (\sum_j S_Ej) */
  ierr = PetscFree(all_local_idx_N);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->l2gmap),local_size,all_local_idx_G,PETSC_OWN_POINTER,&temp_is);CHKERRQ(ierr);
  ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  ierr = MatDestroy(&global_schur_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursInit"
PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, Mat A, Mat S, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap, PetscInt seqthreshold)
{
  IS                  *faces,*edges,*all_cc;
  PetscInt            *index_sequential,*index_parallel;
  PetscInt            *auxlocal_sequential,*auxlocal_parallel;
  PetscInt            *auxglobal_sequential,*auxglobal_parallel;
  PetscInt            *auxmapping;//,*idxs;
  PetscInt            i,max_subset_size;
  PetscInt            n_sequential_problems,n_local_sequential_problems,n_parallel_problems,n_local_parallel_problems;
  PetscInt            n_faces,n_edges,n_all_cc;
  PetscBool           is_sorted;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = ISSorted(is_I,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) {
    SETERRQ(PetscObjectComm((PetscObject)is_I),PETSC_ERR_PLIB,"IS for I dofs should be shorted");
  }
  ierr = ISSorted(is_B,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) {
    SETERRQ(PetscObjectComm((PetscObject)is_B),PETSC_ERR_PLIB,"IS for B dofs should be shorted");
  }

  /* reset any previous data */
  ierr = PCBDDCSubSchursReset(sub_schurs);CHKERRQ(ierr);

  /* get index sets for faces and edges */
  ierr = PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,NULL);CHKERRQ(ierr);
  n_all_cc = n_faces+n_edges;
  ierr = PetscMalloc1(n_all_cc,&all_cc);CHKERRQ(ierr);
  for (i=0;i<n_faces;i++) {
    all_cc[i] = faces[i];
  }
  for (i=0;i<n_edges;i++) {
    all_cc[n_faces+i] = edges[i];
  }
  ierr = PetscFree(faces);CHKERRQ(ierr);
  ierr = PetscFree(edges);CHKERRQ(ierr);

  /* map interface's subsets */
  max_subset_size = 0;
  for (i=0;i<n_all_cc;i++) {
    PetscInt subset_size;
    ierr = ISGetLocalSize(all_cc[i],&subset_size);CHKERRQ(ierr);
    max_subset_size = PetscMax(max_subset_size,subset_size);
  }
  ierr = PetscMalloc1(max_subset_size,&auxmapping);CHKERRQ(ierr);
  ierr = PetscMalloc2(graph->ncc,&auxlocal_sequential,
                      graph->ncc,&auxlocal_parallel);CHKERRQ(ierr);
  ierr = PetscMalloc2(graph->ncc,&index_sequential,
                      graph->ncc,&index_parallel);CHKERRQ(ierr);

  /* if threshold is negative uses all sequential problems (possibly using MUMPS) */
  sub_schurs->use_mumps = PETSC_FALSE;
  if (seqthreshold < 0) {
    seqthreshold = max_subset_size;
#if defined(PETSC_HAVE_MUMPS)
    sub_schurs->use_mumps = !!A;
#endif
  }


  /* determine which problem has to be solved in parallel or sequentially */
  n_local_sequential_problems = 0;
  n_local_parallel_problems = 0;
  for (i=0;i<n_all_cc;i++) {
    PetscInt       subset_size,j,min_loc = 0;
    const PetscInt *idxs;

    ierr = ISGetLocalSize(all_cc[i],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(all_cc[i],&idxs);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingApply(graph->l2gmap,subset_size,idxs,auxmapping);CHKERRQ(ierr);
    for (j=1;j<subset_size;j++) {
      if (auxmapping[j]<auxmapping[min_loc]) {
        min_loc = j;
      }
    }
    if (subset_size > seqthreshold) {
      index_parallel[n_local_parallel_problems] = i;
      auxlocal_parallel[n_local_parallel_problems] = idxs[min_loc];
      n_local_parallel_problems++;
    } else {
      index_sequential[n_local_sequential_problems] = i;
      auxlocal_sequential[n_local_sequential_problems] = idxs[min_loc];
      n_local_sequential_problems++;
    }
    ierr = ISRestoreIndices(all_cc[i],&idxs);CHKERRQ(ierr);
  }

  /* Number parallel problems */
  auxglobal_parallel = 0;
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)graph->l2gmap),graph->l2gmap,n_local_parallel_problems,auxlocal_parallel,PETSC_NULL,&n_parallel_problems,&auxglobal_parallel);CHKERRQ(ierr);

  /* Number sequential problems */
  auxglobal_sequential = 0;
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)graph->l2gmap),graph->l2gmap,n_local_sequential_problems,auxlocal_sequential,PETSC_NULL,&n_sequential_problems,&auxglobal_sequential);CHKERRQ(ierr);

  /* update info in sub_schurs */
  if (sub_schurs->use_mumps && A) {
    ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    sub_schurs->A = A;
  }
  ierr = PetscObjectReference((PetscObject)S);CHKERRQ(ierr);
  sub_schurs->S = S;
  ierr = PetscObjectReference((PetscObject)is_I);CHKERRQ(ierr);
  sub_schurs->is_I = is_I;
  ierr = PetscObjectReference((PetscObject)is_B);CHKERRQ(ierr);
  sub_schurs->is_B = is_B;
  ierr = PetscObjectReference((PetscObject)graph->l2gmap);CHKERRQ(ierr);
  sub_schurs->l2gmap = graph->l2gmap;
  ierr = PetscObjectReference((PetscObject)BtoNmap);CHKERRQ(ierr);
  sub_schurs->BtoNmap = BtoNmap;
  sub_schurs->n_subs_seq = n_local_sequential_problems;
  sub_schurs->n_subs_par = n_local_parallel_problems;
  sub_schurs->n_subs_seq_g = n_sequential_problems;
  sub_schurs->n_subs_par_g = n_parallel_problems;
  sub_schurs->n_subs = sub_schurs->n_subs_seq + sub_schurs->n_subs_par;
  sub_schurs->is_subs = all_cc;
  sub_schurs->index_sequential = index_sequential;
  sub_schurs->index_parallel = index_parallel;
  sub_schurs->auxglobal_sequential = auxglobal_sequential;
  sub_schurs->auxglobal_parallel = auxglobal_parallel;

  /* free workspace */
  ierr = PetscFree(auxmapping);CHKERRQ(ierr);
  ierr = PetscFree2(auxlocal_sequential,auxlocal_parallel);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

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
  ierr = MatDestroy(&sub_schurs->A);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->S);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_I);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_B);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&sub_schurs->l2gmap);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&sub_schurs->BtoNmap);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_Ej_all);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDestroy(&sub_schurs->is_subs[i]);CHKERRQ(ierr);
    ierr = ISDestroy(&sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&sub_schurs->work1[i]);CHKERRQ(ierr);
    ierr = VecDestroy(&sub_schurs->work2[i]);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&sub_schurs->is_I_layer);CHKERRQ(ierr);
  if (sub_schurs->n_subs) {
    ierr = PetscFree(sub_schurs->is_subs);CHKERRQ(ierr);
    ierr = PetscFree4(sub_schurs->is_AEj_B,sub_schurs->S_Ej,sub_schurs->work1,sub_schurs->work2);CHKERRQ(ierr);
    ierr = PetscFree2(sub_schurs->index_sequential,sub_schurs->index_parallel);CHKERRQ(ierr);
    ierr = PetscFree(sub_schurs->auxglobal_sequential);CHKERRQ(ierr);
    ierr = PetscFree(sub_schurs->auxglobal_parallel);CHKERRQ(ierr);
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
