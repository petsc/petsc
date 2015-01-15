#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>

static PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursSetUpNew"
PetscErrorCode PCBDDCSubSchursSetUpNew(PCBDDCSubSchurs sub_schurs, PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers)
{
  Mat                    A_II,A_IB,A_BI,A_BB;
  Mat                    AE_II,*AE_IE,*AE_EI,*AE_EE;
  Mat                    global_schur_subsets,*submat_global_schur_subsets,work_mat;
  ISLocalToGlobalMapping l2gmap_subsets;
  IS                     is_I,*is_subset_B,temp_is;
  PetscInt               *nnz,*all_local_idx_G,*all_local_idx_B,*all_local_idx_N,*all_permutation_G;
  PetscInt               i,subset_size,max_subset_size;
  PetscInt               local_size,global_size;
  PetscBool              implicit_schurs;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  //ierr = PetscObjectTypeCompare((PetscObject)sub_schurs->S,MATSCHURCOMPLEMENT,&implicit_schurs);CHKERRQ(ierr);
  implicit_schurs = PETSC_TRUE;
  /* allocate space for schur complements */
  ierr = PetscMalloc4(sub_schurs->n_subs,&sub_schurs->is_AEj_B,
                      sub_schurs->n_subs,&sub_schurs->S_Ej,
                      sub_schurs->n_subs,&sub_schurs->work1,
                      sub_schurs->n_subs,&sub_schurs->work2);CHKERRQ(ierr);

  /* get Schur complement matrices */
  if (implicit_schurs) {
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

    /* IS for I dofs in original numbering */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->is_I),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&sub_schurs->is_I_layer);CHKERRQ(ierr);
    ierr = PetscFree(local_numbering);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_I_layer);CHKERRQ(ierr);
    /* IS for I dofs in boundary numbering */
    if (implicit_schurs) {
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
    if (implicit_schurs) {
      ierr = ISGetSize(sub_schurs->is_I,&n_I);CHKERRQ(ierr);
      ierr = ISCreateStride(PetscObjectComm((PetscObject)sub_schurs->is_I),n_I,0,1,&is_I);CHKERRQ(ierr);

      /* II block is the same */
      ierr = PetscObjectReference((PetscObject)A_II);CHKERRQ(ierr);
      AE_II = A_II;
    }
  }

  if (implicit_schurs) {
    /* subsets in original and boundary numbering */
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISDuplicate(sub_schurs->is_subs[i],&sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
      ierr = ISSort(sub_schurs->is_AEj_B[i]);CHKERRQ(ierr);
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
  ierr = PetscMalloc1(local_size,&all_local_idx_N);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_size,&nnz);CHKERRQ(ierr);

  /* Get local indices in local whole numbering and local boundary numbering */
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs_seq;i++) {
    PetscInt j;
    const    PetscInt *idxs;

    PetscInt local_problem_index = sub_schurs->index_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_AEj_B[local_problem_index],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_AEj_B[local_problem_index],&idxs);CHKERRQ(ierr);
    /* subset indices in local numbering */
    ierr = PetscMemcpy(all_local_idx_N+local_size,idxs,subset_size*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_AEj_B[local_problem_index],&idxs);CHKERRQ(ierr);
    for (j=0;j<subset_size;j++) nnz[local_size+j] = subset_size;
    local_size += subset_size;
  }

  /* subset indices in local boundary numbering */
  ierr = ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N,&subset_size,all_local_idx_B);CHKERRQ(ierr);
  if (subset_size != local_size) {
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %d != %d\n",subset_size,local_size);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all);CHKERRQ(ierr);

  /* Number dofs on all subsets (parallel) and sort numbering */
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->l2gmap,local_size,all_local_idx_N,PETSC_NULL,&global_size,&all_local_idx_G);CHKERRQ(ierr);
  ierr = PetscMalloc1(local_size,&all_permutation_G);CHKERRQ(ierr);
  for (i=0;i<local_size;i++) {
    all_permutation_G[i]=i;
  }
  ierr = PetscSortIntWithPermutation(local_size,all_local_idx_G,all_permutation_G);CHKERRQ(ierr);

  /* Local matrix of all local Schur on subsets */
  ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatSetSizes(sub_schurs->S_Ej_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
  ierr = MatSetType(sub_schurs->S_Ej_all,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(sub_schurs->S_Ej_all,0,nnz);CHKERRQ(ierr);
  ierr = PetscFree(nnz);CHKERRQ(ierr);

  if (implicit_schurs) {
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
  }
  ierr = MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Global matrix of all assembled Schur on subsets */
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,local_size,all_local_idx_G,PETSC_COPY_VALUES,&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size,l2gmap_subsets,&work_mat);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(work_mat,sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(work_mat,MAT_INITIAL_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&work_mat);CHKERRQ(ierr);

  /* Get local part of (\sum_j S_Ej) */
  for (i=0;i<local_size;i++) {
    all_local_idx_N[i] = all_local_idx_G[all_permutation_G[i]];
  }
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->l2gmap),local_size,all_local_idx_N,PETSC_OWN_POINTER,&temp_is);CHKERRQ(ierr);
  ierr = MatGetSubMatrices(global_schur_subsets,1,&temp_is,&temp_is,MAT_INITIAL_MATRIX,&submat_global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&global_schur_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is);CHKERRQ(ierr);
  for (i=0;i<local_size;i++) {
    all_local_idx_G[all_permutation_G[i]] = i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_G,PETSC_OWN_POINTER,&temp_is);CHKERRQ(ierr);
  ierr = ISSetPermutation(temp_is);CHKERRQ(ierr);
  ierr = MatPermute(submat_global_schur_subsets[0],temp_is,temp_is,&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&submat_global_schur_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is);CHKERRQ(ierr);
  ierr = PetscFree(all_permutation_G);CHKERRQ(ierr);
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

  /* if threshold is negative, uses all sequential problems */
  if (seqthreshold < 0) seqthreshold = max_subset_size;

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
  if (A) {
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
  sub_schurs->n_subs = ncc;
  ierr = PetscMalloc4(sub_schurs->n_subs,&sub_schurs->is_AEj_B,
                      sub_schurs->n_subs,&sub_schurs->S_Ej,
                      sub_schurs->n_subs,&sub_schurs->work1,
                      sub_schurs->n_subs,&sub_schurs->work2);CHKERRQ(ierr);
  ierr = PetscMalloc4(ncc,&is_subset_B,ncc,&AE_IE,ncc,&AE_EI,ncc,&AE_EE);CHKERRQ(ierr);

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
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)ItoNmap),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&sub_schurs->is_I_layer);CHKERRQ(ierr);
    ierr = PetscFree(local_numbering);CHKERRQ(ierr);
    ierr = ISSort(sub_schurs->is_I_layer);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,sub_schurs->is_I_layer,&is_I);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&ItoNmap);CHKERRQ(ierr);

    /* II block */
    ierr = MatGetSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II);CHKERRQ(ierr);
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    ierr = PetscObjectReference((PetscObject)is_A_I);CHKERRQ(ierr);
    sub_schurs->is_I_layer = is_A_I;

    /* IS for I dofs in I numbering (strided 1) */
    ierr = ISGetSize(is_A_I,&n_I);CHKERRQ(ierr);
    ierr = ISCreateStride(PetscObjectComm((PetscObject)is_A_I),n_I,0,1,&is_I);CHKERRQ(ierr);

    /* II block is the same */
    ierr = PetscObjectReference((PetscObject)A_II);CHKERRQ(ierr);
    AE_II = A_II;
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
