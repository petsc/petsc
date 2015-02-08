#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

PETSC_STATIC_INLINE PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat M, Mat *S);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCComputeExplicitSchur"
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat M, Mat *S)
{
  Mat            B, C, D, Bd, Cd, AinvBd;
  KSP            ksp;
  PC             pc;
  PetscBool      isLU, isILU, isCHOL, Bdense, Cdense;
  PetscReal      fill = 2.0;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)M),&size);CHKERRQ(ierr);
  if (size != 1) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for parallel matrices");
  }
  ierr = MatSchurComplementGetSubMatrices(M, NULL, NULL, &B, &C, &D);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(M, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCLU, &isLU);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCILU, &isILU);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCCHOLESKY, &isCHOL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) B, MATSEQDENSE, &Bdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) C, MATSEQDENSE, &Cdense);CHKERRQ(ierr);
  if (!Bdense) {
    ierr = MatConvert(B, MATSEQDENSE, MAT_INITIAL_MATRIX, &Bd);CHKERRQ(ierr);
  } else {
    Bd = B;
  }

  if (isLU || isILU || isCHOL) {
    Mat fact;

    ierr = KSPSetUp(ksp);CHKERRQ(ierr);
    ierr = PCFactorGetMatrix(pc, &fact);CHKERRQ(ierr);
    ierr = MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd);CHKERRQ(ierr);
    ierr = MatMatSolve(fact, Bd, AinvBd);CHKERRQ(ierr);
  } else {
    Mat Ainvd;

    ierr = PCComputeExplicitOperator(pc, &Ainvd);CHKERRQ(ierr);
    ierr = MatMatMult(Ainvd, Bd, MAT_INITIAL_MATRIX, fill, &AinvBd);CHKERRQ(ierr);
    ierr = MatDestroy(&Ainvd);CHKERRQ(ierr);
  }
  if (!Bdense) {
    ierr = MatDestroy(&Bd);CHKERRQ(ierr);
  }
  if (!Cdense) {
    ierr = MatConvert(C, MATSEQDENSE, MAT_INITIAL_MATRIX, &Cd);CHKERRQ(ierr);
  } else {
    Cd = C;
  }

  ierr = MatMatMult(Cd, AinvBd, MAT_INITIAL_MATRIX, fill, S);CHKERRQ(ierr);
  ierr = MatDestroy(&AinvBd);CHKERRQ(ierr);
  if (!Cdense) {
    ierr = MatDestroy(&Cd);CHKERRQ(ierr);
  }

  if (D) {
    Mat       Dd;
    PetscBool Ddense;

    ierr = PetscObjectTypeCompare((PetscObject)D,MATSEQDENSE,&Ddense);CHKERRQ(ierr);
    if (!Ddense) {
      ierr = MatConvert(D, MATSEQDENSE, MAT_INITIAL_MATRIX, &Dd);CHKERRQ(ierr);
    } else {
      Dd = D;
    }
    ierr = MatAYPX(*S,-1.0,Dd,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    if (!Ddense) {
      ierr = MatDestroy(&Dd);CHKERRQ(ierr);
    }
  } else {
    ierr = MatScale(*S,-1.0);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursSetUp"
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs sub_schurs, PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers, PetscBool compute_Stilda, PetscBool deluxe, PetscBool invert_Stildas, PetscBool use_edges, PetscBool use_faces)
{
  Mat                    A_II,A_IB,A_BI,A_BB;
  Mat                    AE_II,*AE_IE,*AE_EI,*AE_EE;
  Mat                    S_all,global_schur_subsets,work_mat;
  Mat                    S_Ej_tilda_all,S_Ej_inv_all;
  ISLocalToGlobalMapping l2gmap_subsets;
  IS                     is_I,*is_subset_B,temp_is;
  PetscInt               *nnz,*all_local_idx_G,*all_local_idx_B,*all_local_idx_N;
  PetscInt               i,subset_size,max_subset_size;
  PetscInt               extra,local_size,global_size;
  PetscBool              is_symmetric,Stilda_computed;
  PetscBLASInt           B_N,B_ierr,B_lwork,*pivots;
  PetscScalar            *work;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* get Schur complement matrices */
  if (!sub_schurs->use_mumps) {
    if (compute_Stilda) {
      SETERRQ(PetscObjectComm((PetscObject)sub_schurs->l2gmap),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS");
    }
    ierr = MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,&A_IB,&A_BI,&A_BB);CHKERRQ(ierr);
    ierr = PetscMalloc4(sub_schurs->n_subs,&is_subset_B,
                        sub_schurs->n_subs,&AE_IE,
                        sub_schurs->n_subs,&AE_EI,
                        sub_schurs->n_subs,&AE_EE);CHKERRQ(ierr);
  } else {
    is_subset_B = NULL;
    AE_IE = NULL;
    AE_EI = NULL;
    AE_EE = NULL;
  }

  /* determine interior problems */
  ierr = ISDestroy(&sub_schurs->is_I_layer);CHKERRQ(ierr);
  ierr = ISGetLocalSize(sub_schurs->is_I,&i);CHKERRQ(ierr);
  if (nlayers >= 0 && i) { /* Interior problems can be different from the original one */
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    if (xadj == NULL || adjncy == NULL) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot request layering without adjacency");
    }
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
  /* Get info on subset sizes and sum of all subsets sizes */
  max_subset_size = 0;
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs_seq;i++) {
    PetscInt j = sub_schurs->index_sequential[i];
    ierr = ISGetLocalSize(sub_schurs->is_subs[j],&subset_size);CHKERRQ(ierr);
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
    ierr = ISGetLocalSize(sub_schurs->is_subs[local_problem_index],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_subs[local_problem_index],&idxs);CHKERRQ(ierr);
    /* subset indices in local numbering */
    ierr = PetscMemcpy(all_local_idx_N+local_size+extra,idxs,subset_size*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_subs[local_problem_index],&idxs);CHKERRQ(ierr);
    for (j=0;j<subset_size;j++) nnz[local_size+j] = subset_size;
    local_size += subset_size;
  }

  S_all = NULL;
  if (sub_schurs->A) {
    ierr = MatIsSymmetric(sub_schurs->A,0.0,&is_symmetric);CHKERRQ(ierr);
  }
  if (sub_schurs->n_subs) {
    if (!sub_schurs->use_mumps) {
      /* subsets in original and boundary numbering */
      for (i=0;i<sub_schurs->n_subs;i++) {
        ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[i],&is_subset_B[i]);CHKERRQ(ierr);
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
        ierr = MatDestroy(&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
        ierr = MatCreateSchurComplement(AE_II,AE_II,AE_IE[i],AE_EI[i],AE_EE[i],&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
        ierr = MatDestroy(&AE_EE[i]);CHKERRQ(ierr);
        ierr = MatDestroy(&AE_IE[i]);CHKERRQ(ierr);
        ierr = MatDestroy(&AE_EI[i]);CHKERRQ(ierr);
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
        ierr = ISDestroy(&is_subset_B[i]);CHKERRQ(ierr);
      }
      ierr = PetscFree4(is_subset_B,AE_IE,AE_EI,AE_EE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
    } else {
      Mat           A,F;
      IS            is_A_all;
      PetscInt      *idxs_schur,n_I;

      /* get working mat */
      ierr = ISGetLocalSize(sub_schurs->is_I_layer,&n_I);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size+n_I,all_local_idx_N,PETSC_COPY_VALUES,&is_A_all);CHKERRQ(ierr);
      ierr = MatGetSubMatrixUnsorted(sub_schurs->A,is_A_all,is_A_all,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
      ierr = ISDestroy(&is_A_all);CHKERRQ(ierr);

      if (n_I) {
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
        ierr = MatDestroy(&F);CHKERRQ(ierr);
      } else {
        ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&S_all);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&A);CHKERRQ(ierr);
#endif
    }
  } else {
    ierr = PetscFree(nnz);CHKERRQ(ierr);
    ierr = PetscFree(all_local_idx_N);CHKERRQ(ierr);
  }

  if (!sub_schurs->n_subs_seq_g) {
    PetscFunctionReturn(0);
  }

  /* subset indices in local boundary numbering */
  if (!sub_schurs->is_Ej_all) {
    ierr = ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B);CHKERRQ(ierr);
    if (subset_size != local_size) {
      SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %d != %d\n",subset_size,local_size);
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all);CHKERRQ(ierr);
  }

  /* Local matrix of all local Schur on subsets transposed */
  if (!sub_schurs->S_Ej_all) {
    ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->S_Ej_all);CHKERRQ(ierr);
    ierr = MatSetSizes(sub_schurs->S_Ej_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
    ierr = MatSetType(sub_schurs->S_Ej_all,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(sub_schurs->S_Ej_all,0,nnz);CHKERRQ(ierr);
  } else {
    ierr = MatZeroEntries(sub_schurs->S_Ej_all);CHKERRQ(ierr);
  }

  S_Ej_tilda_all = 0;
  S_Ej_inv_all = 0;
  work = NULL;
  pivots = NULL;
  Stilda_computed = PETSC_FALSE;
  if (sub_schurs->n_subs && deluxe) { /* workspace needed only for GETRI */
    PetscScalar lwork;

    B_lwork = -1;
    ierr = PetscBLASIntCast(max_subset_size,&B_N);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,work,&B_N,pivots,&lwork,&B_lwork,&B_ierr));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
    ierr = PetscBLASIntCast((PetscInt)lwork,&B_lwork);CHKERRQ(ierr);
    ierr = PetscMalloc2(B_lwork,&work,max_subset_size,&pivots);CHKERRQ(ierr);
  }

  ierr = PetscBTMemzero(sub_schurs->n_subs,sub_schurs->computed_Stilda_subs);CHKERRQ(ierr);
  if (!sub_schurs->use_mumps) {
    PetscScalar *fill_vals;
    PetscInt    *dummy_idx;

    /* Work arrays */
    ierr = PetscMalloc1(max_subset_size,&dummy_idx);CHKERRQ(ierr);
    /* Loop on local problems end compute Schur complements explicitly */
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs_seq;i++) {
      Mat       S_Ej_expl;
      PetscInt  j,lpi = sub_schurs->index_sequential[i];
      PetscBool Sdense;

      ierr = PCBDDCComputeExplicitSchur(sub_schurs->S_Ej[lpi],&S_Ej_expl);CHKERRQ(ierr);
      ierr = ISGetLocalSize(sub_schurs->is_subs[lpi],&subset_size);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)S_Ej_expl,MATSEQDENSE,&Sdense);CHKERRQ(ierr);
      if (Sdense) {
        for (j=0;j<subset_size;j++) {
          dummy_idx[j]=local_size+j;
        }
        ierr = MatDenseGetArray(S_Ej_expl,&fill_vals);CHKERRQ(ierr);
        ierr = MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,fill_vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(S_Ej_expl,&fill_vals);CHKERRQ(ierr);
      } else {
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented for sparse matrices");
      }
      local_size += subset_size;
      ierr = MatDestroy(&sub_schurs->S_Ej[lpi]);CHKERRQ(ierr);
      sub_schurs->S_Ej[lpi] = S_Ej_expl;
    }
    /* Stildas are not computed without mumps */
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = MatDestroy(&sub_schurs->S_Ej_tilda[i]);CHKERRQ(ierr);
    }
    ierr = PetscFree(dummy_idx);CHKERRQ(ierr);
  } else {
    PetscInt  *dummy_idx,n_all;

    if (compute_Stilda) {
      ierr = MatCreate(PETSC_COMM_SELF,&S_Ej_tilda_all);CHKERRQ(ierr);
      ierr = MatSetSizes(S_Ej_tilda_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
      ierr = MatSetType(S_Ej_tilda_all,MATAIJ);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(S_Ej_tilda_all,0,nnz);CHKERRQ(ierr);
      if (deluxe) {
        ierr = MatCreate(PETSC_COMM_SELF,&S_Ej_inv_all);CHKERRQ(ierr);
        ierr = MatSetSizes(S_Ej_inv_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
        ierr = MatSetType(S_Ej_inv_all,MATAIJ);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(S_Ej_inv_all,0,nnz);CHKERRQ(ierr);
      }
    }

    ierr = MatGetSize(S_all,&n_all,NULL);CHKERRQ(ierr);
    local_size = 0;
    /* Work arrays */
    ierr = PetscMalloc1(max_subset_size,&dummy_idx);CHKERRQ(ierr);

    Stilda_computed = PETSC_FALSE;
    for (i=0;i<sub_schurs->n_subs;i++) {
      IS  is_E;
      PetscScalar *vals;
      PetscInt j;

      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      for (j=0;j<subset_size;j++) {
        dummy_idx[j]=local_size+j;
      }
      ierr = ISCreateStride(PetscObjectComm((PetscObject)sub_schurs->is_I),subset_size,local_size,1,&is_E);CHKERRQ(ierr);
      if (sub_schurs->S_Ej[i]) {
        ierr = MatGetSubMatrix(S_all,is_E,is_E,MAT_REUSE_MATRIX,&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
      } else {
        ierr = MatGetSubMatrix(S_all,is_E,is_E,MAT_INITIAL_MATRIX,&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
      }
      ierr = MatDenseGetArray(sub_schurs->S_Ej[i],&vals);CHKERRQ(ierr);
      ierr = MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(sub_schurs->S_Ej[i],&vals);CHKERRQ(ierr);

      if (compute_Stilda && ((PetscBTLookup(sub_schurs->is_edge,i) && use_edges) || use_faces)) {
        ierr = PetscBTSet(sub_schurs->computed_Stilda_subs,i);CHKERRQ(ierr);
        Stilda_computed = PETSC_TRUE;
      }

      if (PetscBTLookup(sub_schurs->computed_Stilda_subs,i)) {
        Mat Stilda;
        if (sub_schurs->n_subs == 1) {
          ierr = PetscObjectReference((PetscObject)sub_schurs->S_Ej[i]);CHKERRQ(ierr);
          Stilda = sub_schurs->S_Ej[i];
        } else {
          KSP ksp;
          PC  pc;
          Mat S_EF,S_FE,S_FF,Stilda_impl;
          IS  is_F;
          PetscScalar eps=1.e-8;
          PetscBool chop=PETSC_FALSE;

          ierr = ISComplement(is_E,0,n_all,&is_F);CHKERRQ(ierr);
          ierr = MatGetSubMatrix(S_all,is_E,is_F,MAT_INITIAL_MATRIX,&S_EF);CHKERRQ(ierr);
          ierr = MatGetSubMatrix(S_all,is_F,is_F,MAT_INITIAL_MATRIX,&S_FF);CHKERRQ(ierr);
          ierr = MatGetSubMatrix(S_all,is_F,is_E,MAT_INITIAL_MATRIX,&S_FE);CHKERRQ(ierr);
          ierr = ISDestroy(&is_F);CHKERRQ(ierr);
          if (chop) {
            ierr = MatChop(S_FF,eps);CHKERRQ(ierr);
            ierr = MatConvert(S_FF,MATAIJ,MAT_REUSE_MATRIX,&S_FF);CHKERRQ(ierr);
          }
          ierr = MatCreateSchurComplement(S_FF,S_FF,S_FE,S_EF,sub_schurs->S_Ej[i],&Stilda_impl);CHKERRQ(ierr);
          ierr = MatDestroy(&S_FF);CHKERRQ(ierr);
          ierr = MatDestroy(&S_FE);CHKERRQ(ierr);
          ierr = MatDestroy(&S_EF);CHKERRQ(ierr);
          ierr = MatSchurComplementGetKSP(Stilda_impl,&ksp);CHKERRQ(ierr);
          ierr = KSPSetType(ksp,KSPPREONLY);CHKERRQ(ierr);
          ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
          if (is_symmetric) {
            ierr = PCSetType(pc,PCCHOLESKY);CHKERRQ(ierr);
          } else {
            ierr = PCSetType(pc,PCLU);CHKERRQ(ierr);
          }
          if (!chop) {
            ierr = PCFactorSetUseInPlace(pc,PETSC_TRUE);CHKERRQ(ierr);
          } else {
            ierr = PCFactorSetMatSolverPackage(pc,MATSOLVERMUMPS);CHKERRQ(ierr);
          }
          ierr = KSPSetUp(ksp);CHKERRQ(ierr);
          ierr = PCBDDCComputeExplicitSchur(Stilda_impl,&Stilda);CHKERRQ(ierr);
          ierr = MatDestroy(&Stilda_impl);CHKERRQ(ierr);
        }
/*
        PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_MATLAB);
        ierr = MatView(Stilda,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
        PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);
*/
        ierr = MatDestroy(&sub_schurs->S_Ej_tilda[i]);CHKERRQ(ierr);
        sub_schurs->S_Ej_tilda[i] = Stilda;


        ierr = MatDenseGetArray(sub_schurs->S_Ej_tilda[i],&vals);CHKERRQ(ierr);
        if (deluxe) { /* when using deluxe scaling, we need (S_1^-1+S_2^-1)^-1 */
          PetscScalar *vals2;

          ierr = MatDenseGetArray(sub_schurs->S_Ej[i],&vals2);CHKERRQ(ierr);
          /* need to be optimized (cholesky) */
          ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          if (invert_Stildas) { /* Stildas can be singular */
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,vals,&B_N,pivots,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,vals,&B_N,pivots,work,&B_lwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          }
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,vals2,&B_N,pivots,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,vals2,&B_N,pivots,work,&B_lwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          ierr = MatSetValues(S_Ej_inv_all,subset_size,dummy_idx,subset_size,dummy_idx,vals2,INSERT_VALUES);CHKERRQ(ierr);
          ierr = MatDenseRestoreArray(sub_schurs->S_Ej[i],&vals2);CHKERRQ(ierr);
        }
        ierr = MatSetValues(S_Ej_tilda_all,subset_size,dummy_idx,subset_size,dummy_idx,vals,INSERT_VALUES);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(sub_schurs->S_Ej_tilda[i],&vals);CHKERRQ(ierr);
      } else {
        ierr = MatDestroy(&sub_schurs->S_Ej_tilda[i]);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&is_E);CHKERRQ(ierr);
      local_size += subset_size;
    }
    ierr = PetscFree(dummy_idx);CHKERRQ(ierr);
  }
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = MatDestroy(&S_all);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (compute_Stilda) {
    ierr = MatAssemblyBegin(S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (deluxe) {
      ierr = MatAssemblyBegin(S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }

  /* Global matrix of all assembled Schur on subsets */
  ierr = PCBDDCSubsetNumbering(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->l2gmap,local_size,all_local_idx_N+extra,PETSC_NULL,&global_size,&all_local_idx_G);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,local_size,all_local_idx_G,PETSC_COPY_VALUES,&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatCreateIS(PetscObjectComm((PetscObject)sub_schurs->l2gmap),1,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size,l2gmap_subsets,&work_mat);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatISSetLocalMat(work_mat,sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)work_mat),&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatSetSizes(global_schur_subsets,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size);CHKERRQ(ierr);
  ierr = MatSetType(global_schur_subsets,MATMPIAIJ);CHKERRQ(ierr);
  ierr = MatISSetMPIXAIJPreallocation_Private(work_mat,global_schur_subsets,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
  /* Get local part of (\sum_j S_Ej) */
  ierr = PetscFree(all_local_idx_N);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->l2gmap),local_size,all_local_idx_G,PETSC_OWN_POINTER,&temp_is);CHKERRQ(ierr);
  if (sub_schurs->sum_S_Ej_all) {
    ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_REUSE_MATRIX,&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  } else {
    ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_INITIAL_MATRIX,&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  }

  if (Stilda_computed) {
    ierr = MatISSetLocalMat(work_mat,S_Ej_tilda_all);CHKERRQ(ierr);
    ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
    if (sub_schurs->sum_S_Ej_tilda_all) {
      ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_REUSE_MATRIX,&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
    } else {
      ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_INITIAL_MATRIX,&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
    }
    if (deluxe) {
      PetscInt    cum;
      PetscScalar *array,*array2;
      ierr = MatISSetLocalMat(work_mat,S_Ej_inv_all);CHKERRQ(ierr);
      ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
      if (sub_schurs->sum_S_Ej_inv_all) {
        ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_REUSE_MATRIX,&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
      } else {
        ierr = MatGetSubMatrixUnsorted(global_schur_subsets,temp_is,temp_is,MAT_INITIAL_MATRIX,&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
      }
      /* invert blocks of sum_S_Ej_inv_all */
      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&array);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&array2);CHKERRQ(ierr);
      cum = 0;
      for (i=0;i<sub_schurs->n_subs;i++) {
        ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
        if (PetscBTLookup(sub_schurs->computed_Stilda_subs,i)) {
          /* need to be optimized (cholesky) */
          ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,array+cum,&B_N,pivots,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,array+cum,&B_N,pivots,work,&B_lwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          if (invert_Stildas) { /* Stildas can be singular */
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,array2+cum,&B_N,pivots,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,array2+cum,&B_N,pivots,work,&B_lwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          }
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
        }
        cum += subset_size*subset_size;
      }
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&array);CHKERRQ(ierr);
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&array2);CHKERRQ(ierr);
    }
  }

  ierr = PetscFree2(work,pivots);CHKERRQ(ierr);
  ierr = MatDestroy(&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&S_Ej_tilda_all);CHKERRQ(ierr);
  ierr = MatDestroy(&S_Ej_inv_all);CHKERRQ(ierr);
  ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
  ierr = ISDestroy(&temp_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursInit"
PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, Mat A, Mat S, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap, PetscInt seqthreshold)
{
  IS                  *faces,*edges,*all_cc,vertices;
  PetscInt            *index_sequential,*index_parallel;
  PetscInt            *auxlocal_sequential,*auxlocal_parallel;
  PetscInt            *auxglobal_sequential,*auxglobal_parallel;
  PetscInt            *auxmapping;
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
  ierr = PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices);CHKERRQ(ierr);
  n_all_cc = n_faces+n_edges;
  ierr = PetscBTCreate(n_all_cc,&sub_schurs->is_edge);CHKERRQ(ierr);
  ierr = PetscBTCreate(n_all_cc,&sub_schurs->computed_Stilda_subs);CHKERRQ(ierr);
  ierr = PetscMalloc1(n_all_cc,&all_cc);CHKERRQ(ierr);
  for (i=0;i<n_faces;i++) {
    all_cc[i] = faces[i];
  }
  for (i=0;i<n_edges;i++) {
    all_cc[n_faces+i] = edges[i];
    ierr = PetscBTSet(sub_schurs->is_edge,n_faces+i);CHKERRQ(ierr);
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
  if (!sub_schurs->use_mumps) { /* for adaptive selection */
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISSort(sub_schurs->is_subs[i]);CHKERRQ(ierr);
    }
  }
  sub_schurs->is_Ej_com = vertices;
  sub_schurs->index_sequential = index_sequential;
  sub_schurs->index_parallel = index_parallel;
  sub_schurs->auxglobal_sequential = auxglobal_sequential;
  sub_schurs->auxglobal_parallel = auxglobal_parallel;


  /* allocate space for schur complements */
  ierr = PetscCalloc2(sub_schurs->n_subs,&sub_schurs->S_Ej,
                      sub_schurs->n_subs,&sub_schurs->S_Ej_tilda);CHKERRQ(ierr);
  sub_schurs->S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_inv_all = NULL;
  sub_schurs->sum_S_Ej_tilda_all = NULL;
  sub_schurs->is_Ej_all = NULL;

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
  ierr = MatDestroy(&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_Ej_all);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_Ej_com);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&sub_schurs->is_edge);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&sub_schurs->computed_Stilda_subs);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDestroy(&sub_schurs->is_subs[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->S_Ej[i]);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->S_Ej_tilda[i]);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&sub_schurs->is_I_layer);CHKERRQ(ierr);
  if (sub_schurs->n_subs) {
    ierr = PetscFree(sub_schurs->is_subs);CHKERRQ(ierr);
    ierr = PetscFree2(sub_schurs->S_Ej,sub_schurs->S_Ej_tilda);CHKERRQ(ierr);
  }
  ierr = PetscFree2(sub_schurs->index_sequential,sub_schurs->index_parallel);CHKERRQ(ierr);
  ierr = PetscFree(sub_schurs->auxglobal_sequential);CHKERRQ(ierr);
  ierr = PetscFree(sub_schurs->auxglobal_parallel);CHKERRQ(ierr);
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
