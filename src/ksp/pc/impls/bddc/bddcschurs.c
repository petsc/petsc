#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <petscblaslapack.h>

PETSC_STATIC_INLINE PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat,PetscBool,MatReuse,Mat*);
static PetscErrorCode PCBDDCMumpsInteriorSolve(PC,Vec,Vec);
static PetscErrorCode PCBDDCMumpsCorrectionSolve(PC,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsCorrectionSolve_Private"
static PetscErrorCode PCBDDCMumpsCorrectionSolve_Private(PC pc, Vec rhs, Vec sol, PetscBool transpose)
{
  PCBDDCReuseMumps ctx;
#if defined(PETSC_HAVE_MUMPS)
  PetscInt         ival;
#endif
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void **)&ctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsGetIcntl(ctx->F,26,&ival);CHKERRQ(ierr);
  ierr = MatMumpsSetIcntl(ctx->F,26,-1);CHKERRQ(ierr);
#endif
  if (transpose) {
    ierr = MatSolveTranspose(ctx->F,rhs,sol);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(ctx->F,rhs,sol);CHKERRQ(ierr);
  }
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ctx->F,26,ival);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsCorrectionSolve"
static PetscErrorCode PCBDDCMumpsCorrectionSolve(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCMumpsCorrectionSolve_Private(pc,rhs,sol,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsCorrectionSolveTranspose"
static PetscErrorCode PCBDDCMumpsCorrectionSolveTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCMumpsCorrectionSolve_Private(pc,rhs,sol,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCReuseMumpsReset"
static PetscErrorCode PCBDDCReuseMumpsReset(PCBDDCReuseMumps reuse)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&reuse->F);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->sol);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->rhs);CHKERRQ(ierr);
  ierr = PCDestroy(&reuse->interior_solver);CHKERRQ(ierr);
  ierr = PCDestroy(&reuse->correction_solver);CHKERRQ(ierr);
  ierr = ISDestroy(&reuse->is_R);CHKERRQ(ierr);
  ierr = ISDestroy(&reuse->is_B);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&reuse->correction_scatter_B);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->sol_B);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->rhs_B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsInteriorSolve_Private"
static PetscErrorCode PCBDDCMumpsInteriorSolve_Private(PC pc, Vec rhs, Vec sol, PetscBool transpose)
{
  PCBDDCReuseMumps ctx;
  PetscScalar      *array,*array_mumps;
#if defined(PETSC_HAVE_MUMPS)
  PetscInt         ival;
#endif
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void **)&ctx);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsGetIcntl(ctx->F,26,&ival);CHKERRQ(ierr);
  ierr = MatMumpsSetIcntl(ctx->F,26,0);CHKERRQ(ierr);
#endif
  /* copy rhs into factored matrix workspace (can it be avoided?, MatSolve_MUMPS has another copy b->x internally) */
  ierr = VecGetArrayRead(rhs,(const PetscScalar**)&array);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->rhs,&array_mumps);CHKERRQ(ierr);
  ierr = PetscMemcpy(array_mumps,array,ctx->n_I*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->rhs,&array_mumps);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(rhs,(const PetscScalar**)&array);CHKERRQ(ierr);

  if (transpose) {
    ierr = MatSolveTranspose(ctx->F,ctx->rhs,ctx->sol);CHKERRQ(ierr);
  } else {
    ierr = MatSolve(ctx->F,ctx->rhs,ctx->sol);CHKERRQ(ierr);
  }

  /* get back data to caller worskpace */
  ierr = VecGetArrayRead(ctx->sol,(const PetscScalar**)&array_mumps);CHKERRQ(ierr);
  ierr = VecGetArray(sol,&array);CHKERRQ(ierr);
  ierr = PetscMemcpy(array,array_mumps,ctx->n_I*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(sol,&array);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->sol,(const PetscScalar**)&array_mumps);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ctx->F,26,ival);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsInteriorSolve"
static PetscErrorCode PCBDDCMumpsInteriorSolve(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCMumpsInteriorSolve_Private(pc,rhs,sol,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCMumpsInteriorSolveTranspose"
static PetscErrorCode PCBDDCMumpsInteriorSolveTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCMumpsInteriorSolve_Private(pc,rhs,sol,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCComputeExplicitSchur"
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat M, PetscBool issym, MatReuse reuse, Mat *S)
{
  Mat            B, C, D, Bd, Cd, AinvBd;
  KSP            ksp;
  PC             pc;
  PetscBool      isLU, isILU, isCHOL, Bdense, Cdense;
  PetscReal      fill = 2.0;
  PetscInt       n_I;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)M),&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for parallel matrices");
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool Sdense;

    ierr = PetscObjectTypeCompare((PetscObject)*S, MATSEQDENSE, &Sdense);CHKERRQ(ierr);
    if (!Sdense) SETERRQ(PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"S should dense");
  }
  ierr = MatSchurComplementGetSubMatrices(M, NULL, NULL, &B, &C, &D);CHKERRQ(ierr);
  ierr = MatSchurComplementGetKSP(M, &ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCLU, &isLU);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCILU, &isILU);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) pc, PCCHOLESKY, &isCHOL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) B, MATSEQDENSE, &Bdense);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) C, MATSEQDENSE, &Cdense);CHKERRQ(ierr);
  ierr = MatGetSize(B,&n_I,NULL);CHKERRQ(ierr);
  if (n_I) {
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
      PetscBool ex = PETSC_TRUE;

      if (ex) {
        Mat Ainvd;

        ierr = PCComputeExplicitOperator(pc, &Ainvd);CHKERRQ(ierr);
        ierr = MatMatMult(Ainvd, Bd, MAT_INITIAL_MATRIX, fill, &AinvBd);CHKERRQ(ierr);
        ierr = MatDestroy(&Ainvd);CHKERRQ(ierr);
      } else {
        Vec         sol,rhs;
        PetscScalar *arrayrhs,*arraysol;
        PetscInt    i,nrhs,n;

        ierr = MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd);CHKERRQ(ierr);
        ierr = MatGetSize(Bd,&n,&nrhs);CHKERRQ(ierr);
        ierr = MatDenseGetArray(Bd,&arrayrhs);CHKERRQ(ierr);
        ierr = MatDenseGetArray(AinvBd,&arraysol);CHKERRQ(ierr);
        ierr = KSPGetSolution(ksp,&sol);CHKERRQ(ierr);
        ierr = KSPGetRhs(ksp,&rhs);CHKERRQ(ierr);
        for (i=0;i<nrhs;i++) {
          ierr = VecPlaceArray(rhs,arrayrhs+i*n);CHKERRQ(ierr);
          ierr = VecPlaceArray(sol,arraysol+i*n);CHKERRQ(ierr);
          ierr = KSPSolve(ksp,rhs,sol);CHKERRQ(ierr);
          ierr = VecResetArray(rhs);CHKERRQ(ierr);
          ierr = VecResetArray(sol);CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(Bd,&arrayrhs);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(AinvBd,&arrayrhs);CHKERRQ(ierr);
      }
    }
    if (!Bdense & !issym) {
      ierr = MatDestroy(&Bd);CHKERRQ(ierr);
    }

    if (!issym) {
      if (!Cdense) {
        ierr = MatConvert(C, MATSEQDENSE, MAT_INITIAL_MATRIX, &Cd);CHKERRQ(ierr);
      } else {
        Cd = C;
      }
      ierr = MatMatMult(Cd, AinvBd, reuse, fill, S);CHKERRQ(ierr);
      if (!Cdense) {
        ierr = MatDestroy(&Cd);CHKERRQ(ierr);
      }
    } else {
      ierr = MatTransposeMatMult(Bd, AinvBd, reuse, fill, S);CHKERRQ(ierr);
      if (!Bdense) {
        ierr = MatDestroy(&Bd);CHKERRQ(ierr);
      }
    }
    ierr = MatDestroy(&AinvBd);CHKERRQ(ierr);
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
    if (n_I) {
      ierr = MatAYPX(*S,-1.0,Dd,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    } else {
      if (reuse == MAT_INITIAL_MATRIX) {
        ierr = MatDuplicate(Dd,MAT_COPY_VALUES,S);CHKERRQ(ierr);
      } else {
        ierr = MatCopy(Dd,*S,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
      }
    }
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
PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs sub_schurs, Mat Ain, Mat Sin, PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers, PetscBool faster_deluxe, PetscBool compute_Stilda, PetscBool reuse_solvers)
{
  Mat                    F,A_II,A_IB,A_BI,A_BB,AE_II;
  Mat                    S_all;
  Mat                    global_schur_subsets,work_mat,*submats;
  ISLocalToGlobalMapping l2gmap_subsets;
  IS                     is_I,is_I_layer;
  IS                     all_subsets,all_subsets_mult,all_subsets_n;
  PetscInt               *nnz,*all_local_idx_N;
  PetscInt               *auxnum1,*auxnum2;
  PetscInt               i,subset_size,max_subset_size;
  PetscInt               extra,local_size,global_size;
  PetscBLASInt           B_N,B_ierr,B_lwork,*pivots;
  PetscScalar            *Bwork;
  PetscSubcomm           subcomm;
  PetscMPIInt            color,rank;
  MPI_Comm               comm_n;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  /* update info in sub_schurs */
  ierr = MatDestroy(&sub_schurs->A);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->S);CHKERRQ(ierr);
  sub_schurs->is_hermitian = PETSC_FALSE;
  sub_schurs->is_posdef = PETSC_FALSE;
  if (Ain) {
    PetscBool isseqaij;
    /* determine if we are dealing with hermitian positive definite problems */
#if !defined(PETSC_USE_COMPLEX)
    if (Ain->symmetric_set) {
      sub_schurs->is_hermitian = Ain->symmetric;
    }
#else
    if (Ain->hermitian_set) {
      sub_schurs->is_hermitian = Ain->hermitian;
    }
#endif
    if (Ain->spd_set) {
      sub_schurs->is_posdef = Ain->spd;
    }

    /* check */
    ierr = PetscObjectTypeCompare((PetscObject)Ain,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (compute_Stilda && (!sub_schurs->is_hermitian || !sub_schurs->is_posdef)) {
      PetscInt lsize;

      ierr = MatGetSize(Ain,&lsize,NULL);CHKERRQ(ierr);
      if (lsize) {
        PetscScalar val;
        PetscReal   norm;
        Vec         vec1,vec2,vec3;

        ierr = MatCreateVecs(Ain,&vec1,&vec2);CHKERRQ(ierr);
        ierr = VecDuplicate(vec1,&vec3);CHKERRQ(ierr);
        ierr = VecSetRandom(vec1,NULL);CHKERRQ(ierr);
        ierr = MatMult(Ain,vec1,vec2);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
        ierr = MatMultTranspose(Ain,vec1,vec3);CHKERRQ(ierr);
#else
        ierr = MatMultHermitianTranspose(Ain,vec1,vec3);CHKERRQ(ierr);
#endif
        ierr = VecAXPY(vec3,-1.0,vec2);CHKERRQ(ierr);
        ierr = VecNorm(vec3,NORM_INFINITY,&norm);CHKERRQ(ierr);
        if (norm > PetscSqrtReal(PETSC_SMALL)) {
          sub_schurs->is_hermitian = PETSC_FALSE;
        } else {
          sub_schurs->is_hermitian = PETSC_TRUE;
        }
        ierr = VecDot(vec1,vec2,&val);CHKERRQ(ierr);
        if (PetscRealPart(val) > 0. && PetscAbsReal(PetscImaginaryPart(val)) < PETSC_SMALL) sub_schurs->is_posdef = PETSC_TRUE;
        ierr = VecDestroy(&vec1);CHKERRQ(ierr);
        ierr = VecDestroy(&vec2);CHKERRQ(ierr);
        ierr = VecDestroy(&vec3);CHKERRQ(ierr);
      } else {
        sub_schurs->is_hermitian = PETSC_TRUE;
        sub_schurs->is_posdef = PETSC_TRUE;
      }
    }
    if (isseqaij) {
      ierr = PetscObjectReference((PetscObject)Ain);CHKERRQ(ierr);
      sub_schurs->A = Ain;
    } else {
      ierr = MatConvert(Ain,MATSEQAIJ,MAT_INITIAL_MATRIX,&sub_schurs->A);CHKERRQ(ierr);
    }
  }
  if (compute_Stilda && (!sub_schurs->is_hermitian || !sub_schurs->is_posdef)) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"General matrix pencils are not currently supported (%D,%D)",sub_schurs->is_hermitian,sub_schurs->is_posdef);

  ierr = PetscObjectReference((PetscObject)Sin);CHKERRQ(ierr);
  sub_schurs->S = Sin;
  if (sub_schurs->use_mumps) {
    sub_schurs->use_mumps = (PetscBool)(!!sub_schurs->A);
  }

  /* preliminary checks */
  if (!sub_schurs->use_mumps && compute_Stilda) SETERRQ(PetscObjectComm((PetscObject)sub_schurs->l2gmap),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS");

  /* restrict work on active processes */
  color = 0;
  if (!sub_schurs->n_subs) color = 1; /* this can happen if we are in a multilevel case or if the subdomain is disconnected */
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank);CHKERRQ(ierr);
  ierr = PetscSubcommCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&subcomm);CHKERRQ(ierr);
  ierr = PetscSubcommSetNumber(subcomm,2);CHKERRQ(ierr);
  ierr = PetscSubcommSetTypeGeneral(subcomm,color,rank);CHKERRQ(ierr);
  ierr = PetscCommDuplicate(PetscSubcommChild(subcomm),&comm_n,NULL);CHKERRQ(ierr);
  ierr = PetscSubcommDestroy(&subcomm);CHKERRQ(ierr);
  if (!sub_schurs->n_subs) {
    ierr = PetscCommDestroy(&comm_n);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* ierr = PetscCommDuplicate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&comm_n,NULL);CHKERRQ(ierr); */

  /* get Schur complement matrices */
  if (!sub_schurs->use_mumps) {
    Mat       tA_IB,tA_BI,tA_BB;
    PetscBool isseqsbaij;
    ierr = MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,&tA_IB,&tA_BI,&tA_BB);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)tA_BB,MATSEQSBAIJ,&isseqsbaij);CHKERRQ(ierr);
    if (isseqsbaij) {
      ierr = MatConvert(tA_BB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BB);CHKERRQ(ierr);
      ierr = MatConvert(tA_IB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_IB);CHKERRQ(ierr);
      ierr = MatConvert(tA_BI,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BI);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)tA_BB);CHKERRQ(ierr);
      A_BB = tA_BB;
      ierr = PetscObjectReference((PetscObject)tA_IB);CHKERRQ(ierr);
      A_IB = tA_IB;
      ierr = PetscObjectReference((PetscObject)tA_BI);CHKERRQ(ierr);
      A_BI = tA_BI;
    }
  } else {
    A_II = NULL;
    A_IB = NULL;
    A_BI = NULL;
    A_BB = NULL;
  }
  S_all = NULL;

  /* determine interior problems */
  ierr = ISGetLocalSize(sub_schurs->is_I,&i);CHKERRQ(ierr);
  if (nlayers >= 0 && i) { /* Interior problems can be different from the original one */
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    if (!xadj || !adjncy) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot request layering without adjacency");
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
      if (n_local_dofs > n_I+n_B) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error querying layer %D. Out of bound access (%D > %D)",layer,n_local_dofs,n_I+n_B);
      ierr = PCBDDCAdjGetNextLayer_Private(local_numbering+n_local_dofs,n_prev_added,touched,xadj,adjncy,&n_added);CHKERRQ(ierr);
      n_prev_added = n_added;
      n_local_dofs += n_added;
      if (!n_added) break;
    }
    ierr = PetscBTDestroy(&touched);CHKERRQ(ierr);

    /* IS for I layer dofs in original numbering */
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->is_I),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&is_I_layer);CHKERRQ(ierr);
    ierr = PetscFree(local_numbering);CHKERRQ(ierr);
    ierr = ISSort(is_I_layer);CHKERRQ(ierr);
    /* IS for I layer dofs in I numbering */
    if (!sub_schurs->use_mumps) {
      ISLocalToGlobalMapping ItoNmap;
      ierr = ISLocalToGlobalMappingCreateIS(sub_schurs->is_I,&ItoNmap);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,is_I_layer,&is_I);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ItoNmap);CHKERRQ(ierr);

      /* II block */
      ierr = MatGetSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II);CHKERRQ(ierr);
    }
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    ierr = PetscObjectReference((PetscObject)sub_schurs->is_I);CHKERRQ(ierr);
    is_I_layer = sub_schurs->is_I;

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
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    max_subset_size = PetscMax(subset_size,max_subset_size);
    local_size += subset_size;
  }

  /* Work arrays for local indices */
  extra = 0;
  if (sub_schurs->use_mumps && is_I_layer) {
    ierr = ISGetLocalSize(is_I_layer,&extra);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(local_size+extra,&all_local_idx_N);CHKERRQ(ierr);
  if (extra) {
    const PetscInt *idxs;
    ierr = ISGetIndices(is_I_layer,&idxs);CHKERRQ(ierr);
    ierr = PetscMemcpy(all_local_idx_N,idxs,extra*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(is_I_layer,&idxs);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(local_size,&nnz);CHKERRQ(ierr);
  ierr = PetscMalloc1(sub_schurs->n_subs,&auxnum1);CHKERRQ(ierr);
  ierr = PetscMalloc1(sub_schurs->n_subs,&auxnum2);CHKERRQ(ierr);

  /* Get local indices in local numbering */
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscInt j;
    const    PetscInt *idxs;

    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
    /* start (smallest in global ordering) and multiplicity */
    auxnum1[i] = idxs[0];
    auxnum2[i] = subset_size;
    /* subset indices in local numbering */
    ierr = PetscMemcpy(all_local_idx_N+local_size+extra,idxs,subset_size*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
    for (j=0;j<subset_size;j++) nnz[local_size+j] = subset_size;
    local_size += subset_size;
  }

  /* allocate extra workspace needed only for GETRI */
  Bwork = NULL;
  pivots = NULL;
  if (local_size && (!sub_schurs->is_hermitian || !sub_schurs->is_posdef)) {
    PetscScalar lwork;

    B_lwork = -1;
    ierr = PetscBLASIntCast(local_size,&B_N);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,Bwork,&B_N,pivots,&lwork,&B_lwork,&B_ierr));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork);CHKERRQ(ierr);
    ierr = PetscMalloc2(B_lwork,&Bwork,B_N,&pivots);CHKERRQ(ierr);
  }

  /* prepare parallel matrices for summing up properly schurs on subsets */
  ierr = ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum1,PETSC_OWN_POINTER,&all_subsets_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(sub_schurs->l2gmap,all_subsets_n,&all_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum2,PETSC_OWN_POINTER,&all_subsets_mult);CHKERRQ(ierr);
  ierr = PCBDDCSubsetNumbering(all_subsets,all_subsets_mult,&global_size,&all_subsets_n);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_mult);CHKERRQ(ierr);
  ierr = ISGetLocalSize(all_subsets_n,&i);CHKERRQ(ierr);
  if (i != local_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",i,local_size);
  ierr = ISLocalToGlobalMappingCreateIS(all_subsets_n,&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatCreateIS(comm_n,1,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size,l2gmap_subsets,NULL,&work_mat);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingDestroy(&l2gmap_subsets);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)work_mat),&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatSetSizes(global_schur_subsets,PETSC_DECIDE,PETSC_DECIDE,global_size,global_size);CHKERRQ(ierr);
  ierr = MatSetType(global_schur_subsets,MATMPIAIJ);CHKERRQ(ierr);

  /* subset indices in local boundary numbering */
  if (!sub_schurs->is_Ej_all) {
    PetscInt *all_local_idx_B;

    ierr = PetscMalloc1(local_size,&all_local_idx_B);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B);CHKERRQ(ierr);
    if (subset_size != local_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %D != %D\n",subset_size,local_size);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all);CHKERRQ(ierr);
  }

  /* Local matrix of all local Schur on subsets (transposed) */
  if (!sub_schurs->S_Ej_all) {
    ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->S_Ej_all);CHKERRQ(ierr);
    ierr = MatSetSizes(sub_schurs->S_Ej_all,PETSC_DECIDE,PETSC_DECIDE,local_size,local_size);CHKERRQ(ierr);
    ierr = MatSetType(sub_schurs->S_Ej_all,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(sub_schurs->S_Ej_all,0,nnz);CHKERRQ(ierr);
  }

  /* Compute Schur complements explicitly */
  F = NULL;
  if (!sub_schurs->use_mumps) {
    Mat         S_Ej_expl;
    PetscScalar *work;
    PetscInt    j,*dummy_idx;
    PetscBool   Sdense;

    ierr = PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&work);CHKERRQ(ierr);
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      IS  is_subset_B;
      Mat AE_EE,AE_IE,AE_EI,S_Ej;

      /* subsets in original and boundary numbering */
      ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[i],&is_subset_B);CHKERRQ(ierr);
      /* EE block */
      ierr = MatGetSubMatrix(A_BB,is_subset_B,is_subset_B,MAT_INITIAL_MATRIX,&AE_EE);CHKERRQ(ierr);
      /* IE block */
      ierr = MatGetSubMatrix(A_IB,is_I,is_subset_B,MAT_INITIAL_MATRIX,&AE_IE);CHKERRQ(ierr);
      /* EI block */
      if (sub_schurs->is_hermitian) {
        ierr = MatCreateTranspose(AE_IE,&AE_EI);CHKERRQ(ierr);
      } else {
        ierr = MatGetSubMatrix(A_BI,is_subset_B,is_I,MAT_INITIAL_MATRIX,&AE_EI);CHKERRQ(ierr);
      }
      ierr = ISDestroy(&is_subset_B);CHKERRQ(ierr);
      ierr = MatCreateSchurComplement(AE_II,AE_II,AE_IE,AE_EI,AE_EE,&S_Ej);CHKERRQ(ierr);
      ierr = MatDestroy(&AE_EE);CHKERRQ(ierr);
      ierr = MatDestroy(&AE_IE);CHKERRQ(ierr);
      ierr = MatDestroy(&AE_EI);CHKERRQ(ierr);
      if (AE_II == A_II) { /* we can reuse the same ksp */
        KSP ksp;
        ierr = MatSchurComplementGetKSP(sub_schurs->S,&ksp);CHKERRQ(ierr);
        ierr = MatSchurComplementSetKSP(S_Ej,ksp);CHKERRQ(ierr);
      } else { /* build new ksp object which inherits ksp and pc types from the original one */
        KSP       origksp,schurksp;
        PC        origpc,schurpc;
        KSPType   ksp_type;
        PetscInt  n_internal;
        PetscBool ispcnone;

        ierr = MatSchurComplementGetKSP(sub_schurs->S,&origksp);CHKERRQ(ierr);
        ierr = MatSchurComplementGetKSP(S_Ej,&schurksp);CHKERRQ(ierr);
        ierr = KSPGetType(origksp,&ksp_type);CHKERRQ(ierr);
        ierr = KSPSetType(schurksp,ksp_type);CHKERRQ(ierr);
        ierr = KSPGetPC(schurksp,&schurpc);CHKERRQ(ierr);
        ierr = KSPGetPC(origksp,&origpc);CHKERRQ(ierr);
        ierr = PetscObjectTypeCompare((PetscObject)origpc,PCNONE,&ispcnone);CHKERRQ(ierr);
        if (!ispcnone) {
          PCType pc_type;
          ierr = PCGetType(origpc,&pc_type);CHKERRQ(ierr);
          ierr = PCSetType(schurpc,pc_type);CHKERRQ(ierr);
        } else {
          ierr = PCSetType(schurpc,PCLU);CHKERRQ(ierr);
        }
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
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&S_Ej_expl);CHKERRQ(ierr);
      ierr = PCBDDCComputeExplicitSchur(S_Ej,sub_schurs->is_hermitian,MAT_REUSE_MATRIX,&S_Ej_expl);CHKERRQ(ierr);
      ierr = PetscObjectTypeCompare((PetscObject)S_Ej_expl,MATSEQDENSE,&Sdense);CHKERRQ(ierr);
      if (Sdense) {
        for (j=0;j<subset_size;j++) {
          dummy_idx[j]=local_size+j;
        }
        ierr = MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES);CHKERRQ(ierr);
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented for sparse matrices");
      ierr = MatDestroy(&S_Ej);CHKERRQ(ierr);
      ierr = MatDestroy(&S_Ej_expl);CHKERRQ(ierr);
      local_size += subset_size;
    }
    ierr = PetscFree2(dummy_idx,work);CHKERRQ(ierr);
    /* free */
    ierr = ISDestroy(&is_I);CHKERRQ(ierr);
    ierr = MatDestroy(&AE_II);CHKERRQ(ierr);
    ierr = PetscFree(all_local_idx_N);CHKERRQ(ierr);
  } else {
    Mat         A;
    IS          is_A_all;
    PetscScalar *work,*S_data;
    PetscInt    n_I,n_I_all,*dummy_idx,size_schur,size_active_schur,cum,cum2;
    PetscBool   mumps_S;

    /* get working mat */
    n_I = 0;
    if (is_I_layer) {
      ierr = ISGetLocalSize(is_I_layer,&n_I);CHKERRQ(ierr);
    }
    if (!sub_schurs->is_dir) {
      ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size+n_I,all_local_idx_N,PETSC_COPY_VALUES,&is_A_all);CHKERRQ(ierr);
      size_schur = local_size;
    } else {
      IS list[2];

      ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size+n_I,all_local_idx_N,PETSC_COPY_VALUES,&list[0]);CHKERRQ(ierr);
      list[1] = sub_schurs->is_dir;
      ierr = ISConcatenate(PETSC_COMM_SELF,2,list,&is_A_all);CHKERRQ(ierr);
      ierr = ISDestroy(&list[0]);CHKERRQ(ierr);
      ierr = ISGetLocalSize(sub_schurs->is_dir,&size_schur);CHKERRQ(ierr);
      size_schur += local_size;
    }
    ierr = PetscFree(all_local_idx_N);CHKERRQ(ierr);
    size_active_schur = local_size; /* size active schurs does not count any dirichlet dof on the interface */
    ierr = MatGetSubMatrix(sub_schurs->A,is_A_all,is_A_all,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatSetOptionsPrefix(A,"sub_schurs_");CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SYMMETRIC,sub_schurs->is_hermitian);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_HERMITIAN,sub_schurs->is_hermitian);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,sub_schurs->is_posdef);CHKERRQ(ierr);

    if (n_I) {
      IS is_schur;

      if (sub_schurs->is_hermitian && sub_schurs->is_posdef) {
        ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
      } else {
        ierr = MatGetFactor(A,MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
      }
      /* subsets ordered last */
      ierr = ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is_schur);CHKERRQ(ierr);
      ierr = MatFactorSetSchurIS(F,is_schur);CHKERRQ(ierr);
      ierr = ISDestroy(&is_schur);CHKERRQ(ierr);

      /* factorization step */
      if (sub_schurs->is_hermitian && sub_schurs->is_posdef) {
        ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        ierr = MatMumpsSetIcntl(F,19,2);CHKERRQ(ierr);
#endif
        ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
      } else {
        ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        ierr = MatMumpsSetIcntl(F,19,3);CHKERRQ(ierr);
#endif
        ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
      }

      /* get explicit Schur Complement computed during numeric factorization */
      ierr = MatFactorGetSchurComplement(F,&S_all);CHKERRQ(ierr);

      /* we can reuse the solvers if we are not using the economic version */
      ierr = ISGetLocalSize(sub_schurs->is_I,&n_I_all);CHKERRQ(ierr);
      reuse_solvers = (PetscBool)(reuse_solvers && (n_I == n_I_all));
      mumps_S = PETSC_TRUE;
    } else { /* we can't use MUMPS when size_schur == size_of_the_problem */
      ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&S_all);CHKERRQ(ierr);
      reuse_solvers = PETSC_FALSE;
      mumps_S = PETSC_FALSE;
    }

    if (reuse_solvers) {
      Mat              A_II;
      Vec              vec1_B;
      PCBDDCReuseMumps msolv_ctx;

      if (sub_schurs->reuse_mumps) {
        ierr = PCBDDCReuseMumpsReset(sub_schurs->reuse_mumps);CHKERRQ(ierr);
      } else {
        ierr = PetscNew(&sub_schurs->reuse_mumps);CHKERRQ(ierr);
      }
      msolv_ctx = sub_schurs->reuse_mumps;
      ierr = MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatGetSize(A_II,&msolv_ctx->n_I,NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)F);CHKERRQ(ierr);
      msolv_ctx->F = F;
      ierr = MatCreateVecs(F,&msolv_ctx->sol,&msolv_ctx->rhs);CHKERRQ(ierr);

      /* interior solver */
      ierr = PCCreate(PETSC_COMM_SELF,&msolv_ctx->interior_solver);CHKERRQ(ierr);
      ierr = PCSetOperators(msolv_ctx->interior_solver,A_II,A_II);CHKERRQ(ierr);
      ierr = PCSetType(msolv_ctx->interior_solver,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetContext(msolv_ctx->interior_solver,msolv_ctx);CHKERRQ(ierr);
      ierr = PCShellSetApply(msolv_ctx->interior_solver,PCBDDCMumpsInteriorSolve);CHKERRQ(ierr);
      ierr = PCShellSetApplyTranspose(msolv_ctx->interior_solver,PCBDDCMumpsInteriorSolveTranspose);CHKERRQ(ierr);

      /* correction solver */
      ierr = PCCreate(PETSC_COMM_SELF,&msolv_ctx->correction_solver);CHKERRQ(ierr);
      ierr = PCSetOperators(msolv_ctx->correction_solver,A,A);CHKERRQ(ierr);
      ierr = PCSetType(msolv_ctx->correction_solver,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetContext(msolv_ctx->correction_solver,msolv_ctx);CHKERRQ(ierr);
      ierr = PCShellSetApply(msolv_ctx->correction_solver,PCBDDCMumpsCorrectionSolve);CHKERRQ(ierr);
      ierr = PCShellSetApplyTranspose(msolv_ctx->correction_solver,PCBDDCMumpsCorrectionSolveTranspose);CHKERRQ(ierr);

      /* scatter and vecs for Schur complement solver */
      ierr = MatCreateVecs(S_all,&msolv_ctx->sol_B,&msolv_ctx->rhs_B);CHKERRQ(ierr);
      ierr = MatCreateVecs(sub_schurs->S,&vec1_B,NULL);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_A_all,&msolv_ctx->is_B);CHKERRQ(ierr);
      ierr = VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,NULL,&msolv_ctx->correction_scatter_B);CHKERRQ(ierr);
      ierr = VecDestroy(&vec1_B);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)is_A_all);CHKERRQ(ierr);
      msolv_ctx->is_R = is_A_all;
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = ISDestroy(&is_A_all);CHKERRQ(ierr);

    /* Work arrays */
    ierr = PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&work);CHKERRQ(ierr);

    /* matrices for adaptive selection */
    if (compute_Stilda) {
      if (!sub_schurs->sum_S_Ej_tilda_all) {
        ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
        ierr = MatSetSizes(sub_schurs->sum_S_Ej_tilda_all,PETSC_DECIDE,PETSC_DECIDE,size_active_schur,size_active_schur);CHKERRQ(ierr);
        ierr = MatSetType(sub_schurs->sum_S_Ej_tilda_all,MATAIJ);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(sub_schurs->sum_S_Ej_tilda_all,0,nnz);CHKERRQ(ierr);
      }
      if (!sub_schurs->sum_S_Ej_inv_all) {
        ierr = MatCreate(PETSC_COMM_SELF,&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
        ierr = MatSetSizes(sub_schurs->sum_S_Ej_inv_all,PETSC_DECIDE,PETSC_DECIDE,size_active_schur,size_active_schur);CHKERRQ(ierr);
        ierr = MatSetType(sub_schurs->sum_S_Ej_inv_all,MATAIJ);CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(sub_schurs->sum_S_Ej_inv_all,0,nnz);CHKERRQ(ierr);
      }
    }

    /* S_Ej_all */
    cum = cum2 = 0;
    ierr = MatDenseGetArray(S_all,&S_data);CHKERRQ(ierr);
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscInt j;

      /* get S_E */
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      if (mumps_S && sub_schurs->is_hermitian) { /* When using MUMPS data I need to expand to upper triangular (column oriented) */
        PetscInt k;
        for (k=0;k<subset_size;k++) {
          for (j=k;j<subset_size;j++) {
            work[k*subset_size+j] = S_data[cum2+k*size_schur+j];
            work[j*subset_size+k] = S_data[cum2+k*size_schur+j];
          }
        }
      } else { /* copy to workspace */
        PetscInt k;
        for (k=0;k<subset_size;k++) {
          for (j=0;j<subset_size;j++) {
            work[k*subset_size+j] = S_data[cum2+k*size_schur+j];
          }
        }
      }
      /* insert S_E values */
      for (j=0;j<subset_size;j++) dummy_idx[j] = cum+j;
      ierr = MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES);CHKERRQ(ierr);

      /* if adaptivity is requested, invert S_E block */
      if (compute_Stilda) {
        ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        if (sub_schurs->is_hermitian && sub_schurs->is_posdef) { /* TODO add sytrf/i for symmetric non hermitian */
          PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,work,&B_N,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,work,&B_N,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
        } else {
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,work,&B_N,pivots,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,work,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        }
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        ierr = MatSetValues(sub_schurs->sum_S_Ej_inv_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES);CHKERRQ(ierr);
      }
      cum += subset_size;
      cum2 += subset_size*(size_schur + 1);
    }
    ierr = MatDenseRestoreArray(S_all,&S_data);CHKERRQ(ierr);

    if (mumps_S) {
      ierr = MatFactorRestoreSchurComplement(F,&S_all);CHKERRQ(ierr);
    }

    if (compute_Stilda && size_active_schur) {
      if (sub_schurs->n_subs == 1 && size_schur == size_active_schur) { /* we already computed the inverse */
        PetscInt j;
        for (j=0;j<size_schur;j++) dummy_idx[j] = j;
        ierr = MatSetValues(sub_schurs->sum_S_Ej_tilda_all,size_schur,dummy_idx,size_schur,dummy_idx,work,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        if (mumps_S) { /* use MatFactor calls to invert S */
          ierr = MatFactorInvertSchurComplement(F);CHKERRQ(ierr);
          ierr = MatFactorGetSchurComplement(F,&S_all);CHKERRQ(ierr);
        } else { /* we need to invert explicitly since we are not using MUMPS for S */
          ierr = MatDenseGetArray(S_all,&S_data);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_schur,&B_N);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          if (sub_schurs->is_hermitian && sub_schurs->is_posdef) { /* TODO add sytrf/i for symmetric non hermitian */
            PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,S_data,&B_N,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,S_data,&B_N,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
          } else {
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,S_data,&B_N,pivots,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,S_data,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          }
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          ierr = MatDenseRestoreArray(S_all,&S_data);CHKERRQ(ierr);
        }
        /* S_Ej_tilda_all */
        cum = cum2 = 0;
        ierr = MatDenseGetArray(S_all,&S_data);CHKERRQ(ierr);
        for (i=0;i<sub_schurs->n_subs;i++) {
          PetscInt j;

          ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
          /* get (St^-1)_E */
          if (sub_schurs->is_hermitian) { /* Here I don't need to expand to upper triangular (column oriented) */
            PetscInt k;
            for (k=0;k<subset_size;k++) {
              for (j=k;j<subset_size;j++) {
                work[k*subset_size+j] = S_data[cum2+k*size_schur+j];
              }
            }
          } else { /* copy to workspace */
            PetscInt k;
            for (k=0;k<subset_size;k++) {
              for (j=0;j<subset_size;j++) {
                work[k*subset_size+j] = S_data[cum2+k*size_schur+j];
              }
            }
          }
          for (j=0;j<subset_size;j++) dummy_idx[j] = cum+j;
          ierr = MatSetValues(sub_schurs->sum_S_Ej_tilda_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES);CHKERRQ(ierr);
          cum += subset_size;
          cum2 += subset_size*(size_schur + 1);
        }
        ierr = MatDenseRestoreArray(S_all,&S_data);CHKERRQ(ierr);
        if (mumps_S) {
          ierr = MatFactorRestoreSchurComplement(F,&S_all);CHKERRQ(ierr);
        }
      }
    }
    ierr = PetscFree2(dummy_idx,work);CHKERRQ(ierr);
  }
  ierr = PetscFree(nnz);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = ISDestroy(&is_I_layer);CHKERRQ(ierr);
  ierr = MatDestroy(&S_all);CHKERRQ(ierr);
  ierr = MatDestroy(&A_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (compute_Stilda) {
    ierr = MatAssemblyBegin(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }

  /* Global matrix of all assembled Schur on subsets */
  ierr = MatISSetLocalMat(work_mat,sub_schurs->S_Ej_all);CHKERRQ(ierr);
  ierr = MatISSetMPIXAIJPreallocation_Private(work_mat,global_schur_subsets,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);

  /* Get local part of (\sum_j S_Ej) */
  if (!sub_schurs->sum_S_Ej_all) {
    ierr = MatGetSubMatrices(global_schur_subsets,1,&all_subsets_n,&all_subsets_n,MAT_INITIAL_MATRIX,&submats);CHKERRQ(ierr);
    sub_schurs->sum_S_Ej_all = submats[0];
  } else {
    ierr = PetscMalloc1(1,&submats);CHKERRQ(ierr);
    submats[0] = sub_schurs->sum_S_Ej_all;
    ierr = MatGetSubMatrices(global_schur_subsets,1,&all_subsets_n,&all_subsets_n,MAT_REUSE_MATRIX,&submats);CHKERRQ(ierr);
  }

  /* Compute explicitly (\sum_j S_Ej)^-1 (faster scaling during PCApply, needs extra work when doing setup) */
  if (faster_deluxe) {
    Mat         tmpmat;
    PetscScalar *array;
    PetscInt    cum;

    ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all,&array);CHKERRQ(ierr);
    cum = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
      ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
      if (sub_schurs->is_hermitian && sub_schurs->is_posdef) {
        PetscInt j,k;

        PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,array+cum,&B_N,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
        PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,array+cum,&B_N,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
        for (j=0;j<B_N;j++) {
          for (k=j+1;k<B_N;k++) {
            array[k*B_N+j+cum] = array[j*B_N+k+cum];
          }
        }
      } else {
        PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,array+cum,&B_N,pivots,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
        PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,array+cum,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
        if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
      }
      ierr = PetscFPTrapPop();CHKERRQ(ierr);
      cum += subset_size*subset_size;
    }
    ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all,&array);CHKERRQ(ierr);
    ierr = MatMatMult(sub_schurs->S_Ej_all,sub_schurs->sum_S_Ej_all,MAT_INITIAL_MATRIX,1.0,&tmpmat);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->S_Ej_all);CHKERRQ(ierr);
    ierr = MatDestroy(&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
    sub_schurs->S_Ej_all = tmpmat;
  }

  /* Get local part of (\sum_j S^-1_Ej) (\sum_j St^-1_Ej) */
  if (compute_Stilda) {
    ierr = MatISSetLocalMat(work_mat,sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
    ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
    submats[0] = sub_schurs->sum_S_Ej_tilda_all;
    ierr = MatGetSubMatrices(global_schur_subsets,1,&all_subsets_n,&all_subsets_n,MAT_REUSE_MATRIX,&submats);CHKERRQ(ierr);
    ierr = MatISSetLocalMat(work_mat,sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
    ierr = MatISGetMPIXAIJ(work_mat,MAT_REUSE_MATRIX,&global_schur_subsets);CHKERRQ(ierr);
    submats[0] = sub_schurs->sum_S_Ej_inv_all;
    ierr = MatGetSubMatrices(global_schur_subsets,1,&all_subsets_n,&all_subsets_n,MAT_REUSE_MATRIX,&submats);CHKERRQ(ierr);
  }

  /* free workspace */
  ierr = PetscFree(submats);CHKERRQ(ierr);
  ierr = PetscFree2(Bwork,pivots);CHKERRQ(ierr);
  ierr = MatDestroy(&global_schur_subsets);CHKERRQ(ierr);
  ierr = MatDestroy(&work_mat);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_n);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm_n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCBDDCSubSchursInit"
PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap)
{
  IS              *faces,*edges,*all_cc,vertices;
  PetscInt        i,n_faces,n_edges,n_all_cc;
  PetscBool       is_sorted;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = ISSorted(is_I,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) SETERRQ(PetscObjectComm((PetscObject)is_I),PETSC_ERR_PLIB,"IS for I dofs should be shorted");
  ierr = ISSorted(is_B,&is_sorted);CHKERRQ(ierr);
  if (!is_sorted) SETERRQ(PetscObjectComm((PetscObject)is_B),PETSC_ERR_PLIB,"IS for B dofs should be shorted");

  /* reset any previous data */
  ierr = PCBDDCSubSchursReset(sub_schurs);CHKERRQ(ierr);

  /* get index sets for faces and edges (already sorted by global ordering) */
  ierr = PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices);CHKERRQ(ierr);
  n_all_cc = n_faces+n_edges;
  ierr = PetscBTCreate(n_all_cc,&sub_schurs->is_edge);CHKERRQ(ierr);
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
  sub_schurs->is_dir = NULL;
  ierr = PCBDDCGraphGetDirichletDofsB(graph,&sub_schurs->is_dir);CHKERRQ(ierr);

  /* Determine if MUMPS can be used */
  sub_schurs->use_mumps = PETSC_FALSE;
#if defined(PETSC_HAVE_MUMPS)
  sub_schurs->use_mumps = PETSC_TRUE;
#endif

  ierr = PetscObjectReference((PetscObject)is_I);CHKERRQ(ierr);
  sub_schurs->is_I = is_I;
  ierr = PetscObjectReference((PetscObject)is_B);CHKERRQ(ierr);
  sub_schurs->is_B = is_B;
  ierr = PetscObjectReference((PetscObject)graph->l2gmap);CHKERRQ(ierr);
  sub_schurs->l2gmap = graph->l2gmap;
  ierr = PetscObjectReference((PetscObject)BtoNmap);CHKERRQ(ierr);
  sub_schurs->BtoNmap = BtoNmap;
  sub_schurs->n_subs = n_all_cc;
  sub_schurs->is_subs = all_cc;
  if (!sub_schurs->use_mumps) { /* sort by local ordering mumps is not present */
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISSort(sub_schurs->is_subs[i]);CHKERRQ(ierr);
    }
  }
  sub_schurs->is_vertices = vertices;
  sub_schurs->S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_inv_all = NULL;
  sub_schurs->sum_S_Ej_tilda_all = NULL;
  sub_schurs->is_Ej_all = NULL;
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
  ierr = ISDestroy(&sub_schurs->is_vertices);CHKERRQ(ierr);
  ierr = ISDestroy(&sub_schurs->is_dir);CHKERRQ(ierr);
  ierr = PetscBTDestroy(&sub_schurs->is_edge);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISDestroy(&sub_schurs->is_subs[i]);CHKERRQ(ierr);
  }
  if (sub_schurs->n_subs) {
    ierr = PetscFree(sub_schurs->is_subs);CHKERRQ(ierr);
  }
  if (sub_schurs->reuse_mumps) {
    ierr = PCBDDCReuseMumpsReset(sub_schurs->reuse_mumps);CHKERRQ(ierr);
  }
  ierr = PetscFree(sub_schurs->reuse_mumps);CHKERRQ(ierr);
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
