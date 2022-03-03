#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscblaslapack.h>

static inline PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat,PetscBool,MatReuse,Mat*);
static PetscErrorCode PCBDDCReuseSolvers_Interior(PC,Vec,Vec);
static PetscErrorCode PCBDDCReuseSolvers_Correction(PC,Vec,Vec);

/* if v2 is not present, correction is done in-place */
PetscErrorCode PCBDDCReuseSolversBenignAdapt(PCBDDCReuseSolvers ctx, Vec v, Vec v2, PetscBool sol, PetscBool full)
{
  PetscScalar    *array;
  PetscScalar    *array2;

  PetscFunctionBegin;
  if (!ctx->benign_n) PetscFunctionReturn(0);
  if (sol && full) {
    PetscInt n_I,size_schur;

    /* get sizes */
    CHKERRQ(MatGetSize(ctx->benign_csAIB,&size_schur,NULL));
    CHKERRQ(VecGetSize(v,&n_I));
    n_I = n_I - size_schur;
    /* get schur sol from array */
    CHKERRQ(VecGetArray(v,&array));
    CHKERRQ(VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I));
    CHKERRQ(VecRestoreArray(v,&array));
    /* apply interior sol correction */
    CHKERRQ(MatMultTranspose(ctx->benign_csAIB,ctx->benign_dummy_schur_vec,ctx->benign_corr_work));
    CHKERRQ(VecResetArray(ctx->benign_dummy_schur_vec));
    CHKERRQ(MatMultAdd(ctx->benign_AIIm1ones,ctx->benign_corr_work,v,v));
  }
  if (v2) {
    PetscInt nl;

    CHKERRQ(VecGetArrayRead(v,(const PetscScalar**)&array));
    CHKERRQ(VecGetLocalSize(v2,&nl));
    CHKERRQ(VecGetArray(v2,&array2));
    CHKERRQ(PetscArraycpy(array2,array,nl));
  } else {
    CHKERRQ(VecGetArray(v,&array));
    array2 = array;
  }
  if (!sol) { /* change rhs */
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;

      CHKERRQ(ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz));
      CHKERRQ(ISGetIndices(ctx->benign_zerodiag_subs[n],&cols));
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      ctx->benign_save_vals[n] = array2[cols[nz-1]];
      array2[cols[nz-1]] = sum;
      CHKERRQ(ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols));
    }
  } else {
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;
      CHKERRQ(ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz));
      CHKERRQ(ISGetIndices(ctx->benign_zerodiag_subs[n],&cols));
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      array2[cols[nz-1]] = ctx->benign_save_vals[n];
      CHKERRQ(ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols));
    }
  }
  if (v2) {
    CHKERRQ(VecRestoreArrayRead(v,(const PetscScalar**)&array));
    CHKERRQ(VecRestoreArray(v2,&array2));
  } else {
    CHKERRQ(VecRestoreArray(v,&array));
  }
  if (!sol && full) {
    Vec      usedv;
    PetscInt n_I,size_schur;

    /* get sizes */
    CHKERRQ(MatGetSize(ctx->benign_csAIB,&size_schur,NULL));
    CHKERRQ(VecGetSize(v,&n_I));
    n_I = n_I - size_schur;
    /* compute schur rhs correction */
    if (v2) {
      usedv = v2;
    } else {
      usedv = v;
    }
    /* apply schur rhs correction */
    CHKERRQ(MatMultTranspose(ctx->benign_AIIm1ones,usedv,ctx->benign_corr_work));
    CHKERRQ(VecGetArrayRead(usedv,(const PetscScalar**)&array));
    CHKERRQ(VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I));
    CHKERRQ(VecRestoreArrayRead(usedv,(const PetscScalar**)&array));
    CHKERRQ(MatMultAdd(ctx->benign_csAIB,ctx->benign_corr_work,ctx->benign_dummy_schur_vec,ctx->benign_dummy_schur_vec));
    CHKERRQ(VecResetArray(ctx->benign_dummy_schur_vec));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Solve_Private(PC pc, Vec rhs, Vec sol, PetscBool transpose, PetscBool full)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          copy = PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&ctx));
  if (full) {
#if defined(PETSC_HAVE_MUMPS)
    CHKERRQ(MatMumpsSetIcntl(ctx->F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    CHKERRQ(MatMkl_PardisoSetCntl(ctx->F,70,0));
#endif
    copy = ctx->has_vertices;
  } else { /* interior solver */
#if defined(PETSC_HAVE_MUMPS)
    CHKERRQ(MatMumpsSetIcntl(ctx->F,26,0));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    CHKERRQ(MatMkl_PardisoSetCntl(ctx->F,70,1));
#endif
    copy = PETSC_TRUE;
  }
  /* copy rhs into factored matrix workspace */
  if (copy) {
    PetscInt    n;
    PetscScalar *array,*array_solver;

    CHKERRQ(VecGetLocalSize(rhs,&n));
    CHKERRQ(VecGetArrayRead(rhs,(const PetscScalar**)&array));
    CHKERRQ(VecGetArray(ctx->rhs,&array_solver));
    CHKERRQ(PetscArraycpy(array_solver,array,n));
    CHKERRQ(VecRestoreArray(ctx->rhs,&array_solver));
    CHKERRQ(VecRestoreArrayRead(rhs,(const PetscScalar**)&array));

    CHKERRQ(PCBDDCReuseSolversBenignAdapt(ctx,ctx->rhs,NULL,PETSC_FALSE,full));
    if (transpose) {
      CHKERRQ(MatSolveTranspose(ctx->F,ctx->rhs,ctx->sol));
    } else {
      CHKERRQ(MatSolve(ctx->F,ctx->rhs,ctx->sol));
    }
    CHKERRQ(PCBDDCReuseSolversBenignAdapt(ctx,ctx->sol,NULL,PETSC_TRUE,full));

    /* get back data to caller worskpace */
    CHKERRQ(VecGetArrayRead(ctx->sol,(const PetscScalar**)&array_solver));
    CHKERRQ(VecGetArray(sol,&array));
    CHKERRQ(PetscArraycpy(array,array_solver,n));
    CHKERRQ(VecRestoreArray(sol,&array));
    CHKERRQ(VecRestoreArrayRead(ctx->sol,(const PetscScalar**)&array_solver));
  } else {
    if (ctx->benign_n) {
      CHKERRQ(PCBDDCReuseSolversBenignAdapt(ctx,rhs,ctx->rhs,PETSC_FALSE,full));
      if (transpose) {
        CHKERRQ(MatSolveTranspose(ctx->F,ctx->rhs,sol));
      } else {
        CHKERRQ(MatSolve(ctx->F,ctx->rhs,sol));
      }
      CHKERRQ(PCBDDCReuseSolversBenignAdapt(ctx,sol,NULL,PETSC_TRUE,full));
    } else {
      if (transpose) {
        CHKERRQ(MatSolveTranspose(ctx->F,rhs,sol));
      } else {
        CHKERRQ(MatSolve(ctx->F,rhs,sol));
      }
    }
  }
  /* restore defaults */
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(MatMumpsSetIcntl(ctx->F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  CHKERRQ(MatMkl_PardisoSetCntl(ctx->F,70,0));
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Correction(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_CorrectionTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Interior(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_InteriorTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_View(PC pc, PetscViewer viewer)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          iascii;

  PetscFunctionBegin;
  CHKERRQ(PCShellGetContext(pc,&ctx));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  }
  CHKERRQ(MatView(ctx->F,viewer));
  if (iascii) {
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolversReset(PCBDDCReuseSolvers reuse)
{
  PetscInt       i;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&reuse->F));
  CHKERRQ(VecDestroy(&reuse->sol));
  CHKERRQ(VecDestroy(&reuse->rhs));
  CHKERRQ(PCDestroy(&reuse->interior_solver));
  CHKERRQ(PCDestroy(&reuse->correction_solver));
  CHKERRQ(ISDestroy(&reuse->is_R));
  CHKERRQ(ISDestroy(&reuse->is_B));
  CHKERRQ(VecScatterDestroy(&reuse->correction_scatter_B));
  CHKERRQ(VecDestroy(&reuse->sol_B));
  CHKERRQ(VecDestroy(&reuse->rhs_B));
  for (i=0;i<reuse->benign_n;i++) {
    CHKERRQ(ISDestroy(&reuse->benign_zerodiag_subs[i]));
  }
  CHKERRQ(PetscFree(reuse->benign_zerodiag_subs));
  CHKERRQ(PetscFree(reuse->benign_save_vals));
  CHKERRQ(MatDestroy(&reuse->benign_csAIB));
  CHKERRQ(MatDestroy(&reuse->benign_AIIm1ones));
  CHKERRQ(VecDestroy(&reuse->benign_corr_work));
  CHKERRQ(VecDestroy(&reuse->benign_dummy_schur_vec));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCComputeExplicitSchur(Mat M, PetscBool issym, MatReuse reuse, Mat *S)
{
  Mat            B, C, D, Bd, Cd, AinvBd;
  KSP            ksp;
  PC             pc;
  PetscBool      isLU, isILU, isCHOL, Bdense, Cdense;
  PetscReal      fill = 2.0;
  PetscInt       n_I;
  PetscMPIInt    size;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)M),&size));
  PetscCheckFalse(size != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for parallel matrices");
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool Sdense;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)*S, MATSEQDENSE, &Sdense));
    PetscCheck(Sdense,PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"S should dense");
  }
  CHKERRQ(MatSchurComplementGetSubMatrices(M, NULL, NULL, &B, &C, &D));
  CHKERRQ(MatSchurComplementGetKSP(M, &ksp));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) pc, PCLU, &isLU));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) pc, PCILU, &isILU));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) pc, PCCHOLESKY, &isCHOL));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) B, MATSEQDENSE, &Bdense));
  CHKERRQ(PetscObjectTypeCompare((PetscObject) C, MATSEQDENSE, &Cdense));
  CHKERRQ(MatGetSize(B,&n_I,NULL));
  if (n_I) {
    if (!Bdense) {
      CHKERRQ(MatConvert(B, MATSEQDENSE, MAT_INITIAL_MATRIX, &Bd));
    } else {
      Bd = B;
    }

    if (isLU || isILU || isCHOL) {
      Mat fact;
      CHKERRQ(KSPSetUp(ksp));
      CHKERRQ(PCFactorGetMatrix(pc, &fact));
      CHKERRQ(MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd));
      CHKERRQ(MatMatSolve(fact, Bd, AinvBd));
    } else {
      PetscBool ex = PETSC_TRUE;

      if (ex) {
        Mat Ainvd;

        CHKERRQ(PCComputeOperator(pc, MATDENSE, &Ainvd));
        CHKERRQ(MatMatMult(Ainvd, Bd, MAT_INITIAL_MATRIX, fill, &AinvBd));
        CHKERRQ(MatDestroy(&Ainvd));
      } else {
        Vec         sol,rhs;
        PetscScalar *arrayrhs,*arraysol;
        PetscInt    i,nrhs,n;

        CHKERRQ(MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd));
        CHKERRQ(MatGetSize(Bd,&n,&nrhs));
        CHKERRQ(MatDenseGetArray(Bd,&arrayrhs));
        CHKERRQ(MatDenseGetArray(AinvBd,&arraysol));
        CHKERRQ(KSPGetSolution(ksp,&sol));
        CHKERRQ(KSPGetRhs(ksp,&rhs));
        for (i=0;i<nrhs;i++) {
          CHKERRQ(VecPlaceArray(rhs,arrayrhs+i*n));
          CHKERRQ(VecPlaceArray(sol,arraysol+i*n));
          CHKERRQ(KSPSolve(ksp,rhs,sol));
          CHKERRQ(VecResetArray(rhs));
          CHKERRQ(VecResetArray(sol));
        }
        CHKERRQ(MatDenseRestoreArray(Bd,&arrayrhs));
        CHKERRQ(MatDenseRestoreArray(AinvBd,&arrayrhs));
      }
    }
    if (!Bdense & !issym) {
      CHKERRQ(MatDestroy(&Bd));
    }

    if (!issym) {
      if (!Cdense) {
        CHKERRQ(MatConvert(C, MATSEQDENSE, MAT_INITIAL_MATRIX, &Cd));
      } else {
        Cd = C;
      }
      CHKERRQ(MatMatMult(Cd, AinvBd, reuse, fill, S));
      if (!Cdense) {
        CHKERRQ(MatDestroy(&Cd));
      }
    } else {
      CHKERRQ(MatTransposeMatMult(Bd, AinvBd, reuse, fill, S));
      if (!Bdense) {
        CHKERRQ(MatDestroy(&Bd));
      }
    }
    CHKERRQ(MatDestroy(&AinvBd));
  }

  if (D) {
    Mat       Dd;
    PetscBool Ddense;

    CHKERRQ(PetscObjectTypeCompare((PetscObject)D,MATSEQDENSE,&Ddense));
    if (!Ddense) {
      CHKERRQ(MatConvert(D, MATSEQDENSE, MAT_INITIAL_MATRIX, &Dd));
    } else {
      Dd = D;
    }
    if (n_I) {
      CHKERRQ(MatAYPX(*S,-1.0,Dd,SAME_NONZERO_PATTERN));
    } else {
      if (reuse == MAT_INITIAL_MATRIX) {
        CHKERRQ(MatDuplicate(Dd,MAT_COPY_VALUES,S));
      } else {
        CHKERRQ(MatCopy(Dd,*S,SAME_NONZERO_PATTERN));
      }
    }
    if (!Ddense) {
      CHKERRQ(MatDestroy(&Dd));
    }
  } else {
    CHKERRQ(MatScale(*S,-1.0));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursSetUp(PCBDDCSubSchurs sub_schurs, Mat Ain, Mat Sin, PetscBool exact_schur, PetscInt xadj[], PetscInt adjncy[], PetscInt nlayers, Vec scaling, PetscBool compute_Stilda, PetscBool reuse_solvers, PetscBool benign_trick, PetscInt benign_n, PetscInt benign_p0_lidx[], IS benign_zerodiag_subs[], Mat change, IS change_primal)
{
  Mat                    F,A_II,A_IB,A_BI,A_BB,AE_II;
  Mat                    S_all;
  Vec                    gstash,lstash;
  VecScatter             sstash;
  IS                     is_I,is_I_layer;
  IS                     all_subsets,all_subsets_mult,all_subsets_n;
  PetscScalar            *stasharray,*Bwork;
  PetscInt               *nnz,*all_local_idx_N;
  PetscInt               *auxnum1,*auxnum2;
  PetscInt               i,subset_size,max_subset_size;
  PetscInt               n_B,extra,local_size,global_size;
  PetscInt               local_stash_size;
  PetscBLASInt           B_N,B_ierr,B_lwork,*pivots;
  MPI_Comm               comm_n;
  PetscBool              deluxe = PETSC_TRUE;
  PetscBool              use_potr = PETSC_FALSE, use_sytr = PETSC_FALSE;
  PetscViewer            matl_dbg_viewer = NULL;
  PetscErrorCode         ierr;
  PetscBool              flg;

  PetscFunctionBegin;
  CHKERRQ(MatDestroy(&sub_schurs->A));
  CHKERRQ(MatDestroy(&sub_schurs->S));
  if (Ain) {
    CHKERRQ(PetscObjectReference((PetscObject)Ain));
    sub_schurs->A = Ain;
  }

  CHKERRQ(PetscObjectReference((PetscObject)Sin));
  sub_schurs->S = Sin;
  if (sub_schurs->schur_explicit) {
    sub_schurs->schur_explicit = (PetscBool)(!!sub_schurs->A);
  }

  /* preliminary checks */
  PetscCheckFalse(!sub_schurs->schur_explicit && compute_Stilda,PetscObjectComm((PetscObject)sub_schurs->l2gmap),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_PARDISO");

  if (benign_trick) sub_schurs->is_posdef = PETSC_FALSE;

  /* debug (MATLAB) */
  if (sub_schurs->debug) {
    PetscMPIInt size,rank;
    PetscInt    nr,*print_schurs_ranks,print_schurs = PETSC_FALSE;

    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&size));
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank));
    nr   = size;
    CHKERRQ(PetscMalloc1(nr,&print_schurs_ranks));
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->prefix,"BDDC sub_schurs options","PC");CHKERRQ(ierr);
    CHKERRQ(PetscOptionsIntArray("-sub_schurs_debug_ranks","Ranks to debug (all if the option is not used)",NULL,print_schurs_ranks,&nr,&flg));
    if (!flg) print_schurs = PETSC_TRUE;
    else {
      print_schurs = PETSC_FALSE;
      for (i=0;i<nr;i++) if (print_schurs_ranks[i] == (PetscInt)rank) { print_schurs = PETSC_TRUE; break; }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    CHKERRQ(PetscFree(print_schurs_ranks));
    if (print_schurs) {
      char filename[256];

      CHKERRQ(PetscSNPrintf(filename,sizeof(filename),"sub_schurs_Schur_r%d.m",PetscGlobalRank));
      CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&matl_dbg_viewer));
      CHKERRQ(PetscViewerPushFormat(matl_dbg_viewer,PETSC_VIEWER_ASCII_MATLAB));
    }
  }

  /* restrict work on active processes */
  if (sub_schurs->restrict_comm) {
    PetscSubcomm subcomm;
    PetscMPIInt  color,rank;

    color = 0;
    if (!sub_schurs->n_subs) color = 1; /* this can happen if we are in a multilevel case or if the subdomain is disconnected */
    CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank));
    CHKERRQ(PetscSubcommCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&subcomm));
    CHKERRQ(PetscSubcommSetNumber(subcomm,2));
    CHKERRQ(PetscSubcommSetTypeGeneral(subcomm,color,rank));
    CHKERRQ(PetscCommDuplicate(PetscSubcommChild(subcomm),&comm_n,NULL));
    CHKERRQ(PetscSubcommDestroy(&subcomm));
    if (!sub_schurs->n_subs) {
      CHKERRQ(PetscCommDestroy(&comm_n));
      PetscFunctionReturn(0);
    }
  } else {
    CHKERRQ(PetscCommDuplicate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&comm_n,NULL));
  }

  /* get Schur complement matrices */
  if (!sub_schurs->schur_explicit) {
    Mat       tA_IB,tA_BI,tA_BB;
    PetscBool isseqsbaij;
    CHKERRQ(MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,&tA_IB,&tA_BI,&tA_BB));
    CHKERRQ(PetscObjectTypeCompare((PetscObject)tA_BB,MATSEQSBAIJ,&isseqsbaij));
    if (isseqsbaij) {
      CHKERRQ(MatConvert(tA_BB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BB));
      CHKERRQ(MatConvert(tA_IB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_IB));
      CHKERRQ(MatConvert(tA_BI,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BI));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)tA_BB));
      A_BB = tA_BB;
      CHKERRQ(PetscObjectReference((PetscObject)tA_IB));
      A_IB = tA_IB;
      CHKERRQ(PetscObjectReference((PetscObject)tA_BI));
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
  CHKERRQ(ISGetLocalSize(sub_schurs->is_I,&i));
  if (nlayers >= 0 && i) { /* Interior problems can be different from the original one */
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    PetscCheck(xadj,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot request layering without adjacency");
    /* get sizes */
    CHKERRQ(ISGetLocalSize(sub_schurs->is_I,&n_I));
    CHKERRQ(ISGetLocalSize(sub_schurs->is_B,&n_B));

    CHKERRQ(PetscMalloc1(n_I+n_B,&local_numbering));
    CHKERRQ(PetscBTCreate(n_I+n_B,&touched));
    CHKERRQ(PetscBTMemzero(n_I+n_B,touched));

    /* all boundary dofs must be skipped when adding layers */
    CHKERRQ(ISGetIndices(sub_schurs->is_B,&idx_B));
    for (j=0;j<n_B;j++) {
      CHKERRQ(PetscBTSet(touched,idx_B[j]));
    }
    CHKERRQ(PetscArraycpy(local_numbering,idx_B,n_B));
    CHKERRQ(ISRestoreIndices(sub_schurs->is_B,&idx_B));

    /* add prescribed number of layers of dofs */
    n_local_dofs = n_B;
    n_prev_added = n_B;
    for (layer=0;layer<nlayers;layer++) {
      PetscInt n_added = 0;
      if (n_local_dofs == n_I+n_B) break;
      PetscCheckFalse(n_local_dofs > n_I+n_B,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error querying layer %D. Out of bound access (%D > %D)",layer,n_local_dofs,n_I+n_B);
      CHKERRQ(PCBDDCAdjGetNextLayer_Private(local_numbering+n_local_dofs,n_prev_added,touched,xadj,adjncy,&n_added));
      n_prev_added = n_added;
      n_local_dofs += n_added;
      if (!n_added) break;
    }
    CHKERRQ(PetscBTDestroy(&touched));

    /* IS for I layer dofs in original numbering */
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->is_I),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&is_I_layer));
    CHKERRQ(PetscFree(local_numbering));
    CHKERRQ(ISSort(is_I_layer));
    /* IS for I layer dofs in I numbering */
    if (!sub_schurs->schur_explicit) {
      ISLocalToGlobalMapping ItoNmap;
      CHKERRQ(ISLocalToGlobalMappingCreateIS(sub_schurs->is_I,&ItoNmap));
      CHKERRQ(ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,is_I_layer,&is_I));
      CHKERRQ(ISLocalToGlobalMappingDestroy(&ItoNmap));

      /* II block */
      CHKERRQ(MatCreateSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II));
    }
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    CHKERRQ(PetscObjectReference((PetscObject)sub_schurs->is_I));
    is_I_layer = sub_schurs->is_I;

    /* IS for I dofs in I numbering (strided 1) */
    if (!sub_schurs->schur_explicit) {
      CHKERRQ(ISGetSize(sub_schurs->is_I,&n_I));
      CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)sub_schurs->is_I),n_I,0,1,&is_I));

      /* II block is the same */
      CHKERRQ(PetscObjectReference((PetscObject)A_II));
      AE_II = A_II;
    }
  }

  /* Get info on subset sizes and sum of all subsets sizes */
  max_subset_size = 0;
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    max_subset_size = PetscMax(subset_size,max_subset_size);
    local_size += subset_size;
  }

  /* Work arrays for local indices */
  extra = 0;
  CHKERRQ(ISGetLocalSize(sub_schurs->is_B,&n_B));
  if (sub_schurs->schur_explicit && is_I_layer) {
    CHKERRQ(ISGetLocalSize(is_I_layer,&extra));
  }
  CHKERRQ(PetscMalloc1(n_B+extra,&all_local_idx_N));
  if (extra) {
    const PetscInt *idxs;
    CHKERRQ(ISGetIndices(is_I_layer,&idxs));
    CHKERRQ(PetscArraycpy(all_local_idx_N,idxs,extra));
    CHKERRQ(ISRestoreIndices(is_I_layer,&idxs));
  }
  CHKERRQ(PetscMalloc1(sub_schurs->n_subs,&auxnum1));
  CHKERRQ(PetscMalloc1(sub_schurs->n_subs,&auxnum2));

  /* Get local indices in local numbering */
  local_size = 0;
  local_stash_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    const PetscInt *idxs;

    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    CHKERRQ(ISGetIndices(sub_schurs->is_subs[i],&idxs));
    /* start (smallest in global ordering) and multiplicity */
    auxnum1[i] = idxs[0];
    auxnum2[i] = subset_size*subset_size;
    /* subset indices in local numbering */
    CHKERRQ(PetscArraycpy(all_local_idx_N+local_size+extra,idxs,subset_size));
    CHKERRQ(ISRestoreIndices(sub_schurs->is_subs[i],&idxs));
    local_size += subset_size;
    local_stash_size += subset_size*subset_size;
  }

  /* allocate extra workspace needed only for GETRI or SYTRF */
  use_potr = use_sytr = PETSC_FALSE;
  if (benign_trick || (sub_schurs->is_hermitian && sub_schurs->is_posdef)) {
    use_potr = PETSC_TRUE;
  } else if (sub_schurs->is_symmetric) {
    use_sytr = PETSC_TRUE;
  }
  if (local_size && !use_potr) {
    PetscScalar  lwork,dummyscalar = 0.;
    PetscBLASInt dummyint = 0;

    B_lwork = -1;
    CHKERRQ(PetscBLASIntCast(local_size,&B_N));
    CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    if (use_sytr) {
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYTRF Lapack routine %d",(int)B_ierr);
    } else {
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
    }
    CHKERRQ(PetscFPTrapPop());
    CHKERRQ(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
    CHKERRQ(PetscMalloc2(B_lwork,&Bwork,B_N,&pivots));
  } else {
    Bwork = NULL;
    pivots = NULL;
  }

  /* prepare data for summing up properly schurs on subsets */
  CHKERRQ(ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum1,PETSC_OWN_POINTER,&all_subsets_n));
  CHKERRQ(ISLocalToGlobalMappingApplyIS(sub_schurs->l2gmap,all_subsets_n,&all_subsets));
  CHKERRQ(ISDestroy(&all_subsets_n));
  CHKERRQ(ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum2,PETSC_OWN_POINTER,&all_subsets_mult));
  CHKERRQ(ISRenumber(all_subsets,all_subsets_mult,&global_size,&all_subsets_n));
  CHKERRQ(ISDestroy(&all_subsets));
  CHKERRQ(ISDestroy(&all_subsets_mult));
  CHKERRQ(ISGetLocalSize(all_subsets_n,&i));
  PetscCheckFalse(i != local_stash_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",i,local_stash_size);
  CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,local_stash_size,NULL,&lstash));
  CHKERRQ(VecCreateMPI(comm_n,PETSC_DECIDE,global_size,&gstash));
  CHKERRQ(VecScatterCreate(lstash,NULL,gstash,all_subsets_n,&sstash));
  CHKERRQ(ISDestroy(&all_subsets_n));

  /* subset indices in local boundary numbering */
  if (!sub_schurs->is_Ej_all) {
    PetscInt *all_local_idx_B;

    CHKERRQ(PetscMalloc1(local_size,&all_local_idx_B));
    CHKERRQ(ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B));
    PetscCheckFalse(subset_size != local_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %D != %D",subset_size,local_size);
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all));
  }

  if (change) {
    ISLocalToGlobalMapping BtoS;
    IS                     change_primal_B;
    IS                     change_primal_all;

    PetscCheck(!sub_schurs->change_primal_sub,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    PetscCheck(!sub_schurs->change,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    CHKERRQ(PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change_primal_sub));
    for (i=0;i<sub_schurs->n_subs;i++) {
      ISLocalToGlobalMapping NtoS;
      CHKERRQ(ISLocalToGlobalMappingCreateIS(sub_schurs->is_subs[i],&NtoS));
      CHKERRQ(ISGlobalToLocalMappingApplyIS(NtoS,IS_GTOLM_DROP,change_primal,&sub_schurs->change_primal_sub[i]));
      CHKERRQ(ISLocalToGlobalMappingDestroy(&NtoS));
    }
    CHKERRQ(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,change_primal,&change_primal_B));
    CHKERRQ(ISLocalToGlobalMappingCreateIS(sub_schurs->is_Ej_all,&BtoS));
    CHKERRQ(ISGlobalToLocalMappingApplyIS(BtoS,IS_GTOLM_DROP,change_primal_B,&change_primal_all));
    CHKERRQ(ISLocalToGlobalMappingDestroy(&BtoS));
    CHKERRQ(ISDestroy(&change_primal_B));
    CHKERRQ(PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change));
    for (i=0;i<sub_schurs->n_subs;i++) {
      Mat change_sub;

      CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      CHKERRQ(KSPCreate(PETSC_COMM_SELF,&sub_schurs->change[i]));
      CHKERRQ(KSPSetType(sub_schurs->change[i],KSPPREONLY));
      if (!sub_schurs->change_with_qr) {
        CHKERRQ(MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_sub));
      } else {
        Mat change_subt;
        CHKERRQ(MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_subt));
        CHKERRQ(MatConvert(change_subt,MATSEQDENSE,MAT_INITIAL_MATRIX,&change_sub));
        CHKERRQ(MatDestroy(&change_subt));
      }
      CHKERRQ(KSPSetOperators(sub_schurs->change[i],change_sub,change_sub));
      CHKERRQ(MatDestroy(&change_sub));
      CHKERRQ(KSPSetOptionsPrefix(sub_schurs->change[i],sub_schurs->prefix));
      CHKERRQ(KSPAppendOptionsPrefix(sub_schurs->change[i],"sub_schurs_change_"));
    }
    CHKERRQ(ISDestroy(&change_primal_all));
  }

  /* Local matrix of all local Schur on subsets (transposed) */
  if (!sub_schurs->S_Ej_all) {
    Mat         T;
    PetscScalar *v;
    PetscInt    *ii,*jj;
    PetscInt    cum,i,j,k;

    /* MatSeqAIJSetPreallocation + MatSetValues is slow for these kind of matrices (may have large blocks)
       Allocate properly a representative matrix and duplicate */
    CHKERRQ(PetscMalloc3(local_size+1,&ii,local_stash_size,&jj,local_stash_size,&v));
    ii[0] = 0;
    cum   = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      for (j=0;j<subset_size;j++) {
        const PetscInt row = cum+j;
        PetscInt col = cum;

        ii[row+1] = ii[row] + subset_size;
        for (k=ii[row];k<ii[row+1];k++) {
          jj[k] = col;
          col++;
        }
      }
      cum += subset_size;
    }
    CHKERRQ(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,local_size,local_size,ii,jj,v,&T));
    CHKERRQ(MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&sub_schurs->S_Ej_all));
    CHKERRQ(MatDestroy(&T));
    CHKERRQ(PetscFree3(ii,jj,v));
  }
  /* matrices for deluxe scaling and adaptive selection */
  if (compute_Stilda) {
    if (!sub_schurs->sum_S_Ej_tilda_all) {
      CHKERRQ(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_tilda_all));
    }
    if (!sub_schurs->sum_S_Ej_inv_all && deluxe) {
      CHKERRQ(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_inv_all));
    }
  }

  /* Compute Schur complements explicitly */
  F = NULL;
  if (!sub_schurs->schur_explicit) {
    /* this code branch is used when MatFactor with Schur complement support is not present or when explicitly requested;
       it is not efficient, unless the economic version of the scaling is used */
    Mat         S_Ej_expl;
    PetscScalar *work;
    PetscInt    j,*dummy_idx;
    PetscBool   Sdense;

    CHKERRQ(PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&work));
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      IS  is_subset_B;
      Mat AE_EE,AE_IE,AE_EI,S_Ej;

      /* subsets in original and boundary numbering */
      CHKERRQ(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[i],&is_subset_B));
      /* EE block */
      CHKERRQ(MatCreateSubMatrix(A_BB,is_subset_B,is_subset_B,MAT_INITIAL_MATRIX,&AE_EE));
      /* IE block */
      CHKERRQ(MatCreateSubMatrix(A_IB,is_I,is_subset_B,MAT_INITIAL_MATRIX,&AE_IE));
      /* EI block */
      if (sub_schurs->is_symmetric) {
        CHKERRQ(MatCreateTranspose(AE_IE,&AE_EI));
      } else if (sub_schurs->is_hermitian) {
        CHKERRQ(MatCreateHermitianTranspose(AE_IE,&AE_EI));
      } else {
        CHKERRQ(MatCreateSubMatrix(A_BI,is_subset_B,is_I,MAT_INITIAL_MATRIX,&AE_EI));
      }
      CHKERRQ(ISDestroy(&is_subset_B));
      CHKERRQ(MatCreateSchurComplement(AE_II,AE_II,AE_IE,AE_EI,AE_EE,&S_Ej));
      CHKERRQ(MatDestroy(&AE_EE));
      CHKERRQ(MatDestroy(&AE_IE));
      CHKERRQ(MatDestroy(&AE_EI));
      if (AE_II == A_II) { /* we can reuse the same ksp */
        KSP ksp;
        CHKERRQ(MatSchurComplementGetKSP(sub_schurs->S,&ksp));
        CHKERRQ(MatSchurComplementSetKSP(S_Ej,ksp));
      } else { /* build new ksp object which inherits ksp and pc types from the original one */
        KSP       origksp,schurksp;
        PC        origpc,schurpc;
        KSPType   ksp_type;
        PetscInt  n_internal;
        PetscBool ispcnone;

        CHKERRQ(MatSchurComplementGetKSP(sub_schurs->S,&origksp));
        CHKERRQ(MatSchurComplementGetKSP(S_Ej,&schurksp));
        CHKERRQ(KSPGetType(origksp,&ksp_type));
        CHKERRQ(KSPSetType(schurksp,ksp_type));
        CHKERRQ(KSPGetPC(schurksp,&schurpc));
        CHKERRQ(KSPGetPC(origksp,&origpc));
        CHKERRQ(PetscObjectTypeCompare((PetscObject)origpc,PCNONE,&ispcnone));
        if (!ispcnone) {
          PCType pc_type;
          CHKERRQ(PCGetType(origpc,&pc_type));
          CHKERRQ(PCSetType(schurpc,pc_type));
        } else {
          CHKERRQ(PCSetType(schurpc,PCLU));
        }
        CHKERRQ(ISGetSize(is_I,&n_internal));
        if (!n_internal) { /* UMFPACK gives error with 0 sized problems */
          MatSolverType solver = NULL;
          CHKERRQ(PCFactorGetMatSolverType(origpc,(MatSolverType*)&solver));
          if (solver) {
            CHKERRQ(PCFactorSetMatSolverType(schurpc,solver));
          }
        }
        CHKERRQ(KSPSetUp(schurksp));
      }
      CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&S_Ej_expl));
      CHKERRQ(PCBDDCComputeExplicitSchur(S_Ej,sub_schurs->is_symmetric,MAT_REUSE_MATRIX,&S_Ej_expl));
      CHKERRQ(PetscObjectTypeCompare((PetscObject)S_Ej_expl,MATSEQDENSE,&Sdense));
      if (Sdense) {
        for (j=0;j<subset_size;j++) {
          dummy_idx[j]=local_size+j;
        }
        CHKERRQ(MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES));
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented for sparse matrices");
      CHKERRQ(MatDestroy(&S_Ej));
      CHKERRQ(MatDestroy(&S_Ej_expl));
      local_size += subset_size;
    }
    CHKERRQ(PetscFree2(dummy_idx,work));
    /* free */
    CHKERRQ(ISDestroy(&is_I));
    CHKERRQ(MatDestroy(&AE_II));
    CHKERRQ(PetscFree(all_local_idx_N));
  } else {
    Mat               A,cs_AIB_mat = NULL,benign_AIIm1_ones_mat = NULL;
    Vec               Dall = NULL;
    IS                is_A_all,*is_p_r = NULL;
    MatType           Stype;
    PetscScalar       *work,*S_data,*schur_factor,infty = PETSC_MAX_REAL;
    PetscScalar       *SEj_arr = NULL,*SEjinv_arr = NULL;
    const PetscScalar *rS_data;
    PetscInt          n,n_I,size_schur,size_active_schur,cum,cum2;
    PetscBool         economic,solver_S,S_lower_triangular = PETSC_FALSE;
    PetscBool         schur_has_vertices,factor_workaround;
    PetscBool         use_cholesky;
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscBool         oldpin;
#endif

    /* get sizes */
    n_I = 0;
    if (is_I_layer) {
      CHKERRQ(ISGetLocalSize(is_I_layer,&n_I));
    }
    economic = PETSC_FALSE;
    CHKERRQ(ISGetLocalSize(sub_schurs->is_I,&cum));
    if (cum != n_I) economic = PETSC_TRUE;
    CHKERRQ(MatGetLocalSize(sub_schurs->A,&n,NULL));
    size_active_schur = local_size;

    /* import scaling vector (wrong formulation if we have 3D edges) */
    if (scaling && compute_Stilda) {
      const PetscScalar *array;
      PetscScalar       *array2;
      const PetscInt    *idxs;
      PetscInt          i;

      CHKERRQ(ISGetIndices(sub_schurs->is_Ej_all,&idxs));
      CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,size_active_schur,&Dall));
      CHKERRQ(VecGetArrayRead(scaling,&array));
      CHKERRQ(VecGetArray(Dall,&array2));
      for (i=0;i<size_active_schur;i++) array2[i] = array[idxs[i]];
      CHKERRQ(VecRestoreArray(Dall,&array2));
      CHKERRQ(VecRestoreArrayRead(scaling,&array));
      CHKERRQ(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs));
      deluxe = PETSC_FALSE;
    }

    /* size active schurs does not count any dirichlet or vertex dof on the interface */
    factor_workaround = PETSC_FALSE;
    schur_has_vertices = PETSC_FALSE;
    cum = n_I+size_active_schur;
    if (sub_schurs->is_dir) {
      const PetscInt* idxs;
      PetscInt        n_dir;

      CHKERRQ(ISGetLocalSize(sub_schurs->is_dir,&n_dir));
      CHKERRQ(ISGetIndices(sub_schurs->is_dir,&idxs));
      CHKERRQ(PetscArraycpy(all_local_idx_N+cum,idxs,n_dir));
      CHKERRQ(ISRestoreIndices(sub_schurs->is_dir,&idxs));
      cum += n_dir;
      factor_workaround = PETSC_TRUE;
    }
    /* include the primal vertices in the Schur complement */
    if (exact_schur && sub_schurs->is_vertices && (compute_Stilda || benign_n)) {
      PetscInt n_v;

      CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&n_v));
      if (n_v) {
        const PetscInt* idxs;

        CHKERRQ(ISGetIndices(sub_schurs->is_vertices,&idxs));
        CHKERRQ(PetscArraycpy(all_local_idx_N+cum,idxs,n_v));
        CHKERRQ(ISRestoreIndices(sub_schurs->is_vertices,&idxs));
        cum += n_v;
        factor_workaround = PETSC_TRUE;
        schur_has_vertices = PETSC_TRUE;
      }
    }
    size_schur = cum - n_I;
    CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,cum,all_local_idx_N,PETSC_OWN_POINTER,&is_A_all));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = sub_schurs->A->boundtocpu;
    CHKERRQ(MatBindToCPU(sub_schurs->A,PETSC_TRUE));
#endif
    if (cum == n) {
      CHKERRQ(ISSetPermutation(is_A_all));
      CHKERRQ(MatPermute(sub_schurs->A,is_A_all,is_A_all,&A));
    } else {
      CHKERRQ(MatCreateSubMatrix(sub_schurs->A,is_A_all,is_A_all,MAT_INITIAL_MATRIX,&A));
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    CHKERRQ(MatBindToCPU(sub_schurs->A,oldpin));
#endif
    CHKERRQ(MatSetOptionsPrefix(A,sub_schurs->prefix));
    CHKERRQ(MatAppendOptionsPrefix(A,"sub_schurs_"));

    /* if we actually change the basis for the pressures, LDL^T factors will use a lot of memory
       this is a workaround */
    if (benign_n) {
      Vec                    v,benign_AIIm1_ones;
      ISLocalToGlobalMapping N_to_reor;
      IS                     is_p0,is_p0_p;
      PetscScalar            *cs_AIB,*AIIm1_data;
      PetscInt               sizeA;

      CHKERRQ(ISLocalToGlobalMappingCreateIS(is_A_all,&N_to_reor));
      CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,benign_n,benign_p0_lidx,PETSC_COPY_VALUES,&is_p0));
      CHKERRQ(ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,is_p0,&is_p0_p));
      CHKERRQ(ISDestroy(&is_p0));
      CHKERRQ(MatCreateVecs(A,&v,&benign_AIIm1_ones));
      CHKERRQ(VecGetSize(v,&sizeA));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,sizeA,benign_n,NULL,&benign_AIIm1_ones_mat));
      CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,size_schur,benign_n,NULL,&cs_AIB_mat));
      CHKERRQ(MatDenseGetArray(cs_AIB_mat,&cs_AIB));
      CHKERRQ(MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data));
      CHKERRQ(PetscMalloc1(benign_n,&is_p_r));
      /* compute colsum of A_IB restricted to pressures */
      for (i=0;i<benign_n;i++) {
        const PetscScalar *array;
        const PetscInt    *idxs;
        PetscInt          j,nz;

        CHKERRQ(ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,benign_zerodiag_subs[i],&is_p_r[i]));
        CHKERRQ(ISGetLocalSize(is_p_r[i],&nz));
        CHKERRQ(ISGetIndices(is_p_r[i],&idxs));
        for (j=0;j<nz;j++) AIIm1_data[idxs[j]+sizeA*i] = 1.;
        CHKERRQ(ISRestoreIndices(is_p_r[i],&idxs));
        CHKERRQ(VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i));
        CHKERRQ(MatMult(A,benign_AIIm1_ones,v));
        CHKERRQ(VecResetArray(benign_AIIm1_ones));
        CHKERRQ(VecGetArrayRead(v,&array));
        for (j=0;j<size_schur;j++) {
#if defined(PETSC_USE_COMPLEX)
          cs_AIB[i*size_schur + j] = (PetscRealPart(array[j+n_I])/nz + PETSC_i*(PetscImaginaryPart(array[j+n_I])/nz));
#else
          cs_AIB[i*size_schur + j] = array[j+n_I]/nz;
#endif
        }
        CHKERRQ(VecRestoreArrayRead(v,&array));
      }
      CHKERRQ(MatDenseRestoreArray(cs_AIB_mat,&cs_AIB));
      CHKERRQ(MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data));
      CHKERRQ(VecDestroy(&v));
      CHKERRQ(VecDestroy(&benign_AIIm1_ones));
      CHKERRQ(MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
      CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      CHKERRQ(MatZeroRowsColumnsIS(A,is_p0_p,1.0,NULL,NULL));
      CHKERRQ(ISDestroy(&is_p0_p));
      CHKERRQ(ISLocalToGlobalMappingDestroy(&N_to_reor));
    }
    CHKERRQ(MatSetOption(A,MAT_SYMMETRIC,sub_schurs->is_symmetric));
    CHKERRQ(MatSetOption(A,MAT_HERMITIAN,sub_schurs->is_hermitian));
    CHKERRQ(MatSetOption(A,MAT_SPD,sub_schurs->is_posdef));

    /* for complexes, symmetric and hermitian at the same time implies null imaginary part */
    use_cholesky = (PetscBool)((use_potr || use_sytr) && sub_schurs->is_hermitian && sub_schurs->is_symmetric);

    /* when using the benign subspace trick, the local Schur complements are SPD */
    /* MKL_PARDISO does not handle well the computation of a Schur complement from a symmetric indefinite factorization
       Use LU and adapt pivoting perturbation (still, solution is not as accurate as with using MUMPS) */
    if (benign_trick) {
      sub_schurs->is_posdef = PETSC_TRUE;
      CHKERRQ(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,&flg));
      if (flg) use_cholesky = PETSC_FALSE;
    }

    if (n_I) {
      IS        is_schur;
      char      stype[64];
      PetscBool gpu = PETSC_FALSE;

      if (use_cholesky) {
        CHKERRQ(MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_CHOLESKY,&F));
      } else {
        CHKERRQ(MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_LU,&F));
      }
      CHKERRQ(MatSetErrorIfFailure(A,PETSC_TRUE));
#if defined(PETSC_HAVE_MKL_PARDISO)
      if (benign_trick) CHKERRQ(MatMkl_PardisoSetCntl(F,10,10));
#endif
      /* subsets ordered last */
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is_schur));
      CHKERRQ(MatFactorSetSchurIS(F,is_schur));
      CHKERRQ(ISDestroy(&is_schur));

      /* factorization step */
      if (use_cholesky) {
        CHKERRQ(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        CHKERRQ(MatMumpsSetIcntl(F,19,2));
#endif
        CHKERRQ(MatCholeskyFactorNumeric(F,A,NULL));
        S_lower_triangular = PETSC_TRUE;
      } else {
        CHKERRQ(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        CHKERRQ(MatMumpsSetIcntl(F,19,3));
#endif
        CHKERRQ(MatLUFactorNumeric(F,A,NULL));
      }
      CHKERRQ(MatViewFromOptions(F,(PetscObject)A,"-mat_factor_view"));

      if (matl_dbg_viewer) {
        Mat S;
        IS  is;

        CHKERRQ(PetscObjectSetName((PetscObject)A,"A"));
        CHKERRQ(MatView(A,matl_dbg_viewer));
        CHKERRQ(MatFactorCreateSchurComplement(F,&S,NULL));
        CHKERRQ(PetscObjectSetName((PetscObject)S,"S"));
        CHKERRQ(MatView(S,matl_dbg_viewer));
        CHKERRQ(MatDestroy(&S));
        CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n_I,0,1,&is));
        CHKERRQ(PetscObjectSetName((PetscObject)is,"I"));
        CHKERRQ(ISView(is,matl_dbg_viewer));
        CHKERRQ(ISDestroy(&is));
        CHKERRQ(ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is));
        CHKERRQ(PetscObjectSetName((PetscObject)is,"B"));
        CHKERRQ(ISView(is,matl_dbg_viewer));
        CHKERRQ(ISDestroy(&is));
        CHKERRQ(PetscObjectSetName((PetscObject)is_A_all,"IA"));
        CHKERRQ(ISView(is_A_all,matl_dbg_viewer));
      }

      /* get explicit Schur Complement computed during numeric factorization */
      CHKERRQ(MatFactorGetSchurComplement(F,&S_all,NULL));
      CHKERRQ(PetscStrncpy(stype,MATSEQDENSE,sizeof(stype)));
#if defined(PETSC_HAVE_CUDA)
      CHKERRQ(PetscObjectTypeCompareAny((PetscObject)A,&gpu,MATSEQAIJVIENNACL,MATSEQAIJCUSPARSE,""));
#endif
      if (gpu) {
        CHKERRQ(PetscStrncpy(stype,MATSEQDENSECUDA,sizeof(stype)));
      }
      CHKERRQ(PetscOptionsGetString(NULL,sub_schurs->prefix,"-sub_schurs_schur_mat_type",stype,sizeof(stype),NULL));
      CHKERRQ(MatConvert(S_all,stype,MAT_INPLACE_MATRIX,&S_all));
      CHKERRQ(MatSetOption(S_all,MAT_SPD,sub_schurs->is_posdef));
      CHKERRQ(MatSetOption(S_all,MAT_HERMITIAN,sub_schurs->is_hermitian));
      CHKERRQ(MatGetType(S_all,&Stype));

      /* we can reuse the solvers if we are not using the economic version */
      reuse_solvers = (PetscBool)(reuse_solvers && !economic);
      factor_workaround = (PetscBool)(reuse_solvers && factor_workaround);
      if (!sub_schurs->is_posdef && factor_workaround && compute_Stilda && size_active_schur)
        reuse_solvers = factor_workaround = PETSC_FALSE;

      solver_S = PETSC_TRUE;

      /* update the Schur complement with the change of basis on the pressures */
      if (benign_n) {
        const PetscScalar *cs_AIB;
        PetscScalar       *S_data,*AIIm1_data;
        Mat               S2 = NULL,S3 = NULL; /* dbg */
        PetscScalar       *S2_data,*S3_data; /* dbg */
        Vec               v,benign_AIIm1_ones;
        PetscInt          sizeA;

        CHKERRQ(MatDenseGetArray(S_all,&S_data));
        CHKERRQ(MatCreateVecs(A,&v,&benign_AIIm1_ones));
        CHKERRQ(VecGetSize(v,&sizeA));
#if defined(PETSC_HAVE_MUMPS)
        CHKERRQ(MatMumpsSetIcntl(F,26,0));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        CHKERRQ(MatMkl_PardisoSetCntl(F,70,1));
#endif
        CHKERRQ(MatDenseGetArrayRead(cs_AIB_mat,&cs_AIB));
        CHKERRQ(MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data));
        if (matl_dbg_viewer) {
          CHKERRQ(MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S2));
          CHKERRQ(MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S3));
          CHKERRQ(MatDenseGetArray(S2,&S2_data));
          CHKERRQ(MatDenseGetArray(S3,&S3_data));
        }
        for (i=0;i<benign_n;i++) {
          PetscScalar    *array,sum = 0.,one = 1.,*sums;
          const PetscInt *idxs;
          PetscInt       k,j,nz;
          PetscBLASInt   B_k,B_n;

          CHKERRQ(PetscCalloc1(benign_n,&sums));
          CHKERRQ(VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i));
          CHKERRQ(VecCopy(benign_AIIm1_ones,v));
          CHKERRQ(MatSolve(F,v,benign_AIIm1_ones));
          CHKERRQ(MatMult(A,benign_AIIm1_ones,v));
          CHKERRQ(VecResetArray(benign_AIIm1_ones));
          /* p0 dofs (eliminated) are excluded from the sums */
          for (k=0;k<benign_n;k++) {
            CHKERRQ(ISGetLocalSize(is_p_r[k],&nz));
            CHKERRQ(ISGetIndices(is_p_r[k],&idxs));
            for (j=0;j<nz-1;j++) sums[k] -= AIIm1_data[idxs[j]+sizeA*i];
            CHKERRQ(ISRestoreIndices(is_p_r[k],&idxs));
          }
          CHKERRQ(VecGetArrayRead(v,(const PetscScalar**)&array));
          if (matl_dbg_viewer) {
            Vec  vv;
            char name[16];

            CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array+n_I,&vv));
            CHKERRQ(PetscSNPrintf(name,sizeof(name),"Pvs%D",i));
            CHKERRQ(PetscObjectSetName((PetscObject)vv,name));
            CHKERRQ(VecView(vv,matl_dbg_viewer));
          }
          /* perform sparse rank updates on symmetric Schur (TODO: move outside of the loop?) */
          /* cs_AIB already scaled by 1./nz */
          B_k = 1;
          for (k=0;k<benign_n;k++) {
            sum  = sums[k];
            CHKERRQ(PetscBLASIntCast(size_schur,&B_n));

            if (PetscAbsScalar(sum) == 0.0) continue;
            if (k == i) {
              PetscStackCallBLAS("BLASsyrk",BLASsyrk_("L","N",&B_n,&B_k,&sum,cs_AIB+i*size_schur,&B_n,&one,S_data,&B_n));
              if (matl_dbg_viewer) {
                PetscStackCallBLAS("BLASsyrk",BLASsyrk_("L","N",&B_n,&B_k,&sum,cs_AIB+i*size_schur,&B_n,&one,S3_data,&B_n));
              }
            } else { /* XXX Is it correct to use symmetric rank-2 update with half of the sum? */
              sum /= 2.0;
              PetscStackCallBLAS("BLASsyr2k",BLASsyr2k_("L","N",&B_n,&B_k,&sum,cs_AIB+k*size_schur,&B_n,cs_AIB+i*size_schur,&B_n,&one,S_data,&B_n));
              if (matl_dbg_viewer) {
                PetscStackCallBLAS("BLASsyr2k",BLASsyr2k_("L","N",&B_n,&B_k,&sum,cs_AIB+k*size_schur,&B_n,cs_AIB+i*size_schur,&B_n,&one,S3_data,&B_n));
              }
            }
          }
          sum  = 1.;
          PetscStackCallBLAS("BLASsyr2k",BLASsyr2k_("L","N",&B_n,&B_k,&sum,array+n_I,&B_n,cs_AIB+i*size_schur,&B_n,&one,S_data,&B_n));
          if (matl_dbg_viewer) {
            PetscStackCallBLAS("BLASsyr2k",BLASsyr2k_("L","N",&B_n,&B_k,&sum,array+n_I,&B_n,cs_AIB+i*size_schur,&B_n,&one,S2_data,&B_n));
          }
          CHKERRQ(VecRestoreArrayRead(v,(const PetscScalar**)&array));
          /* set p0 entry of AIIm1_ones to zero */
          CHKERRQ(ISGetLocalSize(is_p_r[i],&nz));
          CHKERRQ(ISGetIndices(is_p_r[i],&idxs));
          for (j=0;j<benign_n;j++) AIIm1_data[idxs[nz-1]+sizeA*j] = 0.;
          CHKERRQ(ISRestoreIndices(is_p_r[i],&idxs));
          CHKERRQ(PetscFree(sums));
        }
        CHKERRQ(VecDestroy(&benign_AIIm1_ones));
        if (matl_dbg_viewer) {
          CHKERRQ(MatDenseRestoreArray(S2,&S2_data));
          CHKERRQ(MatDenseRestoreArray(S3,&S3_data));
        }
        if (!S_lower_triangular) { /* I need to expand the upper triangular data (column oriented) */
          PetscInt k,j;
          for (k=0;k<size_schur;k++) {
            for (j=k;j<size_schur;j++) {
              S_data[j*size_schur+k] = PetscConj(S_data[k*size_schur+j]);
            }
          }
        }

        /* restore defaults */
#if defined(PETSC_HAVE_MUMPS)
        CHKERRQ(MatMumpsSetIcntl(F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        CHKERRQ(MatMkl_PardisoSetCntl(F,70,0));
#endif
        CHKERRQ(MatDenseRestoreArrayRead(cs_AIB_mat,&cs_AIB));
        CHKERRQ(MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data));
        CHKERRQ(VecDestroy(&v));
        CHKERRQ(MatDenseRestoreArray(S_all,&S_data));
        if (matl_dbg_viewer) {
          Mat S;

          CHKERRQ(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
          CHKERRQ(MatFactorCreateSchurComplement(F,&S,NULL));
          CHKERRQ(PetscObjectSetName((PetscObject)S,"Sb"));
          CHKERRQ(MatView(S,matl_dbg_viewer));
          CHKERRQ(MatDestroy(&S));
          CHKERRQ(PetscObjectSetName((PetscObject)S2,"S2P"));
          CHKERRQ(MatView(S2,matl_dbg_viewer));
          CHKERRQ(PetscObjectSetName((PetscObject)S3,"S3P"));
          CHKERRQ(MatView(S3,matl_dbg_viewer));
          CHKERRQ(PetscObjectSetName((PetscObject)cs_AIB_mat,"cs"));
          CHKERRQ(MatView(cs_AIB_mat,matl_dbg_viewer));
          CHKERRQ(MatFactorGetSchurComplement(F,&S_all,NULL));
        }
        CHKERRQ(MatDestroy(&S2));
        CHKERRQ(MatDestroy(&S3));
      }
      if (!reuse_solvers) {
        for (i=0;i<benign_n;i++) {
          CHKERRQ(ISDestroy(&is_p_r[i]));
        }
        CHKERRQ(PetscFree(is_p_r));
        CHKERRQ(MatDestroy(&cs_AIB_mat));
        CHKERRQ(MatDestroy(&benign_AIIm1_ones_mat));
      }
    } else { /* we can't use MatFactor when size_schur == size_of_the_problem */
      CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&S_all));
      CHKERRQ(MatGetType(S_all,&Stype));
      reuse_solvers = PETSC_FALSE; /* TODO: why we can't reuse the solvers here? */
      factor_workaround = PETSC_FALSE;
      solver_S = PETSC_FALSE;
    }

    if (reuse_solvers) {
      Mat                A_II,Afake;
      Vec                vec1_B;
      PCBDDCReuseSolvers msolv_ctx;
      PetscInt           n_R;

      if (sub_schurs->reuse_solver) {
        CHKERRQ(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
      } else {
        CHKERRQ(PetscNew(&sub_schurs->reuse_solver));
      }
      msolv_ctx = sub_schurs->reuse_solver;
      CHKERRQ(MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,NULL,NULL,NULL));
      CHKERRQ(PetscObjectReference((PetscObject)F));
      msolv_ctx->F = F;
      CHKERRQ(MatCreateVecs(F,&msolv_ctx->sol,NULL));
      /* currently PETSc has no support for MatSolve(F,x,x), so cheat and let rhs and sol share the same memory */
      {
        PetscScalar *array;
        PetscInt    n;

        CHKERRQ(VecGetLocalSize(msolv_ctx->sol,&n));
        CHKERRQ(VecGetArray(msolv_ctx->sol,&array));
        CHKERRQ(VecCreateSeqWithArray(PetscObjectComm((PetscObject)msolv_ctx->sol),1,n,array,&msolv_ctx->rhs));
        CHKERRQ(VecRestoreArray(msolv_ctx->sol,&array));
      }
      msolv_ctx->has_vertices = schur_has_vertices;

      /* interior solver */
      CHKERRQ(PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->interior_solver));
      CHKERRQ(PCSetOperators(msolv_ctx->interior_solver,A_II,A_II));
      CHKERRQ(PCSetType(msolv_ctx->interior_solver,PCSHELL));
      CHKERRQ(PCShellSetName(msolv_ctx->interior_solver,"Interior solver (w/o Schur factorization)"));
      CHKERRQ(PCShellSetContext(msolv_ctx->interior_solver,msolv_ctx));
      CHKERRQ(PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View));
      CHKERRQ(PCShellSetApply(msolv_ctx->interior_solver,PCBDDCReuseSolvers_Interior));
      CHKERRQ(PCShellSetApplyTranspose(msolv_ctx->interior_solver,PCBDDCReuseSolvers_InteriorTranspose));

      /* correction solver */
      CHKERRQ(PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->correction_solver));
      CHKERRQ(PCSetType(msolv_ctx->correction_solver,PCSHELL));
      CHKERRQ(PCShellSetName(msolv_ctx->correction_solver,"Correction solver (with Schur factorization)"));
      CHKERRQ(PCShellSetContext(msolv_ctx->correction_solver,msolv_ctx));
      CHKERRQ(PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View));
      CHKERRQ(PCShellSetApply(msolv_ctx->correction_solver,PCBDDCReuseSolvers_Correction));
      CHKERRQ(PCShellSetApplyTranspose(msolv_ctx->correction_solver,PCBDDCReuseSolvers_CorrectionTranspose));

      /* scatter and vecs for Schur complement solver */
      CHKERRQ(MatCreateVecs(S_all,&msolv_ctx->sol_B,&msolv_ctx->rhs_B));
      CHKERRQ(MatCreateVecs(sub_schurs->S,&vec1_B,NULL));
      if (!schur_has_vertices) {
        CHKERRQ(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_A_all,&msolv_ctx->is_B));
        CHKERRQ(VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,NULL,&msolv_ctx->correction_scatter_B));
        CHKERRQ(PetscObjectReference((PetscObject)is_A_all));
        msolv_ctx->is_R = is_A_all;
      } else {
        IS              is_B_all;
        const PetscInt* idxs;
        PetscInt        dual,n_v,n;

        CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&n_v));
        dual = size_schur - n_v;
        CHKERRQ(ISGetLocalSize(is_A_all,&n));
        CHKERRQ(ISGetIndices(is_A_all,&idxs));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),dual,idxs+n_I,PETSC_COPY_VALUES,&is_B_all));
        CHKERRQ(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_B_all,&msolv_ctx->is_B));
        CHKERRQ(ISDestroy(&is_B_all));
        CHKERRQ(ISCreateStride(PetscObjectComm((PetscObject)is_A_all),dual,0,1,&is_B_all));
        CHKERRQ(VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,is_B_all,&msolv_ctx->correction_scatter_B));
        CHKERRQ(ISDestroy(&is_B_all));
        CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),n-n_v,idxs,PETSC_COPY_VALUES,&msolv_ctx->is_R));
        CHKERRQ(ISRestoreIndices(is_A_all,&idxs));
      }
      CHKERRQ(ISGetLocalSize(msolv_ctx->is_R,&n_R));
      CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,n_R,n_R,0,NULL,&Afake));
      CHKERRQ(MatAssemblyBegin(Afake,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(Afake,MAT_FINAL_ASSEMBLY));
      CHKERRQ(PCSetOperators(msolv_ctx->correction_solver,Afake,Afake));
      CHKERRQ(MatDestroy(&Afake));
      CHKERRQ(VecDestroy(&vec1_B));

      /* communicate benign info to solver context */
      if (benign_n) {
        PetscScalar *array;

        msolv_ctx->benign_n = benign_n;
        msolv_ctx->benign_zerodiag_subs = is_p_r;
        CHKERRQ(PetscMalloc1(benign_n,&msolv_ctx->benign_save_vals));
        msolv_ctx->benign_csAIB = cs_AIB_mat;
        CHKERRQ(MatCreateVecs(cs_AIB_mat,&msolv_ctx->benign_corr_work,NULL));
        CHKERRQ(VecGetArray(msolv_ctx->benign_corr_work,&array));
        CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array,&msolv_ctx->benign_dummy_schur_vec));
        CHKERRQ(VecRestoreArray(msolv_ctx->benign_corr_work,&array));
        msolv_ctx->benign_AIIm1ones = benign_AIIm1_ones_mat;
      }
    } else {
      if (sub_schurs->reuse_solver) {
        CHKERRQ(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
      }
      CHKERRQ(PetscFree(sub_schurs->reuse_solver));
    }
    CHKERRQ(MatDestroy(&A));
    CHKERRQ(ISDestroy(&is_A_all));

    /* Work arrays */
    CHKERRQ(PetscMalloc1(max_subset_size*max_subset_size,&work));

    /* S_Ej_all */
    cum = cum2 = 0;
    CHKERRQ(MatDenseGetArrayRead(S_all,&rS_data));
    CHKERRQ(MatSeqAIJGetArray(sub_schurs->S_Ej_all,&SEj_arr));
    if (compute_Stilda) {
      CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr));
    }
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscInt j;

      /* get S_E */
      CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      if (S_lower_triangular) { /* I need to expand the upper triangular data (column oriented) */
        PetscInt k;
        for (k=0;k<subset_size;k++) {
          for (j=k;j<subset_size;j++) {
            work[k*subset_size+j] = rS_data[cum2+k*size_schur+j];
            work[j*subset_size+k] = PetscConj(rS_data[cum2+k*size_schur+j]);
          }
        }
      } else { /* just copy to workspace */
        PetscInt k;
        for (k=0;k<subset_size;k++) {
          for (j=0;j<subset_size;j++) {
            work[k*subset_size+j] = rS_data[cum2+k*size_schur+j];
          }
        }
      }
      /* insert S_E values */
      if (sub_schurs->change) {
        Mat change_sub,SEj,T;

        /* change basis */
        CHKERRQ(KSPGetOperators(sub_schurs->change[i],&change_sub,NULL));
        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
        if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
          Mat T2;
          CHKERRQ(MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2));
          CHKERRQ(MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
          CHKERRQ(MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T));
          CHKERRQ(MatDestroy(&T2));
        } else {
          CHKERRQ(MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
        }
        CHKERRQ(MatCopy(T,SEj,SAME_NONZERO_PATTERN));
        CHKERRQ(MatDestroy(&T));
        CHKERRQ(MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1.0,NULL,NULL));
        CHKERRQ(MatDestroy(&SEj));
      }
      if (deluxe) {
        CHKERRQ(PetscArraycpy(SEj_arr,work,subset_size*subset_size));
        /* if adaptivity is requested, invert S_E blocks */
        if (compute_Stilda) {
          Mat               M;
          const PetscScalar *vals;
          PetscBool         isdense,isdensecuda;

          CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&M));
          CHKERRQ(MatSetOption(M,MAT_SPD,sub_schurs->is_posdef));
          CHKERRQ(MatSetOption(M,MAT_HERMITIAN,sub_schurs->is_hermitian));
          if (!PetscBTLookup(sub_schurs->is_edge,i)) {
            CHKERRQ(MatSetType(M,Stype));
          }
          CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&isdense));
          CHKERRQ(PetscObjectTypeCompare((PetscObject)M,MATSEQDENSECUDA,&isdensecuda));
          if (use_cholesky) {
            CHKERRQ(MatCholeskyFactor(M,NULL,NULL));
          } else {
            CHKERRQ(MatLUFactor(M,NULL,NULL,NULL));
          }
          if (isdense) {
            CHKERRQ(MatSeqDenseInvertFactors_Private(M));
#if defined(PETSC_HAVE_CUDA)
          } else if (isdensecuda) {
            CHKERRQ(MatSeqDenseCUDAInvertFactors_Private(M));
#endif
          } else SETERRQ(PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Not implemented for type %s",Stype);
          CHKERRQ(MatDenseGetArrayRead(M,&vals));
          CHKERRQ(PetscArraycpy(SEjinv_arr,vals,subset_size*subset_size));
          CHKERRQ(MatDenseRestoreArrayRead(M,&vals));
          CHKERRQ(MatDestroy(&M));
        }
      } else if (compute_Stilda) { /* not using deluxe */
        Mat         SEj;
        Vec         D;
        PetscScalar *array;

        CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
        CHKERRQ(VecGetArray(Dall,&array));
        CHKERRQ(VecCreateSeqWithArray(PETSC_COMM_SELF,1,subset_size,array+cum,&D));
        CHKERRQ(VecRestoreArray(Dall,&array));
        CHKERRQ(VecShift(D,-1.));
        CHKERRQ(MatDiagonalScale(SEj,D,D));
        CHKERRQ(MatDestroy(&SEj));
        CHKERRQ(VecDestroy(&D));
        CHKERRQ(PetscArraycpy(SEj_arr,work,subset_size*subset_size));
      }
      cum += subset_size;
      cum2 += subset_size*(size_schur + 1);
      SEj_arr += subset_size*subset_size;
      if (SEjinv_arr) SEjinv_arr += subset_size*subset_size;
    }
    CHKERRQ(MatDenseRestoreArrayRead(S_all,&rS_data));
    CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&SEj_arr));
    if (compute_Stilda) {
      CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr));
    }
    if (solver_S) {
      CHKERRQ(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
    }

    /* may prevent from unneeded copies, since MUMPS or MKL_Pardiso always use CPU memory
       however, preliminary tests indicate using GPUs is still faster in the solve phase */
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    if (reuse_solvers) {
      Mat                  St;
      MatFactorSchurStatus st;

      flg  = PETSC_FALSE;
      CHKERRQ(PetscOptionsGetBool(NULL,sub_schurs->prefix,"-sub_schurs_schur_pin_to_cpu",&flg,NULL));
      CHKERRQ(MatFactorGetSchurComplement(F,&St,&st));
      CHKERRQ(MatBindToCPU(St,flg));
      CHKERRQ(MatFactorRestoreSchurComplement(F,&St,st));
    }
#endif

    schur_factor = NULL;
    if (compute_Stilda && size_active_schur) {

      CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr));
      if (sub_schurs->n_subs == 1 && size_schur == size_active_schur && deluxe) { /* we already computed the inverse */
        CHKERRQ(PetscArraycpy(SEjinv_arr,work,size_schur*size_schur));
      } else {
        Mat S_all_inv=NULL;

        if (solver_S) {
          /* for adaptive selection we need S^-1; for solver reusage we need S_\Delta\Delta^-1.
             The latter is not the principal subminor for S^-1. However, the factors can be reused since S_\Delta\Delta is the leading principal submatrix of S */
          if (factor_workaround) {/* invert without calling MatFactorInvertSchurComplement, since we are hacking */
            PetscScalar *data;
            PetscInt     nd = 0;

            if (!use_potr) {
              SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor update not yet implemented for non SPD matrices");
            }
            CHKERRQ(MatFactorGetSchurComplement(F,&S_all_inv,NULL));
            CHKERRQ(MatDenseGetArray(S_all_inv,&data));
            if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
              CHKERRQ(ISGetLocalSize(sub_schurs->is_dir,&nd));
            }

            /* factor and invert activedofs and vertices (dirichlet dofs does not contribute) */
            if (schur_has_vertices) {
              Mat          M;
              PetscScalar *tdata;
              PetscInt     nv = 0, news;

              CHKERRQ(ISGetLocalSize(sub_schurs->is_vertices,&nv));
              news = size_active_schur + nv;
              CHKERRQ(PetscCalloc1(news*news,&tdata));
              for (i=0;i<size_active_schur;i++) {
                CHKERRQ(PetscArraycpy(tdata+i*(news+1),data+i*(size_schur+1),size_active_schur-i));
                CHKERRQ(PetscArraycpy(tdata+i*(news+1)+size_active_schur-i,data+i*size_schur+size_active_schur+nd,nv));
              }
              for (i=0;i<nv;i++) {
                PetscInt k = i+size_active_schur;
                CHKERRQ(PetscArraycpy(tdata+k*(news+1),data+(k+nd)*(size_schur+1),nv-i));
              }

              CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,news,news,tdata,&M));
              CHKERRQ(MatSetOption(M,MAT_SPD,PETSC_TRUE));
              CHKERRQ(MatCholeskyFactor(M,NULL,NULL));
              /* save the factors */
              cum = 0;
              CHKERRQ(PetscMalloc1((size_active_schur*(size_active_schur +1))/2+nd,&schur_factor));
              for (i=0;i<size_active_schur;i++) {
                CHKERRQ(PetscArraycpy(schur_factor+cum,tdata+i*(news+1),size_active_schur-i));
                cum += size_active_schur - i;
              }
              for (i=0;i<nd;i++) schur_factor[cum+i] = PetscSqrtReal(PetscRealPart(data[(i+size_active_schur)*(size_schur+1)]));
              CHKERRQ(MatSeqDenseInvertFactors_Private(M));
              /* move back just the active dofs to the Schur complement */
              for (i=0;i<size_active_schur;i++) {
                CHKERRQ(PetscArraycpy(data+i*size_schur,tdata+i*news,size_active_schur));
              }
              CHKERRQ(PetscFree(tdata));
              CHKERRQ(MatDestroy(&M));
            } else { /* we can factorize and invert just the activedofs */
              Mat         M;
              PetscScalar *aux;

              CHKERRQ(PetscMalloc1(nd,&aux));
              for (i=0;i<nd;i++) aux[i] = 1.0/data[(i+size_active_schur)*(size_schur+1)];
              CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,size_active_schur,size_active_schur,data,&M));
              CHKERRQ(MatDenseSetLDA(M,size_schur));
              CHKERRQ(MatSetOption(M,MAT_SPD,PETSC_TRUE));
              CHKERRQ(MatCholeskyFactor(M,NULL,NULL));
              CHKERRQ(MatSeqDenseInvertFactors_Private(M));
              CHKERRQ(MatDestroy(&M));
              CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,size_schur,nd,data+size_active_schur*size_schur,&M));
              CHKERRQ(MatZeroEntries(M));
              CHKERRQ(MatDestroy(&M));
              CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,nd,size_schur,data+size_active_schur,&M));
              CHKERRQ(MatDenseSetLDA(M,size_schur));
              CHKERRQ(MatZeroEntries(M));
              CHKERRQ(MatDestroy(&M));
              for (i=0;i<nd;i++) data[(i+size_active_schur)*(size_schur+1)] = aux[i];
              CHKERRQ(PetscFree(aux));
            }
            CHKERRQ(MatDenseRestoreArray(S_all_inv,&data));
          } else { /* use MatFactor calls to invert S */
            CHKERRQ(MatFactorInvertSchurComplement(F));
            CHKERRQ(MatFactorGetSchurComplement(F,&S_all_inv,NULL));
          }
        } else { /* we need to invert explicitly since we are not using MatFactor for S */
          CHKERRQ(PetscObjectReference((PetscObject)S_all));
          S_all_inv = S_all;
          CHKERRQ(MatDenseGetArray(S_all_inv,&S_data));
          CHKERRQ(PetscBLASIntCast(size_schur,&B_N));
          CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
          if (use_potr) {
            PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,S_data,&B_N,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,S_data,&B_N,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
          } else if (use_sytr) {
            PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,S_data,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&B_N,S_data,&B_N,pivots,Bwork,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRI Lapack routine %d",(int)B_ierr);
          } else {
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,S_data,&B_N,pivots,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,S_data,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
            PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          }
          CHKERRQ(PetscLogFlops(1.0*size_schur*size_schur*size_schur));
          CHKERRQ(PetscFPTrapPop());
          CHKERRQ(MatDenseRestoreArray(S_all_inv,&S_data));
        }
        /* S_Ej_tilda_all */
        cum = cum2 = 0;
        CHKERRQ(MatDenseGetArrayRead(S_all_inv,&rS_data));
        for (i=0;i<sub_schurs->n_subs;i++) {
          PetscInt j;

          CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
          /* get (St^-1)_E */
          /* Unless we are changing the variables, I don't need to expand to upper triangular since St^-1
             will be properly accessed later during adaptive selection */
          if (S_lower_triangular) {
            PetscInt k;
            if (sub_schurs->change) {
              for (k=0;k<subset_size;k++) {
                for (j=k;j<subset_size;j++) {
                  work[k*subset_size+j] = rS_data[cum2+k*size_schur+j];
                  work[j*subset_size+k] = work[k*subset_size+j];
                }
              }
            } else {
              for (k=0;k<subset_size;k++) {
                for (j=k;j<subset_size;j++) {
                  work[k*subset_size+j] = rS_data[cum2+k*size_schur+j];
                }
              }
            }
          } else {
            PetscInt k;
            for (k=0;k<subset_size;k++) {
              for (j=0;j<subset_size;j++) {
                work[k*subset_size+j] = rS_data[cum2+k*size_schur+j];
              }
            }
          }
          if (sub_schurs->change) {
            Mat change_sub,SEj,T;

            /* change basis */
            CHKERRQ(KSPGetOperators(sub_schurs->change[i],&change_sub,NULL));
            CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
            if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
              Mat T2;
              CHKERRQ(MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2));
              CHKERRQ(MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
              CHKERRQ(MatDestroy(&T2));
              CHKERRQ(MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T));
            } else {
              CHKERRQ(MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
            }
            CHKERRQ(MatCopy(T,SEj,SAME_NONZERO_PATTERN));
            CHKERRQ(MatDestroy(&T));
            /* set diagonal entry to a very large value to pick the basis we are eliminating as the first eigenvectors with adaptive selection */
            CHKERRQ(MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1./PETSC_SMALL,NULL,NULL));
            CHKERRQ(MatDestroy(&SEj));
          }
          CHKERRQ(PetscArraycpy(SEjinv_arr,work,subset_size*subset_size));
          cum += subset_size;
          cum2 += subset_size*(size_schur + 1);
          SEjinv_arr += subset_size*subset_size;
        }
        CHKERRQ(MatDenseRestoreArrayRead(S_all_inv,&rS_data));
        if (solver_S) {
          if (schur_has_vertices) {
            CHKERRQ(MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_FACTORED));
          } else {
            CHKERRQ(MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_INVERTED));
          }
        }
        CHKERRQ(MatDestroy(&S_all_inv));
      }
      CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr));

      /* move back factors if needed */
      if (schur_has_vertices) {
        Mat      S_tmp;
        PetscInt nd = 0;

        PetscCheck(solver_S,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        CHKERRQ(MatFactorGetSchurComplement(F,&S_tmp,NULL));
        if (use_potr) {
          PetscScalar *data;

          CHKERRQ(MatDenseGetArray(S_tmp,&data));
          CHKERRQ(PetscArrayzero(data,size_schur*size_schur));

          if (S_lower_triangular) {
            cum = 0;
            for (i=0;i<size_active_schur;i++) {
              CHKERRQ(PetscArraycpy(data+i*(size_schur+1),schur_factor+cum,size_active_schur-i));
              cum += size_active_schur-i;
            }
          } else {
            CHKERRQ(PetscArraycpy(data,schur_factor,size_schur*size_schur));
          }
          if (sub_schurs->is_dir) {
            CHKERRQ(ISGetLocalSize(sub_schurs->is_dir,&nd));
            for (i=0;i<nd;i++) {
              data[(i+size_active_schur)*(size_schur+1)] = schur_factor[cum+i];
            }
          }
          /* workaround: since I cannot modify the matrices used inside the solvers for the forward and backward substitutions,
             set the diagonal entry of the Schur factor to a very large value */
          for (i=size_active_schur+nd;i<size_schur;i++) {
            data[i*(size_schur+1)] = infty;
          }
          CHKERRQ(MatDenseRestoreArray(S_tmp,&data));
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor update not yet implemented for non SPD matrices");
        CHKERRQ(MatFactorRestoreSchurComplement(F,&S_tmp,MAT_FACTOR_SCHUR_FACTORED));
      }
    } else if (factor_workaround) { /* we need to eliminate any unneeded coupling */
      PetscScalar *data;
      PetscInt    nd = 0;

      if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
        CHKERRQ(ISGetLocalSize(sub_schurs->is_dir,&nd));
      }
      CHKERRQ(MatFactorGetSchurComplement(F,&S_all,NULL));
      CHKERRQ(MatDenseGetArray(S_all,&data));
      for (i=0;i<size_active_schur;i++) {
        CHKERRQ(PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur));
      }
      for (i=size_active_schur+nd;i<size_schur;i++) {
        CHKERRQ(PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur));
        data[i*(size_schur+1)] = infty;
      }
      CHKERRQ(MatDenseRestoreArray(S_all,&data));
      CHKERRQ(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
    }
    CHKERRQ(PetscFree(work));
    CHKERRQ(PetscFree(schur_factor));
    CHKERRQ(VecDestroy(&Dall));
  }
  CHKERRQ(ISDestroy(&is_I_layer));
  CHKERRQ(MatDestroy(&S_all));
  CHKERRQ(MatDestroy(&A_BB));
  CHKERRQ(MatDestroy(&A_IB));
  CHKERRQ(MatDestroy(&A_BI));
  CHKERRQ(MatDestroy(&F));

  CHKERRQ(PetscMalloc1(sub_schurs->n_subs,&nnz));
  for (i=0;i<sub_schurs->n_subs;i++) {
    CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&nnz[i]));
  }
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,sub_schurs->n_subs,nnz,PETSC_OWN_POINTER,&is_I_layer));
  CHKERRQ(MatSetVariableBlockSizes(sub_schurs->S_Ej_all,sub_schurs->n_subs,nnz));
  CHKERRQ(MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY));
  if (compute_Stilda) {
    CHKERRQ(MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_tilda_all,sub_schurs->n_subs,nnz));
    CHKERRQ(MatAssemblyBegin(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY));
    if (deluxe) {
      CHKERRQ(MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_inv_all,sub_schurs->n_subs,nnz));
      CHKERRQ(MatAssemblyBegin(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY));
    }
  }
  CHKERRQ(ISDestroy(&is_I_layer));

  /* Get local part of (\sum_j S_Ej) */
  if (!sub_schurs->sum_S_Ej_all) {
    CHKERRQ(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_all));
  }
  CHKERRQ(VecSet(gstash,0.0));
  CHKERRQ(MatSeqAIJGetArray(sub_schurs->S_Ej_all,&stasharray));
  CHKERRQ(VecPlaceArray(lstash,stasharray));
  CHKERRQ(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&stasharray));
  CHKERRQ(VecResetArray(lstash));
  CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all,&stasharray));
  CHKERRQ(VecPlaceArray(lstash,stasharray));
  CHKERRQ(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all,&stasharray));
  CHKERRQ(VecResetArray(lstash));

  /* Get local part of (\sum_j S^-1_Ej) (\sum_j St^-1_Ej) */
  if (compute_Stilda) {
    CHKERRQ(VecSet(gstash,0.0));
    CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray));
    CHKERRQ(VecPlaceArray(lstash,stasharray));
    CHKERRQ(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
    CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray));
    CHKERRQ(VecResetArray(lstash));
    if (deluxe) {
      CHKERRQ(VecSet(gstash,0.0));
      CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&stasharray));
      CHKERRQ(VecPlaceArray(lstash,stasharray));
      CHKERRQ(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
      CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&stasharray));
      CHKERRQ(VecResetArray(lstash));
    } else {
      PetscScalar *array;
      PetscInt    cum;

      CHKERRQ(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&array));
      cum = 0;
      for (i=0;i<sub_schurs->n_subs;i++) {
        CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
        CHKERRQ(PetscBLASIntCast(subset_size,&B_N));
        CHKERRQ(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
        if (use_potr) {
          PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,array+cum,&B_N,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,array+cum,&B_N,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
        } else if (use_sytr) {
          PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,array+cum,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&B_N,array+cum,&B_N,pivots,Bwork,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRI Lapack routine %d",(int)B_ierr);
        } else {
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,array+cum,&B_N,pivots,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,array+cum,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
          PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        }
        CHKERRQ(PetscLogFlops(1.0*subset_size*subset_size*subset_size));
        CHKERRQ(PetscFPTrapPop());
        cum += subset_size*subset_size;
      }
      CHKERRQ(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&array));
      CHKERRQ(PetscObjectReference((PetscObject)sub_schurs->sum_S_Ej_all));
      CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
      sub_schurs->sum_S_Ej_inv_all = sub_schurs->sum_S_Ej_all;
    }
  }
  CHKERRQ(VecDestroy(&lstash));
  CHKERRQ(VecDestroy(&gstash));
  CHKERRQ(VecScatterDestroy(&sstash));

  if (matl_dbg_viewer) {
    PetscInt cum;

    if (sub_schurs->S_Ej_all) {
      CHKERRQ(PetscObjectSetName((PetscObject)sub_schurs->S_Ej_all,"SE"));
      CHKERRQ(MatView(sub_schurs->S_Ej_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_all) {
      CHKERRQ(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_all,"SSE"));
      CHKERRQ(MatView(sub_schurs->sum_S_Ej_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_inv_all) {
      CHKERRQ(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_inv_all,"SSEm"));
      CHKERRQ(MatView(sub_schurs->sum_S_Ej_inv_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_tilda_all) {
      CHKERRQ(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_tilda_all,"SSEt"));
      CHKERRQ(MatView(sub_schurs->sum_S_Ej_tilda_all,matl_dbg_viewer));
    }
    for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
      IS   is;
      char name[16];

      CHKERRQ(PetscSNPrintf(name,sizeof(name),"IE%D",i));
      CHKERRQ(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      CHKERRQ(ISCreateStride(PETSC_COMM_SELF,subset_size,cum,1,&is));
      CHKERRQ(PetscObjectSetName((PetscObject)is,name));
      CHKERRQ(ISView(is,matl_dbg_viewer));
      CHKERRQ(ISDestroy(&is));
      cum += subset_size;
    }
  }

  /* free workspace */
  CHKERRQ(PetscViewerDestroy(&matl_dbg_viewer));
  CHKERRQ(PetscFree2(Bwork,pivots));
  CHKERRQ(PetscCommDestroy(&comm_n));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, const char* prefix, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap, PetscBool copycc)
{
  IS              *faces,*edges,*all_cc,vertices;
  PetscInt        i,n_faces,n_edges,n_all_cc;
  PetscBool       is_sorted,ispardiso,ismumps;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  CHKERRQ(ISSorted(is_I,&is_sorted));
  PetscCheck(is_sorted,PetscObjectComm((PetscObject)is_I),PETSC_ERR_PLIB,"IS for I dofs should be shorted");
  CHKERRQ(ISSorted(is_B,&is_sorted));
  PetscCheck(is_sorted,PetscObjectComm((PetscObject)is_B),PETSC_ERR_PLIB,"IS for B dofs should be shorted");

  /* reset any previous data */
  CHKERRQ(PCBDDCSubSchursReset(sub_schurs));

  /* get index sets for faces and edges (already sorted by global ordering) */
  CHKERRQ(PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices));
  n_all_cc = n_faces+n_edges;
  CHKERRQ(PetscBTCreate(n_all_cc,&sub_schurs->is_edge));
  CHKERRQ(PetscMalloc1(n_all_cc,&all_cc));
  for (i=0;i<n_faces;i++) {
    if (copycc) {
      CHKERRQ(ISDuplicate(faces[i],&all_cc[i]));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)faces[i]));
      all_cc[i] = faces[i];
    }
  }
  for (i=0;i<n_edges;i++) {
    if (copycc) {
      CHKERRQ(ISDuplicate(edges[i],&all_cc[n_faces+i]));
    } else {
      CHKERRQ(PetscObjectReference((PetscObject)edges[i]));
      all_cc[n_faces+i] = edges[i];
    }
    CHKERRQ(PetscBTSet(sub_schurs->is_edge,n_faces+i));
  }
  CHKERRQ(PetscObjectReference((PetscObject)vertices));
  sub_schurs->is_vertices = vertices;
  CHKERRQ(PCBDDCGraphRestoreCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices));
  sub_schurs->is_dir = NULL;
  CHKERRQ(PCBDDCGraphGetDirichletDofsB(graph,&sub_schurs->is_dir));

  /* Determine if MatFactor can be used */
  CHKERRQ(PetscStrallocpy(prefix,&sub_schurs->prefix));
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMUMPS,sizeof(sub_schurs->mat_solver_type)));
#elif defined(PETSC_HAVE_MKL_PARDISO)
  CHKERRQ(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,sizeof(sub_schurs->mat_solver_type)));
#else
  CHKERRQ(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERPETSC,sizeof(sub_schurs->mat_solver_type)));
#endif
#if defined(PETSC_USE_COMPLEX)
  sub_schurs->is_hermitian  = PETSC_FALSE; /* Hermitian Cholesky is not supported by PETSc and external packages */
#else
  sub_schurs->is_hermitian  = PETSC_TRUE;
#endif
  sub_schurs->is_posdef     = PETSC_TRUE;
  sub_schurs->is_symmetric  = PETSC_TRUE;
  sub_schurs->debug         = PETSC_FALSE;
  sub_schurs->restrict_comm = PETSC_FALSE;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)graph->l2gmap),sub_schurs->prefix,"BDDC sub_schurs options","PC");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-sub_schurs_mat_solver_type","Specific direct solver to use",NULL,sub_schurs->mat_solver_type,sub_schurs->mat_solver_type,sizeof(sub_schurs->mat_solver_type),NULL));
  CHKERRQ(PetscOptionsBool("-sub_schurs_symmetric","Symmetric problem",NULL,sub_schurs->is_symmetric,&sub_schurs->is_symmetric,NULL));
  CHKERRQ(PetscOptionsBool("-sub_schurs_hermitian","Hermitian problem",NULL,sub_schurs->is_hermitian,&sub_schurs->is_hermitian,NULL));
  CHKERRQ(PetscOptionsBool("-sub_schurs_posdef","Positive definite problem",NULL,sub_schurs->is_posdef,&sub_schurs->is_posdef,NULL));
  CHKERRQ(PetscOptionsBool("-sub_schurs_restrictcomm","Restrict communicator on active processes only",NULL,sub_schurs->restrict_comm,&sub_schurs->restrict_comm,NULL));
  CHKERRQ(PetscOptionsBool("-sub_schurs_debug","Debug output",NULL,sub_schurs->debug,&sub_schurs->debug,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMUMPS,&ismumps));
  CHKERRQ(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,&ispardiso));
  sub_schurs->schur_explicit = (PetscBool)(ispardiso || ismumps);

  /* for reals, symmetric and hermitian are synonims */
#if !defined(PETSC_USE_COMPLEX)
  sub_schurs->is_symmetric = (PetscBool)(sub_schurs->is_symmetric && sub_schurs->is_hermitian);
  sub_schurs->is_hermitian = sub_schurs->is_symmetric;
#endif

  CHKERRQ(PetscObjectReference((PetscObject)is_I));
  sub_schurs->is_I = is_I;
  CHKERRQ(PetscObjectReference((PetscObject)is_B));
  sub_schurs->is_B = is_B;
  CHKERRQ(PetscObjectReference((PetscObject)graph->l2gmap));
  sub_schurs->l2gmap = graph->l2gmap;
  CHKERRQ(PetscObjectReference((PetscObject)BtoNmap));
  sub_schurs->BtoNmap = BtoNmap;
  sub_schurs->n_subs = n_all_cc;
  sub_schurs->is_subs = all_cc;
  sub_schurs->S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_all = NULL;
  sub_schurs->sum_S_Ej_inv_all = NULL;
  sub_schurs->sum_S_Ej_tilda_all = NULL;
  sub_schurs->is_Ej_all = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursCreate(PCBDDCSubSchurs *sub_schurs)
{
  PCBDDCSubSchurs schurs_ctx;

  PetscFunctionBegin;
  CHKERRQ(PetscNew(&schurs_ctx));
  schurs_ctx->n_subs = 0;
  *sub_schurs = schurs_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs sub_schurs)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!sub_schurs) PetscFunctionReturn(0);
  CHKERRQ(PetscFree(sub_schurs->prefix));
  CHKERRQ(MatDestroy(&sub_schurs->A));
  CHKERRQ(MatDestroy(&sub_schurs->S));
  CHKERRQ(ISDestroy(&sub_schurs->is_I));
  CHKERRQ(ISDestroy(&sub_schurs->is_B));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&sub_schurs->l2gmap));
  CHKERRQ(ISLocalToGlobalMappingDestroy(&sub_schurs->BtoNmap));
  CHKERRQ(MatDestroy(&sub_schurs->S_Ej_all));
  CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_all));
  CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
  CHKERRQ(MatDestroy(&sub_schurs->sum_S_Ej_tilda_all));
  CHKERRQ(ISDestroy(&sub_schurs->is_Ej_all));
  CHKERRQ(ISDestroy(&sub_schurs->is_vertices));
  CHKERRQ(ISDestroy(&sub_schurs->is_dir));
  CHKERRQ(PetscBTDestroy(&sub_schurs->is_edge));
  for (i=0;i<sub_schurs->n_subs;i++) {
    CHKERRQ(ISDestroy(&sub_schurs->is_subs[i]));
  }
  if (sub_schurs->n_subs) {
    CHKERRQ(PetscFree(sub_schurs->is_subs));
  }
  if (sub_schurs->reuse_solver) {
    CHKERRQ(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
  }
  CHKERRQ(PetscFree(sub_schurs->reuse_solver));
  if (sub_schurs->change) {
    for (i=0;i<sub_schurs->n_subs;i++) {
      CHKERRQ(KSPDestroy(&sub_schurs->change[i]));
      CHKERRQ(ISDestroy(&sub_schurs->change_primal_sub[i]));
    }
  }
  CHKERRQ(PetscFree(sub_schurs->change));
  CHKERRQ(PetscFree(sub_schurs->change_primal_sub));
  sub_schurs->n_subs = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs* sub_schurs)
{
  PetscFunctionBegin;
  CHKERRQ(PCBDDCSubSchursReset(*sub_schurs));
  CHKERRQ(PetscFree(*sub_schurs));
  PetscFunctionReturn(0);
}

static inline PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt* queue_tip,PetscInt n_prev,PetscBT touched,PetscInt* xadj,PetscInt* adjncy,PetscInt* n_added)
{
  PetscInt       i,j,n;

  PetscFunctionBegin;
  n = 0;
  for (i=-n_prev;i<0;i++) {
    PetscInt start_dof = queue_tip[i];
    for (j=xadj[start_dof];j<xadj[start_dof+1];j++) {
      PetscInt dof = adjncy[j];
      if (!PetscBTLookup(touched,dof)) {
        CHKERRQ(PetscBTSet(touched,dof));
        queue_tip[n] = dof;
        n++;
      }
    }
  }
  *n_added = n;
  PetscFunctionReturn(0);
}
