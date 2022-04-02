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
    PetscCall(MatGetSize(ctx->benign_csAIB,&size_schur,NULL));
    PetscCall(VecGetSize(v,&n_I));
    n_I = n_I - size_schur;
    /* get schur sol from array */
    PetscCall(VecGetArray(v,&array));
    PetscCall(VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I));
    PetscCall(VecRestoreArray(v,&array));
    /* apply interior sol correction */
    PetscCall(MatMultTranspose(ctx->benign_csAIB,ctx->benign_dummy_schur_vec,ctx->benign_corr_work));
    PetscCall(VecResetArray(ctx->benign_dummy_schur_vec));
    PetscCall(MatMultAdd(ctx->benign_AIIm1ones,ctx->benign_corr_work,v,v));
  }
  if (v2) {
    PetscInt nl;

    PetscCall(VecGetArrayRead(v,(const PetscScalar**)&array));
    PetscCall(VecGetLocalSize(v2,&nl));
    PetscCall(VecGetArray(v2,&array2));
    PetscCall(PetscArraycpy(array2,array,nl));
  } else {
    PetscCall(VecGetArray(v,&array));
    array2 = array;
  }
  if (!sol) { /* change rhs */
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;

      PetscCall(ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz));
      PetscCall(ISGetIndices(ctx->benign_zerodiag_subs[n],&cols));
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      ctx->benign_save_vals[n] = array2[cols[nz-1]];
      array2[cols[nz-1]] = sum;
      PetscCall(ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols));
    }
  } else {
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;
      PetscCall(ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz));
      PetscCall(ISGetIndices(ctx->benign_zerodiag_subs[n],&cols));
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      array2[cols[nz-1]] = ctx->benign_save_vals[n];
      PetscCall(ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols));
    }
  }
  if (v2) {
    PetscCall(VecRestoreArrayRead(v,(const PetscScalar**)&array));
    PetscCall(VecRestoreArray(v2,&array2));
  } else {
    PetscCall(VecRestoreArray(v,&array));
  }
  if (!sol && full) {
    Vec      usedv;
    PetscInt n_I,size_schur;

    /* get sizes */
    PetscCall(MatGetSize(ctx->benign_csAIB,&size_schur,NULL));
    PetscCall(VecGetSize(v,&n_I));
    n_I = n_I - size_schur;
    /* compute schur rhs correction */
    if (v2) {
      usedv = v2;
    } else {
      usedv = v;
    }
    /* apply schur rhs correction */
    PetscCall(MatMultTranspose(ctx->benign_AIIm1ones,usedv,ctx->benign_corr_work));
    PetscCall(VecGetArrayRead(usedv,(const PetscScalar**)&array));
    PetscCall(VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I));
    PetscCall(VecRestoreArrayRead(usedv,(const PetscScalar**)&array));
    PetscCall(MatMultAdd(ctx->benign_csAIB,ctx->benign_corr_work,ctx->benign_dummy_schur_vec,ctx->benign_dummy_schur_vec));
    PetscCall(VecResetArray(ctx->benign_dummy_schur_vec));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Solve_Private(PC pc, Vec rhs, Vec sol, PetscBool transpose, PetscBool full)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          copy = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&ctx));
  if (full) {
#if defined(PETSC_HAVE_MUMPS)
    PetscCall(MatMumpsSetIcntl(ctx->F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    PetscCall(MatMkl_PardisoSetCntl(ctx->F,70,0));
#endif
    copy = ctx->has_vertices;
  } else { /* interior solver */
#if defined(PETSC_HAVE_MUMPS)
    PetscCall(MatMumpsSetIcntl(ctx->F,26,0));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    PetscCall(MatMkl_PardisoSetCntl(ctx->F,70,1));
#endif
    copy = PETSC_TRUE;
  }
  /* copy rhs into factored matrix workspace */
  if (copy) {
    PetscInt    n;
    PetscScalar *array,*array_solver;

    PetscCall(VecGetLocalSize(rhs,&n));
    PetscCall(VecGetArrayRead(rhs,(const PetscScalar**)&array));
    PetscCall(VecGetArray(ctx->rhs,&array_solver));
    PetscCall(PetscArraycpy(array_solver,array,n));
    PetscCall(VecRestoreArray(ctx->rhs,&array_solver));
    PetscCall(VecRestoreArrayRead(rhs,(const PetscScalar**)&array));

    PetscCall(PCBDDCReuseSolversBenignAdapt(ctx,ctx->rhs,NULL,PETSC_FALSE,full));
    if (transpose) {
      PetscCall(MatSolveTranspose(ctx->F,ctx->rhs,ctx->sol));
    } else {
      PetscCall(MatSolve(ctx->F,ctx->rhs,ctx->sol));
    }
    PetscCall(PCBDDCReuseSolversBenignAdapt(ctx,ctx->sol,NULL,PETSC_TRUE,full));

    /* get back data to caller worskpace */
    PetscCall(VecGetArrayRead(ctx->sol,(const PetscScalar**)&array_solver));
    PetscCall(VecGetArray(sol,&array));
    PetscCall(PetscArraycpy(array,array_solver,n));
    PetscCall(VecRestoreArray(sol,&array));
    PetscCall(VecRestoreArrayRead(ctx->sol,(const PetscScalar**)&array_solver));
  } else {
    if (ctx->benign_n) {
      PetscCall(PCBDDCReuseSolversBenignAdapt(ctx,rhs,ctx->rhs,PETSC_FALSE,full));
      if (transpose) {
        PetscCall(MatSolveTranspose(ctx->F,ctx->rhs,sol));
      } else {
        PetscCall(MatSolve(ctx->F,ctx->rhs,sol));
      }
      PetscCall(PCBDDCReuseSolversBenignAdapt(ctx,sol,NULL,PETSC_TRUE,full));
    } else {
      if (transpose) {
        PetscCall(MatSolveTranspose(ctx->F,rhs,sol));
      } else {
        PetscCall(MatSolve(ctx->F,rhs,sol));
      }
    }
  }
  /* restore defaults */
#if defined(PETSC_HAVE_MUMPS)
  PetscCall(MatMumpsSetIcntl(ctx->F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  PetscCall(MatMkl_PardisoSetCntl(ctx->F,70,0));
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Correction(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_CorrectionTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Interior(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_InteriorTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_View(PC pc, PetscViewer viewer)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          iascii;

  PetscFunctionBegin;
  PetscCall(PCShellGetContext(pc,&ctx));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO));
  }
  PetscCall(MatView(ctx->F,viewer));
  if (iascii) {
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolversReset(PCBDDCReuseSolvers reuse)
{
  PetscInt       i;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&reuse->F));
  PetscCall(VecDestroy(&reuse->sol));
  PetscCall(VecDestroy(&reuse->rhs));
  PetscCall(PCDestroy(&reuse->interior_solver));
  PetscCall(PCDestroy(&reuse->correction_solver));
  PetscCall(ISDestroy(&reuse->is_R));
  PetscCall(ISDestroy(&reuse->is_B));
  PetscCall(VecScatterDestroy(&reuse->correction_scatter_B));
  PetscCall(VecDestroy(&reuse->sol_B));
  PetscCall(VecDestroy(&reuse->rhs_B));
  for (i=0;i<reuse->benign_n;i++) {
    PetscCall(ISDestroy(&reuse->benign_zerodiag_subs[i]));
  }
  PetscCall(PetscFree(reuse->benign_zerodiag_subs));
  PetscCall(PetscFree(reuse->benign_save_vals));
  PetscCall(MatDestroy(&reuse->benign_csAIB));
  PetscCall(MatDestroy(&reuse->benign_AIIm1ones));
  PetscCall(VecDestroy(&reuse->benign_corr_work));
  PetscCall(VecDestroy(&reuse->benign_dummy_schur_vec));
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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)M),&size));
  PetscCheck(size == 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for parallel matrices");
  if (reuse == MAT_REUSE_MATRIX) {
    PetscBool Sdense;

    PetscCall(PetscObjectTypeCompare((PetscObject)*S, MATSEQDENSE, &Sdense));
    PetscCheck(Sdense,PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"S should dense");
  }
  PetscCall(MatSchurComplementGetSubMatrices(M, NULL, NULL, &B, &C, &D));
  PetscCall(MatSchurComplementGetKSP(M, &ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject) pc, PCLU, &isLU));
  PetscCall(PetscObjectTypeCompare((PetscObject) pc, PCILU, &isILU));
  PetscCall(PetscObjectTypeCompare((PetscObject) pc, PCCHOLESKY, &isCHOL));
  PetscCall(PetscObjectTypeCompare((PetscObject) B, MATSEQDENSE, &Bdense));
  PetscCall(PetscObjectTypeCompare((PetscObject) C, MATSEQDENSE, &Cdense));
  PetscCall(MatGetSize(B,&n_I,NULL));
  if (n_I) {
    if (!Bdense) {
      PetscCall(MatConvert(B, MATSEQDENSE, MAT_INITIAL_MATRIX, &Bd));
    } else {
      Bd = B;
    }

    if (isLU || isILU || isCHOL) {
      Mat fact;
      PetscCall(KSPSetUp(ksp));
      PetscCall(PCFactorGetMatrix(pc, &fact));
      PetscCall(MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd));
      PetscCall(MatMatSolve(fact, Bd, AinvBd));
    } else {
      PetscBool ex = PETSC_TRUE;

      if (ex) {
        Mat Ainvd;

        PetscCall(PCComputeOperator(pc, MATDENSE, &Ainvd));
        PetscCall(MatMatMult(Ainvd, Bd, MAT_INITIAL_MATRIX, fill, &AinvBd));
        PetscCall(MatDestroy(&Ainvd));
      } else {
        Vec         sol,rhs;
        PetscScalar *arrayrhs,*arraysol;
        PetscInt    i,nrhs,n;

        PetscCall(MatDuplicate(Bd, MAT_DO_NOT_COPY_VALUES, &AinvBd));
        PetscCall(MatGetSize(Bd,&n,&nrhs));
        PetscCall(MatDenseGetArray(Bd,&arrayrhs));
        PetscCall(MatDenseGetArray(AinvBd,&arraysol));
        PetscCall(KSPGetSolution(ksp,&sol));
        PetscCall(KSPGetRhs(ksp,&rhs));
        for (i=0;i<nrhs;i++) {
          PetscCall(VecPlaceArray(rhs,arrayrhs+i*n));
          PetscCall(VecPlaceArray(sol,arraysol+i*n));
          PetscCall(KSPSolve(ksp,rhs,sol));
          PetscCall(VecResetArray(rhs));
          PetscCall(VecResetArray(sol));
        }
        PetscCall(MatDenseRestoreArray(Bd,&arrayrhs));
        PetscCall(MatDenseRestoreArray(AinvBd,&arrayrhs));
      }
    }
    if (!Bdense & !issym) {
      PetscCall(MatDestroy(&Bd));
    }

    if (!issym) {
      if (!Cdense) {
        PetscCall(MatConvert(C, MATSEQDENSE, MAT_INITIAL_MATRIX, &Cd));
      } else {
        Cd = C;
      }
      PetscCall(MatMatMult(Cd, AinvBd, reuse, fill, S));
      if (!Cdense) {
        PetscCall(MatDestroy(&Cd));
      }
    } else {
      PetscCall(MatTransposeMatMult(Bd, AinvBd, reuse, fill, S));
      if (!Bdense) {
        PetscCall(MatDestroy(&Bd));
      }
    }
    PetscCall(MatDestroy(&AinvBd));
  }

  if (D) {
    Mat       Dd;
    PetscBool Ddense;

    PetscCall(PetscObjectTypeCompare((PetscObject)D,MATSEQDENSE,&Ddense));
    if (!Ddense) {
      PetscCall(MatConvert(D, MATSEQDENSE, MAT_INITIAL_MATRIX, &Dd));
    } else {
      Dd = D;
    }
    if (n_I) {
      PetscCall(MatAYPX(*S,-1.0,Dd,SAME_NONZERO_PATTERN));
    } else {
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(MatDuplicate(Dd,MAT_COPY_VALUES,S));
      } else {
        PetscCall(MatCopy(Dd,*S,SAME_NONZERO_PATTERN));
      }
    }
    if (!Ddense) {
      PetscCall(MatDestroy(&Dd));
    }
  } else {
    PetscCall(MatScale(*S,-1.0));
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
  PetscBool              flg;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&sub_schurs->A));
  PetscCall(MatDestroy(&sub_schurs->S));
  if (Ain) {
    PetscCall(PetscObjectReference((PetscObject)Ain));
    sub_schurs->A = Ain;
  }

  PetscCall(PetscObjectReference((PetscObject)Sin));
  sub_schurs->S = Sin;
  if (sub_schurs->schur_explicit) {
    sub_schurs->schur_explicit = (PetscBool)(!!sub_schurs->A);
  }

  /* preliminary checks */
  PetscCheck(sub_schurs->schur_explicit || !compute_Stilda,PetscObjectComm((PetscObject)sub_schurs->l2gmap),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_PARDISO");

  if (benign_trick) sub_schurs->is_posdef = PETSC_FALSE;

  /* debug (MATLAB) */
  if (sub_schurs->debug) {
    PetscMPIInt size,rank;
    PetscInt    nr,*print_schurs_ranks,print_schurs = PETSC_FALSE;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&size));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank));
    nr   = size;
    PetscCall(PetscMalloc1(nr,&print_schurs_ranks));
    PetscOptionsBegin(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->prefix,"BDDC sub_schurs options","PC");
    PetscCall(PetscOptionsIntArray("-sub_schurs_debug_ranks","Ranks to debug (all if the option is not used)",NULL,print_schurs_ranks,&nr,&flg));
    if (!flg) print_schurs = PETSC_TRUE;
    else {
      print_schurs = PETSC_FALSE;
      for (i=0;i<nr;i++) if (print_schurs_ranks[i] == (PetscInt)rank) { print_schurs = PETSC_TRUE; break; }
    }
    PetscOptionsEnd();
    PetscCall(PetscFree(print_schurs_ranks));
    if (print_schurs) {
      char filename[256];

      PetscCall(PetscSNPrintf(filename,sizeof(filename),"sub_schurs_Schur_r%d.m",PetscGlobalRank));
      PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&matl_dbg_viewer));
      PetscCall(PetscViewerPushFormat(matl_dbg_viewer,PETSC_VIEWER_ASCII_MATLAB));
    }
  }

  /* restrict work on active processes */
  if (sub_schurs->restrict_comm) {
    PetscSubcomm subcomm;
    PetscMPIInt  color,rank;

    color = 0;
    if (!sub_schurs->n_subs) color = 1; /* this can happen if we are in a multilevel case or if the subdomain is disconnected */
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank));
    PetscCall(PetscSubcommCreate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&subcomm));
    PetscCall(PetscSubcommSetNumber(subcomm,2));
    PetscCall(PetscSubcommSetTypeGeneral(subcomm,color,rank));
    PetscCall(PetscCommDuplicate(PetscSubcommChild(subcomm),&comm_n,NULL));
    PetscCall(PetscSubcommDestroy(&subcomm));
    if (!sub_schurs->n_subs) {
      PetscCall(PetscCommDestroy(&comm_n));
      PetscFunctionReturn(0);
    }
  } else {
    PetscCall(PetscCommDuplicate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&comm_n,NULL));
  }

  /* get Schur complement matrices */
  if (!sub_schurs->schur_explicit) {
    Mat       tA_IB,tA_BI,tA_BB;
    PetscBool isseqsbaij;
    PetscCall(MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,&tA_IB,&tA_BI,&tA_BB));
    PetscCall(PetscObjectTypeCompare((PetscObject)tA_BB,MATSEQSBAIJ,&isseqsbaij));
    if (isseqsbaij) {
      PetscCall(MatConvert(tA_BB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BB));
      PetscCall(MatConvert(tA_IB,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_IB));
      PetscCall(MatConvert(tA_BI,MATSEQAIJ,MAT_INITIAL_MATRIX,&A_BI));
    } else {
      PetscCall(PetscObjectReference((PetscObject)tA_BB));
      A_BB = tA_BB;
      PetscCall(PetscObjectReference((PetscObject)tA_IB));
      A_IB = tA_IB;
      PetscCall(PetscObjectReference((PetscObject)tA_BI));
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
  PetscCall(ISGetLocalSize(sub_schurs->is_I,&i));
  if (nlayers >= 0 && i) { /* Interior problems can be different from the original one */
    PetscBT                touched;
    const PetscInt*        idx_B;
    PetscInt               n_I,n_B,n_local_dofs,n_prev_added,j,layer,*local_numbering;

    PetscCheck(xadj,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot request layering without adjacency");
    /* get sizes */
    PetscCall(ISGetLocalSize(sub_schurs->is_I,&n_I));
    PetscCall(ISGetLocalSize(sub_schurs->is_B,&n_B));

    PetscCall(PetscMalloc1(n_I+n_B,&local_numbering));
    PetscCall(PetscBTCreate(n_I+n_B,&touched));
    PetscCall(PetscBTMemzero(n_I+n_B,touched));

    /* all boundary dofs must be skipped when adding layers */
    PetscCall(ISGetIndices(sub_schurs->is_B,&idx_B));
    for (j=0;j<n_B;j++) {
      PetscCall(PetscBTSet(touched,idx_B[j]));
    }
    PetscCall(PetscArraycpy(local_numbering,idx_B,n_B));
    PetscCall(ISRestoreIndices(sub_schurs->is_B,&idx_B));

    /* add prescribed number of layers of dofs */
    n_local_dofs = n_B;
    n_prev_added = n_B;
    for (layer=0;layer<nlayers;layer++) {
      PetscInt n_added = 0;
      if (n_local_dofs == n_I+n_B) break;
      PetscCheck(n_local_dofs <= n_I+n_B,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error querying layer %D. Out of bound access (%D > %D)",layer,n_local_dofs,n_I+n_B);
      PetscCall(PCBDDCAdjGetNextLayer_Private(local_numbering+n_local_dofs,n_prev_added,touched,xadj,adjncy,&n_added));
      n_prev_added = n_added;
      n_local_dofs += n_added;
      if (!n_added) break;
    }
    PetscCall(PetscBTDestroy(&touched));

    /* IS for I layer dofs in original numbering */
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)sub_schurs->is_I),n_local_dofs-n_B,local_numbering+n_B,PETSC_COPY_VALUES,&is_I_layer));
    PetscCall(PetscFree(local_numbering));
    PetscCall(ISSort(is_I_layer));
    /* IS for I layer dofs in I numbering */
    if (!sub_schurs->schur_explicit) {
      ISLocalToGlobalMapping ItoNmap;
      PetscCall(ISLocalToGlobalMappingCreateIS(sub_schurs->is_I,&ItoNmap));
      PetscCall(ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,is_I_layer,&is_I));
      PetscCall(ISLocalToGlobalMappingDestroy(&ItoNmap));

      /* II block */
      PetscCall(MatCreateSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II));
    }
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    PetscCall(PetscObjectReference((PetscObject)sub_schurs->is_I));
    is_I_layer = sub_schurs->is_I;

    /* IS for I dofs in I numbering (strided 1) */
    if (!sub_schurs->schur_explicit) {
      PetscCall(ISGetSize(sub_schurs->is_I,&n_I));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)sub_schurs->is_I),n_I,0,1,&is_I));

      /* II block is the same */
      PetscCall(PetscObjectReference((PetscObject)A_II));
      AE_II = A_II;
    }
  }

  /* Get info on subset sizes and sum of all subsets sizes */
  max_subset_size = 0;
  local_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    max_subset_size = PetscMax(subset_size,max_subset_size);
    local_size += subset_size;
  }

  /* Work arrays for local indices */
  extra = 0;
  PetscCall(ISGetLocalSize(sub_schurs->is_B,&n_B));
  if (sub_schurs->schur_explicit && is_I_layer) {
    PetscCall(ISGetLocalSize(is_I_layer,&extra));
  }
  PetscCall(PetscMalloc1(n_B+extra,&all_local_idx_N));
  if (extra) {
    const PetscInt *idxs;
    PetscCall(ISGetIndices(is_I_layer,&idxs));
    PetscCall(PetscArraycpy(all_local_idx_N,idxs,extra));
    PetscCall(ISRestoreIndices(is_I_layer,&idxs));
  }
  PetscCall(PetscMalloc1(sub_schurs->n_subs,&auxnum1));
  PetscCall(PetscMalloc1(sub_schurs->n_subs,&auxnum2));

  /* Get local indices in local numbering */
  local_size = 0;
  local_stash_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    const PetscInt *idxs;

    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
    PetscCall(ISGetIndices(sub_schurs->is_subs[i],&idxs));
    /* start (smallest in global ordering) and multiplicity */
    auxnum1[i] = idxs[0];
    auxnum2[i] = subset_size*subset_size;
    /* subset indices in local numbering */
    PetscCall(PetscArraycpy(all_local_idx_N+local_size+extra,idxs,subset_size));
    PetscCall(ISRestoreIndices(sub_schurs->is_subs[i],&idxs));
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
    PetscCall(PetscBLASIntCast(local_size,&B_N));
    PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
    if (use_sytr) {
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYTRF Lapack routine %d",(int)B_ierr);
    } else {
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      PetscCheck(!B_ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
    }
    PetscCall(PetscFPTrapPop());
    PetscCall(PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork));
    PetscCall(PetscMalloc2(B_lwork,&Bwork,B_N,&pivots));
  } else {
    Bwork = NULL;
    pivots = NULL;
  }

  /* prepare data for summing up properly schurs on subsets */
  PetscCall(ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum1,PETSC_OWN_POINTER,&all_subsets_n));
  PetscCall(ISLocalToGlobalMappingApplyIS(sub_schurs->l2gmap,all_subsets_n,&all_subsets));
  PetscCall(ISDestroy(&all_subsets_n));
  PetscCall(ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum2,PETSC_OWN_POINTER,&all_subsets_mult));
  PetscCall(ISRenumber(all_subsets,all_subsets_mult,&global_size,&all_subsets_n));
  PetscCall(ISDestroy(&all_subsets));
  PetscCall(ISDestroy(&all_subsets_mult));
  PetscCall(ISGetLocalSize(all_subsets_n,&i));
  PetscCheck(i == local_stash_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",i,local_stash_size);
  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,local_stash_size,NULL,&lstash));
  PetscCall(VecCreateMPI(comm_n,PETSC_DECIDE,global_size,&gstash));
  PetscCall(VecScatterCreate(lstash,NULL,gstash,all_subsets_n,&sstash));
  PetscCall(ISDestroy(&all_subsets_n));

  /* subset indices in local boundary numbering */
  if (!sub_schurs->is_Ej_all) {
    PetscInt *all_local_idx_B;

    PetscCall(PetscMalloc1(local_size,&all_local_idx_B));
    PetscCall(ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B));
    PetscCheck(subset_size == local_size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %D != %D",subset_size,local_size);
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all));
  }

  if (change) {
    ISLocalToGlobalMapping BtoS;
    IS                     change_primal_B;
    IS                     change_primal_all;

    PetscCheck(!sub_schurs->change_primal_sub,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    PetscCheck(!sub_schurs->change,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    PetscCall(PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change_primal_sub));
    for (i=0;i<sub_schurs->n_subs;i++) {
      ISLocalToGlobalMapping NtoS;
      PetscCall(ISLocalToGlobalMappingCreateIS(sub_schurs->is_subs[i],&NtoS));
      PetscCall(ISGlobalToLocalMappingApplyIS(NtoS,IS_GTOLM_DROP,change_primal,&sub_schurs->change_primal_sub[i]));
      PetscCall(ISLocalToGlobalMappingDestroy(&NtoS));
    }
    PetscCall(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,change_primal,&change_primal_B));
    PetscCall(ISLocalToGlobalMappingCreateIS(sub_schurs->is_Ej_all,&BtoS));
    PetscCall(ISGlobalToLocalMappingApplyIS(BtoS,IS_GTOLM_DROP,change_primal_B,&change_primal_all));
    PetscCall(ISLocalToGlobalMappingDestroy(&BtoS));
    PetscCall(ISDestroy(&change_primal_B));
    PetscCall(PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change));
    for (i=0;i<sub_schurs->n_subs;i++) {
      Mat change_sub;

      PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      PetscCall(KSPCreate(PETSC_COMM_SELF,&sub_schurs->change[i]));
      PetscCall(KSPSetType(sub_schurs->change[i],KSPPREONLY));
      if (!sub_schurs->change_with_qr) {
        PetscCall(MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_sub));
      } else {
        Mat change_subt;
        PetscCall(MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_subt));
        PetscCall(MatConvert(change_subt,MATSEQDENSE,MAT_INITIAL_MATRIX,&change_sub));
        PetscCall(MatDestroy(&change_subt));
      }
      PetscCall(KSPSetOperators(sub_schurs->change[i],change_sub,change_sub));
      PetscCall(MatDestroy(&change_sub));
      PetscCall(KSPSetOptionsPrefix(sub_schurs->change[i],sub_schurs->prefix));
      PetscCall(KSPAppendOptionsPrefix(sub_schurs->change[i],"sub_schurs_change_"));
    }
    PetscCall(ISDestroy(&change_primal_all));
  }

  /* Local matrix of all local Schur on subsets (transposed) */
  if (!sub_schurs->S_Ej_all) {
    Mat         T;
    PetscScalar *v;
    PetscInt    *ii,*jj;
    PetscInt    cum,i,j,k;

    /* MatSeqAIJSetPreallocation + MatSetValues is slow for these kind of matrices (may have large blocks)
       Allocate properly a representative matrix and duplicate */
    PetscCall(PetscMalloc3(local_size+1,&ii,local_stash_size,&jj,local_stash_size,&v));
    ii[0] = 0;
    cum   = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
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
    PetscCall(MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,local_size,local_size,ii,jj,v,&T));
    PetscCall(MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&sub_schurs->S_Ej_all));
    PetscCall(MatDestroy(&T));
    PetscCall(PetscFree3(ii,jj,v));
  }
  /* matrices for deluxe scaling and adaptive selection */
  if (compute_Stilda) {
    if (!sub_schurs->sum_S_Ej_tilda_all) {
      PetscCall(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_tilda_all));
    }
    if (!sub_schurs->sum_S_Ej_inv_all && deluxe) {
      PetscCall(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_inv_all));
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

    PetscCall(PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&work));
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      IS  is_subset_B;
      Mat AE_EE,AE_IE,AE_EI,S_Ej;

      /* subsets in original and boundary numbering */
      PetscCall(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[i],&is_subset_B));
      /* EE block */
      PetscCall(MatCreateSubMatrix(A_BB,is_subset_B,is_subset_B,MAT_INITIAL_MATRIX,&AE_EE));
      /* IE block */
      PetscCall(MatCreateSubMatrix(A_IB,is_I,is_subset_B,MAT_INITIAL_MATRIX,&AE_IE));
      /* EI block */
      if (sub_schurs->is_symmetric) {
        PetscCall(MatCreateTranspose(AE_IE,&AE_EI));
      } else if (sub_schurs->is_hermitian) {
        PetscCall(MatCreateHermitianTranspose(AE_IE,&AE_EI));
      } else {
        PetscCall(MatCreateSubMatrix(A_BI,is_subset_B,is_I,MAT_INITIAL_MATRIX,&AE_EI));
      }
      PetscCall(ISDestroy(&is_subset_B));
      PetscCall(MatCreateSchurComplement(AE_II,AE_II,AE_IE,AE_EI,AE_EE,&S_Ej));
      PetscCall(MatDestroy(&AE_EE));
      PetscCall(MatDestroy(&AE_IE));
      PetscCall(MatDestroy(&AE_EI));
      if (AE_II == A_II) { /* we can reuse the same ksp */
        KSP ksp;
        PetscCall(MatSchurComplementGetKSP(sub_schurs->S,&ksp));
        PetscCall(MatSchurComplementSetKSP(S_Ej,ksp));
      } else { /* build new ksp object which inherits ksp and pc types from the original one */
        KSP       origksp,schurksp;
        PC        origpc,schurpc;
        KSPType   ksp_type;
        PetscInt  n_internal;
        PetscBool ispcnone;

        PetscCall(MatSchurComplementGetKSP(sub_schurs->S,&origksp));
        PetscCall(MatSchurComplementGetKSP(S_Ej,&schurksp));
        PetscCall(KSPGetType(origksp,&ksp_type));
        PetscCall(KSPSetType(schurksp,ksp_type));
        PetscCall(KSPGetPC(schurksp,&schurpc));
        PetscCall(KSPGetPC(origksp,&origpc));
        PetscCall(PetscObjectTypeCompare((PetscObject)origpc,PCNONE,&ispcnone));
        if (!ispcnone) {
          PCType pc_type;
          PetscCall(PCGetType(origpc,&pc_type));
          PetscCall(PCSetType(schurpc,pc_type));
        } else {
          PetscCall(PCSetType(schurpc,PCLU));
        }
        PetscCall(ISGetSize(is_I,&n_internal));
        if (!n_internal) { /* UMFPACK gives error with 0 sized problems */
          MatSolverType solver = NULL;
          PetscCall(PCFactorGetMatSolverType(origpc,(MatSolverType*)&solver));
          if (solver) {
            PetscCall(PCFactorSetMatSolverType(schurpc,solver));
          }
        }
        PetscCall(KSPSetUp(schurksp));
      }
      PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&S_Ej_expl));
      PetscCall(PCBDDCComputeExplicitSchur(S_Ej,sub_schurs->is_symmetric,MAT_REUSE_MATRIX,&S_Ej_expl));
      PetscCall(PetscObjectTypeCompare((PetscObject)S_Ej_expl,MATSEQDENSE,&Sdense));
      if (Sdense) {
        for (j=0;j<subset_size;j++) {
          dummy_idx[j]=local_size+j;
        }
        PetscCall(MatSetValues(sub_schurs->S_Ej_all,subset_size,dummy_idx,subset_size,dummy_idx,work,INSERT_VALUES));
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not yet implemented for sparse matrices");
      PetscCall(MatDestroy(&S_Ej));
      PetscCall(MatDestroy(&S_Ej_expl));
      local_size += subset_size;
    }
    PetscCall(PetscFree2(dummy_idx,work));
    /* free */
    PetscCall(ISDestroy(&is_I));
    PetscCall(MatDestroy(&AE_II));
    PetscCall(PetscFree(all_local_idx_N));
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
      PetscCall(ISGetLocalSize(is_I_layer,&n_I));
    }
    economic = PETSC_FALSE;
    PetscCall(ISGetLocalSize(sub_schurs->is_I,&cum));
    if (cum != n_I) economic = PETSC_TRUE;
    PetscCall(MatGetLocalSize(sub_schurs->A,&n,NULL));
    size_active_schur = local_size;

    /* import scaling vector (wrong formulation if we have 3D edges) */
    if (scaling && compute_Stilda) {
      const PetscScalar *array;
      PetscScalar       *array2;
      const PetscInt    *idxs;
      PetscInt          i;

      PetscCall(ISGetIndices(sub_schurs->is_Ej_all,&idxs));
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,size_active_schur,&Dall));
      PetscCall(VecGetArrayRead(scaling,&array));
      PetscCall(VecGetArray(Dall,&array2));
      for (i=0;i<size_active_schur;i++) array2[i] = array[idxs[i]];
      PetscCall(VecRestoreArray(Dall,&array2));
      PetscCall(VecRestoreArrayRead(scaling,&array));
      PetscCall(ISRestoreIndices(sub_schurs->is_Ej_all,&idxs));
      deluxe = PETSC_FALSE;
    }

    /* size active schurs does not count any dirichlet or vertex dof on the interface */
    factor_workaround = PETSC_FALSE;
    schur_has_vertices = PETSC_FALSE;
    cum = n_I+size_active_schur;
    if (sub_schurs->is_dir) {
      const PetscInt* idxs;
      PetscInt        n_dir;

      PetscCall(ISGetLocalSize(sub_schurs->is_dir,&n_dir));
      PetscCall(ISGetIndices(sub_schurs->is_dir,&idxs));
      PetscCall(PetscArraycpy(all_local_idx_N+cum,idxs,n_dir));
      PetscCall(ISRestoreIndices(sub_schurs->is_dir,&idxs));
      cum += n_dir;
      factor_workaround = PETSC_TRUE;
    }
    /* include the primal vertices in the Schur complement */
    if (exact_schur && sub_schurs->is_vertices && (compute_Stilda || benign_n)) {
      PetscInt n_v;

      PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&n_v));
      if (n_v) {
        const PetscInt* idxs;

        PetscCall(ISGetIndices(sub_schurs->is_vertices,&idxs));
        PetscCall(PetscArraycpy(all_local_idx_N+cum,idxs,n_v));
        PetscCall(ISRestoreIndices(sub_schurs->is_vertices,&idxs));
        cum += n_v;
        factor_workaround = PETSC_TRUE;
        schur_has_vertices = PETSC_TRUE;
      }
    }
    size_schur = cum - n_I;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,cum,all_local_idx_N,PETSC_OWN_POINTER,&is_A_all));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = sub_schurs->A->boundtocpu;
    PetscCall(MatBindToCPU(sub_schurs->A,PETSC_TRUE));
#endif
    if (cum == n) {
      PetscCall(ISSetPermutation(is_A_all));
      PetscCall(MatPermute(sub_schurs->A,is_A_all,is_A_all,&A));
    } else {
      PetscCall(MatCreateSubMatrix(sub_schurs->A,is_A_all,is_A_all,MAT_INITIAL_MATRIX,&A));
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(sub_schurs->A,oldpin));
#endif
    PetscCall(MatSetOptionsPrefix(A,sub_schurs->prefix));
    PetscCall(MatAppendOptionsPrefix(A,"sub_schurs_"));

    /* if we actually change the basis for the pressures, LDL^T factors will use a lot of memory
       this is a workaround */
    if (benign_n) {
      Vec                    v,benign_AIIm1_ones;
      ISLocalToGlobalMapping N_to_reor;
      IS                     is_p0,is_p0_p;
      PetscScalar            *cs_AIB,*AIIm1_data;
      PetscInt               sizeA;

      PetscCall(ISLocalToGlobalMappingCreateIS(is_A_all,&N_to_reor));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF,benign_n,benign_p0_lidx,PETSC_COPY_VALUES,&is_p0));
      PetscCall(ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,is_p0,&is_p0_p));
      PetscCall(ISDestroy(&is_p0));
      PetscCall(MatCreateVecs(A,&v,&benign_AIIm1_ones));
      PetscCall(VecGetSize(v,&sizeA));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,sizeA,benign_n,NULL,&benign_AIIm1_ones_mat));
      PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,size_schur,benign_n,NULL,&cs_AIB_mat));
      PetscCall(MatDenseGetArray(cs_AIB_mat,&cs_AIB));
      PetscCall(MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data));
      PetscCall(PetscMalloc1(benign_n,&is_p_r));
      /* compute colsum of A_IB restricted to pressures */
      for (i=0;i<benign_n;i++) {
        const PetscScalar *array;
        const PetscInt    *idxs;
        PetscInt          j,nz;

        PetscCall(ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,benign_zerodiag_subs[i],&is_p_r[i]));
        PetscCall(ISGetLocalSize(is_p_r[i],&nz));
        PetscCall(ISGetIndices(is_p_r[i],&idxs));
        for (j=0;j<nz;j++) AIIm1_data[idxs[j]+sizeA*i] = 1.;
        PetscCall(ISRestoreIndices(is_p_r[i],&idxs));
        PetscCall(VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i));
        PetscCall(MatMult(A,benign_AIIm1_ones,v));
        PetscCall(VecResetArray(benign_AIIm1_ones));
        PetscCall(VecGetArrayRead(v,&array));
        for (j=0;j<size_schur;j++) {
#if defined(PETSC_USE_COMPLEX)
          cs_AIB[i*size_schur + j] = (PetscRealPart(array[j+n_I])/nz + PETSC_i*(PetscImaginaryPart(array[j+n_I])/nz));
#else
          cs_AIB[i*size_schur + j] = array[j+n_I]/nz;
#endif
        }
        PetscCall(VecRestoreArrayRead(v,&array));
      }
      PetscCall(MatDenseRestoreArray(cs_AIB_mat,&cs_AIB));
      PetscCall(MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data));
      PetscCall(VecDestroy(&v));
      PetscCall(VecDestroy(&benign_AIIm1_ones));
      PetscCall(MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE));
      PetscCall(MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE));
      PetscCall(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE));
      PetscCall(MatZeroRowsColumnsIS(A,is_p0_p,1.0,NULL,NULL));
      PetscCall(ISDestroy(&is_p0_p));
      PetscCall(ISLocalToGlobalMappingDestroy(&N_to_reor));
    }
    PetscCall(MatSetOption(A,MAT_SYMMETRIC,sub_schurs->is_symmetric));
    PetscCall(MatSetOption(A,MAT_HERMITIAN,sub_schurs->is_hermitian));
    PetscCall(MatSetOption(A,MAT_SPD,sub_schurs->is_posdef));

    /* for complexes, symmetric and hermitian at the same time implies null imaginary part */
    use_cholesky = (PetscBool)((use_potr || use_sytr) && sub_schurs->is_hermitian && sub_schurs->is_symmetric);

    /* when using the benign subspace trick, the local Schur complements are SPD */
    /* MKL_PARDISO does not handle well the computation of a Schur complement from a symmetric indefinite factorization
       Use LU and adapt pivoting perturbation (still, solution is not as accurate as with using MUMPS) */
    if (benign_trick) {
      sub_schurs->is_posdef = PETSC_TRUE;
      PetscCall(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,&flg));
      if (flg) use_cholesky = PETSC_FALSE;
    }

    if (n_I) {
      IS        is_schur;
      char      stype[64];
      PetscBool gpu = PETSC_FALSE;

      if (use_cholesky) {
        PetscCall(MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_CHOLESKY,&F));
      } else {
        PetscCall(MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_LU,&F));
      }
      PetscCall(MatSetErrorIfFailure(A,PETSC_TRUE));
#if defined(PETSC_HAVE_MKL_PARDISO)
      if (benign_trick) PetscCall(MatMkl_PardisoSetCntl(F,10,10));
#endif
      /* subsets ordered last */
      PetscCall(ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is_schur));
      PetscCall(MatFactorSetSchurIS(F,is_schur));
      PetscCall(ISDestroy(&is_schur));

      /* factorization step */
      if (use_cholesky) {
        PetscCall(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        PetscCall(MatMumpsSetIcntl(F,19,2));
#endif
        PetscCall(MatCholeskyFactorNumeric(F,A,NULL));
        S_lower_triangular = PETSC_TRUE;
      } else {
        PetscCall(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        PetscCall(MatMumpsSetIcntl(F,19,3));
#endif
        PetscCall(MatLUFactorNumeric(F,A,NULL));
      }
      PetscCall(MatViewFromOptions(F,(PetscObject)A,"-mat_factor_view"));

      if (matl_dbg_viewer) {
        Mat S;
        IS  is;

        PetscCall(PetscObjectSetName((PetscObject)A,"A"));
        PetscCall(MatView(A,matl_dbg_viewer));
        PetscCall(MatFactorCreateSchurComplement(F,&S,NULL));
        PetscCall(PetscObjectSetName((PetscObject)S,"S"));
        PetscCall(MatView(S,matl_dbg_viewer));
        PetscCall(MatDestroy(&S));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,n_I,0,1,&is));
        PetscCall(PetscObjectSetName((PetscObject)is,"I"));
        PetscCall(ISView(is,matl_dbg_viewer));
        PetscCall(ISDestroy(&is));
        PetscCall(ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is));
        PetscCall(PetscObjectSetName((PetscObject)is,"B"));
        PetscCall(ISView(is,matl_dbg_viewer));
        PetscCall(ISDestroy(&is));
        PetscCall(PetscObjectSetName((PetscObject)is_A_all,"IA"));
        PetscCall(ISView(is_A_all,matl_dbg_viewer));
      }

      /* get explicit Schur Complement computed during numeric factorization */
      PetscCall(MatFactorGetSchurComplement(F,&S_all,NULL));
      PetscCall(PetscStrncpy(stype,MATSEQDENSE,sizeof(stype)));
#if defined(PETSC_HAVE_CUDA)
      PetscCall(PetscObjectTypeCompareAny((PetscObject)A,&gpu,MATSEQAIJVIENNACL,MATSEQAIJCUSPARSE,""));
#endif
      if (gpu) {
        PetscCall(PetscStrncpy(stype,MATSEQDENSECUDA,sizeof(stype)));
      }
      PetscCall(PetscOptionsGetString(NULL,sub_schurs->prefix,"-sub_schurs_schur_mat_type",stype,sizeof(stype),NULL));
      PetscCall(MatConvert(S_all,stype,MAT_INPLACE_MATRIX,&S_all));
      PetscCall(MatSetOption(S_all,MAT_SPD,sub_schurs->is_posdef));
      PetscCall(MatSetOption(S_all,MAT_HERMITIAN,sub_schurs->is_hermitian));
      PetscCall(MatGetType(S_all,&Stype));

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

        PetscCall(MatDenseGetArray(S_all,&S_data));
        PetscCall(MatCreateVecs(A,&v,&benign_AIIm1_ones));
        PetscCall(VecGetSize(v,&sizeA));
#if defined(PETSC_HAVE_MUMPS)
        PetscCall(MatMumpsSetIcntl(F,26,0));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        PetscCall(MatMkl_PardisoSetCntl(F,70,1));
#endif
        PetscCall(MatDenseGetArrayRead(cs_AIB_mat,&cs_AIB));
        PetscCall(MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data));
        if (matl_dbg_viewer) {
          PetscCall(MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S2));
          PetscCall(MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S3));
          PetscCall(MatDenseGetArray(S2,&S2_data));
          PetscCall(MatDenseGetArray(S3,&S3_data));
        }
        for (i=0;i<benign_n;i++) {
          PetscScalar    *array,sum = 0.,one = 1.,*sums;
          const PetscInt *idxs;
          PetscInt       k,j,nz;
          PetscBLASInt   B_k,B_n;

          PetscCall(PetscCalloc1(benign_n,&sums));
          PetscCall(VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i));
          PetscCall(VecCopy(benign_AIIm1_ones,v));
          PetscCall(MatSolve(F,v,benign_AIIm1_ones));
          PetscCall(MatMult(A,benign_AIIm1_ones,v));
          PetscCall(VecResetArray(benign_AIIm1_ones));
          /* p0 dofs (eliminated) are excluded from the sums */
          for (k=0;k<benign_n;k++) {
            PetscCall(ISGetLocalSize(is_p_r[k],&nz));
            PetscCall(ISGetIndices(is_p_r[k],&idxs));
            for (j=0;j<nz-1;j++) sums[k] -= AIIm1_data[idxs[j]+sizeA*i];
            PetscCall(ISRestoreIndices(is_p_r[k],&idxs));
          }
          PetscCall(VecGetArrayRead(v,(const PetscScalar**)&array));
          if (matl_dbg_viewer) {
            Vec  vv;
            char name[16];

            PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array+n_I,&vv));
            PetscCall(PetscSNPrintf(name,sizeof(name),"Pvs%D",i));
            PetscCall(PetscObjectSetName((PetscObject)vv,name));
            PetscCall(VecView(vv,matl_dbg_viewer));
          }
          /* perform sparse rank updates on symmetric Schur (TODO: move outside of the loop?) */
          /* cs_AIB already scaled by 1./nz */
          B_k = 1;
          for (k=0;k<benign_n;k++) {
            sum  = sums[k];
            PetscCall(PetscBLASIntCast(size_schur,&B_n));

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
          PetscCall(VecRestoreArrayRead(v,(const PetscScalar**)&array));
          /* set p0 entry of AIIm1_ones to zero */
          PetscCall(ISGetLocalSize(is_p_r[i],&nz));
          PetscCall(ISGetIndices(is_p_r[i],&idxs));
          for (j=0;j<benign_n;j++) AIIm1_data[idxs[nz-1]+sizeA*j] = 0.;
          PetscCall(ISRestoreIndices(is_p_r[i],&idxs));
          PetscCall(PetscFree(sums));
        }
        PetscCall(VecDestroy(&benign_AIIm1_ones));
        if (matl_dbg_viewer) {
          PetscCall(MatDenseRestoreArray(S2,&S2_data));
          PetscCall(MatDenseRestoreArray(S3,&S3_data));
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
        PetscCall(MatMumpsSetIcntl(F,26,-1));
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        PetscCall(MatMkl_PardisoSetCntl(F,70,0));
#endif
        PetscCall(MatDenseRestoreArrayRead(cs_AIB_mat,&cs_AIB));
        PetscCall(MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data));
        PetscCall(VecDestroy(&v));
        PetscCall(MatDenseRestoreArray(S_all,&S_data));
        if (matl_dbg_viewer) {
          Mat S;

          PetscCall(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
          PetscCall(MatFactorCreateSchurComplement(F,&S,NULL));
          PetscCall(PetscObjectSetName((PetscObject)S,"Sb"));
          PetscCall(MatView(S,matl_dbg_viewer));
          PetscCall(MatDestroy(&S));
          PetscCall(PetscObjectSetName((PetscObject)S2,"S2P"));
          PetscCall(MatView(S2,matl_dbg_viewer));
          PetscCall(PetscObjectSetName((PetscObject)S3,"S3P"));
          PetscCall(MatView(S3,matl_dbg_viewer));
          PetscCall(PetscObjectSetName((PetscObject)cs_AIB_mat,"cs"));
          PetscCall(MatView(cs_AIB_mat,matl_dbg_viewer));
          PetscCall(MatFactorGetSchurComplement(F,&S_all,NULL));
        }
        PetscCall(MatDestroy(&S2));
        PetscCall(MatDestroy(&S3));
      }
      if (!reuse_solvers) {
        for (i=0;i<benign_n;i++) {
          PetscCall(ISDestroy(&is_p_r[i]));
        }
        PetscCall(PetscFree(is_p_r));
        PetscCall(MatDestroy(&cs_AIB_mat));
        PetscCall(MatDestroy(&benign_AIIm1_ones_mat));
      }
    } else { /* we can't use MatFactor when size_schur == size_of_the_problem */
      PetscCall(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&S_all));
      PetscCall(MatGetType(S_all,&Stype));
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
        PetscCall(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
      } else {
        PetscCall(PetscNew(&sub_schurs->reuse_solver));
      }
      msolv_ctx = sub_schurs->reuse_solver;
      PetscCall(MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,NULL,NULL,NULL));
      PetscCall(PetscObjectReference((PetscObject)F));
      msolv_ctx->F = F;
      PetscCall(MatCreateVecs(F,&msolv_ctx->sol,NULL));
      /* currently PETSc has no support for MatSolve(F,x,x), so cheat and let rhs and sol share the same memory */
      {
        PetscScalar *array;
        PetscInt    n;

        PetscCall(VecGetLocalSize(msolv_ctx->sol,&n));
        PetscCall(VecGetArray(msolv_ctx->sol,&array));
        PetscCall(VecCreateSeqWithArray(PetscObjectComm((PetscObject)msolv_ctx->sol),1,n,array,&msolv_ctx->rhs));
        PetscCall(VecRestoreArray(msolv_ctx->sol,&array));
      }
      msolv_ctx->has_vertices = schur_has_vertices;

      /* interior solver */
      PetscCall(PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->interior_solver));
      PetscCall(PCSetOperators(msolv_ctx->interior_solver,A_II,A_II));
      PetscCall(PCSetType(msolv_ctx->interior_solver,PCSHELL));
      PetscCall(PCShellSetName(msolv_ctx->interior_solver,"Interior solver (w/o Schur factorization)"));
      PetscCall(PCShellSetContext(msolv_ctx->interior_solver,msolv_ctx));
      PetscCall(PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View));
      PetscCall(PCShellSetApply(msolv_ctx->interior_solver,PCBDDCReuseSolvers_Interior));
      PetscCall(PCShellSetApplyTranspose(msolv_ctx->interior_solver,PCBDDCReuseSolvers_InteriorTranspose));

      /* correction solver */
      PetscCall(PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->correction_solver));
      PetscCall(PCSetType(msolv_ctx->correction_solver,PCSHELL));
      PetscCall(PCShellSetName(msolv_ctx->correction_solver,"Correction solver (with Schur factorization)"));
      PetscCall(PCShellSetContext(msolv_ctx->correction_solver,msolv_ctx));
      PetscCall(PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View));
      PetscCall(PCShellSetApply(msolv_ctx->correction_solver,PCBDDCReuseSolvers_Correction));
      PetscCall(PCShellSetApplyTranspose(msolv_ctx->correction_solver,PCBDDCReuseSolvers_CorrectionTranspose));

      /* scatter and vecs for Schur complement solver */
      PetscCall(MatCreateVecs(S_all,&msolv_ctx->sol_B,&msolv_ctx->rhs_B));
      PetscCall(MatCreateVecs(sub_schurs->S,&vec1_B,NULL));
      if (!schur_has_vertices) {
        PetscCall(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_A_all,&msolv_ctx->is_B));
        PetscCall(VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,NULL,&msolv_ctx->correction_scatter_B));
        PetscCall(PetscObjectReference((PetscObject)is_A_all));
        msolv_ctx->is_R = is_A_all;
      } else {
        IS              is_B_all;
        const PetscInt* idxs;
        PetscInt        dual,n_v,n;

        PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&n_v));
        dual = size_schur - n_v;
        PetscCall(ISGetLocalSize(is_A_all,&n));
        PetscCall(ISGetIndices(is_A_all,&idxs));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),dual,idxs+n_I,PETSC_COPY_VALUES,&is_B_all));
        PetscCall(ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_B_all,&msolv_ctx->is_B));
        PetscCall(ISDestroy(&is_B_all));
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)is_A_all),dual,0,1,&is_B_all));
        PetscCall(VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,is_B_all,&msolv_ctx->correction_scatter_B));
        PetscCall(ISDestroy(&is_B_all));
        PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),n-n_v,idxs,PETSC_COPY_VALUES,&msolv_ctx->is_R));
        PetscCall(ISRestoreIndices(is_A_all,&idxs));
      }
      PetscCall(ISGetLocalSize(msolv_ctx->is_R,&n_R));
      PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,n_R,n_R,0,NULL,&Afake));
      PetscCall(MatAssemblyBegin(Afake,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(Afake,MAT_FINAL_ASSEMBLY));
      PetscCall(PCSetOperators(msolv_ctx->correction_solver,Afake,Afake));
      PetscCall(MatDestroy(&Afake));
      PetscCall(VecDestroy(&vec1_B));

      /* communicate benign info to solver context */
      if (benign_n) {
        PetscScalar *array;

        msolv_ctx->benign_n = benign_n;
        msolv_ctx->benign_zerodiag_subs = is_p_r;
        PetscCall(PetscMalloc1(benign_n,&msolv_ctx->benign_save_vals));
        msolv_ctx->benign_csAIB = cs_AIB_mat;
        PetscCall(MatCreateVecs(cs_AIB_mat,&msolv_ctx->benign_corr_work,NULL));
        PetscCall(VecGetArray(msolv_ctx->benign_corr_work,&array));
        PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array,&msolv_ctx->benign_dummy_schur_vec));
        PetscCall(VecRestoreArray(msolv_ctx->benign_corr_work,&array));
        msolv_ctx->benign_AIIm1ones = benign_AIIm1_ones_mat;
      }
    } else {
      if (sub_schurs->reuse_solver) {
        PetscCall(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
      }
      PetscCall(PetscFree(sub_schurs->reuse_solver));
    }
    PetscCall(MatDestroy(&A));
    PetscCall(ISDestroy(&is_A_all));

    /* Work arrays */
    PetscCall(PetscMalloc1(max_subset_size*max_subset_size,&work));

    /* S_Ej_all */
    cum = cum2 = 0;
    PetscCall(MatDenseGetArrayRead(S_all,&rS_data));
    PetscCall(MatSeqAIJGetArray(sub_schurs->S_Ej_all,&SEj_arr));
    if (compute_Stilda) {
      PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr));
    }
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscInt j;

      /* get S_E */
      PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
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
        PetscCall(KSPGetOperators(sub_schurs->change[i],&change_sub,NULL));
        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
        if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
          Mat T2;
          PetscCall(MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2));
          PetscCall(MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
          PetscCall(MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T));
          PetscCall(MatDestroy(&T2));
        } else {
          PetscCall(MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
        }
        PetscCall(MatCopy(T,SEj,SAME_NONZERO_PATTERN));
        PetscCall(MatDestroy(&T));
        PetscCall(MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1.0,NULL,NULL));
        PetscCall(MatDestroy(&SEj));
      }
      if (deluxe) {
        PetscCall(PetscArraycpy(SEj_arr,work,subset_size*subset_size));
        /* if adaptivity is requested, invert S_E blocks */
        if (compute_Stilda) {
          Mat               M;
          const PetscScalar *vals;
          PetscBool         isdense,isdensecuda;

          PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&M));
          PetscCall(MatSetOption(M,MAT_SPD,sub_schurs->is_posdef));
          PetscCall(MatSetOption(M,MAT_HERMITIAN,sub_schurs->is_hermitian));
          if (!PetscBTLookup(sub_schurs->is_edge,i)) {
            PetscCall(MatSetType(M,Stype));
          }
          PetscCall(PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&isdense));
          PetscCall(PetscObjectTypeCompare((PetscObject)M,MATSEQDENSECUDA,&isdensecuda));
          if (use_cholesky) {
            PetscCall(MatCholeskyFactor(M,NULL,NULL));
          } else {
            PetscCall(MatLUFactor(M,NULL,NULL,NULL));
          }
          if (isdense) {
            PetscCall(MatSeqDenseInvertFactors_Private(M));
#if defined(PETSC_HAVE_CUDA)
          } else if (isdensecuda) {
            PetscCall(MatSeqDenseCUDAInvertFactors_Private(M));
#endif
          } else SETERRQ(PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Not implemented for type %s",Stype);
          PetscCall(MatDenseGetArrayRead(M,&vals));
          PetscCall(PetscArraycpy(SEjinv_arr,vals,subset_size*subset_size));
          PetscCall(MatDenseRestoreArrayRead(M,&vals));
          PetscCall(MatDestroy(&M));
        }
      } else if (compute_Stilda) { /* not using deluxe */
        Mat         SEj;
        Vec         D;
        PetscScalar *array;

        PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
        PetscCall(VecGetArray(Dall,&array));
        PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF,1,subset_size,array+cum,&D));
        PetscCall(VecRestoreArray(Dall,&array));
        PetscCall(VecShift(D,-1.));
        PetscCall(MatDiagonalScale(SEj,D,D));
        PetscCall(MatDestroy(&SEj));
        PetscCall(VecDestroy(&D));
        PetscCall(PetscArraycpy(SEj_arr,work,subset_size*subset_size));
      }
      cum += subset_size;
      cum2 += subset_size*(size_schur + 1);
      SEj_arr += subset_size*subset_size;
      if (SEjinv_arr) SEjinv_arr += subset_size*subset_size;
    }
    PetscCall(MatDenseRestoreArrayRead(S_all,&rS_data));
    PetscCall(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&SEj_arr));
    if (compute_Stilda) {
      PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr));
    }
    if (solver_S) {
      PetscCall(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
    }

    /* may prevent from unneeded copies, since MUMPS or MKL_Pardiso always use CPU memory
       however, preliminary tests indicate using GPUs is still faster in the solve phase */
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    if (reuse_solvers) {
      Mat                  St;
      MatFactorSchurStatus st;

      flg  = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL,sub_schurs->prefix,"-sub_schurs_schur_pin_to_cpu",&flg,NULL));
      PetscCall(MatFactorGetSchurComplement(F,&St,&st));
      PetscCall(MatBindToCPU(St,flg));
      PetscCall(MatFactorRestoreSchurComplement(F,&St,st));
    }
#endif

    schur_factor = NULL;
    if (compute_Stilda && size_active_schur) {

      PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr));
      if (sub_schurs->n_subs == 1 && size_schur == size_active_schur && deluxe) { /* we already computed the inverse */
        PetscCall(PetscArraycpy(SEjinv_arr,work,size_schur*size_schur));
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
            PetscCall(MatFactorGetSchurComplement(F,&S_all_inv,NULL));
            PetscCall(MatDenseGetArray(S_all_inv,&data));
            if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
              PetscCall(ISGetLocalSize(sub_schurs->is_dir,&nd));
            }

            /* factor and invert activedofs and vertices (dirichlet dofs does not contribute) */
            if (schur_has_vertices) {
              Mat          M;
              PetscScalar *tdata;
              PetscInt     nv = 0, news;

              PetscCall(ISGetLocalSize(sub_schurs->is_vertices,&nv));
              news = size_active_schur + nv;
              PetscCall(PetscCalloc1(news*news,&tdata));
              for (i=0;i<size_active_schur;i++) {
                PetscCall(PetscArraycpy(tdata+i*(news+1),data+i*(size_schur+1),size_active_schur-i));
                PetscCall(PetscArraycpy(tdata+i*(news+1)+size_active_schur-i,data+i*size_schur+size_active_schur+nd,nv));
              }
              for (i=0;i<nv;i++) {
                PetscInt k = i+size_active_schur;
                PetscCall(PetscArraycpy(tdata+k*(news+1),data+(k+nd)*(size_schur+1),nv-i));
              }

              PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,news,news,tdata,&M));
              PetscCall(MatSetOption(M,MAT_SPD,PETSC_TRUE));
              PetscCall(MatCholeskyFactor(M,NULL,NULL));
              /* save the factors */
              cum = 0;
              PetscCall(PetscMalloc1((size_active_schur*(size_active_schur +1))/2+nd,&schur_factor));
              for (i=0;i<size_active_schur;i++) {
                PetscCall(PetscArraycpy(schur_factor+cum,tdata+i*(news+1),size_active_schur-i));
                cum += size_active_schur - i;
              }
              for (i=0;i<nd;i++) schur_factor[cum+i] = PetscSqrtReal(PetscRealPart(data[(i+size_active_schur)*(size_schur+1)]));
              PetscCall(MatSeqDenseInvertFactors_Private(M));
              /* move back just the active dofs to the Schur complement */
              for (i=0;i<size_active_schur;i++) {
                PetscCall(PetscArraycpy(data+i*size_schur,tdata+i*news,size_active_schur));
              }
              PetscCall(PetscFree(tdata));
              PetscCall(MatDestroy(&M));
            } else { /* we can factorize and invert just the activedofs */
              Mat         M;
              PetscScalar *aux;

              PetscCall(PetscMalloc1(nd,&aux));
              for (i=0;i<nd;i++) aux[i] = 1.0/data[(i+size_active_schur)*(size_schur+1)];
              PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,size_active_schur,size_active_schur,data,&M));
              PetscCall(MatDenseSetLDA(M,size_schur));
              PetscCall(MatSetOption(M,MAT_SPD,PETSC_TRUE));
              PetscCall(MatCholeskyFactor(M,NULL,NULL));
              PetscCall(MatSeqDenseInvertFactors_Private(M));
              PetscCall(MatDestroy(&M));
              PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,size_schur,nd,data+size_active_schur*size_schur,&M));
              PetscCall(MatZeroEntries(M));
              PetscCall(MatDestroy(&M));
              PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,nd,size_schur,data+size_active_schur,&M));
              PetscCall(MatDenseSetLDA(M,size_schur));
              PetscCall(MatZeroEntries(M));
              PetscCall(MatDestroy(&M));
              for (i=0;i<nd;i++) data[(i+size_active_schur)*(size_schur+1)] = aux[i];
              PetscCall(PetscFree(aux));
            }
            PetscCall(MatDenseRestoreArray(S_all_inv,&data));
          } else { /* use MatFactor calls to invert S */
            PetscCall(MatFactorInvertSchurComplement(F));
            PetscCall(MatFactorGetSchurComplement(F,&S_all_inv,NULL));
          }
        } else { /* we need to invert explicitly since we are not using MatFactor for S */
          PetscCall(PetscObjectReference((PetscObject)S_all));
          S_all_inv = S_all;
          PetscCall(MatDenseGetArray(S_all_inv,&S_data));
          PetscCall(PetscBLASIntCast(size_schur,&B_N));
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
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
          PetscCall(PetscLogFlops(1.0*size_schur*size_schur*size_schur));
          PetscCall(PetscFPTrapPop());
          PetscCall(MatDenseRestoreArray(S_all_inv,&S_data));
        }
        /* S_Ej_tilda_all */
        cum = cum2 = 0;
        PetscCall(MatDenseGetArrayRead(S_all_inv,&rS_data));
        for (i=0;i<sub_schurs->n_subs;i++) {
          PetscInt j;

          PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
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
            PetscCall(KSPGetOperators(sub_schurs->change[i],&change_sub,NULL));
            PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj));
            if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
              Mat T2;
              PetscCall(MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2));
              PetscCall(MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
              PetscCall(MatDestroy(&T2));
              PetscCall(MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T));
            } else {
              PetscCall(MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T));
            }
            PetscCall(MatCopy(T,SEj,SAME_NONZERO_PATTERN));
            PetscCall(MatDestroy(&T));
            /* set diagonal entry to a very large value to pick the basis we are eliminating as the first eigenvectors with adaptive selection */
            PetscCall(MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1./PETSC_SMALL,NULL,NULL));
            PetscCall(MatDestroy(&SEj));
          }
          PetscCall(PetscArraycpy(SEjinv_arr,work,subset_size*subset_size));
          cum += subset_size;
          cum2 += subset_size*(size_schur + 1);
          SEjinv_arr += subset_size*subset_size;
        }
        PetscCall(MatDenseRestoreArrayRead(S_all_inv,&rS_data));
        if (solver_S) {
          if (schur_has_vertices) {
            PetscCall(MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_FACTORED));
          } else {
            PetscCall(MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_INVERTED));
          }
        }
        PetscCall(MatDestroy(&S_all_inv));
      }
      PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr));

      /* move back factors if needed */
      if (schur_has_vertices) {
        Mat      S_tmp;
        PetscInt nd = 0;

        PetscCheck(solver_S,PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        PetscCall(MatFactorGetSchurComplement(F,&S_tmp,NULL));
        if (use_potr) {
          PetscScalar *data;

          PetscCall(MatDenseGetArray(S_tmp,&data));
          PetscCall(PetscArrayzero(data,size_schur*size_schur));

          if (S_lower_triangular) {
            cum = 0;
            for (i=0;i<size_active_schur;i++) {
              PetscCall(PetscArraycpy(data+i*(size_schur+1),schur_factor+cum,size_active_schur-i));
              cum += size_active_schur-i;
            }
          } else {
            PetscCall(PetscArraycpy(data,schur_factor,size_schur*size_schur));
          }
          if (sub_schurs->is_dir) {
            PetscCall(ISGetLocalSize(sub_schurs->is_dir,&nd));
            for (i=0;i<nd;i++) {
              data[(i+size_active_schur)*(size_schur+1)] = schur_factor[cum+i];
            }
          }
          /* workaround: since I cannot modify the matrices used inside the solvers for the forward and backward substitutions,
             set the diagonal entry of the Schur factor to a very large value */
          for (i=size_active_schur+nd;i<size_schur;i++) {
            data[i*(size_schur+1)] = infty;
          }
          PetscCall(MatDenseRestoreArray(S_tmp,&data));
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor update not yet implemented for non SPD matrices");
        PetscCall(MatFactorRestoreSchurComplement(F,&S_tmp,MAT_FACTOR_SCHUR_FACTORED));
      }
    } else if (factor_workaround) { /* we need to eliminate any unneeded coupling */
      PetscScalar *data;
      PetscInt    nd = 0;

      if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
        PetscCall(ISGetLocalSize(sub_schurs->is_dir,&nd));
      }
      PetscCall(MatFactorGetSchurComplement(F,&S_all,NULL));
      PetscCall(MatDenseGetArray(S_all,&data));
      for (i=0;i<size_active_schur;i++) {
        PetscCall(PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur));
      }
      for (i=size_active_schur+nd;i<size_schur;i++) {
        PetscCall(PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur));
        data[i*(size_schur+1)] = infty;
      }
      PetscCall(MatDenseRestoreArray(S_all,&data));
      PetscCall(MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED));
    }
    PetscCall(PetscFree(work));
    PetscCall(PetscFree(schur_factor));
    PetscCall(VecDestroy(&Dall));
  }
  PetscCall(ISDestroy(&is_I_layer));
  PetscCall(MatDestroy(&S_all));
  PetscCall(MatDestroy(&A_BB));
  PetscCall(MatDestroy(&A_IB));
  PetscCall(MatDestroy(&A_BI));
  PetscCall(MatDestroy(&F));

  PetscCall(PetscMalloc1(sub_schurs->n_subs,&nnz));
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&nnz[i]));
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,sub_schurs->n_subs,nnz,PETSC_OWN_POINTER,&is_I_layer));
  PetscCall(MatSetVariableBlockSizes(sub_schurs->S_Ej_all,sub_schurs->n_subs,nnz));
  PetscCall(MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY));
  if (compute_Stilda) {
    PetscCall(MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_tilda_all,sub_schurs->n_subs,nnz));
    PetscCall(MatAssemblyBegin(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY));
    if (deluxe) {
      PetscCall(MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_inv_all,sub_schurs->n_subs,nnz));
      PetscCall(MatAssemblyBegin(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY));
    }
  }
  PetscCall(ISDestroy(&is_I_layer));

  /* Get local part of (\sum_j S_Ej) */
  if (!sub_schurs->sum_S_Ej_all) {
    PetscCall(MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_all));
  }
  PetscCall(VecSet(gstash,0.0));
  PetscCall(MatSeqAIJGetArray(sub_schurs->S_Ej_all,&stasharray));
  PetscCall(VecPlaceArray(lstash,stasharray));
  PetscCall(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&stasharray));
  PetscCall(VecResetArray(lstash));
  PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all,&stasharray));
  PetscCall(VecPlaceArray(lstash,stasharray));
  PetscCall(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
  PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all,&stasharray));
  PetscCall(VecResetArray(lstash));

  /* Get local part of (\sum_j S^-1_Ej) (\sum_j St^-1_Ej) */
  if (compute_Stilda) {
    PetscCall(VecSet(gstash,0.0));
    PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray));
    PetscCall(VecPlaceArray(lstash,stasharray));
    PetscCall(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
    PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray));
    PetscCall(VecResetArray(lstash));
    if (deluxe) {
      PetscCall(VecSet(gstash,0.0));
      PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&stasharray));
      PetscCall(VecPlaceArray(lstash,stasharray));
      PetscCall(VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD));
      PetscCall(VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE));
      PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&stasharray));
      PetscCall(VecResetArray(lstash));
    } else {
      PetscScalar *array;
      PetscInt    cum;

      PetscCall(MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&array));
      cum = 0;
      for (i=0;i<sub_schurs->n_subs;i++) {
        PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
        PetscCall(PetscBLASIntCast(subset_size,&B_N));
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
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
        PetscCall(PetscLogFlops(1.0*subset_size*subset_size*subset_size));
        PetscCall(PetscFPTrapPop());
        cum += subset_size*subset_size;
      }
      PetscCall(MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&array));
      PetscCall(PetscObjectReference((PetscObject)sub_schurs->sum_S_Ej_all));
      PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
      sub_schurs->sum_S_Ej_inv_all = sub_schurs->sum_S_Ej_all;
    }
  }
  PetscCall(VecDestroy(&lstash));
  PetscCall(VecDestroy(&gstash));
  PetscCall(VecScatterDestroy(&sstash));

  if (matl_dbg_viewer) {
    PetscInt cum;

    if (sub_schurs->S_Ej_all) {
      PetscCall(PetscObjectSetName((PetscObject)sub_schurs->S_Ej_all,"SE"));
      PetscCall(MatView(sub_schurs->S_Ej_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_all) {
      PetscCall(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_all,"SSE"));
      PetscCall(MatView(sub_schurs->sum_S_Ej_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_inv_all) {
      PetscCall(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_inv_all,"SSEm"));
      PetscCall(MatView(sub_schurs->sum_S_Ej_inv_all,matl_dbg_viewer));
    }
    if (sub_schurs->sum_S_Ej_tilda_all) {
      PetscCall(PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_tilda_all,"SSEt"));
      PetscCall(MatView(sub_schurs->sum_S_Ej_tilda_all,matl_dbg_viewer));
    }
    for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
      IS   is;
      char name[16];

      PetscCall(PetscSNPrintf(name,sizeof(name),"IE%D",i));
      PetscCall(ISGetLocalSize(sub_schurs->is_subs[i],&subset_size));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,subset_size,cum,1,&is));
      PetscCall(PetscObjectSetName((PetscObject)is,name));
      PetscCall(ISView(is,matl_dbg_viewer));
      PetscCall(ISDestroy(&is));
      cum += subset_size;
    }
  }

  /* free workspace */
  if (matl_dbg_viewer) PetscCall(PetscViewerFlush(matl_dbg_viewer));
  if (sub_schurs->debug) PetscCallMPI(MPI_Barrier(comm_n));
  PetscCall(PetscViewerDestroy(&matl_dbg_viewer));
  PetscCall(PetscFree2(Bwork,pivots));
  PetscCall(PetscCommDestroy(&comm_n));
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, const char* prefix, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap, PetscBool copycc)
{
  IS              *faces,*edges,*all_cc,vertices;
  PetscInt        i,n_faces,n_edges,n_all_cc;
  PetscBool       is_sorted,ispardiso,ismumps;

  PetscFunctionBegin;
  PetscCall(ISSorted(is_I,&is_sorted));
  PetscCheck(is_sorted,PetscObjectComm((PetscObject)is_I),PETSC_ERR_PLIB,"IS for I dofs should be shorted");
  PetscCall(ISSorted(is_B,&is_sorted));
  PetscCheck(is_sorted,PetscObjectComm((PetscObject)is_B),PETSC_ERR_PLIB,"IS for B dofs should be shorted");

  /* reset any previous data */
  PetscCall(PCBDDCSubSchursReset(sub_schurs));

  /* get index sets for faces and edges (already sorted by global ordering) */
  PetscCall(PCBDDCGraphGetCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices));
  n_all_cc = n_faces+n_edges;
  PetscCall(PetscBTCreate(n_all_cc,&sub_schurs->is_edge));
  PetscCall(PetscMalloc1(n_all_cc,&all_cc));
  for (i=0;i<n_faces;i++) {
    if (copycc) {
      PetscCall(ISDuplicate(faces[i],&all_cc[i]));
    } else {
      PetscCall(PetscObjectReference((PetscObject)faces[i]));
      all_cc[i] = faces[i];
    }
  }
  for (i=0;i<n_edges;i++) {
    if (copycc) {
      PetscCall(ISDuplicate(edges[i],&all_cc[n_faces+i]));
    } else {
      PetscCall(PetscObjectReference((PetscObject)edges[i]));
      all_cc[n_faces+i] = edges[i];
    }
    PetscCall(PetscBTSet(sub_schurs->is_edge,n_faces+i));
  }
  PetscCall(PetscObjectReference((PetscObject)vertices));
  sub_schurs->is_vertices = vertices;
  PetscCall(PCBDDCGraphRestoreCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices));
  sub_schurs->is_dir = NULL;
  PetscCall(PCBDDCGraphGetDirichletDofsB(graph,&sub_schurs->is_dir));

  /* Determine if MatFactor can be used */
  PetscCall(PetscStrallocpy(prefix,&sub_schurs->prefix));
#if defined(PETSC_HAVE_MUMPS)
  PetscCall(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMUMPS,sizeof(sub_schurs->mat_solver_type)));
#elif defined(PETSC_HAVE_MKL_PARDISO)
  PetscCall(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,sizeof(sub_schurs->mat_solver_type)));
#else
  PetscCall(PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERPETSC,sizeof(sub_schurs->mat_solver_type)));
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
  PetscOptionsBegin(PetscObjectComm((PetscObject)graph->l2gmap),sub_schurs->prefix,"BDDC sub_schurs options","PC");
  PetscCall(PetscOptionsString("-sub_schurs_mat_solver_type","Specific direct solver to use",NULL,sub_schurs->mat_solver_type,sub_schurs->mat_solver_type,sizeof(sub_schurs->mat_solver_type),NULL));
  PetscCall(PetscOptionsBool("-sub_schurs_symmetric","Symmetric problem",NULL,sub_schurs->is_symmetric,&sub_schurs->is_symmetric,NULL));
  PetscCall(PetscOptionsBool("-sub_schurs_hermitian","Hermitian problem",NULL,sub_schurs->is_hermitian,&sub_schurs->is_hermitian,NULL));
  PetscCall(PetscOptionsBool("-sub_schurs_posdef","Positive definite problem",NULL,sub_schurs->is_posdef,&sub_schurs->is_posdef,NULL));
  PetscCall(PetscOptionsBool("-sub_schurs_restrictcomm","Restrict communicator on active processes only",NULL,sub_schurs->restrict_comm,&sub_schurs->restrict_comm,NULL));
  PetscCall(PetscOptionsBool("-sub_schurs_debug","Debug output",NULL,sub_schurs->debug,&sub_schurs->debug,NULL));
  PetscOptionsEnd();
  PetscCall(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMUMPS,&ismumps));
  PetscCall(PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,&ispardiso));
  sub_schurs->schur_explicit = (PetscBool)(ispardiso || ismumps);

  /* for reals, symmetric and hermitian are synonims */
#if !defined(PETSC_USE_COMPLEX)
  sub_schurs->is_symmetric = (PetscBool)(sub_schurs->is_symmetric && sub_schurs->is_hermitian);
  sub_schurs->is_hermitian = sub_schurs->is_symmetric;
#endif

  PetscCall(PetscObjectReference((PetscObject)is_I));
  sub_schurs->is_I = is_I;
  PetscCall(PetscObjectReference((PetscObject)is_B));
  sub_schurs->is_B = is_B;
  PetscCall(PetscObjectReference((PetscObject)graph->l2gmap));
  sub_schurs->l2gmap = graph->l2gmap;
  PetscCall(PetscObjectReference((PetscObject)BtoNmap));
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
  PetscCall(PetscNew(&schurs_ctx));
  schurs_ctx->n_subs = 0;
  *sub_schurs = schurs_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs sub_schurs)
{
  PetscInt       i;

  PetscFunctionBegin;
  if (!sub_schurs) PetscFunctionReturn(0);
  PetscCall(PetscFree(sub_schurs->prefix));
  PetscCall(MatDestroy(&sub_schurs->A));
  PetscCall(MatDestroy(&sub_schurs->S));
  PetscCall(ISDestroy(&sub_schurs->is_I));
  PetscCall(ISDestroy(&sub_schurs->is_B));
  PetscCall(ISLocalToGlobalMappingDestroy(&sub_schurs->l2gmap));
  PetscCall(ISLocalToGlobalMappingDestroy(&sub_schurs->BtoNmap));
  PetscCall(MatDestroy(&sub_schurs->S_Ej_all));
  PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_all));
  PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_inv_all));
  PetscCall(MatDestroy(&sub_schurs->sum_S_Ej_tilda_all));
  PetscCall(ISDestroy(&sub_schurs->is_Ej_all));
  PetscCall(ISDestroy(&sub_schurs->is_vertices));
  PetscCall(ISDestroy(&sub_schurs->is_dir));
  PetscCall(PetscBTDestroy(&sub_schurs->is_edge));
  for (i=0;i<sub_schurs->n_subs;i++) {
    PetscCall(ISDestroy(&sub_schurs->is_subs[i]));
  }
  if (sub_schurs->n_subs) {
    PetscCall(PetscFree(sub_schurs->is_subs));
  }
  if (sub_schurs->reuse_solver) {
    PetscCall(PCBDDCReuseSolversReset(sub_schurs->reuse_solver));
  }
  PetscCall(PetscFree(sub_schurs->reuse_solver));
  if (sub_schurs->change) {
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscCall(KSPDestroy(&sub_schurs->change[i]));
      PetscCall(ISDestroy(&sub_schurs->change_primal_sub[i]));
    }
  }
  PetscCall(PetscFree(sub_schurs->change));
  PetscCall(PetscFree(sub_schurs->change_primal_sub));
  sub_schurs->n_subs = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs* sub_schurs)
{
  PetscFunctionBegin;
  PetscCall(PCBDDCSubSchursReset(*sub_schurs));
  PetscCall(PetscFree(*sub_schurs));
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
        PetscCall(PetscBTSet(touched,dof));
        queue_tip[n] = dof;
        n++;
      }
    }
  }
  *n_added = n;
  PetscFunctionReturn(0);
}
