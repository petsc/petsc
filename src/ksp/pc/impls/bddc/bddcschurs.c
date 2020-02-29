#include <../src/ksp/pc/impls/bddc/bddc.h>
#include <../src/ksp/pc/impls/bddc/bddcprivate.h>
#include <../src/mat/impls/dense/seq/dense.h>
#include <petscblaslapack.h>

PETSC_STATIC_INLINE PetscErrorCode PCBDDCAdjGetNextLayer_Private(PetscInt*,PetscInt,PetscBT,PetscInt*,PetscInt*,PetscInt*);
static PetscErrorCode PCBDDCComputeExplicitSchur(Mat,PetscBool,MatReuse,Mat*);
static PetscErrorCode PCBDDCReuseSolvers_Interior(PC,Vec,Vec);
static PetscErrorCode PCBDDCReuseSolvers_Correction(PC,Vec,Vec);

/* if v2 is not present, correction is done in-place */
PetscErrorCode PCBDDCReuseSolversBenignAdapt(PCBDDCReuseSolvers ctx, Vec v, Vec v2, PetscBool sol, PetscBool full)
{
  PetscScalar    *array;
  PetscScalar    *array2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!ctx->benign_n) PetscFunctionReturn(0);
  if (sol && full) {
    PetscInt n_I,size_schur;

    /* get sizes */
    ierr = MatGetSize(ctx->benign_csAIB,&size_schur,NULL);CHKERRQ(ierr);
    ierr = VecGetSize(v,&n_I);CHKERRQ(ierr);
    n_I = n_I - size_schur;
    /* get schur sol from array */
    ierr = VecGetArray(v,&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I);CHKERRQ(ierr);
    ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
    /* apply interior sol correction */
    ierr = MatMultTranspose(ctx->benign_csAIB,ctx->benign_dummy_schur_vec,ctx->benign_corr_work);CHKERRQ(ierr);
    ierr = VecResetArray(ctx->benign_dummy_schur_vec);CHKERRQ(ierr);
    ierr = MatMultAdd(ctx->benign_AIIm1ones,ctx->benign_corr_work,v,v);CHKERRQ(ierr);
  }
  if (v2) {
    PetscInt nl;

    ierr = VecGetArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = VecGetLocalSize(v2,&nl);CHKERRQ(ierr);
    ierr = VecGetArray(v2,&array2);CHKERRQ(ierr);
    ierr = PetscArraycpy(array2,array,nl);CHKERRQ(ierr);
  } else {
    ierr = VecGetArray(v,&array);CHKERRQ(ierr);
    array2 = array;
  }
  if (!sol) { /* change rhs */
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;

      ierr = ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz);CHKERRQ(ierr);
      ierr = ISGetIndices(ctx->benign_zerodiag_subs[n],&cols);CHKERRQ(ierr);
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      ctx->benign_save_vals[n] = array2[cols[nz-1]];
      array2[cols[nz-1]] = sum;
      ierr = ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols);CHKERRQ(ierr);
    }
  } else {
    PetscInt n;
    for (n=0;n<ctx->benign_n;n++) {
      PetscScalar    sum = 0.;
      const PetscInt *cols;
      PetscInt       nz,i;
      ierr = ISGetLocalSize(ctx->benign_zerodiag_subs[n],&nz);CHKERRQ(ierr);
      ierr = ISGetIndices(ctx->benign_zerodiag_subs[n],&cols);CHKERRQ(ierr);
      for (i=0;i<nz-1;i++) sum += array[cols[i]];
#if defined(PETSC_USE_COMPLEX)
      sum = -(PetscRealPart(sum)/nz + PETSC_i*(PetscImaginaryPart(sum)/nz));
#else
      sum = -sum/nz;
#endif
      for (i=0;i<nz-1;i++) array2[cols[i]] += sum;
      array2[cols[nz-1]] = ctx->benign_save_vals[n];
      ierr = ISRestoreIndices(ctx->benign_zerodiag_subs[n],&cols);CHKERRQ(ierr);
    }
  }
  if (v2) {
    ierr = VecRestoreArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = VecRestoreArray(v2,&array2);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(v,&array);CHKERRQ(ierr);
  }
  if (!sol && full) {
    Vec      usedv;
    PetscInt n_I,size_schur;

    /* get sizes */
    ierr = MatGetSize(ctx->benign_csAIB,&size_schur,NULL);CHKERRQ(ierr);
    ierr = VecGetSize(v,&n_I);CHKERRQ(ierr);
    n_I = n_I - size_schur;
    /* compute schur rhs correction */
    if (v2) {
      usedv = v2;
    } else {
      usedv = v;
    }
    /* apply schur rhs correction */
    ierr = MatMultTranspose(ctx->benign_AIIm1ones,usedv,ctx->benign_corr_work);CHKERRQ(ierr);
    ierr = VecGetArrayRead(usedv,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = VecPlaceArray(ctx->benign_dummy_schur_vec,array+n_I);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(usedv,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = MatMultAdd(ctx->benign_csAIB,ctx->benign_corr_work,ctx->benign_dummy_schur_vec,ctx->benign_dummy_schur_vec);CHKERRQ(ierr);
    ierr = VecResetArray(ctx->benign_dummy_schur_vec);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Solve_Private(PC pc, Vec rhs, Vec sol, PetscBool transpose, PetscBool full)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          copy = PETSC_FALSE;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void **)&ctx);CHKERRQ(ierr);
  if (full) {
#if defined(PETSC_HAVE_MUMPS)
    ierr = MatMumpsSetIcntl(ctx->F,26,-1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    ierr = MatMkl_PardisoSetCntl(ctx->F,70,0);CHKERRQ(ierr);
#endif
    copy = ctx->has_vertices;
  } else { /* interior solver */
#if defined(PETSC_HAVE_MUMPS)
    ierr = MatMumpsSetIcntl(ctx->F,26,0);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
    ierr = MatMkl_PardisoSetCntl(ctx->F,70,1);CHKERRQ(ierr);
#endif
    copy = PETSC_TRUE;
  }
  /* copy rhs into factored matrix workspace */
  if (copy) {
    PetscInt    n;
    PetscScalar *array,*array_solver;

    ierr = VecGetLocalSize(rhs,&n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(rhs,(const PetscScalar**)&array);CHKERRQ(ierr);
    ierr = VecGetArray(ctx->rhs,&array_solver);CHKERRQ(ierr);
    ierr = PetscArraycpy(array_solver,array,n);CHKERRQ(ierr);
    ierr = VecRestoreArray(ctx->rhs,&array_solver);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(rhs,(const PetscScalar**)&array);CHKERRQ(ierr);

    ierr = PCBDDCReuseSolversBenignAdapt(ctx,ctx->rhs,NULL,PETSC_FALSE,full);CHKERRQ(ierr);
    if (transpose) {
      ierr = MatSolveTranspose(ctx->F,ctx->rhs,ctx->sol);CHKERRQ(ierr);
    } else {
      ierr = MatSolve(ctx->F,ctx->rhs,ctx->sol);CHKERRQ(ierr);
    }
    ierr = PCBDDCReuseSolversBenignAdapt(ctx,ctx->sol,NULL,PETSC_TRUE,full);CHKERRQ(ierr);

    /* get back data to caller worskpace */
    ierr = VecGetArrayRead(ctx->sol,(const PetscScalar**)&array_solver);CHKERRQ(ierr);
    ierr = VecGetArray(sol,&array);CHKERRQ(ierr);
    ierr = PetscArraycpy(array,array_solver,n);CHKERRQ(ierr);
    ierr = VecRestoreArray(sol,&array);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(ctx->sol,(const PetscScalar**)&array_solver);CHKERRQ(ierr);
  } else {
    if (ctx->benign_n) {
      ierr = PCBDDCReuseSolversBenignAdapt(ctx,rhs,ctx->rhs,PETSC_FALSE,full);CHKERRQ(ierr);
      if (transpose) {
        ierr = MatSolveTranspose(ctx->F,ctx->rhs,sol);CHKERRQ(ierr);
      } else {
        ierr = MatSolve(ctx->F,ctx->rhs,sol);CHKERRQ(ierr);
      }
      ierr = PCBDDCReuseSolversBenignAdapt(ctx,sol,NULL,PETSC_TRUE,full);CHKERRQ(ierr);
    } else {
      if (transpose) {
        ierr = MatSolveTranspose(ctx->F,rhs,sol);CHKERRQ(ierr);
      } else {
        ierr = MatSolve(ctx->F,rhs,sol);CHKERRQ(ierr);
      }
    }
  }
  /* restore defaults */
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(ctx->F,26,-1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
  ierr = MatMkl_PardisoSetCntl(ctx->F,70,0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Correction(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_CorrectionTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_Interior(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_FALSE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_InteriorTranspose(PC pc, Vec rhs, Vec sol)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCBDDCReuseSolvers_Solve_Private(pc,rhs,sol,PETSC_TRUE,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolvers_View(PC pc, PetscViewer viewer)
{
  PCBDDCReuseSolvers ctx;
  PetscBool          iascii;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = PCShellGetContext(pc,(void **)&ctx);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  }
  ierr = MatView(ctx->F,viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCBDDCReuseSolversReset(PCBDDCReuseSolvers reuse)
{
  PetscInt       i;
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
  for (i=0;i<reuse->benign_n;i++) {
    ierr = ISDestroy(&reuse->benign_zerodiag_subs[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(reuse->benign_zerodiag_subs);CHKERRQ(ierr);
  ierr = PetscFree(reuse->benign_save_vals);CHKERRQ(ierr);
  ierr = MatDestroy(&reuse->benign_csAIB);CHKERRQ(ierr);
  ierr = MatDestroy(&reuse->benign_AIIm1ones);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->benign_corr_work);CHKERRQ(ierr);
  ierr = VecDestroy(&reuse->benign_dummy_schur_vec);CHKERRQ(ierr);
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

        ierr = PCComputeOperator(pc, MATDENSE, &Ainvd);CHKERRQ(ierr);
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

  PetscFunctionBegin;
  ierr = MatDestroy(&sub_schurs->A);CHKERRQ(ierr);
  ierr = MatDestroy(&sub_schurs->S);CHKERRQ(ierr);
  if (Ain) {
    ierr = PetscObjectReference((PetscObject)Ain);CHKERRQ(ierr);
    sub_schurs->A = Ain;
  }

  ierr = PetscObjectReference((PetscObject)Sin);CHKERRQ(ierr);
  sub_schurs->S = Sin;
  if (sub_schurs->schur_explicit) {
    sub_schurs->schur_explicit = (PetscBool)(!!sub_schurs->A);
  }

  /* preliminary checks */
  if (!sub_schurs->schur_explicit && compute_Stilda) SETERRQ(PetscObjectComm((PetscObject)sub_schurs->l2gmap),PETSC_ERR_SUP,"Adaptive selection of constraints requires MUMPS and/or MKL_PARDISO");

  if (benign_trick) sub_schurs->is_posdef = PETSC_FALSE;

  /* debug (MATLAB) */
  if (sub_schurs->debug) {
    PetscMPIInt size,rank;
    PetscInt    nr,*print_schurs_ranks,print_schurs = PETSC_FALSE;
    PetscBool   flg;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&rank);CHKERRQ(ierr);
    nr   = size;
    ierr = PetscMalloc1(nr,&print_schurs_ranks);CHKERRQ(ierr);
    ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)sub_schurs->l2gmap),sub_schurs->prefix,"BDDC sub_schurs options","PC");CHKERRQ(ierr);
    ierr = PetscOptionsIntArray("-sub_schurs_debug_ranks","Ranks to debug (all if the option is not used)",NULL,print_schurs_ranks,&nr,&flg);CHKERRQ(ierr);
    if (!flg) print_schurs = PETSC_TRUE;
    else {
      print_schurs = PETSC_FALSE;
      for (i=0;i<nr;i++) if (print_schurs_ranks[i] == (PetscInt)rank) { print_schurs = PETSC_TRUE; break; }
    }
    ierr = PetscOptionsEnd();CHKERRQ(ierr);
    ierr = PetscFree(print_schurs_ranks);CHKERRQ(ierr);
    if (print_schurs) {
      char filename[256];

      ierr = PetscSNPrintf(filename,sizeof(filename),"sub_schurs_Schur_r%d.m",PetscGlobalRank);CHKERRQ(ierr);
      ierr = PetscViewerASCIIOpen(PETSC_COMM_SELF,filename,&matl_dbg_viewer);CHKERRQ(ierr);
      ierr = PetscViewerPushFormat(matl_dbg_viewer,PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    }
  }


  /* restrict work on active processes */
  if (sub_schurs->restrict_comm) {
    PetscSubcomm subcomm;
    PetscMPIInt  color,rank;

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
  } else {
    ierr = PetscCommDuplicate(PetscObjectComm((PetscObject)sub_schurs->l2gmap),&comm_n,NULL);CHKERRQ(ierr);
  }

  /* get Schur complement matrices */
  if (!sub_schurs->schur_explicit) {
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

    if (!xadj) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Cannot request layering without adjacency");
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
    ierr = PetscArraycpy(local_numbering,idx_B,n_B);CHKERRQ(ierr);
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
    if (!sub_schurs->schur_explicit) {
      ISLocalToGlobalMapping ItoNmap;
      ierr = ISLocalToGlobalMappingCreateIS(sub_schurs->is_I,&ItoNmap);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(ItoNmap,IS_GTOLM_DROP,is_I_layer,&is_I);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&ItoNmap);CHKERRQ(ierr);

      /* II block */
      ierr = MatCreateSubMatrix(A_II,is_I,is_I,MAT_INITIAL_MATRIX,&AE_II);CHKERRQ(ierr);
    }
  } else {
    PetscInt n_I;

    /* IS for I dofs in original numbering */
    ierr = PetscObjectReference((PetscObject)sub_schurs->is_I);CHKERRQ(ierr);
    is_I_layer = sub_schurs->is_I;

    /* IS for I dofs in I numbering (strided 1) */
    if (!sub_schurs->schur_explicit) {
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
  ierr = ISGetLocalSize(sub_schurs->is_B,&n_B);CHKERRQ(ierr);
  if (sub_schurs->schur_explicit && is_I_layer) {
    ierr = ISGetLocalSize(is_I_layer,&extra);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(n_B+extra,&all_local_idx_N);CHKERRQ(ierr);
  if (extra) {
    const PetscInt *idxs;
    ierr = ISGetIndices(is_I_layer,&idxs);CHKERRQ(ierr);
    ierr = PetscArraycpy(all_local_idx_N,idxs,extra);CHKERRQ(ierr);
    ierr = ISRestoreIndices(is_I_layer,&idxs);CHKERRQ(ierr);
  }
  ierr = PetscMalloc1(sub_schurs->n_subs,&auxnum1);CHKERRQ(ierr);
  ierr = PetscMalloc1(sub_schurs->n_subs,&auxnum2);CHKERRQ(ierr);

  /* Get local indices in local numbering */
  local_size = 0;
  local_stash_size = 0;
  for (i=0;i<sub_schurs->n_subs;i++) {
    const PetscInt *idxs;

    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
    ierr = ISGetIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
    /* start (smallest in global ordering) and multiplicity */
    auxnum1[i] = idxs[0];
    auxnum2[i] = subset_size*subset_size;
    /* subset indices in local numbering */
    ierr = PetscArraycpy(all_local_idx_N+local_size+extra,idxs,subset_size);CHKERRQ(ierr);
    ierr = ISRestoreIndices(sub_schurs->is_subs[i],&idxs);CHKERRQ(ierr);
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
    ierr = PetscBLASIntCast(local_size,&B_N);CHKERRQ(ierr);
    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    if (use_sytr) {
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to SYTRF Lapack routine %d",(int)B_ierr);
    } else {
      PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,&dummyscalar,&B_N,&dummyint,&lwork,&B_lwork,&B_ierr));
      if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in query to GETRI Lapack routine %d",(int)B_ierr);
    }
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
    ierr = PetscBLASIntCast((PetscInt)PetscRealPart(lwork),&B_lwork);CHKERRQ(ierr);
    ierr = PetscMalloc2(B_lwork,&Bwork,B_N,&pivots);CHKERRQ(ierr);
  } else {
    Bwork = NULL;
    pivots = NULL;
  }

  /* prepare data for summing up properly schurs on subsets */
  ierr = ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum1,PETSC_OWN_POINTER,&all_subsets_n);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingApplyIS(sub_schurs->l2gmap,all_subsets_n,&all_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_n);CHKERRQ(ierr);
  ierr = ISCreateGeneral(comm_n,sub_schurs->n_subs,auxnum2,PETSC_OWN_POINTER,&all_subsets_mult);CHKERRQ(ierr);
  ierr = ISRenumber(all_subsets,all_subsets_mult,&global_size,&all_subsets_n);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_mult);CHKERRQ(ierr);
  ierr = ISGetLocalSize(all_subsets_n,&i);CHKERRQ(ierr);
  if (i != local_stash_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Invalid size of new subset! %D != %D",i,local_stash_size);
  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,local_stash_size,NULL,&lstash);CHKERRQ(ierr);
  ierr = VecCreateMPI(comm_n,PETSC_DECIDE,global_size,&gstash);CHKERRQ(ierr);
  ierr = VecScatterCreate(lstash,NULL,gstash,all_subsets_n,&sstash);CHKERRQ(ierr);
  ierr = ISDestroy(&all_subsets_n);CHKERRQ(ierr);

  /* subset indices in local boundary numbering */
  if (!sub_schurs->is_Ej_all) {
    PetscInt *all_local_idx_B;

    ierr = PetscMalloc1(local_size,&all_local_idx_B);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApply(sub_schurs->BtoNmap,IS_GTOLM_DROP,local_size,all_local_idx_N+extra,&subset_size,all_local_idx_B);CHKERRQ(ierr);
    if (subset_size != local_size) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error in sub_schurs serial (BtoNmap)! %D != %D",subset_size,local_size);
    ierr = ISCreateGeneral(PETSC_COMM_SELF,local_size,all_local_idx_B,PETSC_OWN_POINTER,&sub_schurs->is_Ej_all);CHKERRQ(ierr);
  }

  if (change) {
    ISLocalToGlobalMapping BtoS;
    IS                     change_primal_B;
    IS                     change_primal_all;

    if (sub_schurs->change_primal_sub) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    if (sub_schurs->change) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    ierr = PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change_primal_sub);CHKERRQ(ierr);
    for (i=0;i<sub_schurs->n_subs;i++) {
      ISLocalToGlobalMapping NtoS;
      ierr = ISLocalToGlobalMappingCreateIS(sub_schurs->is_subs[i],&NtoS);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(NtoS,IS_GTOLM_DROP,change_primal,&sub_schurs->change_primal_sub[i]);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&NtoS);CHKERRQ(ierr);
    }
    ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,change_primal,&change_primal_B);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingCreateIS(sub_schurs->is_Ej_all,&BtoS);CHKERRQ(ierr);
    ierr = ISGlobalToLocalMappingApplyIS(BtoS,IS_GTOLM_DROP,change_primal_B,&change_primal_all);CHKERRQ(ierr);
    ierr = ISLocalToGlobalMappingDestroy(&BtoS);CHKERRQ(ierr);
    ierr = ISDestroy(&change_primal_B);CHKERRQ(ierr);
    ierr = PetscMalloc1(sub_schurs->n_subs,&sub_schurs->change);CHKERRQ(ierr);
    for (i=0;i<sub_schurs->n_subs;i++) {
      Mat change_sub;

      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      ierr = KSPCreate(PETSC_COMM_SELF,&sub_schurs->change[i]);CHKERRQ(ierr);
      ierr = KSPSetType(sub_schurs->change[i],KSPPREONLY);CHKERRQ(ierr);
      if (!sub_schurs->change_with_qr) {
        ierr = MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_sub);CHKERRQ(ierr);
      } else {
        Mat change_subt;
        ierr = MatCreateSubMatrix(change,sub_schurs->is_subs[i],sub_schurs->is_subs[i],MAT_INITIAL_MATRIX,&change_subt);CHKERRQ(ierr);
        ierr = MatConvert(change_subt,MATSEQDENSE,MAT_INITIAL_MATRIX,&change_sub);CHKERRQ(ierr);
        ierr = MatDestroy(&change_subt);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(sub_schurs->change[i],change_sub,change_sub);CHKERRQ(ierr);
      ierr = MatDestroy(&change_sub);CHKERRQ(ierr);
      ierr = KSPSetOptionsPrefix(sub_schurs->change[i],sub_schurs->prefix);CHKERRQ(ierr);
      ierr = KSPAppendOptionsPrefix(sub_schurs->change[i],"sub_schurs_change_");CHKERRQ(ierr);
    }
    ierr = ISDestroy(&change_primal_all);CHKERRQ(ierr);
  }

  /* Local matrix of all local Schur on subsets (transposed) */
  if (!sub_schurs->S_Ej_all) {
    Mat         T;
    PetscScalar *v;
    PetscInt    *ii,*jj;
    PetscInt    cum,i,j,k;

    /* MatSeqAIJSetPreallocation + MatSetValues is slow for these kind of matrices (may have large blocks)
       Allocate properly a representative matrix and duplicate */
    ierr  = PetscMalloc3(local_size+1,&ii,local_stash_size,&jj,local_stash_size,&v);CHKERRQ(ierr);
    ii[0] = 0;
    cum   = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
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
    ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,local_size,local_size,ii,jj,v,&T);CHKERRQ(ierr);
    ierr = MatDuplicate(T,MAT_DO_NOT_COPY_VALUES,&sub_schurs->S_Ej_all);CHKERRQ(ierr);
    ierr = MatDestroy(&T);CHKERRQ(ierr);
    ierr = PetscFree3(ii,jj,v);CHKERRQ(ierr);
  }
  /* matrices for deluxe scaling and adaptive selection */
  if (compute_Stilda) {
    if (!sub_schurs->sum_S_Ej_tilda_all) {
      ierr = MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_tilda_all);CHKERRQ(ierr);
    }
    if (!sub_schurs->sum_S_Ej_inv_all && deluxe) {
      ierr = MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
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

    ierr = PetscMalloc2(max_subset_size,&dummy_idx,max_subset_size*max_subset_size,&work);CHKERRQ(ierr);
    local_size = 0;
    for (i=0;i<sub_schurs->n_subs;i++) {
      IS  is_subset_B;
      Mat AE_EE,AE_IE,AE_EI,S_Ej;

      /* subsets in original and boundary numbering */
      ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,sub_schurs->is_subs[i],&is_subset_B);CHKERRQ(ierr);
      /* EE block */
      ierr = MatCreateSubMatrix(A_BB,is_subset_B,is_subset_B,MAT_INITIAL_MATRIX,&AE_EE);CHKERRQ(ierr);
      /* IE block */
      ierr = MatCreateSubMatrix(A_IB,is_I,is_subset_B,MAT_INITIAL_MATRIX,&AE_IE);CHKERRQ(ierr);
      /* EI block */
      if (sub_schurs->is_symmetric) {
        ierr = MatCreateTranspose(AE_IE,&AE_EI);CHKERRQ(ierr);
      } else if (sub_schurs->is_hermitian) {
        ierr = MatCreateHermitianTranspose(AE_IE,&AE_EI);CHKERRQ(ierr);
      } else {
        ierr = MatCreateSubMatrix(A_BI,is_subset_B,is_I,MAT_INITIAL_MATRIX,&AE_EI);CHKERRQ(ierr);
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
        if (!n_internal) { /* UMFPACK gives error with 0 sized problems */
          MatSolverType solver = NULL;
          ierr = PCFactorGetMatSolverType(origpc,(MatSolverType*)&solver);CHKERRQ(ierr);
          if (solver) {
            ierr = PCFactorSetMatSolverType(schurpc,solver);CHKERRQ(ierr);
          }
        }
        ierr = KSPSetUp(schurksp);CHKERRQ(ierr);
      }
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&S_Ej_expl);CHKERRQ(ierr);
      ierr = PCBDDCComputeExplicitSchur(S_Ej,sub_schurs->is_symmetric,MAT_REUSE_MATRIX,&S_Ej_expl);CHKERRQ(ierr);
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
      ierr = ISGetLocalSize(is_I_layer,&n_I);CHKERRQ(ierr);
    }
    economic = PETSC_FALSE;
    ierr = ISGetLocalSize(sub_schurs->is_I,&cum);CHKERRQ(ierr);
    if (cum != n_I) economic = PETSC_TRUE;
    ierr = MatGetLocalSize(sub_schurs->A,&n,NULL);CHKERRQ(ierr);
    size_active_schur = local_size;

    /* import scaling vector (wrong formulation if we have 3D edges) */
    if (scaling && compute_Stilda) {
      const PetscScalar *array;
      PetscScalar       *array2;
      const PetscInt    *idxs;
      PetscInt          i;

      ierr = ISGetIndices(sub_schurs->is_Ej_all,&idxs);CHKERRQ(ierr);
      ierr = VecCreateSeq(PETSC_COMM_SELF,size_active_schur,&Dall);CHKERRQ(ierr);
      ierr = VecGetArrayRead(scaling,&array);CHKERRQ(ierr);
      ierr = VecGetArray(Dall,&array2);CHKERRQ(ierr);
      for (i=0;i<size_active_schur;i++) array2[i] = array[idxs[i]];
      ierr = VecRestoreArray(Dall,&array2);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(scaling,&array);CHKERRQ(ierr);
      ierr = ISRestoreIndices(sub_schurs->is_Ej_all,&idxs);CHKERRQ(ierr);
      deluxe = PETSC_FALSE;
    }

    /* size active schurs does not count any dirichlet or vertex dof on the interface */
    factor_workaround = PETSC_FALSE;
    schur_has_vertices = PETSC_FALSE;
    cum = n_I+size_active_schur;
    if (sub_schurs->is_dir) {
      const PetscInt* idxs;
      PetscInt        n_dir;

      ierr = ISGetLocalSize(sub_schurs->is_dir,&n_dir);CHKERRQ(ierr);
      ierr = ISGetIndices(sub_schurs->is_dir,&idxs);CHKERRQ(ierr);
      ierr = PetscArraycpy(all_local_idx_N+cum,idxs,n_dir);CHKERRQ(ierr);
      ierr = ISRestoreIndices(sub_schurs->is_dir,&idxs);CHKERRQ(ierr);
      cum += n_dir;
      factor_workaround = PETSC_TRUE;
    }
    /* include the primal vertices in the Schur complement */
    if (exact_schur && sub_schurs->is_vertices && (compute_Stilda || benign_n)) {
      PetscInt n_v;

      ierr = ISGetLocalSize(sub_schurs->is_vertices,&n_v);CHKERRQ(ierr);
      if (n_v) {
        const PetscInt* idxs;

        ierr = ISGetIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
        ierr = PetscArraycpy(all_local_idx_N+cum,idxs,n_v);CHKERRQ(ierr);
        ierr = ISRestoreIndices(sub_schurs->is_vertices,&idxs);CHKERRQ(ierr);
        cum += n_v;
        factor_workaround = PETSC_TRUE;
        schur_has_vertices = PETSC_TRUE;
      }
    }
    size_schur = cum - n_I;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,cum,all_local_idx_N,PETSC_OWN_POINTER,&is_A_all);CHKERRQ(ierr);
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    oldpin = sub_schurs->A->boundtocpu;
    ierr = MatBindToCPU(sub_schurs->A,PETSC_TRUE);CHKERRQ(ierr);
#endif
    if (cum == n) {
      ierr = ISSetPermutation(is_A_all);CHKERRQ(ierr);
      ierr = MatPermute(sub_schurs->A,is_A_all,is_A_all,&A);CHKERRQ(ierr);
    } else {
      ierr = MatCreateSubMatrix(sub_schurs->A,is_A_all,is_A_all,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    }
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    ierr = MatBindToCPU(sub_schurs->A,oldpin);CHKERRQ(ierr);
#endif
    ierr = MatSetOptionsPrefix(A,sub_schurs->prefix);CHKERRQ(ierr);
    ierr = MatAppendOptionsPrefix(A,"sub_schurs_");CHKERRQ(ierr);

    /* if we actually change the basis for the pressures, LDL^T factors will use a lot of memory
       this is a workaround */
    if (benign_n) {
      Vec                    v,benign_AIIm1_ones;
      ISLocalToGlobalMapping N_to_reor;
      IS                     is_p0,is_p0_p;
      PetscScalar            *cs_AIB,*AIIm1_data;
      PetscInt               sizeA;

      ierr = ISLocalToGlobalMappingCreateIS(is_A_all,&N_to_reor);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF,benign_n,benign_p0_lidx,PETSC_COPY_VALUES,&is_p0);CHKERRQ(ierr);
      ierr = ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,is_p0,&is_p0_p);CHKERRQ(ierr);
      ierr = ISDestroy(&is_p0);CHKERRQ(ierr);
      ierr = MatCreateVecs(A,&v,&benign_AIIm1_ones);CHKERRQ(ierr);
      ierr = VecGetSize(v,&sizeA);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,sizeA,benign_n,NULL,&benign_AIIm1_ones_mat);CHKERRQ(ierr);
      ierr = MatCreateSeqDense(PETSC_COMM_SELF,size_schur,benign_n,NULL,&cs_AIB_mat);CHKERRQ(ierr);
      ierr = MatDenseGetArray(cs_AIB_mat,&cs_AIB);CHKERRQ(ierr);
      ierr = MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data);CHKERRQ(ierr);
      ierr = PetscMalloc1(benign_n,&is_p_r);CHKERRQ(ierr);
      /* compute colsum of A_IB restricted to pressures */
      for (i=0;i<benign_n;i++) {
        const PetscScalar *array;
        const PetscInt    *idxs;
        PetscInt          j,nz;

        ierr = ISGlobalToLocalMappingApplyIS(N_to_reor,IS_GTOLM_DROP,benign_zerodiag_subs[i],&is_p_r[i]);CHKERRQ(ierr);
        ierr = ISGetLocalSize(is_p_r[i],&nz);CHKERRQ(ierr);
        ierr = ISGetIndices(is_p_r[i],&idxs);CHKERRQ(ierr);
        for (j=0;j<nz;j++) AIIm1_data[idxs[j]+sizeA*i] = 1.;
        ierr = ISRestoreIndices(is_p_r[i],&idxs);CHKERRQ(ierr);
        ierr = VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i);CHKERRQ(ierr);
        ierr = MatMult(A,benign_AIIm1_ones,v);CHKERRQ(ierr);
        ierr = VecResetArray(benign_AIIm1_ones);CHKERRQ(ierr);
        ierr = VecGetArrayRead(v,&array);CHKERRQ(ierr);
        for (j=0;j<size_schur;j++) {
#if defined(PETSC_USE_COMPLEX)
          cs_AIB[i*size_schur + j] = (PetscRealPart(array[j+n_I])/nz + PETSC_i*(PetscImaginaryPart(array[j+n_I])/nz));
#else
          cs_AIB[i*size_schur + j] = array[j+n_I]/nz;
#endif
        }
        ierr = VecRestoreArrayRead(v,&array);CHKERRQ(ierr);
      }
      ierr = MatDenseRestoreArray(cs_AIB_mat,&cs_AIB);CHKERRQ(ierr);
      ierr = MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data);CHKERRQ(ierr);
      ierr = VecDestroy(&v);CHKERRQ(ierr);
      ierr = VecDestroy(&benign_AIIm1_ones);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_KEEP_NONZERO_PATTERN,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
      ierr = MatZeroRowsColumnsIS(A,is_p0_p,1.0,NULL,NULL);CHKERRQ(ierr);
      ierr = ISDestroy(&is_p0_p);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingDestroy(&N_to_reor);CHKERRQ(ierr);
    }
    ierr = MatSetOption(A,MAT_SYMMETRIC,sub_schurs->is_symmetric);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_HERMITIAN,sub_schurs->is_hermitian);CHKERRQ(ierr);
    ierr = MatSetOption(A,MAT_SPD,sub_schurs->is_posdef);CHKERRQ(ierr);

    /* for complexes, symmetric and hermitian at the same time implies null imaginary part */
    use_cholesky = (PetscBool)((use_potr || use_sytr) && sub_schurs->is_hermitian && sub_schurs->is_symmetric);

    /* when using the benign subspace trick, the local Schur complements are SPD */
    if (benign_trick) sub_schurs->is_posdef = PETSC_TRUE;

    if (n_I) {
      IS        is_schur;
      char      stype[64];
      PetscBool gpu;

      if (use_cholesky) {
        ierr = MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
      } else {
        ierr = MatGetFactor(A,sub_schurs->mat_solver_type,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
      }
      ierr = MatSetErrorIfFailure(A,PETSC_TRUE);CHKERRQ(ierr);

      /* subsets ordered last */
      ierr = ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is_schur);CHKERRQ(ierr);
      ierr = MatFactorSetSchurIS(F,is_schur);CHKERRQ(ierr);
      ierr = ISDestroy(&is_schur);CHKERRQ(ierr);

      /* factorization step */
      if (use_cholesky) {
        ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        ierr = MatMumpsSetIcntl(F,19,2);CHKERRQ(ierr);
#endif
        ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
        S_lower_triangular = PETSC_TRUE;
      } else {
        ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS) /* be sure that icntl 19 is not set by command line */
        ierr = MatMumpsSetIcntl(F,19,3);CHKERRQ(ierr);
#endif
        ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
      }
      ierr = MatViewFromOptions(F,(PetscObject)A,"-mat_factor_view");CHKERRQ(ierr);

      if (matl_dbg_viewer) {
        Mat S;
        IS  is;

        ierr = PetscObjectSetName((PetscObject)A,"A");CHKERRQ(ierr);
        ierr = MatView(A,matl_dbg_viewer);CHKERRQ(ierr);
        ierr = MatFactorCreateSchurComplement(F,&S,NULL);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)S,"S");CHKERRQ(ierr);
        ierr = MatView(S,matl_dbg_viewer);CHKERRQ(ierr);
        ierr = MatDestroy(&S);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,n_I,0,1,&is);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)is,"I");CHKERRQ(ierr);
        ierr = ISView(is,matl_dbg_viewer);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
        ierr = ISCreateStride(PETSC_COMM_SELF,size_schur,n_I,1,&is);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)is,"B");CHKERRQ(ierr);
        ierr = ISView(is,matl_dbg_viewer);CHKERRQ(ierr);
        ierr = ISDestroy(&is);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)is_A_all,"IA");CHKERRQ(ierr);
        ierr = ISView(is_A_all,matl_dbg_viewer);CHKERRQ(ierr);
      }

      /* get explicit Schur Complement computed during numeric factorization */
      ierr = MatFactorGetSchurComplement(F,&S_all,NULL);CHKERRQ(ierr);
      ierr = PetscStrncpy(stype,MATSEQDENSE,sizeof(stype));CHKERRQ(ierr);
      ierr = PetscObjectTypeCompareAny((PetscObject)A,&gpu,MATSEQAIJVIENNACL,MATSEQAIJCUSPARSE,"");CHKERRQ(ierr);
      if (gpu) {
        ierr = PetscStrncpy(stype,MATSEQDENSECUDA,sizeof(stype));CHKERRQ(ierr);
      }
      ierr = PetscOptionsGetString(NULL,sub_schurs->prefix,"-sub_schurs_schur_mat_type",stype,sizeof(stype),NULL);CHKERRQ(ierr);
      ierr = MatConvert(S_all,stype,MAT_INPLACE_MATRIX,&S_all);CHKERRQ(ierr);
      ierr = MatSetOption(S_all,MAT_SPD,sub_schurs->is_posdef);CHKERRQ(ierr);
      ierr = MatSetOption(S_all,MAT_HERMITIAN,sub_schurs->is_hermitian);CHKERRQ(ierr);
      ierr = MatGetType(S_all,&Stype);CHKERRQ(ierr);

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

        ierr = MatDenseGetArray(S_all,&S_data);CHKERRQ(ierr);
        ierr = MatCreateVecs(A,&v,&benign_AIIm1_ones);CHKERRQ(ierr);
        ierr = VecGetSize(v,&sizeA);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
        ierr = MatMumpsSetIcntl(F,26,0);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        ierr = MatMkl_PardisoSetCntl(F,70,1);CHKERRQ(ierr);
#endif
        ierr = MatDenseGetArrayRead(cs_AIB_mat,&cs_AIB);CHKERRQ(ierr);
        ierr = MatDenseGetArray(benign_AIIm1_ones_mat,&AIIm1_data);CHKERRQ(ierr);
        if (matl_dbg_viewer) {
          ierr = MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S2);CHKERRQ(ierr);
          ierr = MatDuplicate(S_all,MAT_DO_NOT_COPY_VALUES,&S3);CHKERRQ(ierr);
          ierr = MatDenseGetArray(S2,&S2_data);CHKERRQ(ierr);
          ierr = MatDenseGetArray(S3,&S3_data);CHKERRQ(ierr);
        }
        for (i=0;i<benign_n;i++) {
          PetscScalar    *array,sum = 0.,one = 1.,*sums;
          const PetscInt *idxs;
          PetscInt       k,j,nz;
          PetscBLASInt   B_k,B_n;

          ierr = PetscCalloc1(benign_n,&sums);CHKERRQ(ierr);
          ierr = VecPlaceArray(benign_AIIm1_ones,AIIm1_data+sizeA*i);CHKERRQ(ierr);
          ierr = VecCopy(benign_AIIm1_ones,v);CHKERRQ(ierr);
          ierr = MatSolve(F,v,benign_AIIm1_ones);CHKERRQ(ierr);
          ierr = MatMult(A,benign_AIIm1_ones,v);CHKERRQ(ierr);
          ierr = VecResetArray(benign_AIIm1_ones);CHKERRQ(ierr);
          /* p0 dofs (eliminated) are excluded from the sums */
          for (k=0;k<benign_n;k++) {
            ierr = ISGetLocalSize(is_p_r[k],&nz);CHKERRQ(ierr);
            ierr = ISGetIndices(is_p_r[k],&idxs);CHKERRQ(ierr);
            for (j=0;j<nz-1;j++) sums[k] -= AIIm1_data[idxs[j]+sizeA*i];
            ierr = ISRestoreIndices(is_p_r[k],&idxs);CHKERRQ(ierr);
          }
          ierr = VecGetArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
          if (matl_dbg_viewer) {
            Vec  vv;
            char name[16];

            ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array+n_I,&vv);CHKERRQ(ierr);
            ierr = PetscSNPrintf(name,sizeof(name),"Pvs%D",i);CHKERRQ(ierr);
            ierr = PetscObjectSetName((PetscObject)vv,name);CHKERRQ(ierr);
            ierr = VecView(vv,matl_dbg_viewer);CHKERRQ(ierr);
          }
          /* perform sparse rank updates on symmetric Schur (TODO: move outside of the loop?) */
          /* cs_AIB already scaled by 1./nz */
          B_k = 1;
          for (k=0;k<benign_n;k++) {
            sum  = sums[k];
            ierr = PetscBLASIntCast(size_schur,&B_n);CHKERRQ(ierr);

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
          ierr = VecRestoreArrayRead(v,(const PetscScalar**)&array);CHKERRQ(ierr);
          /* set p0 entry of AIIm1_ones to zero */
          ierr = ISGetLocalSize(is_p_r[i],&nz);CHKERRQ(ierr);
          ierr = ISGetIndices(is_p_r[i],&idxs);CHKERRQ(ierr);
          for (j=0;j<benign_n;j++) AIIm1_data[idxs[nz-1]+sizeA*j] = 0.;
          ierr = ISRestoreIndices(is_p_r[i],&idxs);CHKERRQ(ierr);
          ierr = PetscFree(sums);CHKERRQ(ierr);
        }
        ierr = VecDestroy(&benign_AIIm1_ones);CHKERRQ(ierr);
        if (matl_dbg_viewer) {
          ierr = MatDenseRestoreArray(S2,&S2_data);CHKERRQ(ierr);
          ierr = MatDenseRestoreArray(S3,&S3_data);CHKERRQ(ierr);
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
        ierr = MatMumpsSetIcntl(F,26,-1);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MKL_PARDISO)
        ierr = MatMkl_PardisoSetCntl(F,70,0);CHKERRQ(ierr);
#endif
        ierr = MatDenseRestoreArrayRead(cs_AIB_mat,&cs_AIB);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(benign_AIIm1_ones_mat,&AIIm1_data);CHKERRQ(ierr);
        ierr = VecDestroy(&v);CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(S_all,&S_data);CHKERRQ(ierr);
        if (matl_dbg_viewer) {
          Mat S;

          ierr = MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
          ierr = MatFactorCreateSchurComplement(F,&S,NULL);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)S,"Sb");CHKERRQ(ierr);
          ierr = MatView(S,matl_dbg_viewer);CHKERRQ(ierr);
          ierr = MatDestroy(&S);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)S2,"S2P");CHKERRQ(ierr);
          ierr = MatView(S2,matl_dbg_viewer);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)S3,"S3P");CHKERRQ(ierr);
          ierr = MatView(S3,matl_dbg_viewer);CHKERRQ(ierr);
          ierr = PetscObjectSetName((PetscObject)cs_AIB_mat,"cs");CHKERRQ(ierr);
          ierr = MatView(cs_AIB_mat,matl_dbg_viewer);CHKERRQ(ierr);
          ierr = MatFactorGetSchurComplement(F,&S_all,NULL);CHKERRQ(ierr);
        }
        ierr = MatDestroy(&S2);CHKERRQ(ierr);
        ierr = MatDestroy(&S3);CHKERRQ(ierr);
      }
      if (!reuse_solvers) {
        for (i=0;i<benign_n;i++) {
          ierr = ISDestroy(&is_p_r[i]);CHKERRQ(ierr);
        }
        ierr = PetscFree(is_p_r);CHKERRQ(ierr);
        ierr = MatDestroy(&cs_AIB_mat);CHKERRQ(ierr);
        ierr = MatDestroy(&benign_AIIm1_ones_mat);CHKERRQ(ierr);
      }
    } else { /* we can't use MatFactor when size_schur == size_of_the_problem */
      ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&S_all);CHKERRQ(ierr);
      ierr = MatGetType(S_all,&Stype);CHKERRQ(ierr);
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
        ierr = PCBDDCReuseSolversReset(sub_schurs->reuse_solver);CHKERRQ(ierr);
      } else {
        ierr = PetscNew(&sub_schurs->reuse_solver);CHKERRQ(ierr);
      }
      msolv_ctx = sub_schurs->reuse_solver;
      ierr = MatSchurComplementGetSubMatrices(sub_schurs->S,&A_II,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)F);CHKERRQ(ierr);
      msolv_ctx->F = F;
      ierr = MatCreateVecs(F,&msolv_ctx->sol,NULL);CHKERRQ(ierr);
      /* currently PETSc has no support for MatSolve(F,x,x), so cheat and let rhs and sol share the same memory */
      {
        PetscScalar *array;
        PetscInt    n;

        ierr = VecGetLocalSize(msolv_ctx->sol,&n);CHKERRQ(ierr);
        ierr = VecGetArray(msolv_ctx->sol,&array);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PetscObjectComm((PetscObject)msolv_ctx->sol),1,n,array,&msolv_ctx->rhs);CHKERRQ(ierr);
        ierr = VecRestoreArray(msolv_ctx->sol,&array);CHKERRQ(ierr);
      }
      msolv_ctx->has_vertices = schur_has_vertices;

      /* interior solver */
      ierr = PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->interior_solver);CHKERRQ(ierr);
      ierr = PCSetOperators(msolv_ctx->interior_solver,A_II,A_II);CHKERRQ(ierr);
      ierr = PCSetType(msolv_ctx->interior_solver,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetName(msolv_ctx->interior_solver,"Interior solver (w/o Schur factorization)");CHKERRQ(ierr);
      ierr = PCShellSetContext(msolv_ctx->interior_solver,msolv_ctx);CHKERRQ(ierr);
      ierr = PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View);CHKERRQ(ierr);
      ierr = PCShellSetApply(msolv_ctx->interior_solver,PCBDDCReuseSolvers_Interior);CHKERRQ(ierr);
      ierr = PCShellSetApplyTranspose(msolv_ctx->interior_solver,PCBDDCReuseSolvers_InteriorTranspose);CHKERRQ(ierr);

      /* correction solver */
      ierr = PCCreate(PetscObjectComm((PetscObject)A_II),&msolv_ctx->correction_solver);CHKERRQ(ierr);
      ierr = PCSetType(msolv_ctx->correction_solver,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetName(msolv_ctx->correction_solver,"Correction solver (with Schur factorization)");CHKERRQ(ierr);
      ierr = PCShellSetContext(msolv_ctx->correction_solver,msolv_ctx);CHKERRQ(ierr);
      ierr = PCShellSetView(msolv_ctx->interior_solver,PCBDDCReuseSolvers_View);CHKERRQ(ierr);
      ierr = PCShellSetApply(msolv_ctx->correction_solver,PCBDDCReuseSolvers_Correction);CHKERRQ(ierr);
      ierr = PCShellSetApplyTranspose(msolv_ctx->correction_solver,PCBDDCReuseSolvers_CorrectionTranspose);CHKERRQ(ierr);

      /* scatter and vecs for Schur complement solver */
      ierr = MatCreateVecs(S_all,&msolv_ctx->sol_B,&msolv_ctx->rhs_B);CHKERRQ(ierr);
      ierr = MatCreateVecs(sub_schurs->S,&vec1_B,NULL);CHKERRQ(ierr);
      if (!schur_has_vertices) {
        ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_A_all,&msolv_ctx->is_B);CHKERRQ(ierr);
        ierr = VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,NULL,&msolv_ctx->correction_scatter_B);CHKERRQ(ierr);
        ierr = PetscObjectReference((PetscObject)is_A_all);CHKERRQ(ierr);
        msolv_ctx->is_R = is_A_all;
      } else {
        IS              is_B_all;
        const PetscInt* idxs;
        PetscInt        dual,n_v,n;

        ierr = ISGetLocalSize(sub_schurs->is_vertices,&n_v);CHKERRQ(ierr);
        dual = size_schur - n_v;
        ierr = ISGetLocalSize(is_A_all,&n);CHKERRQ(ierr);
        ierr = ISGetIndices(is_A_all,&idxs);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),dual,idxs+n_I,PETSC_COPY_VALUES,&is_B_all);CHKERRQ(ierr);
        ierr = ISGlobalToLocalMappingApplyIS(sub_schurs->BtoNmap,IS_GTOLM_DROP,is_B_all,&msolv_ctx->is_B);CHKERRQ(ierr);
        ierr = ISDestroy(&is_B_all);CHKERRQ(ierr);
        ierr = ISCreateStride(PetscObjectComm((PetscObject)is_A_all),dual,0,1,&is_B_all);CHKERRQ(ierr);
        ierr = VecScatterCreate(vec1_B,msolv_ctx->is_B,msolv_ctx->sol_B,is_B_all,&msolv_ctx->correction_scatter_B);CHKERRQ(ierr);
        ierr = ISDestroy(&is_B_all);CHKERRQ(ierr);
        ierr = ISCreateGeneral(PetscObjectComm((PetscObject)is_A_all),n-n_v,idxs,PETSC_COPY_VALUES,&msolv_ctx->is_R);CHKERRQ(ierr);
        ierr = ISRestoreIndices(is_A_all,&idxs);CHKERRQ(ierr);
      }
      ierr = ISGetLocalSize(msolv_ctx->is_R,&n_R);CHKERRQ(ierr);
      ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n_R,n_R,0,NULL,&Afake);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(Afake,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(Afake,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = PCSetOperators(msolv_ctx->correction_solver,Afake,Afake);CHKERRQ(ierr);
      ierr = MatDestroy(&Afake);CHKERRQ(ierr);
      ierr = VecDestroy(&vec1_B);CHKERRQ(ierr);

      /* communicate benign info to solver context */
      if (benign_n) {
        PetscScalar *array;

        msolv_ctx->benign_n = benign_n;
        msolv_ctx->benign_zerodiag_subs = is_p_r;
        ierr = PetscMalloc1(benign_n,&msolv_ctx->benign_save_vals);CHKERRQ(ierr);
        msolv_ctx->benign_csAIB = cs_AIB_mat;
        ierr = MatCreateVecs(cs_AIB_mat,&msolv_ctx->benign_corr_work,NULL);CHKERRQ(ierr);
        ierr = VecGetArray(msolv_ctx->benign_corr_work,&array);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,size_schur,array,&msolv_ctx->benign_dummy_schur_vec);CHKERRQ(ierr);
        ierr = VecRestoreArray(msolv_ctx->benign_corr_work,&array);CHKERRQ(ierr);
        msolv_ctx->benign_AIIm1ones = benign_AIIm1_ones_mat;
      }
    } else {
      if (sub_schurs->reuse_solver) {
        ierr = PCBDDCReuseSolversReset(sub_schurs->reuse_solver);CHKERRQ(ierr);
      }
      ierr = PetscFree(sub_schurs->reuse_solver);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = ISDestroy(&is_A_all);CHKERRQ(ierr);

    /* Work arrays */
    ierr = PetscMalloc1(max_subset_size*max_subset_size,&work);CHKERRQ(ierr);

    /* S_Ej_all */
    cum = cum2 = 0;
    ierr = MatDenseGetArrayRead(S_all,&rS_data);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArray(sub_schurs->S_Ej_all,&SEj_arr);CHKERRQ(ierr);
    if (compute_Stilda) {
      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr);CHKERRQ(ierr);
    }
    for (i=0;i<sub_schurs->n_subs;i++) {
      PetscInt j;

      /* get S_E */
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
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
        ierr = KSPGetOperators(sub_schurs->change[i],&change_sub,NULL);CHKERRQ(ierr);
        ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj);CHKERRQ(ierr);
        if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
          Mat T2;
          ierr = MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2);CHKERRQ(ierr);
          ierr = MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T);CHKERRQ(ierr);
          ierr = MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T);CHKERRQ(ierr);
          ierr = MatDestroy(&T2);CHKERRQ(ierr);
        } else {
          ierr = MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T);CHKERRQ(ierr);
        }
        ierr = MatCopy(T,SEj,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
        ierr = MatDestroy(&T);CHKERRQ(ierr);
        ierr = MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1.0,NULL,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&SEj);CHKERRQ(ierr);
      }
      if (deluxe) {
        ierr = PetscArraycpy(SEj_arr,work,subset_size*subset_size);CHKERRQ(ierr);
        /* if adaptivity is requested, invert S_E blocks */
        if (compute_Stilda) {
          Mat               M;
          const PetscScalar *vals;
          PetscBool         isdense,isdensecuda;

          ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&M);CHKERRQ(ierr);
          ierr = MatSetOption(M,MAT_SPD,sub_schurs->is_posdef);CHKERRQ(ierr);
          ierr = MatSetOption(M,MAT_HERMITIAN,sub_schurs->is_hermitian);CHKERRQ(ierr);
          if (!PetscBTLookup(sub_schurs->is_edge,i)) {
            ierr = MatSetType(M,Stype);CHKERRQ(ierr);
          }
          ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQDENSE,&isdense);CHKERRQ(ierr);
          ierr = PetscObjectTypeCompare((PetscObject)M,MATSEQDENSECUDA,&isdensecuda);CHKERRQ(ierr);
          if (use_cholesky) {
            ierr = MatCholeskyFactor(M,NULL,NULL);CHKERRQ(ierr);
          } else {
            ierr = MatLUFactor(M,NULL,NULL,NULL);CHKERRQ(ierr);
          }
          if (isdense) {
            ierr = MatSeqDenseInvertFactors_Private(M);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
          } else if (isdensecuda) {
            ierr = MatSeqDenseCUDAInvertFactors_Private(M);CHKERRQ(ierr);
#endif
          } else SETERRQ1(PetscObjectComm((PetscObject)M),PETSC_ERR_SUP,"Not implemented for type %s",Stype);
          ierr = MatDenseGetArrayRead(M,&vals);CHKERRQ(ierr);
          ierr = PetscArraycpy(SEjinv_arr,vals,subset_size*subset_size);CHKERRQ(ierr);
          ierr = MatDenseRestoreArrayRead(M,&vals);CHKERRQ(ierr);
          ierr = MatDestroy(&M);CHKERRQ(ierr);
        }
      } else if (compute_Stilda) { /* not using deluxe */
        Mat         SEj;
        Vec         D;
        PetscScalar *array;

        ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj);CHKERRQ(ierr);
        ierr = VecGetArray(Dall,&array);CHKERRQ(ierr);
        ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,subset_size,array+cum,&D);CHKERRQ(ierr);
        ierr = VecRestoreArray(Dall,&array);CHKERRQ(ierr);
        ierr = VecShift(D,-1.);CHKERRQ(ierr);
        ierr = MatDiagonalScale(SEj,D,D);CHKERRQ(ierr);
        ierr = MatDestroy(&SEj);CHKERRQ(ierr);
        ierr = VecDestroy(&D);CHKERRQ(ierr);
        ierr = PetscArraycpy(SEj_arr,work,subset_size*subset_size);CHKERRQ(ierr);
      }
      cum += subset_size;
      cum2 += subset_size*(size_schur + 1);
      SEj_arr += subset_size*subset_size;
      if (SEjinv_arr) SEjinv_arr += subset_size*subset_size;
    }
    ierr = MatDenseRestoreArrayRead(S_all,&rS_data);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&SEj_arr);CHKERRQ(ierr);
    if (compute_Stilda) {
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&SEjinv_arr);CHKERRQ(ierr);
    }
    if (solver_S) {
      ierr = MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
    }

    /* may prevent from unneeded copies, since MUMPS or MKL_Pardiso always use CPU memory
       however, preliminary tests indicate using GPUs is still faster in the solve phase */
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    if (reuse_solvers) {
      Mat                  St;
      MatFactorSchurStatus st;
      PetscBool            flg = PETSC_FALSE;

      ierr = PetscOptionsGetBool(NULL,sub_schurs->prefix,"-sub_schurs_schur_pin_to_cpu",&flg,NULL);CHKERRQ(ierr);
      ierr = MatFactorGetSchurComplement(F,&St,&st);CHKERRQ(ierr);
      ierr = MatBindToCPU(St,flg);CHKERRQ(ierr);
      ierr = MatFactorRestoreSchurComplement(F,&St,st);CHKERRQ(ierr);
    }
#endif

    schur_factor = NULL;
    if (compute_Stilda && size_active_schur) {

      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr);CHKERRQ(ierr);
      if (sub_schurs->n_subs == 1 && size_schur == size_active_schur && deluxe) { /* we already computed the inverse */
        ierr = PetscArraycpy(SEjinv_arr,work,size_schur*size_schur);CHKERRQ(ierr);
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
            ierr = MatFactorGetSchurComplement(F,&S_all_inv,NULL);CHKERRQ(ierr);
            ierr = MatDenseGetArray(S_all_inv,&data);CHKERRQ(ierr);
            if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
              ierr = ISGetLocalSize(sub_schurs->is_dir,&nd);CHKERRQ(ierr);
            }

            /* factor and invert activedofs and vertices (dirichlet dofs does not contribute) */
            if (schur_has_vertices) {
              Mat          M;
              PetscScalar *tdata;
              PetscInt     nv = 0, news;

              ierr = ISGetLocalSize(sub_schurs->is_vertices,&nv);CHKERRQ(ierr);
              news = size_active_schur + nv;
              ierr = PetscCalloc1(news*news,&tdata);CHKERRQ(ierr);
              for (i=0;i<size_active_schur;i++) {
                ierr = PetscArraycpy(tdata+i*(news+1),data+i*(size_schur+1),size_active_schur-i);CHKERRQ(ierr);
                ierr = PetscArraycpy(tdata+i*(news+1)+size_active_schur-i,data+i*size_schur+size_active_schur+nd,nv);CHKERRQ(ierr);
              }
              for (i=0;i<nv;i++) {
                PetscInt k = i+size_active_schur;
                ierr = PetscArraycpy(tdata+k*(news+1),data+(k+nd)*(size_schur+1),nv-i);CHKERRQ(ierr);
              }

              ierr = MatCreateSeqDense(PETSC_COMM_SELF,news,news,tdata,&M);CHKERRQ(ierr);
              ierr = MatSetOption(M,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
              ierr = MatCholeskyFactor(M,NULL,NULL);CHKERRQ(ierr);
              /* save the factors */
              cum = 0;
              ierr = PetscMalloc1((size_active_schur*(size_active_schur +1))/2+nd,&schur_factor);CHKERRQ(ierr);
              for (i=0;i<size_active_schur;i++) {
                ierr = PetscArraycpy(schur_factor+cum,tdata+i*(news+1),size_active_schur-i);CHKERRQ(ierr);
                cum += size_active_schur - i;
              }
              for (i=0;i<nd;i++) schur_factor[cum+i] = PetscSqrtReal(PetscRealPart(data[(i+size_active_schur)*(size_schur+1)]));
              ierr = MatSeqDenseInvertFactors_Private(M);CHKERRQ(ierr);
              /* move back just the active dofs to the Schur complement */
              for (i=0;i<size_active_schur;i++) {
                ierr = PetscArraycpy(data+i*size_schur,tdata+i*news,size_active_schur);CHKERRQ(ierr);
              }
              ierr = PetscFree(tdata);CHKERRQ(ierr);
              ierr = MatDestroy(&M);CHKERRQ(ierr);
            } else { /* we can factorize and invert just the activedofs */
              Mat         M;
              PetscScalar *aux;

              ierr = PetscMalloc1(nd,&aux);CHKERRQ(ierr);
              for (i=0;i<nd;i++) aux[i] = 1.0/data[(i+size_active_schur)*(size_schur+1)];
              ierr = MatCreateSeqDense(PETSC_COMM_SELF,size_active_schur,size_active_schur,data,&M);CHKERRQ(ierr);
              ierr = MatSeqDenseSetLDA(M,size_schur);CHKERRQ(ierr);
              ierr = MatSetOption(M,MAT_SPD,PETSC_TRUE);CHKERRQ(ierr);
              ierr = MatCholeskyFactor(M,NULL,NULL);CHKERRQ(ierr);
              ierr = MatSeqDenseInvertFactors_Private(M);CHKERRQ(ierr);
              ierr = MatDestroy(&M);CHKERRQ(ierr);
              ierr = MatCreateSeqDense(PETSC_COMM_SELF,size_schur,nd,data+size_active_schur*size_schur,&M);CHKERRQ(ierr);
              ierr = MatZeroEntries(M);CHKERRQ(ierr);
              ierr = MatDestroy(&M);CHKERRQ(ierr);
              ierr = MatCreateSeqDense(PETSC_COMM_SELF,nd,size_schur,data+size_active_schur,&M);CHKERRQ(ierr);
              ierr = MatSeqDenseSetLDA(M,size_schur);CHKERRQ(ierr);
              ierr = MatZeroEntries(M);CHKERRQ(ierr);
              ierr = MatDestroy(&M);CHKERRQ(ierr);
              for (i=0;i<nd;i++) data[(i+size_active_schur)*(size_schur+1)] = aux[i];
              ierr = PetscFree(aux);CHKERRQ(ierr);
            }
            ierr = MatDenseRestoreArray(S_all_inv,&data);CHKERRQ(ierr);
          } else { /* use MatFactor calls to invert S */
            ierr = MatFactorInvertSchurComplement(F);CHKERRQ(ierr);
            ierr = MatFactorGetSchurComplement(F,&S_all_inv,NULL);CHKERRQ(ierr);
          }
        } else { /* we need to invert explicitly since we are not using MatFactor for S */
          ierr = PetscObjectReference((PetscObject)S_all);CHKERRQ(ierr);
          S_all_inv = S_all;
          ierr = MatDenseGetArray(S_all_inv,&S_data);CHKERRQ(ierr);
          ierr = PetscBLASIntCast(size_schur,&B_N);CHKERRQ(ierr);
          ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
          if (use_potr) {
            PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,S_data,&B_N,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,S_data,&B_N,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
          } else if (use_sytr) {
            PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,S_data,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRF Lapack routine %d",(int)B_ierr);
#if defined(PETSC_MISSING_LAPACK_SYTRI)
            SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRI - Lapack routine is unavailable.");
#else
            PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&B_N,S_data,&B_N,pivots,Bwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRI Lapack routine %d",(int)B_ierr);
#endif
          } else {
            PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,S_data,&B_N,pivots,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
            PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,S_data,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
            if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
          }
          ierr = PetscLogFlops(1.0*size_schur*size_schur*size_schur);CHKERRQ(ierr);
          ierr = PetscFPTrapPop();CHKERRQ(ierr);
          ierr = MatDenseRestoreArray(S_all_inv,&S_data);CHKERRQ(ierr);
        }
        /* S_Ej_tilda_all */
        cum = cum2 = 0;
        ierr = MatDenseGetArrayRead(S_all_inv,&rS_data);CHKERRQ(ierr);
        for (i=0;i<sub_schurs->n_subs;i++) {
          PetscInt j;

          ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
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
            ierr = KSPGetOperators(sub_schurs->change[i],&change_sub,NULL);CHKERRQ(ierr);
            ierr = MatCreateSeqDense(PETSC_COMM_SELF,subset_size,subset_size,work,&SEj);CHKERRQ(ierr);
            if (!sub_schurs->change_with_qr) { /* currently there's no support for PtAP with P SeqAIJ */
              Mat T2;
              ierr = MatTransposeMatMult(change_sub,SEj,MAT_INITIAL_MATRIX,1.0,&T2);CHKERRQ(ierr);
              ierr = MatMatMult(T2,change_sub,MAT_INITIAL_MATRIX,1.0,&T);CHKERRQ(ierr);
              ierr = MatDestroy(&T2);CHKERRQ(ierr);
              ierr = MatConvert(T,MATSEQDENSE,MAT_INPLACE_MATRIX,&T);CHKERRQ(ierr);
            } else {
              ierr = MatPtAP(SEj,change_sub,MAT_INITIAL_MATRIX,1.0,&T);CHKERRQ(ierr);
            }
            ierr = MatCopy(T,SEj,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
            ierr = MatDestroy(&T);CHKERRQ(ierr);
            /* set diagonal entry to a very large value to pick the basis we are eliminating as the first eigenvectors with adaptive selection */
            ierr = MatZeroRowsColumnsIS(SEj,sub_schurs->change_primal_sub[i],1./PETSC_SMALL,NULL,NULL);CHKERRQ(ierr);
            ierr = MatDestroy(&SEj);CHKERRQ(ierr);
          }
          ierr = PetscArraycpy(SEjinv_arr,work,subset_size*subset_size);CHKERRQ(ierr);
          cum += subset_size;
          cum2 += subset_size*(size_schur + 1);
          SEjinv_arr += subset_size*subset_size;
        }
        ierr = MatDenseRestoreArrayRead(S_all_inv,&rS_data);CHKERRQ(ierr);
        if (solver_S) {
          if (schur_has_vertices) {
            ierr = MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_FACTORED);CHKERRQ(ierr);
          } else {
            ierr = MatFactorRestoreSchurComplement(F,&S_all_inv,MAT_FACTOR_SCHUR_INVERTED);CHKERRQ(ierr);
          }
        }
        ierr = MatDestroy(&S_all_inv);CHKERRQ(ierr);
      }
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&SEjinv_arr);CHKERRQ(ierr);

      /* move back factors if needed */
      if (schur_has_vertices) {
        Mat      S_tmp;
        PetscInt nd = 0;

        if (!solver_S) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
        ierr = MatFactorGetSchurComplement(F,&S_tmp,NULL);CHKERRQ(ierr);
        if (use_potr) {
          PetscScalar *data;

          ierr = MatDenseGetArray(S_tmp,&data);CHKERRQ(ierr);
          ierr = PetscArrayzero(data,size_schur*size_schur);CHKERRQ(ierr);

          if (S_lower_triangular) {
            cum = 0;
            for (i=0;i<size_active_schur;i++) {
              ierr = PetscArraycpy(data+i*(size_schur+1),schur_factor+cum,size_active_schur-i);CHKERRQ(ierr);
              cum += size_active_schur-i;
	    }
          } else {
            ierr = PetscArraycpy(data,schur_factor,size_schur*size_schur);CHKERRQ(ierr);
          }
          if (sub_schurs->is_dir) {
            ierr = ISGetLocalSize(sub_schurs->is_dir,&nd);CHKERRQ(ierr);
	    for (i=0;i<nd;i++) {
	      data[(i+size_active_schur)*(size_schur+1)] = schur_factor[cum+i];
	    }
          }
          /* workaround: since I cannot modify the matrices used inside the solvers for the forward and backward substitutions,
             set the diagonal entry of the Schur factor to a very large value */
          for (i=size_active_schur+nd;i<size_schur;i++) {
            data[i*(size_schur+1)] = infty;
          }
          ierr = MatDenseRestoreArray(S_tmp,&data);CHKERRQ(ierr);
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor update not yet implemented for non SPD matrices");
        ierr = MatFactorRestoreSchurComplement(F,&S_tmp,MAT_FACTOR_SCHUR_FACTORED);CHKERRQ(ierr);
      }
    } else if (factor_workaround) { /* we need to eliminate any unneeded coupling */
      PetscScalar *data;
      PetscInt    nd = 0;

      if (sub_schurs->is_dir) { /* dirichlet dofs could have different scalings */
        ierr = ISGetLocalSize(sub_schurs->is_dir,&nd);CHKERRQ(ierr);
      }
      ierr = MatFactorGetSchurComplement(F,&S_all,NULL);CHKERRQ(ierr);
      ierr = MatDenseGetArray(S_all,&data);CHKERRQ(ierr);
      for (i=0;i<size_active_schur;i++) {
        ierr = PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur);CHKERRQ(ierr);
      }
      for (i=size_active_schur+nd;i<size_schur;i++) {
        ierr = PetscArrayzero(data+i*size_schur+size_active_schur,size_schur-size_active_schur);CHKERRQ(ierr);
        data[i*(size_schur+1)] = infty;
      }
      ierr = MatDenseRestoreArray(S_all,&data);CHKERRQ(ierr);
      ierr = MatFactorRestoreSchurComplement(F,&S_all,MAT_FACTOR_SCHUR_UNFACTORED);CHKERRQ(ierr);
    }
    ierr = PetscFree(work);CHKERRQ(ierr);
    ierr = PetscFree(schur_factor);CHKERRQ(ierr);
    ierr = VecDestroy(&Dall);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&is_I_layer);CHKERRQ(ierr);
  ierr = MatDestroy(&S_all);CHKERRQ(ierr);
  ierr = MatDestroy(&A_BB);CHKERRQ(ierr);
  ierr = MatDestroy(&A_IB);CHKERRQ(ierr);
  ierr = MatDestroy(&A_BI);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  ierr = PetscMalloc1(sub_schurs->n_subs,&nnz);CHKERRQ(ierr);
  for (i=0;i<sub_schurs->n_subs;i++) {
    ierr = ISGetLocalSize(sub_schurs->is_subs[i],&nnz[i]);CHKERRQ(ierr);
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,sub_schurs->n_subs,nnz,PETSC_OWN_POINTER,&is_I_layer);CHKERRQ(ierr);
  ierr = MatSetVariableBlockSizes(sub_schurs->S_Ej_all,sub_schurs->n_subs,nnz);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(sub_schurs->S_Ej_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (compute_Stilda) {
    ierr = MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_tilda_all,sub_schurs->n_subs,nnz);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(sub_schurs->sum_S_Ej_tilda_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (deluxe) {
      ierr = MatSetVariableBlockSizes(sub_schurs->sum_S_Ej_inv_all,sub_schurs->n_subs,nnz);CHKERRQ(ierr);
      ierr = MatAssemblyBegin(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(sub_schurs->sum_S_Ej_inv_all,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }
  ierr = ISDestroy(&is_I_layer);CHKERRQ(ierr);

  /* Get local part of (\sum_j S_Ej) */
  if (!sub_schurs->sum_S_Ej_all) {
    ierr = MatDuplicate(sub_schurs->S_Ej_all,MAT_DO_NOT_COPY_VALUES,&sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
  }
  ierr = VecSet(gstash,0.0);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(sub_schurs->S_Ej_all,&stasharray);CHKERRQ(ierr);
  ierr = VecPlaceArray(lstash,stasharray);CHKERRQ(ierr);
  ierr = VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(sub_schurs->S_Ej_all,&stasharray);CHKERRQ(ierr);
  ierr = VecResetArray(lstash);CHKERRQ(ierr);
  ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_all,&stasharray);CHKERRQ(ierr);
  ierr = VecPlaceArray(lstash,stasharray);CHKERRQ(ierr);
  ierr = VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_all,&stasharray);CHKERRQ(ierr);
  ierr = VecResetArray(lstash);CHKERRQ(ierr);

  /* Get local part of (\sum_j S^-1_Ej) (\sum_j St^-1_Ej) */
  if (compute_Stilda) {
    ierr = VecSet(gstash,0.0);CHKERRQ(ierr);
    ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray);CHKERRQ(ierr);
    ierr = VecPlaceArray(lstash,stasharray);CHKERRQ(ierr);
    ierr = VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&stasharray);CHKERRQ(ierr);
    ierr = VecResetArray(lstash);CHKERRQ(ierr);
    if (deluxe) {
      ierr = VecSet(gstash,0.0);CHKERRQ(ierr);
      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_inv_all,&stasharray);CHKERRQ(ierr);
      ierr = VecPlaceArray(lstash,stasharray);CHKERRQ(ierr);
      ierr = VecScatterBegin(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(sstash,lstash,gstash,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterBegin(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = VecScatterEnd(sstash,gstash,lstash,INSERT_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_inv_all,&stasharray);CHKERRQ(ierr);
      ierr = VecResetArray(lstash);CHKERRQ(ierr);
    } else {
      PetscScalar *array;
      PetscInt    cum;

      ierr = MatSeqAIJGetArray(sub_schurs->sum_S_Ej_tilda_all,&array);CHKERRQ(ierr);
      cum = 0;
      for (i=0;i<sub_schurs->n_subs;i++) {
        ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
        ierr = PetscBLASIntCast(subset_size,&B_N);CHKERRQ(ierr);
        ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
        if (use_potr) {
          PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&B_N,array+cum,&B_N,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&B_N,array+cum,&B_N,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in POTRI Lapack routine %d",(int)B_ierr);
        } else if (use_sytr) {
          PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&B_N,array+cum,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRF Lapack routine %d",(int)B_ierr);
#if defined(PETSC_MISSING_LAPACK_SYTRI)
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SYTRI - Lapack routine is unavailable.");
#else
          PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&B_N,array+cum,&B_N,pivots,Bwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in SYTRI Lapack routine %d",(int)B_ierr);
#endif
        } else {
          PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&B_N,&B_N,array+cum,&B_N,pivots,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRF Lapack routine %d",(int)B_ierr);
          PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&B_N,array+cum,&B_N,pivots,Bwork,&B_lwork,&B_ierr));
          if (B_ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in GETRI Lapack routine %d",(int)B_ierr);
        }
        ierr = PetscLogFlops(1.0*subset_size*subset_size*subset_size);CHKERRQ(ierr);
        ierr = PetscFPTrapPop();CHKERRQ(ierr);
        cum += subset_size*subset_size;
      }
      ierr = MatSeqAIJRestoreArray(sub_schurs->sum_S_Ej_tilda_all,&array);CHKERRQ(ierr);
      ierr = PetscObjectReference((PetscObject)sub_schurs->sum_S_Ej_all);CHKERRQ(ierr);
      ierr = MatDestroy(&sub_schurs->sum_S_Ej_inv_all);CHKERRQ(ierr);
      sub_schurs->sum_S_Ej_inv_all = sub_schurs->sum_S_Ej_all;
    }
  }
  ierr = VecDestroy(&lstash);CHKERRQ(ierr);
  ierr = VecDestroy(&gstash);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&sstash);CHKERRQ(ierr);

  if (matl_dbg_viewer) {
    PetscInt cum;

    if (sub_schurs->S_Ej_all) {
      ierr = PetscObjectSetName((PetscObject)sub_schurs->S_Ej_all,"SE");CHKERRQ(ierr);
      ierr = MatView(sub_schurs->S_Ej_all,matl_dbg_viewer);CHKERRQ(ierr);
    }
    if (sub_schurs->sum_S_Ej_all) {
      ierr = PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_all,"SSE");CHKERRQ(ierr);
      ierr = MatView(sub_schurs->sum_S_Ej_all,matl_dbg_viewer);CHKERRQ(ierr);
    }
    if (sub_schurs->sum_S_Ej_inv_all) {
      ierr = PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_inv_all,"SSEm");CHKERRQ(ierr);
      ierr = MatView(sub_schurs->sum_S_Ej_inv_all,matl_dbg_viewer);CHKERRQ(ierr);
    }
    if (sub_schurs->sum_S_Ej_tilda_all) {
      ierr = PetscObjectSetName((PetscObject)sub_schurs->sum_S_Ej_tilda_all,"SSEt");CHKERRQ(ierr);
      ierr = MatView(sub_schurs->sum_S_Ej_tilda_all,matl_dbg_viewer);CHKERRQ(ierr);
    }
    for (i=0,cum=0;i<sub_schurs->n_subs;i++) {
      IS   is;
      char name[16];

      ierr = PetscSNPrintf(name,sizeof(name),"IE%D",i);CHKERRQ(ierr);
      ierr = ISGetLocalSize(sub_schurs->is_subs[i],&subset_size);CHKERRQ(ierr);
      ierr = ISCreateStride(PETSC_COMM_SELF,subset_size,cum,1,&is);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject)is,name);CHKERRQ(ierr);
      ierr = ISView(is,matl_dbg_viewer);CHKERRQ(ierr);
      ierr = ISDestroy(&is);CHKERRQ(ierr);
      cum += subset_size;
    }
  }

  /* free workspace */
  ierr = PetscViewerDestroy(&matl_dbg_viewer);CHKERRQ(ierr);
  ierr = PetscFree2(Bwork,pivots);CHKERRQ(ierr);
  ierr = PetscCommDestroy(&comm_n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursInit(PCBDDCSubSchurs sub_schurs, const char* prefix, IS is_I, IS is_B, PCBDDCGraph graph, ISLocalToGlobalMapping BtoNmap, PetscBool copycc)
{
  IS              *faces,*edges,*all_cc,vertices;
  PetscInt        i,n_faces,n_edges,n_all_cc;
  PetscBool       is_sorted,ispardiso,ismumps;
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
    if (copycc) {
      ierr = ISDuplicate(faces[i],&all_cc[i]);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)faces[i]);CHKERRQ(ierr);
      all_cc[i] = faces[i];
    }
  }
  for (i=0;i<n_edges;i++) {
    if (copycc) {
      ierr = ISDuplicate(edges[i],&all_cc[n_faces+i]);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectReference((PetscObject)edges[i]);CHKERRQ(ierr);
      all_cc[n_faces+i] = edges[i];
    }
    ierr = PetscBTSet(sub_schurs->is_edge,n_faces+i);CHKERRQ(ierr);
  }
  ierr = PetscObjectReference((PetscObject)vertices);CHKERRQ(ierr);
  sub_schurs->is_vertices = vertices;
  ierr = PCBDDCGraphRestoreCandidatesIS(graph,&n_faces,&faces,&n_edges,&edges,&vertices);CHKERRQ(ierr);
  sub_schurs->is_dir = NULL;
  ierr = PCBDDCGraphGetDirichletDofsB(graph,&sub_schurs->is_dir);CHKERRQ(ierr);

  /* Determine if MatFactor can be used */
  ierr = PetscStrallocpy(prefix,&sub_schurs->prefix);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MUMPS)
  ierr = PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMUMPS,64);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_MKL_PARDISO)
  ierr = PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,64);CHKERRQ(ierr);
#else
  ierr = PetscStrncpy(sub_schurs->mat_solver_type,MATSOLVERPETSC,64);CHKERRQ(ierr);
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
  ierr = PetscOptionsString("-sub_schurs_mat_solver_type","Specific direct solver to use",NULL,sub_schurs->mat_solver_type,sub_schurs->mat_solver_type,64,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sub_schurs_symmetric","Symmetric problem",NULL,sub_schurs->is_symmetric,&sub_schurs->is_symmetric,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sub_schurs_hermitian","Hermitian problem",NULL,sub_schurs->is_hermitian,&sub_schurs->is_hermitian,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sub_schurs_posdef","Positive definite problem",NULL,sub_schurs->is_posdef,&sub_schurs->is_posdef,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sub_schurs_restrictcomm","Restrict communicator on active processes only",NULL,sub_schurs->restrict_comm,&sub_schurs->restrict_comm,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-sub_schurs_debug","Debug output",NULL,sub_schurs->debug,&sub_schurs->debug,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMUMPS,&ismumps);CHKERRQ(ierr);
  ierr = PetscStrcmp(sub_schurs->mat_solver_type,MATSOLVERMKL_PARDISO,&ispardiso);CHKERRQ(ierr);
  sub_schurs->schur_explicit = (PetscBool)(ispardiso || ismumps);

  /* for reals, symmetric and hermitian are synonims */
#if !defined(PETSC_USE_COMPLEX)
  sub_schurs->is_symmetric = (PetscBool)(sub_schurs->is_symmetric && sub_schurs->is_hermitian);
  sub_schurs->is_hermitian = sub_schurs->is_symmetric;
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&schurs_ctx);CHKERRQ(ierr);
  schurs_ctx->n_subs = 0;
  *sub_schurs = schurs_ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursReset(PCBDDCSubSchurs sub_schurs)
{
  PetscInt       i;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!sub_schurs) PetscFunctionReturn(0);
  ierr = PetscFree(sub_schurs->prefix);CHKERRQ(ierr);
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
  if (sub_schurs->reuse_solver) {
    ierr = PCBDDCReuseSolversReset(sub_schurs->reuse_solver);CHKERRQ(ierr);
  }
  ierr = PetscFree(sub_schurs->reuse_solver);CHKERRQ(ierr);
  if (sub_schurs->change) {
    for (i=0;i<sub_schurs->n_subs;i++) {
      ierr = KSPDestroy(&sub_schurs->change[i]);CHKERRQ(ierr);
      ierr = ISDestroy(&sub_schurs->change_primal_sub[i]);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(sub_schurs->change);CHKERRQ(ierr);
  ierr = PetscFree(sub_schurs->change_primal_sub);CHKERRQ(ierr);
  sub_schurs->n_subs = 0;
  PetscFunctionReturn(0);
}

PetscErrorCode PCBDDCSubSchursDestroy(PCBDDCSubSchurs* sub_schurs)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCBDDCSubSchursReset(*sub_schurs);CHKERRQ(ierr);
  ierr = PetscFree(*sub_schurs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
