
/*  --------------------------------------------------------------------

     This file implements a subclass of the SeqAIJ matrix class that uses
     the SuperLU sparse solver.
*/

/*
     Defines the data structure for the base matrix type (SeqAIJ)
*/
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/

/*
     SuperLU include files
*/
EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #include <slu_cdefs.h>
  #else
    #include <slu_zdefs.h>
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #include <slu_sdefs.h>
  #else
    #include <slu_ddefs.h>
  #endif
#endif
#include <slu_util.h>
EXTERN_C_END

/*
     This is the data that defines the SuperLU factored matrix type
*/
typedef struct {
  SuperMatrix       A, L, U, B, X;
  superlu_options_t options;
  PetscInt         *perm_c; /* column permutation vector */
  PetscInt         *perm_r; /* row permutations from partial pivoting */
  PetscInt         *etree;
  PetscReal        *R, *C;
  char              equed[1];
  PetscInt          lwork;
  void             *work;
  PetscReal         rpg, rcond;
  mem_usage_t       mem_usage;
  MatStructure      flg;
  SuperLUStat_t     stat;
  Mat               A_dup;
  PetscScalar      *rhs_dup;
  GlobalLU_t        Glu;
  PetscBool         needconversion;

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscBool CleanUpSuperLU;
} Mat_SuperLU;

/*
    Utility function
*/
static PetscErrorCode MatView_Info_SuperLU(Mat A, PetscViewer viewer)
{
  Mat_SuperLU      *lu = (Mat_SuperLU *)A->data;
  superlu_options_t options;

  PetscFunctionBegin;
  options = lu->options;

  PetscCall(PetscViewerASCIIPrintf(viewer, "SuperLU run parameters:\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  Equil: %s\n", (options.Equil != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ColPerm: %" PetscInt_FMT "\n", options.ColPerm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  IterRefine: %" PetscInt_FMT "\n", options.IterRefine));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  SymmetricMode: %s\n", (options.SymmetricMode != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  DiagPivotThresh: %g\n", options.DiagPivotThresh));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  PivotGrowth: %s\n", (options.PivotGrowth != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ConditionNumber: %s\n", (options.ConditionNumber != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  RowPerm: %" PetscInt_FMT "\n", options.RowPerm));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  ReplaceTinyPivot: %s\n", (options.ReplaceTinyPivot != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  PrintStat: %s\n", (options.PrintStat != NO) ? "YES" : "NO"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  lwork: %" PetscInt_FMT "\n", lu->lwork));
  if (A->factortype == MAT_FACTOR_ILU) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_DropTol: %g\n", options.ILU_DropTol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_FillTol: %g\n", options.ILU_FillTol));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_FillFactor: %g\n", options.ILU_FillFactor));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_DropRule: %" PetscInt_FMT "\n", options.ILU_DropRule));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_Norm: %" PetscInt_FMT "\n", options.ILU_Norm));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  ILU_MILU: %" PetscInt_FMT "\n", options.ILU_MILU));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SuperLU_Private(Mat A, Vec b, Vec x)
{
  Mat_SuperLU       *lu = (Mat_SuperLU *)A->data;
  const PetscScalar *barray;
  PetscScalar       *xarray;
  PetscInt           info, i, n;
  PetscReal          ferr, berr;
  static PetscBool   cite = PETSC_FALSE;

  PetscFunctionBegin;
  if (lu->lwork == -1) PetscFunctionReturn(0);
  PetscCall(PetscCitationsRegister("@article{superlu99,\n  author  = {James W. Demmel and Stanley C. Eisenstat and\n             John R. Gilbert and Xiaoye S. Li and Joseph W. H. Liu},\n  title = {A supernodal approach to sparse partial "
                                   "pivoting},\n  journal = {SIAM J. Matrix Analysis and Applications},\n  year = {1999},\n  volume  = {20},\n  number = {3},\n  pages = {720-755}\n}\n",
                                   &cite));

  PetscCall(VecGetLocalSize(x, &n));
  lu->B.ncol = 1; /* Set the number of right-hand side */
  if (lu->options.Equil && !lu->rhs_dup) {
    /* superlu overwrites b when Equil is used, thus create rhs_dup to keep user's b unchanged */
    PetscCall(PetscMalloc1(n, &lu->rhs_dup));
  }
  if (lu->options.Equil) {
    /* Copy b into rsh_dup */
    PetscCall(VecGetArrayRead(b, &barray));
    PetscCall(PetscArraycpy(lu->rhs_dup, barray, n));
    PetscCall(VecRestoreArrayRead(b, &barray));
    barray = lu->rhs_dup;
  } else {
    PetscCall(VecGetArrayRead(b, &barray));
  }
  PetscCall(VecGetArray(x, &xarray));

#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
  ((DNformat *)lu->B.Store)->nzval = (singlecomplex *)barray;
  ((DNformat *)lu->X.Store)->nzval = (singlecomplex *)xarray;
  #else
  ((DNformat *)lu->B.Store)->nzval = (doublecomplex *)barray;
  ((DNformat *)lu->X.Store)->nzval = (doublecomplex *)xarray;
  #endif
#else
  ((DNformat *)lu->B.Store)->nzval = (void *)barray;
  ((DNformat *)lu->X.Store)->nzval = xarray;
#endif

  lu->options.Fact = FACTORED; /* Indicate the factored form of A is supplied. */
  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:cgssvx", cgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #else
    PetscStackCallExternalVoid("SuperLU:zgssvx", zgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:sgssvx", sgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #else
    PetscStackCallExternalVoid("SuperLU:dgssvx", dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #endif
#endif
  } else if (A->factortype == MAT_FACTOR_ILU) {
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:cgsisx", cgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #else
    PetscStackCallExternalVoid("SuperLU:zgsisx", zgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:sgsisx", sgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #else
    PetscStackCallExternalVoid("SuperLU:dgsisx", dgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &info));
  #endif
#endif
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");
  if (!lu->options.Equil) PetscCall(VecRestoreArrayRead(b, &barray));
  PetscCall(VecRestoreArray(x, &xarray));

  if (!info || info == lu->A.ncol + 1) {
    if (lu->options.IterRefine) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Iterative Refinement:\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %8s%8s%16s%16s\n", "rhs", "Steps", "FERR", "BERR"));
      for (i = 0; i < 1; ++i) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  %8d%8d%16e%16e\n", i + 1, lu->stat.RefineSteps, ferr, berr));
    }
  } else if (info > 0) {
    if (lu->lwork == -1) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  ** Estimated memory: %" PetscInt_FMT " bytes\n", info - lu->A.ncol));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Warning: gssvx() returns info %" PetscInt_FMT "\n", info));
    }
  } else PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "info = %" PetscInt_FMT ", the %" PetscInt_FMT "-th argument in gssvx() had an illegal value", info, -info);

  if (lu->options.PrintStat) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatSolve__SuperLU():\n"));
    PetscStackCallExternalVoid("SuperLU:StatPrint", StatPrint(&lu->stat));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SuperLU(Mat A, Vec b, Vec x)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)A->data;

  PetscFunctionBegin;
  if (A->factorerrortype) {
    PetscCall(PetscInfo(A, "MatSolve is called with singular matrix factor, skip\n"));
    PetscCall(VecSetInf(x));
    PetscFunctionReturn(0);
  }

  lu->options.Trans = TRANS;
  PetscCall(MatSolve_SuperLU_Private(A, b, x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolveTranspose_SuperLU(Mat A, Vec b, Vec x)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)A->data;

  PetscFunctionBegin;
  if (A->factorerrortype) {
    PetscCall(PetscInfo(A, "MatSolve is called with singular matrix factor, skip\n"));
    PetscCall(VecSetInf(x));
    PetscFunctionReturn(0);
  }

  lu->options.Trans = NOTRANS;
  PetscCall(MatSolve_SuperLU_Private(A, b, x));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_SuperLU(Mat F, Mat A, const MatFactorInfo *info)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)F->data;
  Mat_SeqAIJ  *aa;
  PetscInt     sinfo;
  PetscReal    ferr, berr;
  NCformat    *Ustore;
  SCformat    *Lstore;

  PetscFunctionBegin;
  if (lu->flg == SAME_NONZERO_PATTERN) { /* successing numerical factorization */
    lu->options.Fact = SamePattern;
    /* Ref: ~SuperLU_3.0/EXAMPLE/dlinsolx2.c */
    Destroy_SuperMatrix_Store(&lu->A);
    if (lu->A_dup) PetscCall(MatCopy_SeqAIJ(A, lu->A_dup, SAME_NONZERO_PATTERN));

    if (lu->lwork >= 0) {
      PetscStackCallExternalVoid("SuperLU:Destroy_SuperNode_Matrix", Destroy_SuperNode_Matrix(&lu->L));
      PetscStackCallExternalVoid("SuperLU:Destroy_CompCol_Matrix", Destroy_CompCol_Matrix(&lu->U));
      lu->options.Fact = SamePattern;
    }
  }

  /* Create the SuperMatrix for lu->A=A^T:
       Since SuperLU likes column-oriented matrices,we pass it the transpose,
       and then solve A^T X = B in MatSolve(). */
  if (lu->A_dup) {
    aa = (Mat_SeqAIJ *)(lu->A_dup)->data;
  } else {
    aa = (Mat_SeqAIJ *)(A)->data;
  }
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCallExternalVoid("SuperLU:cCreate_CompCol_Matrix", cCreate_CompCol_Matrix(&lu->A, A->cmap->n, A->rmap->n, aa->nz, (singlecomplex *)aa->a, aa->j, aa->i, SLU_NC, SLU_C, SLU_GE));
  #else
  PetscStackCallExternalVoid("SuperLU:zCreate_CompCol_Matrix", zCreate_CompCol_Matrix(&lu->A, A->cmap->n, A->rmap->n, aa->nz, (doublecomplex *)aa->a, aa->j, aa->i, SLU_NC, SLU_Z, SLU_GE));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCallExternalVoid("SuperLU:sCreate_CompCol_Matrix", sCreate_CompCol_Matrix(&lu->A, A->cmap->n, A->rmap->n, aa->nz, aa->a, aa->j, aa->i, SLU_NC, SLU_S, SLU_GE));
  #else
  PetscStackCallExternalVoid("SuperLU:dCreate_CompCol_Matrix", dCreate_CompCol_Matrix(&lu->A, A->cmap->n, A->rmap->n, aa->nz, aa->a, aa->j, aa->i, SLU_NC, SLU_D, SLU_GE));
  #endif
#endif

  /* Numerical factorization */
  lu->B.ncol = 0; /* Indicate not to solve the system */
  if (F->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:cgssvx", cgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #else
    PetscStackCallExternalVoid("SuperLU:zgssvx", zgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:sgssvx", sgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #else
    PetscStackCallExternalVoid("SuperLU:dgssvx", dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #endif
#endif
  } else if (F->factortype == MAT_FACTOR_ILU) {
    /* Compute the incomplete factorization, condition number and pivot growth */
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:cgsisx", cgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #else
    PetscStackCallExternalVoid("SuperLU:zgsisx", zgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    PetscStackCallExternalVoid("SuperLU:sgsisx", sgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #else
    PetscStackCallExternalVoid("SuperLU:dgsisx", dgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &lu->Glu, &lu->mem_usage, &lu->stat, &sinfo));
  #endif
#endif
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");
  if (!sinfo || sinfo == lu->A.ncol + 1) {
    if (lu->options.PivotGrowth) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Recip. pivot growth = %e\n", lu->rpg));
    if (lu->options.ConditionNumber) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Recip. condition number = %e\n", lu->rcond));
  } else if (sinfo > 0) {
    if (A->erroriffailure) {
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_MAT_LU_ZRPVT, "Zero pivot in row %" PetscInt_FMT, sinfo);
    } else {
      if (sinfo <= lu->A.ncol) {
        if (lu->options.ILU_FillTol == 0.0) F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
        PetscCall(PetscInfo(F, "Number of zero pivots %" PetscInt_FMT ", ILU_FillTol %g\n", sinfo, lu->options.ILU_FillTol));
      } else if (sinfo == lu->A.ncol + 1) {
        /*
         U is nonsingular, but RCOND is less than machine
                      precision, meaning that the matrix is singular to
                      working precision. Nevertheless, the solution and
                      error bounds are computed because there are a number
                      of situations where the computed solution can be more
                      accurate than the value of RCOND would suggest.
         */
        PetscCall(PetscInfo(F, "Matrix factor U is nonsingular, but is singular to working precision. The solution is computed. info %" PetscInt_FMT, sinfo));
      } else { /* sinfo > lu->A.ncol + 1 */
        F->factorerrortype = MAT_FACTOR_OUTMEMORY;
        PetscCall(PetscInfo(F, "Number of bytes allocated when memory allocation fails %" PetscInt_FMT "\n", sinfo));
      }
    }
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "info = %" PetscInt_FMT ", the %" PetscInt_FMT "-th argument in gssvx() had an illegal value", sinfo, -sinfo);

  if (lu->options.PrintStat) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatLUFactorNumeric_SuperLU():\n"));
    PetscStackCallExternalVoid("SuperLU:StatPrint", StatPrint(&lu->stat));
    Lstore = (SCformat *)lu->L.Store;
    Ustore = (NCformat *)lu->U.Store;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  No of nonzeros in factor L = %" PetscInt_FMT "\n", Lstore->nnz));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  No of nonzeros in factor U = %" PetscInt_FMT "\n", Ustore->nnz));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  No of nonzeros in L+U = %" PetscInt_FMT "\n", Lstore->nnz + Ustore->nnz - lu->A.ncol));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  L\\U MB %.3f\ttotal MB needed %.3f\n", lu->mem_usage.for_lu / 1e6, lu->mem_usage.total_needed / 1e6));
  }

  lu->flg                = SAME_NONZERO_PATTERN;
  F->ops->solve          = MatSolve_SuperLU;
  F->ops->solvetranspose = MatSolveTranspose_SuperLU;
  F->ops->matsolve       = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SuperLU(Mat A)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)A->data;

  PetscFunctionBegin;
  if (lu->CleanUpSuperLU) { /* Free the SuperLU datastructures */
    PetscStackCallExternalVoid("SuperLU:Destroy_SuperMatrix_Store", Destroy_SuperMatrix_Store(&lu->A));
    PetscStackCallExternalVoid("SuperLU:Destroy_SuperMatrix_Store", Destroy_SuperMatrix_Store(&lu->B));
    PetscStackCallExternalVoid("SuperLU:Destroy_SuperMatrix_Store", Destroy_SuperMatrix_Store(&lu->X));
    PetscStackCallExternalVoid("SuperLU:StatFree", StatFree(&lu->stat));
    if (lu->lwork >= 0) {
      PetscStackCallExternalVoid("SuperLU:Destroy_SuperNode_Matrix", Destroy_SuperNode_Matrix(&lu->L));
      PetscStackCallExternalVoid("SuperLU:Destroy_CompCol_Matrix", Destroy_CompCol_Matrix(&lu->U));
    }
  }
  PetscCall(PetscFree(lu->etree));
  PetscCall(PetscFree(lu->perm_r));
  PetscCall(PetscFree(lu->perm_c));
  PetscCall(PetscFree(lu->R));
  PetscCall(PetscFree(lu->C));
  PetscCall(PetscFree(lu->rhs_dup));
  PetscCall(MatDestroy(&lu->A_dup));
  PetscCall(PetscFree(A->data));

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSuperluSetILUDropTol_C", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SuperLU(Mat A, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_SuperLU(A, viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatSolve_SuperLU(Mat A, Mat B, Mat X)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)A->data;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix B must be MATDENSE matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix X must be MATDENSE matrix");
  lu->options.Trans = TRANS;
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatMatSolve_SuperLU() is not implemented yet");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SuperLU(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)(F->data);
  PetscInt     indx;
  PetscBool    flg, set;
  PetscReal    real_input;
  const char  *colperm[]    = {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}; /* MY_PERMC - not supported by the petsc interface yet */
  const char  *iterrefine[] = {"NOREFINE", "SINGLE", "DOUBLE", "EXTRA"};
  const char  *rowperm[]    = {"NOROWPERM", "LargeDiag"}; /* MY_PERMC - not supported by the petsc interface yet */

  PetscFunctionBegin;
  /* Set options to F */
  PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "SuperLU Options", "Mat");
  PetscCall(PetscOptionsBool("-mat_superlu_equil", "Equil", "None", (PetscBool)lu->options.Equil, (PetscBool *)&lu->options.Equil, NULL));
  PetscCall(PetscOptionsEList("-mat_superlu_colperm", "ColPerm", "None", colperm, 4, colperm[3], &indx, &flg));
  if (flg) lu->options.ColPerm = (colperm_t)indx;
  PetscCall(PetscOptionsEList("-mat_superlu_iterrefine", "IterRefine", "None", iterrefine, 4, iterrefine[0], &indx, &flg));
  if (flg) lu->options.IterRefine = (IterRefine_t)indx;
  PetscCall(PetscOptionsBool("-mat_superlu_symmetricmode", "SymmetricMode", "None", (PetscBool)lu->options.SymmetricMode, &flg, &set));
  if (set && flg) lu->options.SymmetricMode = YES;
  PetscCall(PetscOptionsReal("-mat_superlu_diagpivotthresh", "DiagPivotThresh", "None", lu->options.DiagPivotThresh, &real_input, &flg));
  if (flg) lu->options.DiagPivotThresh = (double)real_input;
  PetscCall(PetscOptionsBool("-mat_superlu_pivotgrowth", "PivotGrowth", "None", (PetscBool)lu->options.PivotGrowth, &flg, &set));
  if (set && flg) lu->options.PivotGrowth = YES;
  PetscCall(PetscOptionsBool("-mat_superlu_conditionnumber", "ConditionNumber", "None", (PetscBool)lu->options.ConditionNumber, &flg, &set));
  if (set && flg) lu->options.ConditionNumber = YES;
  PetscCall(PetscOptionsEList("-mat_superlu_rowperm", "rowperm", "None", rowperm, 2, rowperm[lu->options.RowPerm], &indx, &flg));
  if (flg) lu->options.RowPerm = (rowperm_t)indx;
  PetscCall(PetscOptionsBool("-mat_superlu_replacetinypivot", "ReplaceTinyPivot", "None", (PetscBool)lu->options.ReplaceTinyPivot, &flg, &set));
  if (set && flg) lu->options.ReplaceTinyPivot = YES;
  PetscCall(PetscOptionsBool("-mat_superlu_printstat", "PrintStat", "None", (PetscBool)lu->options.PrintStat, &flg, &set));
  if (set && flg) lu->options.PrintStat = YES;
  PetscCall(PetscOptionsInt("-mat_superlu_lwork", "size of work array in bytes used by factorization", "None", lu->lwork, &lu->lwork, NULL));
  if (lu->lwork > 0) {
    /* lwork is in bytes, hence PetscMalloc() is used here, not PetscMalloc1()*/
    PetscCall(PetscMalloc(lu->lwork, &lu->work));
  } else if (lu->lwork != 0 && lu->lwork != -1) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "   Warning: lwork %" PetscInt_FMT " is not supported by SUPERLU. The default lwork=0 is used.\n", lu->lwork));
    lu->lwork = 0;
  }
  /* ilu options */
  PetscCall(PetscOptionsReal("-mat_superlu_ilu_droptol", "ILU_DropTol", "None", lu->options.ILU_DropTol, &real_input, &flg));
  if (flg) lu->options.ILU_DropTol = (double)real_input;
  PetscCall(PetscOptionsReal("-mat_superlu_ilu_filltol", "ILU_FillTol", "None", lu->options.ILU_FillTol, &real_input, &flg));
  if (flg) lu->options.ILU_FillTol = (double)real_input;
  PetscCall(PetscOptionsReal("-mat_superlu_ilu_fillfactor", "ILU_FillFactor", "None", lu->options.ILU_FillFactor, &real_input, &flg));
  if (flg) lu->options.ILU_FillFactor = (double)real_input;
  PetscCall(PetscOptionsInt("-mat_superlu_ilu_droprull", "ILU_DropRule", "None", lu->options.ILU_DropRule, &lu->options.ILU_DropRule, NULL));
  PetscCall(PetscOptionsInt("-mat_superlu_ilu_norm", "ILU_Norm", "None", lu->options.ILU_Norm, &indx, &flg));
  if (flg) lu->options.ILU_Norm = (norm_t)indx;
  PetscCall(PetscOptionsInt("-mat_superlu_ilu_milu", "ILU_MILU", "None", lu->options.ILU_MILU, &indx, &flg));
  if (flg) lu->options.ILU_MILU = (milu_t)indx;
  PetscOptionsEnd();

  lu->flg                 = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpSuperLU      = PETSC_TRUE;
  F->ops->lufactornumeric = MatLUFactorNumeric_SuperLU;

  /* if we are here, the nonzero pattern has changed unless the user explicitly called MatLUFactorSymbolic */
  PetscCall(MatDestroy(&lu->A_dup));
  if (lu->needconversion) PetscCall(MatConvert(A, MATSEQAIJ, MAT_INITIAL_MATRIX, &lu->A_dup));
  if (lu->options.Equil == YES && !lu->A_dup) { /* superlu overwrites input matrix and rhs when Equil is used, thus create A_dup to keep user's A unchanged */
    PetscCall(MatDuplicate_SeqAIJ(A, MAT_COPY_VALUES, &lu->A_dup));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSuperluSetILUDropTol_SuperLU(Mat F, PetscReal dtol)
{
  Mat_SuperLU *lu = (Mat_SuperLU *)F->data;

  PetscFunctionBegin;
  lu->options.ILU_DropTol = dtol;
  PetscFunctionReturn(0);
}

/*@
  MatSuperluSetILUDropTol - Set SuperLU ILU drop tolerance

   Logically Collective on F

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-SuperLU interface
-  dtol - drop tolerance

  Options Database Key:
.   -mat_superlu_ilu_droptol <dtol> - the drop tolerance

   Level: beginner

   References:
.  * - SuperLU Users' Guide

.seealso: `MatGetFactor()`
@*/
PetscErrorCode MatSuperluSetILUDropTol(Mat F, PetscReal dtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(F, dtol, 2);
  PetscTryMethod(F, "MatSuperluSetILUDropTol_C", (Mat, PetscReal), (F, dtol));
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_superlu(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSUPERLU;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSUPERLU = "superlu" - A solver package providing solvers LU and ILU for sequential matrices
  via the external package SuperLU.

  Use ./configure --download-superlu to have PETSc installed with SuperLU

  Use -pc_type lu -pc_factor_mat_solver_type superlu to use this direct solver

  Options Database Keys:
+ -mat_superlu_equil <FALSE>            - Equil (None)
. -mat_superlu_colperm <COLAMD>         - (choose one of) NATURAL MMD_ATA MMD_AT_PLUS_A COLAMD
. -mat_superlu_iterrefine <NOREFINE>    - (choose one of) NOREFINE SINGLE DOUBLE EXTRA
. -mat_superlu_symmetricmode: <FALSE>   - SymmetricMode (None)
. -mat_superlu_diagpivotthresh <1>      - DiagPivotThresh (None)
. -mat_superlu_pivotgrowth <FALSE>      - PivotGrowth (None)
. -mat_superlu_conditionnumber <FALSE>  - ConditionNumber (None)
. -mat_superlu_rowperm <NOROWPERM>      - (choose one of) NOROWPERM LargeDiag
. -mat_superlu_replacetinypivot <FALSE> - ReplaceTinyPivot (None)
. -mat_superlu_printstat <FALSE>        - PrintStat (None)
. -mat_superlu_lwork <0>                - size of work array in bytes used by factorization (None)
. -mat_superlu_ilu_droptol <0>          - ILU_DropTol (None)
. -mat_superlu_ilu_filltol <0>          - ILU_FillTol (None)
. -mat_superlu_ilu_fillfactor <0>       - ILU_FillFactor (None)
. -mat_superlu_ilu_droprull <0>         - ILU_DropRule (None)
. -mat_superlu_ilu_norm <0>             - ILU_Norm (None)
- -mat_superlu_ilu_milu <0>             - ILU_MILU (None)

   Notes:
    Do not confuse this with `MATSOLVERSUPERLU_DIST` which is for parallel sparse solves

    Cannot use ordering provided by PETSc, provides its own.

   Level: beginner

.seealso: `PCLU`, `PCILU`, `MATSOLVERSUPERLU_DIST`, `MATSOLVERMUMPS`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

static PetscErrorCode MatGetFactor_seqaij_superlu(Mat A, MatFactorType ftype, Mat *F)
{
  Mat          B;
  Mat_SuperLU *lu;
  PetscInt     m = A->rmap->n, n = A->cmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(PetscStrallocpy("superlu", &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  B->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_SuperLU;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_SuperLU;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSUPERLU, &B->solvertype));

  B->ops->getinfo = MatGetInfo_External;
  B->ops->destroy = MatDestroy_SuperLU;
  B->ops->view    = MatView_SuperLU;
  B->factortype   = ftype;
  B->assembled    = PETSC_TRUE; /* required by -ksp_view */
  B->preallocated = PETSC_TRUE;

  PetscCall(PetscNew(&lu));

  if (ftype == MAT_FACTOR_LU) {
    set_default_options(&lu->options);
    /* Comments from SuperLU_4.0/SRC/dgssvx.c:
      "Whether or not the system will be equilibrated depends on the
       scaling of the matrix A, but if equilibration is used, A is
       overwritten by diag(R)*A*diag(C) and B by diag(R)*B
       (if options->Trans=NOTRANS) or diag(C)*B (if options->Trans = TRANS or CONJ)."
     We set 'options.Equil = NO' as default because additional space is needed for it.
    */
    lu->options.Equil = NO;
  } else if (ftype == MAT_FACTOR_ILU) {
    /* Set the default input options of ilu: */
    PetscStackCallExternalVoid("SuperLU:ilu_set_default_options", ilu_set_default_options(&lu->options));
  }
  lu->options.PrintStat = NO;

  /* Initialize the statistics variables. */
  PetscStackCallExternalVoid("SuperLU:StatInit", StatInit(&lu->stat));
  lu->lwork = 0; /* allocate space internally by system malloc */

  /* Allocate spaces (notice sizes are for the transpose) */
  PetscCall(PetscMalloc1(m, &lu->etree));
  PetscCall(PetscMalloc1(n, &lu->perm_r));
  PetscCall(PetscMalloc1(m, &lu->perm_c));
  PetscCall(PetscMalloc1(n, &lu->R));
  PetscCall(PetscMalloc1(m, &lu->C));

  /* create rhs and solution x without allocate space for .Store */
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCallExternalVoid("SuperLU:cCreate_Dense_Matrix(", cCreate_Dense_Matrix(&lu->B, m, 1, NULL, m, SLU_DN, SLU_C, SLU_GE));
  PetscStackCallExternalVoid("SuperLU:cCreate_Dense_Matrix(", cCreate_Dense_Matrix(&lu->X, m, 1, NULL, m, SLU_DN, SLU_C, SLU_GE));
  #else
  PetscStackCallExternalVoid("SuperLU:zCreate_Dense_Matrix", zCreate_Dense_Matrix(&lu->B, m, 1, NULL, m, SLU_DN, SLU_Z, SLU_GE));
  PetscStackCallExternalVoid("SuperLU:zCreate_Dense_Matrix", zCreate_Dense_Matrix(&lu->X, m, 1, NULL, m, SLU_DN, SLU_Z, SLU_GE));
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
  PetscStackCallExternalVoid("SuperLU:sCreate_Dense_Matrix", sCreate_Dense_Matrix(&lu->B, m, 1, NULL, m, SLU_DN, SLU_S, SLU_GE));
  PetscStackCallExternalVoid("SuperLU:sCreate_Dense_Matrix", sCreate_Dense_Matrix(&lu->X, m, 1, NULL, m, SLU_DN, SLU_S, SLU_GE));
  #else
  PetscStackCallExternalVoid("SuperLU:dCreate_Dense_Matrix", dCreate_Dense_Matrix(&lu->B, m, 1, NULL, m, SLU_DN, SLU_D, SLU_GE));
  PetscStackCallExternalVoid("SuperLU:dCreate_Dense_Matrix", dCreate_Dense_Matrix(&lu->X, m, 1, NULL, m, SLU_DN, SLU_D, SLU_GE));
  #endif
#endif

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_superlu));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSuperluSetILUDropTol_C", MatSuperluSetILUDropTol_SuperLU));
  B->data = lu;

  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_seqsell_superlu(Mat A, MatFactorType ftype, Mat *F)
{
  Mat_SuperLU *lu;

  PetscFunctionBegin;
  PetscCall(MatGetFactor_seqaij_superlu(A, ftype, F));
  lu                 = (Mat_SuperLU *)((*F)->data);
  lu->needconversion = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SuperLU(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSUPERLU, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_seqaij_superlu));
  PetscCall(MatSolverTypeRegister(MATSOLVERSUPERLU, MATSEQAIJ, MAT_FACTOR_ILU, MatGetFactor_seqaij_superlu));
  PetscCall(MatSolverTypeRegister(MATSOLVERSUPERLU, MATSEQSELL, MAT_FACTOR_LU, MatGetFactor_seqsell_superlu));
  PetscCall(MatSolverTypeRegister(MATSOLVERSUPERLU, MATSEQSELL, MAT_FACTOR_ILU, MatGetFactor_seqsell_superlu));
  PetscFunctionReturn(0);
}
