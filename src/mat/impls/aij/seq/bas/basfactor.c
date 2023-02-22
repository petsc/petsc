
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/aij/seq/bas/spbas.h>

PetscErrorCode MatICCFactorSymbolic_SeqAIJ_Bas(Mat fact, Mat A, IS perm, const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data;
  Mat_SeqSBAIJ   *b;
  PetscBool       perm_identity, missing;
  PetscInt        reallocs = 0, i, *ai = a->i, *aj = a->j, am = A->rmap->n, *ui;
  const PetscInt *rip, *riip;
  PetscInt        j;
  PetscInt        d;
  PetscInt        ncols, *cols, *uj;
  PetscReal       fill = info->fill, levels = info->levels;
  IS              iperm;
  spbas_matrix    Pattern_0, Pattern_P;

  PetscFunctionBegin;
  PetscCheck(A->rmap->n == A->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Must be square matrix, rows %" PetscInt_FMT " columns %" PetscInt_FMT, A->rmap->n, A->cmap->n);
  PetscCall(MatMissingDiagonal(A, &missing, &d));
  PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, d);
  PetscCall(ISIdentity(perm, &perm_identity));
  PetscCall(ISInvertPermutation(perm, PETSC_DECIDE, &iperm));

  /* ICC(0) without matrix ordering: simply copies fill pattern */
  if (!levels && perm_identity) {
    PetscCall(PetscMalloc1(am + 1, &ui));
    ui[0] = 0;

    for (i = 0; i < am; i++) ui[i + 1] = ui[i] + ai[i + 1] - a->diag[i];
    PetscCall(PetscMalloc1(ui[am] + 1, &uj));
    cols = uj;
    for (i = 0; i < am; i++) {
      aj    = a->j + a->diag[i];
      ncols = ui[i + 1] - ui[i];
      for (j = 0; j < ncols; j++) *cols++ = *aj++;
    }
  } else { /* case: levels>0 || (levels=0 && !perm_identity) */
    PetscCall(ISGetIndices(iperm, &riip));
    PetscCall(ISGetIndices(perm, &rip));

    /* Create spbas_matrix for pattern */
    PetscCall(spbas_pattern_only(am, am, ai, aj, &Pattern_0));

    /* Apply the permutation */
    PetscCall(spbas_apply_reordering(&Pattern_0, rip, riip));

    /* Raise the power */
    PetscCall(spbas_power(Pattern_0, (int)levels + 1, &Pattern_P));
    PetscCall(spbas_delete(Pattern_0));

    /* Keep only upper triangle of pattern */
    PetscCall(spbas_keep_upper(&Pattern_P));

    /* Convert to Sparse Row Storage  */
    PetscCall(spbas_matrix_to_crs(Pattern_P, NULL, &ui, &uj));
    PetscCall(spbas_delete(Pattern_P));
  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */

  b               = (Mat_SeqSBAIJ *)(fact)->data;
  b->singlemalloc = PETSC_FALSE;

  PetscCall(PetscMalloc1(ui[am] + 1, &b->a));

  b->j    = uj;
  b->i    = ui;
  b->diag = NULL;
  b->ilen = NULL;
  b->imax = NULL;
  b->row  = perm;
  b->col  = perm;

  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(PetscObjectReference((PetscObject)perm));

  b->icol          = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  PetscCall(PetscMalloc1(am + 1, &b->solve_work));
  b->maxnz = b->nz = ui[am];
  b->free_a        = PETSC_TRUE;
  b->free_ij       = PETSC_TRUE;

  (fact)->info.factor_mallocs   = reallocs;
  (fact)->info.fill_ratio_given = fill;
  if (ai[am] != 0) {
    (fact)->info.fill_ratio_needed = ((PetscReal)ui[am]) / ((PetscReal)ai[am]);
  } else {
    (fact)->info.fill_ratio_needed = 0.0;
  }
  /*  (fact)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_inplace; */
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ_Bas(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat             C  = B;
  Mat_SeqSBAIJ   *b  = (Mat_SeqSBAIJ *)C->data;
  IS              ip = b->row, iip = b->icol;
  const PetscInt *rip, *riip;
  PetscInt        mbs = A->rmap->n, *bi = b->i, *bj = b->j;
  MatScalar      *ba      = b->a;
  PetscReal       shiftnz = info->shiftamount;
  PetscReal       droptol = -1;
  PetscBool       perm_identity;
  spbas_matrix    Pattern, matrix_L, matrix_LT;
  PetscReal       mem_reduction;

  PetscFunctionBegin;
  /* Reduce memory requirements:   erase values of B-matrix */
  PetscCall(PetscFree(ba));
  /*   Compress (maximum) sparseness pattern of B-matrix */
  PetscCall(spbas_compress_pattern(bi, bj, mbs, mbs, SPBAS_DIAGONAL_OFFSETS, &Pattern, &mem_reduction));
  PetscCall(PetscFree(bi));
  PetscCall(PetscFree(bj));

  PetscCall(PetscInfo(NULL, "    compression rate for spbas_compress_pattern %g \n", (double)mem_reduction));

  /* Make Cholesky decompositions with larger Manteuffel shifts until no more    negative diagonals are found. */
  PetscCall(ISGetIndices(ip, &rip));
  PetscCall(ISGetIndices(iip, &riip));

  if (info->usedt) droptol = info->dt;

  for (int ierr = NEGATIVE_DIAGONAL; ierr == NEGATIVE_DIAGONAL;) {
    PetscBool success;

    ierr = (int)spbas_incomplete_cholesky(A, rip, riip, Pattern, droptol, shiftnz, &matrix_LT, &success);
    if (!success) {
      shiftnz *= 1.5;
      if (shiftnz < 1e-5) shiftnz = 1e-5;
      PetscCall(PetscInfo(NULL, "spbas_incomplete_cholesky found a negative diagonal. Trying again with Manteuffel shift=%g\n", (double)shiftnz));
    }
  }
  PetscCall(spbas_delete(Pattern));

  PetscCall(PetscInfo(NULL, "    memory_usage for  spbas_incomplete_cholesky  %g bytes per row\n", (double)(PetscReal)(spbas_memory_requirement(matrix_LT) / (PetscReal)mbs)));

  PetscCall(ISRestoreIndices(ip, &rip));
  PetscCall(ISRestoreIndices(iip, &riip));

  /* Convert spbas_matrix to compressed row storage */
  PetscCall(spbas_transpose(matrix_LT, &matrix_L));
  PetscCall(spbas_delete(matrix_LT));
  PetscCall(spbas_matrix_to_crs(matrix_L, &ba, &bi, &bj));
  b->i = bi;
  b->j = bj;
  b->a = ba;
  PetscCall(spbas_delete(matrix_L));

  /* Set the appropriate solution functions */
  PetscCall(ISIdentity(ip, &perm_identity));
  if (perm_identity) {
    (B)->ops->solve          = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    (B)->ops->solvetranspose = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    (B)->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    (B)->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  } else {
    (B)->ops->solve          = MatSolve_SeqSBAIJ_1_inplace;
    (B)->ops->solvetranspose = MatSolve_SeqSBAIJ_1_inplace;
    (B)->ops->forwardsolve   = MatForwardSolve_SeqSBAIJ_1_inplace;
    (B)->ops->backwardsolve  = MatBackwardSolve_SeqSBAIJ_1_inplace;
  }

  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;

  PetscCall(PetscLogFlops(C->rmap->n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatFactorGetSolverType_seqaij_bas(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERBAS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_bas(Mat A, MatFactorType ftype, Mat *B)
{
  PetscInt n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  if (ftype == MAT_FACTOR_ICC) {
    PetscCall(MatSetType(*B, MATSEQSBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(*B, 1, MAT_SKIP_ALLOCATION, NULL));

    (*B)->ops->iccfactorsymbolic     = MatICCFactorSymbolic_SeqAIJ_Bas;
    (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_Bas;
    PetscCall(PetscObjectComposeFunction((PetscObject)*B, "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_bas));
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_LU]));
    PetscCall(PetscStrallocpy(MATORDERINGND, (char **)&(*B)->preferredordering[MAT_FACTOR_CHOLESKY]));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");
  (*B)->factortype = ftype;

  PetscCall(PetscFree((*B)->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERBAS, &(*B)->solvertype));
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(PetscStrallocpy(MATORDERINGNATURAL, (char **)&(*B)->preferredordering[MAT_FACTOR_ICC]));
  PetscFunctionReturn(PETSC_SUCCESS);
}
