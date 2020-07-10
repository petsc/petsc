
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/aij/seq/bas/spbas.h>

PetscErrorCode MatICCFactorSymbolic_SeqAIJ_Bas(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ   *b;
  PetscErrorCode ierr;
  PetscBool      perm_identity,missing;
  PetscInt       reallocs=0,i,*ai=a->i,*aj=a->j,am=A->rmap->n,*ui;
  const PetscInt *rip,*riip;
  PetscInt       j;
  PetscInt       d;
  PetscInt       ncols,*cols,*uj;
  PetscReal      fill=info->fill,levels=info->levels;
  IS             iperm;
  spbas_matrix   Pattern_0, Pattern_P;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatMissingDiagonal(A,&missing,&d);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",d);
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);

  /* ICC(0) without matrix ordering: simply copies fill pattern */
  if (!levels && perm_identity) {
    ierr  = PetscMalloc1(am+1,&ui);CHKERRQ(ierr);
    ui[0] = 0;

    for (i=0; i<am; i++) {
      ui[i+1] = ui[i] + ai[i+1] - a->diag[i];
    }
    ierr = PetscMalloc1(ui[am]+1,&uj);CHKERRQ(ierr);
    cols = uj;
    for (i=0; i<am; i++) {
      aj    = a->j + a->diag[i];
      ncols = ui[i+1] - ui[i];
      for (j=0; j<ncols; j++) *cols++ = *aj++;
    }
  } else { /* case: levels>0 || (levels=0 && !perm_identity) */
    ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);
    ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

    /* Create spbas_matrix for pattern */
    ierr = spbas_pattern_only(am, am, ai, aj, &Pattern_0);CHKERRQ(ierr);

    /* Apply the permutation */
    ierr = spbas_apply_reordering(&Pattern_0, rip, riip);CHKERRQ(ierr);

    /* Raise the power */
    ierr = spbas_power(Pattern_0, (int) levels+1, &Pattern_P);CHKERRQ(ierr);
    ierr = spbas_delete(Pattern_0);CHKERRQ(ierr);

    /* Keep only upper triangle of pattern */
    ierr = spbas_keep_upper(&Pattern_P);CHKERRQ(ierr);

    /* Convert to Sparse Row Storage  */
    ierr = spbas_matrix_to_crs(Pattern_P, NULL, &ui, &uj);CHKERRQ(ierr);
    ierr = spbas_delete(Pattern_P);CHKERRQ(ierr);
  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */

  b               = (Mat_SeqSBAIJ*)(fact)->data;
  b->singlemalloc = PETSC_FALSE;

  ierr = PetscMalloc1(ui[am]+1,&b->a);CHKERRQ(ierr);

  b->j    = uj;
  b->i    = ui;
  b->diag = 0;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->col  = perm;

  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);

  b->icol          = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr             = PetscMalloc1(am+1,&b->solve_work);CHKERRQ(ierr);
  ierr             = PetscLogObjectMemory((PetscObject)(fact),(ui[am]-am)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz         = b->nz = ui[am];
  b->free_a        = PETSC_TRUE;
  b->free_ij       = PETSC_TRUE;

  (fact)->info.factor_mallocs   = reallocs;
  (fact)->info.fill_ratio_given = fill;
  if (ai[am] != 0) {
    (fact)->info.fill_ratio_needed = ((PetscReal)ui[am])/((PetscReal)ai[am]);
  } else {
    (fact)->info.fill_ratio_needed = 0.0;
  }
  /*  (fact)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_inplace; */
  PetscFunctionReturn(0);
}


PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ_Bas(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C = B;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  IS             ip=b->row,iip = b->icol;
  PetscErrorCode ierr;
  const PetscInt *rip,*riip;
  PetscInt       mbs=A->rmap->n,*bi=b->i,*bj=b->j;

  MatScalar    *ba     = b->a;
  PetscReal    shiftnz = info->shiftamount;
  PetscReal    droptol = -1;
  PetscBool    perm_identity;
  spbas_matrix Pattern, matrix_L,matrix_LT;
  PetscReal    mem_reduction;

  PetscFunctionBegin;
  /* Reduce memory requirements:   erase values of B-matrix */
  ierr = PetscFree(ba);CHKERRQ(ierr);
  /*   Compress (maximum) sparseness pattern of B-matrix */
  ierr = spbas_compress_pattern(bi, bj, mbs, mbs, SPBAS_DIAGONAL_OFFSETS,&Pattern, &mem_reduction);CHKERRQ(ierr);
  ierr = PetscFree(bi);CHKERRQ(ierr);
  ierr = PetscFree(bj);CHKERRQ(ierr);

  ierr = PetscInfo1(NULL,"    compression rate for spbas_compress_pattern %g \n",(double)mem_reduction);CHKERRQ(ierr);

  /* Make Cholesky decompositions with larger Manteuffel shifts until no more    negative diagonals are found. */
  ierr = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISGetIndices(iip,&riip);CHKERRQ(ierr);

  if (info->usedt) {
    droptol = info->dt;
  }
  for (ierr = NEGATIVE_DIAGONAL; ierr == NEGATIVE_DIAGONAL;)
  {
    ierr = spbas_incomplete_cholesky(A, rip, riip, Pattern, droptol, shiftnz,&matrix_LT);CHKERRQ(ierr);
    if (ierr == NEGATIVE_DIAGONAL) {
      shiftnz *= 1.5;
      if (shiftnz < 1e-5) shiftnz=1e-5;
      ierr = PetscInfo1(NULL,"spbas_incomplete_cholesky found a negative diagonal. Trying again with Manteuffel shift=%g\n",(double)shiftnz);CHKERRQ(ierr);
    }
  }
  ierr = spbas_delete(Pattern);CHKERRQ(ierr);

  ierr = PetscInfo1(NULL,"    memory_usage for  spbas_incomplete_cholesky  %g bytes per row\n", (double)(PetscReal) (spbas_memory_requirement(matrix_LT)/ (PetscReal) mbs));CHKERRQ(ierr);

  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iip,&riip);CHKERRQ(ierr);

  /* Convert spbas_matrix to compressed row storage */
  ierr = spbas_transpose(matrix_LT, &matrix_L);CHKERRQ(ierr);
  ierr = spbas_delete(matrix_LT);CHKERRQ(ierr);
  ierr = spbas_matrix_to_crs(matrix_L, &ba, &bi, &bj);CHKERRQ(ierr);
  b->i =bi; b->j=bj; b->a=ba;
  ierr = spbas_delete(matrix_L);CHKERRQ(ierr);

  /* Set the appropriate solution functions */
  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
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

  ierr = PetscLogFlops(C->rmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatFactorGetSolverType_seqaij_bas(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERBAS;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqaij_bas(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt       n = A->rmap->n;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_ICC) {
    ierr = MatSetType(*B,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*B,1,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);

    (*B)->ops->iccfactorsymbolic     = MatICCFactorSymbolic_SeqAIJ_Bas;
    (*B)->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_Bas;
     ierr = PetscObjectComposeFunction((PetscObject)*B,"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_bas);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  (*B)->factortype = ftype;

  ierr = PetscFree((*B)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERBAS,&(*B)->solvertype);CHKERRQ(ierr);
  (*B)->useordering = PETSC_TRUE;
  PetscFunctionReturn(0);
}
