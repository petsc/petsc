
/*
  Defines projective product routines where A is a SeqAIJ matrix
          C = P^T * A * P
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <petscbt.h>
#include <petsctime.h>

#if defined(PETSC_HAVE_HYPRE)
PETSC_INTERN PetscErrorCode MatPtAPSymbolic_AIJ_AIJ_wHYPRE(Mat, Mat, PetscReal, Mat);
#endif

PetscErrorCode MatProductSymbolic_PtAP_SeqAIJ_SeqAIJ(Mat C)
{
  Mat_Product        *product = C->product;
  Mat                 A = product->A, P = product->B;
  MatProductAlgorithm alg  = product->alg;
  PetscReal           fill = product->fill;
  PetscBool           flg;
  Mat                 Pt;

  PetscFunctionBegin;
  /* "scalable" */
  PetscCall(PetscStrcmp(alg, "scalable", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(A, P, fill, C));
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }

  /* "rap" */
  PetscCall(PetscStrcmp(alg, "rap", &flg));
  if (flg) {
    Mat_MatTransMatMult *atb;

    PetscCall(PetscNew(&atb));
    PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &Pt));
    PetscCall(MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Pt, A, P, fill, C));

    atb->At                = Pt;
    atb->data              = C->product->data;
    atb->destroy           = C->product->destroy;
    C->product->data       = atb;
    C->product->destroy    = MatDestroy_SeqAIJ_MatTransMatMult;
    C->ops->ptapnumeric    = MatPtAPNumeric_SeqAIJ_SeqAIJ;
    C->ops->productnumeric = MatProductNumeric_PtAP;
    PetscFunctionReturn(0);
  }

  /* hypre */
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscStrcmp(alg, "hypre", &flg));
  if (flg) {
    PetscCall(MatPtAPSymbolic_AIJ_AIJ_wHYPRE(A, P, fill, C));
    PetscFunctionReturn(0);
  }
#endif

  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatProductType is not supported");
}

PetscErrorCode MatPtAPSymbolic_SeqAIJ_SeqAIJ_SparseAxpy(Mat A, Mat P, PetscReal fill, Mat C)
{
  PetscFreeSpaceList free_space = NULL, current_space = NULL;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ *)A->data, *p = (Mat_SeqAIJ *)P->data, *c;
  PetscInt          *pti, *ptj, *ptJ, *ai = a->i, *aj = a->j, *ajj, *pi = p->i, *pj = p->j, *pjj;
  PetscInt          *ci, *cj, *ptadenserow, *ptasparserow, *ptaj, nspacedouble = 0;
  PetscInt           an = A->cmap->N, am = A->rmap->N, pn = P->cmap->N, pm = P->rmap->N;
  PetscInt           i, j, k, ptnzi, arow, anzj, ptanzi, prow, pnzj, cnzi, nlnk, *lnk;
  MatScalar         *ca;
  PetscBT            lnkbt;
  PetscReal          afill;

  PetscFunctionBegin;
  /* Get ij structure of P^T */
  PetscCall(MatGetSymbolicTranspose_SeqAIJ(P, &pti, &ptj));
  ptJ = ptj;

  /* Allocate ci array, arrays for fill computation and */
  /* free space for accumulating nonzero column info */
  PetscCall(PetscMalloc1(pn + 1, &ci));
  ci[0] = 0;

  PetscCall(PetscCalloc1(2 * an + 1, &ptadenserow));
  ptasparserow = ptadenserow + an;

  /* create and initialize a linked list */
  nlnk = pn + 1;
  PetscCall(PetscLLCreate(pn, pn, nlnk, lnk, lnkbt));

  /* Set initial free space to be fill*(nnz(A)+ nnz(P)) */
  PetscCall(PetscFreeSpaceGet(PetscRealIntMultTruncate(fill, PetscIntSumTruncate(ai[am], pi[pm])), &free_space));
  current_space = free_space;

  /* Determine symbolic info for each row of C: */
  for (i = 0; i < pn; i++) {
    ptnzi  = pti[i + 1] - pti[i];
    ptanzi = 0;
    /* Determine symbolic row of PtA: */
    for (j = 0; j < ptnzi; j++) {
      arow = *ptJ++;
      anzj = ai[arow + 1] - ai[arow];
      ajj  = aj + ai[arow];
      for (k = 0; k < anzj; k++) {
        if (!ptadenserow[ajj[k]]) {
          ptadenserow[ajj[k]]    = -1;
          ptasparserow[ptanzi++] = ajj[k];
        }
      }
    }
    /* Using symbolic info for row of PtA, determine symbolic info for row of C: */
    ptaj = ptasparserow;
    cnzi = 0;
    for (j = 0; j < ptanzi; j++) {
      prow = *ptaj++;
      pnzj = pi[prow + 1] - pi[prow];
      pjj  = pj + pi[prow];
      /* add non-zero cols of P into the sorted linked list lnk */
      PetscCall(PetscLLAddSorted(pnzj, pjj, pn, &nlnk, lnk, lnkbt));
      cnzi += nlnk;
    }

    /* If free space is not available, make more free space */
    /* Double the amount of total space in the list */
    if (current_space->local_remaining < cnzi) {
      PetscCall(PetscFreeSpaceGet(PetscIntSumTruncate(cnzi, current_space->total_array_size), &current_space));
      nspacedouble++;
    }

    /* Copy data into free space, and zero out denserows */
    PetscCall(PetscLLClean(pn, pn, cnzi, lnk, current_space->array, lnkbt));

    current_space->array += cnzi;
    current_space->local_used += cnzi;
    current_space->local_remaining -= cnzi;

    for (j = 0; j < ptanzi; j++) ptadenserow[ptasparserow[j]] = 0;

    /* Aside: Perhaps we should save the pta info for the numerical factorization. */
    /*        For now, we will recompute what is needed. */
    ci[i + 1] = ci[i] + cnzi;
  }
  /* nnz is now stored in ci[ptm], column indices are in the list of free space */
  /* Allocate space for cj, initialize cj, and */
  /* destroy list of free space and other temporary array(s) */
  PetscCall(PetscMalloc1(ci[pn] + 1, &cj));
  PetscCall(PetscFreeSpaceContiguous(&free_space, cj));
  PetscCall(PetscFree(ptadenserow));
  PetscCall(PetscLLDestroy(lnk, lnkbt));

  PetscCall(PetscCalloc1(ci[pn] + 1, &ca));

  /* put together the new matrix */
  PetscCall(MatSetSeqAIJWithArrays_private(PetscObjectComm((PetscObject)A), pn, pn, ci, cj, ca, ((PetscObject)A)->type_name, C));
  PetscCall(MatSetBlockSizes(C, PetscAbs(P->cmap->bs), PetscAbs(P->cmap->bs)));

  /* MatCreateSeqAIJWithArrays flags matrix so PETSc doesn't free the user's arrays. */
  /* Since these are PETSc arrays, change flags to free them as necessary. */
  c          = (Mat_SeqAIJ *)((C)->data);
  c->free_a  = PETSC_TRUE;
  c->free_ij = PETSC_TRUE;
  c->nonew   = 0;

  C->ops->ptapnumeric = MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy;

  /* set MatInfo */
  afill = (PetscReal)ci[pn] / (ai[am] + pi[pm] + 1.e-5);
  if (afill < 1.0) afill = 1.0;
  C->info.mallocs           = nspacedouble;
  C->info.fill_ratio_given  = fill;
  C->info.fill_ratio_needed = afill;

  /* Clean up. */
  PetscCall(MatRestoreSymbolicTranspose_SeqAIJ(P, &pti, &ptj));
#if defined(PETSC_USE_INFO)
  if (ci[pn] != 0) {
    PetscCall(PetscInfo(C, "Reallocs %" PetscInt_FMT "; Fill ratio: given %g needed %g.\n", nspacedouble, (double)fill, (double)afill));
    PetscCall(PetscInfo(C, "Use MatPtAP(A,P,MatReuse,%g,&C) for best performance.\n", (double)afill));
  } else {
    PetscCall(PetscInfo(C, "Empty matrix product\n"));
  }
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ_SparseAxpy(Mat A, Mat P, Mat C)
{
  Mat_SeqAIJ *a  = (Mat_SeqAIJ *)A->data;
  Mat_SeqAIJ *p  = (Mat_SeqAIJ *)P->data;
  Mat_SeqAIJ *c  = (Mat_SeqAIJ *)C->data;
  PetscInt   *ai = a->i, *aj = a->j, *apj, *apjdense, *pi = p->i, *pj = p->j, *pJ = p->j, *pjj;
  PetscInt   *ci = c->i, *cj = c->j, *cjj;
  PetscInt    am = A->rmap->N, cn = C->cmap->N, cm = C->rmap->N;
  PetscInt    i, j, k, anzi, pnzi, apnzj, nextap, pnzj, prow, crow;
  MatScalar  *aa, *apa, *pa, *pA, *paj, *ca, *caj;

  PetscFunctionBegin;
  /* Allocate temporary array for storage of one row of A*P (cn: non-scalable) */
  PetscCall(PetscCalloc2(cn, &apa, cn, &apjdense));
  PetscCall(PetscMalloc1(cn, &apj));
  /* trigger CPU copies if needed and flag CPU mask for C */
#if defined(PETSC_HAVE_DEVICE)
  {
    const PetscScalar *dummy;
    PetscCall(MatSeqAIJGetArrayRead(A, &dummy));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &dummy));
    PetscCall(MatSeqAIJGetArrayRead(P, &dummy));
    PetscCall(MatSeqAIJRestoreArrayRead(P, &dummy));
    if (C->offloadmask != PETSC_OFFLOAD_UNALLOCATED) C->offloadmask = PETSC_OFFLOAD_CPU;
  }
#endif
  aa = a->a;
  pa = p->a;
  pA = p->a;
  ca = c->a;

  /* Clear old values in C */
  PetscCall(PetscArrayzero(ca, ci[cm]));

  for (i = 0; i < am; i++) {
    /* Form sparse row of A*P */
    anzi  = ai[i + 1] - ai[i];
    apnzj = 0;
    for (j = 0; j < anzi; j++) {
      prow = *aj++;
      pnzj = pi[prow + 1] - pi[prow];
      pjj  = pj + pi[prow];
      paj  = pa + pi[prow];
      for (k = 0; k < pnzj; k++) {
        if (!apjdense[pjj[k]]) {
          apjdense[pjj[k]] = -1;
          apj[apnzj++]     = pjj[k];
        }
        apa[pjj[k]] += (*aa) * paj[k];
      }
      PetscCall(PetscLogFlops(2.0 * pnzj));
      aa++;
    }

    /* Sort the j index array for quick sparse axpy. */
    /* Note: a array does not need sorting as it is in dense storage locations. */
    PetscCall(PetscSortInt(apnzj, apj));

    /* Compute P^T*A*P using outer product (P^T)[:,j]*(A*P)[j,:]. */
    pnzi = pi[i + 1] - pi[i];
    for (j = 0; j < pnzi; j++) {
      nextap = 0;
      crow   = *pJ++;
      cjj    = cj + ci[crow];
      caj    = ca + ci[crow];
      /* Perform sparse axpy operation.  Note cjj includes apj. */
      for (k = 0; nextap < apnzj; k++) {
        PetscAssert(k < ci[crow + 1] - ci[crow], PETSC_COMM_SELF, PETSC_ERR_PLIB, "k too large k %" PetscInt_FMT ", crow %" PetscInt_FMT, k, crow);
        if (cjj[k] == apj[nextap]) caj[k] += (*pA) * apa[apj[nextap++]];
      }
      PetscCall(PetscLogFlops(2.0 * apnzj));
      pA++;
    }

    /* Zero the current row info for A*P */
    for (j = 0; j < apnzj; j++) {
      apa[apj[j]]      = 0.;
      apjdense[apj[j]] = 0;
    }
  }

  /* Assemble the final matrix and clean up */
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscFree2(apa, apjdense));
  PetscCall(PetscFree(apj));
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqAIJ_SeqAIJ(Mat A, Mat P, Mat C)
{
  Mat_MatTransMatMult *atb;

  PetscFunctionBegin;
  MatCheckProduct(C, 3);
  atb = (Mat_MatTransMatMult *)C->product->data;
  PetscCheck(atb, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Missing data structure");
  PetscCall(MatTranspose(P, MAT_REUSE_MATRIX, &atb->At));
  PetscCheck(C->ops->matmultnumeric, PetscObjectComm((PetscObject)C), PETSC_ERR_PLIB, "Missing numeric operation");
  /* when using rap, MatMatMatMultSymbolic used a different data */
  if (atb->data) C->product->data = atb->data;
  PetscCall((*C->ops->matmatmultnumeric)(atb->At, A, P, C));
  C->product->data = atb;
  PetscFunctionReturn(0);
}
