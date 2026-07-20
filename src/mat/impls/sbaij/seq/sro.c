#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>

/*@
  MatReorderingSeqSBAIJ - Prepare an updated index structure for a symmetric reordering of a `MATSEQSBAIJ` matrix.

  Not Collective

  Input Parameters:
+ A    - the `MATSEQSBAIJ` matrix
- perm - the (assumed symmetric) permutation to be applied

  Level: developer

  Notes:
  This routine currently raises `PETSC_ERR_SUP`; matrix reordering is not supported for `MATSEQSBAIJ` matrices,
  so callers should convert to `MATSEQAIJ` first.

  The intent (unimplemented) is to compute a new index set `(inew, jnew)` for `A` and a value map so that
  all nonzero entries `A(perm(i), perm(k))` are stored in the upper triangle; the matrix itself is not permuted.

.seealso: `Mat`, `MATSEQSBAIJ`, `MatGetOrdering()`, `MatPermute()`
@*/
PetscErrorCode MatReorderingSeqSBAIJ(Mat A, IS perm)
{
  Mat_SeqSBAIJ  *a   = (Mat_SeqSBAIJ *)A->data;
  const PetscInt mbs = a->mbs;

  PetscFunctionBegin;
  if (!mbs) PetscFunctionReturn(PETSC_SUCCESS);
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Matrix reordering is not supported for sbaij matrix. Use aij format");
#if 0
  const PetscInt *rip,*riip;
  PetscInt       *ai,*aj,*r;
  PetscInt       *nzr,nz,jmin,jmax,j,k,ajk,i;
  IS             iperm;  /* inverse of perm */
  PetscCall(ISGetIndices(perm,&rip));

  PetscCall(ISInvertPermutation(perm,PETSC_DECIDE,&iperm));
  PetscCall(ISGetIndices(iperm,&riip));

  for (i=0; i<mbs; i++) PetscCheck(rip[i] == riip[i],PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Non-symmetric permutation, use symmetric permutation for symmetric matrices");
  PetscCall(ISRestoreIndices(iperm,&riip));
  PetscCall(ISDestroy(&iperm));

  if (!a->inew) PetscCall(PetscMalloc2(mbs+1,&ai, 2*a->i[mbs],&aj));
  else {
    ai = a->inew;
    aj = a->jnew;
  }
  PetscCall(PetscArraycpy(ai,a->i,mbs+1));
  PetscCall(PetscArraycpy(aj,a->j,a->i[mbs]));

  /*
     Phase 1: Find row index r in which to store each nonzero.
              Initialize count of nonzeros to be stored in each row (nzr).
              At the end of this phase, a nonzero a(*,*)=a(r(),aj())
              s.t. a(perm(r),perm(aj)) will fall into upper triangle part.
  */

  PetscCall(PetscMalloc1(mbs,&nzr));
  PetscCall(PetscMalloc1(ai[mbs],&r));
  for (i=0; i<mbs; i++) nzr[i] = 0;
  for (i=0; i<ai[mbs]; i++) r[i] = 0;

  /*  for each nonzero element */
  for (i=0; i<mbs; i++) {
    nz = ai[i+1] - ai[i];
    j  = ai[i];
    /* printf("nz = %d, j=%d\n",nz,j); */
    while (nz--) {
      /*  --- find row (=r[j]) and column (=aj[j]) in which to store a[j] ...*/
      k = aj[j];                          /* col. index */
      /* printf("nz = %d, k=%d\n", nz,k); */
      /* for entry that will be permuted into lower triangle, swap row and col. index */
      if (rip[k] < rip[i]) aj[j] = i;
      else k = i;

      r[j] = k; j++;
      nzr[k]++;  /* increment count of nonzeros in that row */
    }
  }

  /* Phase 2: Find new ai and permutation to apply to (aj,a).
              Determine pointers (r) to delimit rows in permuted (aj,a).
              Note: r is different from r used in phase 1.
              At the end of this phase, (aj[j],a[j]) will be stored in
              (aj[r(j)],a[r(j)]).
  */
  for (i=0; i<mbs; i++) {
    ai[i+1] = ai[i] + nzr[i];
    nzr[i]  = ai[i+1];
  }

  /* determine where each (aj[j], a[j]) is stored in new (aj,a)
     for each nonzero element (in reverse order) */
  jmin = ai[0]; jmax = ai[mbs];
  nz   = jmax - jmin;
  j    = jmax-1;
  while (nz--) {
    i = r[j];  /* row value */
    if (aj[j] == i) r[j] = ai[i]; /* put diagonal nonzero at beginning of row */
    else { /* put off-diagonal nonzero in last unused location in row */
      nzr[i]--; r[j] = nzr[i];
    }
    j--;
  }

  a->a2anew = aj + ai[mbs];
  PetscCall(PetscArraycpy(a->a2anew,r,ai[mbs]));

  /* Phase 3: permute (aj,a) to upper triangular form (wrt new ordering) */
  for (j=jmin; j<jmax; j++) {
    while (r[j] != j) {
      k   = r[j]; r[j] = r[k]; r[k] = k;
      ajk = aj[k]; aj[k] = aj[j]; aj[j] = ajk;
      /* ak = aa[k]; aa[k] = aa[j]; aa[j] = ak; */
    }
  }
  PetscCall(ISRestoreIndices(perm,&rip));

  a->inew = ai;
  a->jnew = aj;

  PetscCall(ISDestroy(&a->row));
  PetscCall(ISDestroy(&a->icol));
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(ISDestroy(&a->row));
  a->row  = perm;
  PetscCall(PetscObjectReference((PetscObject)perm));
  PetscCall(ISDestroy(&a->icol));
  a->icol = perm;

  PetscCall(PetscFree(nzr));
  PetscCall(PetscFree(r));
  PetscFunctionReturn(PETSC_SUCCESS);
#endif
}
