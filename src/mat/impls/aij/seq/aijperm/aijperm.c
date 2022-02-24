
/*
  Defines basic operations for the MATSEQAIJPERM matrix class.
  This class is derived from the MATSEQAIJ class and retains the
  compressed row storage (aka Yale sparse matrix format) but augments
  it with some permutation information that enables some operations
  to be more vectorizable.  A physically rearranged copy of the matrix
  may be stored if the user desires.

  Eventually a variety of permutations may be supported.
*/

#include <../src/mat/impls/aij/seq/aij.h>

#if defined(PETSC_USE_AVX512_KERNELS) && defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
#include <immintrin.h>

#if !defined(_MM_SCALE_8)
#define _MM_SCALE_8    8
#endif
#if !defined(_MM_SCALE_4)
#define _MM_SCALE_4    4
#endif
#endif

#define NDIM 512
/* NDIM specifies how many rows at a time we should work with when
 * performing the vectorized mat-vec.  This depends on various factors
 * such as vector register length, etc., and I really need to add a
 * way for the user (or the library) to tune this.  I'm setting it to
 * 512 for now since that is what Ed D'Azevedo was using in his Fortran
 * routines. */

typedef struct {
  PetscObjectState nonzerostate; /* used to determine if the nonzero structure has changed and hence the permutations need updating */

  PetscInt         ngroup;
  PetscInt         *xgroup;
  /* Denotes where groups of rows with same number of nonzeros
   * begin and end, i.e., xgroup[i] gives us the position in iperm[]
   * where the ith group begins. */

  PetscInt         *nzgroup; /*  how many nonzeros each row that is a member of group i has. */
  PetscInt         *iperm;  /* The permutation vector. */

  /* Some of this stuff is for Ed's recursive triangular solve.
   * I'm not sure what I need yet. */
  PetscInt         blocksize;
  PetscInt         nstep;
  PetscInt         *jstart_list;
  PetscInt         *jend_list;
  PetscInt         *action_list;
  PetscInt         *ngroup_list;
  PetscInt         **ipointer_list;
  PetscInt         **xgroup_list;
  PetscInt         **nzgroup_list;
  PetscInt         **iperm_list;
} Mat_SeqAIJPERM;

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJPERM_SeqAIJ(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  /* This routine is only called to convert a MATAIJPERM to its base PETSc type, */
  /* so we will ignore 'MatType type'. */
  Mat            B       = *newmat;
  Mat_SeqAIJPERM *aijperm=(Mat_SeqAIJPERM*)A->spptr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
    aijperm=(Mat_SeqAIJPERM*)B->spptr;
  }

  /* Reset the original function pointers. */
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJ;
  B->ops->destroy     = MatDestroy_SeqAIJ;
  B->ops->duplicate   = MatDuplicate_SeqAIJ;
  B->ops->mult        = MatMult_SeqAIJ;
  B->ops->multadd     = MatMultAdd_SeqAIJ;

  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijperm_seqaij_C",NULL));

  /* Free everything in the Mat_SeqAIJPERM data structure.*/
  CHKERRQ(PetscFree(aijperm->xgroup));
  CHKERRQ(PetscFree(aijperm->nzgroup));
  CHKERRQ(PetscFree(aijperm->iperm));
  CHKERRQ(PetscFree(B->spptr));

  /* Change the type of B to MATSEQAIJ. */
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B, MATSEQAIJ));

  *newmat = B;
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqAIJPERM(Mat A)
{
  Mat_SeqAIJPERM *aijperm = (Mat_SeqAIJPERM*) A->spptr;

  PetscFunctionBegin;
  if (aijperm) {
    /* If MatHeaderMerge() was used then this SeqAIJPERM matrix will not have a spprt. */
    CHKERRQ(PetscFree(aijperm->xgroup));
    CHKERRQ(PetscFree(aijperm->nzgroup));
    CHKERRQ(PetscFree(aijperm->iperm));
    CHKERRQ(PetscFree(A->spptr));
  }
  /* Change the type of A back to SEQAIJ and use MatDestroy_SeqAIJ()
   * to destroy everything that remains. */
  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A, MATSEQAIJ));
  /* Note that I don't call MatSetType().  I believe this is because that
   * is only to be called when *building* a matrix.  I could be wrong, but
   * that is how things work for the SuperLU matrix class. */
  CHKERRQ(MatDestroy_SeqAIJ(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqAIJPERM(Mat A, MatDuplicateOption op, Mat *M)
{
  Mat_SeqAIJPERM *aijperm      = (Mat_SeqAIJPERM*) A->spptr;
  Mat_SeqAIJPERM *aijperm_dest;
  PetscBool      perm;

  PetscFunctionBegin;
  CHKERRQ(MatDuplicate_SeqAIJ(A,op,M));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)*M,MATSEQAIJPERM,&perm));
  if (perm) {
    aijperm_dest = (Mat_SeqAIJPERM *) (*M)->spptr;
    CHKERRQ(PetscFree(aijperm_dest->xgroup));
    CHKERRQ(PetscFree(aijperm_dest->nzgroup));
    CHKERRQ(PetscFree(aijperm_dest->iperm));
  } else {
    CHKERRQ(PetscNewLog(*M,&aijperm_dest));
    (*M)->spptr = (void*) aijperm_dest;
    CHKERRQ(PetscObjectChangeTypeName((PetscObject)*M,MATSEQAIJPERM));
    CHKERRQ(PetscObjectComposeFunction((PetscObject)*M,"MatConvert_seqaijperm_seqaij_C",MatConvert_SeqAIJPERM_SeqAIJ));
  }
  CHKERRQ(PetscArraycpy(aijperm_dest,aijperm,1));
  /* Allocate space for, and copy the grouping and permutation info.
   * I note that when the groups are initially determined in
   * MatSeqAIJPERM_create_perm, xgroup and nzgroup may be sized larger than
   * necessary.  But at this point, we know how large they need to be, and
   * allocate only the necessary amount of memory.  So the duplicated matrix
   * may actually use slightly less storage than the original! */
  CHKERRQ(PetscMalloc1(A->rmap->n, &aijperm_dest->iperm));
  CHKERRQ(PetscMalloc1(aijperm->ngroup+1, &aijperm_dest->xgroup));
  CHKERRQ(PetscMalloc1(aijperm->ngroup, &aijperm_dest->nzgroup));
  CHKERRQ(PetscArraycpy(aijperm_dest->iperm,aijperm->iperm,A->rmap->n));
  CHKERRQ(PetscArraycpy(aijperm_dest->xgroup,aijperm->xgroup,aijperm->ngroup+1));
  CHKERRQ(PetscArraycpy(aijperm_dest->nzgroup,aijperm->nzgroup,aijperm->ngroup));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqAIJPERM_create_perm(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJPERM *aijperm = (Mat_SeqAIJPERM*) A->spptr;
  PetscInt       m;       /* Number of rows in the matrix. */
  PetscInt       *ia;       /* From the CSR representation; points to the beginning  of each row. */
  PetscInt       maxnz;      /* Maximum number of nonzeros in any row. */
  PetscInt       *rows_in_bucket;
  /* To construct the permutation, we sort each row into one of maxnz
   * buckets based on how many nonzeros are in the row. */
  PetscInt       nz;
  PetscInt       *nz_in_row;         /* the number of nonzero elements in row k. */
  PetscInt       *ipnz;
  /* When constructing the iperm permutation vector,
   * ipnz[nz] is used to point to the next place in the permutation vector
   * that a row with nz nonzero elements should be placed.*/
  PetscInt       i, ngroup, istart, ipos;

  PetscFunctionBegin;
  if (aijperm->nonzerostate == A->nonzerostate) PetscFunctionReturn(0); /* permutation exists and matches current nonzero structure */
  aijperm->nonzerostate = A->nonzerostate;
 /* Free anything previously put in the Mat_SeqAIJPERM data structure. */
  CHKERRQ(PetscFree(aijperm->xgroup));
  CHKERRQ(PetscFree(aijperm->nzgroup));
  CHKERRQ(PetscFree(aijperm->iperm));

  m  = A->rmap->n;
  ia = a->i;

  /* Allocate the arrays that will hold the permutation vector. */
  CHKERRQ(PetscMalloc1(m, &aijperm->iperm));

  /* Allocate some temporary work arrays that will be used in
   * calculating the permuation vector and groupings. */
  CHKERRQ(PetscMalloc1(m, &nz_in_row));

  /* Now actually figure out the permutation and grouping. */

  /* First pass: Determine number of nonzeros in each row, maximum
   * number of nonzeros in any row, and how many rows fall into each
   * "bucket" of rows with same number of nonzeros. */
  maxnz = 0;
  for (i=0; i<m; i++) {
    nz_in_row[i] = ia[i+1]-ia[i];
    if (nz_in_row[i] > maxnz) maxnz = nz_in_row[i];
  }
  CHKERRQ(PetscMalloc1(PetscMax(maxnz,m)+1, &rows_in_bucket));
  CHKERRQ(PetscMalloc1(PetscMax(maxnz,m)+1, &ipnz));

  for (i=0; i<=maxnz; i++) {
    rows_in_bucket[i] = 0;
  }
  for (i=0; i<m; i++) {
    nz = nz_in_row[i];
    rows_in_bucket[nz]++;
  }

  /* Allocate space for the grouping info.  There will be at most (maxnz + 1)
   * groups.  (It is maxnz + 1 instead of simply maxnz because there may be
   * rows with no nonzero elements.)  If there are (maxnz + 1) groups,
   * then xgroup[] must consist of (maxnz + 2) elements, since the last
   * element of xgroup will tell us where the (maxnz + 1)th group ends.
   * We allocate space for the maximum number of groups;
   * that is potentially a little wasteful, but not too much so.
   * Perhaps I should fix it later. */
  CHKERRQ(PetscMalloc1(maxnz+2, &aijperm->xgroup));
  CHKERRQ(PetscMalloc1(maxnz+1, &aijperm->nzgroup));

  /* Second pass.  Look at what is in the buckets and create the groupings.
   * Note that it is OK to have a group of rows with no non-zero values. */
  ngroup = 0;
  istart = 0;
  for (i=0; i<=maxnz; i++) {
    if (rows_in_bucket[i] > 0) {
      aijperm->nzgroup[ngroup] = i;
      aijperm->xgroup[ngroup]  = istart;
      ngroup++;
      istart += rows_in_bucket[i];
    }
  }

  aijperm->xgroup[ngroup] = istart;
  aijperm->ngroup         = ngroup;

  /* Now fill in the permutation vector iperm. */
  ipnz[0] = 0;
  for (i=0; i<maxnz; i++) {
    ipnz[i+1] = ipnz[i] + rows_in_bucket[i];
  }

  for (i=0; i<m; i++) {
    nz                   = nz_in_row[i];
    ipos                 = ipnz[nz];
    aijperm->iperm[ipos] = i;
    ipnz[nz]++;
  }

  /* Clean up temporary work arrays. */
  CHKERRQ(PetscFree(rows_in_bucket));
  CHKERRQ(PetscFree(ipnz));
  CHKERRQ(PetscFree(nz_in_row));
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqAIJPERM(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Since a MATSEQAIJPERM matrix is really just a MATSEQAIJ with some
   * extra information, call the AssemblyEnd routine for a MATSEQAIJ.
   * I'm not sure if this is the best way to do this, but it avoids
   * a lot of code duplication.
   * I also note that currently MATSEQAIJPERM doesn't know anything about
   * the Mat_CompressedRow data structure that SeqAIJ now uses when there
   * are many zero rows.  If the SeqAIJ assembly end routine decides to use
   * this, this may break things.  (Don't know... haven't looked at it.) */
  a->inode.use = PETSC_FALSE;
  CHKERRQ(MatAssemblyEnd_SeqAIJ(A, mode));

  /* Now calculate the permutation and grouping information. */
  CHKERRQ(MatSeqAIJPERM_create_perm(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqAIJPERM(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y;
  const MatScalar   *aa;
  const PetscInt    *aj,*ai;
#if !(defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJPERM) && defined(notworking))
  PetscInt          i,j;
#endif
#if defined(PETSC_USE_AVX512_KERNELS) && defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m512d           vec_x,vec_y,vec_vals;
  __m256i           vec_idx,vec_ipos,vec_j;
  __mmask8           mask;
#endif

  /* Variables that don't appear in MatMult_SeqAIJ. */
  Mat_SeqAIJPERM    *aijperm = (Mat_SeqAIJPERM*) A->spptr;
  PetscInt          *iperm;  /* Points to the permutation vector. */
  PetscInt          *xgroup;
  /* Denotes where groups of rows with same number of nonzeros
   * begin and end in iperm. */
  PetscInt          *nzgroup;
  PetscInt          ngroup;
  PetscInt          igroup;
  PetscInt          jstart,jend;
  /* jstart is used in loops to denote the position in iperm where a
   * group starts; jend denotes the position where it ends.
   * (jend + 1 is where the next group starts.) */
  PetscInt          iold,nz;
  PetscInt          istart,iend,isize;
  PetscInt          ipos;
  PetscScalar       yp[NDIM];
  PetscInt          ip[NDIM];    /* yp[] and ip[] are treated as vector "registers" for performing the mat-vec. */

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(xx,&x));
  CHKERRQ(VecGetArray(yy,&y));
  aj   = a->j;   /* aj[k] gives column index for element aa[k]. */
  aa   = a->a; /* Nonzero elements stored row-by-row. */
  ai   = a->i;  /* ai[k] is the position in aa and aj where row k starts. */

  /* Get the info we need about the permutations and groupings. */
  iperm   = aijperm->iperm;
  ngroup  = aijperm->ngroup;
  xgroup  = aijperm->xgroup;
  nzgroup = aijperm->nzgroup;

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJPERM) && defined(notworking)
  fortranmultaijperm_(&m,x,ii,aj,aa,y);
#else

  for (igroup=0; igroup<ngroup; igroup++) {
    jstart = xgroup[igroup];
    jend   = xgroup[igroup+1] - 1;
    nz     = nzgroup[igroup];

    /* Handle the special cases where the number of nonzeros per row
     * in the group is either 0 or 1. */
    if (nz == 0) {
      for (i=jstart; i<=jend; i++) {
        y[iperm[i]] = 0.0;
      }
    } else if (nz == 1) {
      for (i=jstart; i<=jend; i++) {
        iold    = iperm[i];
        ipos    = ai[iold];
        y[iold] = aa[ipos] * x[aj[ipos]];
      }
    } else {

      /* We work our way through the current group in chunks of NDIM rows
       * at a time. */

      for (istart=jstart; istart<=jend; istart+=NDIM) {
        /* Figure out where the chunk of 'isize' rows ends in iperm.
         * 'isize may of course be less than NDIM for the last chunk. */
        iend = istart + (NDIM - 1);

        if (iend > jend) iend = jend;

        isize = iend - istart + 1;

        /* Initialize the yp[] array that will be used to hold part of
         * the permuted results vector, and figure out where in aa each
         * row of the chunk will begin. */
        for (i=0; i<isize; i++) {
          iold = iperm[istart + i];
          /* iold is a row number from the matrix A *before* reordering. */
          ip[i] = ai[iold];
          /* ip[i] tells us where the ith row of the chunk begins in aa. */
          yp[i] = (PetscScalar) 0.0;
        }

        /* If the number of zeros per row exceeds the number of rows in
         * the chunk, we should vectorize along nz, that is, perform the
         * mat-vec one row at a time as in the usual CSR case. */
        if (nz > isize) {
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI preferstream
#endif
          for (i=0; i<isize; i++) {
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI prefervector
#endif

#if defined(PETSC_USE_AVX512_KERNELS) && defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
            vec_y = _mm512_setzero_pd();
            ipos = ip[i];
            for (j=0; j<(nz>>3); j++) {
              vec_idx  = _mm256_loadu_si256((__m256i const*)&aj[ipos]);
              vec_vals = _mm512_loadu_pd(&aa[ipos]);
              vec_x    = _mm512_i32gather_pd(vec_idx,x,_MM_SCALE_8);
              vec_y    = _mm512_fmadd_pd(vec_x,vec_vals,vec_y);
              ipos += 8;
            }
            if ((nz&0x07)>2) {
              mask     = (__mmask8)(0xff >> (8-(nz&0x07)));
              vec_idx  = _mm256_loadu_si256((__m256i const*)&aj[ipos]);
              vec_vals = _mm512_loadu_pd(&aa[ipos]);
              vec_x    = _mm512_mask_i32gather_pd(vec_x,mask,vec_idx,x,_MM_SCALE_8);
              vec_y    = _mm512_mask3_fmadd_pd(vec_x,vec_vals,vec_y,mask);
            } else if ((nz&0x07)==2) {
              yp[i] += aa[ipos]*x[aj[ipos]];
              yp[i] += aa[ipos+1]*x[aj[ipos+1]];
            } else if ((nz&0x07)==1) {
              yp[i] += aa[ipos]*x[aj[ipos]];
            }
            yp[i] += _mm512_reduce_add_pd(vec_y);
#else
            for (j=0; j<nz; j++) {
              ipos   = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
#endif
          }
        } else {
          /* Otherwise, there are enough rows in the chunk to make it
           * worthwhile to vectorize across the rows, that is, to do the
           * matvec by operating with "columns" of the chunk. */
          for (j=0; j<nz; j++) {
#if defined(PETSC_USE_AVX512_KERNELS) && defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
            vec_j = _mm256_set1_epi32(j);
            for (i=0; i<((isize>>3)<<3); i+=8) {
              vec_y    = _mm512_loadu_pd(&yp[i]);
              vec_ipos = _mm256_loadu_si256((__m256i const*)&ip[i]);
              vec_ipos = _mm256_add_epi32(vec_ipos,vec_j);
              vec_idx  = _mm256_i32gather_epi32(aj,vec_ipos,_MM_SCALE_4);
              vec_vals = _mm512_i32gather_pd(vec_ipos,aa,_MM_SCALE_8);
              vec_x    = _mm512_i32gather_pd(vec_idx,x,_MM_SCALE_8);
              vec_y    = _mm512_fmadd_pd(vec_x,vec_vals,vec_y);
              _mm512_storeu_pd(&yp[i],vec_y);
            }
            for (i=isize-(isize&0x07); i<isize; i++) {
              ipos = ip[i]+j;
              yp[i] += aa[ipos]*x[aj[ipos]];
            }
#else
            for (i=0; i<isize; i++) {
              ipos   = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
#endif
          }
        }

#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI ivdep
#endif
        /* Put results from yp[] into non-permuted result vector y. */
        for (i=0; i<isize; i++) {
          y[iperm[istart+i]] = yp[i];
        }
      } /* End processing chunk of isize rows of a group. */
    } /* End handling matvec for chunk with nz > 1. */
  } /* End loop over igroup. */
#endif
  CHKERRQ(PetscLogFlops(PetscMax(2.0*a->nz - A->rmap->n,0)));
  CHKERRQ(VecRestoreArrayRead(xx,&x));
  CHKERRQ(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

/* MatMultAdd_SeqAIJPERM() calculates yy = ww + A * xx.
 * Note that the names I used to designate the vectors differs from that
 * used in MatMultAdd_SeqAIJ().  I did this to keep my notation consistent
 * with the MatMult_SeqAIJPERM() routine, which is very similar to this one. */
/*
    I hate having virtually identical code for the mult and the multadd!!!
*/
PetscErrorCode MatMultAdd_SeqAIJPERM(Mat A,Vec xx,Vec ww,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  const PetscScalar *x;
  PetscScalar       *y,*w;
  const MatScalar   *aa;
  const PetscInt    *aj,*ai;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJPERM)
  PetscInt i,j;
#endif

  /* Variables that don't appear in MatMultAdd_SeqAIJ. */
  Mat_SeqAIJPERM * aijperm;
  PetscInt       *iperm;    /* Points to the permutation vector. */
  PetscInt       *xgroup;
  /* Denotes where groups of rows with same number of nonzeros
   * begin and end in iperm. */
  PetscInt *nzgroup;
  PetscInt ngroup;
  PetscInt igroup;
  PetscInt jstart,jend;
  /* jstart is used in loops to denote the position in iperm where a
   * group starts; jend denotes the position where it ends.
   * (jend + 1 is where the next group starts.) */
  PetscInt    iold,nz;
  PetscInt    istart,iend,isize;
  PetscInt    ipos;
  PetscScalar yp[NDIM];
  PetscInt    ip[NDIM];
  /* yp[] and ip[] are treated as vector "registers" for performing
   * the mat-vec. */

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(xx,&x));
  CHKERRQ(VecGetArrayPair(yy,ww,&y,&w));

  aj = a->j;   /* aj[k] gives column index for element aa[k]. */
  aa = a->a;   /* Nonzero elements stored row-by-row. */
  ai = a->i;   /* ai[k] is the position in aa and aj where row k starts. */

  /* Get the info we need about the permutations and groupings. */
  aijperm = (Mat_SeqAIJPERM*) A->spptr;
  iperm   = aijperm->iperm;
  ngroup  = aijperm->ngroup;
  xgroup  = aijperm->xgroup;
  nzgroup = aijperm->nzgroup;

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJPERM)
  fortranmultaddaijperm_(&m,x,ii,aj,aa,y,w);
#else

  for (igroup=0; igroup<ngroup; igroup++) {
    jstart = xgroup[igroup];
    jend   = xgroup[igroup+1] - 1;

    nz = nzgroup[igroup];

    /* Handle the special cases where the number of nonzeros per row
     * in the group is either 0 or 1. */
    if (nz == 0) {
      for (i=jstart; i<=jend; i++) {
        iold    = iperm[i];
        y[iold] = w[iold];
      }
    }
    else if (nz == 1) {
      for (i=jstart; i<=jend; i++) {
        iold    = iperm[i];
        ipos    = ai[iold];
        y[iold] = w[iold] + aa[ipos] * x[aj[ipos]];
      }
    }
    /* For the general case: */
    else {

      /* We work our way through the current group in chunks of NDIM rows
       * at a time. */

      for (istart=jstart; istart<=jend; istart+=NDIM) {
        /* Figure out where the chunk of 'isize' rows ends in iperm.
         * 'isize may of course be less than NDIM for the last chunk. */
        iend = istart + (NDIM - 1);
        if (iend > jend) iend = jend;
        isize = iend - istart + 1;

        /* Initialize the yp[] array that will be used to hold part of
         * the permuted results vector, and figure out where in aa each
         * row of the chunk will begin. */
        for (i=0; i<isize; i++) {
          iold = iperm[istart + i];
          /* iold is a row number from the matrix A *before* reordering. */
          ip[i] = ai[iold];
          /* ip[i] tells us where the ith row of the chunk begins in aa. */
          yp[i] = w[iold];
        }

        /* If the number of zeros per row exceeds the number of rows in
         * the chunk, we should vectorize along nz, that is, perform the
         * mat-vec one row at a time as in the usual CSR case. */
        if (nz > isize) {
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI preferstream
#endif
          for (i=0; i<isize; i++) {
#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI prefervector
#endif
            for (j=0; j<nz; j++) {
              ipos   = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }
        /* Otherwise, there are enough rows in the chunk to make it
         * worthwhile to vectorize across the rows, that is, to do the
         * matvec by operating with "columns" of the chunk. */
        else {
          for (j=0; j<nz; j++) {
            for (i=0; i<isize; i++) {
              ipos   = ip[i] + j;
              yp[i] += aa[ipos] * x[aj[ipos]];
            }
          }
        }

#if defined(PETSC_HAVE_CRAY_VECTOR)
#pragma _CRI ivdep
#endif
        /* Put results from yp[] into non-permuted result vector y. */
        for (i=0; i<isize; i++) {
          y[iperm[istart+i]] = yp[i];
        }
      } /* End processing chunk of isize rows of a group. */

    } /* End handling matvec for chunk with nz > 1. */
  } /* End loop over igroup. */

#endif
  CHKERRQ(PetscLogFlops(2.0*a->nz));
  CHKERRQ(VecRestoreArrayRead(xx,&x));
  CHKERRQ(VecRestoreArrayPair(yy,ww,&y,&w));
  PetscFunctionReturn(0);
}

/* MatConvert_SeqAIJ_SeqAIJPERM converts a SeqAIJ matrix into a
 * SeqAIJPERM matrix.  This routine is called by the MatCreate_SeqAIJPERM()
 * routine, but can also be used to convert an assembled SeqAIJ matrix
 * into a SeqAIJPERM one. */
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqAIJPERM(Mat A,MatType type,MatReuse reuse,Mat *newmat)
{
  Mat            B = *newmat;
  Mat_SeqAIJPERM *aijperm;
  PetscBool      sametype;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&B));
  }
  CHKERRQ(PetscObjectTypeCompare((PetscObject)A,type,&sametype));
  if (sametype) PetscFunctionReturn(0);

  CHKERRQ(PetscNewLog(B,&aijperm));
  B->spptr = (void*) aijperm;

  /* Set function pointers for methods that we inherit from AIJ but override. */
  B->ops->duplicate   = MatDuplicate_SeqAIJPERM;
  B->ops->assemblyend = MatAssemblyEnd_SeqAIJPERM;
  B->ops->destroy     = MatDestroy_SeqAIJPERM;
  B->ops->mult        = MatMult_SeqAIJPERM;
  B->ops->multadd     = MatMultAdd_SeqAIJPERM;

  aijperm->nonzerostate = -1;  /* this will trigger the generation of the permutation information the first time through MatAssembly()*/
  /* If A has already been assembled, compute the permutation. */
  if (A->assembled) {
    CHKERRQ(MatSeqAIJPERM_create_perm(B));
  }

  CHKERRQ(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqaijperm_seqaij_C",MatConvert_SeqAIJPERM_SeqAIJ));

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJPERM));
  *newmat = B;
  PetscFunctionReturn(0);
}

/*@C
   MatCreateSeqAIJPERM - Creates a sparse matrix of type SEQAIJPERM.
   This type inherits from AIJ, but calculates some additional permutation
   information that is used to allow better vectorization of some
   operations.  At the cost of increased storage, the AIJ formatted
   matrix can be copied to a format in which pieces of the matrix are
   stored in ELLPACK format, allowing the vectorized matrix multiply
   routine to use stride-1 memory accesses.  As with the AIJ type, it is
   important to preallocate matrix storage in order to get good assembly
   performance.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   Notes:
   If nnz is given then nz is ignored

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJPERM(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqAIJPERM(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscFunctionBegin;
  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,m,n,m,n));
  CHKERRQ(MatSetType(*A,MATSEQAIJPERM));
  CHKERRQ(MatSeqAIJSetPreallocation_SeqAIJ(*A,nz,nnz));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqAIJPERM(Mat A)
{
  PetscFunctionBegin;
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatConvert_SeqAIJ_SeqAIJPERM(A,MATSEQAIJPERM,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}
