
/*
  Defines the basic matrix operations for the SELL matrix storage format.
*/
#include <../src/mat/impls/sell/seq/sell.h>  /*I   "petscmat.h"  I*/
#include <petscblaslapack.h>
#include <petsc/private/kernels/blocktranspose.h>

static PetscBool  cited = PETSC_FALSE;
static const char citation[] =
  "@inproceedings{ZhangELLPACK2018,\n"
  " author = {Hong Zhang and Richard T. Mills and Karl Rupp and Barry F. Smith},\n"
  " title = {Vectorized Parallel Sparse Matrix-Vector Multiplication in {PETSc} Using {AVX-512}},\n"
  " booktitle = {Proceedings of the 47th International Conference on Parallel Processing},\n"
  " year = 2018\n"
"}\n";

#if defined(PETSC_HAVE_IMMINTRIN_H) && (defined(__AVX512F__) || (defined(__AVX2__) && defined(__FMA__)) || defined(__AVX__)) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)

  #include <immintrin.h>

  #if !defined(_MM_SCALE_8)
  #define _MM_SCALE_8    8
  #endif

  #if defined(__AVX512F__)
  /* these do not work
   vec_idx  = _mm512_loadunpackhi_epi32(vec_idx,acolidx);
   vec_vals = _mm512_loadunpackhi_pd(vec_vals,aval);
  */
    #define AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y) \
    /* if the mask bit is set, copy from acolidx, otherwise from vec_idx */ \
    vec_idx  = _mm256_loadu_si256((__m256i const*)acolidx); \
    vec_vals = _mm512_loadu_pd(aval); \
    vec_x    = _mm512_i32gather_pd(vec_idx,x,_MM_SCALE_8); \
    vec_y    = _mm512_fmadd_pd(vec_x,vec_vals,vec_y)
  #elif defined(__AVX2__) && defined(__FMA__)
    #define AVX2_Mult_Private(vec_idx,vec_x,vec_vals,vec_y) \
    vec_vals = _mm256_loadu_pd(aval); \
    vec_idx  = _mm_loadu_si128((__m128i const*)acolidx); /* SSE2 */ \
    vec_x    = _mm256_i32gather_pd(x,vec_idx,_MM_SCALE_8); \
    vec_y    = _mm256_fmadd_pd(vec_x,vec_vals,vec_y)
  #endif
#endif  /* PETSC_HAVE_IMMINTRIN_H */

/*@C
 MatSeqSELLSetPreallocation - For good matrix assembly performance
 the user should preallocate the matrix storage by setting the parameter nz
 (or the array nnz).  By setting these parameters accurately, performance
 during matrix assembly can be increased significantly.

 Collective

 Input Parameters:
 +  B - The matrix
 .  nz - number of nonzeros per row (same for all rows)
 -  nnz - array containing the number of nonzeros in the various rows
 (possibly different for each row) or NULL

 Notes:
 If nnz is given then nz is ignored.

 Specify the preallocated storage with either nz or nnz (not both).
 Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
 allocation.  For large problems you MUST preallocate memory or you
 will get TERRIBLE performance, see the users' manual chapter on matrices.

 You can call MatGetInfo() to get information on how effective the preallocation was;
 for example the fields mallocs,nz_allocated,nz_used,nz_unneeded;
 You can also run with the option -info and look for messages with the string
 malloc in them to see if additional memory allocation was needed.

 Developers: Use nz of MAT_SKIP_ALLOCATION to not allocate any space for the matrix
 entries or columns indices.

 The maximum number of nonzeos in any row should be as accurate as possible.
 If it is underestimated, you will get bad performance due to reallocation
 (MatSeqXSELLReallocateSELL).

 Level: intermediate

 .seealso: MatCreate(), MatCreateSELL(), MatSetValues(), MatGetInfo()

 @*/
PetscErrorCode MatSeqSELLSetPreallocation(Mat B,PetscInt rlenmax,const PetscInt rlen[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  PetscCall(PetscTryMethod(B,"MatSeqSELLSetPreallocation_C",(Mat,PetscInt,const PetscInt[]),(B,rlenmax,rlen)));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqSELLSetPreallocation_SeqSELL(Mat B,PetscInt maxallocrow,const PetscInt rlen[])
{
  Mat_SeqSELL    *b;
  PetscInt       i,j,totalslices;
  PetscBool      skipallocation=PETSC_FALSE,realalloc=PETSC_FALSE;

  PetscFunctionBegin;
  if (maxallocrow >= 0 || rlen) realalloc = PETSC_TRUE;
  if (maxallocrow == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    maxallocrow    = 0;
  }

  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));

  /* FIXME: if one preallocates more space than needed, the matrix does not shrink automatically, but for best performance it should */
  if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 5;
  PetscCheckFalse(maxallocrow < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"maxallocrow cannot be less than 0: value %" PetscInt_FMT,maxallocrow);
  if (rlen) {
    for (i=0; i<B->rmap->n; i++) {
      PetscCheckFalse(rlen[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,rlen[i]);
      PetscCheckFalse(rlen[i] > B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be greater than row length: local row %" PetscInt_FMT " value %" PetscInt_FMT " rowlength %" PetscInt_FMT,i,rlen[i],B->cmap->n);
    }
  }

  B->preallocated = PETSC_TRUE;

  b = (Mat_SeqSELL*)B->data;

  totalslices = PetscCeilInt(B->rmap->n,8);
  b->totalslices = totalslices;
  if (!skipallocation) {
    if (B->rmap->n & 0x07) PetscCall(PetscInfo(B,"Padding rows to the SEQSELL matrix because the number of rows is not the multiple of 8 (value %" PetscInt_FMT ")\n",B->rmap->n));

    if (!b->sliidx) { /* sliidx gives the starting index of each slice, the last element is the total space allocated */
      PetscCall(PetscMalloc1(totalslices+1,&b->sliidx));
      PetscCall(PetscLogObjectMemory((PetscObject)B,(totalslices+1)*sizeof(PetscInt)));
    }
    if (!rlen) { /* if rlen is not provided, allocate same space for all the slices */
      if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 10;
      else if (maxallocrow < 0) maxallocrow = 1;
      for (i=0; i<=totalslices; i++) b->sliidx[i] = i*8*maxallocrow;
    } else {
      maxallocrow = 0;
      b->sliidx[0] = 0;
      for (i=1; i<totalslices; i++) {
        b->sliidx[i] = 0;
        for (j=0;j<8;j++) {
          b->sliidx[i] = PetscMax(b->sliidx[i],rlen[8*(i-1)+j]);
        }
        maxallocrow = PetscMax(b->sliidx[i],maxallocrow);
        PetscCall(PetscIntSumError(b->sliidx[i-1],8*b->sliidx[i],&b->sliidx[i]));
      }
      /* last slice */
      b->sliidx[totalslices] = 0;
      for (j=(totalslices-1)*8;j<B->rmap->n;j++) b->sliidx[totalslices] = PetscMax(b->sliidx[totalslices],rlen[j]);
      maxallocrow = PetscMax(b->sliidx[totalslices],maxallocrow);
      b->sliidx[totalslices] = b->sliidx[totalslices-1] + 8*b->sliidx[totalslices];
    }

    /* allocate space for val, colidx, rlen */
    /* FIXME: should B's old memory be unlogged? */
    PetscCall(MatSeqXSELLFreeSELL(B,&b->val,&b->colidx));
    /* FIXME: assuming an element of the bit array takes 8 bits */
    PetscCall(PetscMalloc2(b->sliidx[totalslices],&b->val,b->sliidx[totalslices],&b->colidx));
    PetscCall(PetscLogObjectMemory((PetscObject)B,b->sliidx[totalslices]*(sizeof(PetscScalar)+sizeof(PetscInt))));
    /* b->rlen will count nonzeros in each row so far. We dont copy rlen to b->rlen because the matrix has not been set. */
    PetscCall(PetscCalloc1(8*totalslices,&b->rlen));
    PetscCall(PetscLogObjectMemory((PetscObject)B,8*totalslices*sizeof(PetscInt)));

    b->singlemalloc = PETSC_TRUE;
    b->free_val     = PETSC_TRUE;
    b->free_colidx  = PETSC_TRUE;
  } else {
    b->free_val    = PETSC_FALSE;
    b->free_colidx = PETSC_FALSE;
  }

  b->nz               = 0;
  b->maxallocrow      = maxallocrow;
  b->rlenmax          = maxallocrow;
  b->maxallocmat      = b->sliidx[totalslices];
  B->info.nz_unneeded = (double)b->maxallocmat;
  if (realalloc) {
    PetscCall(MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetRow_SeqSELL(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_SeqSELL *a = (Mat_SeqSELL*)A->data;
  PetscInt    shift;

  PetscFunctionBegin;
  PetscCheckFalse(row < 0 || row >= A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %" PetscInt_FMT " out of range",row);
  if (nz) *nz = a->rlen[row];
  shift = a->sliidx[row>>3]+(row&0x07);
  if (!a->getrowcols) {

    PetscCall(PetscMalloc2(a->rlenmax,&a->getrowcols,a->rlenmax,&a->getrowvals));
  }
  if (idx) {
    PetscInt j;
    for (j=0; j<a->rlen[row]; j++) a->getrowcols[j] = a->colidx[shift+8*j];
    *idx = a->getrowcols;
  }
  if (v) {
    PetscInt j;
    for (j=0; j<a->rlen[row]; j++) a->getrowvals[j] = a->val[shift+8*j];
    *v = a->getrowvals;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatRestoreRow_SeqSELL(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqSELL_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       i;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    B    = *newmat;
    PetscCall(MatZeroEntries(B));
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
    PetscCall(MatSetType(B,MATSEQAIJ));
    PetscCall(MatSeqAIJSetPreallocation(B,0,a->rlen));
  }

  for (i=0; i<A->rmap->n; i++) {
    PetscInt    nz = 0,*cols = NULL;
    PetscScalar *vals = NULL;

    PetscCall(MatGetRow_SeqSELL(A,i,&nz,&cols,&vals));
    PetscCall(MatSetValues(B,1,&i,nz,cols,vals,INSERT_VALUES));
    PetscCall(MatRestoreRow_SeqSELL(A,i,&nz,&cols,&vals));
  }

  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  B->rmap->bs = A->rmap->bs;

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatConvert_SeqAIJ_SeqSELL(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               B;
  Mat_SeqAIJ        *a=(Mat_SeqAIJ*)A->data;
  PetscInt          *ai=a->i,m=A->rmap->N,n=A->cmap->N,i,*rowlengths,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;

  if (reuse == MAT_REUSE_MATRIX) {
    B = *newmat;
  } else {
    if (PetscDefined(USE_DEBUG) || !a->ilen) {
      PetscCall(PetscMalloc1(m,&rowlengths));
      for (i=0; i<m; i++) {
        rowlengths[i] = ai[i+1] - ai[i];
      }
    }
    if (PetscDefined(USE_DEBUG) && a->ilen) {
      PetscBool eq;
      PetscCall(PetscMemcmp(rowlengths,a->ilen,m*sizeof(PetscInt),&eq));
      PetscCheck(eq,PETSC_COMM_SELF,PETSC_ERR_PLIB,"SeqAIJ ilen array incorrect");
      PetscCall(PetscFree(rowlengths));
      rowlengths = a->ilen;
    } else if (a->ilen) rowlengths = a->ilen;
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
    PetscCall(MatSetSizes(B,m,n,m,n));
    PetscCall(MatSetType(B,MATSEQSELL));
    PetscCall(MatSeqSELLSetPreallocation(B,0,rowlengths));
    if (rowlengths != a->ilen) PetscCall(PetscFree(rowlengths));
  }

  for (row=0; row<m; row++) {
    PetscCall(MatGetRow_SeqAIJ(A,row,&ncols,(PetscInt**)&cols,(PetscScalar**)&vals));
    PetscCall(MatSetValues_SeqSELL(B,1,&row,ncols,cols,vals,INSERT_VALUES));
    PetscCall(MatRestoreRow_SeqAIJ(A,row,&ncols,(PetscInt**)&cols,(PetscScalar**)&vals));
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  B->rmap->bs = A->rmap->bs;

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A,&B));
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_SeqSELL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscInt          totalslices=a->totalslices;
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m512d           vec_x,vec_y,vec_vals;
  __m256i           vec_idx;
  __mmask8          mask;
  __m512d           vec_x2,vec_y2,vec_vals2,vec_x3,vec_y3,vec_vals3,vec_x4,vec_y4,vec_vals4;
  __m256i           vec_idx2,vec_idx3,vec_idx4;
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m128i           vec_idx;
  __m256d           vec_x,vec_y,vec_y2,vec_vals;
  MatScalar         yval;
  PetscInt          r,rows_left,row,nnz_in_row;
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m128d           vec_x_tmp;
  __m256d           vec_x,vec_y,vec_y2,vec_vals;
  MatScalar         yval;
  PetscInt          r,rows_left,row,nnz_in_row;
#else
  PetscScalar       sum[8];
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  for (i=0; i<totalslices; i++) { /* loop over slices */
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    vec_y  = _mm512_setzero_pd();
    vec_y2 = _mm512_setzero_pd();
    vec_y3 = _mm512_setzero_pd();
    vec_y4 = _mm512_setzero_pd();

    j = a->sliidx[i]>>3; /* 8 bytes are read at each time, corresponding to a slice columnn */
    switch ((a->sliidx[i+1]-a->sliidx[i])/8 & 3) {
    case 3:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      j += 3;
      break;
    case 2:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      j += 2;
      break;
    case 1:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      j += 1;
      break;
    }
    #pragma novector
    for (; j<(a->sliidx[i+1]>>3); j+=4) {
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx4,vec_x4,vec_vals4,vec_y4);
      acolidx += 8; aval += 8;
    }

    vec_y = _mm512_add_pd(vec_y,vec_y2);
    vec_y = _mm512_add_pd(vec_y,vec_y3);
    vec_y = _mm512_add_pd(vec_y,vec_y4);
    if (i == totalslices-1 && A->rmap->n & 0x07) { /* if last slice has padding rows */
      mask = (__mmask8)(0xff >> (8-(A->rmap->n & 0x07)));
      _mm512_mask_storeu_pd(&y[8*i],mask,vec_y);
    } else {
      _mm512_storeu_pd(&y[8*i],vec_y);
    }
  }
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX2__) && defined(__FMA__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  for (i=0; i<totalslices; i++) { /* loop over full slices */
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    /* last slice may have padding rows. Don't use vectorization. */
    if (i == totalslices-1 && (A->rmap->n & 0x07)) {
      rows_left = A->rmap->n - 8*i;
      for (r=0; r<rows_left; ++r) {
        yval = (MatScalar)0;
        row = 8*i + r;
        nnz_in_row = a->rlen[row];
        for (j=0; j<nnz_in_row; ++j) yval += aval[8*j+r] * x[acolidx[8*j+r]];
        y[row] = yval;
      }
      break;
    }

    vec_y  = _mm256_setzero_pd();
    vec_y2 = _mm256_setzero_pd();

    /* Process slice of height 8 (512 bits) via two subslices of height 4 (256 bits) via AVX */
    #pragma novector
    #pragma unroll(2)
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      AVX2_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      aval += 4; acolidx += 4;
      AVX2_Mult_Private(vec_idx,vec_x,vec_vals,vec_y2);
      aval += 4; acolidx += 4;
    }

    _mm256_storeu_pd(y+i*8,vec_y);
    _mm256_storeu_pd(y+i*8+4,vec_y2);
  }
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  for (i=0; i<totalslices; i++) { /* loop over full slices */
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    vec_y  = _mm256_setzero_pd();
    vec_y2 = _mm256_setzero_pd();

    /* last slice may have padding rows. Don't use vectorization. */
    if (i == totalslices-1 && (A->rmap->n & 0x07)) {
      rows_left = A->rmap->n - 8*i;
      for (r=0; r<rows_left; ++r) {
        yval = (MatScalar)0;
        row = 8*i + r;
        nnz_in_row = a->rlen[row];
        for (j=0; j<nnz_in_row; ++j) yval += aval[8*j + r] * x[acolidx[8*j + r]];
        y[row] = yval;
      }
      break;
    }

    /* Process slice of height 8 (512 bits) via two subslices of height 4 (256 bits) via AVX */
    #pragma novector
    #pragma unroll(2)
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      vec_vals  = _mm256_loadu_pd(aval);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,0);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,1);
      vec_y     = _mm256_add_pd(_mm256_mul_pd(vec_x,vec_vals),vec_y);
      aval     += 4;

      vec_vals  = _mm256_loadu_pd(aval);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,0);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,1);
      vec_y2    = _mm256_add_pd(_mm256_mul_pd(vec_x,vec_vals),vec_y2);
      aval     += 4;
    }

    _mm256_storeu_pd(y + i*8,     vec_y);
    _mm256_storeu_pd(y + i*8 + 4, vec_y2);
  }
#else
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=0; j<8; j++) sum[j] = 0.0;
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      sum[0] += aval[j] * x[acolidx[j]];
      sum[1] += aval[j+1] * x[acolidx[j+1]];
      sum[2] += aval[j+2] * x[acolidx[j+2]];
      sum[3] += aval[j+3] * x[acolidx[j+3]];
      sum[4] += aval[j+4] * x[acolidx[j+4]];
      sum[5] += aval[j+5] * x[acolidx[j+5]];
      sum[6] += aval[j+6] * x[acolidx[j+6]];
      sum[7] += aval[j+7] * x[acolidx[j+7]];
    }
    if (i == totalslices-1 && (A->rmap->n & 0x07)) { /* if last slice has padding rows */
      for (j=0; j<(A->rmap->n & 0x07); j++) y[8*i+j] = sum[j];
    } else {
      for (j=0; j<8; j++) y[8*i+j] = sum[j];
    }
  }
#endif

  PetscCall(PetscLogFlops(2.0*a->nz-a->nonzerorowcnt)); /* theoretical minimal FLOPs */
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmultadd.h>
PetscErrorCode MatMultAdd_SeqSELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscScalar       *y,*z;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscInt          totalslices=a->totalslices;
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m512d           vec_x,vec_y,vec_vals;
  __m256i           vec_idx;
  __mmask8          mask;
  __m512d           vec_x2,vec_y2,vec_vals2,vec_x3,vec_y3,vec_vals3,vec_x4,vec_y4,vec_vals4;
  __m256i           vec_idx2,vec_idx3,vec_idx4;
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  __m128d           vec_x_tmp;
  __m256d           vec_x,vec_y,vec_y2,vec_vals;
  MatScalar         yval;
  PetscInt          r,row,nnz_in_row;
#else
  PetscScalar       sum[8];
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArrayPair(yy,zz,&y,&z));
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX512F__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  for (i=0; i<totalslices; i++) { /* loop over slices */
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    if (i == totalslices-1 && A->rmap->n & 0x07) { /* if last slice has padding rows */
      mask   = (__mmask8)(0xff >> (8-(A->rmap->n & 0x07)));
      vec_y  = _mm512_mask_loadu_pd(vec_y,mask,&y[8*i]);
    } else {
      vec_y  = _mm512_loadu_pd(&y[8*i]);
    }
    vec_y2 = _mm512_setzero_pd();
    vec_y3 = _mm512_setzero_pd();
    vec_y4 = _mm512_setzero_pd();

    j = a->sliidx[i]>>3; /* 8 bytes are read at each time, corresponding to a slice columnn */
    switch ((a->sliidx[i+1]-a->sliidx[i])/8 & 3) {
    case 3:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      j += 3;
      break;
    case 2:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      j += 2;
      break;
    case 1:
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      j += 1;
      break;
    }
    #pragma novector
    for (; j<(a->sliidx[i+1]>>3); j+=4) {
      AVX512_Mult_Private(vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(vec_idx4,vec_x4,vec_vals4,vec_y4);
      acolidx += 8; aval += 8;
    }

    vec_y = _mm512_add_pd(vec_y,vec_y2);
    vec_y = _mm512_add_pd(vec_y,vec_y3);
    vec_y = _mm512_add_pd(vec_y,vec_y4);
    if (i == totalslices-1 && A->rmap->n & 0x07) { /* if last slice has padding rows */
      _mm512_mask_storeu_pd(&z[8*i],mask,vec_y);
    } else {
      _mm512_storeu_pd(&z[8*i],vec_y);
    }
  }
#elif defined(PETSC_HAVE_IMMINTRIN_H) && defined(__AVX__) && defined(PETSC_USE_REAL_DOUBLE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_64BIT_INDICES)
  for (i=0; i<totalslices; i++) { /* loop over full slices */
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    /* last slice may have padding rows. Don't use vectorization. */
    if (i == totalslices-1 && (A->rmap->n & 0x07)) {
      for (r=0; r<(A->rmap->n & 0x07); ++r) {
        row        = 8*i + r;
        yval       = (MatScalar)0.0;
        nnz_in_row = a->rlen[row];
        for (j=0; j<nnz_in_row; ++j) yval += aval[8*j+r] * x[acolidx[8*j+r]];
        z[row] = y[row] + yval;
      }
      break;
    }

    vec_y  = _mm256_loadu_pd(y+8*i);
    vec_y2 = _mm256_loadu_pd(y+8*i+4);

    /* Process slice of height 8 (512 bits) via two subslices of height 4 (256 bits) via AVX */
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      vec_vals  = _mm256_loadu_pd(aval);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,0);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,1);
      vec_y     = _mm256_add_pd(_mm256_mul_pd(vec_x,vec_vals),vec_y);
      aval     += 4;

      vec_vals  = _mm256_loadu_pd(aval);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,0);
      vec_x_tmp = _mm_loadl_pd(vec_x_tmp, x + *acolidx++);
      vec_x_tmp = _mm_loadh_pd(vec_x_tmp, x + *acolidx++);
      vec_x     = _mm256_insertf128_pd(vec_x,vec_x_tmp,1);
      vec_y2    = _mm256_add_pd(_mm256_mul_pd(vec_x,vec_vals),vec_y2);
      aval     += 4;
    }

    _mm256_storeu_pd(z+i*8,vec_y);
    _mm256_storeu_pd(z+i*8+4,vec_y2);
  }
#else
  for (i=0; i<totalslices; i++) { /* loop over slices */
    for (j=0; j<8; j++) sum[j] = 0.0;
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      sum[0] += aval[j] * x[acolidx[j]];
      sum[1] += aval[j+1] * x[acolidx[j+1]];
      sum[2] += aval[j+2] * x[acolidx[j+2]];
      sum[3] += aval[j+3] * x[acolidx[j+3]];
      sum[4] += aval[j+4] * x[acolidx[j+4]];
      sum[5] += aval[j+5] * x[acolidx[j+5]];
      sum[6] += aval[j+6] * x[acolidx[j+6]];
      sum[7] += aval[j+7] * x[acolidx[j+7]];
    }
    if (i == totalslices-1 && (A->rmap->n & 0x07)) {
      for (j=0; j<(A->rmap->n & 0x07); j++) z[8*i+j] = y[8*i+j] + sum[j];
    } else {
      for (j=0; j<8; j++) z[8*i+j] = y[8*i+j] + sum[j];
    }
  }
#endif

  PetscCall(PetscLogFlops(2.0*a->nz));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArrayPair(yy,zz,&y,&z));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqSELL(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j,r,row,nnz_in_row,totalslices=a->totalslices;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  if (A->symmetric) {
    PetscCall(MatMultAdd_SeqSELL(A,xx,zz,yy));
    PetscFunctionReturn(0);
  }
  if (zz != yy) PetscCall(VecCopy(zz,yy));
  PetscCall(VecGetArrayRead(xx,&x));
  PetscCall(VecGetArray(yy,&y));
  for (i=0; i<a->totalslices; i++) { /* loop over slices */
    if (i == totalslices-1 && (A->rmap->n & 0x07)) {
      for (r=0; r<(A->rmap->n & 0x07); ++r) {
        row        = 8*i + r;
        nnz_in_row = a->rlen[row];
        for (j=0; j<nnz_in_row; ++j) y[acolidx[8*j+r]] += aval[8*j+r] * x[row];
      }
      break;
    }
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      y[acolidx[j]]   += aval[j] * x[8*i];
      y[acolidx[j+1]] += aval[j+1] * x[8*i+1];
      y[acolidx[j+2]] += aval[j+2] * x[8*i+2];
      y[acolidx[j+3]] += aval[j+3] * x[8*i+3];
      y[acolidx[j+4]] += aval[j+4] * x[8*i+4];
      y[acolidx[j+5]] += aval[j+5] * x[8*i+5];
      y[acolidx[j+6]] += aval[j+6] * x[8*i+6];
      y[acolidx[j+7]] += aval[j+7] * x[8*i+7];
    }
  }
  PetscCall(PetscLogFlops(2.0*a->sliidx[a->totalslices]));
  PetscCall(VecRestoreArrayRead(xx,&x));
  PetscCall(VecRestoreArray(yy,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqSELL(Mat A,Vec xx,Vec yy)
{
  PetscFunctionBegin;
  if (A->symmetric) {
    PetscCall(MatMult_SeqSELL(A,xx,yy));
  } else {
    PetscCall(VecSet(yy,0.0));
    PetscCall(MatMultTransposeAdd_SeqSELL(A,xx,yy,yy));
  }
  PetscFunctionReturn(0);
}

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqSELL(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       *diag,i;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !(a->colidx)) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscCall(PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n"));
  } else {
    diag = a->diag;
    for (i=0; i<A->rmap->n; i++) {
      if (diag[i] == -1) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        PetscCall(PetscInfo(A,"Matrix is missing diagonal number %" PetscInt_FMT "\n",i));
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_SeqSELL(Mat A)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       i,j,m=A->rmap->n,shift;

  PetscFunctionBegin;
  if (!a->diag) {
    PetscCall(PetscMalloc1(m,&a->diag));
    PetscCall(PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt)));
    a->free_diag = PETSC_TRUE;
  }
  for (i=0; i<m; i++) { /* loop over rows */
    shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
    a->diag[i] = -1;
    for (j=0; j<a->rlen[i]; j++) {
      if (a->colidx[shift+j*8] == i) {
        a->diag[i] = shift+j*8;
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Negative shift indicates do not generate an error if there is a zero diagonal, just invert it anyways
*/
PetscErrorCode MatInvertDiagonal_SeqSELL(Mat A,PetscScalar omega,PetscScalar fshift)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*) A->data;
  PetscInt       i,*diag,m = A->rmap->n;
  MatScalar      *val = a->val;
  PetscScalar    *idiag,*mdiag;

  PetscFunctionBegin;
  if (a->idiagvalid) PetscFunctionReturn(0);
  PetscCall(MatMarkDiagonal_SeqSELL(A));
  diag = a->diag;
  if (!a->idiag) {
    PetscCall(PetscMalloc3(m,&a->idiag,m,&a->mdiag,m,&a->ssor_work));
    PetscCall(PetscLogObjectMemory((PetscObject)A, 3*m*sizeof(PetscScalar)));
    val  = a->val;
  }
  mdiag = a->mdiag;
  idiag = a->idiag;

  if (omega == 1.0 && PetscRealPart(fshift) <= 0.0) {
    for (i=0; i<m; i++) {
      mdiag[i] = val[diag[i]];
      if (!PetscAbsScalar(mdiag[i])) { /* zero diagonal */
        if (PetscRealPart(fshift)) {
          PetscCall(PetscInfo(A,"Zero diagonal on row %" PetscInt_FMT "\n",i));
          A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
          A->factorerror_zeropivot_value = 0.0;
          A->factorerror_zeropivot_row   = i;
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Zero diagonal on row %" PetscInt_FMT,i);
      }
      idiag[i] = 1.0/val[diag[i]];
    }
    PetscCall(PetscLogFlops(m));
  } else {
    for (i=0; i<m; i++) {
      mdiag[i] = val[diag[i]];
      idiag[i] = omega/(fshift + val[diag[i]]);
    }
    PetscCall(PetscLogFlops(2.0*m));
  }
  a->idiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_SeqSELL(Mat A)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
  PetscCall(PetscArrayzero(a->val,a->sliidx[a->totalslices]));
  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqSELL(Mat A)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%" PetscInt_FMT ", Cols=%" PetscInt_FMT ", NZ=%" PetscInt_FMT,A->rmap->n,A->cmap->n,a->nz);
#endif
  PetscCall(MatSeqXSELLFreeSELL(A,&a->val,&a->colidx));
  PetscCall(ISDestroy(&a->row));
  PetscCall(ISDestroy(&a->col));
  PetscCall(PetscFree(a->diag));
  PetscCall(PetscFree(a->rlen));
  PetscCall(PetscFree(a->sliidx));
  PetscCall(PetscFree3(a->idiag,a->mdiag,a->ssor_work));
  PetscCall(PetscFree(a->solve_work));
  PetscCall(ISDestroy(&a->icol));
  PetscCall(PetscFree(a->saved_values));
  PetscCall(PetscFree2(a->getrowcols,a->getrowvals));

  PetscCall(PetscFree(A->data));

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSeqSELLSetPreallocation_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqSELL(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented = flg;
    break;
  case MAT_KEEP_NONZERO_PATTERN:
    a->keepnonzeropattern = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
    a->nonew = (flg ? 0 : 1);
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew = (flg ? -1 : 0);
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew = (flg ? -2 : 0);
    break;
  case MAT_UNUSED_NONZERO_LOCATION_ERR:
    a->nounused = (flg ? -1 : 0);
    break;
  case MAT_FORCE_DIAGONAL_ENTRIES:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
  case MAT_SORTED_FULL:
    PetscCall(PetscInfo(A,"Option %s ignored\n",MatOptions[op]));
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_SeqSELL(Mat A,Vec v)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       i,j,n,shift;
  PetscScalar    *x,zero=0.0;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(v,&n));
  PetscCheckFalse(n != A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");

  if (A->factortype == MAT_FACTOR_ILU || A->factortype == MAT_FACTOR_LU) {
    PetscInt *diag=a->diag;
    PetscCall(VecGetArray(v,&x));
    for (i=0; i<n; i++) x[i] = 1.0/a->val[diag[i]];
    PetscCall(VecRestoreArray(v,&x));
    PetscFunctionReturn(0);
  }

  PetscCall(VecSet(v,zero));
  PetscCall(VecGetArray(v,&x));
  for (i=0; i<n; i++) { /* loop over rows */
    shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
    x[i] = 0;
    for (j=0; j<a->rlen[i]; j++) {
      if (a->colidx[shift+j*8] == i) {
        x[i] = a->val[shift+j*8];
        break;
      }
    }
  }
  PetscCall(VecRestoreArray(v,&x));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDiagonalScale_SeqSELL(Mat A,Vec ll,Vec rr)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  const PetscScalar *l,*r;
  PetscInt          i,j,m,n,row;

  PetscFunctionBegin;
  if (ll) {
    /* The local size is used so that VecMPI can be passed to this routine
       by MatDiagonalScale_MPISELL */
    PetscCall(VecGetLocalSize(ll,&m));
    PetscCheckFalse(m != A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    PetscCall(VecGetArrayRead(ll,&l));
    for (i=0; i<a->totalslices; i++) { /* loop over slices */
      if (i == a->totalslices-1 && (A->rmap->n & 0x07)) { /* if last slice has padding rows */
        for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
          if (row < (A->rmap->n & 0x07)) a->val[j] *= l[8*i+row];
        }
      } else {
        for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
          a->val[j] *= l[8*i+row];
        }
      }
    }
    PetscCall(VecRestoreArrayRead(ll,&l));
    PetscCall(PetscLogFlops(a->nz));
  }
  if (rr) {
    PetscCall(VecGetLocalSize(rr,&n));
    PetscCheckFalse(n != A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    PetscCall(VecGetArrayRead(rr,&r));
    for (i=0; i<a->totalslices; i++) { /* loop over slices */
      if (i == a->totalslices-1 && (A->rmap->n & 0x07)) { /* if last slice has padding rows */
        for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
          if (row < (A->rmap->n & 0x07)) a->val[j] *= r[a->colidx[j]];
        }
      } else {
        for (j=a->sliidx[i]; j<a->sliidx[i+1]; j++) {
          a->val[j] *= r[a->colidx[j]];
        }
      }
    }
    PetscCall(VecRestoreArrayRead(rr,&r));
    PetscCall(PetscLogFlops(a->nz));
  }
  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetValues_SeqSELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;
  PetscInt    *cp,i,k,low,high,t,row,col,l;
  PetscInt    shift;
  MatScalar   *vp;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over requested rows */
    row = im[k];
    if (row<0) continue;
    PetscCheck(row < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,row,A->rmap->n-1);
    shift = a->sliidx[row>>3]+(row&0x07); /* starting index of the row */
    cp = a->colidx+shift; /* pointer to the row */
    vp = a->val+shift; /* pointer to the row */
    for (l=0; l<n; l++) { /* loop over requested columns */
      col = in[l];
      if (col<0) continue;
      PetscCheck(col < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: row %" PetscInt_FMT " max %" PetscInt_FMT,col,A->cmap->n-1);
      high = a->rlen[row]; low = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (*(cp+t*8) > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (*(cp+8*i) > col) break;
        if (*(cp+8*i) == col) {
          *v++ = *(vp+8*i);
          goto finished;
        }
      }
      *v++ = 0.0;
    finished:;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqSELL_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscInt          i,j,m=A->rmap->n,shift;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer,&format));
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscInt nofinalvalue = 0;
    /*
    if (m && ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != A->cmap->n-1))) {
      nofinalvalue = 1;
    }
    */
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%% Size = %" PetscInt_FMT " %" PetscInt_FMT " \n",m,A->cmap->n));
    PetscCall(PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %" PetscInt_FMT " \n",a->nz));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer,"zzz = zeros(%" PetscInt_FMT ",4);\n",a->nz+nofinalvalue));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer,"zzz = zeros(%" PetscInt_FMT ",3);\n",a->nz+nofinalvalue));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer,"zzz = [\n"));

    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j])));
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)a->val[shift+8*j]));
#endif
      }
    }
    /*
    if (nofinalvalue) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e %18.16e\n",m,A->cmap->n,0.,0.));
#else
      PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT "  %18.16e\n",m,A->cmap->n,0.0));
#endif
    }
    */
    PetscCall(PetscObjectGetName((PetscObject)A,&name));
    PetscCall(PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name));
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (i=0; i<m; i++) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i));
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j])));
        } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j])));
        } else if (PetscRealPart(a->val[shift+8*j]) != 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j])));
        }
#else
        if (a->val[shift+8*j] != 0.0) PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_DENSE) {
    PetscInt    cnt=0,jcnt;
    PetscScalar value;
#if defined(PETSC_USE_COMPLEX)
    PetscBool   realonly=PETSC_TRUE;
    for (i=0; i<a->sliidx[a->totalslices]; i++) {
      if (PetscImaginaryPart(a->val[i]) != 0.0) {
        realonly = PETSC_FALSE;
        break;
      }
    }
#endif

    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    for (i=0; i<m; i++) {
      jcnt = 0;
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<A->cmap->n; j++) {
        if (jcnt < a->rlen[i] && j == a->colidx[shift+8*j]) {
          value = a->val[cnt++];
          jcnt++;
        } else {
          value = 0.0;
        }
#if defined(PETSC_USE_COMPLEX)
        if (realonly) {
          PetscCall(PetscViewerASCIIPrintf(viewer," %7.5e ",(double)PetscRealPart(value)));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer," %7.5e+%7.5e i ",(double)PetscRealPart(value),(double)PetscImaginaryPart(value)));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer," %7.5e ",(double)value));
#endif
      }
      PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else if (format == PETSC_VIEWER_ASCII_MATRIXMARKET) {
    PetscInt fshift=1;
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
#if defined(PETSC_USE_COMPLEX)
    PetscCall(PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate complex general\n"));
#else
    PetscCall(PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate real general\n"));
#endif
    PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "\n", m, A->cmap->n, a->nz));
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT " %g %g\n",i+fshift,a->colidx[shift+8*j]+fshift,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j])));
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"%" PetscInt_FMT " %" PetscInt_FMT " %g\n",i+fshift,a->colidx[shift+8*j]+fshift,(double)a->val[shift+8*j]));
#endif
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  } else if (format == PETSC_VIEWER_NATIVE) {
    for (i=0; i<a->totalslices; i++) { /* loop over slices */
      PetscInt row;
      PetscCall(PetscViewerASCIIPrintf(viewer,"slice %" PetscInt_FMT ": %" PetscInt_FMT " %" PetscInt_FMT "\n",i,a->sliidx[i],a->sliidx[i+1]));
      for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[j]) > 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"  %" PetscInt_FMT " %" PetscInt_FMT " %g + %g i\n",8*i+row,a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j])));
        } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer,"  %" PetscInt_FMT " %" PetscInt_FMT " %g - %g i\n",8*i+row,a->colidx[j],(double)PetscRealPart(a->val[j]),-(double)PetscImaginaryPart(a->val[j])));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer,"  %" PetscInt_FMT " %" PetscInt_FMT " %g\n",8*i+row,a->colidx[j],(double)PetscRealPart(a->val[j])));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer,"  %" PetscInt_FMT " %" PetscInt_FMT " %g\n",8*i+row,a->colidx[j],(double)a->val[j]));
#endif
      }
    }
  } else {
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
    if (A->factortype) {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07);
        PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i));
        /* L part */
        for (j=shift; j<a->diag[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j])));
          } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j]))));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)PetscRealPart(a->val[j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)a->val[j]));
#endif
        }
        /* diagonal */
        j = a->diag[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[j]) > 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)PetscImaginaryPart(1.0/a->val[j])));
        } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)(-PetscImaginaryPart(1.0/a->val[j]))));
        } else {
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)PetscRealPart(1.0/a->val[j])));
        }
#else
        PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)(1.0/a->val[j])));
#endif

        /* U part */
        for (j=a->diag[i]+1; j<shift+8*a->rlen[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j])));
          } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j]))));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)PetscRealPart(a->val[j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[j],(double)a->val[j]));
#endif
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    } else {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07);
        PetscCall(PetscViewerASCIIPrintf(viewer,"row %" PetscInt_FMT ":",i));
        for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j])));
          } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j])));
          } else {
            PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j])));
          }
#else
          PetscCall(PetscViewerASCIIPrintf(viewer," (%" PetscInt_FMT ", %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]));
#endif
        }
        PetscCall(PetscViewerASCIIPrintf(viewer,"\n"));
      }
    }
    PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
  }
  PetscCall(PetscViewerFlush(viewer));
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode MatView_SeqSELL_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A=(Mat)Aa;
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscInt          i,j,m=A->rmap->n,shift;
  int               color;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer));
  PetscCall(PetscViewerGetFormat(viewer,&format));
  PetscCall(PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr));

  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) >=  0.) continue;
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
      }
    }
    color = PETSC_DRAW_CYAN;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (a->val[shift+8*j] !=  0.) continue;
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
      }
    }
    color = PETSC_DRAW_RED;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) <=  0.) continue;
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv=0.0,maxv=0.0;
    PetscInt  count=0;
    PetscDraw popup;
    for (i=0; i<a->sliidx[a->totalslices]; i++) {
      if (PetscAbsScalar(a->val[i]) > maxv) maxv = PetscAbsScalar(a->val[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    PetscCall(PetscDrawGetPopup(draw,&popup));
    PetscCall(PetscDrawScalePopup(popup,minv,maxv));

    ierr = PetscDrawCollectiveBegin(draw);PetscCall(ierr);
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8];
        x_r = x_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(a->val[count]),minv,maxv);
        PetscCall(PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color));
        count++;
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);PetscCall(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode MatView_SeqSELL_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscBool      isnull;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawIsNull(draw,&isnull));
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr  = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;         xl = -w;     yl = -h;
  PetscCall(PetscDrawSetCoordinates(draw,xl,yl,xr,yr));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer));
  PetscCall(PetscDrawZoom(draw,MatView_SeqSELL_Draw_Zoom,A));
  PetscCall(PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqSELL(Mat A,PetscViewer viewer)
{
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw));
  if (iascii) {
    PetscCall(MatView_SeqSELL_ASCII(A,viewer));
  } else if (isbinary) {
    /* PetscCall(MatView_SeqSELL_Binary(A,viewer)); */
  } else if (isdraw) {
    PetscCall(MatView_SeqSELL_Draw(A,viewer));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqSELL(Mat A,MatAssemblyType mode)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       i,shift,row_in_slice,row,nrow,*cp,lastcol,j,k;
  MatScalar      *vp;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  /* To do: compress out the unused elements */
  PetscCall(MatMarkDiagonal_SeqSELL(A));
  PetscCall(PetscInfo(A,"Matrix size: %" PetscInt_FMT " X %" PetscInt_FMT "; storage space: %" PetscInt_FMT " allocated %" PetscInt_FMT " used (%" PetscInt_FMT " nonzeros+%" PetscInt_FMT " paddedzeros)\n",A->rmap->n,A->cmap->n,a->maxallocmat,a->sliidx[a->totalslices],a->nz,a->sliidx[a->totalslices]-a->nz));
  PetscCall(PetscInfo(A,"Number of mallocs during MatSetValues() is %" PetscInt_FMT "\n",a->reallocs));
  PetscCall(PetscInfo(A,"Maximum nonzeros in any row is %" PetscInt_FMT "\n",a->rlenmax));
  /* Set unused slots for column indices to last valid column index. Set unused slots for values to zero. This allows for a use of unmasked intrinsics -> higher performance */
  for (i=0; i<a->totalslices; ++i) {
    shift = a->sliidx[i];    /* starting index of the slice */
    cp    = a->colidx+shift; /* pointer to the column indices of the slice */
    vp    = a->val+shift;    /* pointer to the nonzero values of the slice */
    for (row_in_slice=0; row_in_slice<8; ++row_in_slice) { /* loop over rows in the slice */
      row  = 8*i + row_in_slice;
      nrow = a->rlen[row]; /* number of nonzeros in row */
      /*
        Search for the nearest nonzero. Normally setting the index to zero may cause extra communication.
        But if the entire slice are empty, it is fine to use 0 since the index will not be loaded.
      */
      lastcol = 0;
      if (nrow>0) { /* nonempty row */
        lastcol = cp[8*(nrow-1)+row_in_slice]; /* use the index from the last nonzero at current row */
      } else if (!row_in_slice) { /* first row of the currect slice is empty */
        for (j=1;j<8;j++) {
          if (a->rlen[8*i+j]) {
            lastcol = cp[j];
            break;
          }
        }
      } else {
        if (a->sliidx[i+1] != shift) lastcol = cp[row_in_slice-1]; /* use the index from the previous row */
      }

      for (k=nrow; k<(a->sliidx[i+1]-shift)/8; ++k) {
        cp[8*k+row_in_slice] = lastcol;
        vp[8*k+row_in_slice] = (MatScalar)0;
      }
    }
  }

  A->info.mallocs += a->reallocs;
  a->reallocs      = 0;

  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_SeqSELL(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = a->maxallocmat;
  info->nz_used      = a->sliidx[a->totalslices]; /* include padding zeros */
  info->nz_unneeded  = (a->maxallocmat-a->sliidx[a->totalslices]);
  info->assemblies   = A->num_ass;
  info->mallocs      = A->info.mallocs;
  info->memory       = ((PetscObject)A)->mem;
  if (A->factortype) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetValues_SeqSELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       shift,i,k,l,low,high,t,ii,row,col,nrow;
  PetscInt       *cp,nonew=a->nonew,lastcol=-1;
  MatScalar      *vp,value;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
    PetscCheck(row < A->rmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %" PetscInt_FMT " max %" PetscInt_FMT,row,A->rmap->n-1);
    shift = a->sliidx[row>>3]+(row&0x07); /* starting index of the row */
    cp    = a->colidx+shift; /* pointer to the row */
    vp    = a->val+shift; /* pointer to the row */
    nrow  = a->rlen[row];
    low   = 0;
    high  = nrow;

    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l];
      if (col<0) continue;
      PetscCheck(col < A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %" PetscInt_FMT " max %" PetscInt_FMT,col,A->cmap->n-1);
      if (a->roworiented) {
        value = v[l+k*n];
      } else {
        value = v[k+l*m];
      }
      if ((value == 0.0 && a->ignorezeroentries) && (is == ADD_VALUES)) continue;

      /* search in this row for the specified column, i indicates the column to be set */
      if (col <= lastcol) low = 0;
      else high = nrow;
      lastcol = col;
      while (high-low > 5) {
        t = (low+high)/2;
        if (*(cp+t*8) > col) high = t;
        else low = t;
      }
      for (i=low; i<high; i++) {
        if (*(cp+i*8) > col) break;
        if (*(cp+i*8) == col) {
          if (is == ADD_VALUES) *(vp+i*8) += value;
          else *(vp+i*8) = value;
          low = i + 1;
          goto noinsert;
        }
      }
      if (value == 0.0 && a->ignorezeroentries) goto noinsert;
      if (nonew == 1) goto noinsert;
      PetscCheckFalse(nonew == -1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%" PetscInt_FMT ", %" PetscInt_FMT ") in the matrix", row, col);
      /* If the current row length exceeds the slice width (e.g. nrow==slice_width), allocate a new space, otherwise do nothing */
      MatSeqXSELLReallocateSELL(A,A->rmap->n,1,nrow,a->sliidx,row/8,row,col,a->colidx,a->val,cp,vp,nonew,MatScalar);
      /* add the new nonzero to the high position, shift the remaining elements in current row to the right by one slot */
      for (ii=nrow-1; ii>=i; ii--) {
        *(cp+(ii+1)*8) = *(cp+ii*8);
        *(vp+(ii+1)*8) = *(vp+ii*8);
      }
      a->rlen[row]++;
      *(cp+i*8) = col;
      *(vp+i*8) = value;
      a->nz++;
      A->nonzerostate++;
      low = i+1; high++; nrow++;
noinsert:;
    }
    a->rlen[row] = nrow;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqSELL(Mat A,Mat B,MatStructure str)
{
  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;
    Mat_SeqSELL *b=(Mat_SeqSELL*)B->data;

    PetscCheckFalse(a->sliidx[a->totalslices] != b->sliidx[b->totalslices],PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    PetscCall(PetscArraycpy(b->val,a->val,a->sliidx[a->totalslices]));
  } else {
    PetscCall(MatCopy_Basic(A,B,str));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqSELL(Mat A)
{
  PetscFunctionBegin;
  PetscCall(MatSeqSELLSetPreallocation(A,PETSC_DEFAULT,NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqSELLGetArray_SeqSELL(Mat A,PetscScalar *array[])
{
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
  *array = a->val;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqSELLRestoreArray_SeqSELL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_SeqSELL(Mat A)
{
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;
  PetscInt    i;
  MatScalar   *aval=a->val;

  PetscFunctionBegin;
  for (i=0; i<a->sliidx[a->totalslices]; i++) aval[i]=PetscRealPart(aval[i]);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_SeqSELL(Mat A)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data;
  PetscInt       i;
  MatScalar      *aval=a->val;

  PetscFunctionBegin;
  for (i=0; i<a->sliidx[a->totalslices]; i++) aval[i] = PetscImaginaryPart(aval[i]);
  PetscCall(MatSeqSELLInvalidateDiagonal(A));
  PetscFunctionReturn(0);
}

PetscErrorCode MatScale_SeqSELL(Mat inA,PetscScalar alpha)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)inA->data;
  MatScalar      *aval=a->val;
  PetscScalar    oalpha=alpha;
  PetscBLASInt   one=1,size;

  PetscFunctionBegin;
  PetscCall(PetscBLASIntCast(a->sliidx[a->totalslices],&size));
  PetscStackCallBLAS("BLASscal",BLASscal_(&size,&oalpha,aval,&one));
  PetscCall(PetscLogFlops(a->nz));
  PetscCall(MatSeqSELLInvalidateDiagonal(inA));
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqSELL(Mat Y,PetscScalar a)
{
  Mat_SeqSELL    *y=(Mat_SeqSELL*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !y->nz) {
    PetscCall(MatSeqSELLSetPreallocation(Y,1,NULL));
  }
  PetscCall(MatShift_Basic(Y,a));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqSELL(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqSELL       *a=(Mat_SeqSELL*)A->data;
  PetscScalar       *x,sum,*t;
  const MatScalar   *idiag=NULL,*mdiag;
  const PetscScalar *b,*xb;
  PetscInt          n,m=A->rmap->n,i,j,shift;
  const PetscInt    *diag;

  PetscFunctionBegin;
  its = its*lits;

  if (fshift != a->fshift || omega != a->omega) a->idiagvalid = PETSC_FALSE; /* must recompute idiag[] */
  if (!a->idiagvalid) PetscCall(MatInvertDiagonal_SeqSELL(A,omega,fshift));
  a->fshift = fshift;
  a->omega  = omega;

  diag  = a->diag;
  t     = a->ssor_work;
  idiag = a->idiag;
  mdiag = a->mdiag;

  PetscCall(VecGetArray(xx,&x));
  PetscCall(VecGetArrayRead(bb,&b));
  /* We count flops by assuming the upper triangular and lower triangular parts have the same number of nonzeros */
  PetscCheckFalse(flag == SOR_APPLY_UPPER,PETSC_COMM_SELF,PETSC_ERR_SUP,"SOR_APPLY_UPPER is not implemented");
  PetscCheckFalse(flag == SOR_APPLY_LOWER,PETSC_COMM_SELF,PETSC_ERR_SUP,"SOR_APPLY_LOWER is not implemented");
  PetscCheckFalse(flag & SOR_EISENSTAT,PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if ((flag & SOR_FORWARD_SWEEP) || (flag & SOR_LOCAL_FORWARD_SWEEP)) {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
        sum   = b[i];
        n     = (diag[i]-shift)/8;
        for (j=0; j<n; j++) sum -= a->val[shift+j*8]*x[a->colidx[shift+j*8]];
        t[i]  = sum;
        x[i]  = sum*idiag[i];
      }
      xb   = t;
      PetscCall(PetscLogFlops(a->nz));
    } else xb = b;
    if ((flag & SOR_BACKWARD_SWEEP) || (flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=m-1; i>=0; i--) {
        shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
        sum   = xb[i];
        n     = a->rlen[i]-(diag[i]-shift)/8-1;
        for (j=1; j<=n; j++) sum -= a->val[diag[i]+j*8]*x[a->colidx[diag[i]+j*8]];
        if (xb == b) {
          x[i] = sum*idiag[i];
        } else {
          x[i] = (1.-omega)*x[i]+sum*idiag[i];  /* omega in idiag */
        }
      }
      PetscCall(PetscLogFlops(a->nz)); /* assumes 1/2 in upper */
    }
    its--;
  }
  while (its--) {
    if ((flag & SOR_FORWARD_SWEEP) || (flag & SOR_LOCAL_FORWARD_SWEEP)) {
      for (i=0; i<m; i++) {
        /* lower */
        shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
        sum   = b[i];
        n     = (diag[i]-shift)/8;
        for (j=0; j<n; j++) sum -= a->val[shift+j*8]*x[a->colidx[shift+j*8]];
        t[i]  = sum;             /* save application of the lower-triangular part */
        /* upper */
        n     = a->rlen[i]-(diag[i]-shift)/8-1;
        for (j=1; j<=n; j++) sum -= a->val[diag[i]+j*8]*x[a->colidx[diag[i]+j*8]];
        x[i]  = (1.-omega)*x[i]+sum*idiag[i];  /* omega in idiag */
      }
      xb   = t;
      PetscCall(PetscLogFlops(2.0*a->nz));
    } else xb = b;
    if ((flag & SOR_BACKWARD_SWEEP) || (flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=m-1; i>=0; i--) {
        shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
        sum = xb[i];
        if (xb == b) {
          /* whole matrix (no checkpointing available) */
          n     = a->rlen[i];
          for (j=0; j<n; j++) sum -= a->val[shift+j*8]*x[a->colidx[shift+j*8]];
          x[i] = (1.-omega)*x[i]+(sum+mdiag[i]*x[i])*idiag[i];
        } else { /* lower-triangular part has been saved, so only apply upper-triangular */
          n     = a->rlen[i]-(diag[i]-shift)/8-1;
          for (j=1; j<=n; j++) sum -= a->val[diag[i]+j*8]*x[a->colidx[diag[i]+j*8]];
          x[i]  = (1.-omega)*x[i]+sum*idiag[i];  /* omega in idiag */
        }
      }
      if (xb == b) {
        PetscCall(PetscLogFlops(2.0*a->nz));
      } else {
        PetscCall(PetscLogFlops(a->nz)); /* assumes 1/2 in upper */
      }
    }
  }
  PetscCall(VecRestoreArray(xx,&x));
  PetscCall(VecRestoreArrayRead(bb,&b));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqSELL,
                                       MatGetRow_SeqSELL,
                                       MatRestoreRow_SeqSELL,
                                       MatMult_SeqSELL,
                               /* 4*/  MatMultAdd_SeqSELL,
                                       MatMultTranspose_SeqSELL,
                                       MatMultTransposeAdd_SeqSELL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 10*/ NULL,
                                       NULL,
                                       NULL,
                                       MatSOR_SeqSELL,
                                       NULL,
                               /* 15*/ MatGetInfo_SeqSELL,
                                       MatEqual_SeqSELL,
                                       MatGetDiagonal_SeqSELL,
                                       MatDiagonalScale_SeqSELL,
                                       NULL,
                               /* 20*/ NULL,
                                       MatAssemblyEnd_SeqSELL,
                                       MatSetOption_SeqSELL,
                                       MatZeroEntries_SeqSELL,
                               /* 24*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 29*/ MatSetUp_SeqSELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 34*/ MatDuplicate_SeqSELL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 39*/ NULL,
                                       NULL,
                                       NULL,
                                       MatGetValues_SeqSELL,
                                       MatCopy_SeqSELL,
                               /* 44*/ NULL,
                                       MatScale_SeqSELL,
                                       MatShift_SeqSELL,
                                       NULL,
                                       NULL,
                               /* 49*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 54*/ MatFDColoringCreate_SeqXAIJ,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 59*/ NULL,
                                       MatDestroy_SeqSELL,
                                       MatView_SeqSELL,
                                       NULL,
                                       NULL,
                               /* 64*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 69*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 74*/ NULL,
                                       MatFDColoringApply_AIJ, /* reuse the FDColoring function for AIJ */
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 79*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 84*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 89*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 94*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /* 99*/ NULL,
                                       NULL,
                                       NULL,
                                       MatConjugate_SeqSELL,
                                       NULL,
                               /*104*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*109*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       MatMissingDiagonal_SeqSELL,
                               /*114*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*119*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*124*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*129*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*134*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                                       NULL,
                               /*139*/ NULL,
                                       NULL,
                                       NULL,
                                       MatFDColoringSetUp_SeqXAIJ,
                                       NULL,
                               /*144*/ NULL,
                                       NULL,
                                       NULL,
                                       NULL
};

PetscErrorCode MatStoreValues_SeqSELL(Mat mat)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)mat->data;

  PetscFunctionBegin;
  PetscCheck(a->nonew,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!a->saved_values) {
    PetscCall(PetscMalloc1(a->sliidx[a->totalslices]+1,&a->saved_values));
    PetscCall(PetscLogObjectMemory((PetscObject)mat,(a->sliidx[a->totalslices]+1)*sizeof(PetscScalar)));
  }

  /* copy values over */
  PetscCall(PetscArraycpy(a->saved_values,a->val,a->sliidx[a->totalslices]));
  PetscFunctionReturn(0);
}

PetscErrorCode MatRetrieveValues_SeqSELL(Mat mat)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)mat->data;

  PetscFunctionBegin;
  PetscCheck(a->nonew,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  PetscCheck(a->saved_values,PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
  PetscCall(PetscArraycpy(a->val,a->saved_values,a->sliidx[a->totalslices]));
  PetscFunctionReturn(0);
}

/*@C
 MatSeqSELLRestoreArray - returns access to the array where the data for a MATSEQSELL matrix is stored obtained by MatSeqSELLGetArray()

 Not Collective

 Input Parameters:
 .  mat - a MATSEQSELL matrix
 .  array - pointer to the data

 Level: intermediate

 .seealso: MatSeqSELLGetArray(), MatSeqSELLRestoreArrayF90()
 @*/
PetscErrorCode MatSeqSELLRestoreArray(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscCall(PetscUseMethod(A,"MatSeqSELLRestoreArray_C",(Mat,PetscScalar**),(A,array)));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqSELL(Mat B)
{
  Mat_SeqSELL    *b;
  PetscMPIInt    size;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(citation,&cited));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
  PetscCheckFalse(size > 1,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Comm must be of size 1");

  PetscCall(PetscNewLog(B,&b));

  B->data = (void*)b;

  PetscCall(PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps)));

  b->row                = NULL;
  b->col                = NULL;
  b->icol               = NULL;
  b->reallocs           = 0;
  b->ignorezeroentries  = PETSC_FALSE;
  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = NULL;
  b->solve_work         = NULL;
  B->spptr              = NULL;
  b->saved_values       = NULL;
  b->idiag              = NULL;
  b->mdiag              = NULL;
  b->ssor_work          = NULL;
  b->omega              = 1.0;
  b->fshift             = 0.0;
  b->idiagvalid         = PETSC_FALSE;
  b->keepnonzeropattern = PETSC_FALSE;

  PetscCall(PetscObjectChangeTypeName((PetscObject)B,MATSEQSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqSELLGetArray_C",MatSeqSELLGetArray_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqSELLRestoreArray_C",MatSeqSELLRestoreArray_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatSeqSELLSetPreallocation_C",MatSeqSELLSetPreallocation_SeqSELL));
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqsell_seqaij_C",MatConvert_SeqSELL_SeqAIJ));
  PetscFunctionReturn(0);
}

/*
 Given a matrix generated with MatGetFactor() duplicates all the information in A into B
 */
PetscErrorCode MatDuplicateNoCreate_SeqSELL(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqSELL    *c = (Mat_SeqSELL*)C->data,*a = (Mat_SeqSELL*)A->data;
  PetscInt       i,m=A->rmap->n;
  PetscInt       totalslices=a->totalslices;

  PetscFunctionBegin;
  C->factortype = A->factortype;
  c->row        = NULL;
  c->col        = NULL;
  c->icol       = NULL;
  c->reallocs   = 0;
  C->assembled = PETSC_TRUE;

  PetscCall(PetscLayoutReference(A->rmap,&C->rmap));
  PetscCall(PetscLayoutReference(A->cmap,&C->cmap));

  PetscCall(PetscMalloc1(8*totalslices,&c->rlen));
  PetscCall(PetscLogObjectMemory((PetscObject)C,m*sizeof(PetscInt)));
  PetscCall(PetscMalloc1(totalslices+1,&c->sliidx));
  PetscCall(PetscLogObjectMemory((PetscObject)C, (totalslices+1)*sizeof(PetscInt)));

  for (i=0; i<m; i++) c->rlen[i] = a->rlen[i];
  for (i=0; i<totalslices+1; i++) c->sliidx[i] = a->sliidx[i];

  /* allocate the matrix space */
  if (mallocmatspace) {
    PetscCall(PetscMalloc2(a->maxallocmat,&c->val,a->maxallocmat,&c->colidx));
    PetscCall(PetscLogObjectMemory((PetscObject)C,a->maxallocmat*(sizeof(PetscScalar)+sizeof(PetscInt))));

    c->singlemalloc = PETSC_TRUE;

    if (m > 0) {
      PetscCall(PetscArraycpy(c->colidx,a->colidx,a->maxallocmat));
      if (cpvalues == MAT_COPY_VALUES) {
        PetscCall(PetscArraycpy(c->val,a->val,a->maxallocmat));
      } else {
        PetscCall(PetscArrayzero(c->val,a->maxallocmat));
      }
    }
  }

  c->ignorezeroentries = a->ignorezeroentries;
  c->roworiented       = a->roworiented;
  c->nonew             = a->nonew;
  if (a->diag) {
    PetscCall(PetscMalloc1(m,&c->diag));
    PetscCall(PetscLogObjectMemory((PetscObject)C,m*sizeof(PetscInt)));
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag = NULL;

  c->solve_work         = NULL;
  c->saved_values       = NULL;
  c->idiag              = NULL;
  c->ssor_work          = NULL;
  c->keepnonzeropattern = a->keepnonzeropattern;
  c->free_val           = PETSC_TRUE;
  c->free_colidx        = PETSC_TRUE;

  c->maxallocmat  = a->maxallocmat;
  c->maxallocrow  = a->maxallocrow;
  c->rlenmax      = a->rlenmax;
  c->nz           = a->nz;
  C->preallocated = PETSC_TRUE;

  c->nonzerorowcnt = a->nonzerorowcnt;
  C->nonzerostate  = A->nonzerostate;

  PetscCall(PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist));
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqSELL(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),B));
  PetscCall(MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n));
  if (!(A->rmap->n % A->rmap->bs) && !(A->cmap->n % A->cmap->bs)) {
    PetscCall(MatSetBlockSizesFromMats(*B,A,A));
  }
  PetscCall(MatSetType(*B,((PetscObject)A)->type_name));
  PetscCall(MatDuplicateNoCreate_SeqSELL(*B,A,cpvalues,PETSC_TRUE));
  PetscFunctionReturn(0);
}

/*MC
   MATSEQSELL - MATSEQSELL = "seqsell" - A matrix type to be used for sequential sparse matrices,
   based on the sliced Ellpack format

   Options Database Keys:
. -mat_type seqsell - sets the matrix type to "seqsell" during a call to MatSetFromOptions()

   Level: beginner

.seealso: MatCreateSeqSell(), MATSELL, MATMPISELL, MATSEQAIJ, MATAIJ, MATMPIAIJ
M*/

/*MC
   MATSELL - MATSELL = "sell" - A matrix type to be used for sparse matrices.

   This matrix type is identical to MATSEQSELL when constructed with a single process communicator,
   and MATMPISELL otherwise.  As a result, for single process communicators,
  MatSeqSELLSetPreallocation() is supported, and similarly MatMPISELLSetPreallocation() is supported
  for communicators controlling multiple processes.  It is recommended that you call both of
  the above preallocation routines for simplicity.

   Options Database Keys:
. -mat_type sell - sets the matrix type to "sell" during a call to MatSetFromOptions()

  Level: beginner

  Notes:
   This format is only supported for real scalars, double precision, and 32 bit indices (the defaults).

   It can provide better performance on Intel and AMD processes with AVX2 or AVX512 support for matrices that have a similar number of
   non-zeros in contiguous groups of rows. However if the computation is memory bandwidth limited it may not provide much improvement.

  Developer Notes:
   On Intel (and AMD) systems some of the matrix operations use SIMD (AVX) instructions to achieve higher performance.

   The sparse matrix format is as follows. For simplicity we assume a slice size of 2, it is actually 8
.vb
                            (2 0  3 4)
   Consider the matrix A =  (5 0  6 0)
                            (0 0  7 8)
                            (0 0  9 9)

   symbolically the Ellpack format can be written as

        (2 3 4 |)           (0 2 3 |)
   v =  (5 6 0 |)  colidx = (0 2 2 |)
        --------            ---------
        (7 8 |)             (2 3 |)
        (9 9 |)             (2 3 |)

    The data for 2 contiguous rows of the matrix are stored together (in column-major format) (with any left-over rows handled as a special case).
    Any of the rows in a slice fewer columns than the rest of the slice (row 1 above) are padded with a previous valid column in their "extra" colidx[] locations and
    zeros in their "extra" v locations so that the matrix operations do not need special code to handle different length rows within the 2 rows in a slice.

    The one-dimensional representation of v used in the code is (2 5 3 6 4 0 7 9 8 9)  and for colidx is (0 0 2 2 3 2 2 2 3 3)

.ve

      See MatMult_SeqSELL() for how this format is used with the SIMD operations to achieve high performance.

 References:
. * - Hong Zhang, Richard T. Mills, Karl Rupp, and Barry F. Smith, Vectorized Parallel Sparse Matrix-Vector Multiplication in {PETSc} Using {AVX-512},
   Proceedings of the 47th International Conference on Parallel Processing, 2018.

.seealso: MatCreateSeqSELL(), MatCreateSeqAIJ(), MatCreateSell(), MATSEQSELL, MATMPISELL, MATSEQAIJ, MATMPIAIJ, MATAIJ
M*/

/*@C
       MatCreateSeqSELL - Creates a sparse matrix in SELL format.

 Collective on comm

 Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  rlenmax - maximum number of nonzeros in a row
-  rlen - array containing the number of nonzeros in the various rows
 (possibly different for each row) or NULL

 Output Parameter:
.  A - the matrix

 It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
 MatXXXXSetPreallocation() paradigm instead of this routine directly.
 [MatXXXXSetPreallocation() is, for example, MatSeqSELLSetPreallocation]

 Notes:
 If nnz is given then nz is ignored

 Specify the preallocated storage with either rlenmax or rlen (not both).
 Set rlenmax=PETSC_DEFAULT and rlen=NULL for PETSc to control dynamic memory
 allocation.  For large problems you MUST preallocate memory or you
 will get TERRIBLE performance, see the users' manual chapter on matrices.

 Level: intermediate

 .seealso: MatCreate(), MatCreateSELL(), MatSetValues(), MatSeqSELLSetPreallocation(), MATSELL, MATSEQSELL, MATMPISELL

 @*/
PetscErrorCode MatCreateSeqSELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt maxallocrow,const PetscInt rlen[],Mat *A)
{
  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,m,n));
  PetscCall(MatSetType(*A,MATSEQSELL));
  PetscCall(MatSeqSELLSetPreallocation_SeqSELL(*A,maxallocrow,rlen));
  PetscFunctionReturn(0);
}

PetscErrorCode MatEqual_SeqSELL(Mat A,Mat B,PetscBool * flg)
{
  Mat_SeqSELL    *a=(Mat_SeqSELL*)A->data,*b=(Mat_SeqSELL*)B->data;
  PetscInt       totalslices=a->totalslices;

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) || (A->cmap->n != B->cmap->n) ||(a->nz != b->nz) || (a->rlenmax != b->rlenmax)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  /* if the a->colidx are the same */
  PetscCall(PetscArraycmp(a->colidx,b->colidx,a->sliidx[totalslices],flg));
  if (!*flg) PetscFunctionReturn(0);
  /* if a->val are the same */
  PetscCall(PetscArraycmp(a->val,b->val,a->sliidx[totalslices],flg));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqSELLInvalidateDiagonal(Mat A)
{
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;

  PetscFunctionBegin;
  a->idiagvalid  = PETSC_FALSE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatConjugate_SeqSELL(Mat A)
{
#if defined(PETSC_USE_COMPLEX)
  Mat_SeqSELL *a=(Mat_SeqSELL*)A->data;
  PetscInt    i;
  PetscScalar *val = a->val;

  PetscFunctionBegin;
  for (i=0; i<a->sliidx[a->totalslices]; i++) {
    val[i] = PetscConj(val[i]);
  }
#else
  PetscFunctionBegin;
#endif
  PetscFunctionReturn(0);
}
