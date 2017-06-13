
/*
  Defines the basic matrix operations for the ELL matrix storage format.
*/
#include <../src/mat/impls/ell/seq/ell.h>  /*I   "petscmat.h"  I*/
#include <petscblaslapack.h>
#include <petsc/private/kernels/blocktranspose.h>
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__INTEL_COMPILER)
#include <immintrin.h>

/* these do not work
 vec_idx  = _mm512_loadunpackhi_epi32(vec_idx,acolidx);
 vec_vals = _mm512_loadunpackhi_pd(vec_vals,aval);
*/
#define AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y) \
/* if the mast bit is set, copy from acolidx, otherwise from vec_idx */ \
vec_idx  = _mm256_maskz_load_epi32(maskallset,acolidx); \
vec_vals = _mm512_load_pd(aval); \
vec_x    = _mm512_mask_i32gather_pd(vec_x,mask,vec_idx,x,_MM_SCALE_8); \
vec_y    = _mm512_mask3_fmadd_pd(vec_x,vec_vals,vec_y,mask); \

#endif

/*@C
 MatSeqELLSetPreallocation - For good matrix assembly performance
 the user should preallocate the matrix storage by setting the parameter nz
 (or the array nnz).  By setting these parameters accurately, performance
 during matrix assembly can be increased significantly.

 Collective on MPI_Comm

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

 The maximum number of nonzeos in any row should be as accuate as possible.
 If it is underesitmated, you will get bad performance due to reallocation
 (MatSeqXELLReallocateELL).

 Level: intermediate

 .seealso: MatCreate(), MatCreateELL(), MatSetValues(), MatGetInfo()

 @*/
PetscErrorCode  MatSeqELLSetPreallocation(Mat B,PetscInt rlenmax,const PetscInt rlen[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidType(B,1);
  ierr = PetscTryMethod(B,"MatSeqELLSetPreallocation_C",(Mat,PetscInt,const PetscInt[]),(B,rlenmax,rlen));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqELLSetPreallocation_SeqELL(Mat B,PetscInt maxallocrow,const PetscInt rlen[])
{
  Mat_SeqELL     *b;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscBool      skipallocation=PETSC_FALSE,realalloc=PETSC_FALSE;

  PetscFunctionBegin;
  if (maxallocrow >= 0 || rlen) realalloc = PETSC_TRUE;
  if (maxallocrow == MAT_SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    maxallocrow    = 0;
  }

  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 5;
  if (maxallocrow < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"maxallocrow cannot be less than 0: value %D",maxallocrow);
  if (rlen) {
    for (i=0; i<B->rmap->n; i++) {
      if (rlen[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be less than 0: local row %D value %D",i,rlen[i]);
      if (rlen[i] > B->cmap->n) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"rlen cannot be greater than row length: local row %D value %D rowlength %D",i,rlen[i],B->cmap->n);
    }
  }

  B->preallocated = PETSC_TRUE;

  b = (Mat_SeqELL*)B->data;

  if (!skipallocation) {
    if (B->rmap->n & 0x07) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"SEQELL matrix requires the number of rows be the multiple of 8: value %D",B->rmap->n);

    if (!b->rlen) {
      /* sliidx gives the starting index of each slice, the last element is the total space allocated */
      ierr = PetscMalloc1(B->rmap->n,&b->rlen);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,B->rmap->n*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMalloc1(B->rmap->n/8+1,&b->sliidx);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)B,(B->rmap->n/8+1)*sizeof(PetscInt));CHKERRQ(ierr);
    }
    if (!rlen) { /* if rlen is not provided, allocate same space for all the slices */
      if (maxallocrow == PETSC_DEFAULT || maxallocrow == PETSC_DECIDE) maxallocrow = 10;
      else if (maxallocrow < 0) maxallocrow = 1;
      for (i=0; i<=B->rmap->n/8; i++) b->sliidx[i] = i*8*maxallocrow;
    } else {
      maxallocrow = 0;
      b->sliidx[0] = 0;
      for (i=1; i<=B->rmap->n/8; i++) {
        b->sliidx[i] = 0;
        for (j=0;j<8;j++) {
          b->sliidx[i] = PetscMax(b->sliidx[i],rlen[8*(i-1)+j]);
        }
        maxallocrow = PetscMax(b->sliidx[i],maxallocrow);
        b->sliidx[i] = b->sliidx[i-1] + 8*b->sliidx[i];
      }
    }
    /* b->rlen will count nonzeros in each row so far. We dont copy rlen to b->rlen because the matrix has not been set. */
    for (i=0; i<B->rmap->n; i++) b->rlen[i] = 0;

    /* allocate space for val, colidx and bt */
    /* FIXME: should B's old memory be unlogged? */
    ierr = MatSeqXELLFreeELL(B,&b->val,&b->colidx,&b->bt);CHKERRQ(ierr);
    ierr = PetscMalloc3(b->sliidx[B->rmap->n/8],&b->val,b->sliidx[B->rmap->n/8],&b->colidx,b->sliidx[B->rmap->n/8]/8,&b->bt);CHKERRQ(ierr);
    ierr = PetscMemzero(b->bt,b->sliidx[B->rmap->n/8]/8);CHKERRQ(ierr); /* must set to zero */
    ierr = PetscLogObjectMemory((PetscObject)B,b->sliidx[B->rmap->n/8]*(sizeof(PetscScalar)+sizeof(PetscInt))+b->sliidx[B->rmap->n/8]/4);CHKERRQ(ierr);
    b->singlemalloc = PETSC_TRUE;
    b->free_val     = PETSC_TRUE;
    b->free_colidx  = PETSC_TRUE;
    b->free_bt      = PETSC_TRUE;
  } else {
    b->free_val    = PETSC_FALSE;
    b->free_colidx = PETSC_FALSE;
    b->free_bt     = PETSC_FALSE;
  }

  b->nz               = 0;
  b->maxallocrow      = maxallocrow;
  b->rlenmax          = maxallocrow;
  b->maxallocmat      = b->sliidx[B->rmap->n/8];
  B->info.nz_unneeded = (double)b->maxallocmat;
  if (realalloc) {
    ierr = MatSetOption(B,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatConvert_SeqELL_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscInt       m = A->rmap->N,i,j,row;
  PetscBool      bflag;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,a->rlen);CHKERRQ(ierr);
  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);

  for (i=0; i<m/8; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      bflag = a->bt[j>>3] & (char)(1<<row);
      if (bflag) {
        ierr = MatSetValue(B,8*i+row,a->colidx[j],a->val[j],INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  B->rmap->bs = A->rmap->bs;

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/aij.h>

PetscErrorCode MatConvert_SeqAIJ_SeqELL(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat               B;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          *ai=a->i,m=A->rmap->N,n=A->cmap->N,i,*rowlengths,row,ncols;
  const PetscInt    *cols;
  const PetscScalar *vals;

  PetscFunctionBegin;
  if (n != m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Matrix must be square");
  if (A->rmap->bs > 1) {
    ierr = MatConvert_Basic(A,newtype,reuse,newmat);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  /* Can we just use ilen? */
  ierr = PetscMalloc1(m,&rowlengths);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    rowlengths[i] = ai[i+1] - ai[i];
  }

  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQELL);CHKERRQ(ierr);
  ierr = MatSeqELLSetPreallocation(B,0,rowlengths);CHKERRQ(ierr);
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = MatSetOption(B,MAT_ROW_ORIENTED,PETSC_TRUE);CHKERRQ(ierr);

  for (row=0; row<m; row++) {
    ierr = MatGetRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
    ierr = MatSetValues(B,1,&row,ncols,cols,vals,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,row,&ncols,&cols,&vals);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  B->rmap->bs = A->rmap->bs;

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
PetscErrorCode MatMult_SeqELL(Mat A,Vec xx,Vec yy)
{
  Mat_SeqELL        *a=(Mat_SeqELL*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n; /* number of rows */
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__INTEL_COMPILER)
  __m512d           vec_x,vec_y,vec_vals;
  __m256i           vec_idx;
  __mmask8          mask,maskallset=0xff;
  __m512d           vec_x2,vec_y2,vec_vals2,vec_x3,vec_y3,vec_vals3,vec_x4,vec_y4,vec_vals4;
  __m256i           vec_idx2,vec_idx3,vec_idx4;
  __mmask8          mask2,mask3,mask4;
#else
  PetscScalar       sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8;
  char              bflag1,bflag2,bflag3,bflag4,bflag5,bflag6,bflag7,bflag8;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<(m>>3); i++) { /* loop over slices */
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__INTEL_COMPILER)
    PetscPrefetchBlock(acolidx,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);
    PetscPrefetchBlock(aval,a->sliidx[i+1]-a->sliidx[i],0,PETSC_PREFETCH_HINT_T0);

    vec_y  = _mm512_setzero_pd();
    vec_y2 = _mm512_setzero_pd();
    vec_y3 = _mm512_setzero_pd();
    vec_y4 = _mm512_setzero_pd();

    j = a->sliidx[i]>>3; /* index of the bit array; 8 bits are read at each time, corresponding to a slice columnn */
    switch ((a->sliidx[i+1]-a->sliidx[i])/8 & 3) {
    case 3:
      mask  = (__mmask8)a->bt[j];
      mask2 = (__mmask8)a->bt[j+1];
      mask3 = (__mmask8)a->bt[j+2];
      AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask2,vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask3,vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      j += 3;
      break;
    case 2:
      mask  = (__mmask8)a->bt[j];
      mask2 = (__mmask8)a->bt[j+1];
      AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask2,vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      j += 2;
      break;
    case 1:
      mask = (__mmask8)a->bt[j];
      AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      j += 1;
      break;
    }
    #pragma novector
    for (; j<(a->sliidx[i+1]>>3); j+=4) {
      mask  = (__mmask8)a->bt[j];
      mask2 = (__mmask8)a->bt[j+1];
      mask3 = (__mmask8)a->bt[j+2];
      mask4 = (__mmask8)a->bt[j+3];
      /*
      vec_idx  = _mm512_loadunpackhi_epi32(vec_idx,acolidx);
      vec_vals = _mm512_loadunpackhi_pd(vec_vals,aval);
      */
      AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask2,vec_idx2,vec_x2,vec_vals2,vec_y2);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask3,vec_idx3,vec_x3,vec_vals3,vec_y3);
      acolidx += 8; aval += 8;
      AVX512_Mult_Private(mask4,vec_idx4,vec_x4,vec_vals4,vec_y4);
      acolidx += 8; aval += 8;
    }

    vec_y = _mm512_add_pd(vec_y,vec_y2);
    vec_y = _mm512_add_pd(vec_y,vec_y3);
    vec_y = _mm512_add_pd(vec_y,vec_y4);
    _mm512_store_pd(&y[8*i],vec_y);
#else
    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;
    sum4 = 0.0;
    sum5 = 0.0;
    sum6 = 0.0;
    sum7 = 0.0;
    sum8 = 0.0;
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      bflag1 = a->bt[j>>3] & (char)(1);
      bflag2 = a->bt[j>>3] & (char)(1<<1);
      bflag3 = a->bt[j>>3] & (char)(1<<2);
      bflag4 = a->bt[j>>3] & (char)(1<<3);
      bflag5 = a->bt[j>>3] & (char)(1<<4);
      bflag6 = a->bt[j>>3] & (char)(1<<5);
      bflag7 = a->bt[j>>3] & (char)(1<<6);
      bflag8 = a->bt[j>>3] & (char)(1<<7);
      if (bflag1) sum1 += aval[j]*x[acolidx[j]] ;
      if (bflag2) sum2 += aval[j+1]*x[acolidx[j+1]];
      if (bflag3) sum3 += aval[j+2]*x[acolidx[j+2]];
      if (bflag4) sum4 += aval[j+3]*x[acolidx[j+3]];
      if (bflag5) sum5 += aval[j+4]*x[acolidx[j+4]];
      if (bflag6) sum6 += aval[j+5]*x[acolidx[j+5]];
      if (bflag7) sum7 += aval[j+6]*x[acolidx[j+6]];
      if (bflag8) sum8 += aval[j+7]*x[acolidx[j+7]];
    }
    y[8*i]   = sum1;
    y[8*i+1] = sum2;
    y[8*i+2] = sum3;
    y[8*i+3] = sum4;
    y[8*i+4] = sum5;
    y[8*i+5] = sum6;
    y[8*i+6] = sum7;
    y[8*i+7] = sum8;
#endif
  }

  ierr = PetscLogFlops(2.0*a->nz-a->nonzerorowcnt);CHKERRQ(ierr); /* theoretical minimal FLOPs */
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <../src/mat/impls/aij/seq/ftn-kernels/fmultadd.h>
PetscErrorCode MatMultAdd_SeqELL(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqELL        *a=(Mat_SeqELL*)A->data;
  PetscScalar       *y,*z;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscErrorCode    ierr;
  PetscInt          m=A->rmap->n; /* number of rows */
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j;
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__INTEL_COMPILER)
  __m512d           vec_x,vec_y,vec_z,vec_vals;
  __m256i           vec_idx;
  __mmask8          mask,maskallset=0xff;
#else
  PetscScalar       sum1,sum2,sum3,sum4,sum5,sum6,sum7,sum8;
  char              bflag1,bflag2,bflag3,bflag4,bflag5,bflag6,bflag7,bflag8;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  for (i=0; i<(m>>3); i++) { /* loop over slices */
#if defined(PETSC_HAVE_IMMINTRIN_H) && defined(__INTEL_COMPILER)
    vec_y = _mm512_load_pd(&y[8*i]);
    #pragma novector
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      mask = (__mmask8)a->bt[j>>3];
      AVX512_Mult_Private(mask,vec_idx,vec_x,vec_vals,vec_y);
      acolidx += 8; aval += 8;
    }
    _mm512_store_pd(&z[8*i],vec_y);
#else
    sum1 = y[8*i];
    sum2 = y[8*i+1];
    sum3 = y[8*i+2];
    sum4 = y[8*i+3];
    sum5 = y[8*i+4];
    sum6 = y[8*i+5];
    sum7 = y[8*i+6];
    sum8 = y[8*i+7];
    for (j=a->sliidx[i]; j<a->sliidx[i+1]; j+=8) {
      bflag1 = a->bt[j>>3] & (char)(1);
      bflag2 = a->bt[j>>3] & (char)(1<<1);
      bflag3 = a->bt[j>>3] & (char)(1<<2);
      bflag4 = a->bt[j>>3] & (char)(1<<3);
      bflag5 = a->bt[j>>3] & (char)(1<<4);
      bflag6 = a->bt[j>>3] & (char)(1<<5);
      bflag7 = a->bt[j>>3] & (char)(1<<6);
      bflag8 = a->bt[j>>3] & (char)(1<<7);
      if (bflag1) sum1 += aval[j]*x[acolidx[j]];
      if (bflag2) sum2 += aval[j+1]*x[acolidx[j+1]];
      if (bflag3) sum3 += aval[j+2]*x[acolidx[j+2]];
      if (bflag4) sum4 += aval[j+3]*x[acolidx[j+3]];
      if (bflag5) sum5 += aval[j+4]*x[acolidx[j+4]];
      if (bflag6) sum6 += aval[j+5]*x[acolidx[j+5]];
      if (bflag7) sum7 += aval[j+6]*x[acolidx[j+6]];
      if (bflag8) sum8 += aval[j+7]*x[acolidx[j+7]];
    }
    z[8*i]   = sum1;
    z[8*i+1] = sum2;
    z[8*i+2] = sum3;
    z[8*i+3] = sum4;
    z[8*i+4] = sum5;
    z[8*i+5] = sum6;
    z[8*i+6] = sum7;
    z[8*i+7] = sum8;
#endif
  }

  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayPair(yy,zz,&y,&z);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTransposeAdd_SeqELL(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqELL        *a=(Mat_SeqELL*)A->data;
  PetscScalar       *y;
  const PetscScalar *x;
  const MatScalar   *aval=a->val;
  PetscInt          m=A->rmap->n; /* number of rows */
  const PetscInt    *acolidx=a->colidx;
  PetscInt          i,j,row;
  char              bflag;
  PetscErrorCode    ierr;

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aval)
#endif

  PetscFunctionBegin;
  if (zz != yy) { ierr = VecCopy(zz,yy);CHKERRQ(ierr); }
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  for (i=0; i<m/8; i++) { /* loop over slices */
    for (j=a->sliidx[i],row=0; j<a->sliidx[i+1]; j++,row=((row+1)&0x07)) {
      bflag = a->bt[j>>3] & (char)(1<<row);
      if (bflag) y[acolidx[j]] += aval[j]*x[8*i+row];
    }
  }
  ierr = PetscLogFlops(2.0*a->sliidx[m/8]);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_SeqELL(Mat A,Vec xx,Vec yy)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(yy,0.0);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd_SeqELL(A,xx,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     Checks for missing diagonals
*/
PetscErrorCode MatMissingDiagonal_SeqELL(Mat A,PetscBool  *missing,PetscInt *d)
{
  Mat_SeqELL *a = (Mat_SeqELL*)A->data;
  PetscInt   *diag,i;

  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  if (A->rmap->n > 0 && !(a->colidx)) {
    *missing = PETSC_TRUE;
    if (d) *d = 0;
    PetscInfo(A,"Matrix has no entries therefore is missing diagonal\n");
  } else {
    diag = a->diag;
    for (i=0; i<A->rmap->n; i++) {
      if (diag[i] == -1) {
        *missing = PETSC_TRUE;
        if (d) *d = i;
        PetscInfo1(A,"Matrix is missing diagonal number %D\n",i);
        break;
      }
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMarkDiagonal_SeqELL(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m=A->rmap->n,shift;

  PetscFunctionBegin;
  if (!a->diag) {
    ierr         = PetscMalloc1(m,&a->diag);CHKERRQ(ierr);
    ierr         = PetscLogObjectMemory((PetscObject)A,m*sizeof(PetscInt));CHKERRQ(ierr);
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
PetscErrorCode  MatInvertDiagonal_SeqELL(Mat A,PetscScalar omega,PetscScalar fshift)
{
  Mat_SeqELL     *a = (Mat_SeqELL*) A->data;
  PetscErrorCode ierr;
  PetscInt       i,*diag,m = A->rmap->n;
  MatScalar      *val = a->val;
  PetscScalar    *idiag,*mdiag;

  PetscFunctionBegin;
  if (a->idiagvalid) PetscFunctionReturn(0);
  ierr = MatMarkDiagonal_SeqELL(A);CHKERRQ(ierr);
  diag = a->diag;
  if (!a->idiag) {
    ierr = PetscMalloc3(m,&a->idiag,m,&a->mdiag,m,&a->ssor_work);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A, 3*m*sizeof(PetscScalar));CHKERRQ(ierr);
    val  = a->val;
  }
  mdiag = a->mdiag;
  idiag = a->idiag;

  if (omega == 1.0 && PetscRealPart(fshift) <= 0.0) {
    for (i=0; i<m; i++) {
      mdiag[i] = val[diag[i]];
      if (!PetscAbsScalar(mdiag[i])) { /* zero diagonal */
        if (PetscRealPart(fshift)) {
          ierr = PetscInfo1(A,"Zero diagonal on row %D\n",i);CHKERRQ(ierr);
          A->factorerrortype             = MAT_FACTOR_NUMERIC_ZEROPIVOT;
          A->factorerror_zeropivot_value = 0.0;
          A->factorerror_zeropivot_row   = i;
        } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Zero diagonal on row %D",i);
      }
      idiag[i] = 1.0/val[diag[i]];
    }
    ierr = PetscLogFlops(m);CHKERRQ(ierr);
  } else {
    for (i=0; i<m; i++) {
      mdiag[i] = val[diag[i]];
      idiag[i] = omega/(fshift + val[diag[i]]);
    }
    ierr = PetscLogFlops(2.0*m);CHKERRQ(ierr);
  }
  a->idiagvalid = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_SeqELL(Mat A)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(a->val,(a->sliidx[A->rmap->n/8])*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SeqELL(Mat A)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D",A->rmap->n,A->cmap->n,a->nz);
#endif
  ierr = MatSeqXELLFreeELL(A,&a->val,&a->colidx,&a->bt);CHKERRQ(ierr);
  ierr = ISDestroy(&a->row);CHKERRQ(ierr);
  ierr = ISDestroy(&a->col);CHKERRQ(ierr);
  ierr = PetscFree(a->diag);CHKERRQ(ierr);
  ierr = PetscFree(a->ibdiag);CHKERRQ(ierr);
  ierr = PetscFree(a->rlen);CHKERRQ(ierr);
  ierr = PetscFree(a->sliidx);CHKERRQ(ierr);
  ierr = PetscFree3(a->idiag,a->mdiag,a->ssor_work);CHKERRQ(ierr);
  ierr = PetscFree(a->solve_work);CHKERRQ(ierr);
  ierr = ISDestroy(&a->icol);CHKERRQ(ierr);
  ierr = PetscFree(a->saved_values);CHKERRQ(ierr);

  ierr = PetscFree(A->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)A,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatStoreValues_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatRetrieveValues_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
#endif
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSeqELLSetPreallocation_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetOption_SeqELL(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqELL     *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

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
  case MAT_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    ierr = PetscInfo1(A,"Option %s ignored\n",MatOptions[op]);CHKERRQ(ierr);
    break;
  case MAT_SPD:
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
    /* These options are handled directly by MatSetOption() */
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %d",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetDiagonal_SeqELL(Mat A,Vec v)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       i,j,n,shift;
  PetscScalar    *x,zero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");

  if (A->factortype == MAT_FACTOR_ILU || A->factortype == MAT_FACTOR_LU) {
    PetscInt *diag=a->diag;
    ierr = VecGetArray(v,&x);CHKERRQ(ierr);
    for (i=0; i<n; i++) x[i] = 1.0/a->val[diag[i]];
    ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
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
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatSetValues_SeqELL(Mat,PetscInt,const PetscInt[],PetscInt,const PetscInt[],const PetscScalar[],InsertMode);

PetscErrorCode MatGetValues_SeqELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqELL  *a = (Mat_SeqELL*)A->data;
  PetscInt    *cp,i,k,low,high,t,row,col,l;
  PetscInt    shift;
  MatScalar   *vp;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row<0) continue;
#if defined(PETSC_USE_DEBUG)
    if (row >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->n-1);
#endif
    shift = a->sliidx[row>>3]+(row&0x07); /* starting index of the row */
    cp = a->colidx+shift; /* pointer to the row */
    vp = a->val+shift; /* pointer to the row */
    for (l=0; l<n; l++) { /* loop over added rows */
      col = in[l];
      if (col<0) continue;
#if defined(PETSC_USE_DEBUG)
      if (col >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: row %D max %D",col,A->cmap->n-1);
#endif
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

PetscErrorCode MatView_SeqELL_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqELL        *a = (Mat_SeqELL*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,shift;
  const char        *name;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscInt nofinalvalue = 0;
    /*
    if (m && ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != A->cmap->n-1))) {
      nofinalvalue = 1;
    }
    */
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",m,A->cmap->n);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %D \n",a->nz);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,4);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,a->colidx[shift+8*j]+1,(double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
      }
    }
    /*
    if (nofinalvalue) {
#if defined(PETSC_USE_COMPLEX)
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e %18.16e\n",m,A->cmap->n,0.,0.);CHKERRQ(ierr);
#else
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",m,A->cmap->n,0.0);CHKERRQ(ierr);
#endif
    }
    */
    ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO || format == PETSC_VIEWER_ASCII_INFO) {
    PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0 && PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
        } else if (PetscRealPart(a->val[shift+8*j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]));CHKERRQ(ierr);
        }
#else
        if (a->val[shift*8*j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]);CHKERRQ(ierr);}
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_DENSE) {
    PetscInt    cnt = 0,jcnt;
    PetscScalar value;
#if defined(PETSC_USE_COMPLEX)
    PetscBool   realonly = PETSC_TRUE;

    for (i=0; i<a->sliidx[A->rmap->n/8]; i++) {
      if (PetscImaginaryPart(a->val[i]) != 0.0) {
        realonly = PETSC_FALSE;
        break;
      }
    }
#endif

    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
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
          ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",(double)PetscRealPart(value));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," %7.5e+%7.5e i ",(double)PetscRealPart(value),(double)PetscImaginaryPart(value));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",(double)value);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATRIXMARKET) {
    PetscInt fshift=1;
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    ierr = PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate complex general\n");CHKERRQ(ierr);
#else
    ierr = PetscViewerASCIIPrintf(viewer,"%%%%MatrixMarket matrix coordinate real general\n");CHKERRQ(ierr);
#endif
    ierr = PetscViewerASCIIPrintf(viewer,"%D %D %D\n", m, A->cmap->n, a->nz);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D %g %g\n",i+fshift,a->colidx[shift+8*j]+fshift,(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D %g\n",i+fshift,a->colidx[shift+8*j]+fshift,(double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    if (A->factortype) {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07);
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        /* L part */
        for (j=shift; j<a->diag[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[shift+8*j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->val[shift+8*j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j])));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(a->val[j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)a->val[j]);CHKERRQ(ierr);
#endif
        }
        /* diagonal */
        j = a->diag[i];
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->val[j]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)PetscImaginaryPart(1.0/a->val[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]),(double)(-PetscImaginaryPart(1.0/a->val[j])));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(1.0/a->val[j]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)(1.0/a->val[j]));CHKERRQ(ierr);
#endif

        /* U part */
        for (j=a->diag[i]+1; j<shift+8*a->rlen[i]; j+=8) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)PetscImaginaryPart(a->val[j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->val[j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[j],(double)PetscRealPart(a->val[j]),(double)(-PetscImaginaryPart(a->val[j])));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)PetscRealPart(a->val[j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[j],(double)a->val[j]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } else {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07);
        ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
        for (j=0; j<a->rlen[i]; j++) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->val[j]) > 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
          } else if (PetscImaginaryPart(a->[j]) < 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]),(double)-PetscImaginaryPart(a->val[shift+8*j]));CHKERRQ(ierr);
          } else {
            ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)PetscRealPart(a->val[shift+8*j]));CHKERRQ(ierr);
          }
#else
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->colidx[shift+8*j],(double)a->val[shift+8*j]);CHKERRQ(ierr);
#endif
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode MatView_SeqELL_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A  = (Mat) Aa;
  Mat_SeqELL        *a = (Mat_SeqELL*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,shift;
  int               color;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) >=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    color = PETSC_DRAW_CYAN;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (a->val[shift+8*j] !=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    color = PETSC_DRAW_RED;
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8]; x_r = x_l + 1.0;
        if (PetscRealPart(a->val[shift+8*j]) <=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscInt  count = 0;
    PetscDraw popup;

    for (i=0; i<a->sliidx[A->rmap->n/8]; i++) {
      if (PetscAbsScalar(a->val[i]) > maxv) maxv = PetscAbsScalar(a->val[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup,minv,maxv);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      shift = a->sliidx[i>>3]+(i&0x07);
      y_l = m - i - 1.0;
      y_r = y_l + 1.0;
      for (j=0; j<a->rlen[i]; j++) {
        x_l = a->colidx[shift+j*8];
        x_r = x_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(a->val[count]),minv,maxv);
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
        count++;
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
PetscErrorCode MatView_SeqELL_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscBool      isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr  = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;         xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqELL_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqELL(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);
  if (iascii) {
    ierr = MatView_SeqELL_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    /* ierr = MatView_SeqELL_Binary(A,viewer);CHKERRQ(ierr); */
  } else if (isdraw) {
    ierr = MatView_SeqELL_Draw(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_SeqELL(Mat A,MatAssemblyType mode)
{
  Mat_SeqELL  *a = (Mat_SeqELL*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);
  /* To do: compress out the unused elements */

  ierr = MatMarkDiagonal_SeqELL(A);CHKERRQ(ierr);
  ierr = PetscInfo6(A,"Matrix size: %D X %D; storage space: %D allocated %D used (%D nonzeros+%D paddedzeros)\n",A->rmap->n,A->cmap->n,a->maxallocmat,a->sliidx[A->rmap->n/8],a->nz,a->sliidx[A->rmap->n/8]-a->nz);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Number of mallocs during MatSetValues() is %D\n",a->reallocs);CHKERRQ(ierr);
  ierr = PetscInfo1(A,"Maximum nonzeros in any row is %D\n",a->rlenmax);CHKERRQ(ierr);

  A->info.mallocs    += a->reallocs;
  a->reallocs         = 0;

  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatGetInfo_SeqELL(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqELL *a = (Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  info->block_size   = 1.0;
  info->nz_allocated = (double)a->maxallocmat;
  info->nz_used      = (double)a->sliidx[A->rmap->n/8]; /* include padding zeros */
  info->nz_unneeded  = (double)(a->maxallocmat-a->sliidx[A->rmap->n/8]);
  info->assemblies   = (double)A->num_ass;
  info->mallocs      = (double)A->info.mallocs;
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

PetscErrorCode MatSetValues_SeqELL(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       shift,i,k,l,low,high,t,ii,row,col,nrow;
  PetscInt       *cp,nonew=a->nonew,lastcol=-1;
  MatScalar      *vp,value;
  char           *bp;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row = im[k];
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)
    if (row >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap->n-1);
#endif
    shift = a->sliidx[row>>3]+(row&0x07); /* starting index of the row */
    cp    = a->colidx+shift; /* pointer to the row */
    vp    = a->val+shift; /* pointer to the row */
    bp    = a->bt+shift/8; /* pointer to the row */
    nrow  = a->rlen[row];
    low   = 0;
    high  = nrow;

    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l];
      if (col<0) continue;
#if defined(PETSC_USE_DEBUG)
      if (col >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col too large: row %D max %D",col,A->cmap->n-1);
#endif
      if (a->roworiented) {
        value = v[l+k*n];
      } else {
        value = v[k+l*m];
      }
      if ((value == 0.0 && a->ignorezeroentries) && (is == ADD_VALUES)) continue;

      /* search in this row for the specified colmun, i indicates the column to be set */
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
      if (nonew == -1) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero (%D, %D) in the matrix", row, col);
      /* If the current row length exceeds the slice width (e.g. nrow==slice_width), allocate a new space, otherwise do nothing */
      MatSeqXELLReallocateELL(A,A->rmap->n,1,nrow,a->sliidx,row/8,row,col,a->colidx,a->val,a->bt,cp,vp,bp,nonew,MatScalar);
      /* add the new nonzero to the high position, shift the remaining elements in current row to the right by one slot */
      for (ii=nrow-1; ii>=i; ii--) {
        *(cp+(ii+1)*8) = *(cp+ii*8);
        *(vp+(ii+1)*8) = *(vp+ii*8);
        if (*(bp+ii) & (char)1<<(row&0x07)) *(bp+ii+1) |= (char)1<<(row&0x07);
      }
      a->rlen[row]++;
      *(cp+i*8) = col;
      *(vp+i*8) = value;
      *(bp+i)  |= (char)1<<(row&0x07); /* set the mask bit */
      a->nz++;
      A->nonzerostate++;
      low = i+1; high++; nrow++;
noinsert:;
    }
    a->rlen[row] = nrow;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCopy_SeqELL(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqELL *a = (Mat_SeqELL*)A->data;
    Mat_SeqELL *b = (Mat_SeqELL*)B->data;

    if (a->sliidx[A->rmap->n/8] != b->sliidx[B->rmap->n/8]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    ierr = PetscMemcpy(b->val,a->val,a->sliidx[A->rmap->n/8]*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetUp_SeqELL(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSeqELLSetPreallocation(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqELLGetArray_SeqELL(Mat A,PetscScalar *array[])
{
  Mat_SeqELL *a=(Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  *array = a->val;
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqELLRestoreArray_SeqELL(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatRealPart_SeqELL(Mat A)
{
  Mat_SeqELL  *a=(Mat_SeqELL*)A->data;
  PetscInt    i,matsize=a->sliidx[A->rmap->n/8];
  MatScalar   *aval=a->val;

  PetscFunctionBegin;
  for (i=0; i<matsize; i++) aval[i]=PetscRealPart(aval[i]);
  PetscFunctionReturn(0);
}

PetscErrorCode MatImaginaryPart_SeqELL(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;
  PetscInt       i,matsize=a->sliidx[A->rmap->n/8];
  MatScalar      *aval=a->val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (i=0; i<matsize; i++) aval[i] = PetscImaginaryPart(aval[i]);
  ierr = MatSeqELLInvalidateDiagonal(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatShift_SeqELL(Mat Y,PetscScalar a)
{
  PetscErrorCode ierr;
  Mat_SeqELL     *y=(Mat_SeqELL*)Y->data;

  PetscFunctionBegin;
  if (!Y->preallocated || !y->nz) {
    ierr = MatSeqELLSetPreallocation(Y,1,NULL);CHKERRQ(ierr);
  }
  ierr = MatShift_Basic(Y,a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatSOR_SeqELL(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqELL        *a = (Mat_SeqELL*)A->data;
  PetscScalar       *x,sum,*t;
  const MatScalar   *idiag=0,*mdiag;
  const PetscScalar *b,*xb;
  PetscErrorCode    ierr;
  PetscInt          n,m = A->rmap->n,i,j,shift;
  const PetscInt    *diag;

  PetscFunctionBegin;
  its = its*lits;

  if (fshift != a->fshift || omega != a->omega) a->idiagvalid = PETSC_FALSE; /* must recompute idiag[] */
  if (!a->idiagvalid) {ierr = MatInvertDiagonal_SeqELL(A,omega,fshift);CHKERRQ(ierr);}
  a->fshift = fshift;
  a->omega  = omega;

  diag  = a->diag;
  t     = a->ssor_work;
  idiag = a->idiag;
  mdiag = a->mdiag;

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  /* We count flops by assuming the upper triangular and lower triangular parts have the same number of nonzeros */
  if (flag == SOR_APPLY_UPPER) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SOR_APPLY_UPPER is not implemented");
  if (flag == SOR_APPLY_LOWER) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SOR_APPLY_LOWER is not implemented");
  if (flag & SOR_EISENSTAT) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No support yet for Eisenstat");

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        shift = a->sliidx[i>>3]+(i&0x07); /* starting index of the row i */
        sum   = b[i];
        n     = (diag[i]-shift)/8;
        for (j=0; j<n; j++) sum -= a->val[shift+j*8]*x[a->colidx[shift+j*8]];
        t[i]  = sum;
        x[i]  = sum*idiag[i];
      }
      xb   = t;
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr);
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
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
      ierr = PetscLogFlops(a->nz);CHKERRQ(ierr); /* assumes 1/2 in upper */
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
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
      ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
    } else xb = b;
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
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
        ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);
      } else {
        ierr = PetscLogFlops(a->nz);CHKERRQ(ierr); /* assumes 1/2 in upper */
      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqELL,
                                       0,
                                       0,
                                       MatMult_SeqELL,
                               /* 4*/  MatMultAdd_SeqELL,
                                       MatMultTranspose_SeqELL,
                                       MatMultTransposeAdd_SeqELL,
                                       0,
                                       0,
                                       0,
                               /* 10*/ 0,
                                       0,
                                       0,
                                       MatSOR_SeqELL,
                                       0,
                               /* 15*/ MatGetInfo_SeqELL,
                                       MatEqual_SeqELL,
                                       MatGetDiagonal_SeqELL,
                                       0,
                                       0,
                               /* 20*/ 0,
                                       MatAssemblyEnd_SeqELL,
                                       MatSetOption_SeqELL,
                                       MatZeroEntries_SeqELL,
                               /* 24*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 29*/ MatSetUp_SeqELL,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 34*/ MatDuplicate_SeqELL,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 39*/ 0,
                                       0,
                                       0,
                                       MatGetValues_SeqELL,
                                       MatCopy_SeqELL,
                               /* 44*/ 0,
                                       0,
                                       MatShift_SeqELL,
                                       0,
                                       0,
                               /* 49*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 54*/ MatFDColoringCreate_SeqXAIJ,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 59*/ 0,
                                       MatDestroy_SeqELL,
                                       MatView_SeqELL,
                                       0,
                                       0,
                               /* 64*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 69*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 74*/ 0,
                                       MatFDColoringApply_AIJ, /* reuse the FDColoring function for AIJ */
                                       0,
                                       0,
                                       0,
                               /* 79*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 84*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 89*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 94*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /* 99*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*104*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*109*/ 0,
                                       0,
                                       0,
                                       0,
                                       MatMissingDiagonal_SeqELL,
                               /*114*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*119*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*124*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*129*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*134*/ 0,
                                       0,
                                       0,
                                       0,
                                       0,
                               /*139*/ 0,
                                       0,
                                       0,
                                       MatFDColoringSetUp_SeqXAIJ,
                                       0,
                                /*144*/0
};

PetscErrorCode  MatStoreValues_SeqELL(Mat mat)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)mat->data;
  PetscErrorCode ierr;
  size_t         matsize=a->sliidx[mat->rmap->n/8];

  PetscFunctionBegin;
  if (!a->nonew) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");

  /* allocate space for values if not already there */
  if (!a->saved_values) {
    ierr = PetscMalloc1(matsize+1,&a->saved_values);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)mat,(matsize+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscMemcpy(a->saved_values,a->val,matsize*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatRetrieveValues_SeqELL(Mat mat)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)mat->data;
  PetscErrorCode ierr;
  PetscInt       matsize=a->sliidx[mat->rmap->n/8];

  PetscFunctionBegin;
  if (!a->nonew) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NEW_NONZERO_LOCATIONS,PETSC_FALSE);first");
  if (!a->saved_values) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
  /* copy values over */
  ierr = PetscMemcpy(a->val,a->saved_values,matsize*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 MatSeqELLRestoreArray - returns access to the array where the data for a MATSEQELL matrix is stored obtained by MatSeqELLGetArray()

 Not Collective

 Input Parameters:
 .  mat - a MATSEQELL matrix
 .  array - pointer to the data

 Level: intermediate

 .seealso: MatSeqELLGetArray(), MatSeqELLRestoreArrayF90()
 @*/
PetscErrorCode  MatSeqELLRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatSeqELLRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_SeqELL(Mat B)
{
  Mat_SeqELL     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Comm must be of size 1");

  ierr = PetscNewLog(B,&b);CHKERRQ(ierr);

  B->data = (void*)b;

  ierr = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);

  b->row                = 0;
  b->col                = 0;
  b->icol               = 0;
  b->reallocs           = 0;
  b->ignorezeroentries  = PETSC_FALSE;
  b->roworiented        = PETSC_TRUE;
  b->nonew              = 0;
  b->diag               = 0;
  b->solve_work         = 0;
  B->spptr              = 0;
  b->saved_values       = 0;
  b->idiag              = 0;
  b->mdiag              = 0;
  b->ssor_work          = 0;
  b->omega              = 1.0;
  b->fshift             = 0.0;
  b->idiagvalid         = PETSC_FALSE;
  b->ibdiagvalid        = PETSC_FALSE;
  b->keepnonzeropattern = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLGetArray_C",MatSeqELLGetArray_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLRestoreArray_C",MatSeqELLRestoreArray_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatStoreValues_C",MatStoreValues_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatRetrieveValues_C",MatRetrieveValues_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqELLSetPreallocation_C",MatSeqELLSetPreallocation_SeqELL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqell_seqaij_C",MatConvert_SeqELL_SeqAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 Given a matrix generated with MatGetFactor() duplicates all the information in A into B
 */
PetscErrorCode MatDuplicateNoCreate_SeqELL(Mat C,Mat A,MatDuplicateOption cpvalues,PetscBool mallocmatspace)
{
  Mat_SeqELL     *c,*a=(Mat_SeqELL*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m=A->rmap->n;

  PetscFunctionBegin;
  c = (Mat_SeqELL*)C->data;

  C->factortype = A->factortype;
  c->row        = 0;
  c->col        = 0;
  c->icol       = 0;
  c->reallocs   = 0;

  C->assembled = PETSC_TRUE;

  ierr = PetscLayoutReference(A->rmap,&C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&C->cmap);CHKERRQ(ierr);

  ierr = PetscMalloc1(m,&c->rlen);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)C, m*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc1(m/8+1,&c->sliidx);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)C, (m/8+1)*sizeof(PetscInt));CHKERRQ(ierr);

  for (i=0; i<m; i++) c->rlen[i] = a->rlen[i];
  for (i=0; i<m/8+1; i++) c->sliidx[i] = a->sliidx[i];

  /* allocate the matrix space */
  if (mallocmatspace) {
    ierr = PetscMalloc3(a->maxallocmat,&c->val,a->maxallocmat,&c->colidx,a->maxallocmat/8,&c->bt);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,a->maxallocmat*(sizeof(PetscScalar)+sizeof(PetscInt))+a->maxallocmat/8*sizeof(char));CHKERRQ(ierr);

    c->singlemalloc = PETSC_TRUE;

    if (m > 0) {
      ierr = PetscMemcpy(c->colidx,a->colidx,(a->maxallocmat)*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(c->bt,a->bt,a->maxallocmat/8*sizeof(char));CHKERRQ(ierr);
      if (cpvalues == MAT_COPY_VALUES) {
        ierr = PetscMemcpy(c->val,a->val,a->maxallocmat*sizeof(PetscScalar));CHKERRQ(ierr);
      } else {
        ierr = PetscMemzero(c->val,a->maxallocmat*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    }
  }

  c->ignorezeroentries = a->ignorezeroentries;
  c->roworiented       = a->roworiented;
  c->nonew             = a->nonew;
  if (a->diag) {
    ierr = PetscMalloc1(m,&c->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)C,m*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag = 0;

  c->solve_work         = 0;
  c->saved_values       = 0;
  c->idiag              = 0;
  c->ssor_work          = 0;
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

  ierr = PetscFunctionListDuplicate(((PetscObject)A)->qlist,&((PetscObject)C)->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatDuplicate_SeqELL(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  if (!(A->rmap->n % A->rmap->bs) && !(A->cmap->n % A->cmap->bs)) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
  }
  ierr = MatSetType(*B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqELL(*B,A,cpvalues,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
 MatCreateSeqELL - Creates a sparse matrix in ELL format.

 Collective on MPI_Comm

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
 MatXXXXSetPreallocation() paradgm instead of this routine directly.
 [MatXXXXSetPreallocation() is, for example, MatSeqELLSetPreallocation]

 Notes:
 If nnz is given then nz is ignored

 Specify the preallocated storage with either rlenmax or rlen (not both).
 Set rlenmax=PETSC_DEFAULT and rlen=NULL for PETSc to control dynamic memory
 allocation.  For large problems you MUST preallocate memory or you
 will get TERRIBLE performance, see the users' manual chapter on matrices.

 Level: intermediate

 .seealso: MatCreate(), MatCreateELL(), MatSetValues(), MatCreateSeqELLWithArrays()

 @*/
PetscErrorCode  MatCreateSeqELL(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt maxallocrow,const PetscInt rlen[],Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQELL);CHKERRQ(ierr);
  ierr = MatSeqELLSetPreallocation_SeqELL(*A,maxallocrow,rlen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatEqual_SeqELL(Mat A,Mat B,PetscBool * flg)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data,*b=(Mat_SeqELL*)B->data;
  PetscInt       matsize;
  PetscErrorCode ierr;
#if defined(PETSC_USE_COMPLEX)
  PetscInt k;
#endif

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->rmap->n != B->rmap->n) || (A->cmap->n != B->cmap->n) ||(a->nz != b->nz) || (a->rlenmax != b->rlenmax)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  matsize = a->sliidx[A->rmap->n/8];
  /* if the a->bt are the same */
  ierr = PetscMemcmp(a->bt,b->bt,matsize/8*sizeof(char),flg);CHKERRQ(ierr);
  if (!*flg) PetscFunctionReturn(0);
  /* if the a->colidx are the same */
  ierr = PetscMemcmp(a->colidx,b->colidx,(matsize)*sizeof(PetscInt),flg);CHKERRQ(ierr);
  if (!*flg) PetscFunctionReturn(0);
  /* if a->val are the same */
#if defined(PETSC_USE_COMPLEX)
  for (k=0; k<matsize; k++) {
    if (PetscRealPart(a->val[k]) != PetscRealPart(b->val[k]) || PetscImaginaryPart(a->val[k]) != PetscImaginaryPart(b->val[k])) {
      *flg = PETSC_FALSE;
      PetscFunctionReturn(0);
    }
  }
#else
  ierr = PetscMemcmp(a->val,b->val,(matsize)*sizeof(PetscScalar),flg);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatSeqELLInvalidateDiagonal(Mat A)
{
  Mat_SeqELL     *a=(Mat_SeqELL*)A->data;

  PetscFunctionBegin;
  a->idiagvalid  = PETSC_FALSE;
  a->ibdiagvalid = PETSC_FALSE;
  PetscFunctionReturn(0);
}
