
/*
    Factorization code for BAIJ format.
*/
#include <../src/mat/impls/baij/seq/baij.h>
#include <../src/mat/blockinvert.h>

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetNumericFactorization"
/*
   This is used to set the numeric factorization for both LU and ILU symbolic factorization
*/
PetscErrorCode MatSeqBAIJSetNumericFactorization(Mat fact,PetscBool  natural)
{
  PetscFunctionBegin;
  if (natural){
    switch (fact->rmap->bs){
    case 1:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;
      break;
    case 2:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering;
      break;
    case 3:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering;
      break;
    case 4:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering;
      break;
    case 5:
       fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering;
       break;
    case 6:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering;
      break;
    case 7:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering;
      break;
    case 15:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_15_NaturalOrdering;
      break;
    default:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;
      break;
    }
  } else {
    switch (fact->rmap->bs){
    case 1:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;
      break;
    case 2:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;
      break;
    case 3:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;
      break;
    case 4:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;
      break;
    case 5:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;
      break;
    case 6:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6;
      break;
    case 7:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7;
      break;
    default:
      fact->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;
      break;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSeqBAIJSetNumericFactorization_inplace"
PetscErrorCode MatSeqBAIJSetNumericFactorization_inplace(Mat inA,PetscBool  natural)
{
  PetscFunctionBegin;
  if (natural) {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1_inplace;
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_NaturalOrdering_inplace;
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_NaturalOrdering_inplace;
      break;
    case 4:
#if defined(PETSC_USE_REAL_MAT_SINGLE)
      {
        PetscBool   sse_enabled_local;
        PetscErrorCode ierr;
        ierr = PetscSSEIsEnabled(inA->comm,&sse_enabled_local,PETSC_NULL);CHKERRQ(ierr);
        if (sse_enabled_local) {
#  if defined(PETSC_HAVE_SSE)
          int i,*AJ=a->j,nz=a->nz,n=a->mbs;
          if (n==(unsigned short)n) {
            unsigned short *aj=(unsigned short *)AJ;
            for (i=0;i<nz;i++) {
              aj[i] = (unsigned short)AJ[i];
            }
            inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE_usj;
            inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE_usj;
            ierr = PetscInfo(inA,"Using special SSE, in-place natural ordering, ushort j index factor BS=4\n");CHKERRQ(ierr);
          } else {
        /* Scale the column indices for easier indexing in MatSolve. */
/*            for (i=0;i<nz;i++) { */
/*              AJ[i] = AJ[i]*4; */
/*            } */
            inA->ops->setunfactored   = MatSetUnfactored_SeqBAIJ_4_NaturalOrdering_SSE;
            inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_SSE;
            ierr = PetscInfo(inA,"Using special SSE, in-place natural ordering, int j index factor BS=4\n");CHKERRQ(ierr);
          }
#  else
        /* This should never be reached.  If so, problem in PetscSSEIsEnabled. */
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"SSE Hardware unavailable");
#  endif
        } else {
          inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace;
        }
      }
#else
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_NaturalOrdering_inplace;
#endif
      break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_NaturalOrdering_inplace;
      break;
    case 6:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_NaturalOrdering_inplace;
      break;
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_NaturalOrdering_inplace;
      break;
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N_inplace;
      break;
    }
  } else {
    switch (inA->rmap->bs) {
    case 1:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1_inplace;
      break;
    case 2:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2_inplace;
      break;
    case 3:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3_inplace;
      break;
    case 4:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4_inplace;
      break;
    case 5:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5_inplace;
      break;
    case 6:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6_inplace;
      break;
    case 7:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7_inplace;
      break;
    default:
      inA->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N_inplace;
      break;
    }
  }
  PetscFunctionReturn(0);
}

/*
    The symbolic factorization code is identical to that for AIJ format,
  except for very small changes since this is now a SeqBAIJ datastructure.
  NOT good code reuse.
*/
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_SeqBAIJ"
PetscErrorCode MatLUFactorSymbolic_SeqBAIJ(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data,*b;
  PetscInt           n=a->mbs,bs = A->rmap->bs,bs2=a->bs2;
  PetscBool          row_identity,col_identity,both_identity;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           i,*ai=a->i,*aj=a->j;
  PetscInt           *bi,*bj,*ajtmp;
  PetscInt           *bdiag,row,nnz,nzi,reallocs=0,nzbd,*im;
  PetscReal          f;
  PetscInt           nlnk,*lnk,k,**bi_ptr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix must be square");
  if (bs>1){  /* check shifttype */
    if (info->shifttype == MAT_SHIFT_NONZERO || info->shifttype == MAT_SHIFT_POSITIVE_DEFINITE)
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only MAT_SHIFT_NONE and MAT_SHIFT_INBLOCKS are supported for BAIJ matrix");
  }

  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);
  bi[0] = bdiag[0] = 0;

  /* linked list for storing column indices of the active row */
  nlnk = n + 1;
  ierr = PetscLLCreate(n,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  ierr = PetscMalloc2(n+1,PetscInt**,&bi_ptr,n+1,PetscInt,&im);CHKERRQ(ierr);

  /* initial FreeSpace size is f*(ai[n]+1) */
  f = info->fill;
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (i=0; i<n; i++) {
    /* copy previous fill into linked list */
    nzi = 0;
    nnz = ai[r[i]+1] - ai[r[i]];
    if (!nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    ajtmp = aj + ai[r[i]];
    ierr = PetscLLAddPerm(nnz,ajtmp,ic,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzi += nlnk;

    /* add pivot rows into linked list */
    row = lnk[n];
    while (row < i) {
      nzbd    = bdiag[row] + 1; /* num of entries in the row with column index <= row */
      ajtmp   = bi_ptr[row] + nzbd; /* points to the entry next to the diagonal */
      ierr = PetscLLAddSortedLU(ajtmp,row,nlnk,lnk,lnkbt,i,nzbd,im);CHKERRQ(ierr);
      nzi += nlnk;
      row  = lnk[row];
    }
    bi[i+1] = bi[i] + nzi;
    im[i]   = nzi;

    /* mark bdiag */
    nzbd = 0;
    nnz  = nzi;
    k    = lnk[n];
    while (nnz-- && k < i){
      nzbd++;
      k = lnk[k];
    }
    bdiag[i] = nzbd; /* note : bdaig[i] = nnzL as input for PetscFreeSpaceContiguous_LU() */

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzi) {
      nnz = 2*(n - i)*nzi; /* estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(nnz,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(n,n,nzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    bi_ptr[i] = current_space->array;
    current_space->array           += nzi;
    current_space->local_used      += nzi;
    current_space->local_remaining -= nzi;
  }

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  /* copy free_space into bj and free free_space; set bi, bj, bdiag in new datastructure; */
  ierr = PetscMalloc((bi[n]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous_LU(&free_space,bj,n,bi,bdiag);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFree2(bi_ptr,im);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(B,bs,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,isicol);CHKERRQ(ierr);
  b    = (Mat_SeqBAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr          = PetscMalloc((bdiag[0]+1)*sizeof(MatScalar)*bs2,&b->a);CHKERRQ(ierr);
  b->j          = bj;
  b->i          = bi;
  b->diag       = bdiag;
  b->free_diag  = PETSC_TRUE;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((bs*n+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(B,(bdiag[0]+1)*(sizeof(PetscInt)+sizeof(PetscScalar)*bs2));CHKERRQ(ierr);

  b->maxnz = b->nz = bdiag[0]+1;
  B->factortype            =  MAT_FACTOR_LU;
  B->info.factor_mallocs   = reallocs;
  B->info.fill_ratio_given = f;

  if (ai[n] != 0) {
    B->info.fill_ratio_needed = ((PetscReal)(bdiag[0]+1))/((PetscReal)ai[n]);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = B->info.fill_ratio_needed;
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,f,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G);\n",af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix\n");CHKERRQ(ierr);
  }
#endif

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  both_identity = (PetscBool) (row_identity && col_identity);
  ierr = MatSeqBAIJSetNumericFactorization(B,both_identity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 }

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_SeqBAIJ_inplace"
PetscErrorCode MatLUFactorSymbolic_SeqBAIJ_inplace(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqBAIJ        *a = (Mat_SeqBAIJ*)A->data,*b;
  PetscInt           n=a->mbs,bs = A->rmap->bs,bs2=a->bs2;
  PetscBool          row_identity,col_identity,both_identity;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           i,*ai=a->i,*aj=a->j;
  PetscInt           *bi,*bj,*ajtmp;
  PetscInt           *bdiag,row,nnz,nzi,reallocs=0,nzbd,*im;
  PetscReal          f;
  PetscInt           nlnk,*lnk,k,**bi_ptr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix must be square");
  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);

  bi[0] = bdiag[0] = 0;

  /* linked list for storing column indices of the active row */
  nlnk = n + 1;
  ierr = PetscLLCreate(n,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  ierr = PetscMalloc2(n+1,PetscInt**,&bi_ptr,n+1,PetscInt,&im);CHKERRQ(ierr);

  /* initial FreeSpace size is f*(ai[n]+1) */
  f = info->fill;
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (i=0; i<n; i++) {
    /* copy previous fill into linked list */
    nzi = 0;
    nnz = ai[r[i]+1] - ai[r[i]];
    if (!nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    ajtmp = aj + ai[r[i]];
    ierr = PetscLLAddPerm(nnz,ajtmp,ic,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzi += nlnk;

    /* add pivot rows into linked list */
    row = lnk[n];
    while (row < i) {
      nzbd    = bdiag[row] - bi[row] + 1; /* num of entries in the row with column index <= row */
      ajtmp   = bi_ptr[row] + nzbd; /* points to the entry next to the diagonal */
      ierr = PetscLLAddSortedLU(ajtmp,row,nlnk,lnk,lnkbt,i,nzbd,im);CHKERRQ(ierr);
      nzi += nlnk;
      row  = lnk[row];
    }
    bi[i+1] = bi[i] + nzi;
    im[i]   = nzi;

    /* mark bdiag */
    nzbd = 0;
    nnz  = nzi;
    k    = lnk[n];
    while (nnz-- && k < i){
      nzbd++;
      k = lnk[k];
    }
    bdiag[i] = bi[i] + nzbd;

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzi) {
      nnz = (n - i)*nzi; /* estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(nnz,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(n,n,nzi,lnk,current_space->array,lnkbt);CHKERRQ(ierr);
    bi_ptr[i] = current_space->array;
    current_space->array           += nzi;
    current_space->local_used      += nzi;
    current_space->local_remaining -= nzi;
  }
#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = ((PetscReal)bi[n])/((PetscReal)ai[n]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,f,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G);\n",af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix\n");CHKERRQ(ierr);
  }
#endif

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((bi[n]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFree2(bi_ptr,im);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqBAIJSetPreallocation_SeqBAIJ(B,bs,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,isicol);CHKERRQ(ierr);
  b    = (Mat_SeqBAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr          = PetscMalloc((bi[n]+1)*sizeof(MatScalar)*bs2,&b->a);CHKERRQ(ierr);
  b->j          = bj;
  b->i          = bi;
  b->diag       = bdiag;
  b->free_diag  = PETSC_TRUE;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  b->pivotinblocks = (info->pivotinblocks) ? PETSC_TRUE : PETSC_FALSE;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((bs*n+bs)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(B,(bi[n]-n)*(sizeof(PetscInt)+sizeof(PetscScalar)*bs2));CHKERRQ(ierr);

  b->maxnz = b->nz = bi[n] ;
  (B)->factortype            =  MAT_FACTOR_LU;
  (B)->info.factor_mallocs   = reallocs;
  (B)->info.fill_ratio_given = f;

  if (ai[n] != 0) {
    (B)->info.fill_ratio_needed = ((PetscReal)bi[n])/((PetscReal)ai[n]);
  } else {
    (B)->info.fill_ratio_needed = 0.0;
  }

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  both_identity = (PetscBool) (row_identity && col_identity);
  ierr = MatSeqBAIJSetNumericFactorization_inplace(B,both_identity);CHKERRQ(ierr);
  PetscFunctionReturn(0);
 }

