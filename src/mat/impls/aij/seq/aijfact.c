
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <petscbt.h>
#include <../src/mat/utils/freespace.h>

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetOrdering_Flow_SeqAIJ"
/*
      Computes an ordering to get most of the large numerical values in the lower triangular part of the matrix

      This code does not work and is not called anywhere. It would be registered with MatOrderingRegisterAll()
*/
PetscErrorCode MatGetOrdering_Flow_SeqAIJ(Mat mat,const MatOrderingType type,IS *irow,IS *icol)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)mat->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,jj,k, kk,n = mat->rmap->n, current = 0, newcurrent = 0,*order;
  const PetscInt    *ai = a->i, *aj = a->j;
  const PetscScalar *aa = a->a;
  PetscBool         *done;
  PetscReal         best,past = 0,future;

  PetscFunctionBegin;
  /* pick initial row */
  best = -1;
  for (i=0; i<n; i++) {
    future = 0.0;
    for (j=ai[i]; j<ai[i+1]; j++) {
      if (aj[j] != i) future  += PetscAbsScalar(aa[j]); else past = PetscAbsScalar(aa[j]);
    }
    if (!future) future = 1.e-10; /* if there is zero in the upper diagonal part want to rank this row high */
    if (past/future > best) {
      best = past/future;
      current = i;
    }
  }

  ierr = PetscMalloc(n*sizeof(PetscBool),&done);CHKERRQ(ierr);
  ierr = PetscMemzero(done,n*sizeof(PetscBool));CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&order);CHKERRQ(ierr);
  order[0] = current;
  for (i=0; i<n-1; i++) {
    done[current] = PETSC_TRUE;
    best          = -1;
    /* loop over all neighbors of current pivot */
    for (j=ai[current]; j<ai[current+1]; j++) {
      jj = aj[j];
      if (done[jj]) continue;
      /* loop over columns of potential next row computing weights for below and above diagonal */
      past = future = 0.0;
      for (k=ai[jj]; k<ai[jj+1]; k++) {
        kk = aj[k];
        if (done[kk]) past += PetscAbsScalar(aa[k]);
        else if (kk != jj) future  += PetscAbsScalar(aa[k]);
      }
      if (!future) future = 1.e-10; /* if there is zero in the upper diagonal part want to rank this row high */
      if (past/future > best) {
        best = past/future;
        newcurrent = jj;
      }
    }
    if (best == -1) { /* no neighbors to select from so select best of all that remain */
      best = -1;
      for (k=0; k<n; k++) {
        if (done[k]) continue;
        future = 0.0;
        past   = 0.0;
        for (j=ai[k]; j<ai[k+1]; j++) {
          kk = aj[j];
          if (done[kk]) past += PetscAbsScalar(aa[j]);
          else if (kk != k) future  += PetscAbsScalar(aa[j]);
        }
        if (!future) future = 1.e-10; /* if there is zero in the upper diagonal part want to rank this row high */
        if (past/future > best) {
          best = past/future;
          newcurrent = k;
        }
      }
    }
    if (current == newcurrent) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"newcurrent cannot be current");
    current = newcurrent;
    order[i+1] = current;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,order,PETSC_COPY_VALUES,irow);CHKERRQ(ierr);
  *icol = *irow;
  ierr = PetscObjectReference((PetscObject)*irow);CHKERRQ(ierr);
  ierr = PetscFree(done);CHKERRQ(ierr);
  ierr = PetscFree(order);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactorAvailable_seqaij_petsc"
PetscErrorCode MatGetFactorAvailable_seqaij_petsc(Mat A,MatFactorType ftype,PetscBool  *flg)
{
  PetscFunctionBegin;
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_petsc"
PetscErrorCode MatGetFactor_seqaij_petsc(Mat A,MatFactorType ftype,Mat *B)
{
  PetscInt           n = A->rmap->n;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (A->hermitian && (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC))SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Hermitian Factor is not supported");
#endif
  ierr = MatCreate(((PetscObject)A)->comm,B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU || ftype == MAT_FACTOR_ILUDT){
    ierr = MatSetType(*B,MATSEQAIJ);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = MatILUFactorSymbolic_SeqAIJ;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJ;
    ierr = MatSetBlockSizes(*B,A->rmap->bs,A->cmap->bs);CHKERRQ(ierr);
  } else if (ftype == MAT_FACTOR_CHOLESKY || ftype == MAT_FACTOR_ICC) {
    ierr = MatSetType(*B,MATSEQSBAIJ);CHKERRQ(ierr);
    ierr = MatSeqSBAIJSetPreallocation(*B,1,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
    (*B)->ops->iccfactorsymbolic      = MatICCFactorSymbolic_SeqAIJ;
    (*B)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqAIJ;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  (*B)->factortype = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_inplace"
PetscErrorCode MatLUFactorSymbolic_SeqAIJ_inplace(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           i,n=A->rmap->n,*ai=a->i,*aj=a->j;
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

  /* get new row pointers */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  bi[0] = 0;

  /* bdiag is location of diagonal in factor */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);
  bdiag[0] = 0;

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
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,isicol);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr          = PetscMalloc((bi[n]+1)*sizeof(PetscScalar),&b->a);CHKERRQ(ierr);
  b->j          = bj; 
  b->i          = bi;
  b->diag       = bdiag;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr          = PetscMalloc((n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);

  /* In b structure:  Free imax, ilen, old a, old j.  Allocate solve_work, new a, new j */
  ierr = PetscLogObjectMemory(B,(bi[n]-n)*(sizeof(PetscInt)+sizeof(PetscScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = bi[n] ;

  (B)->factortype            = MAT_FACTOR_LU;
  (B)->info.factor_mallocs   = reallocs;
  (B)->info.fill_ratio_given = f;

  if (ai[n]) {
    (B)->info.fill_ratio_needed = ((PetscReal)bi[n])/((PetscReal)ai[n]);
  } else {
    (B)->info.fill_ratio_needed = 0.0;
  }
  (B)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_inplace;
  if (a->inode.size) {
    (B)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Inode_inplace;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ"
PetscErrorCode MatLUFactorSymbolic_SeqAIJ(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           i,n=A->rmap->n,*ai=a->i,*aj=a->j;
  PetscInt           *bi,*bj,*ajtmp;
  PetscInt           *bdiag,row,nnz,nzi,reallocs=0,nzbd,*im;
  PetscReal          f;
  PetscInt           nlnk,*lnk,k,**bi_ptr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT            lnkbt;

  PetscFunctionBegin; 
  /* Uncomment the oldatastruct part only while testing new data structure for MatSolve() */
  /*
  PetscBool          olddatastruct=PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-lu_old",&olddatastruct,PETSC_NULL);CHKERRQ(ierr);
  if(olddatastruct){
    ierr = MatLUFactorSymbolic_SeqAIJ_inplace(B,A,isrow,iscol,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  */
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
    while (row < i){
      nzbd  = bdiag[row] + 1; /* num of entries in the row with column index <= row */
      ajtmp = bi_ptr[row] + nzbd; /* points to the entry next to the diagonal */   
      ierr  = PetscLLAddSortedLU(ajtmp,row,nlnk,lnk,lnkbt,i,nzbd,im);CHKERRQ(ierr);
      nzi  += nlnk;
      row   = lnk[row];
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
    bdiag[i] = nzbd; /* note: bdiag[i] = nnzL as input for PetscFreeSpaceContiguous_LU() */

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

  /*   copy free_space into bj and free free_space; set bi, bj, bdiag in new datastructure; */
  ierr = PetscMalloc((bi[n]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous_LU(&free_space,bj,n,bi,bdiag);CHKERRQ(ierr); 
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFree2(bi_ptr,im);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,isicol);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)(B)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr = PetscMalloc((bdiag[0]+1)*sizeof(PetscScalar),&b->a);CHKERRQ(ierr);
  b->j          = bj; 
  b->i          = bi;
  b->diag       = bdiag;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr          = PetscMalloc((n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);

  /* In b structure:  Free imax, ilen, old a, old j.  Allocate solve_work, new a, new j */
  ierr = PetscLogObjectMemory(B,(bdiag[0]+1)*(sizeof(PetscInt)+sizeof(PetscScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = bdiag[0]+1;
  B->factortype            = MAT_FACTOR_LU;
  B->info.factor_mallocs   = reallocs;
  B->info.fill_ratio_given = f;

  if (ai[n]) {
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
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ;
  if (a->inode.size) {
    B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Inode;
  }
  PetscFunctionReturn(0); 
}

/*
    Trouble in factorization, should we dump the original matrix?
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFactorDumpMatrix"
PetscErrorCode MatFactorDumpMatrix(Mat A)
{
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-mat_factor_dump_on_error",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    PetscViewer viewer;
    char        filename[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(filename,PETSC_MAX_PATH_LEN,"matrix_factor_error.%d",PetscGlobalRank);CHKERRQ(ierr);
    ierr = PetscViewerBinaryOpen(((PetscObject)A)->comm,filename,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
    ierr = MatView(A,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ"
PetscErrorCode MatLUFactorNumeric_SeqAIJ(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat              C=B;
  Mat_SeqAIJ       *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ *)C->data;
  IS               isrow = b->row,isicol = b->icol;
  PetscErrorCode   ierr;
  const PetscInt   *r,*ic,*ics;
  const PetscInt   n=A->rmap->n,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j,*bdiag=b->diag;
  PetscInt         i,j,k,nz,nzL,row,*pj;
  const PetscInt   *ajtmp,*bjtmp;
  MatScalar        *rtmp,*pc,multiplier,*pv;
  const  MatScalar *aa=a->a,*v;
  PetscBool        row_identity,col_identity;
  FactorShiftCtx   sctx;
  const PetscInt   *ddiag;
  PetscReal        rs;
  MatScalar        d;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    ddiag          = a->diag;
    sctx.shift_top = info->zeropivot;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[ddiag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top   *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  } 

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ics  = ic;

  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<n; i++){
      /* zero rtmp */
      /* L part */
      nz    = bi[i+1] - bi[i];
      bjtmp = bj + bi[i];
      for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      /* U part */
      nz = bdiag[i]-bdiag[i+1];
      bjtmp = bj + bdiag[i+1]+1;
      for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0; 
   
      /* load in initial (unfactored row) */
      nz    = ai[r[i]+1] - ai[r[i]];
      ajtmp = aj + ai[r[i]];
      v     = aa + ai[r[i]];
      for (j=0; j<nz; j++) {
        rtmp[ics[ajtmp[j]]] = v[j];
      }
      /* ZeropivotApply() */
      rtmp[i] += sctx.shift_amount;  /* shift the diagonal of the matrix */
    
      /* elimination */
      bjtmp = bj + bi[i];
      row   = *bjtmp++;
      nzL   = bi[i+1] - bi[i];
      for(k=0; k < nzL;k++) {
        pc = rtmp + row;
        if (*pc != 0.0) {
          pv         = b->a + bdiag[row];
          multiplier = *pc * (*pv); 
          *pc        = multiplier;
          pj = b->j + bdiag[row+1]+1; /* beginning of U(row,:) */
	  pv = b->a + bdiag[row+1]+1;
	  nz = bdiag[row]-bdiag[row+1]-1; /* num of entries in U(row,:) excluding diag */
          for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
          ierr = PetscLogFlops(1+2*nz);CHKERRQ(ierr);
        }
        row = *bjtmp++;
      }

      /* finished row so stick it into b->a */
      rs = 0.0;
      /* L part */
      pv   = b->a + bi[i] ;
      pj   = b->j + bi[i] ;
      nz   = bi[i+1] - bi[i];
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]]; rs += PetscAbsScalar(pv[j]);
      }

      /* U part */
      pv = b->a + bdiag[i+1]+1;
      pj = b->j + bdiag[i+1]+1;
      nz = bdiag[i] - bdiag[i+1]-1;
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]]; rs += PetscAbsScalar(pv[j]);
      }

      sctx.rs  = rs;
      sctx.pv  = rtmp[i];
      ierr = MatPivotCheck(A,info,&sctx,i);CHKERRQ(ierr);
      if(sctx.newshift) break; /* break for-loop */
      rtmp[i] = sctx.pv; /* sctx.pv might be updated in the case of MAT_SHIFT_INBLOCKS */

      /* Mark diagonal and invert diagonal for simplier triangular solves */
      pv  = b->a + bdiag[i];
      *pv = 1.0/rtmp[i];

    } /* endof for (i=0; i<n; i++){ */

    /* MatPivotRefine() */
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && !sctx.newshift && sctx.shift_fraction>0 && sctx.nshift<sctx.nshift_max){
      /* 
       * if no shift in this attempt & shifting & started shifting & can refine,
       * then try lower shift
       */
      sctx.shift_hi       = sctx.shift_fraction;
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;
      sctx.shift_amount   = sctx.shift_fraction * sctx.shift_top;
      sctx.newshift       = PETSC_TRUE;
      sctx.nshift++;
    }
  } while (sctx.newshift);

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    C->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    C->ops->solve = MatSolve_SeqAIJ; 
  }
  C->ops->solveadd           = MatSolveAdd_SeqAIJ;
  C->ops->solvetranspose     = MatSolveTranspose_SeqAIJ;
  C->ops->solvetransposeadd  = MatSolveTransposeAdd_SeqAIJ;
  C->ops->matsolve           = MatMatSolve_SeqAIJ;
  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);

  /* MatShiftView(A,info,&sctx) */
  if (sctx.nshift){
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %G, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,sctx.shift_amount,sctx.shift_fraction,sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS){
      ierr = PetscInfo2(A,"number of shift_inblocks applied %D, each shift_amount %G\n",sctx.nshift,info->shiftamount);CHKERRQ(ierr);
    }
  }
  ierr = Mat_CheckInode_FactorLU(C,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_inplace"
PetscErrorCode MatLUFactorNumeric_SeqAIJ_inplace(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat             C=B;
  Mat_SeqAIJ      *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ *)C->data;
  IS              isrow = b->row,isicol = b->icol;
  PetscErrorCode  ierr;
  const PetscInt   *r,*ic,*ics;
  PetscInt        nz,row,i,j,n=A->rmap->n,diag;
  const PetscInt  *ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  const PetscInt  *ajtmp,*bjtmp,*diag_offset = b->diag,*pj;
  MatScalar       *pv,*rtmp,*pc,multiplier,d;
  const MatScalar *v,*aa=a->a;
  PetscReal       rs=0.0;
  FactorShiftCtx  sctx;
  const PetscInt  *ddiag;
  PetscBool       row_identity, col_identity;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    ddiag          = a->diag;
    sctx.shift_top = info->zeropivot;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[ddiag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top   *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  } 

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ics  = ic;

  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<n; i++){
      nz    = bi[i+1] - bi[i];
      bjtmp = bj + bi[i];
      for  (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      /* load in initial (unfactored row) */
      nz    = ai[r[i]+1] - ai[r[i]];
      ajtmp = aj + ai[r[i]];
      v     = aa + ai[r[i]];
      for (j=0; j<nz; j++) {
        rtmp[ics[ajtmp[j]]] = v[j];
      }
      rtmp[ics[r[i]]] += sctx.shift_amount; /* shift the diagonal of the matrix */

      row = *bjtmp++;
      while  (row < i) {
        pc = rtmp + row;
        if (*pc != 0.0) {
          pv         = b->a + diag_offset[row];
          pj         = b->j + diag_offset[row] + 1;
          multiplier = *pc / *pv++;
          *pc        = multiplier;
          nz         = bi[row+1] - diag_offset[row] - 1;
          for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
          ierr = PetscLogFlops(1+2*nz);CHKERRQ(ierr);
        }
        row = *bjtmp++;
      }
      /* finished row so stick it into b->a */
      pv   = b->a + bi[i] ;
      pj   = b->j + bi[i] ;
      nz   = bi[i+1] - bi[i];
      diag = diag_offset[i] - bi[i];
      rs   = 0.0;
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]];
        rs   += PetscAbsScalar(pv[j]);
      }
      rs   -= PetscAbsScalar(pv[diag]);

      sctx.rs  = rs;
      sctx.pv  = pv[diag];
      ierr = MatPivotCheck(A,info,&sctx,i);CHKERRQ(ierr);
      if (sctx.newshift) break;
      pv[diag] = sctx.pv;
    } 

    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && !sctx.newshift && sctx.shift_fraction>0 && sctx.nshift<sctx.nshift_max) {
      /*
       * if no shift in this attempt & shifting & started shifting & can refine,
       * then try lower shift
       */
      sctx.shift_hi       = sctx.shift_fraction;
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;
      sctx.shift_amount   = sctx.shift_fraction * sctx.shift_top;
      sctx.newshift       = PETSC_TRUE;
      sctx.nshift++;
    }
  } while (sctx.newshift);

  /* invert diagonal entries for simplier triangular solves */
  for (i=0; i<n; i++) {
    b->a[diag_offset[i]] = 1.0/b->a[diag_offset[i]];
  }
  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    C->ops->solve   = MatSolve_SeqAIJ_NaturalOrdering_inplace;
  } else {
    C->ops->solve   = MatSolve_SeqAIJ_inplace;
  }
  C->ops->solveadd           = MatSolveAdd_SeqAIJ_inplace;
  C->ops->solvetranspose     = MatSolveTranspose_SeqAIJ_inplace;
  C->ops->solvetransposeadd  = MatSolveTransposeAdd_SeqAIJ_inplace;
  C->ops->matsolve           = MatMatSolve_SeqAIJ_inplace;
  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);
  if (sctx.nshift){
     if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %G, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,sctx.shift_amount,sctx.shift_fraction,sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  (C)->ops->solve            = MatSolve_SeqAIJ_inplace;
  (C)->ops->solvetranspose   = MatSolveTranspose_SeqAIJ_inplace;
  ierr = Mat_CheckInode(C,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
   This routine implements inplace ILU(0) with row or/and column permutations. 
   Input: 
     A - original matrix
   Output;
     A - a->i (rowptr) is same as original rowptr, but factored i-the row is stored in rowperm[i] 
         a->j (col index) is permuted by the inverse of colperm, then sorted
         a->a reordered accordingly with a->j
         a->diag (ptr to diagonal elements) is updated.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_InplaceWithPerm"
PetscErrorCode MatLUFactorNumeric_SeqAIJ_InplaceWithPerm(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data; 
  IS             isrow = a->row,isicol = a->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic,*ics;
  PetscInt       i,j,n=A->rmap->n,*ai=a->i,*aj=a->j;
  PetscInt       *ajtmp,nz,row;
  PetscInt       *diag = a->diag,nbdiag,*pj;
  PetscScalar    *rtmp,*pc,multiplier,d;
  MatScalar      *pv,*v;
  PetscReal      rs;
  FactorShiftCtx sctx;
  const  MatScalar *aa=a->a,*vtmp;

  PetscFunctionBegin;
  if (A != B) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"input and output matrix must have same address");

  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal) MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    const PetscInt   *ddiag = a->diag;
    sctx.shift_top = info->zeropivot;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[ddiag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      vtmp  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(vtmp[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top   *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  } 

  ierr  = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr  = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr  = PetscMalloc((n+1)*sizeof(PetscScalar),&rtmp);CHKERRQ(ierr);
  ierr  = PetscMemzero(rtmp,(n+1)*sizeof(PetscScalar));CHKERRQ(ierr);
  ics = ic;

#if defined(MV)
  sctx.shift_top      = 0.;
  sctx.nshift_max     = 0;
  sctx.shift_lo       = 0.;
  sctx.shift_hi       = 0.;
  sctx.shift_fraction = 0.;

  if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    sctx.shift_top = 0.;
    for (i=0; i<n; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (a->a)[diag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = a->a+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    if (sctx.shift_top < info->zeropivot) sctx.shift_top = info->zeropivot;
    sctx.shift_top    *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  }

  sctx.shift_amount = 0.;
  sctx.nshift       = 0;
#endif

  do {
    sctx.newshift = PETSC_FALSE;
    for (i=0; i<n; i++){
      /* load in initial unfactored row */
      nz    = ai[r[i]+1] - ai[r[i]];
      ajtmp = aj + ai[r[i]];
      v     = a->a + ai[r[i]];
      /* sort permuted ajtmp and values v accordingly */
      for (j=0; j<nz; j++) ajtmp[j] = ics[ajtmp[j]];
      ierr = PetscSortIntWithScalarArray(nz,ajtmp,v);CHKERRQ(ierr);

      diag[r[i]] = ai[r[i]]; 
      for (j=0; j<nz; j++) {
        rtmp[ajtmp[j]] = v[j];
        if (ajtmp[j] < i) diag[r[i]]++; /* update a->diag */
      }
      rtmp[r[i]] += sctx.shift_amount; /* shift the diagonal of the matrix */

      row = *ajtmp++;
      while  (row < i) {
        pc = rtmp + row;
        if (*pc != 0.0) {
          pv         = a->a + diag[r[row]];
          pj         = aj + diag[r[row]] + 1;

          multiplier = *pc / *pv++;
          *pc        = multiplier;
          nz         = ai[r[row]+1] - diag[r[row]] - 1;
          for (j=0; j<nz; j++) rtmp[pj[j]] -= multiplier * pv[j];
          ierr = PetscLogFlops(1+2*nz);CHKERRQ(ierr);
        }
        row = *ajtmp++;
      }
      /* finished row so overwrite it onto a->a */
      pv   = a->a + ai[r[i]] ;
      pj   = aj + ai[r[i]] ;
      nz   = ai[r[i]+1] - ai[r[i]];
      nbdiag = diag[r[i]] - ai[r[i]]; /* num of entries before the diagonal */
      
      rs   = 0.0;
      for (j=0; j<nz; j++) {
        pv[j] = rtmp[pj[j]];
        if (j != nbdiag) rs += PetscAbsScalar(pv[j]);
      }

      sctx.rs  = rs;
      sctx.pv  = pv[nbdiag];
      ierr = MatPivotCheck(A,info,&sctx,i);CHKERRQ(ierr);
      if (sctx.newshift) break;
      pv[nbdiag] = sctx.pv;
    } 

    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE && !sctx.newshift && sctx.shift_fraction>0 && sctx.nshift<sctx.nshift_max) {
      /*
       * if no shift in this attempt & shifting & started shifting & can refine,
       * then try lower shift
       */
      sctx.shift_hi        = sctx.shift_fraction;
      sctx.shift_fraction = (sctx.shift_hi+sctx.shift_lo)/2.;
      sctx.shift_amount    = sctx.shift_fraction * sctx.shift_top;
      sctx.newshift         = PETSC_TRUE;
      sctx.nshift++;
    }
  } while (sctx.newshift);

  /* invert diagonal entries for simplier triangular solves */
  for (i=0; i<n; i++) {
    a->a[diag[r[i]]] = 1.0/a->a[diag[r[i]]];
  }

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  A->ops->solve             = MatSolve_SeqAIJ_InplaceWithPerm;
  A->ops->solveadd          = MatSolveAdd_SeqAIJ_inplace;
  A->ops->solvetranspose    = MatSolveTranspose_SeqAIJ_inplace;
  A->ops->solvetransposeadd = MatSolveTransposeAdd_SeqAIJ_inplace;
  A->assembled = PETSC_TRUE;
  A->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(A->cmap->n);CHKERRQ(ierr);
  if (sctx.nshift){
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %G, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,sctx.shift_amount,sctx.shift_fraction,sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    } 
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactor_SeqAIJ"
PetscErrorCode MatLUFactor_SeqAIJ(Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode ierr;
  Mat            C;

  PetscFunctionBegin;
  ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&C);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(C,A,row,col,info);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(C,A,info);CHKERRQ(ierr);
  A->ops->solve            = C->ops->solve;
  A->ops->solvetranspose   = C->ops->solvetranspose;
  ierr = MatHeaderMerge(A,C);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(A,((Mat_SeqAIJ*)(A->data))->icol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------- */


#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_inplace"
PetscErrorCode MatSolve_SeqAIJ_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  PetscInt          i, n = A->rmap->n,*vi,*ai = a->i,*aj = a->j;
  PetscInt          nz;
  const PetscInt    *rout,*cout,*r,*c;
  PetscScalar       *x,*tmp,*tmps,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;
 
  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  tmps   = tmp;
  for (i=1; i<n; i++) {
    v   = aa + ai[i] ;
    vi  = aj + ai[i] ;
    nz  = a->diag[i] - ai[i];
    sum = b[*r++];
    PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--){
    v   = aa + a->diag[i] + 1;
    vi  = aj + a->diag[i] + 1;
    nz  = ai[i+1] - a->diag[i] - 1;
    sum = tmp[i];
    PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
    x[*c--] = tmp[i] = sum*aa[a->diag[i]];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_SeqAIJ_inplace"
PetscErrorCode MatMatSolve_SeqAIJ_inplace(Mat A,Mat B,Mat X)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  IS              iscol = a->col,isrow = a->row;
  PetscErrorCode  ierr;
  PetscInt        i, n = A->rmap->n,*vi,*ai = a->i,*aj = a->j;
  PetscInt        nz,neq; 
  const PetscInt  *rout,*cout,*r,*c;
  PetscScalar     *x,*b,*tmp,*tmps,sum;
  const MatScalar *aa = a->a,*v;
  PetscBool       bisdense,xisdense;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&bisdense);CHKERRQ(ierr);
  if (!bisdense) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"B matrix must be a SeqDense matrix");
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&xisdense);CHKERRQ(ierr);
  if (!xisdense) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"X matrix must be a SeqDense matrix");

  ierr = MatGetArray(B,&b);CHKERRQ(ierr); 
  ierr = MatGetArray(X,&x);CHKERRQ(ierr);
  
  tmp  = a->solve_work;
  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  for (neq=0; neq<B->cmap->n; neq++){
    /* forward solve the lower triangular */
    tmp[0] = b[r[0]];
    tmps   = tmp;
    for (i=1; i<n; i++) {
      v   = aa + ai[i] ;
      vi  = aj + ai[i] ;
      nz  = a->diag[i] - ai[i];
      sum = b[r[i]];
      PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
      tmp[i] = sum;
    }
    /* backward solve the upper triangular */
    for (i=n-1; i>=0; i--){
      v   = aa + a->diag[i] + 1;
      vi  = aj + a->diag[i] + 1;
      nz  = ai[i+1] - a->diag[i] - 1;
      sum = tmp[i];
      PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
      x[c[i]] = tmp[i] = sum*aa[a->diag[i]];
    }

    b += n;
    x += n;
  }
  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr); 
  ierr = MatRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(B->cmap->n*(2.0*a->nz - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_SeqAIJ"
PetscErrorCode MatMatSolve_SeqAIJ(Mat A,Mat B,Mat X)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  IS              iscol = a->col,isrow = a->row;
  PetscErrorCode  ierr;
  PetscInt        i, n = A->rmap->n,*vi,*ai = a->i,*aj = a->j,*adiag = a->diag;
  PetscInt        nz,neq; 
  const PetscInt  *rout,*cout,*r,*c;
  PetscScalar     *x,*b,*tmp,sum;
  const MatScalar *aa = a->a,*v;
  PetscBool       bisdense,xisdense;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&bisdense);CHKERRQ(ierr);
  if (!bisdense) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"B matrix must be a SeqDense matrix");
  ierr = PetscObjectTypeCompare((PetscObject)X,MATSEQDENSE,&xisdense);CHKERRQ(ierr);
  if (!xisdense) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"X matrix must be a SeqDense matrix");

  ierr = MatGetArray(B,&b);CHKERRQ(ierr); 
  ierr = MatGetArray(X,&x);CHKERRQ(ierr);
  
  tmp  = a->solve_work;
  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  for (neq=0; neq<B->cmap->n; neq++){
    /* forward solve the lower triangular */
    tmp[0] = b[r[0]];
    v      = aa;
    vi     = aj;
    for (i=1; i<n; i++) {
      nz  = ai[i+1] - ai[i];
      sum = b[r[i]];
      PetscSparseDenseMinusDot(sum,tmp,v,vi,nz); 
      tmp[i] = sum;
      v += nz; vi += nz;
    }

    /* backward solve the upper triangular */
    for (i=n-1; i>=0; i--){
      v   = aa + adiag[i+1]+1;
      vi  = aj + adiag[i+1]+1;
      nz  = adiag[i]-adiag[i+1]-1;
      sum = tmp[i];
      PetscSparseDenseMinusDot(sum,tmp,v,vi,nz); 
      x[c[i]] = tmp[i] = sum*v[nz]; /* v[nz] = aa[adiag[i]] */
    }
  
    b += n;
    x += n;
  }
  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr); 
  ierr = MatRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(B->cmap->n*(2.0*a->nz - n));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}  

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_InplaceWithPerm"
PetscErrorCode MatSolve_SeqAIJ_InplaceWithPerm(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;
  IS              iscol = a->col,isrow = a->row;
  PetscErrorCode  ierr;
  const PetscInt  *r,*c,*rout,*cout;
  PetscInt        i, n = A->rmap->n,*vi,*ai = a->i,*aj = a->j;
  PetscInt        nz,row;
  PetscScalar     *x,*b,*tmp,*tmps,sum;
  const MatScalar *aa = a->a,*v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  tmps   = tmp;
  for (row=1; row<n; row++) {
    i   = rout[row]; /* permuted row */
    v   = aa + ai[i] ;
    vi  = aj + ai[i] ;
    nz  = a->diag[i] - ai[i];
    sum = b[*r++];
    PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
    tmp[row] = sum;
  }

  /* backward solve the upper triangular */
  for (row=n-1; row>=0; row--){
    i   = rout[row]; /* permuted row */
    v   = aa + a->diag[i] + 1;
    vi  = aj + a->diag[i] + 1;
    nz  = ai[i+1] - a->diag[i] - 1;
    sum = tmp[row];
    PetscSparseDenseMinusDot(sum,tmps,v,vi,nz); 
    x[*c--] = tmp[row] = sum*aa[a->diag[i]];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------- */
#include <../src/mat/impls/aij/seq/ftn-kernels/fsolve.h>
#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_NaturalOrdering_inplace"
PetscErrorCode MatSolve_SeqAIJ_NaturalOrdering_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          n = A->rmap->n;
  const PetscInt    *ai = a->i,*aj = a->j,*adiag = a->diag;
  PetscScalar       *x;
  const PetscScalar *b;
  const MatScalar   *aa = a->a;
#if !defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
  PetscInt          adiag_i,i,nz,ai_i;
  const PetscInt    *vi;
  const MatScalar   *v;
  PetscScalar       sum;
#endif

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_SOLVEAIJ)
  fortransolveaij_(&n,x,ai,aj,adiag,aa,b);
#else
  /* forward solve the lower triangular */
  x[0] = b[0];
  for (i=1; i<n; i++) {
    ai_i = ai[i];
    v    = aa + ai_i;
    vi   = aj + ai_i;
    nz   = adiag[i] - ai_i;
    sum  = b[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);    
    x[i] = sum;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--){
    adiag_i = adiag[i];
    v       = aa + adiag_i + 1;
    vi      = aj + adiag_i + 1;
    nz      = ai[i+1] - adiag_i - 1;
    sum     = x[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    x[i]    = sum*aa[adiag_i];
  }
#endif
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveAdd_SeqAIJ_inplace"
PetscErrorCode MatSolveAdd_SeqAIJ_inplace(Mat A,Vec bb,Vec yy,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  PetscInt          i, n = A->rmap->n,j;
  PetscInt          nz;
  const PetscInt    *rout,*cout,*r,*c,*vi,*ai = a->i,*aj = a->j;
  PetscScalar       *x,*tmp,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;

  PetscFunctionBegin;
  if (yy != xx) {ierr = VecCopy(yy,xx);CHKERRQ(ierr);}

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout + (n-1);

  /* forward solve the lower triangular */
  tmp[0] = b[*r++];
  for (i=1; i<n; i++) {
    v   = aa + ai[i] ;
    vi  = aj + ai[i] ;
    nz  = a->diag[i] - ai[i];
    sum = b[*r++];
    for (j=0; j<nz; j++) sum -= v[j]*tmp[vi[j]];
    tmp[i] = sum;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--){
    v   = aa + a->diag[i] + 1;
    vi  = aj + a->diag[i] + 1;
    nz  = ai[i+1] - a->diag[i] - 1;
    sum = tmp[i];
    for (j=0; j<nz; j++) sum -= v[j]*tmp[vi[j]];
    tmp[i] = sum*aa[a->diag[i]];
    x[*c--] += tmp[i];
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveAdd_SeqAIJ"
PetscErrorCode MatSolveAdd_SeqAIJ(Mat A,Vec bb,Vec yy,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  PetscInt          i, n = A->rmap->n,j;
  PetscInt          nz;
  const PetscInt    *rout,*cout,*r,*c,*vi,*ai = a->i,*aj = a->j,*adiag = a->diag;
  PetscScalar       *x,*tmp,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;

  PetscFunctionBegin;
  if (yy != xx) {ierr = VecCopy(yy,xx);CHKERRQ(ierr);}

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* forward solve the lower triangular */
  tmp[0] = b[r[0]];
  v      = aa;
  vi     = aj;
  for (i=1; i<n; i++) {
    nz  = ai[i+1] - ai[i];
    sum = b[r[i]];
    for (j=0; j<nz; j++) sum -= v[j]*tmp[vi[j]];
    tmp[i] = sum;
    v += nz; vi += nz;
  }

  /* backward solve the upper triangular */
  v  = aa + adiag[n-1];
  vi = aj + adiag[n-1]; 
  for (i=n-1; i>=0; i--){
    nz  = adiag[i] - adiag[i+1] - 1;
    sum = tmp[i];
    for (j=0; j<nz; j++) sum -= v[j]*tmp[vi[j]];
    tmp[i] = sum*v[nz];
    x[c[i]] += tmp[i];
    v += nz+1; vi += nz+1;
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*a->nz);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_SeqAIJ_inplace"
PetscErrorCode MatSolveTranspose_SeqAIJ_inplace(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *rout,*cout,*r,*c,*diag = a->diag,*ai = a->i,*aj = a->j,*vi;
  PetscInt          i,n = A->rmap->n,j;
  PetscInt          nz;
  PetscScalar       *x,*tmp,s1;
  const MatScalar   *aa = a->a,*v;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for (i=0; i<n; i++) {
    v   = aa + diag[i] ;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    s1  = tmp[i];
    s1 *= (*v++);  /* multiply by inverse of diagonal entry */
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for (i=n-1; i>=0; i--){
    v   = aa + diag[i] - 1 ;
    vi  = aj + diag[i] - 1 ;
    nz  = diag[i] - ai[i];
    s1  = tmp[i];
    for (j=0; j>-nz; j--) tmp[vi[j]] -= s1*v[j];
  }

  /* copy tmp into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] = tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*a->nz-A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_SeqAIJ"
PetscErrorCode MatSolveTranspose_SeqAIJ(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *rout,*cout,*r,*c,*adiag = a->diag,*ai = a->i,*aj = a->j,*vi;
  PetscInt          i,n = A->rmap->n,j;
  PetscInt          nz;
  PetscScalar       *x,*tmp,s1;
  const MatScalar   *aa = a->a,*v;
  const PetscScalar *b;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for (i=0; i<n; i++) {
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz  = adiag[i] - adiag[i+1] - 1;
    s1  = tmp[i];
    s1 *= v[nz];  /* multiply by inverse of diagonal entry */
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for (i=n-1; i>=0; i--){
    v   = aa + ai[i];
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    s1  = tmp[i];
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
  }

  /* copy tmp into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] = tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*a->nz-A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTransposeAdd_SeqAIJ_inplace"
PetscErrorCode MatSolveTransposeAdd_SeqAIJ_inplace(Mat A,Vec bb,Vec zz,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *rout,*cout,*r,*c,*diag = a->diag,*ai = a->i,*aj = a->j,*vi;
  PetscInt          i,n = A->rmap->n,j;
  PetscInt          nz;
  PetscScalar       *x,*tmp,s1;
  const MatScalar   *aa = a->a,*v;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (zz != xx) {ierr = VecCopy(zz,xx);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for (i=0; i<n; i++) {
    v   = aa + diag[i] ;
    vi  = aj + diag[i] + 1;
    nz  = ai[i+1] - diag[i] - 1;
    s1  = tmp[i];
    s1 *= (*v++);  /* multiply by inverse of diagonal entry */
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
    tmp[i] = s1;
  }

  /* backward solve the L^T */
  for (i=n-1; i>=0; i--){
    v   = aa + diag[i] - 1 ;
    vi  = aj + diag[i] - 1 ;
    nz  = diag[i] - ai[i];
    s1  = tmp[i];
    for (j=0; j>-nz; j--) tmp[vi[j]] -= s1*v[j];
  }

  /* copy tmp into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] += tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*a->nz-A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTransposeAdd_SeqAIJ"
PetscErrorCode MatSolveTransposeAdd_SeqAIJ(Mat A,Vec bb,Vec zz,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  const PetscInt    *rout,*cout,*r,*c,*adiag = a->diag,*ai = a->i,*aj = a->j,*vi;
  PetscInt          i,n = A->rmap->n,j;
  PetscInt          nz;
  PetscScalar       *x,*tmp,s1;
  const MatScalar   *aa = a->a,*v;
  const PetscScalar *b;

  PetscFunctionBegin;
  if (zz != xx) {ierr = VecCopy(zz,xx);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout;

  /* copy the b into temp work space according to permutation */
  for (i=0; i<n; i++) tmp[i] = b[c[i]]; 

  /* forward solve the U^T */
  for (i=0; i<n; i++) {
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz  = adiag[i] - adiag[i+1] - 1;
    s1  = tmp[i];
    s1 *= v[nz];  /* multiply by inverse of diagonal entry */
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
    tmp[i] = s1;
  }


  /* backward solve the L^T */
  for (i=n-1; i>=0; i--){
    v   = aa + ai[i] ;
    vi  = aj + ai[i];
    nz  = ai[i+1] - ai[i];
    s1  = tmp[i];
    for (j=0; j<nz; j++) tmp[vi[j]] -= s1*v[j];
  }

  /* copy tmp into x according to permutation */
  for (i=0; i<n; i++) x[r[i]] += tmp[i];

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);

  ierr = PetscLogFlops(2.0*a->nz-A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/

extern PetscErrorCode MatDuplicateNoCreate_SeqAIJ(Mat,Mat,MatDuplicateOption,PetscBool );

/* 
   ilu() under revised new data structure.
   Factored arrays bj and ba are stored as
     L(0,:), L(1,:), ...,L(n-1,:),  U(n-1,:),...,U(i,:),U(i-1,:),...,U(0,:)

   bi=fact->i is an array of size n+1, in which 
   bi+
     bi[i]:  points to 1st entry of L(i,:),i=0,...,n-1
     bi[n]:  points to L(n-1,n-1)+1
     
  bdiag=fact->diag is an array of size n+1,in which
     bdiag[i]: points to diagonal of U(i,:), i=0,...,n-1
     bdiag[n]: points to entry of U(n-1,0)-1

   U(i,:) contains bdiag[i] as its last entry, i.e., 
    U(i,:) = (u[i,i+1],...,u[i,n-1],diag[i])
*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJ_ilu0"
PetscErrorCode MatILUFactorSymbolic_SeqAIJ_ilu0(Mat fact,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  PetscErrorCode     ierr;
  const PetscInt     n=A->rmap->n,*ai=a->i,*aj,*adiag=a->diag;
  PetscInt           i,j,k=0,nz,*bi,*bj,*bdiag; 
  PetscBool          missing;
  IS                 isicol;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);
  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);

  ierr = MatDuplicateNoCreate_SeqAIJ(fact,A,MAT_DO_NOT_COPY_VALUES,PETSC_FALSE);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)(fact)->data;

  /* allocate matrix arrays for new data structure */
  ierr = PetscMalloc3(ai[n]+1,PetscScalar,&b->a,ai[n]+1,PetscInt,&b->j,n+1,PetscInt,&b->i);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(fact,ai[n]*(sizeof(PetscScalar)+sizeof(PetscInt))+(n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  b->singlemalloc = PETSC_TRUE;
  if (!b->diag){
    ierr = PetscMalloc((n+1)*sizeof(PetscInt),&b->diag);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(fact,(n+1)*sizeof(PetscInt));CHKERRQ(ierr);
  }
  bdiag = b->diag;  
 
  if (n > 0) {
    ierr = PetscMemzero(b->a,(ai[n])*sizeof(MatScalar));CHKERRQ(ierr);
  }
  
  /* set bi and bj with new data structure */
  bi = b->i;
  bj = b->j;

  /* L part */
  bi[0] = 0;
  for (i=0; i<n; i++){
    nz = adiag[i] - ai[i];
    bi[i+1] = bi[i] + nz;
    aj = a->j + ai[i];
    for (j=0; j<nz; j++){
      /*   *bj = aj[j]; bj++; */
      bj[k++] = aj[j];
    }
  }
  
  /* U part */
  bdiag[n] = bi[n]-1;
  for (i=n-1; i>=0; i--){
    nz = ai[i+1] - adiag[i] - 1;
    aj = a->j + adiag[i] + 1;
    for (j=0; j<nz; j++){
      /*      *bj = aj[j]; bj++; */
      bj[k++] = aj[j];
    }
    /* diag[i] */
    /*    *bj = i; bj++; */
    bj[k++] = i;
    bdiag[i] = bdiag[i+1] + nz + 1;
  }

  fact->factortype             = MAT_FACTOR_ILU;
  fact->info.factor_mallocs    = 0;
  fact->info.fill_ratio_given  = info->fill;
  fact->info.fill_ratio_needed = 1.0;
  fact->ops->lufactornumeric   = MatLUFactorNumeric_SeqAIJ;

  b       = (Mat_SeqAIJ*)(fact)->data;
  b->row  = isrow;
  b->col  = iscol;
  b->icol = isicol;
  ierr    = PetscMalloc((fact->rmap->n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr    = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr    = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJ"
PetscErrorCode MatILUFactorSymbolic_SeqAIJ(Mat fact,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           n=A->rmap->n,*ai=a->i,*aj=a->j;
  PetscInt           *bi,*cols,nnz,*cols_lvl;
  PetscInt           *bdiag,prow,fm,nzbd,reallocs=0,dcount=0;
  PetscInt           i,levels,diagonal_fill;
  PetscBool          col_identity,row_identity;
  PetscReal          f;
  PetscInt           nlnk,*lnk,*lnk_lvl=PETSC_NULL;
  PetscBT            lnkbt;
  PetscInt           nzi,*bj,**bj_ptr,**bjlvl_ptr; 
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL; 
  PetscFreeSpaceList free_space_lvl=PETSC_NULL,current_space_lvl=PETSC_NULL; 
  
  PetscFunctionBegin;
  /* Uncomment the old data struct part only while testing new data structure for MatSolve() */
  /*
  PetscBool          olddatastruct=PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-ilu_old",&olddatastruct,PETSC_NULL);CHKERRQ(ierr);
  if(olddatastruct){
    ierr = MatILUFactorSymbolic_SeqAIJ_inplace(fact,A,isrow,iscol,info);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  */
  
  levels = (PetscInt)info->levels;
  ierr   = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr   = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);

  if (!levels && row_identity && col_identity) { 
    /* special case: ilu(0) with natural ordering */
    ierr = MatILUFactorSymbolic_SeqAIJ_ilu0(fact,A,isrow,iscol,info);CHKERRQ(ierr);
    if (a->inode.size) {
      fact->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Inode;
    }
    PetscFunctionReturn(0);
  }

  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);
  bi[0] = bdiag[0] = 0;

  ierr = PetscMalloc2(n,PetscInt*,&bj_ptr,n,PetscInt*,&bjlvl_ptr);CHKERRQ(ierr); 

  /* create a linked list for storing column indices of the active row */
  nlnk = n + 1;
  ierr = PetscIncompleteLLCreate(n,n,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is f*(ai[n]+1) */
  f             = info->fill;
  diagonal_fill = (PetscInt)info->diagonal_fill;
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space_lvl);CHKERRQ(ierr);
  current_space_lvl = free_space_lvl;
 
  for (i=0; i<n; i++) {
    nzi = 0;
    /* copy current row into linked list */
    nnz  = ai[r[i]+1] - ai[r[i]];
    if (!nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    cols = aj + ai[r[i]];
    lnk[i] = -1; /* marker to indicate if diagonal exists */
    ierr = PetscIncompleteLLInit(nnz,cols,n,ic,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
    nzi += nlnk;

    /* make sure diagonal entry is included */
    if (diagonal_fill && lnk[i] == -1) {
      fm = n;
      while (lnk[fm] < i) fm = lnk[fm];
      lnk[i]     = lnk[fm]; /* insert diagonal into linked list */
      lnk[fm]    = i;
      lnk_lvl[i] = 0;
      nzi++; dcount++; 
    }

    /* add pivot rows into the active row */
    nzbd = 0;
    prow = lnk[n];
    while (prow < i) {
      nnz      = bdiag[prow];
      cols     = bj_ptr[prow] + nnz + 1;
      cols_lvl = bjlvl_ptr[prow] + nnz + 1; 
      nnz      = bi[prow+1] - bi[prow] - nnz - 1;
      ierr = PetscILULLAddSorted(nnz,cols,levels,cols_lvl,prow,nlnk,lnk,lnk_lvl,lnkbt,prow);CHKERRQ(ierr);
      nzi += nlnk;
      prow = lnk[prow];
      nzbd++;
    }
    bdiag[i] = nzbd;
    bi[i+1]  = bi[i] + nzi;

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzi) {
      nnz = 2*nzi*(n - i); /* estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(nnz,&current_space);CHKERRQ(ierr);
      ierr = PetscFreeSpaceGet(nnz,&current_space_lvl);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free_space and free_space_lvl, then initialize lnk */
    ierr = PetscIncompleteLLClean(n,n,nzi,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt);CHKERRQ(ierr);
    bj_ptr[i]    = current_space->array;
    bjlvl_ptr[i] = current_space_lvl->array;

    /* make sure the active row i has diagonal entry */
    if (*(bj_ptr[i]+bdiag[i]) != i) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Row %D has missing diagonal in factored matrix\ntry running with -pc_factor_nonzeros_along_diagonal or -pc_factor_diagonal_fill",i);

    current_space->array           += nzi;
    current_space->local_used      += nzi;
    current_space->local_remaining -= nzi;
    current_space_lvl->array           += nzi;
    current_space_lvl->local_used      += nzi;
    current_space_lvl->local_remaining -= nzi;
  } 

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  /* copy free_space into bj and free free_space; set bi, bj, bdiag in new datastructure; */
  ierr = PetscMalloc((bi[n]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous_LU(&free_space,bj,n,bi,bdiag);CHKERRQ(ierr);
  
  ierr = PetscIncompleteLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space_lvl);CHKERRQ(ierr); 
  ierr = PetscFree2(bj_ptr,bjlvl_ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  {
    PetscReal af = ((PetscReal)(bdiag[0]+1))/((PetscReal)ai[n]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,f,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -[sub_]pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill([sub]pc,%G);\n",af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
    if (diagonal_fill) {
      ierr = PetscInfo1(A,"Detected and replaced %D missing diagonals",dcount);CHKERRQ(ierr);
    }
  }
#endif

  /* put together the new matrix */
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(fact,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(fact,isicol);CHKERRQ(ierr);
  b = (Mat_SeqAIJ*)(fact)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr = PetscMalloc((bdiag[0]+1)*sizeof(PetscScalar),&b->a);CHKERRQ(ierr);
  b->j          = bj;
  b->i          = bi;
  b->diag       = bdiag;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate bdiag, solve_work, new a, new j */
  ierr = PetscLogObjectMemory(fact,(bdiag[0]+1)*(sizeof(PetscInt)+sizeof(PetscScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = bdiag[0]+1;
  (fact)->info.factor_mallocs    = reallocs;
  (fact)->info.fill_ratio_given  = f;
  (fact)->info.fill_ratio_needed = ((PetscReal)(bdiag[0]+1))/((PetscReal)ai[n]);
  (fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ;
  if (a->inode.size) {
    (fact)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Inode;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUFactorSymbolic_SeqAIJ_inplace"
PetscErrorCode MatILUFactorSymbolic_SeqAIJ_inplace(Mat fact,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           n=A->rmap->n,*ai=a->i,*aj=a->j,d;
  PetscInt           *bi,*cols,nnz,*cols_lvl;
  PetscInt           *bdiag,prow,fm,nzbd,reallocs=0,dcount=0;
  PetscInt           i,levels,diagonal_fill;
  PetscBool          col_identity,row_identity;
  PetscReal          f;
  PetscInt           nlnk,*lnk,*lnk_lvl=PETSC_NULL;
  PetscBT            lnkbt;
  PetscInt           nzi,*bj,**bj_ptr,**bjlvl_ptr; 
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL; 
  PetscFreeSpaceList free_space_lvl=PETSC_NULL,current_space_lvl=PETSC_NULL; 
  PetscBool          missing;
  
  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  f             = info->fill;
  levels        = (PetscInt)info->levels;
  diagonal_fill = (PetscInt)info->diagonal_fill;
  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(iscol,&col_identity);CHKERRQ(ierr);
  if (!levels && row_identity && col_identity) { /* special case: ilu(0) with natural ordering */    
    ierr = MatDuplicateNoCreate_SeqAIJ(fact,A,MAT_DO_NOT_COPY_VALUES,PETSC_TRUE);CHKERRQ(ierr);
    (fact)->ops->lufactornumeric =  MatLUFactorNumeric_SeqAIJ_inplace;    
    if (a->inode.size) {
      (fact)->ops->lufactornumeric  = MatLUFactorNumeric_SeqAIJ_Inode_inplace;
    }   
    fact->factortype = MAT_FACTOR_ILU;
    (fact)->info.factor_mallocs    = 0;
    (fact)->info.fill_ratio_given  = info->fill;
    (fact)->info.fill_ratio_needed = 1.0;
    b               = (Mat_SeqAIJ*)(fact)->data;
    ierr = MatMissingDiagonal(A,&missing,&d);CHKERRQ(ierr);
    if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",d);
    b->row              = isrow;
    b->col              = iscol;
    b->icol             = isicol;
    ierr                = PetscMalloc(((fact)->rmap->n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
    ierr                = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
    ierr                = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  /* get new row and diagonal pointers, must be allocated separately because they will be given to the Mat_SeqAIJ and freed separately */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bi);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);
  bi[0] = bdiag[0] = 0;

  ierr = PetscMalloc2(n,PetscInt*,&bj_ptr,n,PetscInt*,&bjlvl_ptr);CHKERRQ(ierr); 

  /* create a linked list for storing column indices of the active row */
  nlnk = n + 1;
  ierr = PetscIncompleteLLCreate(n,n,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is f*(ai[n]+1) */
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;
  ierr = PetscFreeSpaceGet((PetscInt)(f*(ai[n]+1)),&free_space_lvl);CHKERRQ(ierr);
  current_space_lvl = free_space_lvl;
 
  for (i=0; i<n; i++) {
    nzi = 0;
    /* copy current row into linked list */
    nnz  = ai[r[i]+1] - ai[r[i]];
    if (!nnz) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    cols = aj + ai[r[i]];
    lnk[i] = -1; /* marker to indicate if diagonal exists */
    ierr = PetscIncompleteLLInit(nnz,cols,n,ic,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
    nzi += nlnk;

    /* make sure diagonal entry is included */
    if (diagonal_fill && lnk[i] == -1) {
      fm = n;
      while (lnk[fm] < i) fm = lnk[fm];
      lnk[i]     = lnk[fm]; /* insert diagonal into linked list */
      lnk[fm]    = i;
      lnk_lvl[i] = 0;
      nzi++; dcount++; 
    }

    /* add pivot rows into the active row */
    nzbd = 0;
    prow = lnk[n];
    while (prow < i) {
      nnz      = bdiag[prow];
      cols     = bj_ptr[prow] + nnz + 1;
      cols_lvl = bjlvl_ptr[prow] + nnz + 1; 
      nnz      = bi[prow+1] - bi[prow] - nnz - 1;
      ierr = PetscILULLAddSorted(nnz,cols,levels,cols_lvl,prow,nlnk,lnk,lnk_lvl,lnkbt,prow);CHKERRQ(ierr);
      nzi += nlnk;
      prow = lnk[prow];
      nzbd++;
    }
    bdiag[i] = nzbd;
    bi[i+1]  = bi[i] + nzi;

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzi) {
      nnz = nzi*(n - i); /* estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(nnz,&current_space);CHKERRQ(ierr);
      ierr = PetscFreeSpaceGet(nnz,&current_space_lvl);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free_space and free_space_lvl, then initialize lnk */
    ierr = PetscIncompleteLLClean(n,n,nzi,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt);CHKERRQ(ierr);
    bj_ptr[i]    = current_space->array;
    bjlvl_ptr[i] = current_space_lvl->array;

    /* make sure the active row i has diagonal entry */
    if (*(bj_ptr[i]+bdiag[i]) != i) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Row %D has missing diagonal in factored matrix\ntry running with -pc_factor_nonzeros_along_diagonal or -pc_factor_diagonal_fill",i);

    current_space->array           += nzi;
    current_space->local_used      += nzi;
    current_space->local_remaining -= nzi;
    current_space_lvl->array           += nzi;
    current_space_lvl->local_used      += nzi;
    current_space_lvl->local_remaining -= nzi;
  } 

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  /* destroy list of free space and other temporary arrays */
  ierr = PetscMalloc((bi[n]+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,bj);CHKERRQ(ierr); /* copy free_space -> bj */
  ierr = PetscIncompleteLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFreeSpaceDestroy(free_space_lvl);CHKERRQ(ierr); 
  ierr = PetscFree2(bj_ptr,bjlvl_ptr);CHKERRQ(ierr);

#if defined(PETSC_USE_INFO)
  {
    PetscReal af = ((PetscReal)bi[n])/((PetscReal)ai[n]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,f,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -[sub_]pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill([sub]pc,%G);\n",af);CHKERRQ(ierr);
    ierr = PetscInfo(A,"for best performance.\n");CHKERRQ(ierr);
    if (diagonal_fill) {
      ierr = PetscInfo1(A,"Detected and replaced %D missing diagonals",dcount);CHKERRQ(ierr);
    }
  }
#endif

  /* put together the new matrix */
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(fact,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(fact,isicol);CHKERRQ(ierr);
  b = (Mat_SeqAIJ*)(fact)->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  ierr = PetscMalloc(bi[n]*sizeof(PetscScalar),&b->a);CHKERRQ(ierr);
  b->j          = bj;
  b->i          = bi;
  for (i=0; i<n; i++) bdiag[i] += bi[i];
  b->diag       = bdiag;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  /* In b structure:  Free imax, ilen, old a, old j.  
     Allocate bdiag, solve_work, new a, new j */
  ierr = PetscLogObjectMemory(fact,(bi[n]-n) * (sizeof(PetscInt)+sizeof(PetscScalar)));CHKERRQ(ierr);
  b->maxnz             = b->nz = bi[n] ;
  (fact)->info.factor_mallocs    = reallocs;
  (fact)->info.fill_ratio_given  = f;
  (fact)->info.fill_ratio_needed = ((PetscReal)bi[n])/((PetscReal)ai[n]);
  (fact)->ops->lufactornumeric =  MatLUFactorNumeric_SeqAIJ_inplace;
  if (a->inode.size) {
    (fact)->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_Inode_inplace;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqAIJ"
PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C = B;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  IS             ip=b->row,iip = b->icol;
  PetscErrorCode ierr;
  const PetscInt *rip,*riip;
  PetscInt       i,j,mbs=A->rmap->n,*bi=b->i,*bj=b->j,*bdiag=b->diag,*bjtmp;
  PetscInt       *ai=a->i,*aj=a->j;
  PetscInt       k,jmin,jmax,*c2r,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa=a->a,dk,uikdi;
  PetscBool      perm_identity;
  FactorShiftCtx sctx;
  PetscReal      rs;
  MatScalar      d,*v;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    sctx.shift_top = info->zeropivot;
    for (i=0; i<mbs; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[a->diag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top   *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  } 

  ierr  = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr  = ISGetIndices(iip,&riip);CHKERRQ(ierr);
  
  /* allocate working arrays
     c2r: linked list, keep track of pivot rows for a given column. c2r[col]: head of the list for a given col
     il:  for active k row, il[i] gives the index of the 1st nonzero entry in U[i,k:n-1] in bj and ba arrays 
  */
  ierr = PetscMalloc3(mbs,MatScalar,&rtmp,mbs,PetscInt,&il,mbs,PetscInt,&c2r);CHKERRQ(ierr);
 
  do {
    sctx.newshift = PETSC_FALSE;

    for (i=0; i<mbs; i++) c2r[i] = mbs; 
    if (mbs) il[0] = 0;
 
    for (k = 0; k<mbs; k++){
      /* zero rtmp */
      nz = bi[k+1] - bi[k];
      bjtmp = bj + bi[k];
      for (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;
      
      /* load in initial unfactored row */
      bval = ba + bi[k];
      jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
      for (j = jmin; j < jmax; j++){
        col = riip[aj[j]];
        if (col >= k){ /* only take upper triangular entry */
          rtmp[col] = aa[j];
          *bval++   = 0.0; /* for in-place factorization */
        }
      } 
      /* shift the diagonal of the matrix: ZeropivotApply() */   
      rtmp[k] += sctx.shift_amount;  /* shift the diagonal of the matrix */
    
      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i  = c2r[k]; /* first row to be added to k_th row  */ 

      while (i < k){
        nexti = c2r[i]; /* next row to be added to k_th row */
   
        /* compute multiplier, update diag(k) and U(i,k) */
        ili   = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
        uikdi = - ba[ili]*ba[bdiag[i]];  /* diagonal(k) */
        dk   += uikdi*ba[ili]; /* update diag[k] */
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax){
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];         
          /* update il and c2r for row i */
          il[i] = jmin;             
          j = bj[jmin]; c2r[i] = c2r[j]; c2r[j] = i; 
        }      
        i = nexti;         
      }

      /* copy data into U(k,:) */
      rs   = 0.0;
      jmin = bi[k]; jmax = bi[k+1]-1; 
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++){
          col = bj[j]; ba[j] = rtmp[col]; rs += PetscAbsScalar(ba[j]);
        }       
        /* add the k-th row into il and c2r */
        il[k] = jmin;
        i = bj[jmin]; c2r[k] = c2r[i]; c2r[i] = k;
      }  

      /* MatPivotCheck() */
      sctx.rs  = rs;
      sctx.pv  = dk;
      ierr = MatPivotCheck(A,info,&sctx,i);CHKERRQ(ierr);
      if(sctx.newshift) break;
      dk = sctx.pv;
 
      ba[bdiag[k]] = 1.0/dk; /* U(k,k) */
    } 
  } while (sctx.newshift);
  
  ierr = PetscFree3(rtmp,il,c2r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iip,&riip);CHKERRQ(ierr);

  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity){
    B->ops->solve           = MatSolve_SeqSBAIJ_1_NaturalOrdering;
    B->ops->solvetranspose  = MatSolve_SeqSBAIJ_1_NaturalOrdering;
    B->ops->forwardsolve    = MatForwardSolve_SeqSBAIJ_1_NaturalOrdering;
    B->ops->backwardsolve   = MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering;
  } else {
    B->ops->solve           = MatSolve_SeqSBAIJ_1;
    B->ops->solvetranspose  = MatSolve_SeqSBAIJ_1;
    B->ops->forwardsolve    = MatForwardSolve_SeqSBAIJ_1;
    B->ops->backwardsolve   = MatBackwardSolve_SeqSBAIJ_1;
  }

  C->assembled    = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(C->rmap->n);CHKERRQ(ierr);

  /* MatPivotView() */
  if (sctx.nshift){
    if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo4(A,"number of shift_pd tries %D, shift_amount %G, diagonal shifted up by %e fraction top_value %e\n",sctx.nshift,sctx.shift_amount,sctx.shift_fraction,sctx.shift_top);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shift_nz tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_INBLOCKS){
      ierr = PetscInfo2(A,"number of shift_inblocks applied %D, each shift_amount %G\n",sctx.nshift,info->shiftamount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqAIJ_inplace"
PetscErrorCode MatCholeskyFactorNumeric_SeqAIJ_inplace(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat            C = B;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ   *b=(Mat_SeqSBAIJ*)C->data;
  IS             ip=b->row,iip = b->icol;
  PetscErrorCode ierr;
  const PetscInt *rip,*riip;
  PetscInt       i,j,mbs=A->rmap->n,*bi=b->i,*bj=b->j,*bcol,*bjtmp;
  PetscInt       *ai=a->i,*aj=a->j;
  PetscInt       k,jmin,jmax,*jl,*il,col,nexti,ili,nz;
  MatScalar      *rtmp,*ba=b->a,*bval,*aa=a->a,dk,uikdi;
  PetscBool      perm_identity;
  FactorShiftCtx sctx;
  PetscReal      rs;
  MatScalar      d,*v;

  PetscFunctionBegin;
  /* MatPivotSetUp(): initialize shift context sctx */
  ierr = PetscMemzero(&sctx,sizeof(FactorShiftCtx));CHKERRQ(ierr);

  if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) { /* set sctx.shift_top=max{rs} */
    sctx.shift_top = info->zeropivot;
    for (i=0; i<mbs; i++) {
      /* calculate sum(|aij|)-RealPart(aii), amt of shift needed for this row */
      d  = (aa)[a->diag[i]];
      rs = -PetscAbsScalar(d) - PetscRealPart(d);
      v  = aa+ai[i];
      nz = ai[i+1] - ai[i];
      for (j=0; j<nz; j++) 
	rs += PetscAbsScalar(v[j]);
      if (rs>sctx.shift_top) sctx.shift_top = rs;
    }
    sctx.shift_top   *= 1.1;
    sctx.nshift_max   = 5;
    sctx.shift_lo     = 0.;
    sctx.shift_hi     = 1.;
  } 

  ierr  = ISGetIndices(ip,&rip);CHKERRQ(ierr);
  ierr  = ISGetIndices(iip,&riip);CHKERRQ(ierr);
  
  /* initialization */
  ierr = PetscMalloc3(mbs,MatScalar,&rtmp,mbs,PetscInt,&il,mbs,PetscInt,&jl);CHKERRQ(ierr);
  
  do {
    sctx.newshift = PETSC_FALSE;

    for (i=0; i<mbs; i++) jl[i] = mbs; 
    il[0] = 0;
 
    for (k = 0; k<mbs; k++){
      /* zero rtmp */
      nz = bi[k+1] - bi[k];
      bjtmp = bj + bi[k];
      for (j=0; j<nz; j++) rtmp[bjtmp[j]] = 0.0;

      bval = ba + bi[k];
      /* initialize k-th row by the perm[k]-th row of A */
      jmin = ai[rip[k]]; jmax = ai[rip[k]+1];
      for (j = jmin; j < jmax; j++){
        col = riip[aj[j]];
        if (col >= k){ /* only take upper triangular entry */
          rtmp[col] = aa[j];
          *bval++  = 0.0; /* for in-place factorization */
        }
      } 
      /* shift the diagonal of the matrix */
      if (sctx.nshift) rtmp[k] += sctx.shift_amount; 

      /* modify k-th row by adding in those rows i with U(i,k)!=0 */
      dk = rtmp[k];
      i = jl[k]; /* first row to be added to k_th row  */  

      while (i < k){
        nexti = jl[i]; /* next row to be added to k_th row */
        
        /* compute multiplier, update diag(k) and U(i,k) */
        ili = il[i];  /* index of first nonzero element in U(i,k:bms-1) */
        uikdi = - ba[ili]*ba[bi[i]];  /* diagonal(k) */ 
        dk += uikdi*ba[ili];
        ba[ili] = uikdi; /* -U(i,k) */

        /* add multiple of row i to k-th row */
        jmin = ili + 1; jmax = bi[i+1];
        if (jmin < jmax){
          for (j=jmin; j<jmax; j++) rtmp[bj[j]] += uikdi*ba[j];         
          /* update il and jl for row i */
          il[i] = jmin;             
          j = bj[jmin]; jl[i] = jl[j]; jl[j] = i; 
        }      
        i = nexti;         
      }

      /* shift the diagonals when zero pivot is detected */
      /* compute rs=sum of abs(off-diagonal) */
      rs   = 0.0;
      jmin = bi[k]+1; 
      nz   = bi[k+1] - jmin; 
      bcol = bj + jmin;
      for (j=0; j<nz; j++) {
        rs += PetscAbsScalar(rtmp[bcol[j]]);
      }

      sctx.rs = rs;
      sctx.pv = dk;
      ierr = MatPivotCheck(A,info,&sctx,k);CHKERRQ(ierr);
      if (sctx.newshift) break;
      dk = sctx.pv;
   
      /* copy data into U(k,:) */
      ba[bi[k]] = 1.0/dk; /* U(k,k) */
      jmin = bi[k]+1; jmax = bi[k+1];
      if (jmin < jmax) {
        for (j=jmin; j<jmax; j++){
          col = bj[j]; ba[j] = rtmp[col]; 
        }       
        /* add the k-th row into il and jl */
        il[k] = jmin;
        i = bj[jmin]; jl[k] = jl[i]; jl[i] = k;
      }        
    } 
  } while (sctx.newshift);

  ierr = PetscFree3(rtmp,il,jl);CHKERRQ(ierr);
  ierr = ISRestoreIndices(ip,&rip);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iip,&riip);CHKERRQ(ierr);

  ierr = ISIdentity(ip,&perm_identity);CHKERRQ(ierr);
  if (perm_identity){
    B->ops->solve           = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    B->ops->solvetranspose  = MatSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    B->ops->forwardsolve    = MatForwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
    B->ops->backwardsolve   = MatBackwardSolve_SeqSBAIJ_1_NaturalOrdering_inplace;
  } else {
    B->ops->solve           = MatSolve_SeqSBAIJ_1_inplace;
    B->ops->solvetranspose  = MatSolve_SeqSBAIJ_1_inplace;
    B->ops->forwardsolve    = MatForwardSolve_SeqSBAIJ_1_inplace;
    B->ops->backwardsolve   = MatBackwardSolve_SeqSBAIJ_1_inplace;
  }

  C->assembled    = PETSC_TRUE; 
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(C->rmap->n);CHKERRQ(ierr);
  if (sctx.nshift){
    if (info->shifttype == (PetscReal)MAT_SHIFT_NONZERO) {
      ierr = PetscInfo2(A,"number of shiftnz tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    } else if (info->shifttype == (PetscReal)MAT_SHIFT_POSITIVE_DEFINITE) {
      ierr = PetscInfo2(A,"number of shiftpd tries %D, shift_amount %G\n",sctx.nshift,sctx.shift_amount);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0); 
}

/* 
   icc() under revised new data structure.
   Factored arrays bj and ba are stored as
     U(0,:),...,U(i,:),U(n-1,:)

   ui=fact->i is an array of size n+1, in which 
   ui+
     ui[i]:  points to 1st entry of U(i,:),i=0,...,n-1
     ui[n]:  points to U(n-1,n-1)+1
     
  udiag=fact->diag is an array of size n,in which
     udiag[i]: points to diagonal of U(i,:), i=0,...,n-1

   U(i,:) contains udiag[i] as its last entry, i.e., 
    U(i,:) = (u[i,i+1],...,u[i,n-1],diag[i])
*/

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactorSymbolic_SeqAIJ"
PetscErrorCode MatICCFactorSymbolic_SeqAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscInt           reallocs=0,i,*ai=a->i,*aj=a->j,am=A->rmap->n,*ui,*udiag;
  const PetscInt     *rip,*riip;
  PetscInt           jmin,jmax,nzk,k,j,*jl,prow,*il,nextprow;
  PetscInt           nlnk,*lnk,*lnk_lvl=PETSC_NULL,d;
  PetscInt           ncols,ncols_upper,*cols,*ajtmp,*uj,**uj_ptr,**uj_lvl_ptr;
  PetscReal          fill=info->fill,levels=info->levels;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscFreeSpaceList free_space_lvl=PETSC_NULL,current_space_lvl=PETSC_NULL;
  PetscBT            lnkbt;
  IS                 iperm;  
  
  PetscFunctionBegin; 
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatMissingDiagonal(A,&missing,&d);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",d);
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);

  ierr = PetscMalloc((am+1)*sizeof(PetscInt),&ui);CHKERRQ(ierr);
  ierr = PetscMalloc((am+1)*sizeof(PetscInt),&udiag);CHKERRQ(ierr);
  ui[0] = 0;

  /* ICC(0) without matrix ordering: simply rearrange column indices */
  if (!levels && perm_identity) { 
    for (i=0; i<am; i++) {
      ncols    = ai[i+1] - a->diag[i];
      ui[i+1]  = ui[i] + ncols; 
      udiag[i] = ui[i+1] - 1; /* points to the last entry of U(i,:) */
    }
    ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr); 
    cols = uj;
    for (i=0; i<am; i++) {
      aj    = a->j + a->diag[i] + 1; /* 1st entry of U(i,:) without diagonal */ 
      ncols = ai[i+1] - a->diag[i] -1;
      for (j=0; j<ncols; j++) *cols++ = aj[j]; 
      *cols++ = i; /* diagoanl is located as the last entry of U(i,:) */
    }
  } else { /* case: levels>0 || (levels=0 && !perm_identity) */
    ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);
    ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

    /* initialization */
    ierr  = PetscMalloc((am+1)*sizeof(PetscInt),&ajtmp);CHKERRQ(ierr); 

    /* jl: linked list for storing indices of the pivot rows 
       il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
    ierr = PetscMalloc4(am,PetscInt*,&uj_ptr,am,PetscInt*,&uj_lvl_ptr,am,PetscInt,&jl,am,PetscInt,&il);CHKERRQ(ierr);
    for (i=0; i<am; i++){
      jl[i] = am; il[i] = 0;
    }

    /* create and initialize a linked list for storing column indices of the active row k */
    nlnk = am + 1;
    ierr = PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);

    /* initial FreeSpace size is fill*(ai[am]+am)/2 */
    ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+am)/2),&free_space);CHKERRQ(ierr);
    current_space = free_space;
    ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+am)/2),&free_space_lvl);CHKERRQ(ierr);
    current_space_lvl = free_space_lvl;

    for (k=0; k<am; k++){  /* for each active row k */
      /* initialize lnk by the column indices of row rip[k] of A */
      nzk   = 0;
      ncols = ai[rip[k]+1] - ai[rip[k]]; 
      if (!ncols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",rip[k],k);
      ncols_upper = 0;
      for (j=0; j<ncols; j++){
        i = *(aj + ai[rip[k]] + j); /* unpermuted column index */
        if (riip[i] >= k){ /* only take upper triangular entry */
          ajtmp[ncols_upper] = i; 
          ncols_upper++;
        }
      }
      ierr = PetscIncompleteLLInit(ncols_upper,ajtmp,am,riip,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
      nzk += nlnk;

      /* update lnk by computing fill-in for each pivot row to be merged in */
      prow = jl[k]; /* 1st pivot row */
   
      while (prow < k){
        nextprow = jl[prow];
      
        /* merge prow into k-th row */
        jmin = il[prow] + 1;  /* index of the 2nd nzero entry in U(prow,k:am-1) */
        jmax = ui[prow+1]; 
        ncols = jmax-jmin;
        i     = jmin - ui[prow];
        cols  = uj_ptr[prow] + i; /* points to the 2nd nzero entry in U(prow,k:am-1) */
        uj    = uj_lvl_ptr[prow] + i; /* levels of cols */
        j     = *(uj - 1); 
        ierr = PetscICCLLAddSorted(ncols,cols,levels,uj,am,nlnk,lnk,lnk_lvl,lnkbt,j);CHKERRQ(ierr); 
        nzk += nlnk;

        /* update il and jl for prow */
        if (jmin < jmax){
          il[prow] = jmin;
          j = *cols; jl[prow] = jl[j]; jl[j] = prow;  
        } 
        prow = nextprow; 
      }  

      /* if free space is not available, make more free space */
      if (current_space->local_remaining<nzk) {
        i  = am - k + 1; /* num of unfactored rows */
        i *= PetscMin(nzk, i-1); /* i*nzk, i*(i-1): estimated and max additional space needed */
        ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
        ierr = PetscFreeSpaceGet(i,&current_space_lvl);CHKERRQ(ierr);
        reallocs++;
      }

      /* copy data into free_space and free_space_lvl, then initialize lnk */
      if (nzk == 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Empty row %D in ICC matrix factor",k);
      ierr = PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt);CHKERRQ(ierr);

      /* add the k-th row into il and jl */
      if (nzk > 1){
        i = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */    
        jl[k] = jl[i]; jl[i] = k;
        il[k] = ui[k] + 1;
      } 
      uj_ptr[k]     = current_space->array;
      uj_lvl_ptr[k] = current_space_lvl->array; 

      current_space->array           += nzk;
      current_space->local_used      += nzk;
      current_space->local_remaining -= nzk;

      current_space_lvl->array           += nzk;
      current_space_lvl->local_used      += nzk;
      current_space_lvl->local_remaining -= nzk;

      ui[k+1] = ui[k] + nzk;  
    } 

    ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iperm,&riip);CHKERRQ(ierr);
    ierr = PetscFree4(uj_ptr,uj_lvl_ptr,jl,il);CHKERRQ(ierr);
    ierr = PetscFree(ajtmp);CHKERRQ(ierr);

    /* copy free_space into uj and free free_space; set ui, uj, udiag in new datastructure; */
    ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr);
    ierr = PetscFreeSpaceContiguous_Cholesky(&free_space,uj,am,ui,udiag);CHKERRQ(ierr); /* store matrix factor  */
    ierr = PetscIncompleteLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
    ierr = PetscFreeSpaceDestroy(free_space_lvl);CHKERRQ(ierr);

  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */
  b    = (Mat_SeqSBAIJ*)(fact)->data;
  b->singlemalloc = PETSC_FALSE;
  ierr = PetscMalloc((ui[am]+1)*sizeof(MatScalar),&b->a);CHKERRQ(ierr);
  b->j    = uj;
  b->i    = ui;
  b->diag = udiag;
  b->free_diag = PETSC_TRUE;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->col  = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr = PetscMalloc((am+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(fact,ui[am]*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz   = b->nz = ui[am];
  b->free_a  = PETSC_TRUE; 
  b->free_ij = PETSC_TRUE; 
  
  fact->info.factor_mallocs   = reallocs;
  fact->info.fill_ratio_given = fill;
  if (ai[am] != 0) {
    /* nonzeros in lower triangular part of A (including diagonals) = (ai[am]+am)/2 */
    fact->info.fill_ratio_needed = ((PetscReal)2*ui[am])/(ai[am]+am);
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
    if (ai[am] != 0) {
      PetscReal af = fact->info.fill_ratio_needed; 
      ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,fill,af);CHKERRQ(ierr);
      ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
      ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G) for best performance.\n",af);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
    }
#endif
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatICCFactorSymbolic_SeqAIJ_inplace"
PetscErrorCode MatICCFactorSymbolic_SeqAIJ_inplace(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity,missing;
  PetscInt           reallocs=0,i,*ai=a->i,*aj=a->j,am=A->rmap->n,*ui,*udiag;
  const PetscInt     *rip,*riip;
  PetscInt           jmin,jmax,nzk,k,j,*jl,prow,*il,nextprow;
  PetscInt           nlnk,*lnk,*lnk_lvl=PETSC_NULL,d;
  PetscInt           ncols,ncols_upper,*cols,*ajtmp,*uj,**uj_ptr,**uj_lvl_ptr;
  PetscReal          fill=info->fill,levels=info->levels;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscFreeSpaceList free_space_lvl=PETSC_NULL,current_space_lvl=PETSC_NULL;
  PetscBT            lnkbt;
  IS                 iperm;  
  
  PetscFunctionBegin;    
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  ierr = MatMissingDiagonal(A,&missing,&d);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",d);
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);

  ierr = PetscMalloc((am+1)*sizeof(PetscInt),&ui);CHKERRQ(ierr); 
  ierr = PetscMalloc((am+1)*sizeof(PetscInt),&udiag);CHKERRQ(ierr);
  ui[0] = 0; 

  /* ICC(0) without matrix ordering: simply copies fill pattern */
  if (!levels && perm_identity) { 

    for (i=0; i<am; i++) {
      ui[i+1]  = ui[i] + ai[i+1] - a->diag[i]; 
      udiag[i] = ui[i];
    }
    ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr); 
    cols = uj;
    for (i=0; i<am; i++) {
      aj    = a->j + a->diag[i];  
      ncols = ui[i+1] - ui[i];
      for (j=0; j<ncols; j++) *cols++ = *aj++; 
    }
  } else { /* case: levels>0 || (levels=0 && !perm_identity) */
    ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);
    ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

    /* initialization */
    ierr  = PetscMalloc((am+1)*sizeof(PetscInt),&ajtmp);CHKERRQ(ierr); 

    /* jl: linked list for storing indices of the pivot rows 
       il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
    ierr = PetscMalloc4(am,PetscInt*,&uj_ptr,am,PetscInt*,&uj_lvl_ptr,am,PetscInt,&jl,am,PetscInt,&il);CHKERRQ(ierr);
    for (i=0; i<am; i++){
      jl[i] = am; il[i] = 0;
    }

    /* create and initialize a linked list for storing column indices of the active row k */
    nlnk = am + 1;
    ierr = PetscIncompleteLLCreate(am,am,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);

    /* initial FreeSpace size is fill*(ai[am]+1) */
    ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+1)),&free_space);CHKERRQ(ierr);
    current_space = free_space;
    ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+1)),&free_space_lvl);CHKERRQ(ierr);
    current_space_lvl = free_space_lvl;

    for (k=0; k<am; k++){  /* for each active row k */
      /* initialize lnk by the column indices of row rip[k] of A */
      nzk   = 0;
      ncols = ai[rip[k]+1] - ai[rip[k]]; 
      if (!ncols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",rip[k],k);
      ncols_upper = 0;
      for (j=0; j<ncols; j++){
        i = *(aj + ai[rip[k]] + j); /* unpermuted column index */
        if (riip[i] >= k){ /* only take upper triangular entry */
          ajtmp[ncols_upper] = i; 
          ncols_upper++;
        }
      }
      ierr = PetscIncompleteLLInit(ncols_upper,ajtmp,am,riip,nlnk,lnk,lnk_lvl,lnkbt);CHKERRQ(ierr);
      nzk += nlnk;

      /* update lnk by computing fill-in for each pivot row to be merged in */
      prow = jl[k]; /* 1st pivot row */
   
      while (prow < k){
        nextprow = jl[prow];
      
        /* merge prow into k-th row */
        jmin = il[prow] + 1;  /* index of the 2nd nzero entry in U(prow,k:am-1) */
        jmax = ui[prow+1]; 
        ncols = jmax-jmin;
        i     = jmin - ui[prow];
        cols  = uj_ptr[prow] + i; /* points to the 2nd nzero entry in U(prow,k:am-1) */
        uj    = uj_lvl_ptr[prow] + i; /* levels of cols */
        j     = *(uj - 1); 
        ierr = PetscICCLLAddSorted(ncols,cols,levels,uj,am,nlnk,lnk,lnk_lvl,lnkbt,j);CHKERRQ(ierr); 
        nzk += nlnk;

        /* update il and jl for prow */
        if (jmin < jmax){
          il[prow] = jmin;
          j = *cols; jl[prow] = jl[j]; jl[j] = prow;  
        } 
        prow = nextprow; 
      }  

      /* if free space is not available, make more free space */
      if (current_space->local_remaining<nzk) {
        i = am - k + 1; /* num of unfactored rows */
        i *= PetscMin(nzk, (i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
        ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
        ierr = PetscFreeSpaceGet(i,&current_space_lvl);CHKERRQ(ierr);
        reallocs++;
      }

      /* copy data into free_space and free_space_lvl, then initialize lnk */
      if (!nzk) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Empty row %D in ICC matrix factor",k);
      ierr = PetscIncompleteLLClean(am,am,nzk,lnk,lnk_lvl,current_space->array,current_space_lvl->array,lnkbt);CHKERRQ(ierr);

      /* add the k-th row into il and jl */
      if (nzk > 1){
        i = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */    
        jl[k] = jl[i]; jl[i] = k;
        il[k] = ui[k] + 1;
      } 
      uj_ptr[k]     = current_space->array;
      uj_lvl_ptr[k] = current_space_lvl->array; 

      current_space->array           += nzk;
      current_space->local_used      += nzk;
      current_space->local_remaining -= nzk;

      current_space_lvl->array           += nzk;
      current_space_lvl->local_used      += nzk;
      current_space_lvl->local_remaining -= nzk;

      ui[k+1] = ui[k] + nzk;  
    } 

#if defined(PETSC_USE_INFO)
    if (ai[am] != 0) {
      PetscReal af = (PetscReal)ui[am]/((PetscReal)ai[am]);
      ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,fill,af);CHKERRQ(ierr);
      ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
      ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G) for best performance.\n",af);CHKERRQ(ierr);
    } else {
      ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
    }
#endif

    ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
    ierr = ISRestoreIndices(iperm,&riip);CHKERRQ(ierr);
    ierr = PetscFree4(uj_ptr,uj_lvl_ptr,jl,il);CHKERRQ(ierr);
    ierr = PetscFree(ajtmp);CHKERRQ(ierr);

    /* destroy list of free space and other temporary array(s) */
    ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr);
    ierr = PetscFreeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
    ierr = PetscIncompleteLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
    ierr = PetscFreeSpaceDestroy(free_space_lvl);CHKERRQ(ierr);

  } /* end of case: levels>0 || (levels=0 && !perm_identity) */

  /* put together the new matrix in MATSEQSBAIJ format */

  b    = (Mat_SeqSBAIJ*)fact->data;
  b->singlemalloc = PETSC_FALSE;
  ierr = PetscMalloc((ui[am]+1)*sizeof(MatScalar),&b->a);CHKERRQ(ierr);
  b->j    = uj;
  b->i    = ui;
  b->diag = udiag;
  b->free_diag = PETSC_TRUE;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->col  = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr    = PetscMalloc((am+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(fact,(ui[am]-am)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz   = b->nz = ui[am];
  b->free_a  = PETSC_TRUE; 
  b->free_ij = PETSC_TRUE; 
  
  fact->info.factor_mallocs    = reallocs;
  fact->info.fill_ratio_given  = fill;
  if (ai[am] != 0) {
    fact->info.fill_ratio_needed = ((PetscReal)ui[am])/((PetscReal)ai[am]);
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_inplace;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqAIJ"
PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJ(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity;
  PetscReal          fill = info->fill;
  const PetscInt     *rip,*riip;
  PetscInt           i,am=A->rmap->n,*ai=a->i,*aj=a->j,reallocs=0,prow;
  PetscInt           *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt           nlnk,*lnk,ncols,ncols_upper,*cols,*uj,**ui_ptr,*uj_ptr,*udiag;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT            lnkbt;
  IS                 iperm;  

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);  
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);
  ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);  
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* initialization */
  ierr  = PetscMalloc((am+1)*sizeof(PetscInt),&ui);CHKERRQ(ierr);
  ierr  = PetscMalloc((am+1)*sizeof(PetscInt),&udiag);CHKERRQ(ierr);
  ui[0] = 0; 

  /* jl: linked list for storing indices of the pivot rows 
     il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
  ierr = PetscMalloc4(am,PetscInt*,&ui_ptr,am,PetscInt,&jl,am,PetscInt,&il,am,PetscInt,&cols);CHKERRQ(ierr);
  for (i=0; i<am; i++){
    jl[i] = am; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = am + 1;
  ierr = PetscLLCreate(am,am,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[am]+am)/2 */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+am)/2),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (k=0; k<am; k++){  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk   = 0;
    ncols = ai[rip[k]+1] - ai[rip[k]]; 
    if (!ncols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",rip[k],k);
    ncols_upper = 0;
    for (j=0; j<ncols; j++){
      i = riip[*(aj + ai[rip[k]] + j)];  
      if (i >= k){ /* only take upper triangular entry */
        cols[ncols_upper] = i;
        ncols_upper++;
      }
    }
    ierr = PetscLLAdd(ncols_upper,cols,am,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */
   
    while (prow < k){
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin = il[prow] + 1;  /* index of the 2nd nzero entry in U(prow,k:am-1) */
      jmax = ui[prow+1]; 
      ncols = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:am-1) */
      ierr = PetscLLAddSorted(ncols,uj_ptr,am,nlnk,lnk,lnkbt);CHKERRQ(ierr); 
      nzk += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax){
        il[prow] = jmin;
        j = *uj_ptr; jl[prow] = jl[j]; jl[j] = prow;  
      } 
      prow = nextprow; 
    }  

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i  = am - k + 1; /* num of unfactored rows */
      i *= PetscMin(nzk,i-1); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(am,am,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr); 

    /* add the k-th row into il and jl */
    if (nzk > 1){
      i = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */    
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    } 
    ui_ptr[k] = current_space->array;
    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;  
  } 

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iperm,&riip);CHKERRQ(ierr);
  ierr = PetscFree4(ui_ptr,jl,il,cols);CHKERRQ(ierr);

  /* copy free_space into uj and free free_space; set ui, uj, udiag in new datastructure; */
  ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous_Cholesky(&free_space,uj,am,ui,udiag);CHKERRQ(ierr); /* store matrix factor */
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */

  b = (Mat_SeqSBAIJ*)fact->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc((ui[am]+1)*sizeof(MatScalar),&b->a);CHKERRQ(ierr);
  b->j    = uj;
  b->i    = ui;
  b->diag = udiag;
  b->free_diag = PETSC_TRUE;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->col  = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr    = PetscMalloc((am+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory(fact,ui[am]*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = ui[am];
  
  fact->info.factor_mallocs    = reallocs;
  fact->info.fill_ratio_given  = fill;
  if (ai[am] != 0) {
    /* nonzeros in lower triangular part of A (including diagonals) = (ai[am]+am)/2 */
    fact->info.fill_ratio_needed = ((PetscReal)2*ui[am])/(ai[am]+am);
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[am] != 0) {
    PetscReal af = fact->info.fill_ratio_needed; 
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,fill,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G) for best performance.\n",af);CHKERRQ(ierr);
  } else {
     ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqAIJ_inplace"
PetscErrorCode MatCholeskyFactorSymbolic_SeqAIJ_inplace(Mat fact,Mat A,IS perm,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqSBAIJ       *b;
  PetscErrorCode     ierr;
  PetscBool          perm_identity;
  PetscReal          fill = info->fill;
  const PetscInt     *rip,*riip;
  PetscInt           i,am=A->rmap->n,*ai=a->i,*aj=a->j,reallocs=0,prow;
  PetscInt           *jl,jmin,jmax,nzk,*ui,k,j,*il,nextprow;
  PetscInt           nlnk,*lnk,ncols,ncols_upper,*cols,*uj,**ui_ptr,*uj_ptr;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;
  PetscBT            lnkbt;
  IS                 iperm;  

  PetscFunctionBegin;  
  if (A->rmap->n != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Must be square matrix, rows %D columns %D",A->rmap->n,A->cmap->n);
  /* check whether perm is the identity mapping */
  ierr = ISIdentity(perm,&perm_identity);CHKERRQ(ierr);  
  ierr = ISInvertPermutation(perm,PETSC_DECIDE,&iperm);CHKERRQ(ierr);
  ierr = ISGetIndices(iperm,&riip);CHKERRQ(ierr);  
  ierr = ISGetIndices(perm,&rip);CHKERRQ(ierr);

  /* initialization */
  ierr  = PetscMalloc((am+1)*sizeof(PetscInt),&ui);CHKERRQ(ierr);
  ui[0] = 0; 

  /* jl: linked list for storing indices of the pivot rows 
     il: il[i] points to the 1st nonzero entry of U(i,k:am-1) */
  ierr = PetscMalloc4(am,PetscInt*,&ui_ptr,am,PetscInt,&jl,am,PetscInt,&il,am,PetscInt,&cols);CHKERRQ(ierr);
  for (i=0; i<am; i++){
    jl[i] = am; il[i] = 0;
  }

  /* create and initialize a linked list for storing column indices of the active row k */
  nlnk = am + 1;
  ierr = PetscLLCreate(am,am,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* initial FreeSpace size is fill*(ai[am]+1) */
  ierr = PetscFreeSpaceGet((PetscInt)(fill*(ai[am]+1)),&free_space);CHKERRQ(ierr);
  current_space = free_space;

  for (k=0; k<am; k++){  /* for each active row k */
    /* initialize lnk by the column indices of row rip[k] of A */
    nzk   = 0;
    ncols = ai[rip[k]+1] - ai[rip[k]]; 
    if (!ncols) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",rip[k],k);
    ncols_upper = 0;
    for (j=0; j<ncols; j++){
      i = riip[*(aj + ai[rip[k]] + j)];  
      if (i >= k){ /* only take upper triangular entry */
        cols[ncols_upper] = i;
        ncols_upper++;
      }
    }
    ierr = PetscLLAdd(ncols_upper,cols,am,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    nzk += nlnk;

    /* update lnk by computing fill-in for each pivot row to be merged in */
    prow = jl[k]; /* 1st pivot row */
   
    while (prow < k){
      nextprow = jl[prow];
      /* merge prow into k-th row */
      jmin = il[prow] + 1;  /* index of the 2nd nzero entry in U(prow,k:am-1) */
      jmax = ui[prow+1]; 
      ncols = jmax-jmin;
      uj_ptr = ui_ptr[prow] + jmin - ui[prow]; /* points to the 2nd nzero entry in U(prow,k:am-1) */
      ierr = PetscLLAddSorted(ncols,uj_ptr,am,nlnk,lnk,lnkbt);CHKERRQ(ierr); 
      nzk += nlnk;

      /* update il and jl for prow */
      if (jmin < jmax){
        il[prow] = jmin;
        j = *uj_ptr; jl[prow] = jl[j]; jl[j] = prow;  
      } 
      prow = nextprow; 
    }  

    /* if free space is not available, make more free space */
    if (current_space->local_remaining<nzk) {
      i = am - k + 1; /* num of unfactored rows */
      i = PetscMin(i*nzk, i*(i-1)); /* i*nzk, i*(i-1): estimated and max additional space needed */
      ierr = PetscFreeSpaceGet(i,&current_space);CHKERRQ(ierr);
      reallocs++;
    }

    /* copy data into free space, then initialize lnk */
    ierr = PetscLLClean(am,am,nzk,lnk,current_space->array,lnkbt);CHKERRQ(ierr); 

    /* add the k-th row into il and jl */
    if (nzk-1 > 0){
      i = current_space->array[1]; /* col value of the first nonzero element in U(k, k+1:am-1) */    
      jl[k] = jl[i]; jl[i] = k;
      il[k] = ui[k] + 1;
    } 
    ui_ptr[k] = current_space->array;
    current_space->array           += nzk;
    current_space->local_used      += nzk;
    current_space->local_remaining -= nzk;

    ui[k+1] = ui[k] + nzk;  
  } 

#if defined(PETSC_USE_INFO)
  if (ai[am] != 0) {
    PetscReal af = (PetscReal)(ui[am])/((PetscReal)ai[am]);
    ierr = PetscInfo3(A,"Reallocs %D Fill ratio:given %G needed %G\n",reallocs,fill,af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"Run with -pc_factor_fill %G or use \n",af);CHKERRQ(ierr);
    ierr = PetscInfo1(A,"PCFactorSetFill(pc,%G) for best performance.\n",af);CHKERRQ(ierr);
  } else {
     ierr = PetscInfo(A,"Empty matrix.\n");CHKERRQ(ierr);
  }
#endif

  ierr = ISRestoreIndices(perm,&rip);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iperm,&riip);CHKERRQ(ierr);
  ierr = PetscFree4(ui_ptr,jl,il,cols);CHKERRQ(ierr);

  /* destroy list of free space and other temporary array(s) */
  ierr = PetscMalloc((ui[am]+1)*sizeof(PetscInt),&uj);CHKERRQ(ierr);
  ierr = PetscFreeSpaceContiguous(&free_space,uj);CHKERRQ(ierr);
  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);

  /* put together the new matrix in MATSEQSBAIJ format */

  b = (Mat_SeqSBAIJ*)fact->data;
  b->singlemalloc = PETSC_FALSE;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  ierr = PetscMalloc((ui[am]+1)*sizeof(MatScalar),&b->a);CHKERRQ(ierr);
  b->j    = uj;
  b->i    = ui;
  b->diag = 0;
  b->ilen = 0;
  b->imax = 0;
  b->row  = perm;
  b->col  = perm;
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  ierr    = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr); 
  b->icol = iperm;
  b->pivotinblocks = PETSC_FALSE; /* need to get from MatFactorInfo */
  ierr    = PetscMalloc((am+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);
  ierr    = PetscLogObjectMemory(fact,(ui[am]-am)*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = b->nz = ui[am];
  
  fact->info.factor_mallocs    = reallocs;
  fact->info.fill_ratio_given  = fill;
  if (ai[am] != 0) {
    fact->info.fill_ratio_needed = ((PetscReal)ui[am])/((PetscReal)ai[am]);
  } else {
    fact->info.fill_ratio_needed = 0.0;
  }
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqAIJ_inplace;
  PetscFunctionReturn(0); 
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqAIJ_NaturalOrdering"
PetscErrorCode MatSolve_SeqAIJ_NaturalOrdering(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          n = A->rmap->n;
  const PetscInt    *ai = a->i,*aj = a->j,*adiag = a->diag,*vi;
  PetscScalar       *x,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;
  PetscInt          i,nz;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);

  /* forward solve the lower triangular */
  x[0] = b[0];
  v    = aa;
  vi   = aj;
  for (i=1; i<n; i++) {
    nz  = ai[i+1] - ai[i];
    sum = b[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    v  += nz;
    vi += nz;
    x[i] = sum;
  }
  
  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--){
    v   = aa + adiag[i+1] + 1;
    vi  = aj + adiag[i+1] + 1;
    nz = adiag[i] - adiag[i+1]-1;
    sum = x[i];
    PetscSparseDenseMinusDot(sum,x,v,vi,nz);
    x[i] = sum*v[nz]; /* x[i]=aa[adiag[i]]*sum; v++; */
  }
   
  ierr = PetscLogFlops(2.0*a->nz - A->cmap->n);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ"
PetscErrorCode MatSolve_SeqAIJ(Mat A,Vec bb,Vec xx)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  IS                iscol = a->col,isrow = a->row;
  PetscErrorCode    ierr;
  PetscInt          i,n=A->rmap->n,*vi,*ai=a->i,*aj=a->j,*adiag = a->diag,nz;
  const PetscInt    *rout,*cout,*r,*c;
  PetscScalar       *x,*tmp,sum;
  const PetscScalar *b;
  const MatScalar   *aa = a->a,*v;

  PetscFunctionBegin;
  if (!n) PetscFunctionReturn(0);

  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr); 
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  tmp  = a->solve_work;

  ierr = ISGetIndices(isrow,&rout);CHKERRQ(ierr); r = rout;
  ierr = ISGetIndices(iscol,&cout);CHKERRQ(ierr); c = cout; 

  /* forward solve the lower triangular */
  tmp[0] = b[r[0]];
  v      = aa;
  vi     = aj;
  for (i=1; i<n; i++) {
    nz  = ai[i+1] - ai[i];
    sum = b[r[i]];
    PetscSparseDenseMinusDot(sum,tmp,v,vi,nz); 
    tmp[i] = sum;
    v += nz; vi += nz;
  }

  /* backward solve the upper triangular */
  for (i=n-1; i>=0; i--){
    v   = aa + adiag[i+1]+1;
    vi  = aj + adiag[i+1]+1;
    nz  = adiag[i]-adiag[i+1]-1;
    sum = tmp[i];
    PetscSparseDenseMinusDot(sum,tmp,v,vi,nz); 
    x[c[i]] = tmp[i] = sum*v[nz]; /* v[nz] = aa[adiag[i]] */
  }

  ierr = ISRestoreIndices(isrow,&rout);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&cout);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr); 
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*a->nz - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactor_SeqAIJ"
/*
    This will get a new name and become a varient of MatILUFactor_SeqAIJ() there is no longer seperate functions in the matrix function table for dt factors
*/
PetscErrorCode MatILUDTFactor_SeqAIJ(Mat A,IS isrow,IS iscol,const MatFactorInfo *info,Mat *fact)
{
  Mat                B = *fact;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;  
  PetscErrorCode     ierr;
  const PetscInt     *r,*ic;
  PetscInt           i,n=A->rmap->n,*ai=a->i,*aj=a->j,*ajtmp,*adiag;
  PetscInt           *bi,*bj,*bdiag,*bdiag_rev;
  PetscInt           row,nzi,nzi_bl,nzi_bu,*im,nzi_al,nzi_au;
  PetscInt           nlnk,*lnk;
  PetscBT            lnkbt;
  PetscBool          row_identity,icol_identity;
  MatScalar          *aatmp,*pv,*batmp,*ba,*rtmp,*pc,multiplier,*vtmp,diag_tmp;
  const PetscInt     *ics;
  PetscInt           j,nz,*pj,*bjtmp,k,ncut,*jtmp;
  PetscReal          dt=info->dt,dtcol=info->dtcol,shift=info->shiftamount;
  PetscInt           dtcount=(PetscInt)info->dtcount,nnz_max;
  PetscBool          missing;

  PetscFunctionBegin;

  if (dt      == PETSC_DEFAULT) dt      = 0.005;
  if (dtcol   == PETSC_DEFAULT) dtcol   = 0.01; /* XXX unused! */
  if (dtcount == PETSC_DEFAULT) dtcount = (PetscInt)(1.5*a->rmax);

  /* ------- symbolic factorization, can be reused ---------*/
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);
  adiag=a->diag;

  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);

  /* bdiag is location of diagonal in factor */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag);CHKERRQ(ierr);     /* becomes b->diag */
  ierr = PetscMalloc((n+1)*sizeof(PetscInt),&bdiag_rev);CHKERRQ(ierr); /* temporary */

  /* allocate row pointers bi */
  ierr = PetscMalloc((2*n+2)*sizeof(PetscInt),&bi);CHKERRQ(ierr);

  /* allocate bj and ba; max num of nonzero entries is (ai[n]+2*n*dtcount+2) */
  if (dtcount > n-1) dtcount = n-1; /* diagonal is excluded */
  nnz_max  = ai[n]+2*n*dtcount+2;

  ierr = PetscMalloc((nnz_max+1)*sizeof(PetscInt),&bj);CHKERRQ(ierr);
  ierr = PetscMalloc((nnz_max+1)*sizeof(MatScalar),&ba);CHKERRQ(ierr);

  /* put together the new matrix */
  ierr = MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(B,isicol);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)B->data;
  b->free_a       = PETSC_TRUE;
  b->free_ij      = PETSC_TRUE;
  b->singlemalloc = PETSC_FALSE;
  b->a          = ba;
  b->j          = bj; 
  b->i          = bi;
  b->diag       = bdiag;
  b->ilen       = 0;
  b->imax       = 0;
  b->row        = isrow;
  b->col        = iscol;
  ierr          = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol       = isicol;
  ierr = PetscMalloc((n+1)*sizeof(PetscScalar),&b->solve_work);CHKERRQ(ierr);

  ierr = PetscLogObjectMemory(B,nnz_max*(sizeof(PetscInt)+sizeof(MatScalar)));CHKERRQ(ierr);
  b->maxnz = nnz_max;

  B->factortype            = MAT_FACTOR_ILUDT;
  B->info.factor_mallocs   = 0;
  B->info.fill_ratio_given = ((PetscReal)nnz_max)/((PetscReal)ai[n]);
  CHKMEMQ;
  /* ------- end of symbolic factorization ---------*/

  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ics  = ic;

  /* linked list for storing column indices of the active row */
  nlnk = n + 1;
  ierr = PetscLLCreate(n,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);

  /* im: used by PetscLLAddSortedLU(); jtmp: working array for column indices of active row */
  ierr = PetscMalloc2(n,PetscInt,&im,n,PetscInt,&jtmp);CHKERRQ(ierr);
  /* rtmp, vtmp: working arrays for sparse and contiguous row entries of active row */
  ierr = PetscMalloc2(n,MatScalar,&rtmp,n,MatScalar,&vtmp);CHKERRQ(ierr);
  ierr = PetscMemzero(rtmp,n*sizeof(MatScalar));CHKERRQ(ierr); 

  bi[0]    = 0;
  bdiag[0] = nnz_max-1; /* location of diag[0] in factor B */
  bdiag_rev[n] = bdiag[0];
  bi[2*n+1] = bdiag[0]+1; /* endof bj and ba array */
  for (i=0; i<n; i++) {
    /* copy initial fill into linked list */
    nzi = 0; /* nonzeros for active row i */
    nzi = ai[r[i]+1] - ai[r[i]];
    if (!nzi) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Empty row in matrix: row in original ordering %D in permuted ordering %D",r[i],i);
    nzi_al = adiag[r[i]] - ai[r[i]];
    nzi_au = ai[r[i]+1] - adiag[r[i]] -1;
    ajtmp = aj + ai[r[i]]; 
    ierr = PetscLLAddPerm(nzi,ajtmp,ic,n,nlnk,lnk,lnkbt);CHKERRQ(ierr);
    
    /* load in initial (unfactored row) */
    aatmp = a->a + ai[r[i]];
    for (j=0; j<nzi; j++) {
      rtmp[ics[*ajtmp++]] = *aatmp++;
    }
    
    /* add pivot rows into linked list */
    row = lnk[n]; 
    while (row < i ) {
      nzi_bl = bi[row+1] - bi[row] + 1;
      bjtmp = bj + bdiag[row+1]+1; /* points to 1st column next to the diagonal in U */
      ierr  = PetscLLAddSortedLU(bjtmp,row,nlnk,lnk,lnkbt,i,nzi_bl,im);CHKERRQ(ierr);
      nzi  += nlnk;
      row   = lnk[row]; 
    }
    
    /* copy data from lnk into jtmp, then initialize lnk */
    ierr = PetscLLClean(n,n,nzi,lnk,jtmp,lnkbt);CHKERRQ(ierr);

    /* numerical factorization */
    bjtmp = jtmp;
    row   = *bjtmp++; /* 1st pivot row */
    while  ( row < i ) {
      pc         = rtmp + row;
      pv         = ba + bdiag[row]; /* 1./(diag of the pivot row) */
      multiplier = (*pc) * (*pv); 
      *pc        = multiplier; 
      if (PetscAbsScalar(*pc) > dt){ /* apply tolerance dropping rule */
        pj         = bj + bdiag[row+1] + 1; /* point to 1st entry of U(row,:) */
        pv         = ba + bdiag[row+1] + 1;
        /* if (multiplier < -1.0 or multiplier >1.0) printf("row/prow %d, %d, multiplier %g\n",i,row,multiplier); */
        nz         = bdiag[row] - bdiag[row+1] - 1; /* num of entries in U(row,:), excluding diagonal */
        for (j=0; j<nz; j++) rtmp[*pj++] -= multiplier * (*pv++);
        ierr = PetscLogFlops(1+2*nz);CHKERRQ(ierr);
      } 
      row = *bjtmp++;
    }

    /* copy sparse rtmp into contiguous vtmp; separate L and U part */
    diag_tmp = rtmp[i];  /* save diagonal value - may not needed?? */
    nzi_bl = 0; j = 0;
    while (jtmp[j] < i){ /* Note: jtmp is sorted */
      vtmp[j] = rtmp[jtmp[j]]; rtmp[jtmp[j]]=0.0;
      nzi_bl++; j++;
    }
    nzi_bu = nzi - nzi_bl -1;
    while (j < nzi){
      vtmp[j] = rtmp[jtmp[j]]; rtmp[jtmp[j]]=0.0;
      j++;
    }
    
    bjtmp = bj + bi[i];
    batmp = ba + bi[i];
    /* apply level dropping rule to L part */
    ncut = nzi_al + dtcount; 
    if (ncut < nzi_bl){ 
      ierr = PetscSortSplit(ncut,nzi_bl,vtmp,jtmp);CHKERRQ(ierr);
      ierr = PetscSortIntWithScalarArray(ncut,jtmp,vtmp);CHKERRQ(ierr);
    } else {
      ncut = nzi_bl;
    }
    for (j=0; j<ncut; j++){
      bjtmp[j] = jtmp[j];
      batmp[j] = vtmp[j];
      /* printf(" (%d,%g),",bjtmp[j],batmp[j]); */
    }
    bi[i+1] = bi[i] + ncut;
    nzi = ncut + 1;
      
    /* apply level dropping rule to U part */
    ncut = nzi_au + dtcount; 
    if (ncut < nzi_bu){
      ierr = PetscSortSplit(ncut,nzi_bu,vtmp+nzi_bl+1,jtmp+nzi_bl+1);CHKERRQ(ierr);
      ierr = PetscSortIntWithScalarArray(ncut,jtmp+nzi_bl+1,vtmp+nzi_bl+1);CHKERRQ(ierr);
    } else {
      ncut = nzi_bu;
    }
    nzi += ncut;

    /* mark bdiagonal */
    bdiag[i+1]       = bdiag[i] - (ncut + 1); 
    bdiag_rev[n-i-1] = bdiag[i+1];
    bi[2*n - i]      = bi[2*n - i +1] - (ncut + 1);
    bjtmp = bj + bdiag[i];
    batmp = ba + bdiag[i];
    *bjtmp = i; 
    *batmp = diag_tmp; /* rtmp[i]; */
    if (*batmp == 0.0) {
      *batmp = dt+shift;
      /* printf(" row %d add shift %g\n",i,shift); */
    }
    *batmp = 1.0/(*batmp); /* invert diagonal entries for simplier triangular solves */
    /* printf(" (%d,%g),",*bjtmp,*batmp); */
    
    bjtmp = bj + bdiag[i+1]+1;
    batmp = ba + bdiag[i+1]+1;
    for (k=0; k<ncut; k++){
      bjtmp[k] = jtmp[nzi_bl+1+k];
      batmp[k] = vtmp[nzi_bl+1+k];
      /* printf(" (%d,%g),",bjtmp[k],batmp[k]); */
    }
    /* printf("\n"); */
      
    im[i]   = nzi; /* used by PetscLLAddSortedLU() */
    /*
    printf("row %d: bi %d, bdiag %d\n",i,bi[i],bdiag[i]);
    printf(" ----------------------------\n");
    */
  } /* for (i=0; i<n; i++) */
  /* printf("end of L %d, beginning of U %d\n",bi[n],bdiag[n]); */
  if (bi[n] >= bdiag[n]) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"end of L array %d cannot >= the beginning of U array %d",bi[n],bdiag[n]);

  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);

  ierr = PetscLLDestroy(lnk,lnkbt);CHKERRQ(ierr);
  ierr = PetscFree2(im,jtmp);CHKERRQ(ierr);
  ierr = PetscFree2(rtmp,vtmp);CHKERRQ(ierr);
  ierr = PetscFree(bdiag_rev);CHKERRQ(ierr);

  ierr = PetscLogFlops(B->cmap->n);CHKERRQ(ierr);
  b->maxnz = b->nz = bi[n] + bdiag[0] - bdiag[n]; 

  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&icol_identity);CHKERRQ(ierr);
  if (row_identity && icol_identity) {
    B->ops->solve = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    B->ops->solve = MatSolve_SeqAIJ;
  }
  
  B->ops->solveadd          = 0;
  B->ops->solvetranspose    = 0;
  B->ops->solvetransposeadd = 0;
  B->ops->matsolve          = 0;
  B->assembled              = PETSC_TRUE;
  B->preallocated           = PETSC_TRUE;
  PetscFunctionReturn(0); 
}

/* a wraper of MatILUDTFactor_SeqAIJ() */
#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactorSymbolic_SeqAIJ"
/*
    This will get a new name and become a varient of MatILUFactor_SeqAIJ() there is no longer seperate functions in the matrix function table for dt factors
*/

PetscErrorCode  MatILUDTFactorSymbolic_SeqAIJ(Mat fact,Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  ierr = MatILUDTFactor_SeqAIJ(A,row,col,info,&fact);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}

/* 
   same as MatLUFactorNumeric_SeqAIJ(), except using contiguous array matrix factors 
   - intend to replace existing MatLUFactorNumeric_SeqAIJ() 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUDTFactorNumeric_SeqAIJ"
/*
    This will get a new name and become a varient of MatILUFactor_SeqAIJ() there is no longer seperate functions in the matrix function table for dt factors
*/

PetscErrorCode  MatILUDTFactorNumeric_SeqAIJ(Mat fact,Mat A,const MatFactorInfo *info)
{
  Mat            C=fact;
  Mat_SeqAIJ     *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ *)C->data;
  IS             isrow = b->row,isicol = b->icol;
  PetscErrorCode ierr;
  const PetscInt *r,*ic,*ics;
  PetscInt       i,j,k,n=A->rmap->n,*ai=a->i,*aj=a->j,*bi=b->i,*bj=b->j;
  PetscInt       *ajtmp,*bjtmp,nz,nzl,nzu,row,*bdiag = b->diag,*pj;
  MatScalar      *rtmp,*pc,multiplier,*v,*pv,*aa=a->a;
  PetscReal      dt=info->dt,shift=info->shiftamount;
  PetscBool      row_identity, col_identity;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = PetscMalloc((n+1)*sizeof(MatScalar),&rtmp);CHKERRQ(ierr);
  ics  = ic;

  for (i=0; i<n; i++){     
    /* initialize rtmp array */
    nzl   = bi[i+1] - bi[i];       /* num of nozeros in L(i,:) */
    bjtmp = bj + bi[i];
    for  (j=0; j<nzl; j++) rtmp[*bjtmp++] = 0.0;
    rtmp[i] = 0.0; 
    nzu   = bdiag[i] - bdiag[i+1]; /* num of nozeros in U(i,:) */
    bjtmp = bj + bdiag[i+1] + 1;
    for  (j=0; j<nzu; j++) rtmp[*bjtmp++] = 0.0;     

    /* load in initial unfactored row of A */
    /* printf("row %d\n",i); */
    nz    = ai[r[i]+1] - ai[r[i]];
    ajtmp = aj + ai[r[i]];
    v     = aa + ai[r[i]];
    for (j=0; j<nz; j++) {
      rtmp[ics[*ajtmp++]] = v[j];
      /* printf(" (%d,%g),",ics[ajtmp[j]],rtmp[ics[ajtmp[j]]]); */
    }
    /* printf("\n"); */

    /* numerical factorization */
    bjtmp = bj + bi[i]; /* point to 1st entry of L(i,:) */
    nzl   = bi[i+1] - bi[i]; /* num of entries in L(i,:) */   
    k = 0;
    while (k < nzl){
      row   = *bjtmp++;
      /* printf("  prow %d\n",row); */
      pc         = rtmp + row;
      pv         = b->a + bdiag[row]; /* 1./(diag of the pivot row) */
      multiplier = (*pc) * (*pv); 
      *pc        = multiplier; 
      if (PetscAbsScalar(multiplier) > dt){ 
        pj         = bj + bdiag[row+1] + 1; /* point to 1st entry of U(row,:) */
        pv         = b->a + bdiag[row+1] + 1;
        nz         = bdiag[row] - bdiag[row+1] - 1; /* num of entries in U(row,:), excluding diagonal */
        for (j=0; j<nz; j++) rtmp[*pj++] -= multiplier * (*pv++);
        ierr = PetscLogFlops(1+2*nz);CHKERRQ(ierr);
      } 
      k++;
    }
    
    /* finished row so stick it into b->a */
    /* L-part */
    pv = b->a + bi[i] ;
    pj = bj + bi[i] ;
    nzl = bi[i+1] - bi[i];
    for (j=0; j<nzl; j++) {
      pv[j] = rtmp[pj[j]];
      /* printf(" (%d,%g),",pj[j],pv[j]); */
    }

    /* diagonal: invert diagonal entries for simplier triangular solves */
    if (rtmp[i] == 0.0) rtmp[i] = dt+shift;
    b->a[bdiag[i]] = 1.0/rtmp[i];
    /* printf(" (%d,%g),",i,b->a[bdiag[i]]); */

    /* U-part */
    pv = b->a + bdiag[i+1] + 1;
    pj = bj + bdiag[i+1] + 1;
    nzu = bdiag[i] - bdiag[i+1] - 1;
    for (j=0; j<nzu; j++) {
      pv[j] = rtmp[pj[j]];
      /* printf(" (%d,%g),",pj[j],pv[j]); */
    }
    /* printf("\n"); */
  } 

  ierr = PetscFree(rtmp);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
  
  ierr = ISIdentity(isrow,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(isicol,&col_identity);CHKERRQ(ierr);
  if (row_identity && col_identity) {
    C->ops->solve   = MatSolve_SeqAIJ_NaturalOrdering;
  } else {
    C->ops->solve   = MatSolve_SeqAIJ;
  }
  C->ops->solveadd           = 0;
  C->ops->solvetranspose     = 0;
  C->ops->solvetransposeadd  = 0;
  C->ops->matsolve           = 0;
  C->assembled    = PETSC_TRUE;
  C->preallocated = PETSC_TRUE;
  ierr = PetscLogFlops(C->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
}
