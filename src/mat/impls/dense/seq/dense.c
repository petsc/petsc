
/*
     Defines the basic matrix operations for sequential dense.
*/

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

#include <../src/mat/impls/aij/seq/aij.h>

static PetscErrorCode MatSeqDenseSymmetrize_Private(Mat A, PetscBool hermitian)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
  PetscInt      j, k, n = A->rmap->n;

  PetscFunctionBegin;
  if (A->rmap->n != A->cmap->n) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Cannot symmetrize a rectangular matrix");
  if (!hermitian) {
    for (k=0;k<n;k++) {
      for (j=k;j<n;j++) {
        mat->v[j*mat->lda + k] = mat->v[k*mat->lda + j];
      }
    }
  } else {
    for (k=0;k<n;k++) {
      for (j=k;j<n;j++) {
        mat->v[j*mat->lda + k] = PetscConj(mat->v[k*mat->lda + j]);
      }
    }
  }
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat A)
{
#if defined(PETSC_MISSING_LAPACK_POTRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
    if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
    if (!mat->fwork) {
      mat->lfwork = n;
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    PetscStackCallBLAS("LAPACKgetri",LAPACKgetri_(&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr); /* TODO CHECK FLOPS */
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
    if (A->spd) {
      PetscStackCallBLAS("LAPACKpotri",LAPACKpotri_("L",&n,mat->v,&mat->lda,&info));
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
#if defined (PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      PetscStackCallBLAS("LAPACKhetri",LAPACKhetri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_TRUE);CHKERRQ(ierr);
#endif
    } else { /* symmetric case */
      if (!mat->pivots) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Pivots not present");
      if (!mat->fwork) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Fwork not present");
      PetscStackCallBLAS("LAPACKsytri",LAPACKsytri_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&info));
      ierr = MatSeqDenseSymmetrize_Private(A,PETSC_FALSE);CHKERRQ(ierr);
    }
    if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad Inversion: zero pivot in row %D",(PetscInt)info-1);
    ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr); /* TODO CHECK FLOPS */
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
#endif

  A->ops->solve             = NULL;
  A->ops->matsolve          = NULL;
  A->ops->solvetranspose    = NULL;
  A->ops->matsolvetranspose = NULL;
  A->ops->solveadd          = NULL;
  A->ops->solvetransposeadd = NULL;
  A->factortype             = MAT_FACTOR_NONE;
  ierr                      = PetscFree(A->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroRowsColumns_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n,r = A->rmap->n, i,j;
  PetscScalar       *slot,*bb;
  const PetscScalar *xx;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<N; i++) {
    if (rows[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
    if (rows[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested to be zeroed greater than or equal number of rows %D",rows[i],A->rmap->n);
    if (rows[i] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Col %D requested to be zeroed greater than or equal number of cols %D",rows[i],A->cmap->n);
  }
#endif

  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  for (i=0; i<N; i++) {
    slot = l->v + rows[i]*m;
    ierr = PetscMemzero(slot,r*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<N; i++) {
    slot = l->v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = l->v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPNumeric_SeqDense_SeqDense(Mat A,Mat P,Mat C)
{
  Mat_SeqDense   *c = (Mat_SeqDense*)(C->data);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMatMultNumeric_SeqDense_SeqDense(A,P,c->ptapwork);CHKERRQ(ierr);
  ierr = MatTransposeMatMultNumeric_SeqDense_SeqDense(P,c->ptapwork,C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatPtAPSymbolic_SeqDense_SeqDense(Mat A,Mat P,PetscReal fill,Mat *C)
{
  Mat_SeqDense   *c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreateSeqDense(PetscObjectComm((PetscObject)A),P->cmap->N,P->cmap->N,NULL,C);CHKERRQ(ierr);
  c = (Mat_SeqDense*)((*C)->data);
  ierr = MatCreateSeqDense(PetscObjectComm((PetscObject)A),A->rmap->N,P->cmap->N,NULL,&c->ptapwork);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatPtAP_SeqDense_SeqDense(Mat A,Mat P,MatReuse reuse,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatPtAPSymbolic_SeqDense_SeqDense(A,P,fill,C);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  ierr = (*(*C)->ops->ptapnumeric)(A,P,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_PtAPNumeric,A,P,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqDense(Mat A,MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B = NULL;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat_SeqDense   *b;
  PetscErrorCode ierr;
  PetscInt       *ai=a->i,*aj=a->j,m=A->rmap->N,n=A->cmap->N,i;
  MatScalar      *av=a->a;
  PetscBool      isseqdense;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) {
    ierr = PetscObjectTypeCompare((PetscObject)*newmat,MATSEQDENSE,&isseqdense);CHKERRQ(ierr);
    if (!isseqdense) SETERRQ1(PetscObjectComm((PetscObject)*newmat),PETSC_ERR_USER,"Cannot reuse matrix of type %s",((PetscObject)(*newmat))->type);
  }
  if (reuse != MAT_REUSE_MATRIX) {
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
    ierr = MatSetSizes(B,m,n,m,n);CHKERRQ(ierr);
    ierr = MatSetType(B,MATSEQDENSE);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(B,NULL);CHKERRQ(ierr);
    b    = (Mat_SeqDense*)(B->data);
  } else {
    b    = (Mat_SeqDense*)((*newmat)->data);
    ierr = PetscMemzero(b->v,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  for (i=0; i<m; i++) {
    PetscInt j;
    for (j=0;j<ai[1]-ai[0];j++) {
      b->v[*aj*m+i] = *av;
      aj++;
      av++;
    }
    ai++;
  }

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    if (B) *newmat = B;
    ierr = MatAssemblyBegin(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i, j;
  PetscInt       *rows, *nnz;
  MatScalar      *aa = a->v, *vals;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = PetscCalloc3(A->rmap->n,&rows,A->rmap->n,&nnz,A->rmap->n,&vals);CHKERRQ(ierr);
  for (j=0; j<A->cmap->n; j++) {
    for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || i == j) ++nnz[i];
    aa += a->lda;
  }
  ierr = MatSeqAIJSetPreallocation(B,PETSC_DETERMINE,nnz);CHKERRQ(ierr);
  aa = a->v;
  for (j=0; j<A->cmap->n; j++) {
    PetscInt numRows = 0;
    for (i=0; i<A->rmap->n; i++) if (aa[i] != 0.0 || i == j) {rows[numRows] = i; vals[numRows++] = aa[i];}
    ierr = MatSetValues(B,numRows,rows,1,&j,vals,INSERT_VALUES);CHKERRQ(ierr);
    aa  += a->lda;
  }
  ierr = PetscFree3(rows,nnz,vals);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAXPY_SeqDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense   *x     = (Mat_SeqDense*)X->data,*y = (Mat_SeqDense*)Y->data;
  PetscScalar    oalpha = alpha;
  PetscInt       j;
  PetscBLASInt   N,m,ldax,lday,one = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(X->rmap->n*X->cmap->n,&N);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(X->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(x->lda,&ldax);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(y->lda,&lday);CHKERRQ(ierr);
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&m,&oalpha,x->v+j*ldax,&one,y->v+j*lday,&one));
    }
  } else {
    PetscStackCallBLAS("BLASaxpy",BLASaxpy_(&N,&oalpha,x->v,&one,y->v,&one));
  }
  ierr = PetscObjectStateIncrease((PetscObject)Y);CHKERRQ(ierr);
  ierr = PetscLogFlops(PetscMax(2*N-1,0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetInfo_SeqDense(Mat A,MatInfoType flag,MatInfo *info)
{
  PetscInt N = A->rmap->n*A->cmap->n;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = (double)N;
  info->nz_used           = (double)N;
  info->nz_unneeded       = (double)0;
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = 0;
  info->memory            = ((PetscObject)A)->mem;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_SeqDense(Mat A,PetscScalar alpha)
{
  Mat_SeqDense   *a     = (Mat_SeqDense*)A->data;
  PetscScalar    oalpha = alpha;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,j,nz,lda;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(a->lda,&lda);CHKERRQ(ierr);
  if (lda>A->rmap->n) {
    ierr = PetscBLASIntCast(A->rmap->n,&nz);CHKERRQ(ierr);
    for (j=0; j<A->cmap->n; j++) {
      PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&oalpha,a->v+j*lda,&one));
    }
  } else {
    ierr = PetscBLASIntCast(A->rmap->n*A->cmap->n,&nz);CHKERRQ(ierr);
    PetscStackCallBLAS("BLASscal",BLASscal_(&nz,&oalpha,a->v,&one));
  }
  ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatIsHermitian_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscInt     i,j,m = A->rmap->n,N;
  PetscScalar  *v = a->v;

  PetscFunctionBegin;
  *fl = PETSC_FALSE;
  if (A->rmap->n != A->cmap->n) PetscFunctionReturn(0);
  N = a->lda;

  for (i=0; i<m; i++) {
    for (j=i+1; j<m; j++) {
      if (PetscAbsScalar(v[i+j*N] - PetscConj(v[j+i*N])) > rtol) PetscFunctionReturn(0);
    }
  }
  *fl = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicateNoCreate_SeqDense(Mat newi,Mat A,MatDuplicateOption cpvalues)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data,*l;
  PetscErrorCode ierr;
  PetscInt       lda = (PetscInt)mat->lda,j,m;

  PetscFunctionBegin;
  ierr = PetscLayoutReference(A->rmap,&newi->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&newi->cmap);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(newi,NULL);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES) {
    l = (Mat_SeqDense*)newi->data;
    if (lda>A->rmap->n) {
      m = A->rmap->n;
      for (j=0; j<A->cmap->n; j++) {
        ierr = PetscMemcpy(l->v+j*m,mat->v+j*lda,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } else {
      ierr = PetscMemcpy(l->v,mat->v,A->rmap->n*A->cmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  newi->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*newmat,A,cpvalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


static PetscErrorCode MatLUFactor_SeqDense(Mat,IS,IS,const MatFactorInfo*);

static PetscErrorCode MatLUFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = MatLUFactor_SeqDense(fact,0,0,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscBLASInt      one = 1,info,m;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info));
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
#endif
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    if (A->spd) {
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
#if defined (PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      PetscStackCallBLAS("LAPACKhetrs",LAPACKhetrs_("L",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"HETRS Bad solve");
#endif
    } else { /* symmetric case */
      PetscStackCallBLAS("LAPACKsytrs",LAPACKsytrs_("L",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SYTRS Bad solve");
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_SeqDense(Mat A,Mat B,Mat X)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *b,*x;
  PetscInt       n;
  PetscBLASInt   nrhs,info,m;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  ierr = MatGetSize(B,NULL,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(n,&nrhs);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatDenseGetArray(X,&x);CHKERRQ(ierr);

  ierr = PetscMemcpy(x,b,m*nrhs*sizeof(PetscScalar));CHKERRQ(ierr);

  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("N",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
#endif
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    if (A->spd) {
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&m,&nrhs,mat->v,&mat->lda,x,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
#if defined (PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      PetscStackCallBLAS("LAPACKhetrs",LAPACKhetrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"HETRS Bad solve");
#endif
    } else { /* symmetric case */
      PetscStackCallBLAS("LAPACKsytrs",LAPACKsytrs_("L",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SYTRS Bad solve");
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");

  ierr = MatDenseRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(nrhs*(2.0*m*m - m));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *y;
  PetscBLASInt      one = 1,info,m;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    PetscStackCallBLAS("LAPACKgetrs",LAPACKgetrs_("T",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info));
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS - Bad solve");
#endif
  } else if (A->factortype == MAT_FACTOR_CHOLESKY) {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    if (A->spd) {
      PetscStackCallBLAS("LAPACKpotrs",LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
#if defined (PETSC_USE_COMPLEX)
    } else if (A->hermitian) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatSolveTranspose with Cholesky/Hermitian not available");
#endif
    } else { /* symmetric case */
      PetscStackCallBLAS("LAPACKsytrs",LAPACKsytrs_("L",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info));
      if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"SYTRS Bad solve");
    }
#endif
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *y,sone = 1.0;
  Vec               tmp = 0;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  }
  ierr = MatSolve_SeqDense(A,xx,yy);CHKERRQ(ierr);
  if (tmp) {
    ierr = VecAXPY(yy,sone,tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(yy,sone,zz);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolveTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscScalar       *y,sone = 1.0;
  Vec               tmp = NULL;

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  }
  ierr = MatSolveTranspose_SeqDense(A,xx,yy);CHKERRQ(ierr);
  if (tmp) {
    ierr = VecAXPY(yy,sone,tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(yy,sone,zz);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
static PetscErrorCode MatLUFactor_SeqDense(Mat A,IS row,IS col,const MatFactorInfo *minfo)
{
#if defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n,m,info;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  if (!mat->pivots) {
    ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  PetscStackCallBLAS("LAPACKgetrf",LAPACKgetrf_(&m,&n,mat->v,&mat->lda,mat->pivots,&info));
  ierr = PetscFPTrapPop();CHKERRQ(ierr);

  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");

  A->ops->solve             = MatSolve_SeqDense;
  A->ops->matsolve          = MatMatSolve_SeqDense;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense;
  A->ops->solveadd          = MatSolveAdd_SeqDense;
  A->ops->solvetransposeadd = MatSolveTransposeAdd_SeqDense;
  A->factortype             = MAT_FACTOR_LU;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&A->solvertype);CHKERRQ(ierr);

  ierr = PetscLogFlops((2.0*A->cmap->n*A->cmap->n*A->cmap->n)/3);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/* Cholesky as L*L^T or L*D*L^T and the symmetric/hermitian complex variants */
static PetscErrorCode MatCholeskyFactor_SeqDense(Mat A,IS perm,const MatFactorInfo *factinfo)
{
#if defined(PETSC_MISSING_LAPACK_POTRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   info,n;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (A->spd) {
    PetscStackCallBLAS("LAPACKpotrf",LAPACKpotrf_("L",&n,mat->v,&mat->lda,&info));
#if defined (PETSC_USE_COMPLEX)
  } else if (A->hermitian) {
    if (!mat->pivots) {
      ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    PetscStackCallBLAS("LAPACKhetrf",LAPACKhetrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
#endif
  } else { /* symmetric case */
    if (!mat->pivots) {
      ierr = PetscMalloc1(A->rmap->n,&mat->pivots);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    if (!mat->fwork) {
      PetscScalar dummy;

      mat->lfwork = -1;
      PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,&dummy,&mat->lfwork,&info));
      mat->lfwork = (PetscInt)PetscRealPart(dummy);
      ierr = PetscMalloc1(mat->lfwork,&mat->fwork);CHKERRQ(ierr);
      ierr = PetscLogObjectMemory((PetscObject)A,mat->lfwork*sizeof(PetscBLASInt));CHKERRQ(ierr);
    }
    PetscStackCallBLAS("LAPACKsytrf",LAPACKsytrf_("L",&n,mat->v,&mat->lda,mat->pivots,mat->fwork,&mat->lfwork,&info));
  }
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %D",(PetscInt)info-1);

  A->ops->solve             = MatSolve_SeqDense;
  A->ops->matsolve          = MatMatSolve_SeqDense;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense;
  A->ops->solveadd          = MatSolveAdd_SeqDense;
  A->ops->solvetransposeadd = MatSolveTransposeAdd_SeqDense;
  A->factortype             = MAT_FACTOR_CHOLESKY;

  ierr = PetscFree(A->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&A->solvertype);CHKERRQ(ierr);

  ierr = PetscLogFlops((1.0*A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


PetscErrorCode MatCholeskyFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  PetscErrorCode ierr;
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;

  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = MatCholeskyFactor_SeqDense(fact,0,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCholeskyFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->assembled                  = PETSC_TRUE;
  fact->preallocated               = PETSC_TRUE;
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqDense;
  fact->ops->solve                 = MatSolve_SeqDense;
  fact->ops->matsolve              = MatMatSolve_SeqDense;
  fact->ops->solvetranspose        = MatSolveTranspose_SeqDense;
  fact->ops->solveadd              = MatSolveAdd_SeqDense;
  fact->ops->solvetransposeadd     = MatSolveTransposeAdd_SeqDense;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->preallocated           = PETSC_TRUE;
  fact->assembled              = PETSC_TRUE;
  fact->ops->lufactornumeric   = MatLUFactorNumeric_SeqDense;
  fact->ops->solve             = MatSolve_SeqDense;
  fact->ops->matsolve          = MatMatSolve_SeqDense;
  fact->ops->solvetranspose    = MatSolveTranspose_SeqDense;
  fact->ops->solveadd          = MatSolveAdd_SeqDense;
  fact->ops->solvetransposeadd = MatSolveTransposeAdd_SeqDense;
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatGetFactor_seqdense_petsc(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU) {
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;

  ierr = PetscFree((*fact)->solvertype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(MATSOLVERPETSC,&(*fact)->solvertype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------*/
static PetscErrorCode MatSOR_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *x,*v = mat->v,zero = 0.0,xt;
  const PetscScalar *b;
  PetscErrorCode    ierr;
  PetscInt          m = A->rmap->n,i;
  PetscBLASInt      o = 1,bm;

  PetscFunctionBegin;
  if (shift == -1) shift = 0.0; /* negative shift indicates do not error on zero diagonal; this code never zeros on zero diagonal */
  ierr = PetscBLASIntCast(m,&bm);CHKERRQ(ierr);
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdot */
    ierr = VecSet(xx,zero);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArrayRead(bb,&b);CHKERRQ(ierr);
  its  = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        PetscStackCallBLAS("BLASdot",xt   = b[i] - BLASdot_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i=m-1; i>=0; i--) {
        PetscStackCallBLAS("BLASdot",xt   = b[i] - BLASdot_(&bm,v+i,&bm,x,&o));
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
  }
  ierr = VecRestoreArrayRead(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
PETSC_INTERN PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v   = mat->v,*x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n,_One=1;
  PetscScalar       _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  PetscScalar       *y,_DOne=1.0,_DZero=0.0;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;
  const PetscScalar *v = mat->v,*x;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->rmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y,_DOne=1.0;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *v = mat->v,*x;
  PetscScalar       *y;
  PetscErrorCode    ierr;
  PetscBLASInt      m, n, _One=1;
  PetscScalar       _DOne=1.0;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&n);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemv",BLASgemv_("T",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One));
  ierr = VecRestoreArrayRead(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
static PetscErrorCode MatGetRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  *ncols = A->cmap->n;
  if (cols) {
    ierr = PetscMalloc1(A->cmap->n+1,cols);CHKERRQ(ierr);
    for (i=0; i<A->cmap->n; i++) (*cols)[i] = i;
  }
  if (vals) {
    ierr = PetscMalloc1(A->cmap->n+1,vals);CHKERRQ(ierr);
    v    = mat->v + row;
    for (i=0; i<A->cmap->n; i++) {(*vals)[i] = *v; v += mat->lda;}
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRestoreRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (cols) {ierr = PetscFree(*cols);CHKERRQ(ierr);}
  if (vals) {ierr = PetscFree(*vals);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/
static PetscErrorCode MatSetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],const PetscScalar v[],InsertMode addv)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
  PetscInt     i,j,idx=0;

  PetscFunctionBegin;
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
#if defined(PETSC_USE_DEBUG)
        if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
#endif
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
#if defined(PETSC_USE_DEBUG)
          if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
#if defined(PETSC_USE_DEBUG)
        if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
#endif
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
#if defined(PETSC_USE_DEBUG)
          if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] += v[idx++];
        }
      }
    }
  } else {
    if (addv == INSERT_VALUES) {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
#if defined(PETSC_USE_DEBUG)
        if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
#endif
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
#if defined(PETSC_USE_DEBUG)
          if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
#if defined(PETSC_USE_DEBUG)
        if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->rmap->n-1);
#endif
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
#if defined(PETSC_USE_DEBUG)
          if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->cmap->n-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] += v[idx++];
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],PetscScalar v[])
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
  PetscInt     i,j;

  PetscFunctionBegin;
  /* row-oriented output */
  for (i=0; i<m; i++) {
    if (indexm[i] < 0) {v += n;continue;}
    if (indexm[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested larger than number rows %D",indexm[i],A->rmap->n);
    for (j=0; j<n; j++) {
      if (indexn[j] < 0) {v++; continue;}
      if (indexn[j] >= A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Column %D requested larger than number columns %D",indexn[j],A->cmap->n);
      *v++ = mat->v[indexn[j]*mat->lda + indexm[i]];
    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

static PetscErrorCode MatLoad_SeqDense(Mat newmat,PetscViewer viewer)
{
  Mat_SeqDense   *a;
  PetscErrorCode ierr;
  PetscInt       *scols,i,j,nz,header[4];
  int            fd;
  PetscMPIInt    size;
  PetscInt       *rowlengths = 0,M,N,*cols,grows,gcols;
  PetscScalar    *vals,*svals,*v,*w;
  MPI_Comm       comm;

  PetscFunctionBegin;
  /* force binary viewer to load .info file if it has not yet done so */
  ierr = PetscViewerSetUp(viewer);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_CLASSID) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  /* set global size if not set already*/
  if (newmat->rmap->n < 0 && newmat->rmap->N < 0 && newmat->cmap->n < 0 && newmat->cmap->N < 0) {
    ierr = MatSetSizes(newmat,M,N,M,N);CHKERRQ(ierr);
  } else {
    /* if sizes and type are already set, check if the vector global sizes are correct */
    ierr = MatGetSize(newmat,&grows,&gcols);CHKERRQ(ierr);
    if (M != grows ||  N != gcols) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED, "Matrix in file of different length (%d, %d) than the input matrix (%d, %d)",M,N,grows,gcols);
  }
  a = (Mat_SeqDense*)newmat->data;
  if (!a->user_alloc) {
    ierr = MatSeqDenseSetPreallocation(newmat,NULL);CHKERRQ(ierr);
  }

  if (nz == MATRIX_BINARY_FORMAT_DENSE) { /* matrix in file is dense */
    a = (Mat_SeqDense*)newmat->data;
    v = a->v;
    /* Allocate some temp space to read in the values and then flip them
       from row major to column major */
    ierr = PetscMalloc1(M*N > 0 ? M*N : 1,&w);CHKERRQ(ierr);
    /* read in nonzero values */
    ierr = PetscBinaryRead(fd,w,M*N,PETSC_SCALAR);CHKERRQ(ierr);
    /* now flip the values and store them in the matrix*/
    for (j=0; j<N; j++) {
      for (i=0; i<M; i++) {
        *v++ =w[i*N+j];
      }
    }
    ierr = PetscFree(w);CHKERRQ(ierr);
  } else {
    /* read row lengths */
    ierr = PetscMalloc1(M+1,&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

    a = (Mat_SeqDense*)newmat->data;
    v = a->v;

    /* read column indices and nonzeros */
    ierr = PetscMalloc1(nz+1,&scols);CHKERRQ(ierr);
    cols = scols;
    ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc1(nz+1,&svals);CHKERRQ(ierr);
    vals = svals;
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);

    /* insert into matrix */
    for (i=0; i<M; i++) {
      for (j=0; j<rowlengths[i]; j++) v[i+M*scols[j]] = svals[j];
      svals += rowlengths[i]; scols += rowlengths[i];
    }
    ierr = PetscFree(vals);CHKERRQ(ierr);
    ierr = PetscFree(cols);CHKERRQ(ierr);
    ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  const char        *name;
  PetscScalar       *v;
  PetscViewerFormat format;
#if defined(PETSC_USE_COMPLEX)
  PetscBool         allreal = PETSC_TRUE;
#endif

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
    for (i=0; i<A->rmap->n; i++) {
      v    = a->v + i;
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(*v) != 0.0 && PetscImaginaryPart(*v) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",j,(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v));CHKERRQ(ierr);
        } else if (PetscRealPart(*v)) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,(double)PetscRealPart(*v));CHKERRQ(ierr);
        }
#else
        if (*v) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,(double)*v);CHKERRQ(ierr);
        }
#endif
        v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_FALSE);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    /* determine if matrix has all real values */
    v = a->v;
    for (i=0; i<A->rmap->n*A->cmap->n; i++) {
      if (PetscImaginaryPart(v[i])) { allreal = PETSC_FALSE; break;}
    }
#endif
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = zeros(%D,%D);\n",name,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    }

    for (i=0; i<A->rmap->n; i++) {
      v = a->v + i;
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (allreal) {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)PetscRealPart(*v));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"%18.16e + %18.16ei ",(double)PetscRealPart(*v),(double)PetscImaginaryPart(*v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%18.16e ",(double)*v);CHKERRQ(ierr);
#endif
        v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  int               fd;
  PetscInt          ict,j,n = A->cmap->n,m = A->rmap->n,i,*col_lens,nz = m*n;
  PetscScalar       *v,*anonz,*vals;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_NATIVE) {
    /* store the matrix as a dense matrix */
    ierr = PetscMalloc1(4,&col_lens);CHKERRQ(ierr);

    col_lens[0] = MAT_FILE_CLASSID;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = MATRIX_BINARY_FORMAT_DENSE;

    ierr = PetscBinaryWrite(fd,col_lens,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* write out matrix, by rows */
    ierr = PetscMalloc1(m*n+1,&vals);CHKERRQ(ierr);
    v    = a->v;
    for (j=0; j<n; j++) {
      for (i=0; i<m; i++) {
        vals[j + i*n] = *v++;
      }
    }
    ierr = PetscBinaryWrite(fd,vals,n*m,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc1(4+nz,&col_lens);CHKERRQ(ierr);

    col_lens[0] = MAT_FILE_CLASSID;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = nz;

    /* store lengths of each row and write (including header) to file */
    for (i=0; i<m; i++) col_lens[4+i] = n;
    ierr = PetscBinaryWrite(fd,col_lens,4+m,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);

    /* Possibly should write in smaller increments, not whole matrix at once? */
    /* store column indices (zero start index) */
    ict = 0;
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) col_lens[ict++] = j;
    }
    ierr = PetscBinaryWrite(fd,col_lens,nz,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* store nonzero values */
    ierr = PetscMalloc1(nz+1,&anonz);CHKERRQ(ierr);
    ict  = 0;
    for (i=0; i<m; i++) {
      v = a->v + i;
      for (j=0; j<n; j++) {
        anonz[ict++] = *v; v += a->lda;
      }
    }
    ierr = PetscBinaryWrite(fd,anonz,nz,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(anonz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include <petscdraw.h>
static PetscErrorCode MatView_SeqDense_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A  = (Mat) Aa;
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          m  = A->rmap->n,n = A->cmap->n,i,j;
  int               color = PETSC_DRAW_WHITE;
  PetscScalar       *v = a->v;
  PetscViewer       viewer;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* Loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    /* Blue for negative and Red for positive */
    for (j = 0; j < n; j++) {
      x_l = j; x_r = x_l + 1.0;
      for (i = 0; i < m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
        if (PetscRealPart(v[j*m+i]) >  0.) {
          color = PETSC_DRAW_RED;
        } else if (PetscRealPart(v[j*m+i]) <  0.) {
          color = PETSC_DRAW_BLUE;
        } else {
          continue;
        }
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    PetscReal minv = 0.0, maxv = 0.0;
    PetscDraw popup;

    for (i=0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    if (minv >= maxv) maxv = minv + PETSC_SMALL;
    ierr = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    ierr = PetscDrawScalePopup(popup,minv,maxv);CHKERRQ(ierr);

    ierr = PetscDrawCollectiveBegin(draw);CHKERRQ(ierr);
    for (j=0; j<n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for (i=0; i<m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
        color = PetscDrawRealToColor(PetscAbsScalar(v[j*m+i]),minv,maxv);
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
    ierr = PetscDrawCollectiveEnd(draw);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_SeqDense_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  xr   = A->cmap->n; yr = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr  += w;          yr += h;        xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqDense_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",NULL);CHKERRQ(ierr);
  ierr = PetscDrawSave(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SeqDense(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool      iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERDRAW,&isdraw);CHKERRQ(ierr);

  if (iascii) {
    ierr = MatView_SeqDense_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqDense_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqDense_Draw(A,viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDensePlaceArray_SeqDense(Mat A,const PetscScalar array[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  a->unplacedarray       = a->v;
  a->unplaced_user_alloc = a->user_alloc;
  a->v                   = (PetscScalar*) array;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseResetArray_SeqDense(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  a->v             = a->unplacedarray;
  a->user_alloc    = a->unplaced_user_alloc;
  a->unplacedarray = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_SeqDense(Mat mat)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows %D Cols %D",mat->rmap->n,mat->cmap->n);
#endif
  ierr = PetscFree(l->pivots);CHKERRQ(ierr);
  ierr = PetscFree(l->fwork);CHKERRQ(ierr);
  ierr = MatDestroy(&l->ptapwork);CHKERRQ(ierr);
  if (!l->user_alloc) {ierr = PetscFree(l->v);CHKERRQ(ierr);}
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseGetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDensePlaceArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseResetArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatDenseRestoreArray_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_seqaij_C",NULL);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatConvert_seqdense_elemental_C",NULL);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatSeqDenseSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMult_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMultSymbolic_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatMatMultNumeric_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatPtAP_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatTransposeMatMult_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatTransposeMatMultSymbolic_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)mat,"MatTransposeMatMultNumeric_seqaij_seqdense_C",NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatTranspose_SeqDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       k,j,m,n,M;
  PetscScalar    *v,tmp;

  PetscFunctionBegin;
  v = mat->v; m = A->rmap->n; M = mat->lda; n = A->cmap->n;
  if (reuse == MAT_INPLACE_MATRIX) { /* in place transpose */
    if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can not transpose non-square matrix in place");
    else {
      for (j=0; j<m; j++) {
        for (k=0; k<j; k++) {
          tmp        = v[j + k*M];
          v[j + k*M] = v[k + j*M];
          v[k + j*M] = tmp;
        }
      }
    }
  } else { /* out-of-place transpose */
    Mat          tmat;
    Mat_SeqDense *tmatd;
    PetscScalar  *v2;
    PetscInt     M2;

    if (reuse == MAT_INITIAL_MATRIX) {
      ierr = MatCreate(PetscObjectComm((PetscObject)A),&tmat);CHKERRQ(ierr);
      ierr = MatSetSizes(tmat,A->cmap->n,A->rmap->n,A->cmap->n,A->rmap->n);CHKERRQ(ierr);
      ierr = MatSetType(tmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr = MatSeqDenseSetPreallocation(tmat,NULL);CHKERRQ(ierr);
    } else {
      tmat = *matout;
    }
    tmatd = (Mat_SeqDense*)tmat->data;
    v = mat->v; v2 = tmatd->v; M2 = tmatd->lda;
    for (j=0; j<n; j++) {
      for (k=0; k<m; k++) v2[j + k*M2] = v[k + j*M];
    }
    ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    *matout = tmat;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatEqual_SeqDense(Mat A1,Mat A2,PetscBool  *flg)
{
  Mat_SeqDense *mat1 = (Mat_SeqDense*)A1->data;
  Mat_SeqDense *mat2 = (Mat_SeqDense*)A2->data;
  PetscInt     i,j;
  PetscScalar  *v1,*v2;

  PetscFunctionBegin;
  if (A1->rmap->n != A2->rmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (A1->cmap->n != A2->cmap->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  for (i=0; i<A1->rmap->n; i++) {
    v1 = mat1->v+i; v2 = mat2->v+i;
    for (j=0; j<A1->cmap->n; j++) {
      if (*v1 != *v2) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
      v1 += mat1->lda; v2 += mat2->lda;
    }
  }
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,n,len;
  PetscScalar    *x,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  len  = PetscMin(A->rmap->n,A->cmap->n);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  for (i=0; i<len; i++) {
    x[i] = mat->v[i*mat->lda + i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense      *mat = (Mat_SeqDense*)A->data;
  const PetscScalar *l,*r;
  PetscScalar       x,*v;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->rmap->n,n = A->cmap->n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    ierr = VecGetArrayRead(ll,&l);CHKERRQ(ierr);
    if (m != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vec wrong size");
    for (i=0; i<m; i++) {
      x = l[i];
      v = mat->v + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= mat->lda;}
    }
    ierr = VecRestoreArrayRead(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    ierr = VecGetArrayRead(rr,&r);CHKERRQ(ierr);
    if (n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec wrong size");
    for (i=0; i<n; i++) {
      x = r[i];
      v = mat->v + i*mat->lda;
      for (j=0; j<m; j++) (*v++) *= x;
    }
    ierr = VecRestoreArrayRead(rr,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(1.0*n*m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatNorm_SeqDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v   = mat->v;
  PetscReal      sum  = 0.0;
  PetscInt       lda  =mat->lda,m=A->rmap->n,i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    if (lda>m) {
      for (j=0; j<A->cmap->n; j++) {
        v = mat->v+j*lda;
        for (i=0; i<m; i++) {
          sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
        }
      }
    } else {
#if defined(PETSC_USE_REAL___FP16)
      PetscBLASInt one = 1,cnt = A->cmap->n*A->rmap->n;
      *nrm = BLASnrm2_(&cnt,v,&one);
    }
#else
      for (i=0; i<A->cmap->n*A->rmap->n; i++) {
        sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
      }
    }
    *nrm = PetscSqrtReal(sum);
#endif
    ierr = PetscLogFlops(2.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    *nrm = 0.0;
    for (j=0; j<A->cmap->n; j++) {
      v   = mat->v + j*mat->lda;
      sum = 0.0;
      for (i=0; i<A->rmap->n; i++) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(1.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->rmap->n; j++) {
      v   = mat->v + j;
      sum = 0.0;
      for (i=0; i<A->cmap->n; i++) {
        sum += PetscAbsScalar(*v); v += mat->lda;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(1.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetOption_SeqDense(Mat A,MatOption op,PetscBool flg)
{
  Mat_SeqDense   *aij = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    aij->roworiented = flg;
    break;
  case MAT_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
  case MAT_NEW_DIAGONALS:
  case MAT_KEEP_NONZERO_PATTERN:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
  case MAT_IGNORE_ZERO_ENTRIES:    
  case MAT_IGNORE_LOWER_TRIANGULAR:
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unknown option %s",MatOptions[op]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda=l->lda,m=A->rmap->n,j;

  PetscFunctionBegin;
  if (lda>m) {
    for (j=0; j<A->cmap->n; j++) {
      ierr = PetscMemzero(l->v+j*lda,m*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMemzero(l->v,A->rmap->n*A->cmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroRows_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m  = l->lda, n = A->cmap->n, i,j;
  PetscScalar       *slot,*bb;
  const PetscScalar *xx;

  PetscFunctionBegin;
#if defined(PETSC_USE_DEBUG)
  for (i=0; i<N; i++) {
    if (rows[i] < 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Negative row requested to be zeroed");
    if (rows[i] >= A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Row %D requested to be zeroed greater than or equal number of rows %D",rows[i],A->rmap->n);
  }
#endif

  /* fix right hand side if needed */
  if (x && b) {
    ierr = VecGetArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecGetArray(b,&bb);CHKERRQ(ierr);
    for (i=0; i<N; i++) bb[rows[i]] = diag*xx[rows[i]];
    ierr = VecRestoreArrayRead(x,&xx);CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&bb);CHKERRQ(ierr);
  }

  for (i=0; i<N; i++) {
    slot = l->v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += m;}
  }
  if (diag != 0.0) {
    if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Only coded for square matrices");
    for (i=0; i<N; i++) {
      slot  = l->v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseGetArray_SeqDense(Mat A,PetscScalar *array[])
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (mat->lda != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get array for Dense matrices with LDA different from number of rows");
  *array = mat->v;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDenseRestoreArray_SeqDense(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = 0; /* user cannot accidently use the array later */
  PetscFunctionReturn(0);
}

/*@C
   MatDenseGetArray - gives access to the array where the data for a SeqDense matrix is stored

   Not Collective

   Input Parameter:
.  mat - a MATSEQDENSE or MATMPIDENSE matrix

   Output Parameter:
.   array - pointer to the data

   Level: intermediate

.seealso: MatDenseRestoreArray()
@*/
PetscErrorCode  MatDenseGetArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatDenseGetArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatDenseRestoreArray - returns access to the array where the data for a dense matrix is stored obtained by MatDenseGetArray()

   Not Collective

   Input Parameters:
.  mat - a MATSEQDENSE or MATMPIDENSE matrix
.  array - pointer to the data

   Level: intermediate

.seealso: MatDenseGetArray()
@*/
PetscErrorCode  MatDenseRestoreArray(Mat A,PetscScalar **array)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscUseMethod(A,"MatDenseRestoreArray_C",(Mat,PetscScalar**),(A,array));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,nrows,ncols;
  const PetscInt *irow,*icol;
  PetscScalar    *av,*bv,*v = mat->v;
  Mat            newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  /* Check submatrixcall */
  if (scall == MAT_REUSE_MATRIX) {
    PetscInt n_cols,n_rows;
    ierr = MatGetSize(*B,&n_rows,&n_cols);CHKERRQ(ierr);
    if (n_rows != nrows || n_cols != ncols) {
      /* resize the result matrix to match number of requested rows/columns */
      ierr = MatSetSizes(*B,nrows,ncols,nrows,ncols);CHKERRQ(ierr);
    }
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(PetscObjectComm((PetscObject)A),&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,ncols,nrows,ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(newmat,NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy,column at a time */
  bv = ((Mat_SeqDense*)newmat->data)->v;

  for (i=0; i<ncols; i++) {
    av = v + mat->lda*icol[i];
    for (j=0; j<nrows; j++) *bv++ = av[irow[j]];
  }

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  *B   = newmat;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCreateSubMatrices_SeqDense(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscCalloc1(n+1,B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatCreateSubMatrix_SeqDense(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyBegin_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_SeqDense(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense*)B->data;
  PetscErrorCode ierr;
  PetscInt       lda1=a->lda,lda2=b->lda, m=A->rmap->n,n=A->cmap->n, j;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (m != B->rmap->n || n != B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      ierr = PetscMemcpy(b->v+j*lda2,a->v+j*lda1,m*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMemcpy(b->v,a->v,A->rmap->n*A->cmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  ierr = PetscObjectStateIncrease((PetscObject)B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUp_SeqDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqDenseSetPreallocation(A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatConjugate_SeqDense(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscInt     i,nz = A->rmap->n*A->cmap->n;
  PetscScalar  *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscConj(aa[i]);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatRealPart_SeqDense(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscInt     i,nz = A->rmap->n*A->cmap->n;
  PetscScalar  *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatImaginaryPart_SeqDense(Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscInt     i,nz = A->rmap->n*A->cmap->n;
  PetscScalar  *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
PETSC_INTERN PetscErrorCode MatMatMult_SeqDense_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqDense_SeqDense(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatMultNumeric_SeqDense_SeqDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=B->cmap->n;
  Mat            Cmat;

  PetscFunctionBegin;
  if (A->cmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"A->cmap->n %d != B->rmap->n %d\n",A->cmap->n,B->rmap->n);
  ierr = MatCreate(PETSC_COMM_SELF,&Cmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Cmat,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Cmat,NULL);CHKERRQ(ierr);

  *C = Cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense   *c = (Mat_SeqDense*)C->data;
  PetscBLASInt   m,n,k;
  PetscScalar    _DOne=1.0,_DZero=0.0;
  PetscErrorCode ierr;
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)B,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Second matrix must be dense");

  /* Handle case where where user provided the final C matrix rather than calling MatMatMult() with MAT_INITIAL_MATRIX*/
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    C->ops->matmultnumeric = MatMatMultNumeric_SeqAIJ_SeqDense;
    ierr = (*C->ops->matmultnumeric)(A,B,C);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQDENSE,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"First matrix must be dense");
  ierr = PetscBLASIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","N",&m,&n,&k,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMult_SeqDense_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatTransposeMultSymbolic_SeqDense_SeqDense(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatTransposeMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatMatTransposeMultNumeric_SeqDense_SeqDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatTransposeMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->rmap->n,n=B->rmap->n;
  Mat            Cmat;

  PetscFunctionBegin;
  if (A->cmap->n != B->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"A->cmap->n %d != B->cmap->n %d\n",A->cmap->n,B->cmap->n);
  ierr = MatCreate(PETSC_COMM_SELF,&Cmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Cmat,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Cmat,NULL);CHKERRQ(ierr);

  Cmat->assembled = PETSC_TRUE;

  *C = Cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMatTransposeMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense   *c = (Mat_SeqDense*)C->data;
  PetscBLASInt   m,n,k;
  PetscScalar    _DOne=1.0,_DZero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(A->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(B->rmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->cmap->n,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("N","T",&m,&n,&k,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda));
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMult_SeqDense_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscLogEventBegin(MAT_TransposeMatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatTransposeMatMultSymbolic_SeqDense_SeqDense(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_TransposeMatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = MatTransposeMatMultNumeric_SeqDense_SeqDense(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_TransposeMatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultSymbolic_SeqDense_SeqDense(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;
  PetscInt       m=A->cmap->n,n=B->cmap->n;
  Mat            Cmat;

  PetscFunctionBegin;
  if (A->rmap->n != B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"A->rmap->n %d != B->rmap->n %d\n",A->rmap->n,B->rmap->n);
  ierr = MatCreate(PETSC_COMM_SELF,&Cmat);CHKERRQ(ierr);
  ierr = MatSetSizes(Cmat,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(Cmat,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Cmat,NULL);CHKERRQ(ierr);

  Cmat->assembled = PETSC_TRUE;

  *C = Cmat;
  PetscFunctionReturn(0);
}

PetscErrorCode MatTransposeMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense   *c = (Mat_SeqDense*)C->data;
  PetscBLASInt   m,n,k;
  PetscScalar    _DOne=1.0,_DZero=0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscBLASIntCast(C->rmap->n,&m);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(C->cmap->n,&n);CHKERRQ(ierr);
  ierr = PetscBLASIntCast(A->rmap->n,&k);CHKERRQ(ierr);
  PetscStackCallBLAS("BLASgemm",BLASgemm_("T","N",&m,&n,&k,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMax_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) < PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMaxAbs_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  PetscReal      atmp;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = PetscAbsScalar(aa[i]);
    for (j=1; j<n; j++) {
      atmp = PetscAbsScalar(aa[i+m*j]);
      if (PetscAbsScalar(x[i]) < atmp) {x[i] = atmp; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetRowMin_SeqDense(Mat A,Vec v,PetscInt idx[])
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n,p;
  PetscScalar    *x;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecSet(v,0.0);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&p);CHKERRQ(ierr);
  if (p != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<m; i++) {
    x[i] = aa[i]; if (idx) idx[i] = 0;
    for (j=1; j<n; j++) {
      if (PetscRealPart(x[i]) > PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetColumnVector_SeqDense(Mat A,Vec v,PetscInt col)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x;

  PetscFunctionBegin;
  if (A->factortype) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");

  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(x,a->v+col*a->lda,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode MatGetColumnNorms_SeqDense(Mat A,NormType type,PetscReal *norms)
{
  PetscErrorCode ierr;
  PetscInt       i,j,m,n;
  PetscScalar    *a;

  PetscFunctionBegin;
  ierr = MatGetSize(A,&m,&n);CHKERRQ(ierr);
  ierr = PetscMemzero(norms,n*sizeof(PetscReal));CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] += PetscAbsScalar(a[j]*a[j]);
      }
      a += m;
    }
  } else if (type == NORM_1) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] += PetscAbsScalar(a[j]);
      }
      a += m;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<n; i++) {
      for (j=0; j<m; j++) {
        norms[i] = PetscMax(PetscAbsScalar(a[j]),norms[i]);
      }
      a += m;
    }
  } else SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Unknown NormType");
  ierr = MatDenseRestoreArray(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = PetscSqrtReal(norms[i]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  MatSetRandom_SeqDense(Mat x,PetscRandom rctx)
{
  PetscErrorCode ierr;
  PetscScalar    *a;
  PetscInt       m,n,i;

  PetscFunctionBegin;
  ierr = MatGetSize(x,&m,&n);CHKERRQ(ierr);
  ierr = MatDenseGetArray(x,&a);CHKERRQ(ierr);
  for (i=0; i<m*n; i++) {
    ierr = PetscRandomGetValue(rctx,a+i);CHKERRQ(ierr);
  }
  ierr = MatDenseRestoreArray(x,&a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMissingDiagonal_SeqDense(Mat A,PetscBool  *missing,PetscInt *d)
{
  PetscFunctionBegin;
  *missing = PETSC_FALSE;
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = { MatSetValues_SeqDense,
                                        MatGetRow_SeqDense,
                                        MatRestoreRow_SeqDense,
                                        MatMult_SeqDense,
                                /*  4*/ MatMultAdd_SeqDense,
                                        MatMultTranspose_SeqDense,
                                        MatMultTransposeAdd_SeqDense,
                                        0,
                                        0,
                                        0,
                                /* 10*/ 0,
                                        MatLUFactor_SeqDense,
                                        MatCholeskyFactor_SeqDense,
                                        MatSOR_SeqDense,
                                        MatTranspose_SeqDense,
                                /* 15*/ MatGetInfo_SeqDense,
                                        MatEqual_SeqDense,
                                        MatGetDiagonal_SeqDense,
                                        MatDiagonalScale_SeqDense,
                                        MatNorm_SeqDense,
                                /* 20*/ MatAssemblyBegin_SeqDense,
                                        MatAssemblyEnd_SeqDense,
                                        MatSetOption_SeqDense,
                                        MatZeroEntries_SeqDense,
                                /* 24*/ MatZeroRows_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 29*/ MatSetUp_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 34*/ MatDuplicate_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 39*/ MatAXPY_SeqDense,
                                        MatCreateSubMatrices_SeqDense,
                                        0,
                                        MatGetValues_SeqDense,
                                        MatCopy_SeqDense,
                                /* 44*/ MatGetRowMax_SeqDense,
                                        MatScale_SeqDense,
                                        MatShift_Basic,
                                        0,
                                        MatZeroRowsColumns_SeqDense,
                                /* 49*/ MatSetRandom_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 54*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 59*/ 0,
                                        MatDestroy_SeqDense,
                                        MatView_SeqDense,
                                        0,
                                        0,
                                /* 64*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 69*/ MatGetRowMaxAbs_SeqDense,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 74*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /* 79*/ 0,
                                        0,
                                        0,
                                        0,
                                /* 83*/ MatLoad_SeqDense,
                                        0,
                                        MatIsHermitian_SeqDense,
                                        0,
                                        0,
                                        0,
                                /* 89*/ MatMatMult_SeqDense_SeqDense,
                                        MatMatMultSymbolic_SeqDense_SeqDense,
                                        MatMatMultNumeric_SeqDense_SeqDense,
                                        MatPtAP_SeqDense_SeqDense,
                                        MatPtAPSymbolic_SeqDense_SeqDense,
                                /* 94*/ MatPtAPNumeric_SeqDense_SeqDense,
                                        MatMatTransposeMult_SeqDense_SeqDense,
                                        MatMatTransposeMultSymbolic_SeqDense_SeqDense,
                                        MatMatTransposeMultNumeric_SeqDense_SeqDense,
                                        0,
                                /* 99*/ 0,
                                        0,
                                        0,
                                        MatConjugate_SeqDense,
                                        0,
                                /*104*/ 0,
                                        MatRealPart_SeqDense,
                                        MatImaginaryPart_SeqDense,
                                        0,
                                        0,
                                /*109*/ 0,
                                        0,
                                        MatGetRowMin_SeqDense,
                                        MatGetColumnVector_SeqDense,
                                        MatMissingDiagonal_SeqDense,
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
                                        MatGetColumnNorms_SeqDense,
                                        0,
                                        0,
                                        0,
                                /*129*/ 0,
                                        MatTransposeMatMult_SeqDense_SeqDense,
                                        MatTransposeMatMultSymbolic_SeqDense_SeqDense,
                                        MatTransposeMatMultNumeric_SeqDense_SeqDense,
                                        0,
                                /*134*/ 0,
                                        0,
                                        0,
                                        0,
                                        0,
                                /*139*/ 0,
                                        0,
                                        0
};

/*@C
   MatCreateSeqDense - Creates a sequential dense matrix that
   is stored in column major order (the usual Fortran 77 manner). Many
   of the matrix operations use the BLAS and LAPACK routines.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
-  data - optional location of matrix data in column major order.  Set data=NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=NULL.

   Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateDense(), MatSetValues()
@*/
PetscErrorCode  MatCreateSeqDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,m,n);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   MatSeqDenseSetPreallocation - Sets the array used for storing the matrix elements

   Collective on MPI_Comm

   Input Parameters:
+  B - the matrix
-  data - the array (or NULL)

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   need not call this routine.

   Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateDense(), MatSetValues(), MatSeqDenseSetLDA()

@*/
PetscErrorCode  MatSeqDenseSetPreallocation(Mat B,PetscScalar data[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(B,"MatSeqDenseSetPreallocation_C",(Mat,PetscScalar[]),(B,data));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode  MatSeqDenseSetPreallocation_SeqDense(Mat B,PetscScalar *data)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  B->preallocated = PETSC_TRUE;

  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);

  b       = (Mat_SeqDense*)B->data;
  b->Mmax = B->rmap->n;
  b->Nmax = B->cmap->n;
  if (b->lda <= 0 || b->changelda) b->lda = B->rmap->n;

  ierr = PetscIntMultError(b->lda,b->Nmax,NULL);CHKERRQ(ierr);
  if (!data) { /* petsc-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    ierr = PetscCalloc1((size_t)b->lda*b->Nmax,&b->v);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)B,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRQ(ierr);

    b->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    b->v          = data;
    b->user_alloc = PETSC_TRUE;
  }
  B->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_Elemental(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            mat_elemental;
  PetscErrorCode ierr;
  PetscScalar    *array,*v_colwise;
  PetscInt       M=A->rmap->N,N=A->cmap->N,i,j,k,*rows,*cols;

  PetscFunctionBegin;
  ierr = PetscMalloc3(M*N,&v_colwise,M,&rows,N,&cols);CHKERRQ(ierr);
  ierr = MatDenseGetArray(A,&array);CHKERRQ(ierr);
  /* convert column-wise array into row-wise v_colwise, see MatSetValues_Elemental() */
  k = 0;
  for (j=0; j<N; j++) {
    cols[j] = j;
    for (i=0; i<M; i++) {
      v_colwise[j*M+i] = array[k++];
    }
  }
  for (i=0; i<M; i++) {
    rows[i] = i;
  }
  ierr = MatDenseRestoreArray(A,&array);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject)A), &mat_elemental);CHKERRQ(ierr);
  ierr = MatSetSizes(mat_elemental,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(mat_elemental,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetUp(mat_elemental);CHKERRQ(ierr);

  /* PETSc-Elemental interaface uses axpy for setting off-processor entries, only ADD_VALUES is allowed */
  ierr = MatSetValues(mat_elemental,M,rows,N,cols,v_colwise,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat_elemental, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree3(v_colwise,rows,cols);CHKERRQ(ierr);

  if (reuse == MAT_INPLACE_MATRIX) {
    ierr = MatHeaderReplace(A,&mat_elemental);CHKERRQ(ierr);
  } else {
    *newmat = mat_elemental;
  }
  PetscFunctionReturn(0);
}
#endif

/*@C
  MatSeqDenseSetLDA - Declare the leading dimension of the user-provided array

  Input parameter:
+ A - the matrix
- lda - the leading dimension

  Notes:
  This routine is to be used in conjunction with MatSeqDenseSetPreallocation();
  it asserts that the preallocation has a leading dimension (the LDA parameter
  of Blas and Lapack fame) larger than M, the first dimension of the matrix.

  Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateSeqDense(), MatSeqDenseSetPreallocation(), MatSetMaximumSize()

@*/
PetscErrorCode  MatSeqDenseSetLDA(Mat B,PetscInt lda)
{
  Mat_SeqDense *b = (Mat_SeqDense*)B->data;

  PetscFunctionBegin;
  if (lda < B->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"LDA %D must be at least matrix dimension %D",lda,B->rmap->n);
  b->lda       = lda;
  b->changelda = PETSC_FALSE;
  b->Mmax      = PetscMax(b->Mmax,lda);
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSE - MATSEQDENSE = "seqdense" - A matrix type to be used for sequential dense matrices.

   Options Database Keys:
. -mat_type seqdense - sets the matrix type to "seqdense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqDense()

M*/

PETSC_EXTERN PetscErrorCode MatCreate_SeqDense(Mat B)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr    = PetscNewLog(B,&b);CHKERRQ(ierr);
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->data = (void*)b;

  b->roworiented = PETSC_TRUE;

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseGetArray_C",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDensePlaceArray_C",MatDensePlaceArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseResetArray_C",MatDenseResetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatDenseRestoreArray_C",MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_seqaij_C",MatConvert_SeqDense_SeqAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_ELEMENTAL)
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatConvert_seqdense_elemental_C",MatConvert_SeqDense_Elemental);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSeqDenseSetPreallocation_C",MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqaij_seqdense_C",MatMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultSymbolic_seqaij_seqdense_C",MatMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqaij_seqdense_C",MatMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_seqaij_seqdense_C",MatPtAP_SeqDense_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqaijperm_seqdense_C",MatMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultSymbolic_seqaijperm_seqdense_C",MatMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqaijperm_seqdense_C",MatMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_seqaijperm_seqdense_C",MatPtAP_SeqDense_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMult_seqaijmkl_seqdense_C",MatMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultSymbolic_seqaijmkl_seqdense_C",MatMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatMatMultNumeric_seqaijmkl_seqdense_C",MatMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatPtAP_seqaijmkl_seqdense_C",MatPtAP_SeqDense_SeqDense);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_seqaij_seqdense_C",MatTransposeMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultSymbolic_seqaij_seqdense_C",MatTransposeMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultNumeric_seqaij_seqdense_C",MatTransposeMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_seqaijperm_seqdense_C",MatTransposeMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultSymbolic_seqaijperm_seqdense_C",MatTransposeMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultNumeric_seqaijperm_seqdense_C",MatTransposeMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMult_seqaijmkl_seqdense_C",MatTransposeMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultSymbolic_seqaijmkl_seqdense_C",MatTransposeMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatTransposeMatMultNumeric_seqaijmkl_seqdense_C",MatTransposeMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
