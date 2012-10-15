
/*
     Defines the basic matrix operations for sequential dense.
*/

#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/
#include <petscblaslapack.h>

#include <../src/mat/impls/aij/seq/aij.h>
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqDense_SeqAIJ"
PetscErrorCode  MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype,MatReuse reuse,Mat *newmat)
{
  Mat            B;
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       *rows;
  MatScalar      *aa = a->v;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,A->cmap->n,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscMalloc(A->rmap->n*sizeof(PetscInt),&rows);CHKERRQ(ierr);
  for (i=0; i<A->rmap->n; i++) rows[i] = i;

  for (i=0; i<A->cmap->n; i++) {
    ierr  = MatSetValues(B,A->rmap->n,rows,1,&i,aa,INSERT_VALUES);CHKERRQ(ierr);
    aa   += a->lda;
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatHeaderReplace(A,B);CHKERRQ(ierr);
  } else {
    *newmat = B;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatAXPY_SeqDense"
PetscErrorCode MatAXPY_SeqDense(Mat Y,PetscScalar alpha,Mat X,MatStructure str)
{
  Mat_SeqDense   *x = (Mat_SeqDense*)X->data,*y = (Mat_SeqDense*)Y->data;
  PetscScalar    oalpha = alpha;
  PetscInt       j;
  PetscBLASInt   N,m,ldax,lday,one = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  N    = PetscBLASIntCast(X->rmap->n*X->cmap->n);
  m    = PetscBLASIntCast(X->rmap->n);
  ldax = PetscBLASIntCast(x->lda);
  lday = PetscBLASIntCast(y->lda);
  if (ldax>m || lday>m) {
    for (j=0; j<X->cmap->n; j++) {
      BLASaxpy_(&m,&oalpha,x->v+j*ldax,&one,y->v+j*lday,&one);
    }
  } else {
    BLASaxpy_(&N,&oalpha,x->v,&one,y->v,&one);
  }
  ierr = PetscLogFlops(PetscMax(2*N-1,0));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetInfo_SeqDense"
PetscErrorCode MatGetInfo_SeqDense(Mat A,MatInfoType flag,MatInfo *info)
{
  PetscInt     N = A->rmap->n*A->cmap->n;

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

#undef __FUNCT__
#define __FUNCT__ "MatScale_SeqDense"
PetscErrorCode MatScale_SeqDense(Mat A,PetscScalar alpha)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscScalar    oalpha = alpha;
  PetscErrorCode ierr;
  PetscBLASInt   one = 1,j,nz,lda = PetscBLASIntCast(a->lda);

  PetscFunctionBegin;
  if (lda>A->rmap->n) {
    nz = PetscBLASIntCast(A->rmap->n);
    for (j=0; j<A->cmap->n; j++) {
      BLASscal_(&nz,&oalpha,a->v+j*lda,&one);
    }
  } else {
    nz = PetscBLASIntCast(A->rmap->n*A->cmap->n);
    BLASscal_(&nz,&oalpha,a->v,&one);
  }
  ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatIsHermitian_SeqDense"
PetscErrorCode MatIsHermitian_SeqDense(Mat A,PetscReal rtol,PetscBool  *fl)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscInt       i,j,m = A->rmap->n,N;
  PetscScalar    *v = a->v;

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

#undef __FUNCT__
#define __FUNCT__ "MatDuplicateNoCreate_SeqDense"
PetscErrorCode MatDuplicateNoCreate_SeqDense(Mat newi,Mat A,MatDuplicateOption cpvalues)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data,*l;
  PetscErrorCode ierr;
  PetscInt       lda = (PetscInt)mat->lda,j,m;

  PetscFunctionBegin;
  ierr = PetscLayoutReference(A->rmap,&newi->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutReference(A->cmap,&newi->cmap);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(newi,PETSC_NULL);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SeqDense"
PetscErrorCode MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,newmat);CHKERRQ(ierr);
  ierr = MatSetSizes(*newmat,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatDuplicateNoCreate_SeqDense(*newmat,A,cpvalues);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


extern PetscErrorCode MatLUFactor_SeqDense(Mat,IS,IS,const MatFactorInfo*);

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorNumeric_SeqDense"
PetscErrorCode MatLUFactorNumeric_SeqDense(Mat fact,Mat A,const MatFactorInfo *info_dummy)
{
  MatFactorInfo  info;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicateNoCreate_SeqDense(fact,A,MAT_COPY_VALUES);CHKERRQ(ierr);
  ierr = MatLUFactor_SeqDense(fact,0,0,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolve_SeqDense"
PetscErrorCode MatSolve_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x,*y;
  PetscBLASInt   one = 1,info,m = PetscBLASIntCast(A->rmap->n);

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("N",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
#endif
  } else if (A->factortype == MAT_FACTOR_CHOLESKY){
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
#endif
  }
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatSolve_SeqDense"
PetscErrorCode MatMatSolve_SeqDense(Mat A,Mat B,Mat X)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *b,*x;
  PetscInt       n;
  PetscBLASInt   nrhs,info,m=PetscBLASIntCast(A->rmap->n);
  PetscBool      flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");

  ierr = MatGetSize(B,PETSC_NULL,&n);CHKERRQ(ierr);
  nrhs = PetscBLASIntCast(n);
  ierr = MatDenseGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatDenseGetArray(X,&x);CHKERRQ(ierr);

  ierr = PetscMemcpy(x,b,m*nrhs*sizeof(PetscScalar));CHKERRQ(ierr);

  if (A->factortype == MAT_FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("N",&m,&nrhs,mat->v,&mat->lda,mat->pivots,x,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"GETRS - Bad solve");
#endif
  } else if (A->factortype == MAT_FACTOR_CHOLESKY){
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&nrhs,mat->v,&mat->lda,x,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS Bad solve");
#endif
  }
  else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");

  ierr = MatDenseRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = PetscLogFlops(nrhs*(2.0*m*m - m));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTranspose_SeqDense"
PetscErrorCode MatSolveTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x,*y;
  PetscBLASInt   one = 1,info,m = PetscBLASIntCast(A->rmap->n);

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("T",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS - Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"POTRS - Bad solve");
#endif
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveAdd_SeqDense"
PetscErrorCode MatSolveAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x,*y,sone = 1.0;
  Vec            tmp = 0;
  PetscBLASInt   one = 1,info,m = PetscBLASIntCast(A->rmap->n);

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(A,tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  }
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("N",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad solve");
#endif
  }
  if (tmp) {
    ierr = VecAXPY(yy,sone,tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(yy,sone,zz);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSolveTransposeAdd_SeqDense"
PetscErrorCode MatSolveTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscScalar    *x,*y,sone = 1.0;
  Vec            tmp;
  PetscBLASInt   one = 1,info,m = PetscBLASIntCast(A->rmap->n);

  PetscFunctionBegin;
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(A,tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  }
  ierr = PetscMemcpy(y,x,A->rmap->n*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("T",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS)
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad solve");
#endif
  }
  if (tmp) {
    ierr = VecAXPY(yy,sone,tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(yy,sone,zz);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->cmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
#undef __FUNCT__
#define __FUNCT__ "MatLUFactor_SeqDense"
PetscErrorCode MatLUFactor_SeqDense(Mat A,IS row,IS col,const MatFactorInfo *minfo)
{
#if defined(PETSC_MISSING_LAPACK_GETRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"GETRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n,m,info;

  PetscFunctionBegin;
  n = PetscBLASIntCast(A->cmap->n);
  m = PetscBLASIntCast(A->rmap->n);
  if (!mat->pivots) {
    ierr = PetscMalloc((A->rmap->n+1)*sizeof(PetscBLASInt),&mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,A->rmap->n*sizeof(PetscBLASInt));CHKERRQ(ierr);
  }
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
  LAPACKgetrf_(&m,&n,mat->v,&mat->lda,mat->pivots,&info);
  ierr = PetscFPTrapPop();CHKERRQ(ierr);

  if (info<0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
  A->ops->solve             = MatSolve_SeqDense;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense;
  A->ops->solveadd          = MatSolveAdd_SeqDense;
  A->ops->solvetransposeadd = MatSolveTransposeAdd_SeqDense;
  A->factortype             = MAT_FACTOR_LU;

  ierr = PetscLogFlops((2.0*A->cmap->n*A->cmap->n*A->cmap->n)/3);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactor_SeqDense"
PetscErrorCode MatCholeskyFactor_SeqDense(Mat A,IS perm,const MatFactorInfo *factinfo)
{
#if defined(PETSC_MISSING_LAPACK_POTRF)
  PetscFunctionBegin;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   info,n = PetscBLASIntCast(A->cmap->n);

  PetscFunctionBegin;
  ierr = PetscFree(mat->pivots);CHKERRQ(ierr);

  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  LAPACKpotrf_("L",&n,mat->v,&mat->lda,&info);
  if (info) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %D",(PetscInt)info-1);
  A->ops->solve             = MatSolve_SeqDense;
  A->ops->solvetranspose    = MatSolveTranspose_SeqDense;
  A->ops->solveadd          = MatSolveAdd_SeqDense;
  A->ops->solvetransposeadd = MatSolveTransposeAdd_SeqDense;
  A->factortype             = MAT_FACTOR_CHOLESKY;
  ierr = PetscLogFlops((A->cmap->n*A->cmap->n*A->cmap->n)/3.0);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqDense"
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

#undef __FUNCT__
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqDense"
PetscErrorCode MatCholeskyFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->assembled                  = PETSC_TRUE;
  fact->preallocated               = PETSC_TRUE;
  fact->ops->choleskyfactornumeric = MatCholeskyFactorNumeric_SeqDense;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatLUFactorSymbolic_SeqDense"
PetscErrorCode MatLUFactorSymbolic_SeqDense(Mat fact,Mat A,IS row,IS col,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  fact->preallocated         = PETSC_TRUE;
  fact->assembled            = PETSC_TRUE;
  fact->ops->lufactornumeric = MatLUFactorNumeric_SeqDense;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatGetFactor_seqdense_petsc"
PetscErrorCode MatGetFactor_seqdense_petsc(Mat A,MatFactorType ftype,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,fact);CHKERRQ(ierr);
  ierr = MatSetSizes(*fact,A->rmap->n,A->cmap->n,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(*fact,((PetscObject)A)->type_name);CHKERRQ(ierr);
  if (ftype == MAT_FACTOR_LU){
    (*fact)->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqDense;
  } else {
    (*fact)->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SeqDense;
  }
  (*fact)->factortype = ftype;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* ------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatSOR_SeqDense"
PetscErrorCode MatSOR_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *x,*b,*v = mat->v,zero = 0.0,xt;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap->n,i;
  PetscBLASInt   o = 1,bm = PetscBLASIntCast(m);

  PetscFunctionBegin;
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdot */
    ierr = VecSet(xx,zero);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  its  = its*lits;
  if (its <= 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for (i=0; i<m; i++) {
        xt = b[i] - BLASdot_(&bm,v+i,&bm,x,&o);
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (i=m-1; i>=0; i--) {
        xt = b[i] - BLASdot_(&bm,v+i,&bm,x,&o);
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
  }
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_SeqDense"
PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y;
  PetscErrorCode ierr;
  PetscBLASInt   m, n,_One=1;
  PetscScalar    _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->rmap->n);
  n = PetscBLASIntCast(A->cmap->n);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMult_SeqDense"
PetscErrorCode MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y,_DOne=1.0,_DZero=0.0;
  PetscErrorCode ierr;
  PetscBLASInt   m, n, _One=1;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->rmap->n);
  n = PetscBLASIntCast(A->cmap->n);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n - A->rmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_SeqDense"
PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y,_DOne=1.0;
  PetscErrorCode ierr;
  PetscBLASInt   m, n, _One=1;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->rmap->n);
  n = PetscBLASIntCast(A->cmap->n);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_SeqDense"
PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y;
  PetscErrorCode ierr;
  PetscBLASInt   m, n, _One=1;
  PetscScalar    _DOne=1.0;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->rmap->n);
  n = PetscBLASIntCast(A->cmap->n);
  if (!A->rmap->n || !A->cmap->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("T",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2.0*A->rmap->n*A->cmap->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatGetRow_SeqDense"
PetscErrorCode MatGetRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  *ncols = A->cmap->n;
  if (cols) {
    ierr = PetscMalloc((A->cmap->n+1)*sizeof(PetscInt),cols);CHKERRQ(ierr);
    for (i=0; i<A->cmap->n; i++) (*cols)[i] = i;
  }
  if (vals) {
    ierr = PetscMalloc((A->cmap->n+1)*sizeof(PetscScalar),vals);CHKERRQ(ierr);
    v    = mat->v + row;
    for (i=0; i<A->cmap->n; i++) {(*vals)[i] = *v; v += mat->lda;}
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRestoreRow_SeqDense"
PetscErrorCode MatRestoreRow_SeqDense(Mat A,PetscInt row,PetscInt *ncols,PetscInt **cols,PetscScalar **vals)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  if (cols) {ierr = PetscFree(*cols);CHKERRQ(ierr);}
  if (vals) {ierr = PetscFree(*vals);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatSetValues_SeqDense"
PetscErrorCode MatSetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],const PetscScalar v[],InsertMode addv)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
  PetscInt     i,j,idx=0;

  PetscFunctionBegin;
  if (v) PetscValidScalarPointer(v,6);
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

#undef __FUNCT__
#define __FUNCT__ "MatGetValues_SeqDense"
PetscErrorCode MatGetValues_SeqDense(Mat A,PetscInt m,const PetscInt indexm[],PetscInt n,const PetscInt indexn[],PetscScalar v[])
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

#undef __FUNCT__
#define __FUNCT__ "MatLoad_SeqDense"
PetscErrorCode MatLoad_SeqDense(Mat newmat,PetscViewer viewer)
{
  Mat_SeqDense   *a;
  PetscErrorCode ierr;
  PetscInt       *scols,i,j,nz,header[4];
  int            fd;
  PetscMPIInt    size;
  PetscInt       *rowlengths = 0,M,N,*cols,grows,gcols;
  PetscScalar    *vals,*svals,*v,*w;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
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
  ierr = MatSeqDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);

  if (nz == MATRIX_BINARY_FORMAT_DENSE) { /* matrix in file is dense */
    a    = (Mat_SeqDense*)newmat->data;
    v    = a->v;
    /* Allocate some temp space to read in the values and then flip them
       from row major to column major */
    ierr = PetscMalloc((M*N > 0 ? M*N : 1)*sizeof(PetscScalar),&w);CHKERRQ(ierr);
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
    ierr = PetscMalloc((M+1)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

    a = (Mat_SeqDense*)newmat->data;
    v = a->v;

    /* read column indices and nonzeros */
    ierr = PetscMalloc((nz+1)*sizeof(PetscInt),&scols);CHKERRQ(ierr);
    cols = scols;
    ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&svals);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqDense_ASCII"
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
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
    for (i=0; i<A->rmap->n; i++) {
      v = a->v + i;
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=0; j<A->cmap->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(*v) != 0.0 && PetscImaginaryPart(*v) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G + %G i) ",j,PetscRealPart(*v),PetscImaginaryPart(*v));CHKERRQ(ierr);
        } else if (PetscRealPart(*v)) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",j,PetscRealPart(*v));CHKERRQ(ierr);
        }
#else
        if (*v) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %G) ",j,*v);CHKERRQ(ierr);
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
	if (PetscImaginaryPart(v[i])) { allreal = PETSC_FALSE; break ;}
    }
#endif
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = zeros(%D,%D);\n",name,A->rmap->n,A->cmap->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    } else {
      ierr = PetscObjectPrintClassNamePrefixType((PetscObject)A,viewer,"Matrix Object");CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqDense_Binary"
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
    ierr = PetscMalloc(4*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
    col_lens[0] = MAT_FILE_CLASSID;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = MATRIX_BINARY_FORMAT_DENSE;
    ierr = PetscBinaryWrite(fd,col_lens,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* write out matrix, by rows */
    ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
    v    = a->v;
    for (j=0; j<n; j++) {
      for (i=0; i<m; i++) {
        vals[j + i*n] = *v++;
      }
    }
    ierr = PetscBinaryWrite(fd,vals,n*m,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc((4+nz)*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
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
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&anonz);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqDense_Draw_Zoom"
PetscErrorCode MatView_SeqDense_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A = (Mat) Aa;
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          m = A->rmap->n,n = A->cmap->n,color,i,j;
  PetscScalar       *v = a->v;
  PetscViewer       viewer;
  PetscDraw         popup;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r,scale,maxv = 0.0;
  PetscViewerFormat format;

  PetscFunctionBegin;

  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* Loop over matrix elements drawing boxes */
  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    /* Blue for negative and Red for positive */
    color = PETSC_DRAW_BLUE;
    for (j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for (i = 0; i < m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(v[j*m+i]) >  0.) {
          color = PETSC_DRAW_RED;
        } else if (PetscRealPart(v[j*m+i]) <  0.) {
          color = PETSC_DRAW_BLUE;
        } else {
          continue;
        }
#else
        if (v[j*m+i] >  0.) {
          color = PETSC_DRAW_RED;
        } else if (v[j*m+i] <  0.) {
          color = PETSC_DRAW_BLUE;
        } else {
          continue;
        }
#endif
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    for (i = 0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    scale = (245.0 - PETSC_DRAW_BASIC_COLORS)/maxv;
    ierr  = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    if (popup) {ierr  = PetscDrawScalePopup(popup,0.0,maxv);CHKERRQ(ierr);}
    for (j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for (i = 0; i < m; i++) {
        y_l   = m - i - 1.0;
        y_r   = y_l + 1.0;
        color = PETSC_DRAW_BASIC_COLORS + (int)(scale*PetscAbsScalar(v[j*m+i]));
        ierr  = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqDense_Draw"
PetscErrorCode MatView_SeqDense_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw      draw;
  PetscBool      isnull;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->cmap->n; yr = A->rmap->n; h = yr/10.0; w = xr/10.0;
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqDense_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqDense"
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
  } else {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Viewer type %s not supported by dense matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SeqDense"
PetscErrorCode MatDestroy_SeqDense(Mat mat)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)mat->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)mat,"Rows %D Cols %D",mat->rmap->n,mat->cmap->n);
#endif
  ierr = PetscFree(l->pivots);CHKERRQ(ierr);
  if (!l->user_alloc) {ierr = PetscFree(l->v);CHKERRQ(ierr);}
  ierr = PetscFree(mat->data);CHKERRQ(ierr);

  ierr = PetscObjectChangeTypeName((PetscObject)mat,0);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatDenseGetArray_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatDenseRestoreArray_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatSeqDenseSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMult_seqaij_seqdense_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMultSymbolic_seqaij_seqdense_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatMatMultNumeric_seqaij_seqdense_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTranspose_SeqDense"
PetscErrorCode MatTranspose_SeqDense(Mat A,MatReuse reuse,Mat *matout)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       k,j,m,n,M;
  PetscScalar    *v,tmp;

  PetscFunctionBegin;
  v = mat->v; m = A->rmap->n; M = mat->lda; n = A->cmap->n;
  if (reuse == MAT_REUSE_MATRIX && *matout == A) { /* in place transpose */
    if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Can not transpose non-square matrix in place");
    else {
      for (j=0; j<m; j++) {
        for (k=0; k<j; k++) {
          tmp = v[j + k*M];
          v[j + k*M] = v[k + j*M];
          v[k + j*M] = tmp;
        }
      }
    }
  } else { /* out-of-place transpose */
    Mat          tmat;
    Mat_SeqDense *tmatd;
    PetscScalar  *v2;

    if (reuse == MAT_INITIAL_MATRIX) {
      ierr  = MatCreate(((PetscObject)A)->comm,&tmat);CHKERRQ(ierr);
      ierr  = MatSetSizes(tmat,A->cmap->n,A->rmap->n,A->cmap->n,A->rmap->n);CHKERRQ(ierr);
      ierr  = MatSetType(tmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
      ierr  = MatSeqDenseSetPreallocation(tmat,PETSC_NULL);CHKERRQ(ierr);
    } else {
      tmat = *matout;
    }
    tmatd = (Mat_SeqDense*)tmat->data;
    v = mat->v; v2 = tmatd->v;
    for (j=0; j<n; j++) {
      for (k=0; k<m; k++) v2[j + k*n] = v[k + j*M];
    }
    ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    *matout = tmat;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatEqual_SeqDense"
PetscErrorCode MatEqual_SeqDense(Mat A1,Mat A2,PetscBool  *flg)
{
  Mat_SeqDense *mat1 = (Mat_SeqDense*)A1->data;
  Mat_SeqDense *mat2 = (Mat_SeqDense*)A2->data;
  PetscInt     i,j;
  PetscScalar  *v1 = mat1->v,*v2 = mat2->v;

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

#undef __FUNCT__
#define __FUNCT__ "MatGetDiagonal_SeqDense"
PetscErrorCode MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,n,len;
  PetscScalar    *x,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(v,zero);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  len = PetscMin(A->rmap->n,A->cmap->n);
  if (n != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
  for (i=0; i<len; i++) {
    x[i] = mat->v[i*mat->lda + i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDiagonalScale_SeqDense"
PetscErrorCode MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *l,*r,x,*v;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->rmap->n,n = A->cmap->n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    if (m != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Left scaling vec wrong size");
    for (i=0; i<m; i++) {
      x = l[i];
      v = mat->v + i;
      for (j=0; j<n; j++) { (*v) *= x; v+= m;}
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    ierr = PetscLogFlops(n*m);CHKERRQ(ierr);
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
    if (n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Right scaling vec wrong size");
    for (i=0; i<n; i++) {
      x = r[i];
      v = mat->v + i*m;
      for (j=0; j<m; j++) { (*v++) *= x;}
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
    ierr = PetscLogFlops(n*m);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatNorm_SeqDense"
PetscErrorCode MatNorm_SeqDense(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;
  PetscScalar  *v = mat->v;
  PetscReal    sum = 0.0;
  PetscInt     lda=mat->lda,m=A->rmap->n,i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    if (lda>m) {
      for (j=0; j<A->cmap->n; j++) {
	v = mat->v+j*lda;
	for (i=0; i<m; i++) {
#if defined(PETSC_USE_COMPLEX)
	  sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
	  sum += (*v)*(*v); v++;
#endif
	}
      }
    } else {
      for (i=0; i<A->cmap->n*A->rmap->n; i++) {
#if defined(PETSC_USE_COMPLEX)
	sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
	sum += (*v)*(*v); v++;
#endif
      }
    }
    *nrm = PetscSqrtReal(sum);
    ierr = PetscLogFlops(2.0*A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    *nrm = 0.0;
    for (j=0; j<A->cmap->n; j++) {
      v = mat->v + j*mat->lda;
      sum = 0.0;
      for (i=0; i<A->rmap->n; i++) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->rmap->n; j++) {
      v = mat->v + j;
      sum = 0.0;
      for (i=0; i<A->cmap->n; i++) {
        sum += PetscAbsScalar(*v); v += mat->lda;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(A->cmap->n*A->rmap->n);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"No two norm");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetOption_SeqDense"
PetscErrorCode MatSetOption_SeqDense(Mat A,MatOption op,PetscBool  flg)
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

#undef __FUNCT__
#define __FUNCT__ "MatZeroEntries_SeqDense"
PetscErrorCode MatZeroEntries_SeqDense(Mat A)
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

#undef __FUNCT__
#define __FUNCT__ "MatZeroRows_SeqDense"
PetscErrorCode MatZeroRows_SeqDense(Mat A,PetscInt N,const PetscInt rows[],PetscScalar diag,Vec x,Vec b)
{
  PetscErrorCode    ierr;
  Mat_SeqDense      *l = (Mat_SeqDense*)A->data;
  PetscInt          m = l->lda, n = A->cmap->n, i,j;
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
    for (i=0; i<N; i++) {
      bb[rows[i]] = diag*xx[rows[i]];
    }
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
      slot = l->v + (m+1)*rows[i];
      *slot = diag;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDenseGetArray_SeqDense"
PetscErrorCode MatDenseGetArray_SeqDense(Mat A,PetscScalar *array[])
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  if (mat->lda != A->rmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot get array for Dense matrices with LDA different from number of rows");
  *array = mat->v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDenseRestoreArray_SeqDense"
PetscErrorCode MatDenseRestoreArray_SeqDense(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = 0; /* user cannot accidently use the array later */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDenseGetArray"
/*@C
   MatDenseGetArray - gives access to the array where the data for a SeqDense matrix is stored

   Not Collective

   Input Parameter:
.  mat - a MATSEQDENSE matrix

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

#undef __FUNCT__
#define __FUNCT__ "MatDenseRestoreArray"
/*@C
   MatDenseRestoreArray - returns access to the array where the data for a SeqDense matrix is stored obtained by MatDenseGetArray()

   Not Collective

   Input Parameters:
.  mat - a MATSEQDENSE matrix
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

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrix_SeqDense"
static PetscErrorCode MatGetSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
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
    ierr = MatCreate(((PetscObject)A)->comm,&newmat);CHKERRQ(ierr);
    ierr = MatSetSizes(newmat,nrows,ncols,nrows,ncols);CHKERRQ(ierr);
    ierr = MatSetType(newmat,((PetscObject)A)->type_name);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy,column at a time */
  bv = ((Mat_SeqDense*)newmat->data)->v;

  for (i=0; i<ncols; i++) {
    av = v + mat->lda*icol[i];
    for (j=0; j<nrows; j++) {
      *bv++ = av[irow[j]];
    }
  }

  /* Assemble the matrices so that the correct flags are set */
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Free work space */
  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
  *B = newmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetSubMatrices_SeqDense"
PetscErrorCode MatGetSubMatrices_SeqDense(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc((n+1)*sizeof(Mat),B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqDense(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyBegin_SeqDense"
PetscErrorCode MatAssemblyBegin_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqDense"
PetscErrorCode MatAssemblyEnd_SeqDense(Mat mat,MatAssemblyType mode)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCopy_SeqDense"
PetscErrorCode MatCopy_SeqDense(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense *)B->data;
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
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetUp_SeqDense"
PetscErrorCode MatSetUp_SeqDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqDenseSetPreallocation(A,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatConjugate_SeqDense"
static PetscErrorCode MatConjugate_SeqDense(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscScalar    *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscConj(aa[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatRealPart_SeqDense"
static PetscErrorCode MatRealPart_SeqDense(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscScalar    *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscRealPart(aa[i]);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatImaginaryPart_SeqDense"
static PetscErrorCode MatImaginaryPart_SeqDense(Mat A)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscInt       i,nz = A->rmap->n*A->cmap->n;
  PetscScalar    *aa = a->v;

  PetscFunctionBegin;
  for (i=0; i<nz; i++) aa[i] = PetscImaginaryPart(aa[i]);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "MatMatMult_SeqDense_SeqDense"
PetscErrorCode MatMatMult_SeqDense_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_SeqDense_SeqDense(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatMatMultNumeric_SeqDense_SeqDense(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_SeqDense_SeqDense"
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
  ierr = MatSeqDenseSetPreallocation(Cmat,PETSC_NULL);CHKERRQ(ierr);
  Cmat->assembled    = PETSC_TRUE;
  Cmat->ops->matmult = MatMatMult_SeqDense_SeqDense;
  *C = Cmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultNumeric_SeqDense_SeqDense"
PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense   *c = (Mat_SeqDense*)C->data;
  PetscBLASInt   m,n,k;
  PetscScalar    _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->rmap->n);
  n = PetscBLASIntCast(B->cmap->n);
  k = PetscBLASIntCast(A->cmap->n);
  BLASgemm_("N","N",&m,&n,&k,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMult_SeqDense_SeqDense"
PetscErrorCode MatTransposeMatMult_SeqDense_SeqDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatTransposeMatMultSymbolic_SeqDense_SeqDense(A,B,fill,C);CHKERRQ(ierr);
  }
  ierr = MatTransposeMatMultNumeric_SeqDense_SeqDense(A,B,*C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultSymbolic_SeqDense_SeqDense"
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
  ierr = MatSeqDenseSetPreallocation(Cmat,PETSC_NULL);CHKERRQ(ierr);
  Cmat->assembled = PETSC_TRUE;
  *C = Cmat;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatTransposeMatMultNumeric_SeqDense_SeqDense"
PetscErrorCode MatTransposeMatMultNumeric_SeqDense_SeqDense(Mat A,Mat B,Mat C)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  Mat_SeqDense   *b = (Mat_SeqDense*)B->data;
  Mat_SeqDense   *c = (Mat_SeqDense*)C->data;
  PetscBLASInt   m,n,k;
  PetscScalar    _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  m = PetscBLASIntCast(A->cmap->n);
  n = PetscBLASIntCast(B->cmap->n);
  k = PetscBLASIntCast(A->rmap->n);
  /*
     Note the m and n arguments below are the number rows and columns of A', not A!
  */
  BLASgemm_("T","N",&m,&n,&k,&_DOne,a->v,&a->lda,b->v,&b->lda,&_DZero,c->v,&c->lda);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRowMax_SeqDense"
PetscErrorCode MatGetRowMax_SeqDense(Mat A,Vec v,PetscInt idx[])
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
    for (j=1; j<n; j++){
      if (PetscRealPart(x[i]) < PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRowMaxAbs_SeqDense"
PetscErrorCode MatGetRowMaxAbs_SeqDense(Mat A,Vec v,PetscInt idx[])
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
    for (j=1; j<n; j++){
      atmp = PetscAbsScalar(aa[i+m*j]);
      if (PetscAbsScalar(x[i]) < atmp) {x[i] = atmp; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetRowMin_SeqDense"
PetscErrorCode MatGetRowMin_SeqDense(Mat A,Vec v,PetscInt idx[])
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
    for (j=1; j<n; j++){
      if (PetscRealPart(x[i]) > PetscRealPart(aa[i+m*j])) {x[i] = aa[i + m*j]; if (idx) idx[i] = j;}
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetColumnVector_SeqDense"
PetscErrorCode MatGetColumnVector_SeqDense(Mat A,Vec v,PetscInt col)
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


#undef __FUNCT__
#define __FUNCT__ "MatGetColumnNorms_SeqDense"
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
    for (i=0; i<n; i++ ){
      for (j=0; j<m; j++) {
	norms[i] += PetscAbsScalar(a[j]*a[j]);
      }
      a += m;
    }
  } else if (type == NORM_1) {
    for (i=0; i<n; i++ ){
      for (j=0; j<m; j++) {
	norms[i] += PetscAbsScalar(a[j]);
      }
      a += m;
    }
  } else if (type == NORM_INFINITY) {
    for (i=0; i<n; i++ ){
      for (j=0; j<m; j++) {
	norms[i] = PetscMax(PetscAbsScalar(a[j]),norms[i]);
      }
      a += m;
    }
  } else SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Unknown NormType");
  ierr = MatDenseRestoreArray(A,&a);CHKERRQ(ierr);
  if (type == NORM_2) {
    for (i=0; i<n; i++) norms[i] = PetscSqrtReal(norms[i]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetRandom_SeqDense"
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


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqDense,
       MatGetRow_SeqDense,
       MatRestoreRow_SeqDense,
       MatMult_SeqDense,
/* 4*/ MatMultAdd_SeqDense,
       MatMultTranspose_SeqDense,
       MatMultTransposeAdd_SeqDense,
       0,
       0,
       0,
/*10*/ 0,
       MatLUFactor_SeqDense,
       MatCholeskyFactor_SeqDense,
       MatSOR_SeqDense,
       MatTranspose_SeqDense,
/*15*/ MatGetInfo_SeqDense,
       MatEqual_SeqDense,
       MatGetDiagonal_SeqDense,
       MatDiagonalScale_SeqDense,
       MatNorm_SeqDense,
/*20*/ MatAssemblyBegin_SeqDense,
       MatAssemblyEnd_SeqDense,
       MatSetOption_SeqDense,
       MatZeroEntries_SeqDense,
/*24*/ MatZeroRows_SeqDense,
       0,
       0,
       0,
       0,
/*29*/ MatSetUp_SeqDense,
       0,
       0,
       0,
       0,
/*34*/ MatDuplicate_SeqDense,
       0,
       0,
       0,
       0,
/*39*/ MatAXPY_SeqDense,
       MatGetSubMatrices_SeqDense,
       0,
       MatGetValues_SeqDense,
       MatCopy_SeqDense,
/*44*/ MatGetRowMax_SeqDense,
       MatScale_SeqDense,
       0,
       0,
       0,
/*49*/ MatSetRandom_SeqDense,
       0,
       0,
       0,
       0,
/*54*/ 0,
       0,
       0,
       0,
       0,
/*59*/ 0,
       MatDestroy_SeqDense,
       MatView_SeqDense,
       0,
       0,
/*64*/ 0,
       0,
       0,
       0,
       0,
/*69*/ MatGetRowMaxAbs_SeqDense,
       0,
       0,
       0,
       0,
/*74*/ 0,
       0,
       0,
       0,
       0,
/*79*/ 0,
       0,
       0,
       0,
/*83*/ MatLoad_SeqDense,
       0,
       MatIsHermitian_SeqDense,
       0,
       0,
       0,
/*89*/ MatMatMult_SeqDense_SeqDense,
       MatMatMultSymbolic_SeqDense_SeqDense,
       MatMatMultNumeric_SeqDense_SeqDense,
       0,
       0,
/*94*/ 0,
       0,
       0,
       0,
       0,
/*99*/ 0,
       0,
       0,
       MatConjugate_SeqDense,
       0,
/*104*/0,
       MatRealPart_SeqDense,
       MatImaginaryPart_SeqDense,
       0,
       0,
/*109*/MatMatSolve_SeqDense,
       0,
       MatGetRowMin_SeqDense,
       MatGetColumnVector_SeqDense,
       0,
/*114*/0,
       0,
       0,
       0,
       0,
/*119*/0,
       0,
       0,
       0,
       0,
/*124*/0,
       MatGetColumnNorms_SeqDense,
       0,
       0,
       0,
/*129*/0,
       MatTransposeMatMult_SeqDense_SeqDense,
       MatTransposeMatMultSymbolic_SeqDense_SeqDense,
       MatTransposeMatMultNumeric_SeqDense_SeqDense,
};

#undef __FUNCT__
#define __FUNCT__ "MatCreateSeqDense"
/*@C
   MatCreateSeqDense - Creates a sequential dense matrix that
   is stored in column major order (the usual Fortran 77 manner). Many
   of the matrix operations use the BLAS and LAPACK routines.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
-  data - optional location of matrix data in column major order.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

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

#undef __FUNCT__
#define __FUNCT__ "MatSeqDenseSetPreallocation"
/*@C
   MatSeqDenseSetPreallocation - Sets the array used for storing the matrix elements

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
-  data - the array (or PETSC_NULL)

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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatSeqDenseSetPreallocation_SeqDense"
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

  if (!data) { /* petsc-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    ierr = PetscMalloc(b->lda*b->Nmax*sizeof(PetscScalar),&b->v);CHKERRQ(ierr);
    ierr = PetscMemzero(b->v,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(B,b->lda*b->Nmax*sizeof(PetscScalar));CHKERRQ(ierr);
    b->user_alloc = PETSC_FALSE;
  } else { /* user-allocated storage */
    if (!b->user_alloc) { ierr = PetscFree(b->v);CHKERRQ(ierr); }
    b->v          = data;
    b->user_alloc = PETSC_TRUE;
  }
  B->assembled = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatSeqDenseSetLDA"
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqDense"
PetscErrorCode  MatCreate_SeqDense(Mat B)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(((PetscObject)B)->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  ierr            = PetscNewLog(B,Mat_SeqDense,&b);CHKERRQ(ierr);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->data         = (void*)b;

  b->pivots       = 0;
  b->roworiented  = PETSC_TRUE;
  b->v            = 0;
  b->changelda    = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatDenseGetArray_C","MatDenseGetArray_SeqDense",MatDenseGetArray_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatDenseRestoreArray_C","MatDenseRestoreArray_SeqDense",MatDenseRestoreArray_SeqDense);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqdense_seqaij_C","MatConvert_SeqDense_SeqAIJ",MatConvert_SeqDense_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatGetFactor_petsc_C",
                                     "MatGetFactor_seqdense_petsc",
                                      MatGetFactor_seqdense_petsc);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqDenseSetPreallocation_C",
                                    "MatSeqDenseSetPreallocation_SeqDense",
                                     MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMatMult_seqaij_seqdense_C",
                                     "MatMatMult_SeqAIJ_SeqDense",
                                      MatMatMult_SeqAIJ_SeqDense);CHKERRQ(ierr);


  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMatMultSymbolic_seqaij_seqdense_C",
                                     "MatMatMultSymbolic_SeqAIJ_SeqDense",
                                      MatMatMultSymbolic_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatMatMultNumeric_seqaij_seqdense_C",
                                     "MatMatMultNumeric_SeqAIJ_SeqDense",
                                      MatMatMultNumeric_SeqAIJ_SeqDense);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQDENSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
