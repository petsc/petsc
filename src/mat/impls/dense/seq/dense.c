#define PETSCMAT_DLL

/*
     Defines the basic matrix operations for sequential dense.
*/

#include "src/mat/impls/dense/seq/dense.h"
#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqDense"
PetscErrorCode MatAXPY_SeqDense(const PetscScalar *alpha,Mat X,Mat Y,MatStructure str)
{
  Mat_SeqDense *x = (Mat_SeqDense*)X->data,*y = (Mat_SeqDense*)Y->data;
  PetscInt     j;
  PetscBLASInt N = (PetscBLASInt)X->m*X->n,m=(PetscBLASInt)X->m,ldax = x->lda,lday=y->lda,one = 1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (X->m != Y->m || X->n != Y->n) SETERRQ(PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  if (ldax>m || lday>m) {
    for (j=0; j<X->n; j++) {
      BLASaxpy_(&m,(PetscScalar*)alpha,x->v+j*ldax,&one,y->v+j*lday,&one);
    }
  } else {
    BLASaxpy_(&N,(PetscScalar*)alpha,x->v,&one,y->v,&one);
  }
  ierr = PetscLogFlops(2*N-1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqDense"
PetscErrorCode MatGetInfo_SeqDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscInt     i,N = A->m*A->n,count = 0;
  PetscScalar  *v = a->v;

  PetscFunctionBegin;
  for (i=0; i<N; i++) {if (*v != 0.0) count++; v++;}

  info->rows_global       = (double)A->m;
  info->columns_global    = (double)A->n;
  info->rows_local        = (double)A->m;
  info->columns_local     = (double)A->n;
  info->block_size        = 1.0;
  info->nz_allocated      = (double)N;
  info->nz_used           = (double)count;
  info->nz_unneeded       = (double)(N-count);
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = 0;
  info->memory            = A->mem;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqDense"
PetscErrorCode MatScale_SeqDense(const PetscScalar *alpha,Mat A)
{
  Mat_SeqDense *a = (Mat_SeqDense*)A->data;
  PetscBLASInt one = 1,lda = a->lda,j,nz;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lda>A->m) {
    nz = (PetscBLASInt)A->m;
    for (j=0; j<A->n; j++) {
      BLASscal_(&nz,(PetscScalar*)alpha,a->v+j*lda,&one);
    }
  } else {
    nz = (PetscBLASInt)A->m*A->n;
    BLASscal_(&nz,(PetscScalar*)alpha,a->v,&one);
  }
  ierr = PetscLogFlops(nz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactor_SeqDense"
PetscErrorCode MatLUFactor_SeqDense(Mat A,IS row,IS col,MatFactorInfo *minfo)
{
#if defined(PETSC_MISSING_LAPACK_GETRF) 
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"GETRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n = (PetscBLASInt)A->n,m = (PetscBLASInt)A->m,info;

  PetscFunctionBegin;
  if (!mat->pivots) {
    ierr = PetscMalloc((A->m+1)*sizeof(PetscBLASInt),&mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,A->m*sizeof(PetscBLASInt));CHKERRQ(ierr);
  }
  A->factor = FACTOR_LU;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  LAPACKgetrf_(&m,&n,mat->v,&mat->lda,mat->pivots,&info);
  if (info<0) SETERRQ(PETSC_ERR_LIB,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,"Bad LU factorization");
  ierr = PetscLogFlops((2*A->n*A->n*A->n)/3);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqDense"
PetscErrorCode MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data,*l;
  PetscErrorCode ierr;
  PetscInt       lda = (PetscInt)mat->lda,j,m;
  Mat            newi;

  PetscFunctionBegin;
  ierr = MatCreate(A->comm,A->m,A->n,A->m,A->n,&newi);CHKERRQ(ierr);
  ierr = MatSetType(newi,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(newi,PETSC_NULL);CHKERRQ(ierr);
  if (cpvalues == MAT_COPY_VALUES) {
    l = (Mat_SeqDense*)newi->data;
    if (lda>A->m) {
      m = A->m;
      for (j=0; j<A->n; j++) {
	ierr = PetscMemcpy(l->v+j*m,mat->v+j*lda,m*sizeof(PetscScalar));CHKERRQ(ierr);
      }
    } else {
      ierr = PetscMemcpy(l->v,mat->v,A->m*A->n*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }
  newi->assembled = PETSC_TRUE;
  *newmat = newi;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqDense"
PetscErrorCode MatLUFactorSymbolic_SeqDense(Mat A,IS row,IS col,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqDense(A,MAT_DO_NOT_COPY_VALUES,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqDense"
PetscErrorCode MatLUFactorNumeric_SeqDense(Mat A,MatFactorInfo *info_dummy,Mat *fact)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data,*l = (Mat_SeqDense*)(*fact)->data;
  PetscErrorCode ierr;
  PetscInt       lda1=mat->lda,lda2=l->lda, m=A->m,n=A->n, j;
  MatFactorInfo  info; 

  PetscFunctionBegin;
  /* copy the numerical values */
  if (lda1>m || lda2>m ) {
    for (j=0; j<n; j++) {
      ierr = PetscMemcpy(l->v+j*lda2,mat->v+j*lda1,m*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMemcpy(l->v,mat->v,A->m*A->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  (*fact)->factor = 0;
  ierr = MatLUFactor(*fact,0,0,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SeqDense"
PetscErrorCode MatCholeskyFactorSymbolic_SeqDense(Mat A,IS row,MatFactorInfo *info,Mat *fact)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactor_SeqDense"
PetscErrorCode MatCholeskyFactor_SeqDense(Mat A,IS perm,MatFactorInfo *factinfo)
{
#if defined(PETSC_MISSING_LAPACK_POTRF) 
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_SUP,"POTRF - Lapack routine is unavailable.");
#else
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   n = (PetscBLASInt)A->n,info;
  
  PetscFunctionBegin;
  if (mat->pivots) {
    ierr = PetscFree(mat->pivots);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(A,-A->m*sizeof(PetscInt));CHKERRQ(ierr);
    mat->pivots = 0;
  }
  if (!A->m || !A->n) PetscFunctionReturn(0);
  LAPACKpotrf_("L",&n,mat->v,&mat->lda,&info);
  if (info) SETERRQ1(PETSC_ERR_MAT_CH_ZRPVT,"Bad factorization: zero pivot in row %D",(PetscInt)info-1);
  A->factor = FACTOR_CHOLESKY;
  ierr = PetscLogFlops((A->n*A->n*A->n)/3);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorNumeric_SeqDense"
PetscErrorCode MatCholeskyFactorNumeric_SeqDense(Mat A,MatFactorInfo *info_dummy,Mat *fact)
{
  PetscErrorCode ierr;
  MatFactorInfo  info;

  PetscFunctionBegin;
  info.fill = 1.0;
  ierr = MatCholeskyFactor_SeqDense(*fact,0,&info);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqDense"
PetscErrorCode MatSolve_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m, one = 1,info;
  PetscScalar    *x,*y;
  
  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->m*sizeof(PetscScalar));CHKERRQ(ierr);
  if (A->factor == FACTOR_LU) {
#if defined(PETSC_MISSING_LAPACK_GETRS) 
    SETERRQ(PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("N",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"GETRS - Bad solve");
#endif
  } else if (A->factor == FACTOR_CHOLESKY){
#if defined(PETSC_MISSING_LAPACK_POTRS) 
    SETERRQ(PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"POTRS Bad solve");
#endif
  }
  else SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Matrix must be factored to solve");
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->n*A->n - A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_SeqDense"
PetscErrorCode MatSolveTranspose_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt) A->m,one = 1,info;
  PetscScalar    *x,*y;
  
  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,A->m*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS) 
    SETERRQ(PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("T",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"POTRS - Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS) 
    SETERRQ(PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"POTRS - Bad solve");
#endif
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->n*A->n - A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveAdd_SeqDense"
PetscErrorCode MatSolveAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m,one = 1,info;
  PetscScalar    *x,*y,sone = 1.0;
  Vec            tmp = 0;
  
  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!A->m || !A->n) PetscFunctionReturn(0);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(A,tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  } 
  ierr = PetscMemcpy(y,x,A->m*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS) 
    SETERRQ(PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("N",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS) 
    SETERRQ(PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"Bad solve");
#endif
  }
  if (tmp) {ierr = VecAXPY(&sone,tmp,yy);CHKERRQ(ierr); ierr = VecDestroy(tmp);CHKERRQ(ierr);}
  else     {ierr = VecAXPY(&sone,zz,yy);CHKERRQ(ierr);}
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->n*A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTransposeAdd_SeqDense"
PetscErrorCode MatSolveTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m,one = 1,info;
  PetscScalar    *x,*y,sone = 1.0;
  Vec            tmp;
  
  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    ierr = PetscLogObjectParent(A,tmp);CHKERRQ(ierr);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  } 
  ierr = PetscMemcpy(y,x,A->m*sizeof(PetscScalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
#if defined(PETSC_MISSING_LAPACK_GETRS) 
    SETERRQ(PETSC_ERR_SUP,"GETRS - Lapack routine is unavailable.");
#else
    LAPACKgetrs_("T",&m,&one,mat->v,&mat->lda,mat->pivots,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"Bad solve");
#endif
  } else {
#if defined(PETSC_MISSING_LAPACK_POTRS) 
    SETERRQ(PETSC_ERR_SUP,"POTRS - Lapack routine is unavailable.");
#else
    LAPACKpotrs_("L",&m,&one,mat->v,&mat->lda,y,&m,&info);
    if (info) SETERRQ(PETSC_ERR_LIB,"Bad solve");
#endif
  }
  if (tmp) {
    ierr = VecAXPY(&sone,tmp,yy);CHKERRQ(ierr);
    ierr = VecDestroy(tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(&sone,zz,yy);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->n*A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatRelax_SeqDense"
PetscErrorCode MatRelax_SeqDense(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *x,*b,*v = mat->v,zero = 0.0,xt;
  PetscErrorCode ierr;
  PetscInt       m = A->m,i;
#if !defined(PETSC_USE_COMPLEX)
  PetscBLASInt   bm = (PetscBLASInt)m, o = 1;
#endif

  PetscFunctionBegin;
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLASdot */
    ierr = VecSet(&zero,xx);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  its  = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for (i=0; i<m; i++) {
#if defined(PETSC_USE_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        PetscInt         _i;
        PetscScalar sum = b[i];
        for (_i=0; _i<m; _i++) {
          sum -= PetscConj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i] - BLASdot_(&bm,v+i,&bm,x,&o);
#endif
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (i=m-1; i>=0; i--) {
#if defined(PETSC_USE_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        PetscInt         _i;
        PetscScalar sum = b[i];
        for (_i=0; _i<m; _i++) {
          sum -= PetscConj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i] - BLASdot_(&bm,v+i,&bm,x,&o);
#endif
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
  PetscBLASInt   m = (PetscBLASInt)A->m, n = (PetscBLASInt)A->n,_One=1;
  PetscScalar    _DOne=1.0,_DZero=0.0;

  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("T",&m,&n,&_DOne,v,&mat->lda,x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->m*A->n - A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqDense"
PetscErrorCode MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y,_DOne=1.0,_DZero=0.0;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m, n = (PetscBLASInt)A->n, _One=1;

  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->m*A->n - A->m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqDense"
PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y,_DOne=1.0;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m, n = (PetscBLASInt)A->n, _One=1;

  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr); 
  BLASgemv_("N",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->m*A->n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqDense"
PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscScalar    *v = mat->v,*x,*y;
  PetscErrorCode ierr;
  PetscBLASInt   m = (PetscBLASInt)A->m, n = (PetscBLASInt)A->n, _One=1;
  PetscScalar    _DOne=1.0;

  PetscFunctionBegin;
  if (!A->m || !A->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  BLASgemv_("T",&m,&n,&_DOne,v,&(mat->lda),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscLogFlops(2*A->m*A->n);CHKERRQ(ierr);
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
  *ncols = A->n;
  if (cols) {
    ierr = PetscMalloc((A->n+1)*sizeof(PetscInt),cols);CHKERRQ(ierr);
    for (i=0; i<A->n; i++) (*cols)[i] = i;
  }
  if (vals) {
    ierr = PetscMalloc((A->n+1)*sizeof(PetscScalar),vals);CHKERRQ(ierr);
    v    = mat->v + row;
    for (i=0; i<A->n; i++) {(*vals)[i] = *v; v += mat->lda;}
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
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
#if defined(PETSC_USE_DEBUG)  
        if (indexn[j] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->n-1);
#endif
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
#if defined(PETSC_USE_DEBUG)  
          if (indexm[i] >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->m-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (j=0; j<n; j++) {
        if (indexn[j] < 0) {idx += m; continue;}
#if defined(PETSC_USE_DEBUG)  
        if (indexn[j] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->n-1);
#endif
        for (i=0; i<m; i++) {
          if (indexm[i] < 0) {idx++; continue;}
#if defined(PETSC_USE_DEBUG)  
          if (indexm[i] >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->m-1);
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
        if (indexm[i] >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->m-1);
#endif
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
#if defined(PETSC_USE_DEBUG)  
          if (indexn[j] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->n-1);
#endif
          mat->v[indexn[j]*mat->lda + indexm[i]] = v[idx++];
        }
      }
    } else {
      for (i=0; i<m; i++) {
        if (indexm[i] < 0) { idx += n; continue;}
#if defined(PETSC_USE_DEBUG)  
        if (indexm[i] >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",indexm[i],A->m-1);
#endif
        for (j=0; j<n; j++) {
          if (indexn[j] < 0) { idx++; continue;}
#if defined(PETSC_USE_DEBUG)  
          if (indexn[j] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",indexn[j],A->n-1);
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
  PetscScalar  *vpt = v;

  PetscFunctionBegin;
  /* row-oriented output */ 
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      *vpt++ = mat->v[indexn[j]*mat->lda + indexm[i]];
    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqDense"
PetscErrorCode MatLoad_SeqDense(PetscViewer viewer,const MatType type,Mat *A)
{
  Mat_SeqDense   *a;
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       *scols,i,j,nz,header[4];
  int            fd;
  PetscMPIInt    size;
  PetscInt       *rowlengths = 0,M,N,*cols;
  PetscScalar    *vals,*svals,*v,*w;
  MPI_Comm       comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz == MATRIX_BINARY_FORMAT_DENSE) { /* matrix in file is dense */
    ierr = MatCreate(comm,M,N,M,N,A);CHKERRQ(ierr);
    ierr = MatSetType(*A,type);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*A,PETSC_NULL);CHKERRQ(ierr);
    B    = *A;
    a    = (Mat_SeqDense*)B->data;
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
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    /* read row lengths */
    ierr = PetscMalloc((M+1)*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

    /* create our matrix */   
    ierr = MatCreate(comm,M,N,M,N,A);CHKERRQ(ierr);
    ierr = MatSetType(*A,type);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(*A,PETSC_NULL);CHKERRQ(ierr);
    B = *A;
    a = (Mat_SeqDense*)B->data;
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

    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#include "petscsys.h"

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqDense_ASCII"
static PetscErrorCode MatView_SeqDense_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqDense      *a = (Mat_SeqDense*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j;
  char              *name;
  PetscScalar       *v;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<A->m; i++) {
      v = a->v + i;
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=0; j<A->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(*v) != 0.0 && PetscImaginaryPart(*v) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i) ",j,PetscRealPart(*v),PetscImaginaryPart(*v));CHKERRQ(ierr);
        } else if (PetscRealPart(*v)) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,PetscRealPart(*v));CHKERRQ(ierr);
        }
#else
        if (*v) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",j,*v);CHKERRQ(ierr);
        }
#endif
        v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    PetscTruth allreal = PETSC_TRUE;
    /* determine if matrix has all real values */
    v = a->v;
    for (i=0; i<A->m*A->n; i++) {
	if (PetscImaginaryPart(v[i])) { allreal = PETSC_FALSE; break ;}
    }
#endif
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",A->m,A->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = zeros(%D,%D);\n",name,A->m,A->n);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"%s = [\n",name);CHKERRQ(ierr);
    }

    for (i=0; i<A->m; i++) {
      v = a->v + i;
      for (j=0; j<A->n; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (allreal) {
          ierr = PetscViewerASCIIPrintf(viewer,"%6.4e ",PetscRealPart(*v));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer,"%6.4e + %6.4e i ",PetscRealPart(*v),PetscImaginaryPart(*v));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%6.4e ",*v);CHKERRQ(ierr);
#endif
	v += a->lda;
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      ierr = PetscViewerASCIIPrintf(viewer,"];\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
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
  PetscInt          ict,j,n = A->n,m = A->m,i,*col_lens,nz = m*n;
  PetscScalar       *v,*anonz,*vals;
  PetscViewerFormat format;
  
  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_BINARY_NATIVE) {
    /* store the matrix as a dense matrix */
    ierr = PetscMalloc(4*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
    col_lens[0] = MAT_FILE_COOKIE;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = MATRIX_BINARY_FORMAT_DENSE;
    ierr = PetscBinaryWrite(fd,col_lens,4,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* write out matrix, by rows */
    ierr = PetscMalloc((m*n+1)*sizeof(PetscScalar),&vals);CHKERRQ(ierr);
    v    = a->v;
    for (i=0; i<m; i++) {
      for (j=0; j<n; j++) {
        vals[i + j*m] = *v++;
      }
    }
    ierr = PetscBinaryWrite(fd,vals,n*m,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc((4+nz)*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
    col_lens[0] = MAT_FILE_COOKIE;
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
  PetscInt          m = A->m,n = A->n,color,i,j;
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
    for(j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for(i = 0; i < m; i++) {
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
    for(i = 0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    scale = (245.0 - PETSC_DRAW_BASIC_COLORS)/maxv; 
    ierr  = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    if (popup) {ierr  = PetscDrawScalePopup(popup,0.0,maxv);CHKERRQ(ierr);}
    for(j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for(i = 0; i < m; i++) {
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
  PetscTruth     isnull;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->n; yr = A->m; h = yr/10.0; w = xr/10.0; 
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
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscTruth     issocket,iascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);

  if (issocket) {
    if (a->lda>A->m) SETERRQ(PETSC_ERR_SUP,"Case can not handle LDA");
    ierr = PetscViewerSocketPutScalar(viewer,A->m,A->n,a->v);CHKERRQ(ierr);
  } else if (iascii) {
    ierr = MatView_SeqDense_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqDense_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqDense_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by dense matrix",((PetscObject)viewer)->type_name);
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
  PetscLogObjectState((PetscObject)mat,"Rows %D Cols %D",mat->m,mat->n);
#endif
  if (l->pivots) {ierr = PetscFree(l->pivots);CHKERRQ(ierr);}
  if (!l->user_alloc) {ierr = PetscFree(l->v);CHKERRQ(ierr);}
  ierr = PetscFree(l);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)mat,"MatSeqDenseSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqDense"
PetscErrorCode MatTranspose_SeqDense(Mat A,Mat *matout)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       k,j,m,n,M;
  PetscScalar    *v,tmp;

  PetscFunctionBegin;
  v = mat->v; m = A->m; M = mat->lda; n = A->n;
  if (!matout) { /* in place transpose */
    if (m != n) {
      SETERRQ(PETSC_ERR_SUP,"Can not transpose non-square matrix in place");
    } else {
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

    ierr  = MatCreate(A->comm,A->n,A->m,A->n,A->m,&tmat);CHKERRQ(ierr);
    ierr  = MatSetType(tmat,A->type_name);CHKERRQ(ierr);
    ierr  = MatSeqDenseSetPreallocation(tmat,PETSC_NULL);CHKERRQ(ierr);
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
PetscErrorCode MatEqual_SeqDense(Mat A1,Mat A2,PetscTruth *flg)
{
  Mat_SeqDense *mat1 = (Mat_SeqDense*)A1->data;
  Mat_SeqDense *mat2 = (Mat_SeqDense*)A2->data;
  PetscInt     i,j;
  PetscScalar  *v1 = mat1->v,*v2 = mat2->v;

  PetscFunctionBegin;
  if (A1->m != A2->m) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (A1->n != A2->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  for (i=0; i<A1->m; i++) {
    v1 = mat1->v+i; v2 = mat2->v+i;
    for (j=0; j<A1->n; j++) {
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
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  len = PetscMin(A->m,A->n);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming mat and vec");
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
  PetscInt       i,j,m = A->m,n = A->n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    if (m != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Left scaling vec wrong size");
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
    if (n != A->n) SETERRQ(PETSC_ERR_ARG_SIZ,"Right scaling vec wrong size");
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
  PetscInt     lda=mat->lda,m=A->m,i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    if (lda>m) {
      for (j=0; j<A->n; j++) {
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
      for (i=0; i<A->n*A->m; i++) {
#if defined(PETSC_USE_COMPLEX)
	sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
	sum += (*v)*(*v); v++;
#endif
      }
    }
    *nrm = sqrt(sum);
    ierr = PetscLogFlops(2*A->n*A->m);CHKERRQ(ierr);
  } else if (type == NORM_1) {
    *nrm = 0.0;
    for (j=0; j<A->n; j++) {
      v = mat->v + j*mat->lda;
      sum = 0.0;
      for (i=0; i<A->m; i++) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(A->n*A->m);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->m; j++) {
      v = mat->v + j;
      sum = 0.0;
      for (i=0; i<A->n; i++) {
        sum += PetscAbsScalar(*v); v += mat->lda;
      }
      if (sum > *nrm) *nrm = sum;
    }
    ierr = PetscLogFlops(A->n*A->m);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"No two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqDense"
PetscErrorCode MatSetOption_SeqDense(Mat A,MatOption op)
{
  Mat_SeqDense   *aij = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    aij->roworiented = PETSC_TRUE;
    break;
  case MAT_COLUMN_ORIENTED:
    aij->roworiented = PETSC_FALSE;
    break;
  case MAT_ROWS_SORTED: 
  case MAT_ROWS_UNSORTED:
  case MAT_COLUMNS_SORTED:
  case MAT_COLUMNS_UNSORTED:
  case MAT_NO_NEW_NONZERO_LOCATIONS:
  case MAT_YES_NEW_NONZERO_LOCATIONS:
  case MAT_NEW_NONZERO_LOCATION_ERR:
  case MAT_NO_NEW_DIAGONALS:
  case MAT_YES_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    ierr = PetscLogInfo((A,"MatSetOption_SeqDense:Option ignored\n"));CHKERRQ(ierr);
    break;
  case MAT_SYMMETRIC:
  case MAT_STRUCTURALLY_SYMMETRIC:
  case MAT_NOT_SYMMETRIC:
  case MAT_NOT_STRUCTURALLY_SYMMETRIC:
  case MAT_HERMITIAN:
  case MAT_NOT_HERMITIAN:
  case MAT_SYMMETRY_ETERNAL:
  case MAT_NOT_SYMMETRY_ETERNAL:
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqDense"
PetscErrorCode MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       lda=l->lda,m=A->m,j;

  PetscFunctionBegin;
  if (lda>m) {
    for (j=0; j<A->n; j++) {
      ierr = PetscMemzero(l->v+j*lda,m*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMemzero(l->v,A->m*A->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqDense"
PetscErrorCode MatZeroRows_SeqDense(Mat A,IS is,const PetscScalar *diag)
{
  Mat_SeqDense   *l = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       n = A->n,i,j,N,*rows;
  PetscScalar    *slot;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  for (i=0; i<N; i++) {
    slot = l->v + rows[i];
    for (j=0; j<n; j++) { *slot = 0.0; slot += n;}
  }
  if (diag) {
    for (i=0; i<N; i++) { 
      slot = l->v + (n+1)*rows[i];
      *slot = *diag;
    }
  }
  ISRestoreIndices(is,&rows);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqDense"
PetscErrorCode MatGetArray_SeqDense(Mat A,PetscScalar *array[])
{
  Mat_SeqDense *mat = (Mat_SeqDense*)A->data;

  PetscFunctionBegin;
  *array = mat->v;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqDense"
PetscErrorCode MatRestoreArray_SeqDense(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  *array = 0; /* user cannot accidently use the array later */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqDense"
static PetscErrorCode MatGetSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,PetscInt cs,MatReuse scall,Mat *B)
{
  Mat_SeqDense   *mat = (Mat_SeqDense*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->m,*irow,*icol,nrows,ncols;
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
    if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size");
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreate(A->comm,nrows,ncols,nrows,ncols,&newmat);CHKERRQ(ierr);
    ierr = MatSetType(newmat,A->type_name);CHKERRQ(ierr);
    ierr = MatSeqDenseSetPreallocation(newmat,PETSC_NULL);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy,column at a time */
  bv = ((Mat_SeqDense*)newmat->data)->v;
  
  for (i=0; i<ncols; i++) {
    av = v + m*icol[i];
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
#define __FUNCT__ "MatCopy_SeqDense"
PetscErrorCode MatCopy_SeqDense(Mat A,Mat B,MatStructure str)
{
  Mat_SeqDense   *a = (Mat_SeqDense*)A->data,*b = (Mat_SeqDense *)B->data;
  PetscErrorCode ierr;
  PetscInt       lda1=a->lda,lda2=b->lda, m=A->m,n=A->n, j;

  PetscFunctionBegin;
  /* If the two matrices don't have the same copy implementation, they aren't compatible for fast copy. */
  if (A->ops->copy != B->ops->copy) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (m != B->m || n != B->n) SETERRQ(PETSC_ERR_ARG_SIZ,"size(B) != size(A)");
  if (lda1>m || lda2>m) {
    for (j=0; j<n; j++) {
      ierr = PetscMemcpy(b->v+j*lda2,a->v+j*lda1,m*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  } else {
    ierr = PetscMemcpy(b->v,a->v,A->m*A->n*sizeof(PetscScalar));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqDense"
PetscErrorCode MatSetUpPreallocation_SeqDense(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqDenseSetPreallocation(A,0);CHKERRQ(ierr);
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
       MatSolve_SeqDense,
       MatSolveAdd_SeqDense,
       MatSolveTranspose_SeqDense,
/*10*/ MatSolveTransposeAdd_SeqDense,
       MatLUFactor_SeqDense,
       MatCholeskyFactor_SeqDense,
       MatRelax_SeqDense,
       MatTranspose_SeqDense,
/*15*/ MatGetInfo_SeqDense,
       MatEqual_SeqDense,
       MatGetDiagonal_SeqDense,
       MatDiagonalScale_SeqDense,
       MatNorm_SeqDense,
/*20*/ 0,
       0,
       0,
       MatSetOption_SeqDense,
       MatZeroEntries_SeqDense,
/*25*/ MatZeroRows_SeqDense,
       MatLUFactorSymbolic_SeqDense,
       MatLUFactorNumeric_SeqDense,
       MatCholeskyFactorSymbolic_SeqDense,
       MatCholeskyFactorNumeric_SeqDense,
/*30*/ MatSetUpPreallocation_SeqDense,
       0,
       0,
       MatGetArray_SeqDense,
       MatRestoreArray_SeqDense,
/*35*/ MatDuplicate_SeqDense,
       0,
       0,
       0,
       0,
/*40*/ MatAXPY_SeqDense,
       MatGetSubMatrices_SeqDense,
       0,
       MatGetValues_SeqDense,
       MatCopy_SeqDense,
/*45*/ 0,
       MatScale_SeqDense,
       0,
       0,
       0,
/*50*/ 0,
       0,
       0,
       0,
       0,
/*55*/ 0,
       0,
       0,
       0,
       0,
/*60*/ 0,
       MatDestroy_SeqDense,
       MatView_SeqDense,
       MatGetPetscMaps_Petsc,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       0,
       0,
       0,
/*75*/ 0,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
/*84*/ MatLoad_SeqDense,
       0,
       0,
       0,
       0,
       0,                                     
/*90*/ 0,
       0,
       0,
       0,
       0,
/*95*/ 0,
       0,
       0,
       0};

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
-  data - optional location of matrix data.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

   Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateMPIDense(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatCreateSeqDense(MPI_Comm comm,PetscInt m,PetscInt n,PetscScalar *data,Mat *A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,m,n,A);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(*A,data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqDensePreallocation"
/*@C
   MatSeqDenseSetPreallocation - Sets the array used for storing the matrix elements

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
-  data - the array (or PETSC_NULL)

   Notes:
   The data input variable is intended primarily for Fortran programmers
   who wish to allocate their own matrix memory space.  Most users should
   set data=PETSC_NULL.

   Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateMPIDense(), MatSetValues()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqDenseSetPreallocation(Mat B,PetscScalar data[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscScalar[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqDenseSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,data);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqDensePreallocation_SeqDense"
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqDenseSetPreallocation_SeqDense(Mat B,PetscScalar *data)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  B->preallocated = PETSC_TRUE;
  b               = (Mat_SeqDense*)B->data;
  if (!data) {
    ierr          = PetscMalloc((B->m*B->n+1)*sizeof(PetscScalar),&b->v);CHKERRQ(ierr);
    ierr          = PetscMemzero(b->v,B->m*B->n*sizeof(PetscScalar));CHKERRQ(ierr);
    b->user_alloc = PETSC_FALSE;
    ierr = PetscLogObjectMemory(B,B->n*B->m*sizeof(PetscScalar));CHKERRQ(ierr);
  } else { /* user-allocated storage */
    b->v          = data;
    b->user_alloc = PETSC_TRUE;
  }
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
  This routine is to be used in conjunction with MatSeqDenseSetPreallocation;
  it asserts that the preallocation has a leading dimension (the LDA parameter
  of Blas and Lapack fame) larger than M, the first dimension of the matrix.

  Level: intermediate

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatCreateSeqDense(), MatSeqDenseSetPreallocation()
@*/
PetscErrorCode PETSCMAT_DLLEXPORT MatSeqDenseSetLDA(Mat B,PetscInt lda)
{
  Mat_SeqDense *b = (Mat_SeqDense*)B->data;
  PetscFunctionBegin;
  if (lda < B->m) SETERRQ2(PETSC_ERR_ARG_SIZ,"LDA %D must be at least matrix dimension %D",lda,B->m);
  b->lda = lda;
  PetscFunctionReturn(0);
}

/*MC
   MATSEQDENSE - MATSEQDENSE = "seqdense" - A matrix type to be used for sequential dense matrices.

   Options Database Keys:
. -mat_type seqdense - sets the matrix type to "seqdense" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqDense
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqDense"
PetscErrorCode PETSCMAT_DLLEXPORT MatCreate_SeqDense(Mat B)
{
  Mat_SeqDense   *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);

  ierr            = PetscNew(Mat_SeqDense,&b);CHKERRQ(ierr);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor       = 0;
  B->mapping      = 0;
  ierr = PetscLogObjectMemory(B,sizeof(struct _p_Mat));CHKERRQ(ierr);
  B->data         = (void*)b;

  ierr = PetscMapCreateMPI(B->comm,B->m,B->m,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->n,&B->cmap);CHKERRQ(ierr);

  b->pivots       = 0;
  b->roworiented  = PETSC_TRUE;
  b->v            = 0;
  b->lda          = B->m;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqDenseSetPreallocation_C",
                                    "MatSeqDenseSetPreallocation_SeqDense",
                                     MatSeqDenseSetPreallocation_SeqDense);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
