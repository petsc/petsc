/*$Id: dense.c,v 1.178 1999/11/05 14:45:14 bsmith Exp bsmith $*/
/*
     Defines the basic matrix operations for sequential dense.
*/

#include "src/mat/impls/dense/seq/dense.h"
#include "pinclude/blaslapack.h"

#undef __FUNC__  
#define __FUNC__ "MatAXPY_SeqDense"
int MatAXPY_SeqDense(Scalar *alpha,Mat X,Mat Y)
{
  Mat_SeqDense *x = (Mat_SeqDense*) X->data,*y = (Mat_SeqDense*) Y->data;
  int          N = x->m*x->n, one = 1;

  PetscFunctionBegin;
  BLaxpy_( &N, alpha, x->v, &one, y->v, &one );
  PLogFlops(2*N-1);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqDense"
int MatGetInfo_SeqDense(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  int          i,N = a->m*a->n,count = 0;
  Scalar       *v = a->v;

  PetscFunctionBegin;
  for ( i=0; i<N; i++ ) {if (*v != 0.0) count++; v++;}

  info->rows_global       = (double)a->m;
  info->columns_global    = (double)a->n;
  info->rows_local        = (double)a->m;
  info->columns_local     = (double)a->n;
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

#undef __FUNC__  
#define __FUNC__ "MatScale_SeqDense"
int MatScale_SeqDense(Scalar *alpha,Mat inA)
{
  Mat_SeqDense *a = (Mat_SeqDense *) inA->data;
  int          one = 1, nz;

  PetscFunctionBegin;
  nz = a->m*a->n;
  BLscal_( &nz, alpha, a->v, &one );
  PLogFlops(nz);
  PetscFunctionReturn(0);
}
  
/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide row permutation in the pivots,
   rather than put it in the Mat->row slot.*/
#undef __FUNC__  
#define __FUNC__ "MatLUFactor_SeqDense"
int MatLUFactor_SeqDense(Mat A,IS row,IS col,double f)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          info;

  PetscFunctionBegin;
  if (!mat->pivots) {
    mat->pivots = (int *) PetscMalloc((mat->m+1)*sizeof(int));CHKPTRQ(mat->pivots);
    PLogObjectMemory(A,mat->m*sizeof(int));
  }
  A->factor = FACTOR_LU;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  LAgetrf_(&mat->m,&mat->n,mat->v,&mat->m,mat->pivots,&info);
  if (info<0) SETERRQ(PETSC_ERR_LIB,0,"Bad argument to LU factorization");
  if (info>0) SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,0,"Bad LU factorization");
  PLogFlops((2*mat->n*mat->n*mat->n)/3);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_SeqDense"
int MatDuplicate_SeqDense(Mat A,MatDuplicateOption cpvalues,Mat *newmat)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data, *l;
  int          ierr;
  Mat          newi;

  PetscFunctionBegin;
  ierr = MatCreateSeqDense(A->comm,mat->m,mat->n,PETSC_NULL,&newi);CHKERRQ(ierr);
  l = (Mat_SeqDense *) newi->data;
  if (cpvalues == MAT_COPY_VALUES) {
    ierr = PetscMemcpy(l->v,mat->v,mat->m*mat->n*sizeof(Scalar));CHKERRQ(ierr);
  }
  newi->assembled = PETSC_TRUE;
  *newmat = newi;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorSymbolic_SeqDense"
int MatLUFactorSymbolic_SeqDense(Mat A,IS row,IS col,double f,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatDuplicate_SeqDense(A,MAT_DO_NOT_COPY_VALUES,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLUFactorNumeric_SeqDense"
int MatLUFactorNumeric_SeqDense(Mat A,Mat *fact)
{
  Mat_SeqDense *mat = (Mat_SeqDense*) A->data, *l = (Mat_SeqDense*) (*fact)->data;
  int          ierr;

  PetscFunctionBegin;
  /* copy the numerical values */
  ierr = PetscMemcpy(l->v,mat->v,mat->m*mat->n*sizeof(Scalar));CHKERRQ(ierr);
  (*fact)->factor = 0;
  ierr = MatLUFactor(*fact,0,0,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorSymbolic_SeqDense"
int MatCholeskyFactorSymbolic_SeqDense(Mat A,IS row,double f,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatConvert(A,MATSAME,fact);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactorNumeric_SeqDense"
int MatCholeskyFactorNumeric_SeqDense(Mat A,Mat *fact)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatCholeskyFactor(*fact,0,1.0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCholeskyFactor_SeqDense"
int MatCholeskyFactor_SeqDense(Mat A,IS perm,double f)
{
  Mat_SeqDense  *mat = (Mat_SeqDense *) A->data;
  int           info,ierr;
  
  PetscFunctionBegin;
  if (mat->pivots) {
    ierr = PetscFree(mat->pivots);CHKERRQ(ierr);
    PLogObjectMemory(A,-mat->m*sizeof(int));
    mat->pivots = 0;
  }
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  LApotrf_("L",&mat->n,mat->v,&mat->m,&info);
  if (info) SETERRQ(PETSC_ERR_MAT_CH_ZRPVT,0,"Bad factorization");
  A->factor = FACTOR_CHOLESKY;
  PLogFlops((mat->n*mat->n*mat->n)/3);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolve_SeqDense"
int MatSolve_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          one = 1, info, ierr;
  Scalar       *x, *y;
  
  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,mat->m*sizeof(Scalar));CHKERRQ(ierr);
  if (A->factor == FACTOR_LU) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  } else if (A->factor == FACTOR_CHOLESKY){
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  else SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Matrix must be factored to solve");
  if (info) SETERRQ(PETSC_ERR_LIB,0,"MBad solve");
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(mat->n*mat->n - mat->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveTrans_SeqDense"
int MatSolveTrans_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          ierr,one = 1, info;
  Scalar       *x, *y;
  
  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  ierr = PetscMemcpy(y,x,mat->m*sizeof(Scalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  } else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(PETSC_ERR_LIB,0,"Bad solve");
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(mat->n*mat->n - mat->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveAdd_SeqDense"
int MatSolveAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          ierr,one = 1, info;
  Scalar       *x, *y, sone = 1.0;
  Vec          tmp = 0;
  
  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  } 
  ierr = PetscMemcpy(y,x,mat->m*sizeof(Scalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  } else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(PETSC_ERR_LIB,0,"Bad solve");
  if (tmp) {ierr = VecAXPY(&sone,tmp,yy);CHKERRQ(ierr); ierr = VecDestroy(tmp);CHKERRQ(ierr);}
  else     {ierr = VecAXPY(&sone,zz,yy);CHKERRQ(ierr);}
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(mat->n*mat->n - mat->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSolveTransAdd_SeqDense"
int MatSolveTransAdd_SeqDense(Mat A,Vec xx,Vec zz, Vec yy)
{
  Mat_SeqDense  *mat = (Mat_SeqDense *) A->data;
  int           one = 1, info,ierr;
  Scalar        *x, *y, sone = 1.0;
  Vec           tmp;
  
  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp);CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp);CHKERRQ(ierr);
  } 
  ierr = PetscMemcpy(y,x,mat->m*sizeof(Scalar));CHKERRQ(ierr);
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  } else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(PETSC_ERR_LIB,0,"Bad solve");
  if (tmp) {
    ierr = VecAXPY(&sone,tmp,yy);CHKERRQ(ierr);
    ierr = VecDestroy(tmp);CHKERRQ(ierr);
  } else {
    ierr = VecAXPY(&sone,zz,yy);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(mat->n*mat->n - mat->n);
  PetscFunctionReturn(0);
}
/* ------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatRelax_SeqDense"
int MatRelax_SeqDense(Mat A,Vec bb,double omega,MatSORType flag,
                          double shift,int its,Vec xx)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *x, *b, *v = mat->v, zero = 0.0, xt;
  int          ierr, m = mat->m, i;
#if !defined(PETSC_USE_COMPLEX)
  int          o = 1;
#endif

  PetscFunctionBegin;
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without the second BLdot */
    ierr = VecSet(&zero,xx);CHKERRQ(ierr);
  }
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
#if defined(PETSC_USE_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        int    _i;
        Scalar sum = b[i];
        for ( _i=0; _i<m; _i++ ) {
          sum -= PetscConj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
#endif
        x[i] = (1. - omega)*x[i] + omega*(xt+v[i + i*m]*x[i])/(v[i + i*m]+shift);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP) {
      for ( i=m-1; i>=0; i-- ) {
#if defined(PETSC_USE_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        int    _i;
        Scalar sum = b[i];
        for ( _i=0; _i<m; _i++ ) {
          sum -= PetscConj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
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
#undef __FUNC__  
#define __FUNC__ "MatMultTrans_SeqDense"
int MatMultTrans_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          ierr,_One=1;Scalar _DOne=1.0, _DZero=0.0;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  LAgemv_("T",&(mat->m),&(mat->n),&_DOne,v,&(mat->m),x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(2*mat->m*mat->n - mat->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_SeqDense"
int MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          ierr,_One=1;Scalar _DOne=1.0, _DZero=0.0;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m),x,&_One,&_DZero,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(2*mat->m*mat->n - mat->m);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_SeqDense"
int MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          ierr,_One=1; Scalar _DOne=1.0;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr); 
  LAgemv_( "N", &(mat->m), &(mat->n),&_DOne,v,&(mat->m),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(2*mat->m*mat->n);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransAdd_SeqDense"
int MatMultTransAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          ierr,_One=1;
  Scalar       _DOne=1.0;

  PetscFunctionBegin;
  if (!mat->m || !mat->n) PetscFunctionReturn(0);
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m),x,&_One,&_DOne,y,&_One);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PLogFlops(2*mat->m*mat->n);
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqDense"
int MatGetRow_SeqDense(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v;
  int          i;
  
  PetscFunctionBegin;
  *ncols = mat->n;
  if (cols) {
    *cols = (int *) PetscMalloc((mat->n+1)*sizeof(int));CHKPTRQ(*cols);
    for ( i=0; i<mat->n; i++ ) (*cols)[i] = i;
  }
  if (vals) {
    *vals = (Scalar *) PetscMalloc((mat->n+1)*sizeof(Scalar));CHKPTRQ(*vals);
    v = mat->v + row;
    for ( i=0; i<mat->n; i++ ) {(*vals)[i] = *v; v += mat->m;}
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqDense"
int MatRestoreRow_SeqDense(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  int ierr;
  PetscFunctionBegin;
  if (cols) {ierr = PetscFree(*cols);CHKERRQ(ierr);}
  if (vals) {ierr = PetscFree(*vals);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}
/* ----------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatSetValues_SeqDense"
int MatSetValues_SeqDense(Mat A,int m,int *indexm,int n,
                                    int *indexn,Scalar *v,InsertMode addv)
{ 
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i,j;
 
  PetscFunctionBegin;
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
      for ( j=0; j<n; j++ ) {
        if (indexn[j] < 0) {v += m; continue;}
#if defined(PETSC_USE_BOPT_g)  
        if (indexn[j] >= A->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
        for ( i=0; i<m; i++ ) {
          if (indexm[i] < 0) {v++; continue;}
#if defined(PETSC_USE_BOPT_g)  
          if (indexm[i] >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    } else {
      for ( j=0; j<n; j++ ) {
        if (indexn[j] < 0) {v += m; continue;}
#if defined(PETSC_USE_BOPT_g)  
        if (indexn[j] >= A->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
        for ( i=0; i<m; i++ ) {
          if (indexm[i] < 0) {v++; continue;}
#if defined(PETSC_USE_BOPT_g)  
          if (indexm[i] >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  } else {
    if (addv == INSERT_VALUES) {
      for ( i=0; i<m; i++ ) {
        if (indexm[i] < 0) { v += n; continue;}
#if defined(PETSC_USE_BOPT_g)  
        if (indexm[i] >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
        for ( j=0; j<n; j++ ) {
          if (indexn[j] < 0) { v++; continue;}
#if defined(PETSC_USE_BOPT_g)  
          if (indexn[j] >= A->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    } else {
      for ( i=0; i<m; i++ ) {
        if (indexm[i] < 0) { v += n; continue;}
#if defined(PETSC_USE_BOPT_g)  
        if (indexm[i] >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
#endif
        for ( j=0; j<n; j++ ) {
          if (indexn[j] < 0) { v++; continue;}
#if defined(PETSC_USE_BOPT_g)  
          if (indexn[j] >= A->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
#endif
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetValues_SeqDense"
int MatGetValues_SeqDense(Mat A,int m,int *indexm,int n,int *indexn,Scalar *v)
{ 
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i, j;
  Scalar       *vpt = v;

  PetscFunctionBegin;
  /* row-oriented output */ 
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      *vpt++ = mat->v[indexn[j]*mat->m + indexm[i]];
    }
  }
  PetscFunctionReturn(0);
}

/* -----------------------------------------------------------------*/

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqDense"
int MatLoad_SeqDense(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqDense *a;
  Mat          B;
  int          *scols, i, j, nz, ierr, fd, header[4], size;
  int          *rowlengths = 0, M, N, *cols;
  Scalar       *vals, *svals, *v,*w;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz == MATRIX_BINARY_FORMAT_DENSE) { /* matrix in file is dense */
    ierr = MatCreateSeqDense(comm,M,N,PETSC_NULL,A);CHKERRQ(ierr);
    B    = *A;
    a    = (Mat_SeqDense *) B->data;
    v    = a->v;
    /* Allocate some temp space to read in the values and then flip them
       from row major to column major */
    w = (Scalar *) PetscMalloc((M*N+1)*sizeof(Scalar));CHKPTRQ(w);
    /* read in nonzero values */
    ierr = PetscBinaryRead(fd,w,M*N,PETSC_SCALAR);CHKERRQ(ierr);
    /* now flip the values and store them in the matrix*/
    for ( j=0; j<N; j++ ) {
      for ( i=0; i<M; i++ ) {
        *v++ =w[i*N+j];
      }
    }
    ierr = PetscFree(w);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  } else {
    /* read row lengths */
    rowlengths = (int*) PetscMalloc( (M+1)*sizeof(int) );CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

    /* create our matrix */
    ierr = MatCreateSeqDense(comm,M,N,PETSC_NULL,A);CHKERRQ(ierr);
    B = *A;
    a = (Mat_SeqDense *) B->data;
    v = a->v;

    /* read column indices and nonzeros */
    cols = scols = (int *) PetscMalloc( (nz+1)*sizeof(int) );CHKPTRQ(cols);
    ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT);CHKERRQ(ierr);
    vals = svals = (Scalar *) PetscMalloc( (nz+1)*sizeof(Scalar) );CHKPTRQ(vals);
    ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR);CHKERRQ(ierr);

    /* insert into matrix */  
    for ( i=0; i<M; i++ ) {
      for ( j=0; j<rowlengths[i]; j++ ) v[i+M*scols[j]] = svals[j];
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

#include "sys.h"

#undef __FUNC__  
#define __FUNC__ "MatView_SeqDense_ASCII"
static int MatView_SeqDense_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  int          ierr, i, j, format;
  char         *outputname;
  Scalar       *v;

  PetscFunctionBegin;
  ierr = ViewerGetOutputname(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    PetscFunctionReturn(0);  /* do nothing for now */
  } else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for ( i=0; i<a->m; i++ ) {
      v = a->v + i;
      ierr = ViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      for ( j=0; j<a->n; j++ ) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscReal(*v) != 0.0 && PetscImaginary(*v) != 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g + %g i",j,PetscReal(*v),PetscImaginary(*v));CHKERRQ(ierr);
        } else if (PetscReal(*v)) {
          ierr = ViewerASCIIPrintf(viewer," %d %g ",j,PetscReal(*v));CHKERRQ(ierr);
        }
#else
        if (*v) {
          ierr = ViewerASCIIPrintf(viewer," %d %g ",j,*v);CHKERRQ(ierr);
        }
#endif
        v += a->m;
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
    int allreal = 1;
    /* determine if matrix has all real values */
    v = a->v;
    for ( i=0; i<a->m*a->n; i++ ) {
      if (PetscImaginary(v[i])) { allreal = 0; break ;}
    }
#endif
    for ( i=0; i<a->m; i++ ) {
      v = a->v + i;
      for ( j=0; j<a->n; j++ ) {
#if defined(PETSC_USE_COMPLEX)
        if (allreal) {
          ierr = ViewerASCIIPrintf(viewer,"%6.4e ",PetscReal(*v));CHKERRQ(ierr); v += a->m;
        } else {
          ierr = ViewerASCIIPrintf(viewer,"%6.4e + %6.4e i ",PetscReal(*v),PetscImaginary(*v));CHKERRQ(ierr); v += a->m;
        }
#else
        ierr = ViewerASCIIPrintf(viewer,"%6.4e ",*v);CHKERRQ(ierr); v += a->m;
#endif
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = ViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqDense_Binary"
static int MatView_SeqDense_Binary(Mat A,Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  int          ict, j, n = a->n, m = a->m, i, fd, *col_lens, ierr, nz = m*n;
  int          format;
  Scalar       *v, *anonz,*vals;
  
  PetscFunctionBegin;
  ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_BINARY_NATIVE) {
    /* store the matrix as a dense matrix */
    col_lens = (int *) PetscMalloc( 4*sizeof(int) );CHKPTRQ(col_lens);
    col_lens[0] = MAT_COOKIE;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = MATRIX_BINARY_FORMAT_DENSE;
    ierr = PetscBinaryWrite(fd,col_lens,4,PETSC_INT,1);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* write out matrix, by rows */
    vals = (Scalar *) PetscMalloc((m*n+1)*sizeof(Scalar));CHKPTRQ(vals);
    v    = a->v;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<n; j++ ) {
        vals[i + j*m] = *v++;
      }
    }
    ierr = PetscBinaryWrite(fd,vals,n*m,PETSC_SCALAR,0);CHKERRQ(ierr);
    ierr = PetscFree(vals);CHKERRQ(ierr);
  } else {
    col_lens = (int *) PetscMalloc( (4+nz)*sizeof(int) );CHKPTRQ(col_lens);
    col_lens[0] = MAT_COOKIE;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = nz;

    /* store lengths of each row and write (including header) to file */
    for ( i=0; i<m; i++ ) col_lens[4+i] = n;
    ierr = PetscBinaryWrite(fd,col_lens,4+m,PETSC_INT,1);CHKERRQ(ierr);

    /* Possibly should write in smaller increments, not whole matrix at once? */
    /* store column indices (zero start index) */
    ict = 0;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<n; j++ ) col_lens[ict++] = j;
    }
    ierr = PetscBinaryWrite(fd,col_lens,nz,PETSC_INT,0);CHKERRQ(ierr);
    ierr = PetscFree(col_lens);CHKERRQ(ierr);

    /* store nonzero values */
    anonz = (Scalar *) PetscMalloc((nz+1)*sizeof(Scalar));CHKPTRQ(anonz);
    ict = 0;
    for ( i=0; i<m; i++ ) {
      v = a->v + i;
      for ( j=0; j<n; j++ ) {
        anonz[ict++] = *v; v += a->m;
      }
    }
    ierr = PetscBinaryWrite(fd,anonz,nz,PETSC_SCALAR,0);CHKERRQ(ierr);
    ierr = PetscFree(anonz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqDense_Draw_Zoom"
int MatView_SeqDense_Draw_Zoom(Draw draw, void *Aa)
{
  Mat           A = (Mat) Aa;
  Mat_SeqDense  *a = (Mat_SeqDense *) A->data;
  int           m = a->m,n = a->n,format, color,i, j,ierr;
  Scalar        *v = a->v;
  Viewer        viewer;
  Draw          popup;
  double        xl, yl, xr, yr, x_l, x_r, y_l, y_r, scale, maxv = 0.0;

  PetscFunctionBegin; 

  ierr = PetscObjectQuery((PetscObject) A, "Zoomviewer", (PetscObject*) &viewer);CHKERRQ(ierr); 
  ierr = ViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  ierr = DrawGetCoordinates(draw, &xl, &yl, &xr, &yr);CHKERRQ(ierr);

  /* Loop over matrix elements drawing boxes */
  if (format != VIEWER_FORMAT_DRAW_CONTOUR) {
    /* Blue for negative and Red for positive */
    color = DRAW_BLUE;
    for(j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for(i = 0; i < m; i++) {
        y_l = m - i - 1.0;
        y_r = y_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscReal(v[j*m+i]) >  0.) {
          color = DRAW_RED;
        } else if (PetscReal(v[j*m+i]) <  0.) {
          color = DRAW_BLUE;
        } else {
          continue;
        }
#else
        if (v[j*m+i] >  0.) {
          color = DRAW_RED;
        } else if (v[j*m+i] <  0.) {
          color = DRAW_BLUE;
        } else {
          continue;
        }
#endif
        ierr = DrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color);CHKERRQ(ierr);
      } 
    }
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    for(i = 0; i < m*n; i++) {
      if (PetscAbsScalar(v[i]) > maxv) maxv = PetscAbsScalar(v[i]);
    }
    scale = (245.0 - DRAW_BASIC_COLORS)/maxv; 
    ierr  = DrawGetPopup(draw, &popup);CHKERRQ(ierr);
    ierr  = DrawScalePopup(popup, 0.0, maxv);CHKERRQ(ierr);
    for(j = 0; j < n; j++) {
      x_l = j;
      x_r = x_l + 1.0;
      for(i = 0; i < m; i++) {
        y_l   = m - i - 1.0;
        y_r   = y_l + 1.0;
        color = DRAW_BASIC_COLORS + (int) (scale*PetscAbsScalar(v[j*m+i]));
        ierr  = DrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color);CHKERRQ(ierr);
      } 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqDense_Draw"
int MatView_SeqDense_Draw(Mat A, Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  Draw       draw;
  PetscTruth isnull;
  double     xr, yr, xl, yl, h, w;
  int        ierr;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer, 0, &draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw, &isnull);CHKERRQ(ierr);
  if (isnull == PETSC_TRUE) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject) A, "Zoomviewer", (PetscObject) viewer);CHKERRQ(ierr);
  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw, xl, yl, xr, yr);CHKERRQ(ierr);
  ierr = DrawZoom(draw, MatView_SeqDense_Draw_Zoom, A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject) A, "Zoomviewer", PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqDense"
int MatView_SeqDense(Mat A,Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense*) A->data;
  int          ierr;
  PetscTruth   issocket,isascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,SOCKET_VIEWER,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);

  if (issocket) {
    ierr = ViewerSocketPutScalar_Private(viewer,a->m,a->n,a->v);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = MatView_SeqDense_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqDense_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqDense_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by dense matrix",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqDense"
int MatDestroy_SeqDense(Mat mat)
{
  Mat_SeqDense *l = (Mat_SeqDense *) mat->data;
  int          ierr;

  PetscFunctionBegin;

  if (mat->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->mapping);CHKERRQ(ierr);
  }
  if (mat->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(mat->bmapping);CHKERRQ(ierr);
  }
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)mat,"Rows %d Cols %d",l->m,l->n);
#endif
  if (l->pivots) {ierr = PetscFree(l->pivots);CHKERRQ(ierr);}
  if (!l->user_alloc) {ierr = PetscFree(l->v);CHKERRQ(ierr);}
  ierr = PetscFree(l);CHKERRQ(ierr);
  if (mat->rmap) {
    ierr = MapDestroy(mat->rmap);CHKERRQ(ierr);
  }
  if (mat->cmap) {
    ierr = MapDestroy(mat->cmap);CHKERRQ(ierr);
  }
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqDense"
int MatTranspose_SeqDense(Mat A,Mat *matout)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          k, j, m, n, ierr;
  Scalar       *v, tmp;

  PetscFunctionBegin;
  v = mat->v; m = mat->m; n = mat->n;
  if (matout == PETSC_NULL) { /* in place transpose */
    if (m != n) { /* malloc temp to hold transpose */
      Scalar *w = (Scalar *) PetscMalloc((m+1)*(n+1)*sizeof(Scalar));CHKPTRQ(w);
      for ( j=0; j<m; j++ ) {
        for ( k=0; k<n; k++ ) {
          w[k + j*n] = v[j + k*m];
        }
      }
      ierr = PetscMemcpy(v,w,m*n*sizeof(Scalar));CHKERRQ(ierr);
      ierr = PetscFree(w);CHKERRQ(ierr);
    } else {
      for ( j=0; j<m; j++ ) {
        for ( k=0; k<j; k++ ) {
          tmp = v[j + k*n]; 
          v[j + k*n] = v[k + j*n];
          v[k + j*n] = tmp;
        }
      }
    }
  } else { /* out-of-place transpose */
    Mat          tmat;
    Mat_SeqDense *tmatd;
    Scalar       *v2;
    ierr = MatCreateSeqDense(A->comm,mat->n,mat->m,PETSC_NULL,&tmat);CHKERRQ(ierr);
    tmatd = (Mat_SeqDense *) tmat->data;
    v = mat->v; v2 = tmatd->v;
    for ( j=0; j<n; j++ ) {
      for ( k=0; k<m; k++ ) v2[j + k*n] = v[k + j*m];
    }
    ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    *matout = tmat;
  } 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatEqual_SeqDense"
int MatEqual_SeqDense(Mat A1,Mat A2, PetscTruth *flg)
{
  Mat_SeqDense *mat1 = (Mat_SeqDense *) A1->data;
  Mat_SeqDense *mat2 = (Mat_SeqDense *) A2->data;
  int          i;
  Scalar       *v1 = mat1->v, *v2 = mat2->v;

  PetscFunctionBegin;
  if (A2->type != MATSEQDENSE) SETERRQ(PETSC_ERR_SUP,0,"Matrices should be of type  MATSEQDENSE");
  if (mat1->m != mat2->m) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  if (mat1->n != mat2->n) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
  for ( i=0; i<mat1->m*mat1->n; i++ ) {
    if (*v1 != *v2) {*flg = PETSC_FALSE; PetscFunctionReturn(0);}
    v1++; v2++;
  }
  *flg = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqDense"
int MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          ierr,i, n, len;
  Scalar       *x, zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetSize(v,&n);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  len = PetscMin(mat->m,mat->n);
  if (n != mat->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  for ( i=0; i<len; i++ ) {
    x[i] = mat->v[i*mat->m + i];
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_SeqDense"
int MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *l,*r,x,*v;
  int          ierr,i,j,m = mat->m, n = mat->n;

  PetscFunctionBegin;
  if (ll) {
    ierr = VecGetSize(ll,&m);CHKERRQ(ierr);
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    if (m != mat->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Left scaling vec wrong size");
    for ( i=0; i<m; i++ ) {
      x = l[i];
      v = mat->v + i;
      for ( j=0; j<n; j++ ) { (*v) *= x; v+= m;} 
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    PLogFlops(n*m);
  }
  if (rr) {
    ierr = VecGetSize(rr,&n);CHKERRQ(ierr);
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr);
    if (n != mat->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Right scaling vec wrong size");
    for ( i=0; i<n; i++ ) {
      x = r[i];
      v = mat->v + i*m;
      for ( j=0; j<m; j++ ) { (*v++) *= x;} 
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr);
    PLogFlops(n*m);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqDense"
int MatNorm_SeqDense(Mat A,NormType type,double *norm)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v;
  double       sum = 0.0;
  int          i, j;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(PETSC_USE_COMPLEX)
      sum += PetscReal(PetscConj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
    PLogFlops(2*mat->n*mat->m);
  } else if (type == NORM_1) {
    *norm = 0.0;
    for ( j=0; j<mat->n; j++ ) {
      sum = 0.0;
      for ( i=0; i<mat->m; i++ ) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *norm) *norm = sum;
    }
    PLogFlops(mat->n*mat->m);
  } else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<mat->m; j++ ) {
      v = mat->v + j;
      sum = 0.0;
      for ( i=0; i<mat->n; i++ ) {
        sum += PetscAbsScalar(*v); v += mat->m;
      }
      if (sum > *norm) *norm = sum;
    }
    PLogFlops(mat->n*mat->m);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"No two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqDense"
int MatSetOption_SeqDense(Mat A,MatOption op)
{
  Mat_SeqDense *aij = (Mat_SeqDense *) A->data;
  
  PetscFunctionBegin;
  if (op == MAT_ROW_ORIENTED)            aij->roworiented = 1;
  else if (op == MAT_COLUMN_ORIENTED)    aij->roworiented = 0;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED ||
           op == MAT_COLUMNS_SORTED ||
           op == MAT_COLUMNS_UNSORTED ||
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_NO_NEW_NONZERO_LOCATIONS ||
           op == MAT_YES_NEW_NONZERO_LOCATIONS ||
           op == MAT_NEW_NONZERO_LOCATION_ERR ||
           op == MAT_NO_NEW_DIAGONALS ||
           op == MAT_YES_NEW_DIAGONALS ||
           op == MAT_IGNORE_OFF_PROC_ENTRIES ||
           op == MAT_USE_HASH_TABLE) {
    PLogInfo(A,"MatSetOption_SeqDense:Option ignored\n");
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_SeqDense"
int MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense *l = (Mat_SeqDense *) A->data;
  int          ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(l->v,l->m*l->n*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqDense"
int MatGetBlockSize_SeqDense(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqDense"
int MatZeroRows_SeqDense(Mat A,IS is,Scalar *diag)
{
  Mat_SeqDense *l = (Mat_SeqDense *) A->data;
  int          n = l->n, i, j,ierr,N, *rows;
  Scalar       *slot;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  for ( i=0; i<N; i++ ) {
    slot = l->v + rows[i];
    for ( j=0; j<n; j++ ) { *slot = 0.0; slot += n;}
  }
  if (diag) {
    for ( i=0; i<N; i++ ) { 
      slot = l->v + (n+1)*rows[i];
      *slot = *diag;
    }
  }
  ISRestoreIndices(is,&rows);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqDense"
int MatGetSize_SeqDense(Mat A,int *m,int *n)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;

  PetscFunctionBegin;
  if (m) *m = mat->m;
  if (n) *n = mat->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqDense"
int MatGetOwnershipRange_SeqDense(Mat A,int *m,int *n)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;

  PetscFunctionBegin;
  *m = 0; *n = mat->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetArray_SeqDense"
int MatGetArray_SeqDense(Mat A,Scalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;

  PetscFunctionBegin;
  *array = mat->v;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreArray_SeqDense"
int MatRestoreArray_SeqDense(Mat A,Scalar **array)
{
  PetscFunctionBegin;
  *array = 0; /* user cannot accidently use the array later */
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqDense"
static int MatGetSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,int cs,MatReuse scall,Mat *B)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i, j, ierr, m = mat->m, *irow, *icol, nrows, ncols;
  Scalar       *av, *bv, *v = mat->v;
  Mat          newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols);CHKERRQ(ierr);
  
  /* Check submatrixcall */
  if (scall == MAT_REUSE_MATRIX) {
    int n_cols,n_rows;
    ierr = MatGetSize(*B,&n_rows,&n_cols);CHKERRQ(ierr);
    if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Reused submatrix wrong size");
    newmat = *B;
  } else {
    /* Create and fill new matrix */
    ierr = MatCreateSeqDense(A->comm,nrows,ncols,PETSC_NULL,&newmat);CHKERRQ(ierr);
  }

  /* Now extract the data pointers and do the copy, column at a time */
  bv = ((Mat_SeqDense*)newmat->data)->v;
  
  for ( i=0; i<ncols; i++ ) {
    av = v + m*icol[i];
    for (j=0; j<nrows; j++ ) {
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

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqDense"
int MatGetSubMatrices_SeqDense(Mat A,int n, IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) );CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SeqDense(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCopy_SeqDense"
int MatCopy_SeqDense(Mat A, Mat B,MatStructure str)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data, *b = (Mat_SeqDense *)B->data;
  int          ierr;

  PetscFunctionBegin;
  if (B->type != MATSEQDENSE) {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (a->m != b->m || a->n != b->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"size(B) != size(A)");
  ierr = PetscMemcpy(b->v,a->v,a->m*a->n*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqDense,
       MatGetRow_SeqDense, 
       MatRestoreRow_SeqDense,
       MatMult_SeqDense, 
       MatMultAdd_SeqDense, 
       MatMultTrans_SeqDense, 
       MatMultTransAdd_SeqDense, 
       MatSolve_SeqDense,
       MatSolveAdd_SeqDense,
       MatSolveTrans_SeqDense,
       MatSolveTransAdd_SeqDense,
       MatLUFactor_SeqDense,
       MatCholeskyFactor_SeqDense,
       MatRelax_SeqDense,
       MatTranspose_SeqDense,
       MatGetInfo_SeqDense,
       MatEqual_SeqDense,
       MatGetDiagonal_SeqDense,
       MatDiagonalScale_SeqDense,
       MatNorm_SeqDense,
       0,
       0,
       0, 
       MatSetOption_SeqDense,
       MatZeroEntries_SeqDense,
       MatZeroRows_SeqDense,
       MatLUFactorSymbolic_SeqDense,
       MatLUFactorNumeric_SeqDense,
       MatCholeskyFactorSymbolic_SeqDense,
       MatCholeskyFactorNumeric_SeqDense,
       MatGetSize_SeqDense,
       MatGetSize_SeqDense,
       MatGetOwnershipRange_SeqDense,
       0,
       0, 
       MatGetArray_SeqDense, 
       MatRestoreArray_SeqDense,
       MatDuplicate_SeqDense,
       0,
       0,
       0,
       0,
       MatAXPY_SeqDense,
       MatGetSubMatrices_SeqDense,
       0,
       MatGetValues_SeqDense,
       MatCopy_SeqDense,
       0,
       MatScale_SeqDense,
       0,
       0,
       0,
       MatGetBlockSize_SeqDense,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetMaps_Petsc};

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqDense"
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
int MatCreateSeqDense(MPI_Comm comm,int m,int n,Scalar *data,Mat *A)
{
  Mat          B;
  Mat_SeqDense *b;
  int          ierr,size;
  PetscTruth   flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Comm must be of size 1");

  *A            = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQDENSE,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  b               = (Mat_SeqDense *) PetscMalloc(sizeof(Mat_SeqDense));CHKPTRQ(b);
  ierr            = PetscMemzero(b,sizeof(Mat_SeqDense));CHKERRQ(ierr);
  ierr            = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy = MatDestroy_SeqDense;
  B->ops->view    = MatView_SeqDense;
  B->factor       = 0;
  B->mapping      = 0;
  PLogObjectMemory(B,sizeof(struct _p_Mat));
  B->data         = (void *) b;

  b->m = m;  B->m = m; B->M = m;
  b->n = n;  B->n = n; B->N = n;

  ierr = MapCreateMPI(comm,m,m,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,n,n,&B->cmap);CHKERRQ(ierr);

  b->pivots       = 0;
  b->roworiented  = 1;
  if (data == PETSC_NULL) {
    b->v = (Scalar*) PetscMalloc((m*n+1)*sizeof(Scalar));CHKPTRQ(b->v);
    ierr = PetscMemzero(b->v,m*n*sizeof(Scalar));CHKERRQ(ierr);
    b->user_alloc = 0;
    PLogObjectMemory(B,n*m*sizeof(Scalar));
  } else { /* user-allocated storage */
    b->v = data;
    b->user_alloc = 1;
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B);CHKERRQ(ierr);}

  *A = B;
  PetscFunctionReturn(0);
}






