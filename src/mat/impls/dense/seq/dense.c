#ifndef lint
static char vcid[] = "$Id: dense.c,v 1.56 1995/09/10 20:51:25 curfman Exp bsmith $";
#endif

/*
    Standard Fortran style matrices
*/
#include "petsc.h"
#include "pinclude/plapack.h"
#include "matimpl.h"
#include "math.h"
#include "vec/vecimpl.h"
#include "pinclude/pviewer.h"

typedef struct {
  Scalar *v;
  int    roworiented;
  int    m,n,pad;
  int    *pivots;   /* pivots in LU factorization */
} Mat_Dense;

static int MatGetInfo_Dense(Mat matin,MatInfoType flag,int *nz,
                            int *nzalloc,int *mem)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    i,N = mat->m*mat->n,count = 0;
  Scalar *v = mat->v;
  for ( i=0; i<N; i++ ) {if (*v != 0.0) count++; v++;}
  *nz = count; *nzalloc = N; *mem = (int)matin->mem;
  return 0;
}
  
/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/
static int MatLUFactor_Dense(Mat matin,IS row,IS col,double f)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    info;
  if (!mat->pivots) {
    mat->pivots = (int *) PETSCMALLOC(mat->m*sizeof(int));
    CHKPTRQ(mat->pivots);
    PLogObjectMemory(matin,mat->m*sizeof(int));
  }
  LAgetrf_(&mat->m,&mat->n,mat->v,&mat->m,mat->pivots,&info);
  if (info) SETERRQ(1,"MatLUFactor_Dense: Bad LU factorization");
  matin->factor = FACTOR_LU;
  return 0;
}
static int MatLUFactorSymbolic_Dense(Mat matin,IS row,IS col,double f,
                                     Mat *fact)
{
  int ierr;
  ierr = MatConvert(matin,MATSAME,fact); CHKERRQ(ierr);
  return 0;
}
static int MatLUFactorNumeric_Dense(Mat matin,Mat *fact)
{
  return MatLUFactor(*fact,0,0,1.0);
}
static int MatCholeskyFactorSymbolic_Dense(Mat matin,IS row,double f,Mat *fact)
{
  int ierr;
  ierr = MatConvert(matin,MATSAME,fact); CHKERRQ(ierr);
  return 0;
}
static int MatCholeskyFactorNumeric_Dense(Mat matin,Mat *fact)
{
  return MatCholeskyFactor(*fact,0,1.0);
}
static int MatCholeskyFactor_Dense(Mat matin,IS perm,double f)
{
  Mat_Dense    *mat = (Mat_Dense *) matin->data;
  int       info;
  if (mat->pivots) {
    PETSCFREE(mat->pivots);
    PLogObjectMemory(matin,-mat->m*sizeof(int));
    mat->pivots = 0;
  }
  LApotrf_("L",&mat->n,mat->v,&mat->m,&info);
  if (info) SETERRQ(1,"MatCholeskyFactor_Dense: Bad factorization");
  matin->factor = FACTOR_CHOLESKY;
  return 0;
}

static int MatSolve_Dense(Mat matin,Vec xx,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    one = 1, info;
  Scalar *x, *y;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  if (matin->factor == FACTOR_LU) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,
              y, &mat->m, &info );
  }
  else if (matin->factor == FACTOR_CHOLESKY){
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,
              y, &mat->m, &info );
  }
  else SETERRQ(1,"MatSolve_Dense: Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolve_Dense: Bad solve");
  return 0;
}
static int MatSolveTrans_Dense(Mat matin,Vec xx,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    one = 1, info;
  Scalar *x, *y;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,
              y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,
              y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveTrans_Dense: Bad solve");
  return 0;
}
static int MatSolveAdd_Dense(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    one = 1, info,ierr;
  Scalar *x, *y, sone = 1.0;
  Vec    tmp = 0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(matin,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,
              y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,
              y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveAdd_Dense: Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
static int MatSolveTransAdd_Dense(Mat matin,Vec xx,Vec zz, Vec yy)
{
  Mat_Dense  *mat = (Mat_Dense *) matin->data;
  int     one = 1, info,ierr;
  Scalar  *x, *y, sone = 1.0;
  Vec     tmp;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(matin,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PETSCMEMCPY(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,
              y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,
              y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_Dense: Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  return 0;
}
/* ------------------------------------------------------------------*/
static int MatRelax_Dense(Mat matin,Vec bb,double omega,MatSORType flag,
                          double shift,int its,Vec xx)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *x, *b, *v = mat->v, zero = 0.0, xt;
  int    o = 1,ierr, m = mat->m, i;

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without 
       the second BLdot */
    ierr = VecSet(&zero,xx); CHKERRQ(ierr);
  }
  VecGetArray(xx,&x); VecGetArray(bb,&b);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
#if defined(PETSC_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        int    _i;
        Scalar sum = b[i];
        for ( _i=0; _i<m; _i++ ) {
          sum -= conj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
#endif
        x[i] = (1. - omega)*x[i] + omega*(xt/(v[i + i*m]+shift) + x[i]);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP) {
      for ( i=m-1; i>=0; i-- ) {
#if defined(PETSC_COMPLEX)
        /* cannot use BLAS dot for complex because compiler/linker is 
           not happy about returning a double complex */
        int    _i;
        Scalar sum = b[i];
        for ( _i=0; _i<m; _i++ ) {
          sum -= conj(v[i+_i*m])*x[_i];
        }
        xt = sum;
#else
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
#endif
        x[i] = (1. - omega)*x[i] + omega*(xt/(v[i + i*m]+shift) + x[i]);
      }
    }
  } 
  return 0;
} 

/* -----------------------------------------------------------------*/
static int MatMultTrans_Dense(Mat matin,Vec xx,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v = mat->v, *x, *y;
  int _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x), VecGetArray(yy,&y);
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DZero, y, &_One );
  return 0;
}
static int MatMult_Dense(Mat matin,Vec xx,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v = mat->v, *x, *y;
  int _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DZero, y, &_One );
  return 0;
}
static int MatMultAdd_Dense(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v = mat->v, *x, *y, *z;
  int    _One=1; Scalar _DOne=1.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z);
  if (zz != yy) PETSCMEMCPY(y,z,mat->m*sizeof(Scalar));
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DOne, y, &_One );
  return 0;
}
static int MatMultTransAdd_Dense(Mat matin,Vec xx,Vec zz,Vec yy)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v = mat->v, *x, *y, *z;
  int    _One=1;
  Scalar _DOne=1.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  VecGetArray(zz,&z);
  if (zz != yy) PETSCMEMCPY(y,z,mat->m*sizeof(Scalar));
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DOne, y, &_One );
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatGetRow_Dense(Mat matin,int row,int *ncols,int **cols,
                        Scalar **vals)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v;
  int    i;
  *ncols = mat->n;
  if (cols) {
    *cols = (int *) PETSCMALLOC(mat->n*sizeof(int)); CHKPTRQ(*cols);
    for ( i=0; i<mat->n; i++ ) *cols[i] = i;
  }
  if (vals) {
    *vals = (Scalar *) PETSCMALLOC(mat->n*sizeof(Scalar)); CHKPTRQ(*vals);
    v = mat->v + row;
    for ( i=0; i<mat->n; i++ ) {*vals[i] = *v; v += mat->m;}
  }
  return 0;
}
static int MatRestoreRow_Dense(Mat matin,int row,int *ncols,int **cols,
                            Scalar **vals)
{
  if (cols) { PETSCFREE(*cols); }
  if (vals) { PETSCFREE(*vals); }
  return 0;
}
/* ----------------------------------------------------------------*/
static int MatInsert_Dense(Mat matin,int m,int *indexm,int n,
                        int *indexn,Scalar *v,InsertMode addv)
{ 
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int    i,j;
 
  if (!mat->roworiented) {
    if (addv == INSERTVALUES) {
      for ( j=0; j<n; j++ ) {
        for ( i=0; i<m; i++ ) {
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    }
    else {
      for ( j=0; j<n; j++ ) {
        for ( i=0; i<m; i++ ) {
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  }
  else {
    if (addv == INSERTVALUES) {
      for ( i=0; i<m; i++ ) {
        for ( j=0; j<n; j++ ) {
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    }
    else {
      for ( i=0; i<m; i++ ) {
        for ( j=0; j<n; j++ ) {
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  }
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatCopyPrivate_Dense(Mat matin,Mat *newmat)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int ierr;
  Mat newi;
  Mat_Dense *l;
  ierr = MatCreateSeqDense(matin->comm,mat->m,mat->n,&newi);
  CHKERRQ(ierr);
  l = (Mat_Dense *) newi->data;
  PETSCMEMCPY(l->v,mat->v,mat->m*mat->n*sizeof(Scalar));
  *newmat = newi;
  return 0;
}
#include "viewer.h"

int MatView_Dense(PetscObject obj,Viewer ptr)
{
  Mat         matin = (Mat) obj;
  Mat_Dense   *mat = (Mat_Dense *) matin->data;
  Scalar      *v;
  int         i,j,ierr;
  PetscObject vobj = (PetscObject) ptr;

  if (ptr == 0) {
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }
  if (vobj->cookie == VIEWER_COOKIE && vobj->type == MATLAB_VIEWER) {
    return ViewerMatlabPutArray_Private(ptr,mat->m,mat->n,mat->v); 
  }
  else {
    FILE *fd;
    int format;
    ierr = ViewerFileGetPointer_Private(ptr,&fd); CHKERRQ(ierr);
    ierr = ViewerFileGetFormat_Private(ptr,&format); CHKERRQ(ierr);
    if (format == FILE_FORMAT_INFO) {
      /* do nothing for now */
    }
    else {
      for ( i=0; i<mat->m; i++ ) {
        v = mat->v + i;
        for ( j=0; j<mat->n; j++ ) {
#if defined(PETSC_COMPLEX)
          fprintf(fd,"%6.4e + %6.4e i ",real(*v),imag(*v)); v += mat->m;
#else
          fprintf(fd,"%6.4e ",*v); v += mat->m;
#endif
        }
        fprintf(fd,"\n");
      }
    }
  }
  return 0;
}


static int MatDestroy_Dense(PetscObject obj)
{
  Mat    mat = (Mat) obj;
  Mat_Dense *l = (Mat_Dense *) mat->data;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d Cols %d",l->m,l->n);
#endif
  if (l->pivots) PETSCFREE(l->pivots);
  PETSCFREE(l);
  PLogObjectDestroy(mat);
  PETSCHEADERDESTROY(mat);
  return 0;
}

static int MatTranspose_Dense(Mat matin,Mat *matout)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int       k, j, m, n;
  Scalar    *v, tmp;

  v = mat->v; m = mat->m; n = mat->n;
  if (!matout) { /* in place transpose */
    if (m != n) SETERRQ(1,
      "MatTranspose_Dense: Cannot transpose rectangular matrix in place");
    for ( j=0; j<m; j++ ) {
      for ( k=0; k<j; k++ ) {
        tmp = v[j + k*n]; 
        v[j + k*n] = v[k + j*n];
        v[k + j*n] = tmp;
      }
    }
  }
  else { /* out-of-place transpose */
    int ierr;
    Mat tmat;
    Mat_Dense *tmatd;
    Scalar *v2;
    ierr = MatCreateSeqDense(matin->comm,mat->n,mat->m,&tmat); CHKERRQ(ierr);
    tmatd = (Mat_Dense *) tmat->data;
    v = mat->v; v2 = tmatd->v;
    for ( j=0; j<n; j++ ) {
      for ( k=0; k<m; k++ ) v2[j + k*n] = v[k + j*m];
    }
    ierr = MatAssemblyBegin(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
    *matout = tmat;
  } 
  return 0;
}

static int MatEqual_Dense(Mat matin1,Mat matin2)
{
  Mat_Dense *mat1 = (Mat_Dense *) matin1->data;
  Mat_Dense *mat2 = (Mat_Dense *) matin2->data;
  int    i;
  Scalar *v1 = mat1->v, *v2 = mat2->v;
  if (mat1->m != mat2->m) return 0;
  if (mat1->n != mat2->n) return 0;
  for ( i=0; i<mat1->m*mat1->n; i++ ) {
    if (*v1 != *v2) return 0;
    v1++; v2++;
  }
  return 1;
}

static int MatGetDiagonal_Dense(Mat matin,Vec v)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int       i, n;
  Scalar    *x;
  VecGetArray(v,&x); VecGetSize(v,&n);
  if (n != mat->m) SETERRQ(1,"MatGetDiagonal_Dense:Nonconforming mat and vec");
  for ( i=0; i<mat->m; i++ ) {
    x[i] = mat->v[i*mat->m + i];
  }
  return 0;
}

static int MatScale_Dense(Mat matin,Vec ll,Vec rr)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *l,*r,x,*v;
  int    i,j,m = mat->m, n = mat->n;
  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != mat->m) SETERRQ(1,"MatScale_Dense:Left scaling vec wrong size");
    for ( i=0; i<m; i++ ) {
      x = l[i];
      v = mat->v + i;
      for ( j=0; j<n; j++ ) { (*v) *= x; v+= m;} 
    }
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != mat->n) SETERRQ(1,"MatScale_Dense:Right scaling vec wrong size");
    for ( i=0; i<n; i++ ) {
      x = r[i];
      v = mat->v + i*m;
      for ( j=0; j<m; j++ ) { (*v++) *= x;} 
    }
  }
  return 0;
}

static int MatNorm_Dense(Mat matin,MatNormType type,double *norm)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  Scalar *v = mat->v;
  double sum = 0.0;
  int    i, j;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  }
  else if (type == NORM_1) {
    *norm = 0.0;
    for ( j=0; j<mat->n; j++ ) {
      sum = 0.0;
      for ( i=0; i<mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += abs(*v++); 
#else
        sum += fabs(*v++); 
#endif
      }
      if (sum > *norm) *norm = sum;
    }
  }
  else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for ( j=0; j<mat->m; j++ ) {
      v = mat->v + j;
      sum = 0.0;
      for ( i=0; i<mat->n; i++ ) {
#if defined(PETSC_COMPLEX)
        sum += abs(*v); v += mat->m;
#else
        sum += fabs(*v); v += mat->m;
#endif
      }
      if (sum > *norm) *norm = sum;
    }
  }
  else {
    SETERRQ(1,"MatNorm_Dense:No two norm yet");
  }
  return 0;
}

static int MatSetOption_Dense(Mat aijin,MatOption op)
{
  Mat_Dense *aij = (Mat_Dense *) aijin->data;
  if (op == ROW_ORIENTED)            aij->roworiented = 1;
  else if (op == COLUMN_ORIENTED)    aij->roworiented = 0;
  /* doesn't care about sorted rows or columns */
  return 0;
}

static int MatZeroEntries_Dense(Mat A)
{
  Mat_Dense *l = (Mat_Dense *) A->data;
  PETSCMEMSET(l->v,0,l->m*l->n*sizeof(Scalar));
  return 0;
}

static int MatZeroRows_Dense(Mat A,IS is,Scalar *diag)
{
  Mat_Dense *l = (Mat_Dense *) A->data;
  int    n = l->n, i, j,ierr,N, *rows;
  Scalar *slot;
  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
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
  return 0;
}

static int MatGetSize_Dense(Mat matin,int *m,int *n)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  *m = mat->m; *n = mat->n;
  return 0;
}

static int MatGetOwnershipRange_Dense(Mat matin,int *m,int *n)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  *m = 0; *n = mat->m;
  return 0;
}

static int MatGetArray_Dense(Mat matin,Scalar **array)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  *array = mat->v;
  return 0;
}


static int MatGetSubMatrixInPlace_Dense(Mat matin,IS isrow,IS iscol)
{
  SETERRQ(1,"MatGetSubMatrixInPlace_Dense: not done");
}

static int MatGetSubMatrix_Dense(Mat matin,IS isrow,IS iscol,Mat *submat)
{
  Mat_Dense *mat = (Mat_Dense *) matin->data;
  int     nznew, *smap, i, j, ierr, oldcols = mat->n;
  int     *irow, *icol, nrows, ncols, *cwork;
  Scalar  *vwork, *val;
  Mat     newmat;

  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  smap = (int *) PETSCMALLOC(oldcols*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PETSCMALLOC(ncols*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PETSCMALLOC(ncols*sizeof(Scalar)); CHKPTRQ(vwork);
  PETSCMEMSET((char*)smap,0,oldcols*sizeof(int));
  for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;

  /* Create and fill new matrix */
  ierr = MatCreateSeqDense(matin->comm,nrows,ncols,&newmat);
         CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    nznew = 0;
    val   = mat->v + irow[i];
    for (j=0; j<oldcols; j++) {
      if (smap[j]) {
        cwork[nznew]   = smap[j] - 1;
        vwork[nznew++] = val[j * mat->m];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERTVALUES); 
           CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PETSCFREE(smap); PETSCFREE(cwork); PETSCFREE(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatInsert_Dense,
       MatGetRow_Dense, MatRestoreRow_Dense,
       MatMult_Dense, MatMultAdd_Dense, 
       MatMultTrans_Dense, MatMultTransAdd_Dense, 
       MatSolve_Dense,MatSolveAdd_Dense,
       MatSolveTrans_Dense,MatSolveTransAdd_Dense,
       MatLUFactor_Dense,MatCholeskyFactor_Dense,
       MatRelax_Dense,
       MatTranspose_Dense,
       MatGetInfo_Dense,MatEqual_Dense,
       MatGetDiagonal_Dense,MatScale_Dense,MatNorm_Dense,
       0,0,
       0, MatSetOption_Dense,MatZeroEntries_Dense,MatZeroRows_Dense,0,
       MatLUFactorSymbolic_Dense,MatLUFactorNumeric_Dense,
       MatCholeskyFactorSymbolic_Dense,MatCholeskyFactorNumeric_Dense,
       MatGetSize_Dense,MatGetSize_Dense,MatGetOwnershipRange_Dense,
       0,0,MatGetArray_Dense,0,0,
       MatGetSubMatrix_Dense,MatGetSubMatrixInPlace_Dense,
       MatCopyPrivate_Dense};

/*@C
   MatCreateSeqDense - Creates a sequential dense matrix that 
   is stored in column major order (the usual Fortran 77 manner). Many 
   of the matrix operations use the BLAS and LAPACK routines.

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  m - number of rows
.  n - number of column

   Output Parameter:
.  newmat - the matrix

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatSetValues()
@*/
int MatCreateSeqDense(MPI_Comm comm,int m,int n,Mat *newmat)
{
  int       size = sizeof(Mat_Dense) + m*n*sizeof(Scalar);
  Mat mat;
  Mat_Dense    *l;
  *newmat        = 0;
  PETSCHEADERCREATE(mat,_Mat,MAT_COOKIE,MATDENSE,comm);
  PLogObjectCreate(mat);
  l              = (Mat_Dense *) PETSCMALLOC(size); CHKPTRQ(l);
  mat->ops       = &MatOps;
  mat->destroy   = MatDestroy_Dense;
  mat->view      = MatView_Dense;
  mat->data      = (void *) l;
  mat->factor    = 0;
  PLogObjectMemory(mat,sizeof(struct _Mat) + size);

  l->m           = m;
  l->n           = n;
  l->v           = (Scalar *) (l + 1);
  l->pivots      = 0;
  l->roworiented = 1;

  PETSCMEMSET(l->v,0,m*n*sizeof(Scalar));
  *newmat = mat;
  return 0;
}

int MatCreate_Dense(Mat matin,Mat *newmat)
{
  Mat_Dense *m = (Mat_Dense *) matin->data;
  return MatCreateSeqDense(matin->comm,m->m,m->n,newmat);
}
