
/*
    Standard Fortran style matrices
*/
#include "ptscimpl.h"
#include "plapack.h"
#include "matimpl.h"
#include "math.h"
#include "vec/vecimpl.h"

typedef struct {
  Scalar *v;
  int    roworiented;
  int    m,n,pad;
  int    *pivots;   /* pivots in LU factorization */
} MatiSD;


static int MatiSDnz(Mat matin,int *nz)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int    i,N = mat->m*mat->n,count = 0;
  Scalar *v = mat->v;
  for ( i=0; i<N; i++ ) {if (*v != 0.0) count++; v++;}
  *nz = count; return 0;
}
static int MatiSDmemory(Mat matin,int *mem)
{
  MatiSD *mat = (MatiSD *) matin->data;
  *mem = mat->m*mat->n*sizeof(Scalar); return 0;
}
  
/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/
static int MatiSDlufactor(Mat matin,IS row,IS col)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int    ierr, one  = 1, info;
  if (!mat->pivots) {
    mat->pivots = (int *) MALLOC( mat->m*sizeof(int) );
    CHKPTR(mat->pivots);
  }
  LAgetrf_(&mat->m,&mat->n,mat->v,&mat->m,mat->pivots,&info);
  if (info) SETERR(1,"Bad LU factorization");
  matin->factor = FACTOR_LU;
  return 0;
}
static int MatiSDlufactorsymbolic(Mat matin,IS row,IS col,Mat *fact)
{
  int ierr;
  if (ierr = MatCopy(matin,fact)) SETERR(ierr,0);
  return 0;
}
static int MatiSDlufactornumeric(Mat matin,Mat *fact)
{
  return MatLUFactor(*fact,0,0);
}
static int MatiSDchfactorsymbolic(Mat matin,IS row,Mat *fact)
{
  int ierr;
  if (ierr = MatCopy(matin,fact)) SETERR(ierr,0);
  return 0;
}
static int MatiSDchfactornumeric(Mat matin,Mat *fact)
{
  return MatCholeskyFactor(*fact,0);
}
static int MatiSDchfactor(Mat matin,IS perm)
{
  MatiSD    *mat = (MatiSD *) matin->data;
  int       ierr, one  = 1, info;
  if (mat->pivots) {FREE(mat->pivots); mat->pivots = 0;}
  LApotrf_("L",&mat->n,mat->v,&mat->m,&info);
  if (info) SETERR(1,"Bad Cholesky factorization");
  matin->factor = FACTOR_CHOLESKY;
  return 0;
}

static int MatiSDsolve(Mat matin,Vec xx,Vec yy)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int i,j, one = 1, info;
  Scalar *v = mat->v, *x, *y;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  MEMCPY(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then LU else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,
              y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,
              y, &mat->m, &info );
  }
  if (info) SETERR(1,"Bad solve");
  return 0;
}
static int MatiSDsolvetrans(Mat matin,Vec xx,Vec yy)
{return 0;}

/* ------------------------------------------------------------------*/
static int MatiSDrelax(Mat matin,Vec bb,double omega,int flag,double shift,
                       int its,Vec xx)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *x, *b, *v = mat->v, zero = 0.0, xt;
  int    o = 1, tmp,n = mat->n,ierr, m = mat->m, i, j;

  if (flag & SOR_ZERO_INITIAL_GUESS) {
    /* this is a hack fix, should have another version without 
       the second BLdot */
    if (ierr = VecSet(&zero,xx)) SETERR(ierr,0);
  }
  VecGetArray(xx,&x); VecGetArray(bb,&b);
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
        x[i] = (1. - omega)*x[i] + omega*(xt/(v[i + i*m]+shift) + x[i]);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP) {
      for ( i=m-1; i>=0; i-- ) {
        xt = b[i]-BLdot_(&m,v+i,&m,x,&o);
        x[i] = (1. - omega)*x[i] + omega*(xt/(v[i + i*m]+shift) + x[i]);
      }
    }
  } 
  return 0;
} 

/* -----------------------------------------------------------------*/
static int MatiSDmulttrans(Mat matin,Vec xx,Vec yy)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v = mat->v, *x, *y;
  int _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x), VecGetArray(yy,&y);
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DZero, y, &_One );
  return 0;
}
static int MatiSDmult(Mat matin,Vec xx,Vec yy)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v = mat->v, *x, *y;
  int _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DZero, y, &_One );
  return 0;
}
static int MatiSDmultadd(Mat matin,Vec xx,Vec zz,Vec yy)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v = mat->v, *x, *y, *z;
  int    _One=1; Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z);
  if (zz != yy) MEMCPY(y,z,mat->m*sizeof(Scalar));
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DOne, y, &_One );
  return 0;
}
static int MatiSDmulttransadd(Mat matin,Vec xx,Vec zz,Vec yy)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v = mat->v, *x, *y, *z;
  int    _One=1;
  Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  VecGetArray(zz,&z);
  if (zz != yy) MEMCPY(y,z,mat->m*sizeof(Scalar));
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m), 
         x, &_One, &_DOne, y, &_One );
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatiSDgetrow(Mat matin,int row,int *ncols,int **cols,
                        Scalar **vals)
{
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v;
  int    i;
  *ncols = mat->n;
  if (cols) {
    *cols = (int *) MALLOC(mat->n*sizeof(int)); CHKPTR(*cols);
    for ( i=0; i<mat->n; i++ ) *cols[i] = i;
  }
  if (vals) {
    *vals = (Scalar *) MALLOC(mat->n*sizeof(Scalar)); CHKPTR(*vals);
    v = mat->v + row;
    for ( i=0; i<mat->n; i++ ) {*vals[i] = *v; v += mat->m;}
  }
  return 0;
}
static int MatiSDrestorerow(Mat matin,int row,int *ncols,int **cols,
                            Scalar **vals)
{
  MatiSD *mat = (MatiSD *) matin->data;
  if (cols) { FREE(*cols); }
  if (vals) { FREE(*vals); }
  return 0;
}
/* ----------------------------------------------------------------*/
static int MatiSDinsert(Mat matin,int m,int *indexm,int n,
                        int *indexn,Scalar *v,InsertMode addv)
{ 
  MatiSD *mat = (MatiSD *) matin->data;
  int    i,j;
 
  if (!mat->roworiented) {
    if (addv == InsertValues) {
      for ( j=0; j<n; j++ ) {
        if (indexn[j] < 0) {*v += m; continue;}
        for ( i=0; i<m; i++ ) {
          if (indexm[i] < 0) {*v++; continue;}
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    }
    else {
      for ( j=0; j<n; j++ ) {
        if (indexn[j] < 0) {*v += m; continue;}
        for ( i=0; i<m; i++ ) {
          if (indexm[i] < 0) {*v++; continue;}
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  }
  else {
    if (addv == InsertValues) {
      for ( i=0; i<m; i++ ) {
        if (indexm[i] < 0) {*v += n; continue;}
        for ( j=0; j<n; j++ ) {
          if (indexn[j] < 0) {*v++; continue;}
          mat->v[indexn[j]*mat->m + indexm[i]] = *v++;
        }
      }
    }
    else {
      for ( i=0; i<m; i++ ) {
        if (indexm[i] < 0) {*v += n; continue;}
        for ( j=0; j<n; j++ ) {
          if (indexn[j] < 0) {*v++; continue;}
          mat->v[indexn[j]*mat->m + indexm[i]] += *v++;
        }
      }
    }
  }
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatiSDcopy(Mat matin,Mat *newmat)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int ierr;
  Mat newi;
  MatiSD *l;
  if (ierr = MatCreateSequentialDense(mat->m,mat->n,&newi)) SETERR(ierr,0);
  l = (MatiSD *) newi->data;
  MEMCPY(l->v,mat->v,mat->m*mat->n*sizeof(Scalar));
  *newmat = newi;
  return 0;
}

int MatiSDview(PetscObject obj,Viewer ptr)
{
  Mat    matin = (Mat) obj;
  MatiSD *mat = (MatiSD *) matin->data;
  Scalar *v;
  int i,j;
  for ( i=0; i<mat->m; i++ ) {
    v = mat->v + i;
    for ( j=0; j<mat->n; j++ ) {
#if defined(PETSC_COMPLEX)
      printf("%6.4e + %6.4e i ",real(*v),imag(*v)); v += mat->m;
#else
      printf("%6.4e ",*v); v += mat->m;
#endif
    }
    printf("\n");
  }
  return 0;
}


static int MatiSDdestroy(PetscObject obj)
{
  Mat mat = (Mat) obj;
  MatiSD *l = (MatiSD *) mat->data;
  if (l->pivots) FREE(l->pivots);
  FREE(l);
  FREE(mat);
  return 0;
}

static int MatiSDtrans(Mat matin)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int    k,j;
  Scalar *v = mat->v, tmp;
  if (mat->m != mat->n) {
    SETERR(1,"Cannot transpose rectangular dense matrix");
  }
  for ( j=0; j<mat->m; j++ ) {
    for ( k=0; k<j; k++ ) {
      tmp = v[j + k*mat->n]; 
      v[j + k*mat->n] = v[k + j*mat->n];
      v[k + j*mat->n] = tmp;
    }
  }
  return 0;
}

static int MatiSDequal(Mat matin1,Mat matin2)
{
  MatiSD *mat1 = (MatiSD *) matin1->data;
  MatiSD *mat2 = (MatiSD *) matin2->data;
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

static int MatiSDgetdiag(Mat matin,Vec v)
{
  MatiSD *mat = (MatiSD *) matin->data;
  int    i,j, n;
  Scalar *x, zero = 0.0;
  CHKTYPE(v,SEQVECTOR);
  VecGetArray(v,&x); VecGetSize(v,&n);
  if (n != mat->m) SETERR(1,"Nonconforming matrix and vector");
  for ( i=0; i<mat->m; i++ ) {
    x[i] = mat->v[i*mat->m + i];
  }
  return 0;
}

static int MatiSDscale(Mat matin,Vec l,Vec r)
{
return 0;
}

static int MatiSDnorm(Mat matin,int type,double *norm)
{
  MatiSD *mat = (MatiSD *) matin->data;
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
    SETERR(1,"No support for the two norm yet");
  }
  return 0;
}

static int MatiDenseinsopt(Mat aijin,int op)
{
  MatiSD *aij = (MatiSD *) aijin->data;
  if (op == ROW_ORIENTED)            aij->roworiented = 1;
  else if (op == COLUMN_ORIENTED)    aij->roworiented = 0;
  /* doesn't care about sorted rows or columns */
  return 0;
}

static int MatiZero(Mat A)
{
  MatiSD *l = (MatiSD *) A->data;
  MEMSET(l->v,0,l->m*l->n*sizeof(Scalar));
  return 0;
}

static int MatiZerorows(Mat A,IS is,Scalar *diag)
{
  MatiSD *l = (MatiSD *) A->data;
  int     m = l->m, n = l->n, i, j,ierr,N, *rows;
  Scalar  *slot;
  ierr = ISGetLocalSize(is,&N); CHKERR(ierr);
  ierr = ISGetIndices(is,&rows); CHKERR(ierr);
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
/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatiSDinsert,
       MatiSDgetrow, MatiSDrestorerow,
       MatiSDmult, MatiSDmultadd, MatiSDmulttrans, MatiSDmulttransadd, 
       MatiSDsolve,0,MatiSDsolvetrans,0,
       MatiSDlufactor,MatiSDchfactor,
       MatiSDrelax,
       MatiSDtrans,
       MatiSDnz,MatiSDmemory,MatiSDequal,
       MatiSDcopy,
       MatiSDgetdiag,MatiSDscale,MatiSDnorm,
       0,0,
       0, MatiDenseinsopt,MatiZero,MatiZerorows,0,
       MatiSDlufactorsymbolic,MatiSDlufactornumeric,
       MatiSDchfactorsymbolic,MatiSDchfactornumeric
};
/*@
    MatCreateSequentialDense - Creates a sequential dense matrix that 
        is stored in the usual Fortran 77 manner. Many of the matrix
        operations use the BLAS and LAPACK routines.

  Input Parameters:
.   m, n - the number of rows and columns in the matrix.

  Output Parameter:
.  newmat - the matrix created.

  Keywords: dense matrix, lapack, blas
@*/
int MatCreateSequentialDense(int m,int n,Mat *newmat)
{
  int       size = sizeof(MatiSD) + m*n*sizeof(Scalar);
  Mat mat;
  MatiSD    *l;
  *newmat        = 0;
  CREATEHEADER(mat,_Mat);
  l              = (MatiSD *) MALLOC(size); CHKPTR(l);
  mat->cookie    = MAT_COOKIE;
  mat->ops       = &MatOps;
  mat->destroy   = MatiSDdestroy;
  mat->view      = MatiSDview;
  mat->data      = (void *) l;
  mat->type      = MATDENSESEQ;
  mat->factor    = 0;
  mat->col       = 0;
  mat->row       = 0;
  mat->outofrange= 0;
  mat->Mlow      = 0;
  mat->Mhigh     = m;
  mat->Nlow      = 0;
  mat->Nhigh     = n;

  l->m           = m;
  l->n           = n;
  l->v           = (Scalar *) (l + 1);
  l->pivots      = 0;
  l->roworiented = 1;

  MEMSET(l->v,0,m*n*sizeof(Scalar));
  *newmat = mat;
  return 0;
}

int MatiSDCreate(Mat matin,Mat *newmat)
{
  MatiSD *m = (MatiSD *) matin->data;
  return MatCreateSequentialDense(m->m,m->n,newmat);
}
