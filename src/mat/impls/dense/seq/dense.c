#ifndef lint
static char vcid[] = "$Id: dense.c,v 1.104 1996/04/12 23:30:30 balay Exp bsmith $";
#endif
/*
     Defines the basic matrix operations for sequential dense.
*/

#include "dense.h"
#include "pinclude/plapack.h"
#include "pinclude/pviewer.h"

int MatAXPY_SeqDense(Scalar *alpha,Mat X,Mat Y)
{
  Mat_SeqDense *x = (Mat_SeqDense*) X->data,*y = (Mat_SeqDense*) Y->data;
  int          N = x->m*x->n, one = 1;
  BLaxpy_( &N, alpha, x->v, &one, y->v, &one );
  PLogFlops(2*N-1);
  return 0;
}

static int MatGetInfo_SeqDense(Mat A,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i,N = mat->m*mat->n,count = 0;
  Scalar       *v = mat->v;
  for ( i=0; i<N; i++ ) {if (*v != 0.0) count++; v++;}
  if (nz)      *nz      = count; 
  if (nzalloc) *nzalloc = N;
  if (mem)     *mem     = (int)A->mem;
  return 0;
}

static int MatScale_SeqDense(Scalar *alpha,Mat inA)
{
  Mat_SeqDense *a = (Mat_SeqDense *) inA->data;
  int          one = 1, nz;

  nz = a->m*a->n;
  BLscal_( &nz, alpha, a->v, &one );
  PLogFlops(nz);
  return 0;
}
  
/* ---------------------------------------------------------------*/
/* COMMENT: I have chosen to hide column permutation in the pivots,
   rather than put it in the Mat->col slot.*/
static int MatLUFactor_SeqDense(Mat A,IS row,IS col,double f)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          info;

  if (!mat->pivots) {
    mat->pivots = (int *) PetscMalloc(mat->m*sizeof(int));CHKPTRQ(mat->pivots);
    PLogObjectMemory(A,mat->m*sizeof(int));
  }
  LAgetrf_(&mat->m,&mat->n,mat->v,&mat->m,mat->pivots,&info);
  if (info<0) SETERRQ(1,"MatLUFactor_SeqDense:Bad argument to LU factorization");
  if (info>0) SETERRQ(1,"MatLUFactor_SeqDense:Bad LU factorization");
  A->factor = FACTOR_LU;
  PLogFlops((2*mat->n*mat->n*mat->n)/3);
  return 0;
}
static int MatLUFactorSymbolic_SeqDense(Mat A,IS row,IS col,double f,Mat *fact)
{
  return MatConvert(A,MATSAME,fact);
}
static int MatLUFactorNumeric_SeqDense(Mat A,Mat *fact)
{
  return MatLUFactor(*fact,0,0,1.0);
}
static int MatCholeskyFactorSymbolic_SeqDense(Mat A,IS row,double f,Mat *fact)
{
  return MatConvert(A,MATSAME,fact);
}
static int MatCholeskyFactorNumeric_SeqDense(Mat A,Mat *fact)
{
  return MatCholeskyFactor(*fact,0,1.0);
}
static int MatCholeskyFactor_SeqDense(Mat A,IS perm,double f)
{
  Mat_SeqDense  *mat = (Mat_SeqDense *) A->data;
  int           info;

  if (mat->pivots) {
    PetscFree(mat->pivots);
    PLogObjectMemory(A,-mat->m*sizeof(int));
    mat->pivots = 0;
  }
  LApotrf_("L",&mat->n,mat->v,&mat->m,&info);
  if (info) SETERRQ(1,"MatCholeskyFactor_SeqDense:Bad factorization");
  A->factor = FACTOR_CHOLESKY;
  PLogFlops((mat->n*mat->n*mat->n)/3);
  return 0;
}

static int MatSolve_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          one = 1, info, ierr;
  Scalar       *x, *y;
  
  ierr = VecGetArray(xx,&x); CHKERRQ(ierr); ierr = VecGetArray(yy,&y); CHKERRQ(ierr);
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  if (A->factor == FACTOR_LU) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  }
  else if (A->factor == FACTOR_CHOLESKY){
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  else SETERRQ(1,"MatSolve_SeqDense:Matrix must be factored to solve");
  if (info) SETERRQ(1,"MatSolve_SeqDense:Bad solve");
  PLogFlops(mat->n*mat->n - mat->n);
  return 0;
}
static int MatSolveTrans_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          one = 1, info;
  Scalar       *x, *y;
  
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveTrans_SeqDense:Bad solve");
  PLogFlops(mat->n*mat->n - mat->n);
  return 0;
}
static int MatSolveAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          one = 1, info,ierr;
  Scalar       *x, *y, sone = 1.0;
  Vec          tmp = 0;
  
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "N", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveAdd_SeqDense:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  PLogFlops(mat->n*mat->n - mat->n);
  return 0;
}

static int MatSolveTransAdd_SeqDense(Mat A,Vec xx,Vec zz, Vec yy)
{
  Mat_SeqDense  *mat = (Mat_SeqDense *) A->data;
  int           one = 1, info,ierr;
  Scalar        *x, *y, sone = 1.0;
  Vec           tmp;
  
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (yy == zz) {
    ierr = VecDuplicate(yy,&tmp); CHKERRQ(ierr);
    PLogObjectParent(A,tmp);
    ierr = VecCopy(yy,tmp); CHKERRQ(ierr);
  } 
  PetscMemcpy(y,x,mat->m*sizeof(Scalar));
  /* assume if pivots exist then use LU; else Cholesky */
  if (mat->pivots) {
    LAgetrs_( "T", &mat->m, &one, mat->v, &mat->m, mat->pivots,y, &mat->m, &info );
  }
  else {
    LApotrs_( "L", &mat->m, &one, mat->v, &mat->m,y, &mat->m, &info );
  }
  if (info) SETERRQ(1,"MatSolveTransAdd_SeqDense:Bad solve");
  if (tmp) {VecAXPY(&sone,tmp,yy); VecDestroy(tmp);}
  else VecAXPY(&sone,zz,yy);
  PLogFlops(mat->n*mat->n - mat->n);
  return 0;
}
/* ------------------------------------------------------------------*/
static int MatRelax_SeqDense(Mat A,Vec bb,double omega,MatSORType flag,
                          double shift,int its,Vec xx)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *x, *b, *v = mat->v, zero = 0.0, xt;
  int          o = 1,ierr, m = mat->m, i;

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
int MatMultTrans_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x), VecGetArray(yy,&y);
  LAgemv_("T",&(mat->m),&(mat->n),&_DOne,v,&(mat->m),x,&_One,&_DZero,y,&_One);
  PLogFlops(2*mat->m*mat->n - mat->n);
  return 0;
}
int MatMult_SeqDense(Mat A,Vec xx,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y;
  int          _One=1;Scalar _DOne=1.0, _DZero=0.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  LAgemv_( "N", &(mat->m), &(mat->n), &_DOne, v, &(mat->m),x,&_One,&_DZero,y,&_One);
  PLogFlops(2*mat->m*mat->n - mat->m);
  return 0;
}
int MatMultAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y, *z;
  int          _One=1; Scalar _DOne=1.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z);
  if (zz != yy) PetscMemcpy(y,z,mat->m*sizeof(Scalar));
  LAgemv_( "N", &(mat->m), &(mat->n),&_DOne,v,&(mat->m),x,&_One,&_DOne,y,&_One);
  PLogFlops(2*mat->m*mat->n);
  return 0;
}
int MatMultTransAdd_SeqDense(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v, *x, *y, *z;
  int          _One=1;
  Scalar       _DOne=1.0;
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z);
  if (zz != yy) PetscMemcpy(y,z,mat->n*sizeof(Scalar));
  LAgemv_( "T", &(mat->m), &(mat->n), &_DOne, v, &(mat->m),x,&_One,&_DOne,y,&_One);
  PLogFlops(2*mat->m*mat->n);
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatGetRow_SeqDense(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v;
  int          i;
  
  *ncols = mat->n;
  if (cols) {
    *cols = (int *) PetscMalloc(mat->n*sizeof(int)); CHKPTRQ(*cols);
    for ( i=0; i<mat->n; i++ ) (*cols)[i] = i;
  }
  if (vals) {
    *vals = (Scalar *) PetscMalloc(mat->n*sizeof(Scalar)); CHKPTRQ(*vals);
    v = mat->v + row;
    for ( i=0; i<mat->n; i++ ) {(*vals)[i] = *v; v += mat->m;}
  }
  return 0;
}
static int MatRestoreRow_SeqDense(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  if (cols) { PetscFree(*cols); }
  if (vals) { PetscFree(*vals); }
  return 0;
}
/* ----------------------------------------------------------------*/
static int MatSetValues_SeqDense(Mat A,int m,int *indexm,int n,
                                    int *indexn,Scalar *v,InsertMode addv)
{ 
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i,j;
 
  if (!mat->roworiented) {
    if (addv == INSERT_VALUES) {
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
    if (addv == INSERT_VALUES) {
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

static int MatGetValues_SeqDense(Mat A,int m,int *indexm,int n,int *indexn,Scalar *v)
{ 
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i, j;
  Scalar       *vpt = v;

  /* row-oriented output */ 
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      *vpt++ = mat->v[indexn[j]*mat->m + indexm[i]];
    }
  }
  return 0;
}

/* -----------------------------------------------------------------*/
static int MatConvertSameType_SeqDense(Mat A,Mat *newmat,int cpvalues)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data, *l;
  int          ierr;
  Mat          newi;

  ierr = MatCreateSeqDense(A->comm,mat->m,mat->n,PETSC_NULL,&newi); CHKERRQ(ierr);
  l = (Mat_SeqDense *) newi->data;
  if (cpvalues == COPY_VALUES) {
    PetscMemcpy(l->v,mat->v,mat->m*mat->n*sizeof(Scalar));
  }
  newi->assembled = PETSC_TRUE;
  *newmat = newi;
  return 0;
}

#include "sys.h"

int MatLoad_SeqDense(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqDense *a;
  Mat          B;
  int          *scols, i, j, nz, ierr, fd, header[4], size;
  int          *rowlengths = 0, M, N, *cols;
  Scalar       *vals, *svals, *v;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"MatLoad_SeqDense: view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,BINARY_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_SeqDense:Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz == MATRIX_BINARY_FORMAT_DENSE) { /* matrix in file is dense */
    ierr = MatCreateSeqDense(comm,M,N,PETSC_NULL,A); CHKERRQ(ierr);
    B = *A;
    a = (Mat_SeqDense *) B->data;

    /* read in nonzero values */
    ierr = PetscBinaryRead(fd,a->v,M*N,BINARY_SCALAR); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(B,PETSC_NULL); CHKERRQ(ierr);
  } else {
    /* read row lengths */
    rowlengths = (int*) PetscMalloc( M*sizeof(int) ); CHKPTRQ(rowlengths);
    ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);

    /* create our matrix */
    ierr = MatCreateSeqDense(comm,M,N,PETSC_NULL,A); CHKERRQ(ierr);
    B = *A;
    a = (Mat_SeqDense *) B->data;
    v = a->v;

    /* read column indices and nonzeros */
    cols = scols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(cols);
    ierr = PetscBinaryRead(fd,cols,nz,BINARY_INT); CHKERRQ(ierr);
    vals = svals = (Scalar *) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);
    ierr = PetscBinaryRead(fd,vals,nz,BINARY_SCALAR); CHKERRQ(ierr);

    /* insert into matrix */  
    for ( i=0; i<M; i++ ) {
      for ( j=0; j<rowlengths[i]; j++ ) v[i+M*scols[j]] = svals[j];
      svals += rowlengths[i]; scols += rowlengths[i];
    }
    PetscFree(vals); PetscFree(cols); PetscFree(rowlengths);   

    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  }
  return 0;
}

#include "pinclude/pviewer.h"
#include "sys.h"

static int MatView_SeqDense_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  int          ierr, i, j, format;
  FILE         *fd;
  char         *outputname;
  Scalar       *v;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == ASCII_FORMAT_INFO || format == ASCII_FORMAT_INFO_DETAILED) {
    return 0;  /* do nothing for now */
  } 
  else if (format == ASCII_FORMAT_COMMON) {
    for ( i=0; i<a->m; i++ ) {
      v = a->v + i;
      fprintf(fd,"row %d:",i);
      for ( j=0; j<a->n; j++ ) {
#if defined(PETSC_COMPLEX)
        if (real(*v) != 0.0 && imag(*v) != 0.0)
          fprintf(fd," %d %g + %g i",j,real(*v),imag(*v));
        else if (real(*v)) fprintf(fd," %d %g ",j,real(*v));
        v += a->m;
#else
        if (*v) fprintf(fd," %d %g ",j,*v); 
        v += a->m;
#endif
      }
      fprintf(fd,"\n");
    }
  }
  else {
#if defined(PETSC_COMPLEX)
    int allreal = 1;
    /* determine if matrix has all real values */
    v = a->v;
    for ( i=0; i<a->m*a->n; i++ ) {
      if (imag(v[i])) { allreal = 0; break ;}
    }
#endif
    for ( i=0; i<a->m; i++ ) {
      v = a->v + i;
      for ( j=0; j<a->n; j++ ) {
#if defined(PETSC_COMPLEX)
        if (allreal) {
          fprintf(fd,"%6.4e ",real(*v)); v += a->m;
        } else {
          fprintf(fd,"%6.4e + %6.4e i ",real(*v),imag(*v)); v += a->m;
        }
#else
        fprintf(fd,"%6.4e ",*v); v += a->m;
#endif
      }
      fprintf(fd,"\n");
    }
  }
  fflush(fd);
  return 0;
}

static int MatView_SeqDense_Binary(Mat A,Viewer viewer)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data;
  int          ict, j, n = a->n, m = a->m, i, fd, *col_lens, ierr, nz = m*n;
  int          format;
  Scalar       *v, *anonz,*vals;
  
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
  if (format == BINARY_FORMAT_NATIVE) {
    /* store the matrix as a dense matrix */
    col_lens = (int *) PetscMalloc( 4*sizeof(int) ); CHKPTRQ(col_lens);
    col_lens[0] = MAT_COOKIE;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = MATRIX_BINARY_FORMAT_DENSE;
    ierr = PetscBinaryWrite(fd,col_lens,4,BINARY_INT,1); CHKERRQ(ierr);
    PetscFree(col_lens);

    /* write out matrix, by rows */
    vals = (Scalar *) PetscMalloc(m*n*sizeof(Scalar)); CHKPTRQ(vals);
    v    = a->v;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<n; j++ ) {
        vals[i + j*m] = *v++;
      }
    }
    ierr = PetscBinaryWrite(fd,vals,n*m,BINARY_SCALAR,0); CHKERRQ(ierr);
    PetscFree(vals);
  } else {
    col_lens = (int *) PetscMalloc( (4+nz)*sizeof(int) ); CHKPTRQ(col_lens);
    col_lens[0] = MAT_COOKIE;
    col_lens[1] = m;
    col_lens[2] = n;
    col_lens[3] = nz;

    /* store lengths of each row and write (including header) to file */
    for ( i=0; i<m; i++ ) col_lens[4+i] = n;
    ierr = PetscBinaryWrite(fd,col_lens,4+m,BINARY_INT,1); CHKERRQ(ierr);

    /* Possibly should write in smaller increments, not whole matrix at once? */
    /* store column indices (zero start index) */
    ict = 0;
    for ( i=0; i<m; i++ ) {
      for ( j=0; j<n; j++ ) col_lens[ict++] = j;
    }
    ierr = PetscBinaryWrite(fd,col_lens,nz,BINARY_INT,0); CHKERRQ(ierr);
    PetscFree(col_lens);

    /* store nonzero values */
    anonz = (Scalar *) PetscMalloc((nz)*sizeof(Scalar)); CHKPTRQ(anonz);
    ict = 0;
    for ( i=0; i<m; i++ ) {
      v = a->v + i;
      for ( j=0; j<n; j++ ) {
        anonz[ict++] = *v; v += a->m;
      }
    }
    ierr = PetscBinaryWrite(fd,anonz,nz,BINARY_SCALAR,0); CHKERRQ(ierr);
    PetscFree(anonz);
  }
  return 0;
}

static int MatView_SeqDense(PetscObject obj,Viewer viewer)
{
  Mat          A = (Mat) obj;
  Mat_SeqDense *a = (Mat_SeqDense*) A->data;
  ViewerType   vtype;
  int          ierr;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);

  if (vtype == MATLAB_VIEWER) {
    return ViewerMatlabPutArray_Private(viewer,a->m,a->n,a->v); 
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    return MatView_SeqDense_ASCII(A,viewer);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_SeqDense_Binary(A,viewer);
  }
  return 0;
}

static int MatDestroy_SeqDense(PetscObject obj)
{
  Mat          mat = (Mat) obj;
  Mat_SeqDense *l = (Mat_SeqDense *) mat->data;
#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows %d Cols %d",l->m,l->n);
#endif
  if (l->pivots) PetscFree(l->pivots);
  if (!l->user_alloc) PetscFree(l->v);
  PetscFree(l);
  PLogObjectDestroy(mat);
  PetscHeaderDestroy(mat);
  return 0;
}

static int MatTranspose_SeqDense(Mat A,Mat *matout)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          k, j, m, n;
  Scalar       *v, tmp;

  v = mat->v; m = mat->m; n = mat->n;
  if (matout == PETSC_NULL) { /* in place transpose */
    if (m != n) SETERRQ(1,"MatTranspose_SeqDense:Supports square matrix only in-place");
    for ( j=0; j<m; j++ ) {
      for ( k=0; k<j; k++ ) {
        tmp = v[j + k*n]; 
        v[j + k*n] = v[k + j*n];
        v[k + j*n] = tmp;
      }
    }
  }
  else { /* out-of-place transpose */
    int          ierr;
    Mat          tmat;
    Mat_SeqDense *tmatd;
    Scalar       *v2;
    ierr = MatCreateSeqDense(A->comm,mat->n,mat->m,PETSC_NULL,&tmat); CHKERRQ(ierr);
    tmatd = (Mat_SeqDense *) tmat->data;
    v = mat->v; v2 = tmatd->v;
    for ( j=0; j<n; j++ ) {
      for ( k=0; k<m; k++ ) v2[j + k*n] = v[k + j*m];
    }
    ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    *matout = tmat;
  } 
  return 0;
}

static int MatEqual_SeqDense(Mat A1,Mat A2, PetscTruth *flg)
{
  Mat_SeqDense *mat1 = (Mat_SeqDense *) A1->data;
  Mat_SeqDense *mat2 = (Mat_SeqDense *) A2->data;
  int          i;
  Scalar       *v1 = mat1->v, *v2 = mat2->v;

  if (A2->type != MATSEQDENSE) SETERRQ(1,"MatEqual_SeqDense:Both matrices should be of type  MATSEQDENSE");
  if (mat1->m != mat2->m) {*flg = PETSC_FALSE; return 0;}
  if (mat1->n != mat2->n) {*flg = PETSC_FALSE; return 0;}
  for ( i=0; i<mat1->m*mat1->n; i++ ) {
    if (*v1 != *v2) {*flg = PETSC_FALSE; return 0;}
    v1++; v2++;
  }
  *flg = PETSC_TRUE;
  return 0;
}

static int MatGetDiagonal_SeqDense(Mat A,Vec v)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          i, n, len;
  Scalar       *x, zero = 0.0;

  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetSize(v,&n);
  len = PetscMin(mat->m,mat->n);
  if (n != mat->m) SETERRQ(1,"MatGetDiagonal_SeqDense:Nonconforming mat and vec");
  for ( i=0; i<len; i++ ) {
    x[i] = mat->v[i*mat->m + i];
  }
  return 0;
}

static int MatDiagonalScale_SeqDense(Mat A,Vec ll,Vec rr)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *l,*r,x,*v;
  int          i,j,m = mat->m, n = mat->n;

  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != mat->m) SETERRQ(1,"MatDiagonalScale_SeqDense:Left scaling vec wrong size");
    for ( i=0; i<m; i++ ) {
      x = l[i];
      v = mat->v + i;
      for ( j=0; j<n; j++ ) { (*v) *= x; v+= m;} 
    }
    PLogFlops(n*m);
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != mat->n) SETERRQ(1,"MatDiagonalScale_SeqDense:Right scaling vec wrong size");
    for ( i=0; i<n; i++ ) {
      x = r[i];
      v = mat->v + i*m;
      for ( j=0; j<m; j++ ) { (*v++) *= x;} 
    }
    PLogFlops(n*m);
  }
  return 0;
}

static int MatNorm_SeqDense(Mat A,NormType type,double *norm)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  Scalar       *v = mat->v;
  double       sum = 0.0;
  int          i, j;

  if (type == NORM_FROBENIUS) {
    for (i=0; i<mat->n*mat->m; i++ ) {
#if defined(PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
    PLogFlops(2*mat->n*mat->m);
  }
  else if (type == NORM_1) {
    *norm = 0.0;
    for ( j=0; j<mat->n; j++ ) {
      sum = 0.0;
      for ( i=0; i<mat->m; i++ ) {
        sum += PetscAbsScalar(*v);  v++;
      }
      if (sum > *norm) *norm = sum;
    }
    PLogFlops(mat->n*mat->m);
  }
  else if (type == NORM_INFINITY) {
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
  }
  else {
    SETERRQ(1,"MatNorm_SeqDense:No two norm");
  }
  return 0;
}

static int MatSetOption_SeqDense(Mat A,MatOption op)
{
  Mat_SeqDense *aij = (Mat_SeqDense *) A->data;
  
  if (op == MAT_ROW_ORIENTED)            aij->roworiented = 1;
  else if (op == MAT_COLUMN_ORIENTED)    aij->roworiented = 0;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_COLUMNS_SORTED ||
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_NO_NEW_NONZERO_LOCATIONS ||
           op == MAT_YES_NEW_NONZERO_LOCATIONS ||
           op == MAT_NO_NEW_DIAGONALS ||
           op == MAT_YES_NEW_DIAGONALS)
    PLogInfo(A,"Info:MatSetOption_SeqDense:Option ignored\n");
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqDense:unknown option");}
  return 0;
}

static int MatZeroEntries_SeqDense(Mat A)
{
  Mat_SeqDense *l = (Mat_SeqDense *) A->data;
  PetscMemzero(l->v,l->m*l->n*sizeof(Scalar));
  return 0;
}

static int MatZeroRows_SeqDense(Mat A,IS is,Scalar *diag)
{
  Mat_SeqDense *l = (Mat_SeqDense *) A->data;
  int          n = l->n, i, j,ierr,N, *rows;
  Scalar       *slot;

  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
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

static int MatGetSize_SeqDense(Mat A,int *m,int *n)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  *m = mat->m; *n = mat->n;
  return 0;
}

static int MatGetOwnershipRange_SeqDense(Mat A,int *m,int *n)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  *m = 0; *n = mat->m;
  return 0;
}

static int MatGetArray_SeqDense(Mat A,Scalar **array)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  *array = mat->v;
  return 0;
}

static int MatRestoreArray_SeqDense(Mat A,Scalar **array)
{
  return 0;
}

static int MatGetSubMatrixInPlace_SeqDense(Mat A,IS isrow,IS iscol)
{
  SETERRQ(1,"MatGetSubMatrixInPlace_SeqDense:not done");
}

static int MatGetSubMatrix_SeqDense(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,
                                    Mat *submat)
{
  Mat_SeqDense *mat = (Mat_SeqDense *) A->data;
  int          nznew, *smap, i, j, ierr, oldcols = mat->n;
  int          *irow, *icol, nrows, ncols, *cwork;
  Scalar       *vwork, *val;
  Mat          newmat;

  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols); CHKERRQ(ierr);

  smap = (int *) PetscMalloc(oldcols*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PetscMalloc(ncols*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PetscMalloc(ncols*sizeof(Scalar)); CHKPTRQ(vwork);
  PetscMemzero((char*)smap,oldcols*sizeof(int));
  for ( i=0; i<ncols; i++ ) smap[icol[i]] = i+1;

  /* Create and fill new matrix */
  ierr = MatCreateSeqDense(A->comm,nrows,ncols,PETSC_NULL,&newmat); CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    nznew = 0;
    val   = mat->v + irow[i];
    for (j=0; j<oldcols; j++) {
      if (smap[j]) {
        cwork[nznew]   = smap[j] - 1;
        vwork[nznew++] = val[j * mat->m];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES); 
           CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PetscFree(smap); PetscFree(cwork); PetscFree(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  return 0;
}

static int MatCopy_SeqDense(Mat A, Mat B)
{
  Mat_SeqDense *a = (Mat_SeqDense *) A->data, *b = (Mat_SeqDense *)B->data;
  if (B->type != MATSEQDENSE) return MatCopy_Basic(A,B);
  if (a->m != b->m || a->n != b->n) SETERRQ(1,"MatCopy_SeqDense:size(B) != size(A)");
  PetscMemcpy(b->v,a->v,a->m*a->n*sizeof(Scalar));
  return 0;
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqDense,
       MatGetRow_SeqDense, MatRestoreRow_SeqDense,
       MatMult_SeqDense, MatMultAdd_SeqDense, 
       MatMultTrans_SeqDense, MatMultTransAdd_SeqDense, 
       MatSolve_SeqDense,MatSolveAdd_SeqDense,
       MatSolveTrans_SeqDense,MatSolveTransAdd_SeqDense,
       MatLUFactor_SeqDense,MatCholeskyFactor_SeqDense,
       MatRelax_SeqDense,
       MatTranspose_SeqDense,
       MatGetInfo_SeqDense,MatEqual_SeqDense,
       MatGetDiagonal_SeqDense,MatDiagonalScale_SeqDense,MatNorm_SeqDense,
       0,0,
       0, MatSetOption_SeqDense,MatZeroEntries_SeqDense,MatZeroRows_SeqDense,0,
       MatLUFactorSymbolic_SeqDense,MatLUFactorNumeric_SeqDense,
       MatCholeskyFactorSymbolic_SeqDense,MatCholeskyFactorNumeric_SeqDense,
       MatGetSize_SeqDense,MatGetSize_SeqDense,MatGetOwnershipRange_SeqDense,
       0,0, MatGetArray_SeqDense, MatRestoreArray_SeqDense,0,
       MatGetSubMatrix_SeqDense,MatGetSubMatrixInPlace_SeqDense,
       MatConvertSameType_SeqDense,0,0,0,0,
       MatAXPY_SeqDense,0,0,
       MatGetValues_SeqDense,
       MatCopy_SeqDense,0,MatScale_SeqDense};


/*@C
   MatCreateSeqDense - Creates a sequential dense matrix that 
   is stored in column major order (the usual Fortran 77 manner). Many 
   of the matrix operations use the BLAS and LAPACK routines.

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  m - number of rows
.  n - number of columns
.  data - optional location of matrix data.  Set data=PETSC_NULL for PETSc
   to control all matrix memory allocation.

   Output Parameter:
.  A - the matrix

  Notes:
  The data input variable is intended primarily for Fortran programmers
  who wish to allocate their own matrix memory space.  Most users should
  set data=PETSC_NULL.

.keywords: dense, matrix, LAPACK, BLAS

.seealso: MatCreate(), MatSetValues()
@*/
int MatCreateSeqDense(MPI_Comm comm,int m,int n,Scalar *data,Mat *A)
{
  Mat          B;
  Mat_SeqDense *b;
  int          ierr,flg;

  *A            = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATSEQDENSE,comm);
  PLogObjectCreate(B);
  b             = (Mat_SeqDense *) PetscMalloc(sizeof(Mat_SeqDense)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqDense));
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  B->destroy    = MatDestroy_SeqDense;
  B->view       = MatView_SeqDense;
  B->factor     = 0;
  PLogObjectMemory(B,sizeof(struct _Mat));
  B->data       = (void *) b;

  b->m = m;  B->m = m; B->M = m;
  b->n = n;  B->n = n; B->N = n;
  b->pivots       = 0;
  b->roworiented  = 1;
  if (data == PETSC_NULL) {
    b->v = (Scalar*) PetscMalloc((m*n+1)*sizeof(Scalar)); CHKPTRQ(b->v);
    PetscMemzero(b->v,m*n*sizeof(Scalar));
    b->user_alloc = 0;
    PLogObjectMemory(B,n*m*sizeof(Scalar));
  } 
  else { /* user-allocated storage */
    b->v = data;
    b->user_alloc = 1;
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr);}
  *A = B;
  return 0;
}

int MatCreate_SeqDense(Mat A,Mat *newmat)
{
  Mat_SeqDense *m = (Mat_SeqDense *) A->data;
  return MatCreateSeqDense(A->comm,m->m,m->n,PETSC_NULL,newmat);
}
