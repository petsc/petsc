
#include "aij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

int SpToSymmetricIJ(Matiaij*,int**,int**);
int SpOrderND(int,int*,int*,int*);
int SpOrder1WD(int,int*,int*,int*);
int SpOrderQMD(int,int*,int*,int*);
int SpOrderRCM(int,int*,int*,int*);

static int MatAIJreorder(Mat mat,int type,IS *rperm, IS *cperm)
{
  Matiaij *aij = (Matiaij *) mat->data;
  int     i, ierr, *ia, *ja, *perma;

  perma = (int *) MALLOC( aij->n*sizeof(int) ); CHKPTR(perma);

  if (ierr = SpToSymmetricIJ( aij, &ia, &ja )) SETERR(ierr,0);

  if (type == ORDER_NATURAL) {
    for ( i=0; i<aij->n; i++ ) perma[i] = i;
  }
  else if (type == ORDER_ND) {
    ierr = SpOrderND( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_1WD) {
    ierr = SpOrder1WD( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_RCM) {
    ierr = SpOrderRCM( aij->n, ia, ja, perma );
  }
  else if (type == ORDER_QMD) {
    ierr = SpOrderQMD( aij->n, ia, ja, perma );
  }
  else SETERR(1,"Cannot performing ordering requested");
  if (ierr) SETERR(ierr,0);
  FREE(ia); FREE(ja);

  if (ierr = ISCreateSequential(aij->n,perma,rperm)) SETERR(ierr,0);
  ISSetPermutation(*rperm);
  FREE(perma); 
  *cperm = *rperm; /* so far all permutations are symmetric.*/
  return 0; 
}


#define CHUNCKSIZE 5

/* This version has row oriented v  */
static int MatiAIJAddValues(Mat matin,Scalar *v,int m,int *idxm,int n,
                            int *idxn)
{
  Matiaij *mat = (Matiaij *) matin->data;
  int    *rp,k,a,b,t,ii,row,nrow,i,col,l,rmax, N, sorted = mat->sorted;
  int    *imax = mat->imax, *ai = mat->i, *ailen = mat->ilen;
  int    *aj = mat->j, nonew = mat->nonew;
  Scalar *ap,value, *aa = mat->a;

  for ( k=0; k<m; k++ ) { /* loop over added rows */
    row  = idxm[k];   rp   = aj + ai[row] - 1; ap = aa + ai[row] - 1;
    rmax = imax[row]; nrow = ailen[row]; 
    a = 0;
    for ( l=0; l<n; l++ ) { /* loop over added columns */
      col = idxn[l] + 1; value = *v++; 
      if (nrow) {
        if (!sorted) a = 0; b = nrow;
        while (b-a > 5) {
          t = (b+a)/2;
          if (rp[t] > col) b = t;
          else             a = t;
        }
        for ( i=a; i<b; i++ ) {
          if (rp[i] > col) break;
          if (rp[i] == col) {ap[i] += value;  goto noinsert;}
        } 
        if (nonew) goto noinsert;
        if (nrow >= rmax) {
          /* there is no extra room in row, therefore enlarge */
          int    new_nz = ai[mat->m] + CHUNCKSIZE,len,*new_i,*new_j;
          Scalar *new_a;
          fprintf(stderr,"Warning, enlarging matrix storage \n");

          /* malloc new storage space */
          len     = new_nz*(sizeof(int)+sizeof(Scalar))+(mat->m+1)*sizeof(int);
          new_a  = (Scalar *) MALLOC( len ); CHKPTR(new_a);
          new_j  = (int *) (new_a + new_nz);
          new_i  = new_j + new_nz;

          /* copy over old data into new slots */
          for ( ii=0; ii<row+1; ii++ ) {new_i[ii] = ai[ii];}
          for ( ii=row+1; ii<mat->m+1; ii++ ) {new_i[ii] = ai[ii]+CHUNCKSIZE;}
          MEMCPY(new_j,aj,(ai[row]+nrow-1)*sizeof(int));
          len = (new_nz - CHUNCKSIZE - ai[row] - nrow + 1);
          MEMCPY(new_j+ai[row]-1+nrow+CHUNCKSIZE,aj+ai[row]-1+nrow,
                                                           len*sizeof(int));
          MEMCPY(new_a,aa,(ai[row]+nrow-1)*sizeof(Scalar));
          MEMCPY(new_a+ai[row]-1+nrow+CHUNCKSIZE,aa+ai[row]-1+nrow,
                                                         len*sizeof(Scalar)); 
          /* free up old matrix storage */
          FREE(mat->a); if (!mat->singlemalloc) {FREE(mat->i);FREE(mat->j);}
          aa = mat->a = new_a; ai = mat->i = new_i; aj = mat->j = new_j; 
          mat->singlemalloc = 1;

          rp   = aj + ai[row] - 1; ap = aa + ai[row] - 1;
          rmax = imax[row] = imax[row] + CHUNCKSIZE;
          mat->mem += CHUNCKSIZE*(sizeof(int) + sizeof(Scalar));
        }
        N = nrow++ - 1; mat->nz++;
        /* this has too many shifts here; but alternative was slower*/
        for ( ii=N; ii>=i; ii-- ) {/* shift values up*/
          rp[ii+1] = rp[ii];
          ap[ii+1] = ap[ii];
        }
        rp[i] = col; 
        ap[i] = value; 
        noinsert:;
        a = i + 1;
      }
      else {
        ap[0] = value; rp[0] = col; nrow++; a = 1;
      }
    }
    ailen[row] = nrow;
  }
  return 0;
} 

static int MatiAIJview(PetscObject obj,Viewer ptr)
{
  Mat     aijin = (Mat) obj;
  Matiaij *aij = (Matiaij *) aijin->data;
  int i,j;
  for ( i=0; i<aij->m; i++ ) {
    printf("row %d:",i);
    for ( j=aij->i[i]-1; j<aij->i[i+1]-1; j++ ) {
#if defined(PETSC_COMPLEX)
      printf(" %d %g + %g i",aij->j[j]-1,real(aij->a[j]),imag(aij->a[j]));
#else
      printf(" %d %g ",aij->j[j]-1,aij->a[j]);
#endif
    }
    printf("\n");
  }
  return 0;
}

static int MatiAIJEndAssemble(Mat aijin)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  int    shift = 0,i,j,*ai = aij->i, *aj = aij->j, *imax = aij->imax;
  int    m = aij->m, *ip, N, *ailen = aij->ilen;
  Scalar *a = aij->a, *ap;

  for ( i=1; i<m; i++ ) {
    shift += imax[i-1] - ailen[i-1];
    if (shift) {
      ip = aj + ai[i] - 1; ap = a + ai[i] - 1;
      N = ailen[i];
      for ( j=0; j<N; j++ ) {
        ip[j-shift] = ip[j];
        ap[j-shift] = ap[j]; 
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  shift += imax[m-1] - ailen[m-1];
  ai[m] = ai[m-1] + ailen[m-1];
  FREE(aij->imax);
  FREE(aij->ilen);
  aij->mem -= 2*(aij->m)*sizeof(int);
  return 0;
}

static int MatiAIJzeroentries(Mat mat)
{
  Matiaij *aij = (Matiaij *) mat->data; 
  Scalar  *a = aij->a;
  int     i,n = aij->i[aij->m]-1;
  for ( i=0; i<n; i++ ) a[i] = 0.0;
  return 0;

}
static int MatiAIJdestroy(PetscObject obj)
{
  Mat mat = (Mat) obj;
  Matiaij *aij = (Matiaij *) mat->data;
  FREE(aij->a); 
  if (!aij->singlemalloc) {FREE(aij->i); FREE(aij->j);}
  FREE(aij); FREE(mat);
  return 0;
}

static int MatiAIJCompress(Mat aijin)
{
  return 0;
}

static int MatiAIJinsopt(Mat aijin,int op)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  if      (op == ROW_ORIENTED)              aij->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           aij->roworiented = 0;
  else if (op == COLUMNS_SORTED)            aij->sorted      = 1;
  /* doesn't care about sorted rows */
  else if (op == NO_NEW_NONZERO_LOCATIONS)  aij->nonew = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) aij->nonew = 0;

  if (op == COLUMN_ORIENTED) SETERR(1,"Column oriented input not supported");
  return 0;
}

static int MatiAIJgetdiag(Mat aijin,Vec v)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  int    i,j, n;
  Scalar *x, zero = 0.0;
  CHKTYPE(v,SEQVECTOR);
  VecSet(&zero,v);
  VecGetArray(v,&x); VecGetSize(v,&n);
  if (n != aij->m) SETERR(1,"Nonconforming matrix and vector");
  for ( i=0; i<aij->m; i++ ) {
    for ( j=aij->i[i]-1; j<aij->i[i+1]-1; j++ ) {
      if (aij->j[j]-1 == i) {
        x[i] = aij->a[j];
        break;
      }
    }
  }
  return 0;
}

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
static int MatiAIJmulttrans(Mat aijin,Vec xx,Vec yy)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  Scalar  *x, *y, *v, alpha;
  int     m = aij->m, n, i, *idx;
  CHKTYPE(xx,SEQVECTOR);CHKTYPE(yy,SEQVECTOR);
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  MEMSET(y,0,aij->n*sizeof(Scalar));
  y--; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = aij->j + aij->i[i] - 1;
    v     = aij->a + aij->i[i] - 1;
    n     = aij->i[i+1] - aij->i[i];
    alpha = x[i];
    /* should inline */
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  return 0;
}
static int MatiAIJmulttransadd(Mat aijin,Vec xx,Vec zz,Vec yy)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  Scalar  *x, *y, *z, *v, alpha;
  int     m = aij->m, n, i, *idx;
  CHKTYPE(xx,SEQVECTOR);CHKTYPE(yy,SEQVECTOR);
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  if (zz != yy) VecCopy(zz,yy);
  y--; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx   = aij->j + aij->i[i] - 1;
    v     = aij->a + aij->i[i] - 1;
    n     = aij->i[i+1] - aij->i[i];
    alpha = x[i];
    /* should inline */
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  return 0;
}

static int MatiAIJmult(Mat aijin,Vec xx,Vec yy)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  Scalar  *x, *y, *v, sum;
  int     m = aij->m, n, i, *idx;
  CHKTYPE(xx,SEQVECTOR);CHKTYPE(yy,SEQVECTOR);
  VecGetArray(xx,&x); VecGetArray(yy,&y);
  x--; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx  = aij->j + aij->i[i] - 1;
    v    = aij->a + aij->i[i] - 1;
    n    = aij->i[i+1] - aij->i[i];
    sum  = 0.0;
    SPARSEDENSEDOT(sum,x,v,idx,n); 
    y[i] = sum;
  }
  return 0;
}

static int MatiAIJmultadd(Mat aijin,Vec xx,Vec yy,Vec zz)
{
  Matiaij *aij = (Matiaij *) aijin->data;
  Scalar  *x, *y, *z, *v, sum;
  int     m = aij->m, n, i, *idx;
  CHKTYPE(xx,SEQVECTOR);CHKTYPE(yy,SEQVECTOR); CHKTYPE(zz,SEQVECTOR);
  VecGetArray(xx,&x); VecGetArray(yy,&y); VecGetArray(zz,&z); 
  x--; /* shift for Fortran start by 1 indexing */
  for ( i=0; i<m; i++ ) {
    idx  = aij->j + aij->i[i] - 1;
    v    = aij->a + aij->i[i] - 1;
    n    = aij->i[i+1] - aij->i[i];
    sum  = y[i];
    SPARSEDENSEDOT(sum,x,v,idx,n); 
    y[i] = sum;
  }
  return 0;
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/

static int MatiAIJmarkdiag(Matiaij  *aij)
{
  int    i,j, n, *diag;;
  diag = (int *) MALLOC( aij->m*sizeof(int)); CHKPTR(diag);
  for ( i=0; i<aij->m; i++ ) {
    for ( j=aij->i[i]-1; j<aij->i[i+1]-1; j++ ) {
      if (aij->j[j]-1 == i) {
        diag[i] = j + 1;
        break;
      }
    }
  }
  aij->diag = diag;
  return 0;
}

static int MatiAIJrelax(Mat matin,Vec bb,double omega,int flag,IS is,
                        int its,Vec xx)
{
  Matiaij *mat = (Matiaij *) matin->data;
  Scalar *x, *b, zero = 0.0, d, *xs, sum, *v = mat->a;
  int    ierr,one = 1, tmp, *idx, *diag;
  int    n = mat->n, m = mat->m, i, j;

  if (is) SETERR(1,"No support for ordering in relaxation");
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (ierr = VecSet(&zero,xx)) return ierr;
  }
  VecGetArray(xx,&x); VecGetArray(bb,&b);
  if (!mat->diag) {if (ierr = MatiAIJmarkdiag(mat)) return ierr;}
  diag = mat->diag;
  xs = x - 1; /* shifted by one for index start of a or mat->j*/
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP){
      for ( i=0; i<m; i++ ) {
        d    = mat->a[diag[i]-1];
        n    = mat->i[i+1] - mat->i[i]; 
        idx  = mat->j + mat->i[i] - 1;
        v    = mat->a + mat->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
    if (flag & SOR_BACKWARD_SWEEP){
      for ( i=m-1; i>=0; i-- ) {
        d    = mat->a[diag[i] - 1];
        n    = mat->i[i+1] - mat->i[i]; 
        idx  = mat->j + mat->i[i] - 1;
        v    = mat->a + mat->i[i] - 1;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum/d + x[i]);
      }
    }
  }
  return 0;
} 

static int MatiAIJnz(Mat matin,int *nz)
{
  Matiaij *mat = (Matiaij *) matin->data;
  *nz = mat->nz;
  return 0;
}
static int MatiAIJmem(Mat matin,int *nz)
{
  Matiaij *mat = (Matiaij *) matin->data;
  *nz = mat->mem;
  return 0;
}

int MatiAIJLUFactorSymbolic(Mat,IS,IS,Mat*);
int MatiAIJLUFactorNumeric(Mat,Mat);
int MatiAIJSolve(Mat,Vec,Vec);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatiAIJAddValues,MatiAIJAddValues,
       0, 0,
       MatiAIJmult,MatiAIJmultadd,MatiAIJmulttrans,MatiAIJmulttransadd,
       MatiAIJSolve,0,0,0,
       0,0,
       MatiAIJrelax,
       0,
       MatiAIJnz,MatiAIJmem,0,
       0,
       MatiAIJgetdiag,0,0,
       0,MatiAIJEndAssemble,
       MatiAIJCompress,
       MatiAIJinsopt,MatiAIJzeroentries,0,MatAIJreorder,
       MatiAIJLUFactorSymbolic,MatiAIJLUFactorNumeric,0,0 };



/*@

      MatCreateSequentialAIJ - Creates a sparse matrix in AIJ format.

  Input Parameters:
.   m,n - number of rows and columns
.   nz - total number nonzeros in matrix
.   nzz - number of nonzeros per row or null
.       You must leave room for the diagonal entry even if it is zero.

  Output parameters:
.  newmat - the matrix 

  Keywords: matrix, aij, compressed row, sparse
@*/
int MatCreateSequentialAIJ(int m,int n,int nz,int *nnz, Mat *newmat)
{
  Mat       mat;
  Matiaij   *aij;
  int       i,rl,len;
  *newmat      = 0;
  CREATEHEADER(mat,_Mat);
  mat->data    = (void *) (aij = NEW(Matiaij)); CHKPTR(aij);
  mat->cookie  = MAT_COOKIE;
  mat->ops     = &MatOps;
  mat->destroy = MatiAIJdestroy;
  mat->view    = MatiAIJview;
  mat->type    = MATAIJSEQ;
  mat->factor  = 0;
  mat->row     = 0;
  mat->col     = 0;
  aij->m       = m;
  aij->n       = n;
  aij->imax    = (int *) MALLOC( m*sizeof(int) ); CHKPTR(aij->imax);
  aij->mem     = m*sizeof(int) + sizeof(Matiaij);
  if (!nnz) {
    if (nz <= 0) nz = 1;
    for ( i=0; i<m; i++ ) aij->imax[i] = nz;
    nz = nz*m;
  }
  else {
    nz = 0;
    for ( i=0; i<m; i++ ) {aij->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  aij->nz = nz;
  len     = nz*(sizeof(int) + sizeof(Scalar)) + (aij->m+1)*sizeof(int);
  aij->a  = (Scalar *) MALLOC( len ); CHKPTR(aij->a);
  aij->j  = (int *) (aij->a + nz);
  aij->i  = aij->j + nz;
  aij->singlemalloc = 1;
  aij->mem += len;

  aij->i[0] = 1;
  for (i=1; i<m+1; i++) {
    aij->i[i] = aij->i[i-1] + aij->imax[i-1];
  }

  /* aij->ilen will count nonzeros in each row so far. */
  aij->ilen = (int *) MALLOC((aij->m)*sizeof(int)); 

  /* stick in zeros along diagonal */
  for ( i=0; i<aij->m; i++ ) {
    aij->ilen[i] = 1;
    aij->j[aij->i[i]-1] = i+1;
    aij->a[aij->i[i]-1] = 0.0;
  }
  aij->nz = aij->m;
  aij->mem += (aij->m)*sizeof(int) + len;

  aij->sorted      = 0;
  aij->roworiented = 1;
  aij->nonew       = 0;
  aij->diag        = 0;

  *newmat = mat;
  return 0;
}
