
#ifndef lint
static char vcid[] = "$Id: baij.c,v 1.14 1996/03/18 00:40:53 bsmith Exp bsmith $";
#endif

/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/
#include "baij.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"
#include "petsc.h"

extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

static int MatGetReordering_SeqBAIJ(Mat A,MatOrdering type,IS *rperm,IS *cperm)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, *ia, *ja,n = a->mbs,*idx,i,ishift,oshift;

  /* 
     this is tacky: In the future when we have written special factorization
     and solve routines for the identity permutation we should use a 
     stride index set instead of the general one.
  */
  if (type  == ORDER_NATURAL) {
    idx = (int *) PetscMalloc( n*sizeof(int) ); CHKPTRQ(idx);
    for ( i=0; i<n; i++ ) idx[i] = i;
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,rperm); CHKERRQ(ierr);
    ierr = ISCreateSeq(MPI_COMM_SELF,n,idx,cperm); CHKERRQ(ierr);
    PetscFree(idx);
    ISSetPermutation(*rperm);
    ISSetPermutation(*cperm);
    ISSetIdentity(*rperm);
    ISSetIdentity(*cperm);
    return 0;
  }

  MatReorderingRegisterAll();
  ishift = a->indexshift;
  oshift = -MatReorderingIndexShift[(int)type];
  if (MatReorderingRequiresSymmetric[(int)type]) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->n,a->i,a->j,ishift,oshift,&ia,&ja);
    CHKERRQ(ierr);
    ierr = MatGetReordering_IJ(a->n,ia,ja,type,rperm,cperm); CHKERRQ(ierr);
    PetscFree(ia); PetscFree(ja);
  } else {
    if (ishift == oshift) {
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm);CHKERRQ(ierr);
    }
    else if (ishift == -1) {
      /* temporarily subtract 1 from i and j indices */
      int nz = a->i[a->n] - 1; 
      for ( i=0; i<nz; i++ ) a->j[i]--;
      for ( i=0; i<a->n+1; i++ ) a->i[i]--;
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm);CHKERRQ(ierr);
      for ( i=0; i<nz; i++ ) a->j[i]++;
      for ( i=0; i<a->n+1; i++ ) a->i[i]++;
    } else {
      /* temporarily add 1 to i and j indices */
      int nz = a->i[a->n] - 1; 
      for ( i=0; i<nz; i++ ) a->j[i]++;
      for ( i=0; i<a->n+1; i++ ) a->i[i]++;
      ierr = MatGetReordering_IJ(a->n,a->i,a->j,type,rperm,cperm);CHKERRQ(ierr);
      for ( i=0; i<nz; i++ ) a->j[i]--;
      for ( i=0; i<a->n+1; i++ ) a->i[i]--;
    }
  }
  return 0; 
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/

int MatMarkDiag_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data; 
  int         i,j, *diag, m = a->mbs;

  diag = (int *) PetscMalloc( (m+1)*sizeof(int)); CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for ( i=0; i<m; i++ ) {
    for ( j=a->i[i]; j<a->i[i+1]; j++ ) {
      if (a->j[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
  a->diag = diag;
  return 0;
}

#include "draw.h"
#include "pinclude/pviewer.h"
#include "sys.h"

static int MatView_SeqBAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         i, fd, *col_lens, ierr, bs = a->bs,count,*jj,j,k,l;
  Scalar      *aa;

  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  col_lens = (int *) PetscMalloc((4+a->m)*sizeof(int));CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz*bs*bs;

  /* store lengths of each row and write (including header) to file */
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      col_lens[4+count++] = bs*(a->i[i+1] - a->i[i]);
    }
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,BINARY_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* store column indices (zero start index) */
  jj = (int *) PetscMalloc( a->nz*bs*bs*sizeof(int) ); CHKPTRQ(jj);
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
        for ( l=0; l<bs; l++ ) {
          jj[count++] = bs*a->j[k] + l;
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,jj,bs*bs*a->nz,BINARY_INT,0); CHKERRQ(ierr);
  PetscFree(jj);

  /* store nonzero values */
  aa = (Scalar *) PetscMalloc(a->nz*bs*bs*sizeof(Scalar)); CHKPTRQ(aa);
  count = 0;
  for ( i=0; i<a->mbs; i++ ) {
    for ( j=0; j<bs; j++ ) {
      for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
        for ( l=0; l<bs; l++ ) {
          aa[count++] = a->a[bs*bs*k + l*bs + j];
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,aa,bs*bs*a->nz,BINARY_SCALAR,0); CHKERRQ(ierr);
  PetscFree(aa);
  return 0;
}

static int MatView_SeqBAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  int         ierr, i,j,format,bs = a->bs,k,l;
  FILE        *fd;
  char        *outputname;

  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);
  if (format == ASCII_FORMAT_INFO) {
    /* no need to print additional information */ ;
  } 
  else if (format == ASCII_FORMAT_MATLAB) {
    SETERRQ(1,"MatView_SeqBAIJ_ASCII:Matlab format not supported");
  } 
  else {
    for ( i=0; i<a->mbs; i++ ) {
      for ( j=0; j<bs; j++ ) {
        fprintf(fd,"row %d:",i*bs+j);
        for ( k=a->i[i]; k<a->i[i+1]; k++ ) {
          for ( l=0; l<bs; l++ ) {
#if defined(PETSC_COMPLEX)
          if (imag(a->a[bs*bs*k + l*bs + j]) != 0.0) {
            fprintf(fd," %d %g + %g i",bs*a->j[k]+l,
              real(a->a[bs*bs*k + l*bs + j]),imag(a->a[bs*bs*k + l*bs + j]));
          }
          else {
            fprintf(fd," %d %g",bs*a->j[k]+l,real(a->a[bs*bs*k + l*bs + j]));
          }
#else
            fprintf(fd," %d %g",bs*a->j[k]+l,a->a[bs*bs*k + l*bs + j]);
#endif
          }
        }
        fprintf(fd,"\n");
      }
    } 
  }
  fflush(fd);
  return 0;
}

static int MatView_SeqBAIJ(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  ViewerType  vtype;
  int         ierr;

  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; 
  }

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    SETERRQ(1,"MatView_SeqBAIJ:Matlab viewer not supported");
  }
  else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    return MatView_SeqBAIJ_ASCII(A,viewer);
  }
  else if (vtype == BINARY_FILE_VIEWER) {
    return MatView_SeqBAIJ_Binary(A,viewer);
  }
  else if (vtype == DRAW_VIEWER) {
    SETERRQ(1,"MatView_SeqBAIJ:Draw viewer not supported");
  }
  return 0;
}


static int MatZeroEntries_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data; 
  PetscMemzero(a->a,a->bs*a->bs*a->i[a->mbs]*sizeof(Scalar));
  return 0;
}

int MatDestroy_SeqBAIJ(PetscObject obj)
{
  Mat        A  = (Mat) obj;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
#endif
  PetscFree(a->a); 
  if (!a->singlemalloc) { PetscFree(a->i); PetscFree(a->j);}
  if (a->diag) PetscFree(a->diag);
  if (a->ilen) PetscFree(a->ilen);
  if (a->imax) PetscFree(a->imax);
  if (a->solve_work) PetscFree(a->solve_work);
  if (a->mult_work) PetscFree(a->mult_work);
  PetscFree(a); 
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  return 0;
}

static int MatSetOption_SeqBAIJ(Mat A,MatOption op)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  if      (op == ROW_ORIENTED)              a->roworiented = 1;
  else if (op == COLUMN_ORIENTED)           a->roworiented = 0;
  else if (op == COLUMNS_SORTED)            a->sorted      = 1;
  else if (op == NO_NEW_NONZERO_LOCATIONS)  a->nonew       = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == ROWS_SORTED || 
           op == SYMMETRIC_MATRIX ||
           op == STRUCTURALLY_SYMMETRIC_MATRIX ||
           op == YES_NEW_DIAGONALS)
    PLogInfo((PetscObject)A,"Info:MatSetOption_SeqBAIJ:Option ignored\n");
  else if (op == NO_NEW_DIAGONALS)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqBAIJ:NO_NEW_DIAGONALS");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqBAIJ:unknown option");}
  return 0;
}


/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#include "pinclude/plapack.h"

static int MatMult_SeqBAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBAIJ     *a = (Mat_SeqBAIJ *) A->data;
  Scalar          *xg,*yg;
  register Scalar *x, *y, *v, sum,*xb, sum1,sum2,sum3,sum4,sum5;
  register Scalar x1,x2,x3,x4,x5;
  int             mbs = a->mbs, m = a->m, i, *idx,*ii;
  int             bs = a->bs,j,n,bs2 = bs*bs;

  VecGetArray(xx,&xg); x = xg;  VecGetArray(yy,&yg); y = yg;
  PetscMemzero(y,m*sizeof(Scalar));
  x     = x;
  idx   = a->j;
  v     = a->a;
  ii    = a->i;

  switch (bs) {
    case 1:
     for ( i=0; i<m; i++ ) {
       n    = ii[1] - ii[0]; ii++;
       sum  = 0.0;
       while (n--) sum += *v++ * x[*idx++];
       y[i] = sum;
      }
      break;
    case 2:
      for ( i=0; i<mbs; i++ ) {
        n  = ii[1] - ii[0]; ii++; 
        sum1 = 0.0; sum2 = 0.0;
        for ( j=0; j<n; j++ ) {
          xb = x + 2*(*idx++); x1 = xb[0]; x2 = xb[1];
          sum1 += v[0]*x1 + v[2]*x2;
          sum2 += v[1]*x1 + v[3]*x2;
          v += 4;
        }
        y[0] += sum1; y[1] += sum2;
        y += 2;
      }
      break;
    case 3:
      for ( i=0; i<mbs; i++ ) {
        n  = ii[1] - ii[0]; ii++; 
        sum1 = 0.0; sum2 = 0.0; sum3 = 0.0;
        for ( j=0; j<n; j++ ) {
          xb = x + 3*(*idx++); x1 = xb[0]; x2 = xb[1]; x3 = xb[2];
          sum1 += v[0]*x1 + v[3]*x2 + v[6]*x3;
          sum2 += v[1]*x1 + v[4]*x2 + v[7]*x3;
          sum3 += v[2]*x1 + v[5]*x2 + v[8]*x3;
          v += 9;
        }
        y[0] += sum1; y[1] += sum2; y[2] += sum3;
        y += 3;
      }
      break;
    case 4:
      for ( i=0; i<mbs; i++ ) {
        n  = ii[1] - ii[0]; ii++; 
        sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0;
        for ( j=0; j<n; j++ ) {
          xb = x + 4*(*idx++);
          x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3];
          sum1 += v[0]*x1 + v[4]*x2 + v[8]*x3   + v[12]*x4;
          sum2 += v[1]*x1 + v[5]*x2 + v[9]*x3   + v[13]*x4;
          sum3 += v[2]*x1 + v[6]*x2 + v[10]*x3  + v[14]*x4;
          sum4 += v[3]*x1 + v[7]*x2 + v[11]*x3  + v[15]*x4;
          v += 16;
        }
        y[0] += sum1; y[1] += sum2; y[2] += sum3; y[3] += sum4;
        y += 4;
      }
      break;
    case 5:
      for ( i=0; i<mbs; i++ ) {
        n  = ii[1] - ii[0]; ii++; 
        sum1 = 0.0; sum2 = 0.0; sum3 = 0.0; sum4 = 0.0; sum5 = 0.0;
        for ( j=0; j<n; j++ ) {
          xb = x + 5*(*idx++);
          x1 = xb[0]; x2 = xb[1]; x3 = xb[2]; x4 = xb[3]; x5 = xb[4];
          sum1 += v[0]*x1 + v[5]*x2 + v[10]*x3  + v[15]*x4 + v[20]*x5;
          sum2 += v[1]*x1 + v[6]*x2 + v[11]*x3  + v[16]*x4 + v[21]*x5;
          sum3 += v[2]*x1 + v[7]*x2 + v[12]*x3  + v[17]*x4 + v[22]*x5;
          sum4 += v[3]*x1 + v[8]*x2 + v[13]*x3  + v[18]*x4 + v[23]*x5;
          sum5 += v[4]*x1 + v[9]*x2 + v[14]*x3  + v[19]*x4 + v[24]*x5;
          v += 25;
        }
        y[0] += sum1; y[1] += sum2; y[2] += sum3; y[3] += sum4; y[4] += sum5;
        y += 5;
      }
      break;
      /* block sizes larger then 5 by 5 are handled by BLAS */
    default: {
      int  _One = 1,ncols,k; Scalar _DOne = 1.0, *work,*workt;
      if (!a->mult_work) {
        a->mult_work = (Scalar *) PetscMalloc(a->m*sizeof(Scalar));
        CHKPTRQ(a->mult_work);
      }
      work = a->mult_work;
      for ( i=0; i<mbs; i++ ) {
        n     = ii[1] - ii[0]; ii++;
        ncols = n*bs;
        workt = work;
        for ( j=0; j<n; j++ ) {
          xb = x + bs*(*idx++);
          for ( k=0; k<bs; k++ ) workt[k] = xb[k];
          workt += bs;
        }
        LAgemv_("N",&bs,&ncols,&_DOne,v,&bs,work,&_One,&_DOne,y,&_One);
        v += n*bs2;
        y += bs;
      }
    }
  }
  PLogFlops(2*bs2*a->nz - m);
  return 0;
}

static int MatGetInfo_SeqBAIJ(Mat A,MatInfoType flag,int *nz,int *nza,int *mem)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  if (nz)  *nz  = a->bs*a->bs*a->nz;
  if (nza) *nza = a->maxnz;
  if (mem) *mem = (int)A->mem;
  return 0;
}

extern int MatLUFactorSymbolic_SeqBAIJ(Mat,IS,IS,double,Mat*);
extern int MatLUFactor_SeqBAIJ(Mat,IS,IS,double);
extern int MatSolve_SeqBAIJ(Mat,Vec,Vec);
extern int MatLUFactorNumeric_SeqBAIJ_N(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_1(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_2(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_3(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_4(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBAIJ_5(Mat,Mat*);

static int MatGetSize_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  *m = a->m; *n = a->n;
  return 0;
}

static int MatGetOwnershipRange_SeqBAIJ(Mat A,int *m,int *n)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  *m = 0; *n = a->m;
  return 0;
}

static int MatNorm_SeqBAIJ(Mat A,NormType type,double *norm)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) A->data;
  Scalar      *v = a->a;
  double      sum = 0.0;
  int         i;

  if (type == NORM_FROBENIUS) {
    for (i=0; i<a->nz; i++ ) {
#if defined(PETSC_COMPLEX)
      sum += real(conj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  }
  else {
    SETERRQ(1,"MatNorm_SeqBAIJ:No support for this norm yet");
  }
  return 0;
}

/*
     note: This can only work for identity for row and col. It would 
   be good to check this and otherwise generate an error.
*/
static int MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,double efill,int fill)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) inA->data;
  Mat         outA;
  int         ierr;

  if (fill != 0) SETERRQ(1,"MatILUFactor_SeqBAIJ:Only fill=0 supported");

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;

  a->solve_work = (Scalar *) PetscMalloc((a->m+a->bs)*sizeof(Scalar));CHKPTRQ(a->solve_work);

  if (!a->diag) {
    ierr = MatMarkDiag_SeqBAIJ(inA); CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(inA,&outA); CHKERRQ(ierr);
  return 0;
}

static int MatScale_SeqBAIJ(Scalar *alpha,Mat inA)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ *) inA->data;
  int         one = 1, totalnz = a->bs*a->bs*a->nz;
  BLscal_( &totalnz, alpha, a->a, &one );
  PLogFlops(totalnz);
  return 0;
}

int MatPrintHelp_SeqBAIJ(Mat A)
{
  static int called = 0; 

  if (called) return 0; else called = 1;
  return 0;
}
/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {0,
       0,0,
       MatMult_SeqBAIJ,0,
       0,0,
       MatSolve_SeqBAIJ,0,
       0,0,
       MatLUFactor_SeqBAIJ,0,
       0,
       0,
       MatGetInfo_SeqBAIJ,0,
       0,0,MatNorm_SeqBAIJ,
       0,0,
       0,
       MatSetOption_SeqBAIJ,MatZeroEntries_SeqBAIJ,0,
       MatGetReordering_SeqBAIJ,
       MatLUFactorSymbolic_SeqBAIJ,MatLUFactorNumeric_SeqBAIJ_N,0,0,
       MatGetSize_SeqBAIJ,MatGetSize_SeqBAIJ,MatGetOwnershipRange_SeqBAIJ,
       MatILUFactorSymbolic_SeqBAIJ,0,
       0,0,/*  MatConvert_SeqBAIJ  */ 0,
       0,0,
       MatConvertSameType_SeqBAIJ,0,0,
       MatILUFactor_SeqBAIJ,0,0,
       0,0,
       0,0,
       MatPrintHelp_SeqBAIJ,MatScale_SeqBAIJ,
       0};

/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or nzz).  By setting these parameters accurately, performance can be
   increased by more than a factor of 50.

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  nz - number of block nonzeros per block row (same for all rows)
.  nzz - number of block nonzeros per block row or PETSC_NULL
         (possibly different for each row)

   Output Parameter:
.  A - the matrix 

   Notes:
   The BAIJ format, is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices and the file $(PETSC_DIR)/Performance.

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues()
@*/
int MatCreateSeqBAIJ(MPI_Comm comm,int bs,int m,int n,int nz,int *nnz, Mat *A)
{
  Mat         B;
  Mat_SeqBAIJ *b;
  int         i,len,ierr, flg,mbs = m/bs;

  if (mbs*bs != m) 
    SETERRQ(1,"MatCreateSeqBAIJ:Number rows must be divisible by blocksize");

  *A      = 0;
  PetscHeaderCreate(B,_Mat,MAT_COOKIE,MATSEQBAIJ,comm);
  PLogObjectCreate(B);
  B->data             = (void *) (b = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(b);
  PetscMemcpy(&B->ops,&MatOps,sizeof(struct _MatOps));
  switch (bs) {
    case 1:
      B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;  
      break;
    case 2:
      B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;  
      break;
    case 3:
      B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;  
      break;
    case 4:
      B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;  
      break;
    case 5:
      B->ops.lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;  
      break;
  }
  B->destroy          = MatDestroy_SeqBAIJ;
  B->view             = MatView_SeqBAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  b->row              = 0;
  b->col              = 0;
  b->reallocs         = 0;
  
  b->m       = m;
  b->mbs     = mbs;
  b->n       = n;
  b->imax    = (int *) PetscMalloc( (mbs+1)*sizeof(int) ); CHKPTRQ(b->imax);
  if (nnz == PETSC_NULL) {
    if (nz == PETSC_DEFAULT) nz = 5;
    else if (nz <= 0)        nz = 1;
    for ( i=0; i<mbs; i++ ) b->imax[i] = nz;
    nz = nz*mbs;
  }
  else {
    nz = 0;
    for ( i=0; i<mbs; i++ ) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len   = nz*sizeof(int) + nz*bs*bs*sizeof(Scalar) + (b->m+1)*sizeof(int);
  b->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(b->a);
  PetscMemzero(b->a,nz*bs*bs*sizeof(Scalar));
  b->j  = (int *) (b->a + nz*bs*bs);
  PetscMemzero(b->j,nz*sizeof(int));
  b->i  = b->j + nz;
  b->singlemalloc = 1;

  b->i[0] = 0;
  for (i=1; i<mbs+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each block row so far. */
  b->ilen = (int *) PetscMalloc((mbs+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(mbs+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqBAIJ));
  for ( i=0; i<mbs; i++ ) { b->ilen[i] = 0;}

  b->bs               = bs;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz;
  b->sorted           = 0;
  b->roworiented      = 1;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->mult_work        = 0;
  b->spptr            = 0;

  *A = B;
  ierr = OptionsHasName(PETSC_NULL,"-help", &flg); CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B); CHKERRQ(ierr); }
  return 0;
}

int MatConvertSameType_SeqBAIJ(Mat A,Mat *B,int cpvalues)
{
  Mat         C;
  Mat_SeqBAIJ *c,*a = (Mat_SeqBAIJ *) A->data;
  int         i,len, mbs = a->mbs, bs = a->bs,nz = a->nz;

  if (a->i[mbs] != nz) SETERRQ(1,"MatConvertSameType_SeqBAIJ:Corrupt matrix");

  *B = 0;
  PetscHeaderCreate(C,_Mat,MAT_COOKIE,MATSEQBAIJ,A->comm);
  PLogObjectCreate(C);
  C->data       = (void *) (c = PetscNew(Mat_SeqBAIJ)); CHKPTRQ(c);
  PetscMemcpy(&C->ops,&A->ops,sizeof(struct _MatOps));
  C->destroy    = MatDestroy_SeqBAIJ;
  C->view       = MatView_SeqBAIJ;
  C->factor     = A->factor;
  c->row        = 0;
  c->col        = 0;
  C->assembled  = PETSC_TRUE;

  c->m          = a->m;
  c->n          = a->n;
  c->bs         = a->bs;
  c->mbs        = a->mbs;
  c->nbs        = a->nbs;

  c->imax       = (int *) PetscMalloc((mbs+1)*sizeof(int)); CHKPTRQ(c->imax);
  c->ilen       = (int *) PetscMalloc((mbs+1)*sizeof(int)); CHKPTRQ(c->ilen);
  for ( i=0; i<mbs; i++ ) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = 1;
  len   = (mbs+1)*sizeof(int) + nz*(bs*bs*sizeof(Scalar) + sizeof(int));
  c->a  = (Scalar *) PetscMalloc( len ); CHKPTRQ(c->a);
  c->j  = (int *) (c->a + nz*bs*bs);
  c->i  = c->j + nz;
  PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(int));
  if (mbs > 0) {
    PetscMemcpy(c->j,a->j,nz*sizeof(int));
    if (cpvalues == COPY_VALUES) {
      PetscMemcpy(c->a,a->a,bs*bs*nz*sizeof(Scalar));
    }
  }

  PLogObjectMemory(C,len+2*(mbs+1)*sizeof(int)+sizeof(struct _Mat)+sizeof(Mat_SeqBAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  if (a->diag) {
    c->diag = (int *) PetscMalloc( (mbs+1)*sizeof(int) ); CHKPTRQ(c->diag);
    PLogObjectMemory(C,(mbs+1)*sizeof(int));
    for ( i=0; i<mbs; i++ ) {
      c->diag[i] = a->diag[i];
    }
  }
  else c->diag          = 0;
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */
  c->mult_work          = 0;
  *B = C;
  return 0;
}

int MatLoad_SeqBAIJ(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqBAIJ  *a;
  Mat          B;
  int          i,nz,ierr,fd,header[4],size,*rowlengths=0,M,N,bs=1,flg;
  int          *mask,mbs,*jj,j,rowcount,nzcount,k,*browlengths,maskcount;
  int          kmax,jcount,block,idx,point,nzcountb,extra_rows;
  int          *masked, nmask,tmp,ishift,bs2;
  Scalar       *aa;
  MPI_Comm     comm = ((PetscObject) viewer)->comm;

  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg);CHKERRQ(ierr);
  bs2  = bs*bs;

  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(1,"MatLoad_SeqBAIJ:view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,BINARY_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(1,"MatLoad_SeqBAIJ:not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (M != N) SETERRQ(1,"MatLoad_SeqBAIJ:Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  mbs        = M/bs;
  extra_rows = bs - M + bs*(mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  mbs++;
  if (extra_rows) {
    PLogInfo(0,"MatLoad_SeqBAIJ:Padding loading matrix to match blocksize");
  }

  /* read in row lengths */
  rowlengths = (int*) PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,BINARY_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;

  /* read in column indices */
  jj = (int*) PetscMalloc( (nz+extra_rows)*sizeof(int) ); CHKPTRQ(jj);
  ierr = PetscBinaryRead(fd,jj,nz,BINARY_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) jj[nz+i] = M+i;

  /* loop over row lengths determining block row lengths */
  browlengths = (int *) PetscMalloc(mbs*sizeof(int));CHKPTRQ(browlengths);
  PetscMemzero(browlengths,mbs*sizeof(int));
  mask   = (int *) PetscMalloc( 2*mbs*sizeof(int) ); CHKPTRQ(mask);
  PetscMemzero(mask,mbs*sizeof(int));
  masked = mask + mbs;
  rowcount = 0; nzcount = 0;
  for ( i=0; i<mbs; i++ ) {
    nmask = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[rowcount];
      for ( k=0; k<kmax; k++ ) {
        tmp = jj[nzcount++]/bs;
        if (!mask[tmp]) {masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    browlengths[i] += nmask;
    /* zero out the mask elements we set */
    for ( j=0; j<nmask; j++ ) mask[masked[j]] = 0;
  }

  /* create our matrix */
  ierr = MatCreateSeqBAIJ(comm,bs,M+extra_rows,N+extra_rows,0,browlengths,A);
         CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqBAIJ *) B->data;

  /* set matrix "i" values */
  a->i[0] = 0;
  for ( i=1; i<= mbs; i++ ) {
    a->i[i]      = a->i[i-1] + browlengths[i-1];
    a->ilen[i-1] = browlengths[i-1];
  }
  a->nz = 0;
  for ( i=0; i<mbs; i++ ) a->nz += browlengths[i];

  /* read in nonzero values */
  aa = (Scalar *) PetscMalloc((nz+extra_rows)*sizeof(Scalar));CHKPTRQ(aa);
  ierr = PetscBinaryRead(fd,aa,nz,BINARY_SCALAR); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) aa[nz+i] = 1.0;

  /* set "a" and "j" values into matrix */
  nzcount = 0; jcount = 0;
  for ( i=0; i<mbs; i++ ) {
    nzcountb = nzcount;
    nmask    = 0;
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[i*bs+j];
      for ( k=0; k<kmax; k++ ) {
        tmp = jj[nzcount++]/bs;
	if (!mask[tmp]) { masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    /* sort the masked values */
    PetscSortInt(nmask,masked);

    /* set "j" values into matrix */
    maskcount = 1;
    for ( j=0; j<nmask; j++ ) {
      a->j[jcount++]  = masked[j];
      mask[masked[j]] = maskcount++; 
    }
    /* set "a" values into matrix */
    ishift = bs2*a->i[i];
    for ( j=0; j<bs; j++ ) {
      kmax = rowlengths[i*bs+j];
      for ( k=0; k<kmax; k++ ) {
        tmp    = jj[nzcountb]/bs ;
        block  = mask[tmp] - 1;
        point  = jj[nzcountb] - bs*tmp;
        idx    = ishift + bs2*block + j + bs*point;
        a->a[idx] = aa[nzcountb++];
      }
    }
    /* zero out the mask elements we set */
    for ( j=0; j<nmask; j++ ) mask[masked[j]] = 0;
  }
  if (jcount != a->nz) SETERRQ(1,"MatLoad_SeqBAIJ:Error bad binary matrix");

  PetscFree(rowlengths);   
  PetscFree(browlengths);
  PetscFree(aa);
  PetscFree(jj);
  PetscFree(mask);

  B->assembled = PETSC_TRUE;

  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,ASCII_FORMAT_INFO,0);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_info_detailed",&flg);CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,ASCII_FORMAT_INFO_DETAILED,0);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_matlab",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerFileOpenASCII(B->comm,"stdout",&tviewer);CHKERRQ(ierr);
    ierr = ViewerSetFormat(tviewer,ASCII_FORMAT_MATLAB,"M");CHKERRQ(ierr);
    ierr = MatView(B,tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-mat_view_draw",&flg); CHKERRQ(ierr);
  if (flg) {
    Viewer tviewer;
    ierr = ViewerDrawOpenX(B->comm,0,0,0,0,300,300,&tviewer); CHKERRQ(ierr);
    ierr = MatView(B,(Viewer)tviewer); CHKERRQ(ierr);
    ierr = ViewerFlush(tviewer); CHKERRQ(ierr);
    ierr = ViewerDestroy(tviewer); CHKERRQ(ierr);
  }
  return 0;
}



