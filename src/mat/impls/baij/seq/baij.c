/*$Id: baij.c,v 1.245 2001/08/07 03:02:55 balay Exp bsmith $*/

/*
    Defines the basic matrix operations for the BAIJ (compressed row)
  matrix storage format.
*/
#include "src/mat/impls/baij/seq/baij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "petscsys.h"                     /*I "petscmat.h" I*/

/*
    Special version for Fun3d sequential benchmark
*/
#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvaluesblocked4_ MATSETVALUESBLOCKED4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvaluesblocked4_ matsetvaluesblocked4
#endif

#undef __FUNCT__  
#define __FUNCT__ "matsetvaluesblocked4_"
void matsetvaluesblocked4_(Mat *AA,int *mm,int *im,int *nn,int *in,PetscScalar *v)
{
  Mat         A = *AA;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,N,m = *mm,n = *nn;
  int         *ai=a->i,*ailen=a->ilen;
  int         *aj=a->j,stepval;
  PetscScalar *value = v;
  MatScalar   *ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  stepval = (n-1)*4;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    rp   = aj + ai[row]; 
    ap   = aa + 16*ai[row];
    nrow = ailen[row]; 
    low  = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l]; 
      value = v + k*(stepval+4)*4 + l*4;
      low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  16*i;
          for (ii=0; ii<4; ii++,value+=stepval) {
            for (jj=ii; jj<16; jj+=4) {
              bap[jj] += *value++; 
            }
          }
          goto noinsert2;
        }
      } 
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+16*(ii+1),ap+16*(ii),16*sizeof(MatScalar));
      }
      if (N >= i) {
        PetscMemzero(ap+16*i,16*sizeof(MatScalar));
      }
      rp[i] = col; 
      bap   = ap +  16*i;
      for (ii=0; ii<4; ii++,value+=stepval) {
        for (jj=ii; jj<16; jj+=4) {
          bap[jj] = *value++; 
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
} 

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define matsetvalues4_ MATSETVALUES4
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define matsetvalues4_ matsetvalues4
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues4_"
void matsetvalues4_(Mat *AA,int *mm,int *im,int *nn,int *in,PetscScalar *v)
{
  Mat         A = *AA;
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         *rp,k,low,high,t,ii,row,nrow,i,col,l,N,n = *nn,m = *mm;
  int         *ai=a->i,*ailen=a->ilen;
  int         *aj=a->j,brow,bcol;
  int         ridx,cidx;
  MatScalar   *ap,value,*aa=a->a,*bap;
  
  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; brow = row/4;  
    rp   = aj + ai[brow]; 
    ap   = aa + 16*ai[brow];
    nrow = ailen[brow]; 
    low  = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      col = in[l]; bcol = col/4;
      ridx = row % 4; cidx = col % 4;
      value = v[l + k*n]; 
      low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap  = ap +  16*i + 4*cidx + ridx;
          *bap += value;  
          goto noinsert1;
        }
      } 
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        PetscMemcpy(ap+16*(ii+1),ap+16*(ii),16*sizeof(MatScalar));
      }
      if (N>=i) {
        PetscMemzero(ap+16*i,16*sizeof(MatScalar));
      }
      rp[i]                    = bcol; 
      ap[16*i + 4*cidx + ridx] = value; 
      noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
} 

/*  UGLY, ugly, ugly
   When MatScalar == PetscScalar the function MatSetValuesBlocked_SeqBAIJ_MatScalar() does 
   not exist. Otherwise ..._MatScalar() takes matrix dlements in single precision and 
   inserts them into the single precision data structure. The function MatSetValuesBlocked_SeqBAIJ()
   converts the entries into single precision and then calls ..._MatScalar() to put them
   into the single precision data structures.
*/
#if defined(PETSC_USE_MAT_SINGLE)
EXTERN int MatSetValuesBlocked_SeqBAIJ_MatScalar(Mat,int,int*,int,int*,MatScalar*,InsertMode);
#else
#define MatSetValuesBlocked_SeqBAIJ_MatScalar MatSetValuesBlocked_SeqBAIJ
#endif
#if defined(PETSC_HAVE_DSCPACK)
EXTERN int MatUseDSCPACK_MPIBAIJ(Mat);
#endif

#define CHUNKSIZE  10

/*
     Checks for missing diagonals
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal_SeqBAIJ"
int MatMissingDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
  int         *diag,*jj = a->j,i,ierr;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqBAIJ(A);CHKERRQ(ierr);
  diag = a->diag;
  for (i=0; i<a->mbs; i++) {
    if (jj[diag[i]] != i) {
      SETERRQ1(1,"Matrix is missing diagonal number %d",i);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMarkDiagonal_SeqBAIJ"
int MatMarkDiagonal_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
  int         i,j,*diag,m = a->mbs,ierr;

  PetscFunctionBegin;
  if (a->diag) PetscFunctionReturn(0);

  ierr = PetscMalloc((m+1)*sizeof(int),&diag);CHKERRQ(ierr);
  PetscLogObjectMemory(A,(m+1)*sizeof(int));
  for (i=0; i<m; i++) {
    diag[i] = a->i[i+1];
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
        diag[i] = j;
        break;
      }
    }
  }
  a->diag = diag;
  PetscFunctionReturn(0);
}


EXTERN int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_SeqBAIJ"
static int MatGetRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         ierr,n = a->mbs,i;

  PetscFunctionBegin;
  *nn = n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(n,a->i,a->j,0,oshift,ia,ja);CHKERRQ(ierr);
  } else if (oshift == 1) {
    /* temporarily add 1 to i and j indices */
    int nz = a->i[n]; 
    for (i=0; i<nz; i++) a->j[i]++;
    for (i=0; i<n+1; i++) a->i[i]++;
    *ia = a->i; *ja = a->j;
  } else {
    *ia = a->i; *ja = a->j;
  }

  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_SeqBAIJ" 
static int MatRestoreRowIJ_SeqBAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         i,n = a->mbs,ierr;

  PetscFunctionBegin;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  } else if (oshift == 1) {
    int nz = a->i[n]-1; 
    for (i=0; i<nz; i++) a->j[i]--;
    for (i=0; i<n+1; i++) a->i[i]--;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetBlockSize_SeqBAIJ"
int MatGetBlockSize_SeqBAIJ(Mat mat,int *bs)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ*)mat->data;

  PetscFunctionBegin;
  *bs = baij->bs;
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqBAIJ"
int MatDestroy_SeqBAIJ(Mat A)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%d, Cols=%d, NZ=%d",A->m,A->n,a->nz);
#endif
  ierr = PetscFree(a->a);CHKERRQ(ierr);
  if (!a->singlemalloc) {
    ierr = PetscFree(a->i);CHKERRQ(ierr);
    ierr = PetscFree(a->j);CHKERRQ(ierr);
  }
  if (a->row) {
    ierr = ISDestroy(a->row);CHKERRQ(ierr);
  }
  if (a->col) {
    ierr = ISDestroy(a->col);CHKERRQ(ierr);
  }
  if (a->diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
  if (a->ilen) {ierr = PetscFree(a->ilen);CHKERRQ(ierr);}
  if (a->imax) {ierr = PetscFree(a->imax);CHKERRQ(ierr);}
  if (a->solve_work) {ierr = PetscFree(a->solve_work);CHKERRQ(ierr);}
  if (a->mult_work) {ierr = PetscFree(a->mult_work);CHKERRQ(ierr);}
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  if (a->saved_values) {ierr = PetscFree(a->saved_values);CHKERRQ(ierr);}
#if defined(PETSC_USE_MAT_SINGLE)
  if (a->setvaluescopy) {ierr = PetscFree(a->setvaluescopy);CHKERRQ(ierr);}
#endif
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqBAIJ"
int MatSetOption_SeqBAIJ(Mat A,MatOption op)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;

  PetscFunctionBegin;
  switch (op) {
  case MAT_ROW_ORIENTED:
    a->roworiented    = PETSC_TRUE;
    break;
  case MAT_COLUMN_ORIENTED:
    a->roworiented    = PETSC_FALSE;
    break;
  case MAT_COLUMNS_SORTED:
    a->sorted         = PETSC_TRUE;
    break;
  case MAT_COLUMNS_UNSORTED:
    a->sorted         = PETSC_FALSE;
    break;
  case MAT_KEEP_ZEROED_ROWS:
    a->keepzeroedrows = PETSC_TRUE;
    break;
  case MAT_NO_NEW_NONZERO_LOCATIONS:
    a->nonew          = 1;
    break;
  case MAT_NEW_NONZERO_LOCATION_ERR:
    a->nonew          = -1;
    break;
  case MAT_NEW_NONZERO_ALLOCATION_ERR:
    a->nonew          = -2;
    break;
  case MAT_YES_NEW_NONZERO_LOCATIONS:
    a->nonew          = 0;
    break;
  case MAT_ROWS_SORTED:
  case MAT_ROWS_UNSORTED:
  case MAT_YES_NEW_DIAGONALS:
  case MAT_IGNORE_OFF_PROC_ENTRIES:
  case MAT_USE_HASH_TABLE:
    PetscLogInfo(A,"MatSetOption_SeqBAIJ:Option ignored\n");
    break;
  case MAT_NO_NEW_DIAGONALS:
    SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
  case MAT_USE_SINGLE_PRECISION_SOLVES:
    if (a->bs==4) {
      int        ierr;
      a->single_precision_solves = PETSC_TRUE;
      A->ops->solve              = MatSolve_SeqBAIJ_Update;
      A->ops->solvetranspose     = MatSolveTranspose_SeqBAIJ_Update;
      PetscLogInfo(A,"MatSetOption_SeqBAIJ:Option MAT_USE_SINGLE_PRECISION_SOLVES set\n");
    } else {
      PetscLogInfo(A,"MatSetOption_SeqBAIJ:Option MAT_USE_SINGLE_PRECISION_SOLVES ignored\n");
    }
    break;
  default:
    SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqBAIJ"
int MatGetRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data;
  int          itmp,i,j,k,M,*ai,*aj,bs,bn,bp,*idx_i,bs2,ierr;
  MatScalar    *aa,*aa_i;
  PetscScalar  *v_i;

  PetscFunctionBegin;
  bs  = a->bs;
  ai  = a->i;
  aj  = a->j;
  aa  = a->a;
  bs2 = a->bs2;
  
  if (row < 0 || row >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row out of range");
  
  bn  = row/bs;   /* Block number */
  bp  = row % bs; /* Block Position */
  M   = ai[bn+1] - ai[bn];
  *nz = bs*M;
  
  if (v) {
    *v = 0;
    if (*nz) {
      ierr = PetscMalloc((*nz)*sizeof(PetscScalar),v);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        v_i  = *v + i*bs;
        aa_i = aa + bs2*(ai[bn] + i);
        for (j=bp,k=0; j<bs2; j+=bs,k++) {v_i[k] = aa_i[j];}
      }
    }
  }

  if (idx) {
    *idx = 0;
    if (*nz) {
      ierr = PetscMalloc((*nz)*sizeof(int),idx);CHKERRQ(ierr);
      for (i=0; i<M; i++) { /* for each block in the block row */
        idx_i = *idx + i*bs;
        itmp  = bs*aj[ai[bn] + i];
        for (j=0; j<bs; j++) {idx_i[j] = itmp++;}
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqBAIJ"
int MatRestoreRow_SeqBAIJ(Mat A,int row,int *nz,int **idx,PetscScalar **v)
{
  int ierr;

  PetscFunctionBegin;
  if (idx) {if (*idx) {ierr = PetscFree(*idx);CHKERRQ(ierr);}}
  if (v)   {if (*v)   {ierr = PetscFree(*v);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqBAIJ"
int MatTranspose_SeqBAIJ(Mat A,Mat *B)
{ 
  Mat_SeqBAIJ *a=(Mat_SeqBAIJ *)A->data;
  Mat         C;
  int         i,j,k,ierr,*aj=a->j,*ai=a->i,bs=a->bs,mbs=a->mbs,nbs=a->nbs,len,*col;
  int         *rows,*cols,bs2=a->bs2;
  PetscScalar *array;

  PetscFunctionBegin;
  if (!B && mbs!=nbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Square matrix only for in-place");
  ierr = PetscMalloc((1+nbs)*sizeof(int),&col);CHKERRQ(ierr);
  ierr = PetscMemzero(col,(1+nbs)*sizeof(int));CHKERRQ(ierr);

#if defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscMalloc(a->bs2*a->nz*sizeof(PetscScalar),&array);CHKERRQ(ierr);
  for (i=0; i<a->bs2*a->nz; i++) array[i] = (PetscScalar)a->a[i];
#else
  array = a->a;
#endif

  for (i=0; i<ai[mbs]; i++) col[aj[i]] += 1;
  ierr = MatCreateSeqBAIJ(A->comm,bs,A->n,A->m,PETSC_NULL,col,&C);CHKERRQ(ierr);
  ierr = PetscFree(col);CHKERRQ(ierr);
  ierr = PetscMalloc(2*bs*sizeof(int),&rows);CHKERRQ(ierr);
  cols = rows + bs;
  for (i=0; i<mbs; i++) {
    cols[0] = i*bs;
    for (k=1; k<bs; k++) cols[k] = cols[k-1] + 1;
    len = ai[i+1] - ai[i];
    for (j=0; j<len; j++) {
      rows[0] = (*aj++)*bs;
      for (k=1; k<bs; k++) rows[k] = rows[k-1] + 1;
      ierr = MatSetValues(C,bs,rows,bs,cols,array,INSERT_VALUES);CHKERRQ(ierr);
      array += bs2;
    }
  }
  ierr = PetscFree(rows);CHKERRQ(ierr);
#if defined(PETSC_USE_MAT_SINGLE)
  ierr = PetscFree(array);
#endif
  
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  
  if (B) {
    *B = C;
  } else {
    ierr = MatHeaderCopy(A,C);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Binary"
static int MatView_SeqBAIJ_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         i,fd,*col_lens,ierr,bs = a->bs,count,*jj,j,k,l,bs2=a->bs2;
  PetscScalar *aa;
  FILE        *file;

  PetscFunctionBegin;
  ierr        = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr        = PetscMalloc((4+A->m)*sizeof(int),&col_lens);CHKERRQ(ierr);
  col_lens[0] = MAT_FILE_COOKIE;

  col_lens[1] = A->m;
  col_lens[2] = A->n;
  col_lens[3] = a->nz*bs2;

  /* store lengths of each row and write (including header) to file */
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      col_lens[4+count++] = bs*(a->i[i+1] - a->i[i]);
    }
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+A->m,PETSC_INT,1);CHKERRQ(ierr);
  ierr = PetscFree(col_lens);CHKERRQ(ierr);

  /* store column indices (zero start index) */
  ierr  = PetscMalloc((a->nz+1)*bs2*sizeof(int),&jj);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      for (k=a->i[i]; k<a->i[i+1]; k++) {
        for (l=0; l<bs; l++) {
          jj[count++] = bs*a->j[k] + l;
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,jj,bs2*a->nz,PETSC_INT,0);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);

  /* store nonzero values */
  ierr  = PetscMalloc((a->nz+1)*bs2*sizeof(PetscScalar),&aa);CHKERRQ(ierr);
  count = 0;
  for (i=0; i<a->mbs; i++) {
    for (j=0; j<bs; j++) {
      for (k=a->i[i]; k<a->i[i+1]; k++) {
        for (l=0; l<bs; l++) {
          aa[count++] = a->a[bs2*k + l*bs + j];
        }
      }
    }
  }
  ierr = PetscBinaryWrite(fd,aa,bs2*a->nz,PETSC_SCALAR,0);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);

  ierr = PetscViewerBinaryGetInfoPointer(viewer,&file);CHKERRQ(ierr);
  if (file) {
    fprintf(file,"-matload_block_size %d\n",a->bs);
  }
  PetscFunctionReturn(0);
}

extern int MatMPIBAIJFactorInfo_DSCPACK(Mat,PetscViewer);

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_ASCII"
static int MatView_SeqBAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqBAIJ       *a = (Mat_SeqBAIJ*)A->data;
  int               ierr,i,j,bs = a->bs,k,l,bs2=a->bs2;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer,"  block size is %d\n",bs);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    Mat aij;
    ierr = MatConvert(A,MATSEQAIJ,&aij);CHKERRQ(ierr);
    ierr = MatView(aij,viewer);CHKERRQ(ierr);
    ierr = MatDestroy(aij);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
#if defined(PETSC_HAVE_DSCPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
     ierr = MatMPIBAIJFactorInfo_DSCPACK(A,viewer);CHKERRQ(ierr);
#endif
     PetscFunctionReturn(0); 
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g + %gi) ",bs*a->j[k]+l,
                      PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0 && PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g - %gi) ",bs*a->j[k]+l,
                      PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscRealPart(a->a[bs2*k + l*bs + j]) != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            if (a->a[bs2*k + l*bs + j] != 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
            }
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<a->mbs; i++) {
      for (j=0; j<bs; j++) {
        ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i*bs+j);CHKERRQ(ierr);
        for (k=a->i[i]; k<a->i[i+1]; k++) {
          for (l=0; l<bs; l++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) > 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g + %g i) ",bs*a->j[k]+l,
                PetscRealPart(a->a[bs2*k + l*bs + j]),PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else if (PetscImaginaryPart(a->a[bs2*k + l*bs + j]) < 0.0) {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g - %g i) ",bs*a->j[k]+l,
                PetscRealPart(a->a[bs2*k + l*bs + j]),-PetscImaginaryPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            } else {
              ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",bs*a->j[k]+l,PetscRealPart(a->a[bs2*k + l*bs + j]));CHKERRQ(ierr);
            }
#else
            ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",bs*a->j[k]+l,a->a[bs2*k + l*bs + j]);CHKERRQ(ierr);
#endif
          }
        }
        ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      }
    } 
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Draw_Zoom"
static int MatView_SeqBAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat          A = (Mat) Aa;
  Mat_SeqBAIJ  *a=(Mat_SeqBAIJ*)A->data;
  int          row,ierr,i,j,k,l,mbs=a->mbs,color,bs=a->bs,bs2=a->bs2;
  PetscReal    xl,yl,xr,yr,x_l,x_r,y_l,y_r;
  MatScalar    *aa;
  PetscViewer  viewer;

  PetscFunctionBegin; 

  /* still need to add support for contour plot of nonzeros; see MatView_SeqAIJ_Draw_Zoom()*/
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr); 

  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes */
  color = PETSC_DRAW_BLUE;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) >=  0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  color = PETSC_DRAW_CYAN;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) != 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }

  color = PETSC_DRAW_RED;
  for (i=0,row=0; i<mbs; i++,row+=bs) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      y_l = A->m - row - 1.0; y_r = y_l + 1.0;
      x_l = a->j[j]*bs; x_r = x_l + 1.0;
      aa = a->a + j*bs2;
      for (k=0; k<bs; k++) {
        for (l=0; l<bs; l++) {
          if (PetscRealPart(*aa++) <= 0.) continue;
          ierr = PetscDrawRectangle(draw,x_l+k,y_l-l,x_r+k,y_r-l,color,color,color,color);CHKERRQ(ierr);
        }
      }
    } 
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ_Draw"
static int MatView_SeqBAIJ_Draw(Mat A,PetscViewer viewer)
{
  int          ierr;
  PetscReal    xl,yl,xr,yr,w,h;
  PetscDraw    draw;
  PetscTruth   isnull;

  PetscFunctionBegin; 

  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->n; yr = A->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqBAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBAIJ"
int MatView_SeqBAIJ(Mat A,PetscViewer viewer)
{
  int        ierr;
  PetscTruth issocket,isascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (issocket) {
    SETERRQ(PETSC_ERR_SUP,"Socket viewer not supported");
  } else if (isascii){
    ierr = MatView_SeqBAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqBAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqBAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by SeqBAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqBAIJ"
int MatGetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,PetscScalar *v)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int        *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  int        *ai = a->i,*ailen = a->ilen;
  int        brow,bcol,ridx,cidx,bs=a->bs,bs2=a->bs2;
  MatScalar  *ap,*aa = a->a,zero = 0.0;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (row >= A->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    rp   = aj + ai[brow] ; ap = aa + bs2*ai[brow] ;
    nrow = ailen[brow]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column");
      if (in[l] >= A->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
      col  = in[l] ; 
      bcol = col/bs;
      cidx = col%bs; 
      ridx = row%bs;
      high = nrow; 
      low  = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          *v++ = ap[bs2*i+bs*cidx+ridx];
          goto finished;
        }
      } 
      *v++ = zero;
      finished:;
    }
  }
  PetscFunctionReturn(0);
} 

#if defined(PETSC_USE_MAT_SINGLE)
#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_SeqBAIJ"
int MatSetValuesBlocked_SeqBAIJ(Mat mat,int m,int *im,int n,int *in,PetscScalar *v,InsertMode addv)
{
  Mat_SeqBAIJ *b = (Mat_SeqBAIJ*)mat->data;
  int         ierr,i,N = m*n*b->bs2;
  MatScalar   *vsingle;

  PetscFunctionBegin;  
  if (N > b->setvalueslen) {
    if (b->setvaluescopy) {ierr = PetscFree(b->setvaluescopy);CHKERRQ(ierr);}
    ierr = PetscMalloc(N*sizeof(MatScalar),&b->setvaluescopy);CHKERRQ(ierr);
    b->setvalueslen  = N;
  }
  vsingle = b->setvaluescopy;
  for (i=0; i<N; i++) {
    vsingle[i] = v[i];
  }
  ierr = MatSetValuesBlocked_SeqBAIJ_MatScalar(mat,m,im,n,in,vsingle,addv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
#endif


#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesBlocked_SeqBAIJ"
int MatSetValuesBlocked_SeqBAIJ_MatScalar(Mat A,int m,int *im,int n,int *in,MatScalar *v,InsertMode is)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         *rp,k,low,high,t,ii,jj,row,nrow,i,col,l,rmax,N,sorted=a->sorted;
  int         *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  int         *aj=a->j,nonew=a->nonew,bs2=a->bs2,bs=a->bs,stepval,ierr;
  PetscTruth  roworiented=a->roworiented; 
  MatScalar   *value = v,*ap,*aa = a->a,*bap;

  PetscFunctionBegin;
  if (roworiented) { 
    stepval = (n-1)*bs;
  } else {
    stepval = (m-1)*bs;
  }
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
    if (row >= a->mbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
#endif
    rp   = aj + ai[row]; 
    ap   = aa + bs2*ai[row];
    rmax = imax[row]; 
    nrow = ailen[row]; 
    low  = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
      if (in[l] >= a->nbs) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
#endif
      col = in[l]; 
      if (roworiented) { 
        value = v + k*(stepval+bs)*bs + l*bs;
      } else {
        value = v + l*(stepval+bs)*bs + k*bs;
      }
      if (!sorted) low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          bap  = ap +  bs2*i;
          if (roworiented) { 
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=ii; jj<bs2; jj+=bs) {
                  bap[jj] = *value++; 
                }
              }
            }
          } else {
            if (is == ADD_VALUES) {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++ += *value++; 
                }
              }
            } else {
              for (ii=0; ii<bs; ii++,value+=stepval) {
                for (jj=0; jj<bs; jj++) {
                  *bap++  = *value++; 
                }
              }
            }
          }
          goto noinsert2;
        }
      } 
      if (nonew == 1) goto noinsert2;
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int       new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j;
        MatScalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix");

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+bs2*sizeof(MatScalar))+(a->mbs+1)*sizeof(int);
	ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr);
        new_j   = (int*)(new_a + bs2*new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for (ii=0; ii<row+1; ii++) {new_i[ii] = ai[ii];}
        for (ii=row+1; ii<a->mbs+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow)*sizeof(int));CHKERRQ(ierr);
        len  = (new_nz - CHUNKSIZE - ai[row] - nrow);
        ierr = PetscMemcpy(new_j+ai[row]+nrow+CHUNKSIZE,aj+ai[row]+nrow,len*sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a,aa,(ai[row]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr);
        ierr = PetscMemzero(new_a+bs2*(ai[row]+nrow),bs2*CHUNKSIZE*sizeof(MatScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a+bs2*(ai[row]+nrow+CHUNKSIZE),aa+bs2*(ai[row]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);
        /* free up old matrix storage */
        ierr = PetscFree(a->a);CHKERRQ(ierr);
        if (!a->singlemalloc) {
          ierr = PetscFree(a->i);CHKERRQ(ierr);
          ierr = PetscFree(a->j);CHKERRQ(ierr);
        }
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = PETSC_TRUE;

        rp   = aj + ai[row]; ap = aa + bs2*ai[row];
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(MatScalar)));
        a->maxnz += bs2*CHUNKSIZE;
        a->reallocs++;
        a->nz++;
      }
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ierr = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      if (N >= i) {
        ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      rp[i] = col; 
      bap   = ap +  bs2*i;
      if (roworiented) { 
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=ii; jj<bs2; jj+=bs) {
            bap[jj] = *value++; 
          }
        }
      } else {
        for (ii=0; ii<bs; ii++,value+=stepval) {
          for (jj=0; jj<bs; jj++) {
            *bap++  = *value++; 
          }
        }
      }
      noinsert2:;
      low = i;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqBAIJ"
int MatAssemblyEnd_SeqBAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int        fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  int        m = A->m,*ip,N,*ailen = a->ilen;
  int        mbs = a->mbs,bs2 = a->bs2,rmax = 0,ierr;
  MatScalar  *aa = a->a,*ap;
#if defined(PETSC_HAVE_DSCPACK)
  PetscTruth   flag;
#endif

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0];
  for (i=1; i<mbs; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i]; ap = aa + bs2*ai[i];
      N = ailen[i];
      for (j=0; j<N; j++) {
        ip[j-fshift] = ip[j];
        ierr = PetscMemcpy(ap+(j-fshift)*bs2,ap+j*bs2,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (mbs) {
    fshift += imax[mbs-1] - ailen[mbs-1];
    ai[mbs] = ai[mbs-1] + ailen[mbs-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<mbs; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[mbs]; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    ierr = PetscFree(a->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(A,-(mbs+1)*sizeof(int));
    a->diag = 0;
  } 
  PetscLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Matrix size: %d X %d, block size %d; storage space: %d unneeded, %d used\n",m,A->n,a->bs,fshift*bs2,a->nz*bs2);
  PetscLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Number of mallocs during MatSetValues is %d\n",a->reallocs);
  PetscLogInfo(A,"MatAssemblyEnd_SeqBAIJ:Most nonzeros blocks in any row is %d\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (PetscReal)fshift*bs2;

#if defined(PETSC_HAVE_DSCPACK)
  ierr = PetscOptionsHasName(A->prefix,"-mat_baij_dscpack",&flag);CHKERRQ(ierr);
  if (flag) { ierr = MatUseDSCPACK_MPIBAIJ(A);CHKERRQ(ierr); }
#endif

  PetscFunctionReturn(0);
}



/* 
   This function returns an array of flags which indicate the locations of contiguous
   blocks that should be zeroed. for eg: if bs = 3  and is = [0,1,2,3,5,6,7,8,9]
   then the resulting sizes = [3,1,1,3,1] correspondig to sets [(0,1,2),(3),(5),(6,7,8),(9)]
   Assume: sizes should be long enough to hold all the values.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqBAIJ_Check_Blocks"
static int MatZeroRows_SeqBAIJ_Check_Blocks(int idx[],int n,int bs,int sizes[], int *bs_max)
{
  int        i,j,k,row;
  PetscTruth flg;

  PetscFunctionBegin;
  for (i=0,j=0; i<n; j++) {
    row = idx[i];
    if (row%bs!=0) { /* Not the begining of a block */
      sizes[j] = 1;
      i++; 
    } else if (i+bs > n) { /* complete block doesn't exist (at idx end) */
      sizes[j] = 1;         /* Also makes sure atleast 'bs' values exist for next else */
      i++; 
    } else { /* Begining of the block, so check if the complete block exists */
      flg = PETSC_TRUE;
      for (k=1; k<bs; k++) {
        if (row+k != idx[i+k]) { /* break in the block */
          flg = PETSC_FALSE;
          break;
        }
      }
      if (flg == PETSC_TRUE) { /* No break in the bs */
        sizes[j] = bs;
        i+= bs;
      } else {
        sizes[j] = 1;
        i++;
      }
    }
  }
  *bs_max = j;
  PetscFunctionReturn(0);
}
  
#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqBAIJ"
int MatZeroRows_SeqBAIJ(Mat A,IS is,PetscScalar *diag)
{
  Mat_SeqBAIJ *baij=(Mat_SeqBAIJ*)A->data;
  int         ierr,i,j,k,count,is_n,*is_idx,*rows;
  int         bs=baij->bs,bs2=baij->bs2,*sizes,row,bs_max;
  PetscScalar zero = 0.0;
  MatScalar   *aa;

  PetscFunctionBegin;
  /* Make a copy of the IS and  sort it */
  ierr = ISGetLocalSize(is,&is_n);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&is_idx);CHKERRQ(ierr);

  /* allocate memory for rows,sizes */
  ierr  = PetscMalloc((3*is_n+1)*sizeof(int),&rows);CHKERRQ(ierr);
  sizes = rows + is_n;

  /* copy IS values to rows, and sort them */
  for (i=0; i<is_n; i++) { rows[i] = is_idx[i]; }
  ierr = PetscSortInt(is_n,rows);CHKERRQ(ierr);
  if (baij->keepzeroedrows) {
    for (i=0; i<is_n; i++) { sizes[i] = 1; }
    bs_max = is_n;
  } else {
    ierr = MatZeroRows_SeqBAIJ_Check_Blocks(rows,is_n,bs,sizes,&bs_max);CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(is,&is_idx);CHKERRQ(ierr);

  for (i=0,j=0; i<bs_max; j+=sizes[i],i++) {
    row   = rows[j];
    if (row < 0 || row > A->m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"row %d out of range",row);
    count = (baij->i[row/bs +1] - baij->i[row/bs])*bs;
    aa    = baij->a + baij->i[row/bs]*bs2 + (row%bs);
    if (sizes[i] == bs && !baij->keepzeroedrows) {
      if (diag) {
        if (baij->ilen[row/bs] > 0) {
          baij->ilen[row/bs]       = 1;
          baij->j[baij->i[row/bs]] = row/bs;
          ierr = PetscMemzero(aa,count*bs*sizeof(MatScalar));CHKERRQ(ierr);
        } 
        /* Now insert all the diagonal values for this bs */
        for (k=0; k<bs; k++) {
          ierr = (*A->ops->setvalues)(A,1,rows+j+k,1,rows+j+k,diag,INSERT_VALUES);CHKERRQ(ierr);
        } 
      } else { /* (!diag) */
        baij->ilen[row/bs] = 0;
      } /* end (!diag) */
    } else { /* (sizes[i] != bs) */
#if defined (PETSC_USE_DEBUG)
      if (sizes[i] != 1) SETERRQ(1,"Internal Error. Value should be 1");
#endif
      for (k=0; k<count; k++) { 
        aa[0] =  zero; 
        aa    += bs;
      }
      if (diag) {
        ierr = (*A->ops->setvalues)(A,1,rows+j,1,rows+j,diag,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
  }

  ierr = PetscFree(rows);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqBAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqBAIJ"
int MatSetValues_SeqBAIJ(Mat A,int m,int *im,int n,int *in,PetscScalar *v,InsertMode is)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data;
  int         *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted=a->sorted;
  int         *imax=a->imax,*ai=a->i,*ailen=a->ilen;
  int         *aj=a->j,nonew=a->nonew,bs=a->bs,brow,bcol;
  int         ridx,cidx,bs2=a->bs2,ierr;
  PetscTruth  roworiented=a->roworiented;
  MatScalar   *ap,value,*aa=a->a,*bap;

  PetscFunctionBegin;
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; brow = row/bs;  
    if (row < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
    if (row >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %d max %d",row,A->m);
#endif
    rp   = aj + ai[brow]; 
    ap   = aa + bs2*ai[brow];
    rmax = imax[brow]; 
    nrow = ailen[brow]; 
    low  = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
      if (in[l] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %d max %d",in[l],A->n);
#endif
      col = in[l]; bcol = col/bs;
      ridx = row % bs; cidx = col % bs;
      if (roworiented) {
        value = v[l + k*n]; 
      } else {
        value = v[k + l*m];
      }
      if (!sorted) low = 0; high = nrow;
      while (high-low > 7) {
        t = (low+high)/2;
        if (rp[t] > bcol) high = t;
        else              low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > bcol) break;
        if (rp[i] == bcol) {
          bap  = ap +  bs2*i + bs*cidx + ridx;
          if (is == ADD_VALUES) *bap += value;  
          else                  *bap  = value; 
          goto noinsert1;
        }
      } 
      if (nonew == 1) goto noinsert1;
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int       new_nz = ai[a->mbs] + CHUNKSIZE,len,*new_i,*new_j;
        MatScalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero in the matrix");

        /* Malloc new storage space */
        len     = new_nz*(sizeof(int)+bs2*sizeof(MatScalar))+(a->mbs+1)*sizeof(int);
	ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr);
        new_j   = (int*)(new_a + bs2*new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for (ii=0; ii<brow+1; ii++) {new_i[ii] = ai[ii];}
        for (ii=brow+1; ii<a->mbs+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        ierr = PetscMemcpy(new_j,aj,(ai[brow]+nrow)*sizeof(int));CHKERRQ(ierr);
        len  = (new_nz - CHUNKSIZE - ai[brow] - nrow);
        ierr = PetscMemcpy(new_j+ai[brow]+nrow+CHUNKSIZE,aj+ai[brow]+nrow,len*sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a,aa,(ai[brow]+nrow)*bs2*sizeof(MatScalar));CHKERRQ(ierr);
        ierr = PetscMemzero(new_a+bs2*(ai[brow]+nrow),bs2*CHUNKSIZE*sizeof(MatScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a+bs2*(ai[brow]+nrow+CHUNKSIZE),aa+bs2*(ai[brow]+nrow),bs2*len*sizeof(MatScalar));CHKERRQ(ierr);
        /* free up old matrix storage */
        ierr = PetscFree(a->a);CHKERRQ(ierr);
        if (!a->singlemalloc) {
          ierr = PetscFree(a->i);CHKERRQ(ierr);
          ierr = PetscFree(a->j);CHKERRQ(ierr);
        }
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = PETSC_TRUE;

        rp   = aj + ai[brow]; ap = aa + bs2*ai[brow];
        rmax = imax[brow] = imax[brow] + CHUNKSIZE;
        PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + bs2*sizeof(MatScalar)));
        a->maxnz += bs2*CHUNKSIZE;
        a->reallocs++;
        a->nz++;
      }
      N = nrow++ - 1; 
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ierr     = PetscMemcpy(ap+bs2*(ii+1),ap+bs2*(ii),bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      if (N>=i) {
        ierr = PetscMemzero(ap+bs2*i,bs2*sizeof(MatScalar));CHKERRQ(ierr);
      }
      rp[i]                      = bcol; 
      ap[bs2*i + bs*cidx + ridx] = value; 
      noinsert1:;
      low = i;
    }
    ailen[brow] = nrow;
  }
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor_SeqBAIJ"
int MatILUFactor_SeqBAIJ(Mat inA,IS row,IS col,MatILUInfo *info)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)inA->data;
  Mat         outA;
  int         ierr;
  PetscTruth  row_identity,col_identity;

  PetscFunctionBegin;
  if (info && info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only levels = 0 supported for in-place ILU");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(1,"Row and column permutations must be identity for in-place ILU");
  }

  outA          = inA; 
  inA->factor   = FACTOR_LU;

  if (!a->diag) {
    ierr = MatMarkDiagonal_SeqBAIJ(inA);CHKERRQ(ierr);
  }

  a->row        = row;
  a->col        = col;
  ierr          = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr          = PetscObjectReference((PetscObject)col);CHKERRQ(ierr);
  
  /* Create the invert permutation so that it can be used in MatLUFactorNumeric() */
  ierr = ISInvertPermutation(col,PETSC_DECIDE,&a->icol);CHKERRQ(ierr);
  PetscLogObjectParent(inA,a->icol);
  
  /*
      Blocksize 2, 3, 4, 5, 6 and 7 have a special faster factorization/solver 
      for ILU(0) factorization with natural ordering
  */
  if (a->bs < 8) {
    ierr = MatSeqBAIJ_UpdateFactorNumeric_NaturalOrdering(inA);CHKERRQ(ierr);
  } else {
    if (!a->solve_work) {
      ierr = PetscMalloc((inA->m+a->bs)*sizeof(PetscScalar),&a->solve_work);CHKERRQ(ierr);
      PetscLogObjectMemory(inA,(inA->m+a->bs)*sizeof(PetscScalar));
    }
  }

  ierr = MatLUFactorNumeric(inA,&outA);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_SeqBAIJ"
int MatPrintHelp_SeqBAIJ(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  int               ierr;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATSEQBAIJ and MATMPIBAIJ matrix formats (the defaults):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_block_size <block_size>\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetColumnIndices_SeqBAIJ"
int MatSeqBAIJSetColumnIndices_SeqBAIJ(Mat mat,int *indices)
{
  Mat_SeqBAIJ *baij = (Mat_SeqBAIJ *)mat->data;
  int         i,nz,nbs;

  PetscFunctionBegin;
  nz  = baij->maxnz/baij->bs2;
  nbs = baij->nbs;
  for (i=0; i<nz; i++) {
    baij->j[i] = indices[i];
  }
  baij->nz = nz;
  for (i=0; i<nbs; i++) {
    baij->ilen[i] = baij->imax[i];
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetColumnIndices"
/*@
    MatSeqBAIJSetColumnIndices - Set the column indices for all the rows
       in the matrix.

  Input Parameters:
+  mat - the SeqBAIJ matrix
-  indices - the column indices

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.

    You MUST have set the correct numbers of nonzeros per row in the call to 
  MatCreateSeqBAIJ().

    MUST be called before any calls to MatSetValues();

@*/ 
int MatSeqBAIJSetColumnIndices(Mat mat,int *indices)
{
  int ierr,(*f)(Mat,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqBAIJSetColumnIndices_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowMax_SeqBAIJ"
int MatGetRowMax_SeqBAIJ(Mat A,Vec v)
{
  Mat_SeqBAIJ  *a = (Mat_SeqBAIJ*)A->data;
  int          ierr,i,j,n,row,bs,*ai,*aj,mbs;
  PetscReal    atmp;
  PetscScalar  *x,zero = 0.0;
  MatScalar    *aa;
  int          ncols,brow,krow,kcol; 

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix");  
  bs   = a->bs;
  aa   = a->a;
  ai   = a->i;
  aj   = a->j;
  mbs = a->mbs;

  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<mbs; i++) {
    ncols = ai[1] - ai[0]; ai++;
    brow  = bs*i;
    for (j=0; j<ncols; j++){
      /* bcol = bs*(*aj); */
      for (kcol=0; kcol<bs; kcol++){
        for (krow=0; krow<bs; krow++){         
          atmp = PetscAbsScalar(*aa); aa++;         
          row = brow + krow;    /* row index */
          /* printf("val[%d,%d]: %g\n",row,bcol+kcol,atmp); */
          if (PetscAbsScalar(x[row]) < atmp) x[row] = atmp;
        }
      }
      aj++;
    }   
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqBAIJ"
int MatSetUpPreallocation_SeqBAIJ(Mat A)
{
  int        ierr;

  PetscFunctionBegin;
  ierr =  MatSeqBAIJSetPreallocation(A,1,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqBAIJ"
int MatGetArray_SeqBAIJ(Mat A,PetscScalar **array)
{
  Mat_SeqBAIJ *a = (Mat_SeqBAIJ*)A->data; 
  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqBAIJ"
int MatRestoreArray_SeqBAIJ(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqBAIJ"
int MatAXPY_SeqBAIJ(PetscScalar *a,Mat X,Mat Y,MatStructure str)
{
  int          ierr,one=1;
  Mat_SeqBAIJ  *x  = (Mat_SeqBAIJ *)X->data,*y = (Mat_SeqBAIJ *)Y->data;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {   
    BLaxpy_(&x->nz,a,x->a,&one,y->a,&one);
  } else {
    ierr = MatAXPY_Basic(a,X,Y,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqBAIJ,
       MatGetRow_SeqBAIJ,
       MatRestoreRow_SeqBAIJ,
       MatMult_SeqBAIJ_N,
       MatMultAdd_SeqBAIJ_N,
       MatMultTranspose_SeqBAIJ,
       MatMultTransposeAdd_SeqBAIJ,
       MatSolve_SeqBAIJ_N,
       0,
       0,
       0,
       MatLUFactor_SeqBAIJ,
       0,
       0,
       MatTranspose_SeqBAIJ,
       MatGetInfo_SeqBAIJ,
       MatEqual_SeqBAIJ,
       MatGetDiagonal_SeqBAIJ,
       MatDiagonalScale_SeqBAIJ,
       MatNorm_SeqBAIJ,
       0,
       MatAssemblyEnd_SeqBAIJ,
       0,
       MatSetOption_SeqBAIJ,
       MatZeroEntries_SeqBAIJ,
       MatZeroRows_SeqBAIJ,
       MatLUFactorSymbolic_SeqBAIJ,
       MatLUFactorNumeric_SeqBAIJ_N,
       0,
       0,
       MatSetUpPreallocation_SeqBAIJ,
       MatILUFactorSymbolic_SeqBAIJ,
       0,
       MatGetArray_SeqBAIJ,
       MatRestoreArray_SeqBAIJ,
       MatDuplicate_SeqBAIJ,
       0,
       0,
       MatILUFactor_SeqBAIJ,
       0,
       MatAXPY_SeqBAIJ,
       MatGetSubMatrices_SeqBAIJ,
       MatIncreaseOverlap_SeqBAIJ,
       MatGetValues_SeqBAIJ,
       0,
       MatPrintHelp_SeqBAIJ,
       MatScale_SeqBAIJ,
       0,
       0,
       0,
       MatGetBlockSize_SeqBAIJ,
       MatGetRowIJ_SeqBAIJ,
       MatRestoreRowIJ_SeqBAIJ,
       0,
       0,
       0,
       0,
       0,
       0,
       MatSetValuesBlocked_SeqBAIJ,
       MatGetSubMatrix_SeqBAIJ,
       MatDestroy_SeqBAIJ,
       MatView_SeqBAIJ,
       MatGetPetscMaps_Petsc,
       0,
       0,
       0,
       0,
       0,
       0,
       MatGetRowMax_SeqBAIJ,
       MatConvert_Basic};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_SeqBAIJ"
int MatStoreValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ *aij = (Mat_SeqBAIJ *)mat->data;
  int         nz = aij->i[mat->m]*aij->bs*aij->bs2;
  int         ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(1,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
  }

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    ierr = PetscMalloc((nz+1)*sizeof(PetscScalar),&aij->saved_values);CHKERRQ(ierr);
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->saved_values,aij->a,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_SeqBAIJ"
int MatRetrieveValues_SeqBAIJ(Mat mat)
{
  Mat_SeqBAIJ *aij = (Mat_SeqBAIJ *)mat->data;
  int         nz = aij->i[mat->m]*aij->bs*aij->bs2,ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(1,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
  }
  if (!aij->saved_values) {
    SETERRQ(1,"Must call MatStoreValues(A);first");
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->a,aij->saved_values,nz*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
extern int MatConvert_SeqBAIJ_SeqAIJ(Mat,MatType,Mat*);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqBAIJ"
int MatCreate_SeqBAIJ(Mat B)
{
  int         ierr,size;
  Mat_SeqBAIJ *b;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"Comm must be of size 1");

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);
  ierr    = PetscNew(Mat_SeqBAIJ,&b);CHKERRQ(ierr);
  B->data = (void*)b;
  ierr    = PetscMemzero(b,sizeof(Mat_SeqBAIJ));CHKERRQ(ierr);
  ierr    = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->reallocs         = 0;
  b->saved_values     = 0;
#if defined(PETSC_USE_MAT_SINGLE)
  b->setvalueslen     = 0;
  b->setvaluescopy    = PETSC_NULL;
#endif
  b->single_precision_solves = PETSC_FALSE;

  ierr = PetscMapCreateMPI(B->comm,B->m,B->m,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->n,&B->cmap);CHKERRQ(ierr);

  b->sorted           = PETSC_FALSE;
  b->roworiented      = PETSC_TRUE;
  b->nonew            = 0;
  b->diag             = 0;
  b->solve_work       = 0;
  b->mult_work        = 0;
  B->spptr            = 0;
  B->info.nz_unneeded = (PetscReal)b->maxnz;
  b->keepzeroedrows   = PETSC_FALSE;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqBAIJ",
                                      MatStoreValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqBAIJ",
                                      MatRetrieveValues_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqBAIJSetColumnIndices_C",
                                     "MatSeqBAIJSetColumnIndices_SeqBAIJ",
                                      MatSeqBAIJSetColumnIndices_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_SeqBAIJ_SeqAIJ_C",
                                     "MatConvert_SeqBAIJ_SeqAIJ",
                                      MatConvert_SeqBAIJ_SeqAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqBAIJ"
int MatDuplicate_SeqBAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat         C;
  Mat_SeqBAIJ *c,*a = (Mat_SeqBAIJ*)A->data;
  int         i,len,mbs = a->mbs,nz = a->nz,bs2 =a->bs2,ierr;

  PetscFunctionBegin;
  if (a->i[mbs] != nz) SETERRQ(PETSC_ERR_PLIB,"Corrupt matrix");

  *B = 0;
  ierr = MatCreate(A->comm,A->m,A->n,A->m,A->n,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQBAIJ);CHKERRQ(ierr);
  c    = (Mat_SeqBAIJ*)C->data;

  c->bs         = a->bs;
  c->bs2        = a->bs2;
  c->mbs        = a->mbs;
  c->nbs        = a->nbs;
  ierr = PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  ierr = PetscMalloc((mbs+1)*sizeof(int),&c->imax);CHKERRQ(ierr);
  ierr = PetscMalloc((mbs+1)*sizeof(int),&c->ilen);CHKERRQ(ierr);
  for (i=0; i<mbs; i++) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = PETSC_TRUE;
  len  = (mbs+1)*sizeof(int) + nz*(bs2*sizeof(MatScalar) + sizeof(int));
  ierr = PetscMalloc(len,&c->a);CHKERRQ(ierr);
  c->j = (int*)(c->a + nz*bs2);
  c->i = c->j + nz;
  ierr = PetscMemcpy(c->i,a->i,(mbs+1)*sizeof(int));CHKERRQ(ierr);
  if (mbs > 0) {
    ierr = PetscMemcpy(c->j,a->j,nz*sizeof(int));CHKERRQ(ierr);
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscMemcpy(c->a,a->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(c->a,bs2*nz*sizeof(MatScalar));CHKERRQ(ierr);
    }
  }

  PetscLogObjectMemory(C,len+2*(mbs+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqBAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;

  if (a->diag) {
    ierr = PetscMalloc((mbs+1)*sizeof(int),&c->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(C,(mbs+1)*sizeof(int));
    for (i=0; i<mbs; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag        = 0;
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  C->spptr              = 0;     /* Dangerous -I'm throwing away a->spptr */
  c->mult_work          = 0;
  C->preallocated       = PETSC_TRUE;
  C->assembled          = PETSC_TRUE;
  *B = C;
  ierr = PetscFListDuplicate(A->qlist,&C->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqBAIJ"
int MatLoad_SeqBAIJ(PetscViewer viewer,MatType type,Mat *A)
{
  Mat_SeqBAIJ  *a;
  Mat          B;
  int          i,nz,ierr,fd,header[4],size,*rowlengths=0,M,N,bs=1;
  int          *mask,mbs,*jj,j,rowcount,nzcount,k,*browlengths,maskcount;
  int          kmax,jcount,block,idx,point,nzcountb,extra_rows;
  int          *masked,nmask,tmp,bs2,ishift;
  PetscScalar  *aa;
  MPI_Comm     comm = ((PetscObject)viewer)->comm;

  PetscFunctionBegin;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,PETSC_NULL);CHKERRQ(ierr);
  bs2  = bs*bs;

  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not Mat object");
  M = header[1]; N = header[2]; nz = header[3];

  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format, cannot load as SeqBAIJ");
  }

  if (M != N) SETERRQ(PETSC_ERR_SUP,"Can only do square matrices");

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  mbs        = M/bs;
  extra_rows = bs - M + bs*(mbs);
  if (extra_rows == bs) extra_rows = 0;
  else                  mbs++;
  if (extra_rows) {
    PetscLogInfo(0,"MatLoad_SeqBAIJ:Padding loaded matrix to match blocksize\n");
  }

  /* read in row lengths */
  ierr = PetscMalloc((M+extra_rows)*sizeof(int),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) rowlengths[M+i] = 1;

  /* read in column indices */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(int),&jj);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,jj,nz,PETSC_INT);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) jj[nz+i] = M+i;

  /* loop over row lengths determining block row lengths */
  ierr     = PetscMalloc(mbs*sizeof(int),&browlengths);CHKERRQ(ierr);
  ierr     = PetscMemzero(browlengths,mbs*sizeof(int));CHKERRQ(ierr);
  ierr     = PetscMalloc(2*mbs*sizeof(int),&mask);CHKERRQ(ierr);
  ierr     = PetscMemzero(mask,mbs*sizeof(int));CHKERRQ(ierr);
  masked   = mask + mbs;
  rowcount = 0; nzcount = 0;
  for (i=0; i<mbs; i++) {
    nmask = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[rowcount];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs;
        if (!mask[tmp]) {masked[nmask++] = tmp; mask[tmp] = 1;}
      }
      rowcount++;
    }
    browlengths[i] += nmask;
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }

  /* create our matrix */
  ierr = MatCreateSeqBAIJ(comm,bs,M+extra_rows,N+extra_rows,0,browlengths,A);CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqBAIJ*)B->data;

  /* set matrix "i" values */
  a->i[0] = 0;
  for (i=1; i<= mbs; i++) {
    a->i[i]      = a->i[i-1] + browlengths[i-1];
    a->ilen[i-1] = browlengths[i-1];
  }
  a->nz         = 0;
  for (i=0; i<mbs; i++) a->nz += browlengths[i];

  /* read in nonzero values */
  ierr = PetscMalloc((nz+extra_rows)*sizeof(PetscScalar),&aa);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,aa,nz,PETSC_SCALAR);CHKERRQ(ierr);
  for (i=0; i<extra_rows; i++) aa[nz+i] = 1.0;

  /* set "a" and "j" values into matrix */
  nzcount = 0; jcount = 0;
  for (i=0; i<mbs; i++) {
    nzcountb = nzcount;
    nmask    = 0;
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp = jj[nzcount++]/bs;
	if (!mask[tmp]) { masked[nmask++] = tmp; mask[tmp] = 1;}
      }
    }
    /* sort the masked values */
    ierr = PetscSortInt(nmask,masked);CHKERRQ(ierr);

    /* set "j" values into matrix */
    maskcount = 1;
    for (j=0; j<nmask; j++) {
      a->j[jcount++]  = masked[j];
      mask[masked[j]] = maskcount++; 
    }
    /* set "a" values into matrix */
    ishift = bs2*a->i[i];
    for (j=0; j<bs; j++) {
      kmax = rowlengths[i*bs+j];
      for (k=0; k<kmax; k++) {
        tmp       = jj[nzcountb]/bs ;
        block     = mask[tmp] - 1;
        point     = jj[nzcountb] - bs*tmp;
        idx       = ishift + bs2*block + j + bs*point;
        a->a[idx] = (MatScalar)aa[nzcountb++];
      }
    }
    /* zero out the mask elements we set */
    for (j=0; j<nmask; j++) mask[masked[j]] = 0;
  }
  if (jcount != a->nz) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Bad binary matrix");

  ierr = PetscFree(rowlengths);CHKERRQ(ierr);
  ierr = PetscFree(browlengths);CHKERRQ(ierr);
  ierr = PetscFree(aa);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);
  ierr = PetscFree(mask);CHKERRQ(ierr);

  B->assembled = PETSC_TRUE;
  
  ierr = MatView_Private(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqBAIJ"
/*@C
   MatCreateSeqBAIJ - Creates a sparse matrix in block AIJ (block
   compressed row) format.  For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  bs - size of block
.  m - number of rows
.  n - number of columns
.  nz - number of nonzero blocks  per block row (same for all rows)
-  nnz - array containing the number of nonzero blocks in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   A nonzero block is any block that as 1 or more nonzeros in it

   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/
int MatCreateSeqBAIJ(MPI_Comm comm,int bs,int m,int n,int nz,int *nnz,Mat *A)
{
  int ierr;
 
  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,m,n,A);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQBAIJ);CHKERRQ(ierr);
  ierr = MatSeqBAIJSetPreallocation(*A,bs,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSeqBAIJSetPreallocation"
/*@C
   MatSeqBAIJSetPreallocation - Sets the block size and expected nonzeros
   per row in the matrix. For good matrix assembly performance the
   user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  A - the matrix
.  bs - size of block
.  nz - number of block nonzeros per block row (same for all rows)
-  nnz - array containing the number of block nonzeros in the various block rows 
         (possibly different for each block row) or PETSC_NULL

   Options Database Keys:
.   -mat_no_unroll - uses code that does not unroll the loops in the 
                     block calculations (much slower)
.    -mat_block_size - size of the blocks to use

   Level: intermediate

   Notes:
   The block AIJ format is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For additional details, see the users manual chapter on
   matrices.

.seealso: MatCreate(), MatCreateSeqAIJ(), MatSetValues(), MatCreateMPIBAIJ()
@*/
int MatSeqBAIJSetPreallocation(Mat B,int bs,int nz,int *nnz)
{
  Mat_SeqBAIJ *b;
  int         i,len,ierr,mbs,nbs,bs2,newbs = bs;
  PetscTruth  flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATSEQBAIJ,&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);

  B->preallocated = PETSC_TRUE;
  ierr = PetscOptionsGetInt(B->prefix,"-mat_block_size",&newbs,PETSC_NULL);CHKERRQ(ierr);
  if (nnz && newbs != bs) {
    SETERRQ(1,"Cannot change blocksize from command line if setting nnz");
  }
  bs   = newbs;

  mbs  = B->m/bs;
  nbs  = B->n/bs;
  bs2  = bs*bs;

  if (mbs*bs!=B->m || nbs*bs!=B->n) {
    SETERRQ3(PETSC_ERR_ARG_SIZ,"Number rows %d, cols %d must be divisible by blocksize %d",B->m,B->n,bs);
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<mbs; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %d value %d",i,nnz[i]);
      if (nnz[i] > nbs) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than block row length: local row %d value %d rowlength %d",i,nnz[i],nbs);
    }
  }

  b       = (Mat_SeqBAIJ*)B->data;
  ierr    = PetscOptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg);CHKERRQ(ierr);
  B->ops->solve               = MatSolve_SeqBAIJ_Update;
  B->ops->solvetranspose      = MatSolveTranspose_SeqBAIJ_Update;
  if (!flg) {
    switch (bs) {
    case 1:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_1;  
      B->ops->mult            = MatMult_SeqBAIJ_1;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_1;
      break;
    case 2:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_2;  
      B->ops->mult            = MatMult_SeqBAIJ_2;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_2;
      break;
    case 3:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_3;  
      B->ops->mult            = MatMult_SeqBAIJ_3;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_3;
      break;
    case 4:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_4;  
      B->ops->mult            = MatMult_SeqBAIJ_4;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_4;
      break;
    case 5:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_5;  
      B->ops->mult            = MatMult_SeqBAIJ_5;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_5;
      break;
    case 6:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_6;  
      B->ops->mult            = MatMult_SeqBAIJ_6;
      B->ops->multadd         = MatMultAdd_SeqBAIJ_6;
      break;
    case 7:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_7;  
      B->ops->mult            = MatMult_SeqBAIJ_7; 
      B->ops->multadd         = MatMultAdd_SeqBAIJ_7;
      break;
    default:
      B->ops->lufactornumeric = MatLUFactorNumeric_SeqBAIJ_N;  
      B->ops->mult            = MatMult_SeqBAIJ_N; 
      B->ops->multadd         = MatMultAdd_SeqBAIJ_N;
      break;
    }
  }
  b->bs      = bs;
  b->mbs     = mbs;
  b->nbs     = nbs;
  ierr = PetscMalloc((mbs+1)*sizeof(int),&b->imax);CHKERRQ(ierr);
  if (!nnz) {
    if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
    else if (nz <= 0)        nz = 1;
    for (i=0; i<mbs; i++) b->imax[i] = nz;
    nz = nz*mbs;
  } else {
    nz = 0;
    for (i=0; i<mbs; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len   = nz*sizeof(int) + nz*bs2*sizeof(MatScalar) + (B->m+1)*sizeof(int);
  ierr  = PetscMalloc(len,&b->a);CHKERRQ(ierr);
  ierr  = PetscMemzero(b->a,nz*bs2*sizeof(MatScalar));CHKERRQ(ierr);
  b->j  = (int*)(b->a + nz*bs2);
  ierr = PetscMemzero(b->j,nz*sizeof(int));CHKERRQ(ierr);
  b->i  = b->j + nz;
  b->singlemalloc = PETSC_TRUE;

  b->i[0] = 0;
  for (i=1; i<mbs+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each block row so far. */
  ierr = PetscMalloc((mbs+1)*sizeof(int),&b->ilen);CHKERRQ(ierr);
  PetscLogObjectMemory(B,len+2*(mbs+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqBAIJ));
  for (i=0; i<mbs; i++) { b->ilen[i] = 0;}

  b->bs               = bs;
  b->bs2              = bs2;
  b->mbs              = mbs;
  b->nz               = 0;
  b->maxnz            = nz*bs2;
  B->info.nz_unneeded = (PetscReal)b->maxnz;
  PetscFunctionReturn(0);
}

