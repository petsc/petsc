/*$Id: aij.c,v 1.385 2001/09/07 20:09:22 bsmith Exp $*/
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "src/inline/dot.h"
#include "petscbt.h"


EXTERN int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_SeqAIJ"
int MatGetRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *m,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr,i,ishift;
 
  PetscFunctionBegin;  
  *m     = A->m;
  if (!ia) PetscFunctionReturn(0);
  ishift = a->indexshift;
  if (symmetric && !A->structurally_symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(A->m,a->i,a->j,ishift,oshift,ia,ja);CHKERRQ(ierr);
  } else if (oshift == 0 && ishift == -1) {
    int nz = a->i[A->m] - 1; 
    /* malloc space and  subtract 1 from i and j indices */
    ierr = PetscMalloc((A->m+1)*sizeof(int),ia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(int),ja);CHKERRQ(ierr);
    for (i=0; i<nz; i++) (*ja)[i] = a->j[i] - 1;
    for (i=0; i<A->m+1; i++) (*ia)[i] = a->i[i] - 1;
  } else if (oshift == 1 && ishift == 0) {
    int nz = a->i[A->m]; 
    /* malloc space and  add 1 to i and j indices */
    ierr = PetscMalloc((A->m+1)*sizeof(int),ia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(int),ja);CHKERRQ(ierr);
    for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
    for (i=0; i<A->m+1; i++) (*ia)[i] = a->i[i] + 1;
  } else {
    *ia = a->i; *ja = a->j;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_SeqAIJ"
int MatRestoreRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ishift = a->indexshift,ierr;
 
  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);
  if ((symmetric && !A->structurally_symmetric) || (oshift == 0 && ishift == -1) || (oshift == 1 && ishift == 0)) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnIJ_SeqAIJ"
int MatGetColumnIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr,i,ishift = a->indexshift,*collengths,*cia,*cja,n = A->n,m = A->m;
  int        nz = a->i[m]+ishift,row,*jj,mr,col;
 
  PetscFunctionBegin;  
  *nn     = A->n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(A->m,a->i,a->j,ishift,oshift,ia,ja);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc((n+1)*sizeof(int),&collengths);CHKERRQ(ierr);
    ierr = PetscMemzero(collengths,n*sizeof(int));CHKERRQ(ierr);
    ierr = PetscMalloc((n+1)*sizeof(int),&cia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(int),&cja);CHKERRQ(ierr);
    jj = a->j;
    for (i=0; i<nz; i++) {
      collengths[jj[i] + ishift]++;
    }
    cia[0] = oshift;
    for (i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    ierr = PetscMemzero(collengths,n*sizeof(int));CHKERRQ(ierr);
    jj   = a->j;
    for (row=0; row<m; row++) {
      mr = a->i[row+1] - a->i[row];
      for (i=0; i<mr; i++) {
        col = *jj++ + ishift;
        cja[cia[col] + collengths[col]++ - oshift] = row + oshift;  
      }
    }
    ierr = PetscFree(collengths);CHKERRQ(ierr);
    *ia = cia; *ja = cja;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreColumnIJ_SeqAIJ"
int MatRestoreColumnIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,int **ja,PetscTruth *done)
{
  int ierr;

  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);

  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
  
  PetscFunctionReturn(0); 
}

#define CHUNKSIZE   15

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqAIJ"
int MatSetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,PetscScalar *v,InsertMode is)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;
  int         *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted = a->sorted;
  int         *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  int         *aj = a->j,nonew = a->nonew,shift = a->indexshift,ierr;
  PetscScalar *ap,value,*aa = a->a;
  PetscTruth  ignorezeroentries = ((a->ignorezeroentries && is == ADD_VALUES) ? PETSC_TRUE:PETSC_FALSE);
  PetscTruth  roworiented = a->roworiented;

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
    if (row >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %d max %d",row,A->m);
#endif
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    rmax = imax[row]; nrow = ailen[row]; 
    low = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
      if (in[l] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %d max %d",in[l],A->n);
#endif
      col = in[l] - shift;
      if (roworiented) {
        value = v[l + k*n]; 
      } else {
        value = v[k + l*m];
      }
      if (value == 0.0 && ignorezeroentries) continue;

      if (!sorted) low = 0; high = nrow;
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          if (is == ADD_VALUES) ap[i] += value;  
          else                  ap[i] = value;
          goto noinsert;
        }
      } 
      if (nonew == 1) goto noinsert;
      else if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at (%d,%d) in the matrix",row,col);
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int         new_nz = ai[A->m] + CHUNKSIZE,len,*new_i,*new_j;
        PetscScalar *new_a;

        if (nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at (%d,%d) in the matrix requiring new malloc()",row,col);

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+sizeof(PetscScalar))+(A->m+1)*sizeof(int);
	ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr);
        new_j   = (int*)(new_a + new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for (ii=0; ii<row+1; ii++) {new_i[ii] = ai[ii];}
        for (ii=row+1; ii<A->m+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr);
        len  = (new_nz - CHUNKSIZE - ai[row] - nrow - shift);
        ierr = PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow,len*sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow,len*sizeof(PetscScalar));CHKERRQ(ierr);
        /* free up old matrix storage */
        ierr = PetscFree(a->a);CHKERRQ(ierr);
        if (!a->singlemalloc) {
          ierr = PetscFree(a->i);CHKERRQ(ierr);
          ierr = PetscFree(a->j);CHKERRQ(ierr);
        }
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = PETSC_TRUE;

        rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(PetscScalar)));
        a->maxnz += CHUNKSIZE;
        a->reallocs++;
      }
      N = nrow++ - 1; a->nz++;
      /* shift up all the later entries in this row */
      for (ii=N; ii>=i; ii--) {
        rp[ii+1] = rp[ii];
        ap[ii+1] = ap[ii];
      }
      rp[i] = col; 
      ap[i] = value; 
      noinsert:;
      low = i + 1;
    }
    ailen[row] = nrow;
  }
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqAIJ"
int MatGetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,PetscScalar *v)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  int          *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  int          *ai = a->i,*ailen = a->ilen,shift = a->indexshift;
  PetscScalar  *ap,*aa = a->a,zero = 0.0;

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k];   
    if (row < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %d",row);
    if (row >= A->m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: %d",row);
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    nrow = ailen[row]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %d",in[l]);
      if (in[l] >= A->n) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: %d",in[l]);
      col = in[l] - shift;
      high = nrow; low = 0; /* assume unsorted */
      while (high-low > 5) {
        t = (low+high)/2;
        if (rp[t] > col) high = t;
        else             low  = t;
      }
      for (i=low; i<high; i++) {
        if (rp[i] > col) break;
        if (rp[i] == col) {
          *v++ = ap[i];
          goto finished;
        }
      } 
      *v++ = zero;
      finished:;
    }
  }
  PetscFunctionReturn(0);
} 


#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_Binary"
int MatView_SeqAIJ_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        i,fd,*col_lens,ierr;

  PetscFunctionBegin;  
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscMalloc((4+A->m)*sizeof(int),&col_lens);CHKERRQ(ierr);
  col_lens[0] = MAT_FILE_COOKIE;
  col_lens[1] = A->m;
  col_lens[2] = A->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for (i=0; i<A->m; i++) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+A->m,PETSC_INT,1);CHKERRQ(ierr);
  ierr = PetscFree(col_lens);CHKERRQ(ierr);

  /* store column indices (zero start index) */
  if (a->indexshift) {
    for (i=0; i<a->nz; i++) a->j[i]--;
  }
  ierr = PetscBinaryWrite(fd,a->j,a->nz,PETSC_INT,0);CHKERRQ(ierr);
  if (a->indexshift) {
    for (i=0; i<a->nz; i++) a->j[i]++;
  }

  /* store nonzero values */
  ierr = PetscBinaryWrite(fd,a->a,a->nz,PETSC_SCALAR,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern int MatSeqAIJFactorInfo_SuperLU(Mat,PetscViewer);
extern int MatMPIAIJFactorInfo_SuperLu(Mat,PetscViewer);
extern int MatFactorInfo_Spooles(Mat,PetscViewer);
extern int MatSeqAIJFactorInfo_UMFPACK(Mat,PetscViewer);

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_ASCII"
int MatView_SeqAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  int               ierr,i,j,m = A->m,shift = a->indexshift;
  char              *name;
  PetscViewerFormat format;

  PetscFunctionBegin;  
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_LONG || format == PETSC_VIEWER_ASCII_INFO) {
    if (a->inode.size) {
      ierr = PetscViewerASCIIPrintf(viewer,"using I-node routines: found %d nodes, limit used is %d\n",a->inode.node_count,a->inode.limit);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"not using I-node routines\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    int nofinalvalue = 0;
    if ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != A->n-!shift)) {
      nofinalvalue = 1;
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %d %d \n",m,A->n);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %d \n",a->nz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%d,3);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%d %d  %18.16e + %18.16ei \n",i+1,a->j[j]+!shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%d %d  %18.16e\n",i+1,a->j[j]+!shift,a->a[j]);CHKERRQ(ierr);
#endif
      }
    }
    if (nofinalvalue) {
      ierr = PetscViewerASCIIPrintf(viewer,"%d %d  %18.16e\n",m,A->n,0.0);CHKERRQ(ierr);
    } 
    ierr = PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
#if defined(PETSC_HAVE_SUPERLU) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
     ierr = MatSeqAIJFactorInfo_SuperLU(A,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SUPERLUDIST) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
     ierr = MatMPIAIJFactorInfo_SuperLu(A,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_SPOOLES) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
     ierr = MatFactorInfo_Spooles(A,viewer);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_UMFPACK) && !defined(PETSC_USE_SINGLE) && !defined(PETSC_USE_COMPLEX)
     ierr = MatSeqAIJFactorInfo_UMFPACK(A,viewer);CHKERRQ(ierr);
#endif
     PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g + %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g - %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        if (a->a[j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);}
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    int nzd=0,fshift=1,*sptr;
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = PetscMalloc((m+1)*sizeof(int),&sptr);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      sptr[i] = nzd+1;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) != 0.0 || PetscRealPart(a->a[j]) != 0.0) nzd++;
#else
          if (a->a[j] != 0.0) nzd++;
#endif
        }
      }
    }
    sptr[m] = nzd+1;
    ierr = PetscViewerASCIIPrintf(viewer," %d %d\n\n",m,nzd);CHKERRQ(ierr);
    for (i=0; i<m+1; i+=6) {
      if (i+4<m) {ierr = PetscViewerASCIIPrintf(viewer," %d %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4],sptr[i+5]);CHKERRQ(ierr);}
      else if (i+3<m) {ierr = PetscViewerASCIIPrintf(viewer," %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4]);CHKERRQ(ierr);}
      else if (i+2<m) {ierr = PetscViewerASCIIPrintf(viewer," %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3]);CHKERRQ(ierr);}
      else if (i+1<m) {ierr = PetscViewerASCIIPrintf(viewer," %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2]);CHKERRQ(ierr);}
      else if (i<m)   {ierr = PetscViewerASCIIPrintf(viewer," %d %d\n",sptr[i],sptr[i+1]);CHKERRQ(ierr);}
      else            {ierr = PetscViewerASCIIPrintf(viewer," %d\n",sptr[i]);CHKERRQ(ierr);}
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscFree(sptr);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {ierr = PetscViewerASCIIPrintf(viewer," %d ",a->j[j]+fshift);CHKERRQ(ierr);}
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) != 0.0 || PetscRealPart(a->a[j]) != 0.0) {
            ierr = PetscViewerASCIIPrintf(viewer," %18.16e %18.16e ",PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
          }
#else
          if (a->a[j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," %18.16e ",a->a[j]);CHKERRQ(ierr);}
#endif
        }
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_DENSE) {
    int         cnt = 0,jcnt;
    PetscScalar value;

    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      jcnt = 0;
      for (j=0; j<A->n; j++) {
        if (jcnt < a->i[i+1]-a->i[i] && j == a->j[cnt]) {
          value = a->a[cnt++];
          jcnt++;
        } else {
          value = 0.0;
        }
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer," %7.5e+%7.5e i ",PetscRealPart(value),PetscImaginaryPart(value));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer," %7.5e ",value);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g + %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g - %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%d, %g) ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_Draw_Zoom"
int MatView_SeqAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A = (Mat) Aa;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  int               ierr,i,j,m = A->m,shift = a->indexshift,color;
  PetscReal         xl,yl,xr,yr,x_l,x_r,y_l,y_r,maxv = 0.0;
  PetscViewer       viewer;
  PetscViewerFormat format;

  PetscFunctionBegin; 
  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr); 
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);

  ierr = PetscDrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */

  if (format != PETSC_VIEWER_DRAW_CONTOUR) {
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = PETSC_DRAW_BLUE;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(a->a[j]) >=  0.) continue;
#else
        if (a->a[j] >=  0.) continue;
#endif
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
    color = PETSC_DRAW_CYAN;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        if (a->a[j] !=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
    color = PETSC_DRAW_RED;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(a->a[j]) <=  0.) continue;
#else
        if (a->a[j] <=  0.) continue;
#endif
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    int    nz = a->nz,count;
    PetscDraw   popup;
    PetscReal scale;

    for (i=0; i<nz; i++) {
      if (PetscAbsScalar(a->a[i]) > maxv) maxv = PetscAbsScalar(a->a[i]);
    }
    scale = (245.0 - PETSC_DRAW_BASIC_COLORS)/maxv; 
    ierr  = PetscDrawGetPopup(draw,&popup);CHKERRQ(ierr);
    if (popup) {ierr  = PetscDrawScalePopup(popup,0.0,maxv);CHKERRQ(ierr);}
    count = 0;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        color = PETSC_DRAW_BASIC_COLORS + (int)(scale*PetscAbsScalar(a->a[count]));
        ierr  = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
        count++;
      } 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_Draw"
int MatView_SeqAIJ_Draw(Mat A,PetscViewer viewer)
{
  int        ierr;
  PetscDraw  draw;
  PetscReal  xr,yr,xl,yl,h,w;
  PetscTruth isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = A->n; yr = A->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = PetscDrawZoom(draw,MatView_SeqAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ"
int MatView_SeqAIJ(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;
  int         ierr;
  PetscTruth  issocket,isascii,isbinary,isdraw;

  PetscFunctionBegin;  
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (issocket) {
    if (a->indexshift) {
      SETERRQ(1,"Can only socket send sparse matrix with 0 based indexing");
    }
    ierr = PetscViewerSocketPutSparse_Private(viewer,A->m,A->n,a->nz,a->a,a->i,a->j);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = MatView_SeqAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by SeqAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

EXTERN int Mat_AIJ_CheckInode(Mat,PetscTruth);
#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ"
int MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  int          fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax,ierr;
  int          m = A->m,*ip,N,*ailen = a->ilen,shift = a->indexshift,rmax = 0;
  PetscScalar  *aa = a->a,*ap;
#if defined(PETSC_HAVE_SUPERLUDIST) || defined(PETSC_HAVE_SPOOLES) || defined(PETSC_HAVE_UMFPACK)
  PetscTruth   flag;
#endif

  PetscFunctionBegin;  
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i] + shift; 
      ap = aa + ai[i] + shift;
      N  = ailen[i];
      for (j=0; j<N; j++) {
        ip[j-fshift] = ip[j];
        ap[j-fshift] = ap[j]; 
      }
    } 
    ai[i] = ai[i-1] + ailen[i-1];
  }
  if (m) {
    fshift += imax[m-1] - ailen[m-1];
    ai[m]  = ai[m-1] + ailen[m-1];
  }
  /* reset ilen and imax for each row */
  for (i=0; i<m; i++) {
    ailen[i] = imax[i] = ai[i+1] - ai[i];
  }
  a->nz = ai[m] + shift; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    ierr = PetscFree(a->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Matrix size: %d X %d; storage space: %d unneeded,%d used\n",m,A->n,fshift,a->nz);
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Number of mallocs during MatSetValues() is %d\n",a->reallocs);
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Most nonzeros in any row is %d\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift;
  a->rmax              = rmax;

  /* check out for identical nodes. If found, use inode functions */
  ierr = Mat_AIJ_CheckInode(A,(PetscTruth)(!fshift));CHKERRQ(ierr);

#if defined(PETSC_HAVE_SUPERLUDIST) 
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_superlu_dist",&flag);CHKERRQ(ierr);
  if (flag) { ierr = MatUseSuperLU_DIST_MPIAIJ(A);CHKERRQ(ierr); }
#endif 

#if defined(PETSC_HAVE_SPOOLES) 
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_spooles",&flag);CHKERRQ(ierr);
  if (flag) { ierr = MatUseSpooles_SeqAIJ(A);CHKERRQ(ierr); }
#endif 

#if defined(PETSC_HAVE_UMFPACK) 
  ierr = PetscOptionsHasName(A->prefix,"-mat_aij_umfpack",&flag);CHKERRQ(ierr);
  if (flag) { ierr = MatUseUMFPACK_SeqAIJ(A);CHKERRQ(ierr); }
#endif 

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqAIJ"
int MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        ierr;

  PetscFunctionBegin;  
  ierr = PetscMemzero(a->a,(a->i[A->m]+a->indexshift)*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ"
int MatDestroy_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;  
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%d, Cols=%d, NZ=%d",A->m,A->n,a->nz);
#endif
  if (a->freedata) {
    ierr = PetscFree(a->a);CHKERRQ(ierr);
    if (!a->singlemalloc) {
      ierr = PetscFree(a->i);CHKERRQ(ierr);
      ierr = PetscFree(a->j);CHKERRQ(ierr);
    }
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
  if (a->idiag) {ierr = PetscFree(a->idiag);CHKERRQ(ierr);}
  if (a->solve_work) {ierr = PetscFree(a->solve_work);CHKERRQ(ierr);}
  if (a->inode.size) {ierr = PetscFree(a->inode.size);CHKERRQ(ierr);}
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  if (a->saved_values) {ierr = PetscFree(a->saved_values);CHKERRQ(ierr);}
  if (a->coloring) {ierr = ISColoringDestroy(a->coloring);CHKERRQ(ierr);}
  ierr = PetscFree(a);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCompress_SeqAIJ"
int MatCompress_SeqAIJ(Mat A)
{
  PetscFunctionBegin;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqAIJ"
int MatSetOption_SeqAIJ(Mat A,MatOption op)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;  
  switch (op) {
    case MAT_ROW_ORIENTED:
      a->roworiented       = PETSC_TRUE;
      break;
    case MAT_KEEP_ZEROED_ROWS:
      a->keepzeroedrows    = PETSC_TRUE;
      break;
    case MAT_COLUMN_ORIENTED:
      a->roworiented       = PETSC_FALSE;
      break;
    case MAT_COLUMNS_SORTED:
      a->sorted            = PETSC_TRUE;
      break;
    case MAT_COLUMNS_UNSORTED:
      a->sorted            = PETSC_FALSE;
      break;
    case MAT_NO_NEW_NONZERO_LOCATIONS:
      a->nonew             = 1;
      break;
    case MAT_NEW_NONZERO_LOCATION_ERR:     
      a->nonew             = -1;
      break;
    case MAT_NEW_NONZERO_ALLOCATION_ERR:
      a->nonew             = -2;
      break;
    case MAT_YES_NEW_NONZERO_LOCATIONS:
      a->nonew             = 0;
      break;
    case MAT_IGNORE_ZERO_ENTRIES:
      a->ignorezeroentries = PETSC_TRUE;
      break;
    case MAT_USE_INODES:
      a->inode.use         = PETSC_TRUE;
      break;
    case MAT_DO_NOT_USE_INODES:
      a->inode.use         = PETSC_FALSE;
      break;
    case MAT_ROWS_SORTED:
    case MAT_ROWS_UNSORTED:
    case MAT_YES_NEW_DIAGONALS:
    case MAT_IGNORE_OFF_PROC_ENTRIES:
    case MAT_USE_HASH_TABLE:
    case MAT_USE_SINGLE_PRECISION_SOLVES:
      PetscLogInfo(A,"MatSetOption_SeqAIJ:Option ignored\n");
      break;
    case MAT_NO_NEW_DIAGONALS:
      SETERRQ(PETSC_ERR_SUP,"MAT_NO_NEW_DIAGONALS");
    case MAT_INODE_LIMIT_1:
      a->inode.limit  = 1;
      break;
    case MAT_INODE_LIMIT_2:
      a->inode.limit  = 2;
      break;
    case MAT_INODE_LIMIT_3:
      a->inode.limit  = 3;
      break;
    case MAT_INODE_LIMIT_4:
      a->inode.limit  = 4;
      break;
    case MAT_INODE_LIMIT_5:
      a->inode.limit  = 5;
      break;
    default:
      SETERRQ(PETSC_ERR_SUP,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetDiagonal_SeqAIJ"
int MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  int          i,j,n,shift = a->indexshift,ierr;
  PetscScalar  *x,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<A->m; i++) {
    for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
      if (a->j[j]+shift == i) {
        x[i] = a->a[j];
        break;
      }
    }
  }
  ierr = VecRestoreArray(v,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqAIJ"
int MatMultTransposeAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *x,*y;
  int          ierr,m = A->m,shift = a->indexshift;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  PetscScalar  *v,alpha;
  int          n,i,*idx;
#endif

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  fortranmulttransposeaddaij_(&m,x,a->i,a->j+shift,a->a+shift,y);
#else
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
#endif
  PetscLogFlops(2*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqAIJ"
int MatMultTranspose_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  PetscScalar  zero = 0.0;
  int          ierr;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd_SeqAIJ(A,xx,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJ"
int MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *x,*y,*v;
  int          ierr,m = A->m,*idx,shift = a->indexshift,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  int          n,i,jrow,j;
  PetscScalar  sum;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*v)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  x    = x + shift;    /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  fortranmultaij_(&m,x,ii,idx+shift,v+shift,y);
#else
  v    += shift; /* shift for Fortran start by 1 indexing */
  idx  += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum  = 0.0;
    for (j=0; j<n; j++) {
      sum += v[jrow]*x[idx[jrow]]; jrow++;
     }
    y[i] = sum;
  }
#endif
  PetscLogFlops(2*a->nz - m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJ"
int MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *x,*y,*z,*v;
  int          ierr,m = A->m,*idx,shift = a->indexshift,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
  int          n,i,jrow,j;
PetscScalar    sum;
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }
  x    = x + shift; /* shift for Fortran start by 1 indexing */
  idx  = a->j;
  v    = a->a;
  ii   = a->i;
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
  fortranmultaddaij_(&m,x,ii,idx+shift,v+shift,y,z);
#else
  v   += shift; /* shift for Fortran start by 1 indexing */
  idx += shift;
  for (i=0; i<m; i++) {
    jrow = ii[i];
    n    = ii[i+1] - jrow;
    sum  = y[i];
    for (j=0; j<n; j++) {
      sum += v[jrow]*x[idx[jrow]]; jrow++;
     }
    z[i] = sum;
  }
#endif
  PetscLogFlops(2*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecRestoreArray(zz,&z);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     Adds diagonal pointers to sparse matrix structure.
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMarkDiagonal_SeqAIJ"
int MatMarkDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        i,j,*diag,m = A->m,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  if (a->diag) PetscFunctionReturn(0);

  ierr = PetscMalloc((m+1)*sizeof(int),&diag);CHKERRQ(ierr);
  PetscLogObjectMemory(A,(m+1)*sizeof(int));
  for (i=0; i<A->m; i++) {
    diag[i] = a->i[i+1];
    for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
      if (a->j[j]+shift == i) {
        diag[i] = j - shift;
        break;
      }
    }
  }
  a->diag = diag;
  PetscFunctionReturn(0);
}

/*
     Checks for missing diagonals
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal_SeqAIJ"
int MatMissingDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        *diag,*jj = a->j,i,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  diag = a->diag;
  for (i=0; i<A->m; i++) {
    if (jj[diag[i]+shift] != i-shift) {
      SETERRQ1(1,"Matrix is missing diagonal number %d",i);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRelax_SeqAIJ"
int MatRelax_SeqAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,int its,int lits,Vec xx)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *x,*b,*bs, d,*xs,sum,*v = a->a,*t=0,scale,*ts,*xb,*idiag=0;
  int          ierr,*idx,*diag,n = A->n,m = A->m,i,shift = a->indexshift;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %d and local its %d both positive",its,lits);

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  if (xx != bb) {
    ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  } else {
    b = x;
  }

  if (!a->diag) {ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);}
  diag = a->diag;
  xs   = x + shift; /* shifted by one for index start of a or a->j*/
  if (flag == SOR_APPLY_UPPER) {
   /* apply (U + D/omega) to the vector */
    bs = b + shift;
    for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
	PetscLogFlops(2*n-1);
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = b[i]*d/omega;
        SPARSEDENSEDOT(sum,bs,v,idx,n); 
        x[i] = sum;
    }
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }

  /* setup workspace for Eisenstat */
  if (flag & SOR_EISENSTAT) {
    if (!a->idiag) {
      ierr     = PetscMalloc(2*m*sizeof(PetscScalar),&a->idiag);CHKERRQ(ierr);
      a->ssor  = a->idiag + m;
      v        = a->a;
      for (i=0; i<m; i++) { a->idiag[i] = 1.0/v[diag[i]];}
    }
    t     = a->ssor;
    idiag = a->idiag;
  }
    /* Let  A = L + U + D; where L is lower trianglar,
    U is upper triangular, E is diagonal; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick. This is for
    the case of SSOR preconditioner, so E is D/omega where omega
    is the relaxation factor.
    */

  if (flag == SOR_APPLY_LOWER) {
    SETERRQ(PETSC_ERR_SUP,"SOR_APPLY_LOWER is not implemented");
  } else if ((flag & SOR_EISENSTAT) && omega == 1.0 && shift == 0 && fshift == 0.0) {
    /* special case for omega = 1.0 saves flops and some integer ops */
    PetscScalar *v2;
 
    v2    = a->a;
    /*  x = (E + U)^{-1} b */
    for (i=m-1; i>=0; i--) {
      n    = a->i[i+1] - diag[i] - 1;
      idx  = a->j + diag[i] + 1;
      v    = a->a + diag[i] + 1;
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = sum*idiag[i];

      /*  t = b - (2*E - D)x */
      t[i] = b[i] - (v2[diag[i]])*x[i];
    }

    /*  t = (E + L)^{-1}t */
    diag = a->diag;
    for (i=0; i<m; i++) {
      n    = diag[i] - a->i[i];
      idx  = a->j + a->i[i];
      v    = a->a + a->i[i];
      sum  = t[i];
      SPARSEDENSEMDOT(sum,t,v,idx,n); 
      t[i]  = sum*idiag[i];

      /*  x = x + t */
      x[i] += t[i];
    }

    PetscLogFlops(3*m-1 + 2*a->nz);
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  } else if (flag & SOR_EISENSTAT) {
    /* Let  A = L + U + D; where L is lower trianglar,
    U is upper triangular, E is diagonal; This routine applies

            (L + E)^{-1} A (U + E)^{-1}

    to a vector efficiently using Eisenstat's trick. This is for
    the case of SSOR preconditioner, so E is D/omega where omega
    is the relaxation factor.
    */
    scale = (2.0/omega) - 1.0;

    /*  x = (E + U)^{-1} b */
    for (i=m-1; i>=0; i--) {
      d    = fshift + a->a[diag[i] + shift];
      n    = a->i[i+1] - diag[i] - 1;
      idx  = a->j + diag[i] + (!shift);
      v    = a->a + diag[i] + (!shift);
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = omega*(sum/d);
    }

    /*  t = b - (2*E - D)x */
    v = a->a;
    for (i=0; i<m; i++) { t[i] = b[i] - scale*(v[*diag++ + shift])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t + shift; /* shifted by one for index start of a or a->j*/
    diag = a->diag;
    for (i=0; i<m; i++) {
      d    = fshift + a->a[diag[i]+shift];
      n    = diag[i] - a->i[i];
      idx  = a->j + a->i[i] + shift;
      v    = a->a + a->i[i] + shift;
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      t[i] = omega*(sum/d);
      /*  x = x + t */
      x[i] += t[i];
    }

    PetscLogFlops(6*m-1 + 2*a->nz);
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijforwardzero_(&m,&omega,x,a->i,a->j,diag,a->a,b);
#else
      for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]+shift];
        n    = diag[i] - a->i[i];
	PetscLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
#endif
      xb = x;
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=0; i<m; i++) {
        x[i] *= a->a[diag[i]+shift];
      }
      PetscLogFlops(m);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijbackwardzero_(&m,&omega,x,a->i,a->j,diag,a->a,xb);
#else
      for (i=m-1; i>=0; i--) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
	PetscLogFlops(2*n-1);
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = xb[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
#endif
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijforward_(&m,&omega,x,a->i,a->j,diag,a->a,b);
#else
      for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]+shift];
        n    = a->i[i+1] - a->i[i]; 
	PetscLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
#endif
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijbackward_(&m,&omega,x,a->i,a->j,diag,a->a,b);
#else
      for (i=m-1; i>=0; i--) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - a->i[i]; 
	PetscLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
#endif
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqAIJ"
int MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  info->rows_global    = (double)A->m;
  info->columns_global = (double)A->n;
  info->rows_local     = (double)A->m;
  info->columns_local  = (double)A->n;
  info->block_size     = 1.0;
  info->nz_allocated   = (double)a->maxnz;
  info->nz_used        = (double)a->nz;
  info->nz_unneeded    = (double)(a->maxnz - a->nz);
  info->assemblies     = (double)A->num_ass;
  info->mallocs        = (double)a->reallocs;
  info->memory         = A->mem;
  if (A->factor) {
    info->fill_ratio_given  = A->info.fill_ratio_given;
    info->fill_ratio_needed = A->info.fill_ratio_needed;
    info->factor_mallocs    = A->info.factor_mallocs;
  } else {
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
  }
  PetscFunctionReturn(0);
}

EXTERN int MatLUFactorSymbolic_SeqAIJ(Mat,IS,IS,MatLUInfo*,Mat*);
EXTERN int MatLUFactorNumeric_SeqAIJ(Mat,Mat*);
EXTERN int MatLUFactor_SeqAIJ(Mat,IS,IS,MatLUInfo*);
EXTERN int MatSolve_SeqAIJ(Mat,Vec,Vec);
EXTERN int MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
EXTERN int MatSolveTranspose_SeqAIJ(Mat,Vec,Vec);
EXTERN int MatSolveTransposeAdd_SeqAIJ(Mat,Vec,Vec,Vec);

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqAIJ"
int MatZeroRows_SeqAIJ(Mat A,IS is,PetscScalar *diag)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int         i,ierr,N,*rows,m = A->m - 1,shift = a->indexshift;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  if (a->keepzeroedrows) {
    for (i=0; i<N; i++) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"row out of range");
      ierr = PetscMemzero(&a->a[a->i[rows[i]]+shift],a->ilen[rows[i]]*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (diag) {
      ierr = MatMissingDiagonal_SeqAIJ(A);CHKERRQ(ierr);
      ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
      for (i=0; i<N; i++) {
        a->a[a->diag[rows[i]]] = *diag;
      }
    }
  } else {
    if (diag) {
      for (i=0; i<N; i++) {
        if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"row out of range");
        if (a->ilen[rows[i]] > 0) { 
          a->ilen[rows[i]]          = 1; 
          a->a[a->i[rows[i]]+shift] = *diag;
          a->j[a->i[rows[i]]+shift] = rows[i]+shift;
        } else { /* in case row was completely empty */
          ierr = MatSetValues_SeqAIJ(A,1,&rows[i],1,&rows[i],diag,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    } else {
      for (i=0; i<N; i++) {
        if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"row out of range");
        a->ilen[rows[i]] = 0; 
      }
    }
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqAIJ"
int MatGetRow_SeqAIJ(Mat A,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        *itmp,i,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  if (row < 0 || row >= A->m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Row %d out of range",row);

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row] + shift;
  if (idx) {
    itmp = a->j + a->i[row] + shift;
    if (*nz && shift) {
      ierr = PetscMalloc((*nz)*sizeof(int),idx);CHKERRQ(ierr);
      for (i=0; i<(*nz); i++) {(*idx)[i] = itmp[i] + shift;}
    } else if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqAIJ"
int MatRestoreRow_SeqAIJ(Mat A,int row,int *nz,int **idx,PetscScalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int ierr;

  PetscFunctionBegin;
  if (idx) {if (*idx && a->indexshift) {ierr = PetscFree(*idx);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqAIJ"
int MatNorm_SeqAIJ(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *v = a->a;
  PetscReal    sum = 0.0;
  int          i,j,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<a->nz; i++) {
#if defined(PETSC_USE_COMPLEX)
      sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *nrm = sqrt(sum);
  } else if (type == NORM_1) {
    PetscReal *tmp;
    int    *jj = a->j;
    ierr = PetscMalloc((A->n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMemzero(tmp,A->n*sizeof(PetscReal));CHKERRQ(ierr);
    *nrm = 0.0;
    for (j=0; j<a->nz; j++) {
        tmp[*jj++ + shift] += PetscAbsScalar(*v);  v++;
    }
    for (j=0; j<A->n; j++) {
      if (tmp[j] > *nrm) *nrm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->m; j++) {
      v = a->a + a->i[j] + shift;
      sum = 0.0;
      for (i=0; i<a->i[j+1]-a->i[j]; i++) {
        sum += PetscAbsScalar(*v); v++;
      }
      if (sum > *nrm) *nrm = sum;
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqAIJ"
int MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  Mat          C;
  int          i,ierr,*aj = a->j,*ai = a->i,m = A->m,len,*col;
  int          shift = a->indexshift;
  PetscScalar  *array = a->a;

  PetscFunctionBegin;
  if (!B && m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  ierr = PetscMalloc((1+A->n)*sizeof(int),&col);CHKERRQ(ierr);
  ierr = PetscMemzero(col,(1+A->n)*sizeof(int));CHKERRQ(ierr);
  if (shift) {
    for (i=0; i<ai[m]-1; i++) aj[i] -= 1;
  }
  for (i=0; i<ai[m]+shift; i++) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,A->n,m,0,col,&C);CHKERRQ(ierr);
  ierr = PetscFree(col);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    len    = ai[i+1]-ai[i];
    ierr   = MatSetValues(C,len,aj,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    array += len; 
    aj    += len;
  }
  if (shift) { 
    for (i=0; i<ai[m]-1; i++) aj[i] += 1;
  }

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
#define __FUNCT__ "MatDiagonalScale_SeqAIJ"
int MatDiagonalScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *l,*r,x,*v;
  int          ierr,i,j,m = A->m,n = A->n,M,nz = a->nz,*jj,shift = a->indexshift;

  PetscFunctionBegin;
  if (ll) {
    /* The local size is used so that VecMPI can be passed to this routine
       by MatDiagonalScale_MPIAIJ */
    ierr = VecGetLocalSize(ll,&m);CHKERRQ(ierr);
    if (m != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Left scaling vector wrong length");
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    v = a->a;
    for (i=0; i<m; i++) {
      x = l[i];
      M = a->i[i+1] - a->i[i];
      for (j=0; j<M; j++) { (*v++) *= x;} 
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    PetscLogFlops(nz);
  }
  if (rr) {
    ierr = VecGetLocalSize(rr,&n);CHKERRQ(ierr);
    if (n != A->n) SETERRQ(PETSC_ERR_ARG_SIZ,"Right scaling vector wrong length");
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr); 
    v = a->a; jj = a->j;
    for (i=0; i<nz; i++) {
      (*v++) *= r[*jj++ + shift]; 
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr); 
    PetscLogFlops(nz);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqAIJ"
int MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,int csize,MatReuse scall,Mat *B)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data,*c;
  int          *smap,i,k,kstart,kend,ierr,oldcols = A->n,*lens;
  int          row,mat_i,*mat_j,tcol,first,step,*mat_ilen,sum,lensi;
  int          *irow,*icol,nrows,ncols,shift = a->indexshift,*ssmap;
  int          *starts,*j_new,*i_new,*aj = a->j,*ai = a->i,ii,*ailen = a->ilen;
  PetscScalar  *a_new,*mat_a;
  Mat          C;
  PetscTruth   stride;

  PetscFunctionBegin;
  ierr = ISSorted(isrow,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"ISrow is not sorted");
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"IScol is not sorted");

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscol,&ncols);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(iscol,&first,&step);CHKERRQ(ierr);
  ierr = ISStride(iscol,&stride);CHKERRQ(ierr);
  if (stride && step == 1) { 
    /* special case of contiguous rows */
    ierr   = PetscMalloc((2*nrows+1)*sizeof(int),&lens);CHKERRQ(ierr);
    starts = lens + nrows;
    /* loop over new rows determining lens and starting points */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + ailen[irow[i]];
      for (k=kstart; k<kend; k++) {
        if (aj[k]+shift >= first) {
          starts[i] = k;
          break;
	}
      }
      sum = 0;
      while (k < kend) {
        if (aj[k++]+shift >= first+ncols) break;
        sum++;
      }
      lens[i] = sum;
    }
    /* create submatrix */
    if (scall == MAT_REUSE_MATRIX) {
      int n_cols,n_rows;
      ierr = MatGetSize(*B,&n_rows,&n_cols);CHKERRQ(ierr);
      if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size");
      ierr = MatZeroEntries(*B);CHKERRQ(ierr);
      C = *B;
    } else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ*)C->data;

    /* loop over rows inserting into submatrix */
    a_new    = c->a;
    j_new    = c->j;
    i_new    = c->i;
    i_new[0] = -shift;
    for (i=0; i<nrows; i++) {
      ii    = starts[i];
      lensi = lens[i];
      for (k=0; k<lensi; k++) {
        *j_new++ = aj[ii+k] - first;
      }
      ierr = PetscMemcpy(a_new,a->a + starts[i],lensi*sizeof(PetscScalar));CHKERRQ(ierr);
      a_new      += lensi;
      i_new[i+1]  = i_new[i] + lensi;
      c->ilen[i]  = lensi;
    }
    ierr = PetscFree(lens);CHKERRQ(ierr);
  } else {
    ierr  = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
    ierr  = PetscMalloc((1+oldcols)*sizeof(int),&smap);CHKERRQ(ierr);
    ssmap = smap + shift;
    ierr  = PetscMalloc((1+nrows)*sizeof(int),&lens);CHKERRQ(ierr);
    ierr  = PetscMemzero(smap,oldcols*sizeof(int));CHKERRQ(ierr);
    for (i=0; i<ncols; i++) smap[icol[i]] = i+1;
    /* determine lens of each row */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]+shift; 
      kend    = kstart + a->ilen[irow[i]];
      lens[i] = 0;
      for (k=kstart; k<kend; k++) {
        if (ssmap[aj[k]]) {
          lens[i]++;
        }
      }
    }
    /* Create and fill new matrix */
    if (scall == MAT_REUSE_MATRIX) {
      PetscTruth equal;

      c = (Mat_SeqAIJ *)((*B)->data);
      if ((*B)->m  != nrows || (*B)->n != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(c->ilen,lens,(*B)->m*sizeof(int),&equal);CHKERRQ(ierr);
      if (!equal) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
      }
      ierr = PetscMemzero(c->ilen,(*B)->m*sizeof(int));CHKERRQ(ierr);
      C = *B;
    } else {  
      ierr = MatCreateSeqAIJ(A->comm,nrows,ncols,0,lens,&C);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ *)(C->data);
    for (i=0; i<nrows; i++) {
      row    = irow[i];
      kstart = ai[row]+shift; 
      kend   = kstart + a->ilen[row];
      mat_i  = c->i[i]+shift;
      mat_j  = c->j + mat_i; 
      mat_a  = c->a + mat_i;
      mat_ilen = c->ilen + i;
      for (k=kstart; k<kend; k++) {
        if ((tcol=ssmap[a->j[k]])) {
          *mat_j++ = tcol - (!shift);
          *mat_a++ = a->a[k];
          (*mat_ilen)++;

        }
      }
    }
    /* Free work space */
    ierr = ISRestoreIndices(iscol,&icol);CHKERRQ(ierr);
    ierr = PetscFree(smap);CHKERRQ(ierr);
    ierr = PetscFree(lens);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = ISRestoreIndices(isrow,&irow);CHKERRQ(ierr);
  *B = C;
  PetscFunctionReturn(0);
}

/*
*/
#undef __FUNCT__  
#define __FUNCT__ "MatILUFactor_SeqAIJ"
int MatILUFactor_SeqAIJ(Mat inA,IS row,IS col,MatILUInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)inA->data;
  int        ierr;
  Mat        outA;
  PetscTruth row_identity,col_identity;

  PetscFunctionBegin;
  if (info && info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only levels=0 supported for in-place ilu");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(1,"Row and column permutations must be identity for in-place ILU");
  }

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;
  ierr = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)col);CHKERRQ(ierr);

  /* Create the inverse permutation so that it can be used in MatLUFactorNumeric() */
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);} /* need to remove old one */
  ierr = ISInvertPermutation(col,PETSC_DECIDE,&a->icol);CHKERRQ(ierr);
  PetscLogObjectParent(inA,a->icol);

  if (!a->solve_work) { /* this matrix may have been factored before */
     ierr = PetscMalloc((inA->m+1)*sizeof(PetscScalar),&a->solve_work);CHKERRQ(ierr);
  }

  if (!a->diag) {
    ierr = MatMarkDiagonal_SeqAIJ(inA);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric_SeqAIJ(inA,&outA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatScale_SeqAIJ"
int MatScale_SeqAIJ(PetscScalar *alpha,Mat inA)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)inA->data;
  int        one = 1;

  PetscFunctionBegin;
  BLscal_(&a->nz,alpha,a->a,&one);
  PetscLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SeqAIJ"
int MatGetSubMatrices_SeqAIJ(Mat A,int n,IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    ierr = PetscMalloc((n+1)*sizeof(Mat),B);CHKERRQ(ierr);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqAIJ(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetBlockSize_SeqAIJ"
int MatGetBlockSize_SeqAIJ(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatIncreaseOverlap_SeqAIJ"
int MatIncreaseOverlap_SeqAIJ(Mat A,int is_max,IS *is,int ov)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        shift,row,i,j,k,l,m,n,*idx,ierr,*nidx,isz,val;
  int        start,end,*ai,*aj;
  PetscBT    table;

  PetscFunctionBegin;
  shift = a->indexshift;
  m     = A->m;
  ai    = a->i;
  aj    = a->j+shift;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"illegal overlap value used");

  ierr = PetscMalloc((m+1)*sizeof(int),&nidx);CHKERRQ(ierr); 
  ierr = PetscBTCreate(m,table);CHKERRQ(ierr);

  for (i=0; i<is_max; i++) {
    /* Initialize the two local arrays */
    isz  = 0;
    ierr = PetscBTMemzero(m,table);CHKERRQ(ierr);
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is[i],&n);CHKERRQ(ierr);
    
    /* Enter these into the temp arrays. I.e., mark table[row], enter row into new index */
    for (j=0; j<n ; ++j){
      if(!PetscBTLookupSet(table,idx[j])) { nidx[isz++] = idx[j];}
    }
    ierr = ISRestoreIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISDestroy(is[i]);CHKERRQ(ierr);
    
    k = 0;
    for (j=0; j<ov; j++){ /* for each overlap */
      n = isz;
      for (; k<n ; k++){ /* do only those rows in nidx[k], which are not done yet */
        row   = nidx[k];
        start = ai[row];
        end   = ai[row+1];
        for (l = start; l<end ; l++){
          val = aj[l] + shift;
          if (!PetscBTLookupSet(table,val)) {nidx[isz++] = val;}
        }
      }
    }
    ierr = ISCreateGeneral(PETSC_COMM_SELF,isz,nidx,(is+i));CHKERRQ(ierr);
  }
  ierr = PetscBTDestroy(table);CHKERRQ(ierr);
  ierr = PetscFree(nidx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "MatPermute_SeqAIJ"
int MatPermute_SeqAIJ(Mat A,IS rowp,IS colp,Mat *B)
{ 
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscScalar  *vwork;
  int          i,ierr,nz,m = A->m,n = A->n,*cwork;
  int          *row,*col,*cnew,j,*lens;
  IS           icolp,irowp;

  PetscFunctionBegin;
  ierr = ISInvertPermutation(rowp,PETSC_DECIDE,&irowp);CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&row);CHKERRQ(ierr);
  ierr = ISInvertPermutation(colp,PETSC_DECIDE,&icolp);CHKERRQ(ierr);
  ierr = ISGetIndices(icolp,&col);CHKERRQ(ierr);
  
  /* determine lengths of permuted rows */
  ierr = PetscMalloc((m+1)*sizeof(int),&lens);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    lens[row[i]] = a->i[i+1] - a->i[i];
  }
  ierr = MatCreateSeqAIJ(A->comm,m,n,0,lens,B);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  ierr = PetscMalloc(n*sizeof(int),&cnew);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = MatGetRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    for (j=0; j<nz; j++) { cnew[j] = col[cwork[j]];}
    ierr = MatSetValues(*B,1,&row[i],nz,cnew,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = PetscFree(cnew);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISRestoreIndices(irowp,&row);CHKERRQ(ierr);
  ierr = ISRestoreIndices(icolp,&col);CHKERRQ(ierr);
  ierr = ISDestroy(irowp);CHKERRQ(ierr);
  ierr = ISDestroy(icolp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatPrintHelp_SeqAIJ"
int MatPrintHelp_SeqAIJ(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  int               ierr;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATSEQAIJ and MATMPIAIJ matrix formats (the defaults):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_lu_pivotthreshold <threshold>: Set pivoting threshold\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_oneindex: internal indices begin at 1 instead of the default 0.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_no_inode: Do not use inodes\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_inode_limit <limit>: Set inode limit (max limit=5)\n");CHKERRQ(ierr);
#if defined(PETSC_HAVE_ESSL)
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_essl: Use IBM sparse LU factorization and solve.\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_LUSOL)
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_lusol: Use the Stanford LUSOL sparse factorization and solve.\n");CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_matlab: Use Matlab engine sparse LU factorization and solve.\n");CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
EXTERN int MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg);
EXTERN int MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);
EXTERN int MatILUDTFactor_SeqAIJ(Mat,MatILUInfo*,IS,IS,Mat*);
#undef __FUNCT__  
#define __FUNCT__ "MatCopy_SeqAIJ"
int MatCopy_SeqAIJ(Mat A,Mat B,MatStructure str)
{
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (str == SAME_NONZERO_PATTERN && flg) {
    Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)B->data; 

    if (a->i[A->m]+a->indexshift != b->i[B->m]+a->indexshift) {
      SETERRQ(1,"Number of nonzeros in two matrices are different");
    }
    ierr = PetscMemcpy(b->a,a->a,(a->i[A->m]+a->indexshift)*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqAIJ"
int MatSetUpPreallocation_SeqAIJ(Mat A)
{
  int        ierr;

  PetscFunctionBegin;
  ierr =  MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqAIJ"
int MatGetArray_SeqAIJ(Mat A,PetscScalar **array)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqAIJ"
int MatRestoreArray_SeqAIJ(Mat A,PetscScalar **array)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringApply_SeqAIJ"
int MatFDColoringApply_SeqAIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  int           (*f)(void *,Vec,Vec,void*) = (int (*)(void *,Vec,Vec,void *))coloring->f;
  int           k,ierr,N,start,end,l,row,col,srow,**vscaleforrow,m1,m2;
  PetscScalar   dx,mone = -1.0,*y,*xx,*w3_array;
  PetscScalar   *vscale_array;
  PetscReal     epsilon = coloring->error_rel,umin = coloring->umin; 
  Vec           w1,w2,w3;
  void          *fctx = coloring->fctx;
  PetscTruth    flg;

  PetscFunctionBegin;
  if (!coloring->w1) {
    ierr = VecDuplicate(x1,&coloring->w1);CHKERRQ(ierr);
    PetscLogObjectParent(coloring,coloring->w1);
    ierr = VecDuplicate(x1,&coloring->w2);CHKERRQ(ierr);
    PetscLogObjectParent(coloring,coloring->w2);
    ierr = VecDuplicate(x1,&coloring->w3);CHKERRQ(ierr);
    PetscLogObjectParent(coloring,coloring->w3);
  }
  w1 = coloring->w1; w2 = coloring->w2; w3 = coloring->w3;

  ierr = MatSetUnfactored(J);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(coloring->prefix,"-mat_fd_coloring_dont_rezero",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscLogInfo(coloring,"MatFDColoringApply_SeqAIJ: Not calling MatZeroEntries()\n");
  } else {
    ierr = MatZeroEntries(J);CHKERRQ(ierr);
  }

  ierr = VecGetOwnershipRange(x1,&start,&end);CHKERRQ(ierr);
  ierr = VecGetSize(x1,&N);CHKERRQ(ierr);

  /*
       This is a horrible, horrible, hack. See DMMGComputeJacobian_Multigrid() it inproperly sets
     coloring->F for the coarser grids from the finest
  */
  if (coloring->F) {
    ierr = VecGetLocalSize(coloring->F,&m1);CHKERRQ(ierr);
    ierr = VecGetLocalSize(w1,&m2);CHKERRQ(ierr);
    if (m1 != m2) {
      coloring->F = 0; 
    }    
  }

  if (coloring->F) {
    w1          = coloring->F;
    coloring->F = 0;
  } else {
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
  }

  /* 
      Compute all the scale factors and share with other processors
  */
  ierr = VecGetArray(x1,&xx);CHKERRQ(ierr);xx = xx - start;
  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);vscale_array = vscale_array - start;
  for (k=0; k<coloring->ncolors; k++) { 
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* column of the matrix we are probing for */
      dx  = xx[col];
      if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
      dx                *= epsilon;
      vscale_array[col] = 1.0/dx;
    }
  } 
  vscale_array = vscale_array + start;ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  ierr = VecGhostUpdateBegin(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecGhostUpdateEnd(coloring->vscale,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);

  /*  ierr = VecView(coloring->vscale,PETSC_VIEWER_STDOUT_WORLD);
      ierr = VecView(x1,PETSC_VIEWER_STDOUT_WORLD);*/

  if (coloring->vscaleforrow) vscaleforrow = coloring->vscaleforrow;
  else                        vscaleforrow = coloring->columnsforrow;

  ierr = VecGetArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  /*
      Loop over each color
  */
  for (k=0; k<coloring->ncolors; k++) { 
    ierr = VecCopy(x1,w3);CHKERRQ(ierr);
    ierr = VecGetArray(w3,&w3_array);CHKERRQ(ierr);w3_array = w3_array - start;
    /*
       Loop over each column associated with color adding the 
       perturbation to the vector w3.
    */
    for (l=0; l<coloring->ncolumns[k]; l++) {
      col = coloring->columns[k][l];    /* column of the matrix we are probing for */
      dx  = xx[col];
      if (dx == 0.0) dx = 1.0;
#if !defined(PETSC_USE_COMPLEX)
      if (dx < umin && dx >= 0.0)      dx = umin;
      else if (dx < 0.0 && dx > -umin) dx = -umin;
#else
      if (PetscAbsScalar(dx) < umin && PetscRealPart(dx) >= 0.0)     dx = umin;
      else if (PetscRealPart(dx) < 0.0 && PetscAbsScalar(dx) < umin) dx = -umin;
#endif
      dx            *= epsilon;
      if (!PetscAbsScalar(dx)) SETERRQ(1,"Computed 0 differencing parameter");
      w3_array[col] += dx;
    } 
    w3_array = w3_array + start; ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);

    /*
       Evaluate function at x1 + dx (here dx is a vector of perturbations)
    */

    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
    ierr = VecAXPY(&mone,w1,w2);CHKERRQ(ierr);

    /*
       Loop over rows of vector, putting results into Jacobian matrix
    */
    ierr = VecGetArray(w2,&y);CHKERRQ(ierr);
    for (l=0; l<coloring->nrows[k]; l++) {
      row    = coloring->rows[k][l];
      col    = coloring->columnsforrow[k][l];
      y[row] *= vscale_array[vscaleforrow[k][l]];
      srow   = row + start;
      ierr   = MatSetValues_SeqAIJ(J,1,&srow,1,&col,y+row,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = VecRestoreArray(w2,&y);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  xx = xx + start; ierr  = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"

#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqAIJ"
int MatAXPY_SeqAIJ(PetscScalar *a,Mat X,Mat Y,MatStructure str)
{
  int        ierr,one=1;
  Mat_SeqAIJ *x  = (Mat_SeqAIJ *)X->data,*y = (Mat_SeqAIJ *)Y->data;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    BLaxpy_(&x->nz,a,x->a,&one,y->a,&one);
  } else {
    ierr = MatAXPY_Basic(a,X,Y,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,
       MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,
       MatMultAdd_SeqAIJ,
       MatMultTranspose_SeqAIJ,
       MatMultTransposeAdd_SeqAIJ,
       MatSolve_SeqAIJ,
       MatSolveAdd_SeqAIJ,
       MatSolveTranspose_SeqAIJ,
       MatSolveTransposeAdd_SeqAIJ,
       MatLUFactor_SeqAIJ,
       0,
       MatRelax_SeqAIJ,
       MatTranspose_SeqAIJ,
       MatGetInfo_SeqAIJ,
       MatEqual_SeqAIJ,
       MatGetDiagonal_SeqAIJ,
       MatDiagonalScale_SeqAIJ,
       MatNorm_SeqAIJ,
       0,
       MatAssemblyEnd_SeqAIJ,
       MatCompress_SeqAIJ,
       MatSetOption_SeqAIJ,
       MatZeroEntries_SeqAIJ,
       MatZeroRows_SeqAIJ,
       MatLUFactorSymbolic_SeqAIJ,
       MatLUFactorNumeric_SeqAIJ,
       0,
       0,
       MatSetUpPreallocation_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,
       0,
       MatGetArray_SeqAIJ,
       MatRestoreArray_SeqAIJ,
       MatDuplicate_SeqAIJ,
       0,
       0,
       MatILUFactor_SeqAIJ,
       0,
       MatAXPY_SeqAIJ,
       MatGetSubMatrices_SeqAIJ,
       MatIncreaseOverlap_SeqAIJ,
       MatGetValues_SeqAIJ,
       MatCopy_SeqAIJ,
       MatPrintHelp_SeqAIJ,
       MatScale_SeqAIJ,
       0,
       0,
       MatILUDTFactor_SeqAIJ,
       MatGetBlockSize_SeqAIJ,
       MatGetRowIJ_SeqAIJ,
       MatRestoreRowIJ_SeqAIJ,
       MatGetColumnIJ_SeqAIJ,
       MatRestoreColumnIJ_SeqAIJ,
       MatFDColoringCreate_SeqAIJ,
       0,
       0,
       MatPermute_SeqAIJ,
       0,
       0,
       MatDestroy_SeqAIJ,
       MatView_SeqAIJ,
       MatGetPetscMaps_Petsc,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       0,
       MatSetColoring_SeqAIJ,
       MatSetValuesAdic_SeqAIJ,
       MatSetValuesAdifor_SeqAIJ,
       MatFDColoringApply_SeqAIJ};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJSetColumnIndices_SeqAIJ"

int MatSeqAIJSetColumnIndices_SeqAIJ(Mat mat,int *indices)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        i,nz,n;

  PetscFunctionBegin;

  nz = aij->maxnz;
  n  = mat->n;
  for (i=0; i<nz; i++) {
    aij->j[i] = indices[i];
  }
  aij->nz = nz;
  for (i=0; i<n; i++) {
    aij->ilen[i] = aij->imax[i];
  }

  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJSetColumnIndices"
/*@
    MatSeqAIJSetColumnIndices - Set the column indices for all the rows
       in the matrix.

  Input Parameters:
+  mat - the SeqAIJ matrix
-  indices - the column indices

  Level: advanced

  Notes:
    This can be called if you have precomputed the nonzero structure of the 
  matrix and want to provide it to the matrix object to improve the performance
  of the MatSetValues() operation.

    You MUST have set the correct numbers of nonzeros per row in the call to 
  MatCreateSeqAIJ().

    MUST be called before any calls to MatSetValues();

    The indices should start with zero, not one.

@*/ 
int MatSeqAIJSetColumnIndices(Mat mat,int *indices)
{
  int ierr,(*f)(Mat,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqAIJSetColumnIndices_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_SeqAIJ"
int MatStoreValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        nz = aij->i[mat->m]+aij->indexshift,ierr;

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

#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues"
/*@
    MatStoreValues - Stashes a copy of the matrix values; this allows, for 
       example, reuse of the linear part of a Jacobian, while recomputing the 
       nonlinear portion.

   Collect on Mat

  Input Parameters:
.  mat - the matrix (currently on AIJ matrices support this option)

  Level: advanced

  Common Usage, with SNESSolve():
$    Create Jacobian matrix
$    Set linear terms into matrix
$    Apply boundary conditions to matrix, at this time matrix must have 
$      final nonzero structure (i.e. setting the nonlinear terms and applying 
$      boundary conditions again will not change the nonzero structure
$    ierr = MatSetOption(mat,MAT_NO_NEW_NONZERO_LOCATIONS);
$    ierr = MatStoreValues(mat);
$    Call SNESSetJacobian() with matrix
$    In your Jacobian routine
$      ierr = MatRetrieveValues(mat);
$      Set nonlinear terms in matrix
 
  Common Usage without SNESSolve(), i.e. when you handle nonlinear solve yourself:
$    // build linear portion of Jacobian 
$    ierr = MatSetOption(mat,MAT_NO_NEW_NONZERO_LOCATIONS);
$    ierr = MatStoreValues(mat);
$    loop over nonlinear iterations
$       ierr = MatRetrieveValues(mat);
$       // call MatSetValues(mat,...) to set nonliner portion of Jacobian 
$       // call MatAssemblyBegin/End() on matrix
$       Solve linear system with Jacobian
$    endloop 

  Notes:
    Matrix must already be assemblied before calling this routine
    Must set the matrix option MatSetOption(mat,MAT_NO_NEW_NONZERO_LOCATIONS); before 
    calling this routine.

.seealso: MatRetrieveValues()

@*/ 
int MatStoreValues(Mat mat)
{
  int ierr,(*f)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatStoreValues_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Wrong type of matrix to store values");
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_SeqAIJ"
int MatRetrieveValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        nz = aij->i[mat->m]+aij->indexshift,ierr;

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

#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues"
/*@
    MatRetrieveValues - Retrieves the copy of the matrix values; this allows, for 
       example, reuse of the linear part of a Jacobian, while recomputing the 
       nonlinear portion.

   Collect on Mat

  Input Parameters:
.  mat - the matrix (currently on AIJ matrices support this option)

  Level: advanced

.seealso: MatStoreValues()

@*/ 
int MatRetrieveValues(Mat mat)
{
  int ierr,(*f)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatRetrieveValues_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Wrong type of matrix to retrieve values");
  }
  PetscFunctionReturn(0);
}

/*
   This allows SeqAIJ matrices to be passed to the matlab engine
*/
#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
#include "engine.h"   /* Matlab include file */
#include "mex.h"      /* Matlab include file */
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMatlabEnginePut_SeqAIJ"
int MatMatlabEnginePut_SeqAIJ(PetscObject obj,void *engine)
{
  int         ierr,i,*ai,*aj;
  Mat         B = (Mat)obj;
  PetscScalar *array;
  mxArray     *mat; 
  Mat_SeqAIJ  *aij = (Mat_SeqAIJ*)B->data;

  PetscFunctionBegin;
  mat  = mxCreateSparse(B->n,B->m,aij->nz,mxREAL);
  ierr = PetscMemcpy(mxGetPr(mat),aij->a,aij->nz*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Matlab stores by column, not row so we pass in the transpose of the matrix */
  ierr = PetscMemcpy(mxGetIr(mat),aij->j,aij->nz*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(mxGetJc(mat),aij->i,(B->m+1)*sizeof(int));CHKERRQ(ierr);

  /* Matlab indices start at 0 for sparse (what a surprise) */
  if (aij->indexshift) {
    for (i=0; i<B->m+1; i++) {
      ai[i]--;
    }
    for (i=0; i<aij->nz; i++) {
      aj[i]--;
    }
  }
  ierr = PetscObjectName(obj);CHKERRQ(ierr);
  mxSetName(mat,obj->name);
  engPutArray((Engine *)engine,mat);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatMatlabEngineGet_SeqAIJ"
int MatMatlabEngineGet_SeqAIJ(PetscObject obj,void *engine)
{
  int        ierr,ii;
  Mat        mat = (Mat)obj;
  Mat_SeqAIJ *aij = (Mat_SeqAIJ*)mat->data;
  mxArray    *mmat; 

  PetscFunctionBegin;
  ierr = PetscFree(aij->a);CHKERRQ(ierr);
  aij->indexshift = 0;

  mmat = engGetArray((Engine *)engine,obj->name);

  aij->nz           = (mxGetJc(mmat))[mat->m];
  ierr              = PetscMalloc(aij->nz*(sizeof(int)+sizeof(PetscScalar))+(mat->m+1)*sizeof(int),&aij->a);CHKERRQ(ierr);
  aij->j            = (int*)(aij->a + aij->nz);
  aij->i            = aij->j + aij->nz;
  aij->singlemalloc = PETSC_TRUE;
  aij->freedata     = PETSC_TRUE;

  ierr = PetscMemcpy(aij->a,mxGetPr(mmat),aij->nz*sizeof(PetscScalar));CHKERRQ(ierr);
  /* Matlab stores by column, not row so we pass in the transpose of the matrix */
  ierr = PetscMemcpy(aij->j,mxGetIr(mmat),aij->nz*sizeof(int));CHKERRQ(ierr);
  ierr = PetscMemcpy(aij->i,mxGetJc(mmat),(mat->m+1)*sizeof(int));CHKERRQ(ierr);

  for (ii=0; ii<mat->m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = aij->i[ii+1] - aij->i[ii];
  }

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END
#endif

/* --------------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJ"
/*@C
   MatCreateSeqAIJ - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_aij_no_inode  - Do not use inodes
.  -mat_aij_inode_limit <limit> - Sets inode limit (max limit=5)
-  -mat_aij_oneindex - Internally use indexing starting at 1
        rather than 0.  Note that when calling MatSetValues(),
        the user still MUST index entries starting at 0!

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays()

@*/
int MatCreateSeqAIJ(MPI_Comm comm,int m,int n,int nz,int *nnz,Mat *A)
{
  int ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,m,n,A);CHKERRQ(ierr);
  ierr = MatSetType(*A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*A,nz,nnz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define SKIP_ALLOCATION -4

#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJSetPreallocation"
/*@C
   MatSeqAIJSetPreallocation - For good matrix assembly performance
   the user should preallocate the matrix storage by setting the parameter nz
   (or the array nnz).  By setting these parameters accurately, performance
   during matrix assembly can be increased by more than a factor of 50.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows 
         (possibly different for each row) or PETSC_NULL

   Output Parameter:
.  A - the matrix 

   Notes:
   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=PETSC_NULL for PETSc to control dynamic memory 
   allocation.  For large problems you MUST preallocate memory or you 
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to 
   improve numerical efficiency of matrix-vector products and solves. We 
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Options Database Keys:
+  -mat_aij_no_inode  - Do not use inodes
.  -mat_aij_inode_limit <limit> - Sets inode limit (max limit=5)
-  -mat_aij_oneindex - Internally use indexing starting at 1
        rather than 0.  Note that when calling MatSetValues(),
        the user still MUST index entries starting at 0!

   Level: intermediate

.seealso: MatCreate(), MatCreateMPIAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays()

@*/
int MatSeqAIJSetPreallocation(Mat B,int nz,int *nnz)
{
  Mat_SeqAIJ *b;
  int        i,len=0,ierr;
  PetscTruth flg2,skipallocation = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATSEQAIJ,&flg2);CHKERRQ(ierr);
  if (!flg2) PetscFunctionReturn(0);
  
  if (nz == SKIP_ALLOCATION) {
    skipallocation = PETSC_TRUE;
    nz             = 0;
  }

  if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 5;
  if (nz < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"nz cannot be less than 0: value %d",nz);
  if (nnz) {
    for (i=0; i<B->m; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be less than 0: local row %d value %d",i,nnz[i]);
      if (nnz[i] > B->n) SETERRQ3(PETSC_ERR_ARG_OUTOFRANGE,"nnz cannot be greater than row length: local row %d value %d rowlength %d",i,nnz[i],B->n);
    }
  }

  B->preallocated = PETSC_TRUE;
  b = (Mat_SeqAIJ*)B->data;

  ierr = PetscMalloc((B->m+1)*sizeof(int),&b->imax);CHKERRQ(ierr);
  if (!nnz) {
    if (nz == PETSC_DEFAULT || nz == PETSC_DECIDE) nz = 10;
    else if (nz <= 0)        nz = 1;
    for (i=0; i<B->m; i++) b->imax[i] = nz;
    nz = nz*B->m;
  } else {
    nz = 0;
    for (i=0; i<B->m; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  if (!skipallocation) {
    /* allocate the matrix space */
    len             = nz*(sizeof(int) + sizeof(PetscScalar)) + (B->m+1)*sizeof(int);
    ierr            = PetscMalloc(len,&b->a);CHKERRQ(ierr);
    b->j            = (int*)(b->a + nz);
    ierr            = PetscMemzero(b->j,nz*sizeof(int));CHKERRQ(ierr);
    b->i            = b->j + nz;
    b->i[0] = -b->indexshift;
    for (i=1; i<B->m+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }
    b->singlemalloc = PETSC_TRUE;
    b->freedata     = PETSC_TRUE;
  } else {
    b->freedata     = PETSC_FALSE;
  }

  /* b->ilen will count nonzeros in each row so far. */
  ierr = PetscMalloc((B->m+1)*sizeof(int),&b->ilen);CHKERRQ(ierr);
  PetscLogObjectMemory(B,len+2*(B->m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));
  for (i=0; i<B->m; i++) { b->ilen[i] = 0;}

  b->nz                = 0;
  b->maxnz             = nz;
  B->info.nz_unneeded  = (double)b->maxnz;
  PetscFunctionReturn(0);
}

EXTERN int RegisterApplyPtAPRoutines_Private(Mat);

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJ"
int MatCreate_SeqAIJ(Mat B)
{
  Mat_SeqAIJ *b;
  int        ierr,size;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Comm must be of size 1");

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);

  ierr = PetscNew(Mat_SeqAIJ,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr = PetscMemzero(b,sizeof(Mat_SeqAIJ));CHKERRQ(ierr);
  ierr = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = PetscOptionsGetReal(B->prefix,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(B->prefix,"-pc_ilu_preserve_row_sums",&b->ilu_preserve_row_sums);CHKERRQ(ierr);
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->indexshift       = 0;
  b->reallocs         = 0;
  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_oneindex",&flg);CHKERRQ(ierr);
  if (flg) b->indexshift = -1;
  
  ierr = PetscMapCreateMPI(B->comm,B->m,B->m,&B->rmap);CHKERRQ(ierr);
  ierr = PetscMapCreateMPI(B->comm,B->n,B->n,&B->cmap);CHKERRQ(ierr);

  b->sorted            = PETSC_FALSE;
  b->ignorezeroentries = PETSC_FALSE;
  b->roworiented       = PETSC_TRUE;
  b->nonew             = 0;
  b->diag              = 0;
  b->solve_work        = 0;
  B->spptr             = 0;
  b->inode.use         = PETSC_TRUE;
  b->inode.node_count  = 0;
  b->inode.size        = 0;
  b->inode.limit       = 5;
  b->inode.max_limit   = 5;
  b->saved_values      = 0;
  b->idiag             = 0;
  b->ssor              = 0;
  b->keepzeroedrows    = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);

#if defined(PETSC_HAVE_SUPERLU)
  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_superlu",&flg);CHKERRQ(ierr);
  if (flg) { ierr = MatUseSuperLU_SeqAIJ(B);CHKERRQ(ierr); }
#endif

  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_essl",&flg);CHKERRQ(ierr);
  if (flg) { ierr = MatUseEssl_SeqAIJ(B);CHKERRQ(ierr); }
  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_lusol",&flg);CHKERRQ(ierr);
  if (flg) { ierr = MatUseLUSOL_SeqAIJ(B);CHKERRQ(ierr); }
  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_matlab",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatUseMatlab_SeqAIJ(B);CHKERRQ(ierr);}
  ierr = PetscOptionsHasName(B->prefix,"-mat_aij_dxml",&flg);CHKERRQ(ierr);
  if (flg) {
    if (!b->indexshift) SETERRQ(PETSC_ERR_LIB,"need -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(B);CHKERRQ(ierr);
  }
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetColumnIndices_C",
                                     "MatSeqAIJSetColumnIndices_SeqAIJ",
                                     MatSeqAIJSetColumnIndices_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqAIJ",
                                     MatStoreValues_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqAIJ",
                                     MatRetrieveValues_SeqAIJ);CHKERRQ(ierr);
#if defined(PETSC_HAVE_MATLAB_ENGINE) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"PetscMatlabEnginePut_C","MatMatlabEnginePut_SeqAIJ",MatMatlabEnginePut_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"PetscMatlabEngineGet_C","MatMatlabEngineGet_SeqAIJ",MatMatlabEngineGet_SeqAIJ);CHKERRQ(ierr);
#endif
  ierr = RegisterApplyPtAPRoutines_Private(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqAIJ"
int MatDuplicate_SeqAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat        C;
  Mat_SeqAIJ *c,*a = (Mat_SeqAIJ*)A->data;
  int        i,len,m = A->m,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  *B = 0;
  ierr = MatCreate(A->comm,A->m,A->n,A->m,A->n,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,MATSEQAIJ);CHKERRQ(ierr);
  c    = (Mat_SeqAIJ*)C->data;

  C->factor         = A->factor;
  c->row            = 0;
  c->col            = 0;
  c->icol           = 0;
  c->indexshift     = shift;
  c->keepzeroedrows = a->keepzeroedrows;
  C->assembled      = PETSC_TRUE;

  C->M          = A->m;
  C->N          = A->n;

  ierr = PetscMalloc((m+1)*sizeof(int),&c->imax);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(int),&c->ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = PETSC_TRUE;
  len   = (m+1)*sizeof(int)+(a->i[m])*(sizeof(PetscScalar)+sizeof(int));
  ierr  = PetscMalloc(len,&c->a);CHKERRQ(ierr);
  c->j  = (int*)(c->a + a->i[m] + shift);
  c->i  = c->j + a->i[m] + shift;
  ierr = PetscMemcpy(c->i,a->i,(m+1)*sizeof(int));CHKERRQ(ierr);
  if (m > 0) {
    ierr = PetscMemcpy(c->j,a->j,(a->i[m]+shift)*sizeof(int));CHKERRQ(ierr);
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscMemcpy(c->a,a->a,(a->i[m]+shift)*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(c->a,(a->i[m]+shift)*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }

  PetscLogObjectMemory(C,len+2*(m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));  
  c->sorted      = a->sorted;
  c->roworiented = a->roworiented;
  c->nonew       = a->nonew;
  c->ilu_preserve_row_sums = a->ilu_preserve_row_sums;
  c->saved_values = 0;
  c->idiag        = 0;
  c->ssor         = 0;
  c->ignorezeroentries = a->ignorezeroentries;
  c->freedata     = PETSC_TRUE;

  if (a->diag) {
    ierr = PetscMalloc((m+1)*sizeof(int),&c->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(C,(m+1)*sizeof(int));
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag        = 0;
  c->inode.use          = a->inode.use;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    ierr                = PetscMalloc((m+1)*sizeof(int),&c->inode.size);CHKERRQ(ierr);
    c->inode.node_count = a->inode.node_count;
    ierr                = PetscMemcpy(c->inode.size,a->inode.size,(m+1)*sizeof(int));CHKERRQ(ierr);
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  C->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */
  C->preallocated       = PETSC_TRUE;

  *B = C;
  ierr = PetscFListDuplicate(A->qlist,&C->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqAIJ"
int MatLoad_SeqAIJ(PetscViewer viewer,MatType type,Mat *A)
{
  Mat_SeqAIJ   *a;
  Mat          B;
  int          i,nz,ierr,fd,header[4],size,*rowlengths = 0,M,N,shift;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,"view must have one processor");
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_FILE_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"not matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"Matrix stored in special format on disk,cannot load as SeqAIJ");
  }

  /* read in row lengths */
  ierr = PetscMalloc(M*sizeof(int),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreateSeqAIJ(comm,M,N,0,rowlengths,A);CHKERRQ(ierr);
  B = *A;
  a = (Mat_SeqAIJ*)B->data;
  shift = a->indexshift;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = PetscBinaryRead(fd,a->j,nz,PETSC_INT);CHKERRQ(ierr);
  if (shift) {
    for (i=0; i<nz; i++) {
      a->j[i] += 1;
    }
  }

  /* read in nonzero values */
  ierr = PetscBinaryRead(fd,a->a,nz,PETSC_SCALAR);CHKERRQ(ierr);

  /* set matrix "i" values */
  a->i[0] = -shift;
  for (i=1; i<= M; i++) {
    a->i[i]      = a->i[i-1] + rowlengths[i-1];
    a->ilen[i-1] = rowlengths[i-1];
  }
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_SeqAIJ"
int MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data,*b = (Mat_SeqAIJ *)B->data;
  int        ierr;
  PetscTruth flag;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)B,MATSEQAIJ,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(PETSC_ERR_ARG_INCOMP,"Matrices must be same type");

  /* If the  matrix dimensions are not equal,or no of nonzeros or shift */
  if ((A->m != B->m) || (A->n != B->n) ||(a->nz != b->nz)|| (a->indexshift != b->indexshift)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(A->m+1)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
  
  /* if a->a are the same */
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*sizeof(PetscScalar),flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
  
}

#undef __FUNCT__  
#define __FUNCT__ "MatCreateSeqAIJWithArrays"
/*@C
     MatCreateSeqAIJWithArrays - Creates an sequential AIJ matrix using matrix elements (in CSR format)
              provided by the user.

      Coolective on MPI_Comm

   Input Parameters:
+   comm - must be an MPI communicator of size 1
.   m - number of rows
.   n - number of columns
.   i - row indices
.   j - column indices
-   a - matrix values

   Output Parameter:
.   mat - the matrix

   Level: intermediate

   Notes:
       The i, j, and a arrays are not copied by this routine, the user must free these arrays
    once the matrix is destroyed

       You cannot set new nonzero locations into this matrix, that will generate an error.

       The i and j indices can be either 0- or 1 based

.seealso: MatCreate(), MatCreateMPIAIJ(), MatCreateSeqAIJ()

@*/
int MatCreateSeqAIJWithArrays(MPI_Comm comm,int m,int n,int* i,int*j,PetscScalar *a,Mat *mat)
{
  int        ierr,ii;
  Mat_SeqAIJ *aij;

  PetscFunctionBegin;
  ierr = MatCreateSeqAIJ(comm,m,n,SKIP_ALLOCATION,0,mat);CHKERRQ(ierr);
  aij  = (Mat_SeqAIJ*)(*mat)->data;

  if (i[0] == 1) {
    aij->indexshift = -1;
  } else if (i[0]) {
    SETERRQ(1,"i (row indices) do not start with 0 or 1");
  }  
  aij->i = i;
  aij->j = j;
  aij->a = a;
  aij->singlemalloc = PETSC_FALSE;
  aij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aij->freedata     = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = i[ii+1] - i[ii];
#if defined(PETSC_USE_BOPT_g)
    if (i[ii+1] - i[ii] < 0) SETERRQ2(1,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
#endif    
  }
#if defined(PETSC_USE_BOPT_g)
  for (ii=0; ii<aij->i[m]; ii++) {
    if (j[ii] < -aij->indexshift) SETERRQ2(1,"Negative column index at location = %d index = %d",ii,j[ii]);
    if (j[ii] > n - 1 -aij->indexshift) SETERRQ2(1,"Column index to large at location = %d index = %d",ii,j[ii]);
  }
#endif    

  /* changes indices to start at 0 */
  if (i[0]) {
    aij->indexshift = 0;
    for (ii=0; ii<m; ii++) {
      i[ii]--;
    }
    for (ii=0; ii<i[m]; ii++) {
      j[ii]--;
    }
  }

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetColoring_SeqAIJ"
int MatSetColoring_SeqAIJ(Mat A,ISColoring coloring)
{
  int        ierr;
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;  

  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    ierr        = ISColoringReference(coloring);CHKERRQ(ierr);
    a->coloring = coloring;
  } else if (coloring->ctype == IS_COLORING_GHOSTED) {
    int        *colors,i,*larray;
    ISColoring ocoloring;

    /* set coloring for diagonal portion */
    ierr = PetscMalloc((A->n+1)*sizeof(int),&larray);CHKERRQ(ierr);
    for (i=0; i<A->n; i++) {
      larray[i] = i;
    }
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,A->n,larray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((A->n+1)*sizeof(int),&colors);CHKERRQ(ierr);
    for (i=0; i<A->n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(PETSC_COMM_SELF,A->n,colors,&ocoloring);CHKERRQ(ierr);
    a->coloring = ocoloring;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ADIC) && !defined(PETSC_USE_COMPLEX) && !defined(PETSC_USE_SINGLE)
EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic_SeqAIJ"
int MatSetValuesAdic_SeqAIJ(Mat A,void *advalues)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;  
  int         m = A->m,*ii = a->i,*jj = a->j,nz,i,*color,j,nlen;
  PetscScalar *v = a->a,*values;
  char        *cadvalues = (char *)advalues;

  PetscFunctionBegin;
  if (!a->coloring) SETERRQ(1,"Coloring not set for matrix");
  nlen  = PetscADGetDerivTypeSize();
  color = a->coloring->colors;
  /* loop over rows */
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    /* loop over columns putting computed value into matrix */
    values = PetscADGetGradArray(cadvalues);
    for (j=0; j<nz; j++) {
      *v++ = values[color[*jj++]];
    }
    cadvalues += nlen; /* jump to next row of derivatives */
  }
  PetscFunctionReturn(0);
}

#else

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic_SeqAIJ"
int MatSetValuesAdic_SeqAIJ(Mat A,void *advalues)
{
  PetscFunctionBegin;
  SETERRQ(1,"PETSc installed without ADIC");
}

#endif

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdifor_SeqAIJ"
int MatSetValuesAdifor_SeqAIJ(Mat A,int nl,void *advalues)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;  
  int          m = A->m,*ii = a->i,*jj = a->j,nz,i,*color,j;
  PetscScalar  *v = a->a,*values = (PetscScalar *)advalues;

  PetscFunctionBegin;
  if (!a->coloring) SETERRQ(1,"Coloring not set for matrix");
  color = a->coloring->colors;
  /* loop over rows */
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    /* loop over columns putting computed value into matrix */
    for (j=0; j<nz; j++) {
      *v++ = values[color[*jj++]];
    }
    values += nl; /* jump to next row of derivatives */
  }
  PetscFunctionReturn(0);
}

