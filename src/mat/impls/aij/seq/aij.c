/*$Id: aij.c,v 1.339 2000/02/02 20:08:56 bsmith Exp bsmith $*/
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "sys.h"
#include "src/mat/impls/aij/seq/aij.h"
#include "src/vec/vecimpl.h"
#include "src/inline/spops.h"
#include "src/inline/dot.h"
#include "bitarray.h"


extern int MatToSymmetricIJ_SeqAIJ(int,int*,int*,int,int,int**,int**);

#undef __FUNC__  
#define __FUNC__ "MatGetRowIJ_SeqAIJ"
int MatGetRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *m,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr,i,ishift;
 
  PetscFunctionBegin;  
  *m     = A->m;
  if (!ia) PetscFunctionReturn(0);
  ishift = a->indexshift;
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->m,a->i,a->j,ishift,oshift,ia,ja);CHKERRQ(ierr);
  } else if (oshift == 0 && ishift == -1) {
    int nz = a->i[a->m]; 
    /* malloc space and  subtract 1 from i and j indices */
    *ia = (int*)PetscMalloc((a->m+1)*sizeof(int));CHKPTRQ(*ia);
    *ja = (int*)PetscMalloc((nz+1)*sizeof(int));CHKPTRQ(*ja);
    for (i=0; i<nz; i++) (*ja)[i] = a->j[i] - 1;
    for (i=0; i<a->m+1; i++) (*ia)[i] = a->i[i] - 1;
  } else if (oshift == 1 && ishift == 0) {
    int nz = a->i[a->m] + 1; 
    /* malloc space and  add 1 to i and j indices */
    *ia = (int*)PetscMalloc((a->m+1)*sizeof(int));CHKPTRQ(*ia);
    *ja = (int*)PetscMalloc((nz+1)*sizeof(int));CHKPTRQ(*ja);
    for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
    for (i=0; i<a->m+1; i++) (*ia)[i] = a->i[i] + 1;
  } else {
    *ia = a->i; *ja = a->j;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRowIJ_SeqAIJ"
int MatRestoreRowIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *n,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ishift = a->indexshift,ierr;
 
  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);
  if (symmetric || (oshift == 0 && ishift == -1) || (oshift == 1 && ishift == 0)) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNC__  
#define __FUNC__ "MatGetColumnIJ_SeqAIJ"
int MatGetColumnIJ_SeqAIJ(Mat A,int oshift,PetscTruth symmetric,int *nn,int **ia,int **ja,PetscTruth *done)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr,i,ishift = a->indexshift,*collengths,*cia,*cja,n = A->n,m = A->m;
  int        nz = a->i[m]+ishift,row,*jj,mr,col;
 
  PetscFunctionBegin;  
  *nn     = A->n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(a->m,a->i,a->j,ishift,oshift,ia,ja);CHKERRQ(ierr);
  } else {
    collengths = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ(collengths);
    ierr       = PetscMemzero(collengths,n*sizeof(int));CHKERRQ(ierr);
    cia        = (int*)PetscMalloc((n+1)*sizeof(int));CHKPTRQ(cia);
    cja        = (int*)PetscMalloc((nz+1)*sizeof(int));CHKPTRQ(cja);
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

#undef __FUNC__  
#define __FUNC__ "MatRestoreColumnIJ_SeqAIJ"
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

#undef __FUNC__  
#define __FUNC__ "MatSetValues_SeqAIJ"
int MatSetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v,InsertMode is)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted = a->sorted;
  int        *imax = a->imax,*ai = a->i,*ailen = a->ilen,roworiented = a->roworiented;
  int        *aj = a->j,nonew = a->nonew,shift = a->indexshift,ierr;
  Scalar     *ap,value,*aa = a->a;
  PetscTruth ignorezeroentries = ((a->ignorezeroentries && is == ADD_VALUES) ? PETSC_TRUE:PETSC_FALSE);

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
    if (row >= a->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large: row %d max %d",row,a->m);
#endif
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    rmax = imax[row]; nrow = ailen[row]; 
    low = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_BOPT_g)  
      if (in[l] >= a->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large: col %d max %d",in[l],a->n);
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
      else if (nonew == -1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        int    new_nz = ai[a->m] + CHUNKSIZE,len,*new_i,*new_j;
        Scalar *new_a;

        if (nonew == -2) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Inserting a new nonzero in the matrix");

        /* malloc new storage space */
        len     = new_nz*(sizeof(int)+sizeof(Scalar))+(a->m+1)*sizeof(int);
        new_a   = (Scalar*)PetscMalloc(len);CHKPTRQ(new_a);
        new_j   = (int*)(new_a + new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for (ii=0; ii<row+1; ii++) {new_i[ii] = ai[ii];}
        for (ii=row+1; ii<a->m+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow+shift)*sizeof(int));CHKERRQ(ierr);
        len  = (new_nz - CHUNKSIZE - ai[row] - nrow - shift);
        ierr = PetscMemcpy(new_j+ai[row]+shift+nrow+CHUNKSIZE,aj+ai[row]+shift+nrow,len*sizeof(int));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a,aa,(ai[row]+nrow+shift)*sizeof(Scalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a+ai[row]+shift+nrow+CHUNKSIZE,aa+ai[row]+shift+nrow,len*sizeof(Scalar));CHKERRQ(ierr);
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
        PLogObjectMemory(A,CHUNKSIZE*(sizeof(int) + sizeof(Scalar)));
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

#undef __FUNC__  
#define __FUNC__ "MatGetValues_SeqAIJ"
int MatGetValues_SeqAIJ(Mat A,int m,int *im,int n,int *in,Scalar *v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  int        *ai = a->i,*ailen = a->ilen,shift = a->indexshift;
  Scalar     *ap,*aa = a->a,zero = 0.0;

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k];   
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative row");
    if (row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row too large");
    rp   = aj + ai[row] + shift; ap = aa + ai[row] + shift;
    nrow = ailen[row]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Negative column");
      if (in[l] >= a->n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Column too large");
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


#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ_Binary"
int MatView_SeqAIJ_Binary(Mat A,Viewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        i,fd,*col_lens,ierr;

  PetscFunctionBegin;  
  ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  col_lens = (int*)PetscMalloc((4+a->m)*sizeof(int));CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for (i=0; i<a->m; i++) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,PETSC_INT,1);CHKERRQ(ierr);
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

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ_ASCII"
int MatView_SeqAIJ_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;
  int         ierr,i,j,m = a->m,shift = a->indexshift,format;
  char        *outputname;

  PetscFunctionBegin;  
  ierr = ViewerGetOutputname(viewer,&outputname);CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO_LONG || format == VIEWER_FORMAT_ASCII_INFO) {
    if (a->inode.size) {
      ierr = ViewerASCIIPrintf(viewer,"using I-node routines: found %d nodes, limit used is %d\n",a->inode.node_count,a->inode.limit);CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer,"not using I-node routines\n");CHKERRQ(ierr);
    }
  } else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    int nofinalvalue = 0;
    if ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != a->n-!shift)) {
      nofinalvalue = 1;
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"%% Size = %d %d \n",m,a->n);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"%% Nonzeros = %d \n",a->nz);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"zzz = zeros(%d,3);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = ViewerASCIIPrintf(viewer,"%d %d  %18.16e + %18.16ei \n",i+1,a->j[j]+!shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
#else
        ierr = ViewerASCIIPrintf(viewer,"%d %d  %18.16e\n",i+1,a->j[j]+!shift,a->a[j]);CHKERRQ(ierr);
#endif
      }
    }
    if (nofinalvalue) {
      ierr = ViewerASCIIPrintf(viewer,"%d %d  %18.16e\n",m,a->n,0.0);CHKERRQ(ierr);
    } 
    if (outputname) {ierr = ViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",outputname);CHKERRQ(ierr);}
    else            {ierr = ViewerASCIIPrintf(viewer,"];\n M = spconvert(zzz);\n");CHKERRQ(ierr);}
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == VIEWER_FORMAT_ASCII_COMMON) {
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = ViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g + %g i",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g - %g i",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscRealPart(a->a[j]) != 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        if (a->a[j] != 0.0) {ierr = ViewerASCIIPrintf(viewer," %d %g ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);}
#endif
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == VIEWER_FORMAT_ASCII_SYMMODU) {
    int nzd=0,fshift=1,*sptr;
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    sptr = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(sptr);
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
    ierr = ViewerASCIIPrintf(viewer," %d %d\n\n",m,nzd);CHKERRQ(ierr);
    for (i=0; i<m+1; i+=6) {
      if (i+4<m) {ierr = ViewerASCIIPrintf(viewer," %d %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4],sptr[i+5]);CHKERRQ(ierr);}
      else if (i+3<m) {ierr = ViewerASCIIPrintf(viewer," %d %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4]);CHKERRQ(ierr);}
      else if (i+2<m) {ierr = ViewerASCIIPrintf(viewer," %d %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3]);CHKERRQ(ierr);}
      else if (i+1<m) {ierr = ViewerASCIIPrintf(viewer," %d %d %d\n",sptr[i],sptr[i+1],sptr[i+2]);CHKERRQ(ierr);}
      else if (i<m)   {ierr = ViewerASCIIPrintf(viewer," %d %d\n",sptr[i],sptr[i+1]);CHKERRQ(ierr);}
      else            {ierr = ViewerASCIIPrintf(viewer," %d\n",sptr[i]);CHKERRQ(ierr);}
    }
    ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscFree(sptr);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {ierr = ViewerASCIIPrintf(viewer," %d ",a->j[j]+fshift);CHKERRQ(ierr);}
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {
#if defined(PETSC_USE_COMPLEX)
          if (PetscImaginaryPart(a->a[j]) != 0.0 || PetscRealPart(a->a[j]) != 0.0) {
            ierr = ViewerASCIIPrintf(viewer," %18.16e %18.16e ",PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
          }
#else
          if (a->a[j] != 0.0) {ierr = ViewerASCIIPrintf(viewer," %18.16e ",a->a[j]);CHKERRQ(ierr);}
#endif
        }
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == VIEWER_FORMAT_ASCII_DENSE) {
    int    cnt = 0,jcnt;
    Scalar value;

    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      jcnt = 0;
      for (j=0; j<a->n; j++) {
        if (jcnt < a->i[i+1]-a->i[i] && j == a->j[cnt]) {
          value = a->a[cnt++];
          jcnt++;
        } else {
          value = 0.0;
        }
#if defined(PETSC_USE_COMPLEX)
        ierr = ViewerASCIIPrintf(viewer," %7.5e+%7.5e i ",PetscRealPart(value),PetscImaginaryPart(value));CHKERRQ(ierr);
#else
        ierr = ViewerASCIIPrintf(viewer," %7.5e ",value);CHKERRQ(ierr);
#endif
      }
      ierr = ViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = ViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = ViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g + %g i",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
          ierr = ViewerASCIIPrintf(viewer," %d %g - %g i",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else {
          ierr = ViewerASCIIPrintf(viewer," %d %g ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        ierr = ViewerASCIIPrintf(viewer," %d %g ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);
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
#define __FUNC__ "MatView_SeqAIJ_Draw_Zoom"
int MatView_SeqAIJ_Draw_Zoom(Draw draw,void *Aa)
{
  Mat         A = (Mat) Aa;
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;
  int         ierr,i,j,m = a->m,shift = a->indexshift,color,rank;
  int         format;
  PetscReal   xl,yl,xr,yr,x_l,x_r,y_l,y_r,maxv = 0.0;
  Viewer      viewer;
  MPI_Comm    comm;

  PetscFunctionBegin; 
  /*
      This is nasty. If this is called from an originally parallel matrix
   then all processes call this,but only the first has the matrix so the
   rest should return immediately.
  */
  ierr = PetscObjectGetComm((PetscObject)draw,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (rank) PetscFunctionReturn(0);

  ierr = PetscObjectQuery((PetscObject)A,"Zoomviewer",(PetscObject*)&viewer);CHKERRQ(ierr); 
  ierr = ViewerGetFormat(viewer,&format);CHKERRQ(ierr);

  ierr = DrawGetCoordinates(draw,&xl,&yl,&xr,&yr);CHKERRQ(ierr);
  /* loop over matrix elements drawing boxes */

  if (format != VIEWER_FORMAT_DRAW_CONTOUR) {
    /* Blue for negative, Cyan for zero and  Red for positive */
    color = DRAW_BLUE;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(a->a[j]) >=  0.) continue;
#else
        if (a->a[j] >=  0.) continue;
#endif
        ierr = DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
    color = DRAW_CYAN;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        if (a->a[j] !=  0.) continue;
        ierr = DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
    color = DRAW_RED;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
#if defined(PETSC_USE_COMPLEX)
        if (PetscRealPart(a->a[j]) <=  0.) continue;
#else
        if (a->a[j] <=  0.) continue;
#endif
        ierr = DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
  } else {
    /* use contour shading to indicate magnitude of values */
    /* first determine max of all nonzero values */
    int    nz = a->nz,count;
    Draw   popup;
    PetscReal scale;

    for (i=0; i<nz; i++) {
      if (PetscAbsScalar(a->a[i]) > maxv) maxv = PetscAbsScalar(a->a[i]);
    }
    scale = (245.0 - DRAW_BASIC_COLORS)/maxv; 
    ierr  = DrawGetPopup(draw,&popup);CHKERRQ(ierr);
    if (popup) {ierr  = DrawScalePopup(popup,0.0,maxv);CHKERRQ(ierr);}
    count = 0;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        x_l = a->j[j] + shift; x_r = x_l + 1.0;
        color = DRAW_BASIC_COLORS + (int)(scale*PetscAbsScalar(a->a[count]));
        ierr  = DrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
        count++;
      } 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ_Draw"
int MatView_SeqAIJ_Draw(Mat A,Viewer viewer)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr;
  Draw       draw;
  PetscReal  xr,yr,xl,yl,h,w;
  PetscTruth isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull);CHKERRQ(ierr);
  if (isnull) PetscFunctionReturn(0);

  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",(PetscObject)viewer);CHKERRQ(ierr);
  xr  = a->n; yr = a->m; h = yr/10.0; w = xr/10.0; 
  xr += w;    yr += h;  xl = -w;     yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);
  ierr = DrawZoom(draw,MatView_SeqAIJ_Draw_Zoom,A);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"Zoomviewer",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqAIJ"
int MatView_SeqAIJ(Mat A,Viewer viewer)
{
  Mat_SeqAIJ  *a = (Mat_SeqAIJ*)A->data;
  int         ierr;
  PetscTruth  issocket,isascii,isbinary,isdraw;

  PetscFunctionBegin;  
  ierr = PetscTypeCompare((PetscObject)viewer,SOCKET_VIEWER,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,BINARY_VIEWER,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,DRAW_VIEWER,&isdraw);CHKERRQ(ierr);
  if (issocket) {
    ierr = ViewerSocketPutSparse_Private(viewer,a->m,a->n,a->nz,a->a,a->i,a->j);CHKERRQ(ierr);
  } else if (isascii) {
    ierr = MatView_SeqAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported by SeqAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

extern int Mat_AIJ_CheckInode(Mat);
#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_SeqAIJ"
int MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax,ierr;
  int        m = a->m,*ip,N,*ailen = a->ilen,shift = a->indexshift,rmax = 0;
  Scalar     *aa = a->a,*ap;

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
    PLogObjectMemory(A,-(m+1)*sizeof(int));
    a->diag = 0;
  } 
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Matrix size: %d X %d; storage space: %d unneeded,%d used\n",m,a->n,fshift,a->nz);
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Number of mallocs during MatSetValues() is %d\n",a->reallocs);
  PLogInfo(A,"MatAssemblyEnd_SeqAIJ:Most nonzeros in any row is %d\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift;
  a->rmax              = rmax;

  /* check out for identical nodes. If found, use inode functions */
  ierr = Mat_AIJ_CheckInode(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_SeqAIJ"
int MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        ierr;

  PetscFunctionBegin;  
  ierr = PetscMemzero(a->a,(a->i[a->m]+a->indexshift)*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqAIJ"
int MatDestroy_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        ierr;

  PetscFunctionBegin;  

  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping);CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping);CHKERRQ(ierr);
  }
  if (A->rmap) {
    ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
  }
  if (A->cmap) {
    ierr = MapDestroy(A->cmap);CHKERRQ(ierr);
  }
  if (a->idiag) {ierr = PetscFree(a->idiag);CHKERRQ(ierr);}
#if defined(PETSC_USE_LOG)
  PLogObjectState((PetscObject)A,"Rows=%d, Cols=%d, NZ=%d",a->m,a->n,a->nz);
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
  if (a->solve_work) {ierr = PetscFree(a->solve_work);CHKERRQ(ierr);}
  if (a->inode.size) {ierr = PetscFree(a->inode.size);CHKERRQ(ierr);}
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);}
  if (a->saved_values) {ierr = PetscFree(a->saved_values);CHKERRQ(ierr);}
  ierr = PetscFree(a);CHKERRQ(ierr);

  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatCompress_SeqAIJ"
int MatCompress_SeqAIJ(Mat A)
{
  PetscFunctionBegin;  
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqAIJ"
int MatSetOption_SeqAIJ(Mat A,MatOption op)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;  
  if      (op == MAT_ROW_ORIENTED)                 a->roworiented       = PETSC_TRUE;
  else if (op == MAT_KEEP_ZEROED_ROWS)             a->keepzeroedrows    = PETSC_TRUE;
  else if (op == MAT_COLUMN_ORIENTED)              a->roworiented       = PETSC_FALSE;
  else if (op == MAT_COLUMNS_SORTED)               a->sorted            = PETSC_TRUE;
  else if (op == MAT_COLUMNS_UNSORTED)             a->sorted            = PETSC_FALSE;
  else if (op == MAT_NO_NEW_NONZERO_LOCATIONS)     a->nonew             = 1;
  else if (op == MAT_NEW_NONZERO_LOCATION_ERR)     a->nonew             = -1;
  else if (op == MAT_NEW_NONZERO_ALLOCATION_ERR)   a->nonew             = -2;
  else if (op == MAT_YES_NEW_NONZERO_LOCATIONS)    a->nonew             = 0;
  else if (op == MAT_IGNORE_ZERO_ENTRIES)          a->ignorezeroentries = PETSC_TRUE;
  else if (op == MAT_USE_INODES)                   a->inode.use         = PETSC_TRUE;
  else if (op == MAT_DO_NOT_USE_INODES)            a->inode.use         = PETSC_FALSE;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED ||
           op == MAT_SYMMETRIC ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_YES_NEW_DIAGONALS ||
           op == MAT_IGNORE_OFF_PROC_ENTRIES||
           op == MAT_USE_HASH_TABLE)
    PLogInfo(A,"MatSetOption_SeqAIJ:Option ignored\n");
  else if (op == MAT_NO_NEW_DIAGONALS) {
    SETERRQ(PETSC_ERR_SUP,0,"MAT_NO_NEW_DIAGONALS");
  } else if (op == MAT_INODE_LIMIT_1)          a->inode.limit  = 1;
  else if (op == MAT_INODE_LIMIT_2)            a->inode.limit  = 2;
  else if (op == MAT_INODE_LIMIT_3)            a->inode.limit  = 3;
  else if (op == MAT_INODE_LIMIT_4)            a->inode.limit  = 4;
  else if (op == MAT_INODE_LIMIT_5)            a->inode.limit  = 5;
  else SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqAIJ"
int MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        i,j,n,shift = a->indexshift,ierr;
  Scalar     *x,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming matrix and vector");
  for (i=0; i<a->m; i++) {
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

/* -------------------------------------------------------*/
/* Should check that shapes of vectors and matrices match */
/* -------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "MatMultTranspose_SeqAIJ"
int MatMultTranspose_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *x,*y,*v,alpha,zero = 0.0;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*a->nz - a->n);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultTransposeAdd_SeqAIJ"
int MatMultTransposeAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *x,*y,*v,alpha;
  int        ierr,m = a->m,n,i,*idx,shift = a->indexshift;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  y = y + shift; /* shift for Fortran start by 1 indexing */
  for (i=0; i<m; i++) {
    idx   = a->j + a->i[i] + shift;
    v     = a->a + a->i[i] + shift;
    n     = a->i[i+1] - a->i[i];
    alpha = x[i];
    while (n-->0) {y[*idx++] += alpha * *v++;}
  }
  PLogFlops(2*a->nz);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMult_SeqAIJ"
int MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *x,*y,*v,sum;
  int        ierr,m = a->m,*idx,shift = a->indexshift,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  int        n,i,jrow,j;
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
  PLogFlops(2*a->nz - m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatMultAdd_SeqAIJ"
int MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *x,*y,*z,*v,sum;
  int        ierr,m = a->m,*idx,shift = a->indexshift,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
  int        n,i,jrow,j;
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
  PLogFlops(2*a->nz);
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
#undef __FUNC__  
#define __FUNC__ "MatMarkDiagonal_SeqAIJ"
int MatMarkDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        i,j,*diag,m = a->m,shift = a->indexshift;

  PetscFunctionBegin;
  if (a->diag) PetscFunctionReturn(0);

  diag = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(diag);
  PLogObjectMemory(A,(m+1)*sizeof(int));
  for (i=0; i<a->m; i++) {
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
#undef __FUNC__  
#define __FUNC__ "MatMissingDiagonal_SeqAIJ"
int MatMissingDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  int        *diag,*jj = a->j,i,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  diag = a->diag;
  for (i=0; i<a->m; i++) {
    if (jj[diag[i]+shift] != i-shift) {
      SETERRQ1(1,1,"Matrix is missing diagonal number %d",i);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRelax_SeqAIJ"
int MatRelax_SeqAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,int its,Vec xx)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *x,*b,*bs, d,*xs,sum,*v = a->a,*t=0,scale,*ts,*xb,*idiag=0;
  int        ierr,*idx,*diag,n = a->n,m = a->m,i,shift = a->indexshift;

  PetscFunctionBegin;
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
	PLogFlops(2*n-1);
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
      a->idiag = (Scalar*)PetscMalloc(2*m*sizeof(Scalar));CHKPTRQ(a->idiag);
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
    SETERRQ(PETSC_ERR_SUP,0,"SOR_APPLY_LOWER is not done");
  } else if ((flag & SOR_EISENSTAT) && omega == 1.0 && shift == 0 && fshift == 0.0) {
    /* special case for omega = 1.0 saves flops and some integer ops */
    Scalar *v2;
 
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

    PLogFlops(3*m-1 + 2*a->nz);
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

    PLogFlops(6*m-1 + 2*a->nz);
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]+shift];
        n    = diag[i] - a->i[i];
	PLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
      xb = x;
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=0; i<m; i++) {
        x[i] *= a->a[diag[i]+shift];
      }
      PLogFlops(m);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (i=m-1; i>=0; i--) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - diag[i] - 1;
	PLogFlops(2*n-1);
        idx  = a->j + diag[i] + (!shift);
        v    = a->a + diag[i] + (!shift);
        sum  = xb[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = omega*(sum/d);
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
      for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]+shift];
        n    = a->i[i+1] - a->i[i]; 
	PLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (i=m-1; i>=0; i--) {
        d    = fshift + a->a[diag[i] + shift];
        n    = a->i[i+1] - a->i[i]; 
	PLogFlops(2*n-1);
        idx  = a->j + a->i[i] + shift;
        v    = a->a + a->i[i] + shift;
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + omega*(sum + a->a[diag[i]+shift]*x[i])/d;
      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  if (bb != xx) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
} 

#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqAIJ"
int MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  info->rows_global    = (double)a->m;
  info->columns_global = (double)a->n;
  info->rows_local     = (double)a->m;
  info->columns_local  = (double)a->n;
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

extern int MatLUFactorSymbolic_SeqAIJ(Mat,IS,IS,PetscReal,Mat*);
extern int MatLUFactorNumeric_SeqAIJ(Mat,Mat*);
extern int MatLUFactor_SeqAIJ(Mat,IS,IS,PetscReal);
extern int MatSolve_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveAdd_SeqAIJ(Mat,Vec,Vec,Vec);
extern int MatSolveTranspose_SeqAIJ(Mat,Vec,Vec);
extern int MatSolveTransposeAdd_SeqAIJ(Mat,Vec,Vec,Vec);

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqAIJ"
int MatZeroRows_SeqAIJ(Mat A,IS is,Scalar *diag)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int         i,ierr,N,*rows,m = a->m - 1,shift = a->indexshift;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  if (a->keepzeroedrows) {
    for (i=0; i<N; i++) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
      ierr = PetscMemzero(&a->a[a->i[rows[i]]+shift],a->ilen[rows[i]]*sizeof(Scalar));CHKERRQ(ierr);
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
        if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
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
        if (rows[i] < 0 || rows[i] > m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
        a->ilen[rows[i]] = 0; 
      }
    }
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqAIJ"
int MatGetSize_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (m) *m = a->m; 
  if (n) *n = a->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqAIJ"
int MatGetOwnershipRange_SeqAIJ(Mat A,int *m,int *n)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;

  PetscFunctionBegin;
  if (m) *m = 0;
  if (n) *n = a->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqAIJ"
int MatGetRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        *itmp,i,shift = a->indexshift;

  PetscFunctionBegin;
  if (row < 0 || row >= a->m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Row out of range");

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row] + shift;
  if (idx) {
    itmp = a->j + a->i[row] + shift;
    if (*nz && shift) {
      *idx = (int*)PetscMalloc((*nz)*sizeof(int));CHKPTRQ(*idx);
      for (i=0; i<(*nz); i++) {(*idx)[i] = itmp[i] + shift;}
    } else if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqAIJ"
int MatRestoreRow_SeqAIJ(Mat A,int row,int *nz,int **idx,Scalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int ierr;

  PetscFunctionBegin;
  if (idx) {if (*idx && a->indexshift) {ierr = PetscFree(*idx);CHKERRQ(ierr);}}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqAIJ"
int MatNorm_SeqAIJ(Mat A,NormType type,PetscReal *norm)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *v = a->a;
  PetscReal  sum = 0.0;
  int        i,j,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (i=0; i<a->nz; i++) {
#if defined(PETSC_USE_COMPLEX)
      sum += PetscRealPart(PetscConj(*v)*(*v)); v++;
#else
      sum += (*v)*(*v); v++;
#endif
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) {
    PetscReal *tmp;
    int    *jj = a->j;
    tmp = (PetscReal*)PetscMalloc((a->n+1)*sizeof(PetscReal));CHKPTRQ(tmp);
    ierr = PetscMemzero(tmp,a->n*sizeof(PetscReal));CHKERRQ(ierr);
    *norm = 0.0;
    for (j=0; j<a->nz; j++) {
        tmp[*jj++ + shift] += PetscAbsScalar(*v);  v++;
    }
    for (j=0; j<a->n; j++) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *norm = 0.0;
    for (j=0; j<a->m; j++) {
      v = a->a + a->i[j] + shift;
      sum = 0.0;
      for (i=0; i<a->i[j+1]-a->i[j]; i++) {
        sum += PetscAbsScalar(*v); v++;
      }
      if (sum > *norm) *norm = sum;
    }
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqAIJ"
int MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Mat        C;
  int        i,ierr,*aj = a->j,*ai = a->i,m = a->m,len,*col;
  int        shift = a->indexshift,refct;
  Scalar     *array = a->a;

  PetscFunctionBegin;
  if (!B && m != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Square matrix only for in-place");
  col  = (int*)PetscMalloc((1+a->n)*sizeof(int));CHKPTRQ(col);
  ierr = PetscMemzero(col,(1+a->n)*sizeof(int));CHKERRQ(ierr);
  if (shift) {
    for (i=0; i<ai[m]-1; i++) aj[i] -= 1;
  }
  for (i=0; i<ai[m]+shift; i++) col[aj[i]] += 1;
  ierr = MatCreateSeqAIJ(A->comm,a->n,m,0,col,&C);CHKERRQ(ierr);
  ierr = PetscFree(col);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    len = ai[i+1]-ai[i];
    ierr = MatSetValues(C,len,aj,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    array += len; aj += len;
  }
  if (shift) { 
    for (i=0; i<ai[m]-1; i++) aj[i] += 1;
  }

  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  if (B) {
    *B = C;
  } else {
    PetscOps *Abops;
    MatOps   Aops;

    /* This isn't really an in-place transpose */
    ierr = PetscFree(a->a);CHKERRQ(ierr);
    if (!a->singlemalloc) {
      ierr = PetscFree(a->i);CHKERRQ(ierr);
      ierr = PetscFree(a->j);
    }
    if (a->diag) {ierr = PetscFree(a->diag);CHKERRQ(ierr);}
    if (a->ilen) {ierr = PetscFree(a->ilen);CHKERRQ(ierr);}
    if (a->imax) {ierr = PetscFree(a->imax);CHKERRQ(ierr);}
    if (a->solve_work) {ierr = PetscFree(a->solve_work);CHKERRQ(ierr);}
    if (a->inode.size) {ierr = PetscFree(a->inode.size);CHKERRQ(ierr);}
    ierr = PetscFree(a);CHKERRQ(ierr);
 

    ierr = MapDestroy(A->rmap);CHKERRQ(ierr);
    ierr = MapDestroy(A->cmap);CHKERRQ(ierr);

    /*
      This is horrible, horrible code. We need to keep the 
      the bops and ops Structures, copy everything from C
      including the function pointers..
    */
    ierr     = PetscMemcpy(A->ops,C->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
    ierr     = PetscMemcpy(A->bops,C->bops,sizeof(PetscOps));CHKERRQ(ierr);
    Abops    = A->bops;
    Aops     = A->ops;
    refct    = A->refct;
    ierr     = PetscMemcpy(A,C,sizeof(struct _p_Mat));CHKERRQ(ierr);
    A->bops  = Abops;
    A->ops   = Aops;
    A->qlist = 0;
    A->refct = refct;
    /* copy over the type_name and name */
    ierr     = PetscStrallocpy(C->type_name,&A->type_name);CHKERRQ(ierr);
    ierr     = PetscStrallocpy(C->name,&A->name);CHKERRQ(ierr);

    PetscHeaderDestroy(C);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_SeqAIJ"
int MatDiagonalScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *l,*r,x,*v;
  int        ierr,i,j,m = a->m,n = a->n,M,nz = a->nz,*jj,shift = a->indexshift;

  PetscFunctionBegin;
  if (ll) {
    /* The local size is used so that VecMPI can be passed to this routine
       by MatDiagonalScale_MPIAIJ */
    ierr = VecGetLocalSize(ll,&m);CHKERRQ(ierr);
    if (m != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Left scaling vector wrong length");
    ierr = VecGetArray(ll,&l);CHKERRQ(ierr);
    v = a->a;
    for (i=0; i<m; i++) {
      x = l[i];
      M = a->i[i+1] - a->i[i];
      for (j=0; j<M; j++) { (*v++) *= x;} 
    }
    ierr = VecRestoreArray(ll,&l);CHKERRQ(ierr);
    PLogFlops(nz);
  }
  if (rr) {
    ierr = VecGetLocalSize(rr,&n);CHKERRQ(ierr);
    if (n != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Right scaling vector wrong length");
    ierr = VecGetArray(rr,&r);CHKERRQ(ierr); 
    v = a->a; jj = a->j;
    for (i=0; i<nz; i++) {
      (*v++) *= r[*jj++ + shift]; 
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr); 
    PLogFlops(nz);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqAIJ"
int MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,int csize,MatReuse scall,Mat *B)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data,*c;
  int          *smap,i,k,kstart,kend,ierr,oldcols = a->n,*lens;
  int          row,mat_i,*mat_j,tcol,first,step,*mat_ilen,sum,lensi;
  int          *irow,*icol,nrows,ncols,shift = a->indexshift,*ssmap;
  int          *starts,*j_new,*i_new,*aj = a->j,*ai = a->i,ii,*ailen = a->ilen;
  Scalar       *a_new,*mat_a;
  Mat          C;
  PetscTruth   stride;

  PetscFunctionBegin;
  ierr = ISSorted(isrow,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"ISrow is not sorted");
  ierr = ISSorted(iscol,(PetscTruth*)&i);CHKERRQ(ierr);
  if (!i) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"IScol is not sorted");

  ierr = ISGetIndices(isrow,&irow);CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&nrows);CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&ncols);CHKERRQ(ierr);

  ierr = ISStrideGetInfo(iscol,&first,&step);CHKERRQ(ierr);
  ierr = ISStride(iscol,&stride);CHKERRQ(ierr);
  if (stride && step == 1) { 
    /* special case of contiguous rows */
    lens   = (int*)PetscMalloc((ncols+nrows+1)*sizeof(int));CHKPTRQ(lens);
    starts = lens + ncols;
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
      if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Reused submatrix wrong size");
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
      ierr = PetscMemcpy(a_new,a->a + starts[i],lensi*sizeof(Scalar));CHKERRQ(ierr);
      a_new      += lensi;
      i_new[i+1]  = i_new[i] + lensi;
      c->ilen[i]  = lensi;
    }
    ierr = PetscFree(lens);CHKERRQ(ierr);
  } else {
    ierr  = ISGetIndices(iscol,&icol);CHKERRQ(ierr);
    smap  = (int*)PetscMalloc((1+oldcols)*sizeof(int));CHKPTRQ(smap);
    ssmap = smap + shift;
    lens  = (int*)PetscMalloc((1+nrows)*sizeof(int));CHKPTRQ(lens);
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
      if (c->m  != nrows || c->n != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(c->ilen,lens,c->m*sizeof(int),&equal);CHKERRQ(ierr);
      if (!equal) {
        SETERRQ(PETSC_ERR_ARG_SIZ,0,"Cannot reuse matrix. wrong no of nonzeros");
      }
      ierr = PetscMemzero(c->ilen,c->m*sizeof(int));CHKERRQ(ierr);
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
#undef __FUNC__  
#define __FUNC__ "MatILUFactor_SeqAIJ"
int MatILUFactor_SeqAIJ(Mat inA,IS row,IS col,MatILUInfo *info)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)inA->data;
  int        ierr;
  Mat        outA;
  PetscTruth row_identity,col_identity;

  PetscFunctionBegin;
  if (info && info->levels != 0) SETERRQ(PETSC_ERR_SUP,0,"Only levels=0 supported for in-place ilu");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(1,1,"Row and column permutations must be identity for in-place ILU");
  }

  outA          = inA; 
  inA->factor   = FACTOR_LU;
  a->row        = row;
  a->col        = col;
  ierr = PetscObjectReference((PetscObject)row);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)col);CHKERRQ(ierr);

  /* Create the inverse permutation so that it can be used in MatLUFactorNumeric() */
  if (a->icol) {ierr = ISDestroy(a->icol);CHKERRQ(ierr);} /* if this came from a previous factored; need to remove old one */
  ierr = ISInvertPermutation(col,PETSC_DECIDE,&a->icol);CHKERRQ(ierr);
  PLogObjectParent(inA,a->icol);

  if (!a->solve_work) { /* this matrix may have been factored before */
    a->solve_work = (Scalar*)PetscMalloc((a->m+1)*sizeof(Scalar));CHKPTRQ(a->solve_work);
  }

  if (!a->diag) {
    ierr = MatMarkDiagonal_SeqAIJ(inA);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric_SeqAIJ(inA,&outA);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "pinclude/blaslapack.h"
#undef __FUNC__  
#define __FUNC__ "MatScale_SeqAIJ"
int MatScale_SeqAIJ(Scalar *alpha,Mat inA)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)inA->data;
  int        one = 1;

  PetscFunctionBegin;
  BLscal_(&a->nz,alpha,a->a,&one);
  PLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqAIJ"
int MatGetSubMatrices_SeqAIJ(Mat A,int n,IS *irow,IS *icol,MatReuse scall,Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat*)PetscMalloc((n+1)*sizeof(Mat));CHKPTRQ(*B);
  }

  for (i=0; i<n; i++) {
    ierr = MatGetSubMatrix_SeqAIJ(A,irow[i],icol[i],PETSC_DECIDE,scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqAIJ"
int MatGetBlockSize_SeqAIJ(Mat A,int *bs)
{
  PetscFunctionBegin;
  *bs = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatIncreaseOverlap_SeqAIJ"
int MatIncreaseOverlap_SeqAIJ(Mat A,int is_max,IS *is,int ov)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  int        shift,row,i,j,k,l,m,n,*idx,ierr,*nidx,isz,val;
  int        start,end,*ai,*aj;
  PetscBT    table;

  PetscFunctionBegin;
  shift = a->indexshift;
  m     = a->m;
  ai    = a->i;
  aj    = a->j+shift;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"illegal overlap value used");

  nidx  = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(nidx); 
  ierr  = PetscBTCreate(m,table);CHKERRQ(ierr);

  for (i=0; i<is_max; i++) {
    /* Initialize the two local arrays */
    isz  = 0;
    ierr = PetscBTMemzero(m,table);CHKERRQ(ierr);
                 
    /* Extract the indices, assume there can be duplicate entries */
    ierr = ISGetIndices(is[i],&idx);CHKERRQ(ierr);
    ierr = ISGetSize(is[i],&n);CHKERRQ(ierr);
    
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
#undef __FUNC__  
#define __FUNC__ "MatPermute_SeqAIJ"
int MatPermute_SeqAIJ(Mat A,IS rowp,IS colp,Mat *B)
{ 
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  Scalar     *vwork;
  int        i,ierr,nz,m = a->m,n = a->n,*cwork;
  int        *row,*col,*cnew,j,*lens;
  IS         icolp,irowp;

  PetscFunctionBegin;
  ierr = ISInvertPermutation(rowp,PETSC_DECIDE,&irowp);CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&row);CHKERRQ(ierr);
  ierr = ISInvertPermutation(colp,PETSC_DECIDE,&icolp);CHKERRQ(ierr);
  ierr = ISGetIndices(icolp,&col);CHKERRQ(ierr);
  
  /* determine lengths of permuted rows */
  lens = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(lens);
  for (i=0; i<m; i++) {
    lens[row[i]] = a->i[i+1] - a->i[i];
  }
  ierr = MatCreateSeqAIJ(A->comm,m,n,0,lens,B);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  cnew = (int*)PetscMalloc(n*sizeof(int));CHKPTRQ(cnew);
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

#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_SeqAIJ"
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
  PetscFunctionReturn(0);
}
extern int MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg);
extern int MatFDColoringCreate_SeqAIJ(Mat,ISColoring,MatFDColoring);
extern int MatColoringPatch_SeqAIJ(Mat,int,int *,ISColoring *);
extern int MatILUDTFactor_SeqAIJ(Mat,MatILUInfo*,IS,IS,Mat*);
#undef __FUNC__  
#define __FUNC__ "MatCopy_SeqAIJ"
int MatCopy_SeqAIJ(Mat A,Mat B,MatStructure str)
{
  int    ierr;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN && B->type == MATSEQAIJ) {
    Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)B->data; 

    if (a->i[a->m]+a->indexshift != b->i[b->m]+a->indexshift) {
      SETERRQ(1,1,"Number of nonzeros in two matrices are different");
    }
    ierr = PetscMemcpy(b->a,a->a,(a->i[a->m]+a->indexshift)*sizeof(Scalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
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
       MatGetSize_SeqAIJ,
       MatGetSize_SeqAIJ,
       MatGetOwnershipRange_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,
       0,
       0,
       0,
       MatDuplicate_SeqAIJ,
       0,
       0,
       MatILUFactor_SeqAIJ,
       0,
       0,
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
       MatColoringPatch_SeqAIJ,
       0,
       MatPermute_SeqAIJ,
       0,
       0,
       0,
       0,
       MatGetMaps_Petsc};

extern int MatUseSuperLU_SeqAIJ(Mat);
extern int MatUseEssl_SeqAIJ(Mat);
extern int MatUseDXML_SeqAIJ(Mat);

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatSeqAIJSetColumnIndices_SeqAIJ"

int MatSeqAIJSetColumnIndices_SeqAIJ(Mat mat,int *indices)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        i,nz,n;

  PetscFunctionBegin;
  if (aij->indexshift) SETERRQ(1,1,"No support with 1 based indexing");

  nz = aij->maxnz;
  n  = aij->n;
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

#undef __FUNC__  
#define __FUNC__ "MatSeqAIJSetColumnIndices"
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

@*/ 
int MatSeqAIJSetColumnIndices(Mat mat,int *indices)
{
  int ierr,(*f)(Mat,int *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqAIJSetColumnIndices_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatStoreValues_SeqAIJ"
int MatStoreValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        nz = aij->i[aij->m]+aij->indexshift,ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(1,1,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
  }

  /* allocate space for values if not already there */
  if (!aij->saved_values) {
    aij->saved_values = (Scalar*)PetscMalloc(nz*sizeof(Scalar));CHKPTRQ(aij->saved_values);
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->saved_values,aij->a,nz*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatStoreValues"
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
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatStoreValues_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Wrong type of matrix to store values");
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "MatRetrieveValues_SeqAIJ"
int MatRetrieveValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  int        nz = aij->i[aij->m]+aij->indexshift,ierr;

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(1,1,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
  }
  if (!aij->saved_values) {
    SETERRQ(1,1,"Must call MatStoreValues(A);first");
  }

  /* copy values over */
  ierr = PetscMemcpy(aij->a,aij->saved_values,nz*sizeof(Scalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "MatRetrieveValues"
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
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatRetrieveValues_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Wrong type of matrix to retrieve values");
  }
  PetscFunctionReturn(0);
}

/* --------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqAIJ"
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
  Mat        B;
  Mat_SeqAIJ *b;
  int        i,len,ierr,size;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"Comm must be of size 1");

  if (nz < -2) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,0,"nz cannot be less than -2: value %d",nz);
  if (nnz) {
    for (i=0; i<m; i++) {
      if (nnz[i] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,0,"nnz cannot be less than 0: local row %d value %d",i,nnz[i]);
    }
  }

  *A                  = 0;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQAIJ,"Mat",comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data             = (void*)(b = PetscNew(Mat_SeqAIJ));CHKPTRQ(b);
  ierr = PetscMemzero(b,sizeof(Mat_SeqAIJ));CHKERRQ(ierr);
  ierr = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->ops->destroy          = MatDestroy_SeqAIJ;
  B->ops->view             = MatView_SeqAIJ;
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = OptionsGetDouble(PETSC_NULL,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-pc_ilu_preserve_row_sums",&b->ilu_preserve_row_sums);CHKERRQ(ierr);
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->indexshift       = 0;
  b->reallocs         = 0;
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_oneindex",&flg);CHKERRQ(ierr);
  if (flg) b->indexshift = -1;
  
  b->m = m; B->m = m; B->M = m;
  b->n = n; B->n = n; B->N = n;

  ierr = MapCreateMPI(comm,m,m,&B->rmap);CHKERRQ(ierr);
  ierr = MapCreateMPI(comm,n,n,&B->cmap);CHKERRQ(ierr);

  b->imax = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(b->imax);
  if (!nnz) {
    if (nz == PETSC_DEFAULT) nz = 10;
    else if (nz <= 0)        nz = 1;
    for (i=0; i<m; i++) b->imax[i] = nz;
    nz = nz*m;
  } else {
    nz = 0;
    for (i=0; i<m; i++) {b->imax[i] = nnz[i]; nz += nnz[i];}
  }

  /* allocate the matrix space */
  len             = nz*(sizeof(int) + sizeof(Scalar)) + (b->m+1)*sizeof(int);
  b->a            = (Scalar*)PetscMalloc(len);CHKPTRQ(b->a);
  b->j            = (int*)(b->a + nz);
  ierr            = PetscMemzero(b->j,nz*sizeof(int));CHKERRQ(ierr);
  b->i            = b->j + nz;
  b->singlemalloc = PETSC_TRUE;
  b->freedata     = PETSC_TRUE;

  b->i[0] = -b->indexshift;
  for (i=1; i<m+1; i++) {
    b->i[i] = b->i[i-1] + b->imax[i-1];
  }

  /* b->ilen will count nonzeros in each row so far. */
  b->ilen = (int*)PetscMalloc((m+1)*sizeof(int)); 
  PLogObjectMemory(B,len+2*(m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));
  for (i=0; i<b->m; i++) { b->ilen[i] = 0;}

  b->nz                = 0;
  b->maxnz             = nz;
  b->sorted            = PETSC_FALSE;
  b->ignorezeroentries = PETSC_FALSE;
  b->roworiented       = PETSC_TRUE;
  b->nonew             = 0;
  b->diag              = 0;
  b->solve_work        = 0;
  b->spptr             = 0;
  b->inode.use         = PETSC_TRUE;
  b->inode.node_count  = 0;
  b->inode.size        = 0;
  b->inode.limit       = 5;
  b->inode.max_limit   = 5;
  b->saved_values      = 0;
  B->info.nz_unneeded  = (double)b->maxnz;
  b->idiag             = 0;
  b->ssor              = 0;
  b->keepzeroedrows    = PETSC_FALSE;

  *A = B;

  /*  SuperLU is not currently supported through PETSc */
#if defined(PETSC_HAVE_SUPERLU)
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_superlu",&flg);CHKERRQ(ierr);
  if (flg) { ierr = MatUseSuperLU_SeqAIJ(B);CHKERRQ(ierr); }
#endif
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_essl",&flg);CHKERRQ(ierr);
  if (flg) { ierr = MatUseEssl_SeqAIJ(B);CHKERRQ(ierr); }
  ierr = OptionsHasName(PETSC_NULL,"-mat_aij_dxml",&flg);CHKERRQ(ierr);
  if (flg) {
    if (!b->indexshift) SETERRQ(PETSC_ERR_LIB,0,"need -mat_aij_oneindex with -mat_aij_dxml");
    ierr = MatUseDXML_SeqAIJ(B);CHKERRQ(ierr);
  }
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg);CHKERRQ(ierr);
  if (flg) {ierr = MatPrintHelp(B);CHKERRQ(ierr); }

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetColumnIndices_C",
                                     "MatSeqAIJSetColumnIndices_SeqAIJ",
                                     (void*)MatSeqAIJSetColumnIndices_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqAIJ",
                                     (void*)MatStoreValues_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqAIJ",
                                     (void*)MatRetrieveValues_SeqAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDuplicate_SeqAIJ"
int MatDuplicate_SeqAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat        C;
  Mat_SeqAIJ *c,*a = (Mat_SeqAIJ*)A->data;
  int        i,len,m = a->m,shift = a->indexshift,ierr;

  PetscFunctionBegin;
  *B = 0;
  PetscHeaderCreate(C,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQAIJ,"Mat",A->comm,MatDestroy,MatView);
  PLogObjectCreate(C);
  C->data           = (void*)(c = PetscNew(Mat_SeqAIJ));CHKPTRQ(c);
  ierr              = PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  C->ops->destroy   = MatDestroy_SeqAIJ;
  C->ops->view      = MatView_SeqAIJ;
  C->factor         = A->factor;
  c->row            = 0;
  c->col            = 0;
  c->icol           = 0;
  c->indexshift     = shift;
  c->keepzeroedrows = a->keepzeroedrows;
  C->assembled      = PETSC_TRUE;

  c->m = C->m   = a->m;
  c->n = C->n   = a->n;
  C->M          = a->m;
  C->N          = a->n;

  c->imax       = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(c->imax);
  c->ilen       = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(c->ilen);
  for (i=0; i<m; i++) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = PETSC_TRUE;
  len     = (m+1)*sizeof(int)+(a->i[m])*(sizeof(Scalar)+sizeof(int));
  c->a  = (Scalar*)PetscMalloc(len);CHKPTRQ(c->a);
  c->j  = (int*)(c->a + a->i[m] + shift);
  c->i  = c->j + a->i[m] + shift;
  ierr = PetscMemcpy(c->i,a->i,(m+1)*sizeof(int));CHKERRQ(ierr);
  if (m > 0) {
    ierr = PetscMemcpy(c->j,a->j,(a->i[m]+shift)*sizeof(int));CHKERRQ(ierr);
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscMemcpy(c->a,a->a,(a->i[m]+shift)*sizeof(Scalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(c->a,(a->i[m]+shift)*sizeof(Scalar));CHKERRQ(ierr);
    }
  }

  PLogObjectMemory(C,len+2*(m+1)*sizeof(int)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));  
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
    c->diag = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(c->diag);
    PLogObjectMemory(C,(m+1)*sizeof(int));
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag        = 0;
  c->inode.use          = a->inode.use;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    c->inode.size       = (int*)PetscMalloc((m+1)*sizeof(int));CHKPTRQ(c->inode.size);
    c->inode.node_count = a->inode.node_count;
    ierr = PetscMemcpy(c->inode.size,a->inode.size,(m+1)*sizeof(int));CHKERRQ(ierr);
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  c->solve_work         = 0;
  c->spptr              = 0;      /* Dangerous -I'm throwing away a->spptr */

  *B = C;
  ierr = FListDuplicate(A->qlist,&C->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqAIJ"
int MatLoad_SeqAIJ(Viewer viewer,MatType type,Mat *A)
{
  Mat_SeqAIJ   *a;
  Mat          B;
  int          i,nz,ierr,fd,header[4],size,*rowlengths = 0,M,N,shift;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT);CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"not matrix object in file");
  M = header[1]; N = header[2]; nz = header[3];

  if (nz < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format on disk,cannot load as SeqAIJ");
  }

  /* read in row lengths */
  rowlengths = (int*)PetscMalloc(M*sizeof(int));CHKPTRQ(rowlengths);
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

#undef __FUNC__  
#define __FUNC__ "MatEqual_SeqAIJ"
int MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ *)A->data,*b = (Mat_SeqAIJ *)B->data;
  int        ierr;

  PetscFunctionBegin;
  if (B->type !=MATSEQAIJ)SETERRQ(PETSC_ERR_ARG_INCOMP,0,"Matrices must be same type");

  /* If the  matrix dimensions are not equal,or no of nonzeros or shift */
  if ((a->m != b->m) || (a->n !=b->n) ||(a->nz != b->nz)|| 
      (a->indexshift != b->indexshift)) {
    *flg = PETSC_FALSE; PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(a->m+1)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(int),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
  
  /* if a->a are the same */
  ierr = PetscMemcmp(a->a,b->a,(a->nz)*sizeof(Scalar),flg);CHKERRQ(ierr);

  PetscFunctionReturn(0);
  
}

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqAIJWithArrays"
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
int MatCreateSeqAIJWithArrays(MPI_Comm comm,int m,int n,int* i,int*j,Scalar *a,Mat *mat)
{
  int        ierr,ii;
  Mat_SeqAIJ *aij;

  PetscFunctionBegin;
  ierr = MatCreateSeqAIJ(comm,m,n,0,0,mat);CHKERRQ(ierr);
  aij  = (Mat_SeqAIJ*)(*mat)->data;
  ierr = PetscFree(aij->a);CHKERRQ(ierr);

  if (i[0] == 1) {
    aij->indexshift = -1;
  } else if (i[0]) {
    SETERRQ(1,1,"i (row indices) do not start with 0 or 1");
  }  
  aij->i = i;
  aij->j = j;
  aij->a = a;
  aij->singlemalloc = PETSC_FALSE;
  aij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aij->freedata     = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = i[ii+1] - i[ii];
#if defined(PETSC_BOPT_g)
    if (i[ii+1] - i[i] < 0) SETERRQ2(1,1,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
#endif    
  }
#if defined(PETSC_BOPT_g)
  for (ii=0; ii<aij->i[m]; ii++) {
    if (j[ii] < -aij->indexshift) SETERRQ2(1,1,"Negative column index at location = %d index = %d",ii,j[ii]);
    if (j[ii] > n - 1 -aij->indexshift) SETERRQ2(1,1,"Column index to large at location = %d index = %d",ii,j[ii]);
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



