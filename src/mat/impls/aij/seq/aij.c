
/*
    Defines the basic matrix operations for the AIJ (compressed row)
  matrix storage format.
*/

#include "src/mat/impls/aij/seq/aij.h"          /*I "petscmat.h" I*/
#include "src/inline/spops.h"
#include "src/inline/dot.h"
#include "petscbt.h"

#undef __FUNCT__  
#define __FUNCT__ "MatGetRowIJ_SeqAIJ"
PetscErrorCode MatGetRowIJ_SeqAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscInt *m,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,ishift;
 
  PetscFunctionBegin;  
  *m     = A->m;
  if (!ia) PetscFunctionReturn(0);
  ishift = 0;
  if (symmetric && !A->structurally_symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(A->m,a->i,a->j,ishift,oshift,ia,ja);CHKERRQ(ierr);
  } else if (oshift == 1) {
    PetscInt nz = a->i[A->m]; 
    /* malloc space and  add 1 to i and j indices */
    ierr = PetscMalloc((A->m+1)*sizeof(PetscInt),ia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(PetscInt),ja);CHKERRQ(ierr);
    for (i=0; i<nz; i++) (*ja)[i] = a->j[i] + 1;
    for (i=0; i<A->m+1; i++) (*ia)[i] = a->i[i] + 1;
  } else {
    *ia = a->i; *ja = a->j;
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRowIJ_SeqAIJ"
PetscErrorCode MatRestoreRowIJ_SeqAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscInt *n,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscErrorCode ierr;
 
  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);
  if ((symmetric && !A->structurally_symmetric) || oshift == 1) {
    ierr = PetscFree(*ia);CHKERRQ(ierr);
    ierr = PetscFree(*ja);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0); 
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetColumnIJ_SeqAIJ"
PetscErrorCode MatGetColumnIJ_SeqAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscInt *nn,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,*collengths,*cia,*cja,n = A->n,m = A->m;
  PetscInt       nz = a->i[m],row,*jj,mr,col;
 
  PetscFunctionBegin;  
  *nn     = A->n;
  if (!ia) PetscFunctionReturn(0);
  if (symmetric) {
    ierr = MatToSymmetricIJ_SeqAIJ(A->m,a->i,a->j,0,oshift,ia,ja);CHKERRQ(ierr);
  } else {
    ierr = PetscMalloc((n+1)*sizeof(PetscInt),&collengths);CHKERRQ(ierr);
    ierr = PetscMemzero(collengths,n*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc((n+1)*sizeof(PetscInt),&cia);CHKERRQ(ierr);
    ierr = PetscMalloc((nz+1)*sizeof(PetscInt),&cja);CHKERRQ(ierr);
    jj = a->j;
    for (i=0; i<nz; i++) {
      collengths[jj[i]]++;
    }
    cia[0] = oshift;
    for (i=0; i<n; i++) {
      cia[i+1] = cia[i] + collengths[i];
    }
    ierr = PetscMemzero(collengths,n*sizeof(PetscInt));CHKERRQ(ierr);
    jj   = a->j;
    for (row=0; row<m; row++) {
      mr = a->i[row+1] - a->i[row];
      for (i=0; i<mr; i++) {
        col = *jj++;
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
PetscErrorCode MatRestoreColumnIJ_SeqAIJ(Mat A,PetscInt oshift,PetscTruth symmetric,PetscInt *n,PetscInt *ia[],PetscInt *ja[],PetscTruth *done)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  if (!ia) PetscFunctionReturn(0);

  ierr = PetscFree(*ia);CHKERRQ(ierr);
  ierr = PetscFree(*ja);CHKERRQ(ierr);
  
  PetscFunctionReturn(0); 
}

#define CHUNKSIZE   15

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqAIJ"
PetscErrorCode MatSetValues_SeqAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscInt       *rp,k,low,high,t,ii,row,nrow,i,col,l,rmax,N,sorted = a->sorted;
  PetscInt       *imax = a->imax,*ai = a->i,*ailen = a->ilen;
  PetscErrorCode ierr;
  PetscInt       *aj = a->j,nonew = a->nonew;
  PetscScalar    *ap,value,*aa = a->a;
  PetscTruth     ignorezeroentries = ((a->ignorezeroentries && is == ADD_VALUES) ? PETSC_TRUE:PETSC_FALSE);
  PetscTruth     roworiented = a->roworiented;

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over added rows */
    row  = im[k]; 
    if (row < 0) continue;
#if defined(PETSC_USE_DEBUG)  
    if (row >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->m-1);
#endif
    rp   = aj + ai[row]; ap = aa + ai[row];
    rmax = imax[row]; nrow = ailen[row]; 
    low = 0;
    for (l=0; l<n; l++) { /* loop over added columns */
      if (in[l] < 0) continue;
#if defined(PETSC_USE_DEBUG)  
      if (in[l] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->n-1);
#endif
      col = in[l];
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
      else if (nonew == -1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at (%D,%D) in the matrix",row,col);
      if (nrow >= rmax) {
        /* there is no extra room in row, therefore enlarge */
        PetscInt         new_nz = ai[A->m] + CHUNKSIZE,*new_i,*new_j;
        size_t      len;
        PetscScalar *new_a;

        if (nonew == -2) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Inserting a new nonzero at (%D,%D) in the matrix requiring new malloc()",row,col);

        /* malloc new storage space */
        len     = ((size_t) new_nz)*(sizeof(PetscInt)+sizeof(PetscScalar))+(A->m+1)*sizeof(PetscInt);
	ierr    = PetscMalloc(len,&new_a);CHKERRQ(ierr);
        new_j   = (PetscInt*)(new_a + new_nz);
        new_i   = new_j + new_nz;

        /* copy over old data into new slots */
        for (ii=0; ii<row+1; ii++) {new_i[ii] = ai[ii];}
        for (ii=row+1; ii<A->m+1; ii++) {new_i[ii] = ai[ii]+CHUNKSIZE;}
        ierr = PetscMemcpy(new_j,aj,(ai[row]+nrow)*sizeof(PetscInt));CHKERRQ(ierr);
        len  = (((size_t) new_nz) - CHUNKSIZE - ai[row] - nrow );
        ierr = PetscMemcpy(new_j+ai[row]+nrow+CHUNKSIZE,aj+ai[row]+nrow,len*sizeof(PetscInt));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a,aa,(((size_t) ai[row])+nrow)*sizeof(PetscScalar));CHKERRQ(ierr);
        ierr = PetscMemcpy(new_a+ai[row]+nrow+CHUNKSIZE,aa+ai[row]+nrow,len*sizeof(PetscScalar));CHKERRQ(ierr);
        /* free up old matrix storage */
        ierr = PetscFree(a->a);CHKERRQ(ierr);
        if (!a->singlemalloc) {
          ierr = PetscFree(a->i);CHKERRQ(ierr);
          ierr = PetscFree(a->j);CHKERRQ(ierr);
        }
        aa = a->a = new_a; ai = a->i = new_i; aj = a->j = new_j; 
        a->singlemalloc = PETSC_TRUE;

        rp   = aj + ai[row]; ap = aa + ai[row] ;
        rmax = imax[row] = imax[row] + CHUNKSIZE;
        PetscLogObjectMemory(A,CHUNKSIZE*(sizeof(PetscInt) + sizeof(PetscScalar)));
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
  A->same_nonzero = PETSC_FALSE;
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqAIJ"
PetscErrorCode MatGetValues_SeqAIJ(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)A->data;
  PetscInt     *rp,k,low,high,t,row,nrow,i,col,l,*aj = a->j;
  PetscInt     *ai = a->i,*ailen = a->ilen;
  PetscScalar  *ap,*aa = a->a,zero = 0.0;

  PetscFunctionBegin;  
  for (k=0; k<m; k++) { /* loop over rows */
    row  = im[k];   
    if (row < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",row);
    if (row >= A->m) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->m-1);
    rp   = aj + ai[row]; ap = aa + ai[row];
    nrow = ailen[row]; 
    for (l=0; l<n; l++) { /* loop over columns */
      if (in[l] < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative column: %D",in[l]);
      if (in[l] >= A->n) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[l],A->n-1);
      col = in[l] ;
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
PetscErrorCode MatView_SeqAIJ_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,*col_lens;
  int            fd;

  PetscFunctionBegin;  
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);
  ierr = PetscMalloc((4+A->m)*sizeof(PetscInt),&col_lens);CHKERRQ(ierr);
  col_lens[0] = MAT_FILE_COOKIE;
  col_lens[1] = A->m;
  col_lens[2] = A->n;
  col_lens[3] = a->nz;

  /* store lengths of each row and write (including header) to file */
  for (i=0; i<A->m; i++) {
    col_lens[4+i] = a->i[i+1] - a->i[i];
  }
  ierr = PetscBinaryWrite(fd,col_lens,4+A->m,PETSC_INT,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscFree(col_lens);CHKERRQ(ierr);

  /* store column indices (zero start index) */
  ierr = PetscBinaryWrite(fd,a->j,a->nz,PETSC_INT,PETSC_FALSE);CHKERRQ(ierr);

  /* store nonzero values */
  ierr = PetscBinaryWrite(fd,a->a,a->nz,PETSC_SCALAR,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatSeqAIJFactorInfo_Matlab(Mat,PetscViewer);

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_ASCII"
PetscErrorCode MatView_SeqAIJ_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->m,shift=0;
  char              *name;
  PetscViewerFormat format;

  PetscFunctionBegin;  
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL || format == PETSC_VIEWER_ASCII_INFO) {
    if (a->inode.size) {
      ierr = PetscViewerASCIIPrintf(viewer,"using I-node routines: found %D nodes, limit used is %D\n",a->inode.node_count,a->inode.limit);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"not using I-node routines\n");CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    PetscInt nofinalvalue = 0;
    if ((a->i[m] == a->i[m-1]) || (a->j[a->nz-1] != A->n-!shift)) {
      nofinalvalue = 1;
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %D %D \n",m,A->n);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %D \n",a->nz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%D,3);\n",a->nz+nofinalvalue);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);

    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e + %18.16ei \n",i+1,a->j[j]+!shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
#else
        ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",i+1,a->j[j]+!shift,a->a[j]);CHKERRQ(ierr);
#endif
      }
    }
    if (nofinalvalue) {
      ierr = PetscViewerASCIIPrintf(viewer,"%D %D  %18.16e\n",m,A->n,0.0);CHKERRQ(ierr);
    } 
    ierr = PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
     PetscFunctionReturn(0);
  } else if (format == PETSC_VIEWER_ASCII_COMMON) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0 && PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscRealPart(a->a[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        if (a->a[j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);}
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_SYMMODU) {
    PetscInt nzd=0,fshift=1,*sptr;
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = PetscMalloc((m+1)*sizeof(PetscInt),&sptr);CHKERRQ(ierr);
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
    ierr = PetscViewerASCIIPrintf(viewer," %D %D\n\n",m,nzd);CHKERRQ(ierr);
    for (i=0; i<m+1; i+=6) {
      if (i+4<m) {ierr = PetscViewerASCIIPrintf(viewer," %D %D %D %D %D %D\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4],sptr[i+5]);CHKERRQ(ierr);}
      else if (i+3<m) {ierr = PetscViewerASCIIPrintf(viewer," %D %D %D %D %D\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3],sptr[i+4]);CHKERRQ(ierr);}
      else if (i+2<m) {ierr = PetscViewerASCIIPrintf(viewer," %D %D %D %D\n",sptr[i],sptr[i+1],sptr[i+2],sptr[i+3]);CHKERRQ(ierr);}
      else if (i+1<m) {ierr = PetscViewerASCIIPrintf(viewer," %D %D %D\n",sptr[i],sptr[i+1],sptr[i+2]);CHKERRQ(ierr);}
      else if (i<m)   {ierr = PetscViewerASCIIPrintf(viewer," %D %D\n",sptr[i],sptr[i+1]);CHKERRQ(ierr);}
      else            {ierr = PetscViewerASCIIPrintf(viewer," %D\n",sptr[i]);CHKERRQ(ierr);}
    }
    ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
    ierr = PetscFree(sptr);CHKERRQ(ierr);
    for (i=0; i<m; i++) {
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
        if (a->j[j] >= i) {ierr = PetscViewerASCIIPrintf(viewer," %D ",a->j[j]+fshift);CHKERRQ(ierr);}
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
    PetscInt         cnt = 0,jcnt;
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
      ierr = PetscViewerASCIIPrintf(viewer,"row %D:",i);CHKERRQ(ierr);
      for (j=a->i[i]+shift; j<a->i[i+1]+shift; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(a->a[j]) > 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g + %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else if (PetscImaginaryPart(a->a[j]) < 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g - %g i)",a->j[j]+shift,PetscRealPart(a->a[j]),-PetscImaginaryPart(a->a[j]));CHKERRQ(ierr);
        } else {
          ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[j]+shift,PetscRealPart(a->a[j]));CHKERRQ(ierr);
        }
#else
        ierr = PetscViewerASCIIPrintf(viewer," (%D, %g) ",a->j[j]+shift,a->a[j]);CHKERRQ(ierr);
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
PetscErrorCode MatView_SeqAIJ_Draw_Zoom(PetscDraw draw,void *Aa)
{
  Mat               A = (Mat) Aa;
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode    ierr;
  PetscInt          i,j,m = A->m,color;
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
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        x_l = a->j[j] ; x_r = x_l + 1.0;
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
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        x_l = a->j[j]; x_r = x_l + 1.0;
        if (a->a[j] !=  0.) continue;
        ierr = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
      } 
    }
    color = PETSC_DRAW_RED;
    for (i=0; i<m; i++) {
      y_l = m - i - 1.0; y_r = y_l + 1.0;
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        x_l = a->j[j]; x_r = x_l + 1.0;
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
    PetscInt    nz = a->nz,count;
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
      for (j=a->i[i]; j<a->i[i+1]; j++) {
        x_l = a->j[j]; x_r = x_l + 1.0;
        color = PETSC_DRAW_BASIC_COLORS + (PetscInt)(scale*PetscAbsScalar(a->a[count]));
        ierr  = PetscDrawRectangle(draw,x_l,y_l,x_r,y_r,color,color,color,color);CHKERRQ(ierr);
        count++;
      } 
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqAIJ_Draw"
PetscErrorCode MatView_SeqAIJ_Draw(Mat A,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscDraw      draw;
  PetscReal      xr,yr,xl,yl,h,w;
  PetscTruth     isnull;

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
PetscErrorCode MatView_SeqAIJ(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscTruth     issocket,iascii,isbinary,isdraw;

  PetscFunctionBegin;  
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_SOCKET,&issocket);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (issocket) {
    ierr = PetscViewerSocketPutSparse_Private(viewer,A->m,A->n,a->nz,a->a,a->i,a->j);CHKERRQ(ierr);
  } else if (iascii) {
    ierr = MatView_SeqAIJ_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqAIJ_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqAIJ_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported by SeqAIJ matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ"
PetscErrorCode MatAssemblyEnd_SeqAIJ(Mat A,MatAssemblyType mode)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       fshift = 0,i,j,*ai = a->i,*aj = a->j,*imax = a->imax;
  PetscInt       m = A->m,*ip,N,*ailen = a->ilen,rmax = 0;
  PetscScalar    *aa = a->a,*ap;
  PetscReal      ratio=0.6;

  PetscFunctionBegin;  
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  if (m) rmax = ailen[0]; /* determine row with most nonzeros */
  for (i=1; i<m; i++) {
    /* move each row back by the amount of empty slots (fshift) before it*/
    fshift += imax[i-1] - ailen[i-1];
    rmax   = PetscMax(rmax,ailen[i]);
    if (fshift) {
      ip = aj + ai[i] ; 
      ap = aa + ai[i] ;
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
  a->nz = ai[m]; 

  /* diagonals may have moved, so kill the diagonal pointers */
  if (fshift && a->diag) {
    ierr = PetscFree(a->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(A,-(m+1)*sizeof(PetscInt));
    a->diag = 0;
  } 
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Matrix size: %D X %D; storage space: %D unneeded,%D used\n",m,A->n,fshift,a->nz);
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Number of mallocs during MatSetValues() is %D\n",a->reallocs);
  PetscLogInfo(A,"MatAssemblyEnd_SeqAIJ:Maximum nonzeros in any row is %D\n",rmax);
  a->reallocs          = 0;
  A->info.nz_unneeded  = (double)fshift;
  a->rmax              = rmax;

  /* check for identical nodes. If found, use inode functions */
  if (!a->inode.checked){ 
    ierr = Mat_AIJ_CheckInode(A,(PetscTruth)(!fshift));CHKERRQ(ierr);
  }

  /* check for zero rows. If found a large number of zero rows, use CompressedRow functions */
  if (!a->inode.use && a->compressedrow.use && !A->same_nonzero){
    ierr = Mat_CheckCompressedRow(A,&a->compressedrow,a->i,ratio);CHKERRQ(ierr); 
  } 

  A->same_nonzero = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatZeroEntries_SeqAIJ"
PetscErrorCode MatZeroEntries_SeqAIJ(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  PetscErrorCode ierr;

  PetscFunctionBegin;  
  ierr = PetscMemzero(a->a,(a->i[A->m])*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ"
PetscErrorCode MatDestroy_SeqAIJ(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;  
#if defined(PETSC_USE_LOG)
  PetscLogObjectState((PetscObject)A,"Rows=%D, Cols=%D, NZ=%D",A->m,A->n,a->nz);
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
  if (a->xtoy) {ierr = PetscFree(a->xtoy);CHKERRQ(ierr);}
  if (a->compressedrow.use){ierr = PetscFree(a->compressedrow.i);} 
  
  ierr = PetscFree(a);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSeqAIJSetColumnIndices_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatStoreValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatRetrieveValues_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatConvert_seqaij_seqsbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatConvert_seqaij_seqbaij_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatIsTranspose_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSeqAIJSetPreallocation_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatReorderForNonzeroDiagonal_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatAdjustForInodes_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSeqAIJGetInodeSizes_C","",PETSC_NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCompress_SeqAIJ"
PetscErrorCode MatCompress_SeqAIJ(Mat A)
{
  PetscFunctionBegin;  
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetOption_SeqAIJ"
PetscErrorCode MatSetOption_SeqAIJ(Mat A,MatOption op)
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
    case MAT_USE_COMPRESSEDROW:
      a->compressedrow.use = PETSC_TRUE;
      break;
    case MAT_DO_NOT_USE_COMPRESSEDROW:
      a->compressedrow.use = PETSC_FALSE;
      break;
    case MAT_ROWS_SORTED:
    case MAT_ROWS_UNSORTED:
    case MAT_YES_NEW_DIAGONALS:
    case MAT_IGNORE_OFF_PROC_ENTRIES:
    case MAT_USE_HASH_TABLE:
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
#define __FUNCT__ "MatGetDiagonal_SeqAIJ"
PetscErrorCode MatGetDiagonal_SeqAIJ(Mat A,Vec v)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,j,n;
  PetscScalar    *x,zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != A->m) SETERRQ(PETSC_ERR_ARG_SIZ,"Nonconforming matrix and vector");
  for (i=0; i<A->m; i++) {
    for (j=a->i[i]; j<a->i[i+1]; j++) {
      if (a->j[j] == i) {
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
PetscErrorCode MatMultTransposeAdd_SeqAIJ(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqAIJ        *a = (Mat_SeqAIJ*)A->data;
  PetscScalar       *x,*y;
  PetscErrorCode    ierr;
  PetscInt          m = A->m;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  PetscScalar       *v,alpha;
  PetscInt          n,i,*idx,*ii,*ridx=PETSC_NULL;
  Mat_CompressedRow cprow = a->compressedrow;
  PetscTruth        usecprow; 
#endif

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);

#if defined(PETSC_USE_FORTRAN_KERNEL_MULTTRANSPOSEAIJ)
  fortranmulttransposeaddaij_(&m,x,a->i,a->j,a->a,y);
#else
  if (cprow.use && cprow.checked){
    usecprow = PETSC_TRUE;
  } else {
    usecprow = PETSC_FALSE;
  }

  if (usecprow){
    m    = cprow.nrows;
    ii   = cprow.i;
    ridx = cprow.rindex;
  } else {
    ii = a->i;
  }
  for (i=0; i<m; i++) {
    idx   = a->j + ii[i] ;
    v     = a->a + ii[i] ;
    n     = ii[i+1] - ii[i];
    if (usecprow){
      alpha = x[ridx[i]];
    } else {
      alpha = x[i];
    }
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
PetscErrorCode MatMultTranspose_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  PetscScalar    zero = 0.0;
  PetscErrorCode ierr;

  PetscFunctionBegin; 
  ierr = VecSet(&zero,yy);CHKERRQ(ierr);
  ierr = MatMultTransposeAdd_SeqAIJ(A,xx,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqAIJ"
PetscErrorCode MatMult_SeqAIJ(Mat A,Vec xx,Vec yy)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *x,*y,*aa;
  PetscErrorCode ierr;
  PetscInt       m=A->m,*aj,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  PetscInt       n,i,jrow,j,*ridx=PETSC_NULL;
  PetscScalar    sum;
  PetscTruth     usecprow=a->compressedrow.use;
#endif

#if defined(PETSC_HAVE_PRAGMA_DISJOINT)
#pragma disjoint(*x,*y,*aa)
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  aj  = a->j;
  aa    = a->a;
  ii   = a->i;
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTAIJ)
  fortranmultaij_(&m,x,ii,aj,aa,y);
#else
  if (usecprow && a->compressedrow.checked){ /* use compressed row format */
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i=0; i<m; i++){
      n   = ii[i+1] - ii[i]; 
      aj  = a->j + ii[i];
      aa  = a->a + ii[i];
      sum = 0.0;
      for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; 
      y[*ridx++] = sum;
    }
  } else { /* do not use compressed row format */
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sum  = 0.0;
      for (j=0; j<n; j++) {
        sum += aa[jrow]*x[aj[jrow]]; jrow++;
      }
      y[i] = sum;
    }
  }
#endif
  PetscLogFlops(2*a->nz - m);
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqAIJ"
PetscErrorCode MatMultAdd_SeqAIJ(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *x,*y,*z,*aa;
  PetscErrorCode ierr;
  PetscInt       m = A->m,*aj,*ii;
#if !defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
  PetscInt       n,i,jrow,j,*ridx=PETSC_NULL;
  PetscScalar    sum;
  PetscTruth     usecprow=a->compressedrow.use;
#endif

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&y);CHKERRQ(ierr);
  if (zz != yy) {
    ierr = VecGetArray(zz,&z);CHKERRQ(ierr);
  } else {
    z = y;
  }

  aj  = a->j;
  aa  = a->a;
  ii  = a->i;
#if defined(PETSC_USE_FORTRAN_KERNEL_MULTADDAIJ)
  fortranmultaddaij_(&m,x,ii,aj,aa,y,z);
#else
  if (usecprow && a->compressedrow.checked){ /* use compressed row format */
    m    = a->compressedrow.nrows;
    ii   = a->compressedrow.i;
    ridx = a->compressedrow.rindex;
    for (i=0; i<m; i++){
      n  = ii[i+1] - ii[i]; 
      aj  = a->j + ii[i];
      aa  = a->a + ii[i];
      sum = y[*ridx];
      for (j=0; j<n; j++) sum += (*aa++)*x[*aj++]; 
      z[*ridx++] = sum;
    }
  } else { /* do not use compressed row format */
    for (i=0; i<m; i++) {
      jrow = ii[i];
      n    = ii[i+1] - jrow;
      sum  = y[i];
      for (j=0; j<n; j++) {
        sum += aa[jrow]*x[aj[jrow]]; jrow++;
      }
      z[i] = sum;
    }
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
PetscErrorCode MatMarkDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       i,j,*diag,m = A->m;

  PetscFunctionBegin;
  if (a->diag) PetscFunctionReturn(0);

  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&diag);CHKERRQ(ierr);
  PetscLogObjectMemory(A,(m+1)*sizeof(PetscInt));
  for (i=0; i<A->m; i++) {
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

/*
     Checks for missing diagonals
*/
#undef __FUNCT__  
#define __FUNCT__ "MatMissingDiagonal_SeqAIJ"
PetscErrorCode MatMissingDiagonal_SeqAIJ(Mat A)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data; 
  PetscErrorCode ierr;
  PetscInt       *diag,*jj = a->j,i;

  PetscFunctionBegin;
  ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
  diag = a->diag;
  for (i=0; i<A->m; i++) {
    if (jj[diag[i]] != i) {
      SETERRQ1(PETSC_ERR_PLIB,"Matrix is missing diagonal number %D",i);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRelax_SeqAIJ"
PetscErrorCode MatRelax_SeqAIJ(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal fshift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data;
  PetscScalar        *x,d,*xs,sum,*t,scale,*idiag=0,*mdiag;
  const PetscScalar  *v = a->a, *b, *bs,*xb, *ts;
  PetscErrorCode     ierr;
  PetscInt           n = A->n,m = A->m,i;
  const PetscInt     *idx,*diag;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);

  if (!a->diag) {ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);}
  diag = a->diag;
  if (!a->idiag) {
    ierr     = PetscMalloc(3*m*sizeof(PetscScalar),&a->idiag);CHKERRQ(ierr);
    a->ssor  = a->idiag + m;
    mdiag    = a->ssor + m;

    v        = a->a;

    /* this is wrong when fshift omega changes each iteration */
    if (omega == 1.0 && !fshift) {
      for (i=0; i<m; i++) {
        mdiag[i]    = v[diag[i]];
        a->idiag[i] = 1.0/v[diag[i]];
      }
      PetscLogFlops(m);
    } else {
      for (i=0; i<m; i++) {
        mdiag[i]    = v[diag[i]];
        a->idiag[i] = omega/(fshift + v[diag[i]]);
      }
      PetscLogFlops(2*m);
    }
  }
  t     = a->ssor;
  idiag = a->idiag;
  mdiag = a->idiag + 2*m;

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  if (xx != bb) {
    ierr = VecGetArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);
  } else {
    b = x;
  }

  /* We count flops by assuming the upper triangular and lower triangular parts have the same number of nonzeros */
  xs   = x;
  if (flag == SOR_APPLY_UPPER) {
   /* apply (U + D/omega) to the vector */
    bs = b;
    for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]];
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + 1;
        v    = a->a + diag[i] + 1;
        sum  = b[i]*d/omega;
        SPARSEDENSEDOT(sum,bs,v,idx,n); 
        x[i] = sum;
    }
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);}
    PetscLogFlops(a->nz);
    PetscFunctionReturn(0);
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
      n    = a->i[i+1] - diag[i] - 1;
      idx  = a->j + diag[i] + 1;
      v    = a->a + diag[i] + 1;
      sum  = b[i];
      SPARSEDENSEMDOT(sum,xs,v,idx,n); 
      x[i] = sum*idiag[i];
    }

    /*  t = b - (2*E - D)x */
    v = a->a;
    for (i=0; i<m; i++) { t[i] = b[i] - scale*(v[*diag++])*x[i]; }

    /*  t = (E + L)^{-1}t */
    ts = t; 
    diag = a->diag;
    for (i=0; i<m; i++) {
      n    = diag[i] - a->i[i];
      idx  = a->j + a->i[i];
      v    = a->a + a->i[i];
      sum  = t[i];
      SPARSEDENSEMDOT(sum,ts,v,idx,n); 
      t[i] = sum*idiag[i];
      /*  x = x + t */
      x[i] += t[i];
    }

    PetscLogFlops(6*m-1 + 2*a->nz);
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (bb != xx) {ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijforwardzero_(&m,&omega,x,a->i,a->j,(PetscInt*)diag,idiag,a->a,(void*)b);
#else
      for (i=0; i<m; i++) {
        n    = diag[i] - a->i[i];
        idx  = a->j + a->i[i];
        v    = a->a + a->i[i];
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = sum*idiag[i];
      }
#endif
      xb = x;
      PetscLogFlops(a->nz);
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=0; i<m; i++) {
        x[i] *= mdiag[i];
      }
      PetscLogFlops(m); 
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijbackwardzero_(&m,&omega,x,a->i,a->j,(PetscInt*)diag,idiag,a->a,(void*)xb);
#else
      for (i=m-1; i>=0; i--) {
        n    = a->i[i+1] - diag[i] - 1;
        idx  = a->j + diag[i] + 1;
        v    = a->a + diag[i] + 1;
        sum  = xb[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = sum*idiag[i];
      }
#endif
      PetscLogFlops(a->nz);
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijforward_(&m,&omega,x,a->i,a->j,(PetscInt*)diag,a->a,(void*)b);
#else
      for (i=0; i<m; i++) {
        d    = fshift + a->a[diag[i]];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i];
        v    = a->a + a->i[i];
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + (sum + mdiag[i]*x[i])*idiag[i];
      }
#endif 
      PetscLogFlops(a->nz);
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
#if defined(PETSC_USE_FORTRAN_KERNEL_RELAXAIJ)
      fortranrelaxaijbackward_(&m,&omega,x,a->i,a->j,(PetscInt*)diag,a->a,(void*)b);
#else
      for (i=m-1; i>=0; i--) {
        d    = fshift + a->a[diag[i]];
        n    = a->i[i+1] - a->i[i]; 
        idx  = a->j + a->i[i];
        v    = a->a + a->i[i];
        sum  = b[i];
        SPARSEDENSEMDOT(sum,xs,v,idx,n); 
        x[i] = (1. - omega)*x[i] + (sum + mdiag[i]*x[i])*idiag[i];
      }
#endif
      PetscLogFlops(a->nz);
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  if (bb != xx) {ierr = VecRestoreArray(bb,(PetscScalar**)&b);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqAIJ"
PetscErrorCode MatGetInfo_SeqAIJ(Mat A,MatInfoType flag,MatInfo *info)
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

#undef __FUNCT__  
#define __FUNCT__ "MatZeroRows_SeqAIJ"
PetscErrorCode MatZeroRows_SeqAIJ(Mat A,IS is,const PetscScalar *diag)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,N,*rows,m = A->m - 1;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is,&N);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows);CHKERRQ(ierr);
  if (a->keepzeroedrows) {
    for (i=0; i<N; i++) {
      if (rows[i] < 0 || rows[i] > m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range", rows[i]);
      ierr = PetscMemzero(&a->a[a->i[rows[i]]],a->ilen[rows[i]]*sizeof(PetscScalar));CHKERRQ(ierr);
    }
    if (diag) {
      ierr = MatMissingDiagonal_SeqAIJ(A);CHKERRQ(ierr);
      ierr = MatMarkDiagonal_SeqAIJ(A);CHKERRQ(ierr);
      for (i=0; i<N; i++) {
        a->a[a->diag[rows[i]]] = *diag;
      }
    }
    A->same_nonzero = PETSC_TRUE;
  } else {
    if (diag) {
      for (i=0; i<N; i++) {
        if (rows[i] < 0 || rows[i] > m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range", rows[i]);
        if (a->ilen[rows[i]] > 0) { 
          a->ilen[rows[i]]          = 1; 
          a->a[a->i[rows[i]]] = *diag;
          a->j[a->i[rows[i]]] = rows[i];
        } else { /* in case row was completely empty */
          ierr = MatSetValues_SeqAIJ(A,1,&rows[i],1,&rows[i],diag,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    } else {
      for (i=0; i<N; i++) {
        if (rows[i] < 0 || rows[i] > m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"row %D out of range", rows[i]);
        a->ilen[rows[i]] = 0; 
      }
    }
    A->same_nonzero = PETSC_FALSE;
  }
  ierr = ISRestoreIndices(is,&rows);CHKERRQ(ierr);
  ierr = MatAssemblyEnd_SeqAIJ(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqAIJ"
PetscErrorCode MatGetRow_SeqAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data;
  PetscInt   *itmp;

  PetscFunctionBegin;
  if (row < 0 || row >= A->m) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Row %D out of range",row);

  *nz = a->i[row+1] - a->i[row];
  if (v) *v = a->a + a->i[row];
  if (idx) {
    itmp = a->j + a->i[row];
    if (*nz) {
      *idx = itmp;
    }
    else *idx = 0;
  }
  PetscFunctionReturn(0);
}

/* remove this function? */
#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqAIJ"
PetscErrorCode MatRestoreRow_SeqAIJ(Mat A,PetscInt row,PetscInt *nz,PetscInt **idx,PetscScalar **v)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqAIJ"
PetscErrorCode MatNorm_SeqAIJ(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *v = a->a;
  PetscReal      sum = 0.0;
  PetscErrorCode ierr;
  PetscInt       i,j;

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
    PetscInt    *jj = a->j;
    ierr = PetscMalloc((A->n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMemzero(tmp,A->n*sizeof(PetscReal));CHKERRQ(ierr);
    *nrm = 0.0;
    for (j=0; j<a->nz; j++) {
        tmp[*jj++] += PetscAbsScalar(*v);  v++;
    }
    for (j=0; j<A->n; j++) {
      if (tmp[j] > *nrm) *nrm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) {
    *nrm = 0.0;
    for (j=0; j<A->m; j++) {
      v = a->a + a->i[j];
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
PetscErrorCode MatTranspose_SeqAIJ(Mat A,Mat *B)
{ 
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       i,*aj = a->j,*ai = a->i,m = A->m,len,*col;
  PetscScalar    *array = a->a;

  PetscFunctionBegin;
  if (!B && m != A->n) SETERRQ(PETSC_ERR_ARG_SIZ,"Square matrix only for in-place");
  ierr = PetscMalloc((1+A->n)*sizeof(PetscInt),&col);CHKERRQ(ierr);
  ierr = PetscMemzero(col,(1+A->n)*sizeof(PetscInt));CHKERRQ(ierr);
  
  for (i=0; i<ai[m]; i++) col[aj[i]] += 1;
  ierr = MatCreate(A->comm,A->n,m,A->n,m,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,0,col);CHKERRQ(ierr);
  ierr = PetscFree(col);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    len    = ai[i+1]-ai[i];
    ierr   = MatSetValues(C,len,aj,1,&i,array,INSERT_VALUES);CHKERRQ(ierr);
    array += len; 
    aj    += len;
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

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatIsTranspose_SeqAIJ"
PetscErrorCode MatIsTranspose_SeqAIJ(Mat A,Mat B,PetscReal tol,PetscTruth *f)
{
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ *) A->data,*bij = (Mat_SeqAIJ*) A->data;
  PetscInt       *adx,*bdx,*aii,*bii,*aptr,*bptr; PetscScalar *va,*vb;
  PetscErrorCode ierr;
  PetscInt       ma,na,mb,nb, i;

  PetscFunctionBegin;
  bij = (Mat_SeqAIJ *) B->data;
  
  ierr = MatGetSize(A,&ma,&na);CHKERRQ(ierr);
  ierr = MatGetSize(B,&mb,&nb);CHKERRQ(ierr);
  if (ma!=nb || na!=mb){
    *f = PETSC_FALSE;
    PetscFunctionReturn(0);
  }
  aii = aij->i; bii = bij->i;
  adx = aij->j; bdx = bij->j;
  va  = aij->a; vb = bij->a;
  ierr = PetscMalloc(ma*sizeof(PetscInt),&aptr);CHKERRQ(ierr);
  ierr = PetscMalloc(mb*sizeof(PetscInt),&bptr);CHKERRQ(ierr);
  for (i=0; i<ma; i++) aptr[i] = aii[i];
  for (i=0; i<mb; i++) bptr[i] = bii[i];

  *f = PETSC_TRUE;
  for (i=0; i<ma; i++) {
    while (aptr[i]<aii[i+1]) {
      PetscInt         idc,idr; 
      PetscScalar vc,vr;
      /* column/row index/value */
      idc = adx[aptr[i]]; 
      idr = bdx[bptr[idc]];
      vc  = va[aptr[i]]; 
      vr  = vb[bptr[idc]];
      if (i!=idr || PetscAbsScalar(vc-vr) > tol) {
	*f = PETSC_FALSE;
        goto done;
      } else {
	aptr[i]++; 
        if (B || i!=idc) bptr[idc]++;
      }
    }
  }
 done:
  ierr = PetscFree(aptr);CHKERRQ(ierr);
  if (B) {
    ierr = PetscFree(bptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__
#define __FUNCT__ "MatIsSymmetric_SeqAIJ"
PetscErrorCode MatIsSymmetric_SeqAIJ(Mat A,PetscReal tol,PetscTruth *f)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  ierr = MatIsTranspose_SeqAIJ(A,A,tol,f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatDiagonalScale_SeqAIJ"
PetscErrorCode MatDiagonalScale_SeqAIJ(Mat A,Vec ll,Vec rr)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscScalar    *l,*r,x,*v;
  PetscErrorCode ierr;
  PetscInt       i,j,m = A->m,n = A->n,M,nz = a->nz,*jj;

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
      (*v++) *= r[*jj++]; 
    }
    ierr = VecRestoreArray(rr,&r);CHKERRQ(ierr); 
    PetscLogFlops(nz);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrix_SeqAIJ"
PetscErrorCode MatGetSubMatrix_SeqAIJ(Mat A,IS isrow,IS iscol,PetscInt csize,MatReuse scall,Mat *B)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data,*c;
  PetscErrorCode ierr;
  PetscInt       *smap,i,k,kstart,kend,oldcols = A->n,*lens;
  PetscInt       row,mat_i,*mat_j,tcol,first,step,*mat_ilen,sum,lensi;
  PetscInt       *irow,*icol,nrows,ncols;
  PetscInt       *starts,*j_new,*i_new,*aj = a->j,*ai = a->i,ii,*ailen = a->ilen;
  PetscScalar    *a_new,*mat_a;
  Mat            C;
  PetscTruth     stride;

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
    ierr   = PetscMalloc((2*nrows+1)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
    starts = lens + nrows;
    /* loop over new rows determining lens and starting points */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]; 
      kend    = kstart + ailen[irow[i]];
      for (k=kstart; k<kend; k++) {
        if (aj[k] >= first) {
          starts[i] = k;
          break;
	}
      }
      sum = 0;
      while (k < kend) {
        if (aj[k++] >= first+ncols) break;
        sum++;
      }
      lens[i] = sum;
    }
    /* create submatrix */
    if (scall == MAT_REUSE_MATRIX) {
      PetscInt n_cols,n_rows;
      ierr = MatGetSize(*B,&n_rows,&n_cols);CHKERRQ(ierr);
      if (n_rows != nrows || n_cols != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,"Reused submatrix wrong size");
      ierr = MatZeroEntries(*B);CHKERRQ(ierr);
      C = *B;
    } else {  
      ierr = MatCreate(A->comm,nrows,ncols,PETSC_DETERMINE,PETSC_DETERMINE,&C);CHKERRQ(ierr);
      ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(C,0,lens);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ*)C->data;

    /* loop over rows inserting into submatrix */
    a_new    = c->a;
    j_new    = c->j;
    i_new    = c->i;

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
    ierr  = PetscMalloc((1+oldcols)*sizeof(PetscInt),&smap);CHKERRQ(ierr);
    
    ierr  = PetscMalloc((1+nrows)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
    ierr  = PetscMemzero(smap,oldcols*sizeof(PetscInt));CHKERRQ(ierr);
    for (i=0; i<ncols; i++) smap[icol[i]] = i+1;
    /* determine lens of each row */
    for (i=0; i<nrows; i++) {
      kstart  = ai[irow[i]]; 
      kend    = kstart + a->ilen[irow[i]];
      lens[i] = 0;
      for (k=kstart; k<kend; k++) {
        if (smap[aj[k]]) {
          lens[i]++;
        }
      }
    }
    /* Create and fill new matrix */
    if (scall == MAT_REUSE_MATRIX) {
      PetscTruth equal;

      c = (Mat_SeqAIJ *)((*B)->data);
      if ((*B)->m  != nrows || (*B)->n != ncols) SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong size");
      ierr = PetscMemcmp(c->ilen,lens,(*B)->m*sizeof(PetscInt),&equal);CHKERRQ(ierr);
      if (!equal) {
        SETERRQ(PETSC_ERR_ARG_SIZ,"Cannot reuse matrix. wrong no of nonzeros");
      }
      ierr = PetscMemzero(c->ilen,(*B)->m*sizeof(PetscInt));CHKERRQ(ierr);
      C = *B;
    } else {  
      ierr = MatCreate(A->comm,nrows,ncols,PETSC_DETERMINE,PETSC_DETERMINE,&C);CHKERRQ(ierr);
      ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
      ierr = MatSeqAIJSetPreallocation(C,0,lens);CHKERRQ(ierr);
    }
    c = (Mat_SeqAIJ *)(C->data);
    for (i=0; i<nrows; i++) {
      row    = irow[i];
      kstart = ai[row]; 
      kend   = kstart + a->ilen[row];
      mat_i  = c->i[i];
      mat_j  = c->j + mat_i; 
      mat_a  = c->a + mat_i;
      mat_ilen = c->ilen + i;
      for (k=kstart; k<kend; k++) {
        if ((tcol=smap[a->j[k]])) {
          *mat_j++ = tcol - 1;
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
PetscErrorCode MatILUFactor_SeqAIJ(Mat inA,IS row,IS col,MatFactorInfo *info)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)inA->data;
  PetscErrorCode ierr;
  Mat            outA;
  PetscTruth     row_identity,col_identity;

  PetscFunctionBegin;
  if (info->levels != 0) SETERRQ(PETSC_ERR_SUP,"Only levels=0 supported for in-place ilu");
  ierr = ISIdentity(row,&row_identity);CHKERRQ(ierr);
  ierr = ISIdentity(col,&col_identity);CHKERRQ(ierr);
  if (!row_identity || !col_identity) {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Row and column permutations must be identity for in-place ILU");
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
PetscErrorCode MatScale_SeqAIJ(const PetscScalar *alpha,Mat inA)
{
  Mat_SeqAIJ   *a = (Mat_SeqAIJ*)inA->data;
  PetscBLASInt bnz = (PetscBLASInt)a->nz,one = 1;

  PetscFunctionBegin;
  BLASscal_(&bnz,(PetscScalar*)alpha,a->a,&one);
  PetscLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetSubMatrices_SeqAIJ"
PetscErrorCode MatGetSubMatrices_SeqAIJ(Mat A,PetscInt n,const IS irow[],const IS icol[],MatReuse scall,Mat *B[])
{
  PetscErrorCode ierr;
  PetscInt       i;

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
#define __FUNCT__ "MatIncreaseOverlap_SeqAIJ"
PetscErrorCode MatIncreaseOverlap_SeqAIJ(Mat A,PetscInt is_max,IS is[],PetscInt ov)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       row,i,j,k,l,m,n,*idx,*nidx,isz,val;
  PetscInt       start,end,*ai,*aj;
  PetscBT        table;

  PetscFunctionBegin;
  m     = A->m;
  ai    = a->i;
  aj    = a->j;

  if (ov < 0)  SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"illegal negative overlap value used");

  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&nidx);CHKERRQ(ierr); 
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
          val = aj[l] ;
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
PetscErrorCode MatPermute_SeqAIJ(Mat A,IS rowp,IS colp,Mat *B)
{ 
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,nz,m = A->m,n = A->n,*col;
  PetscInt       *row,*cnew,j,*lens;
  IS             icolp,irowp;
  PetscInt       *cwork;
  PetscScalar    *vwork;

  PetscFunctionBegin;
  ierr = ISInvertPermutation(rowp,PETSC_DECIDE,&irowp);CHKERRQ(ierr);
  ierr = ISGetIndices(irowp,&row);CHKERRQ(ierr);
  ierr = ISInvertPermutation(colp,PETSC_DECIDE,&icolp);CHKERRQ(ierr);
  ierr = ISGetIndices(icolp,&col);CHKERRQ(ierr);
  
  /* determine lengths of permuted rows */
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&lens);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    lens[row[i]] = a->i[i+1] - a->i[i];
  }
  ierr = MatCreate(A->comm,m,n,m,n,B);CHKERRQ(ierr);
  ierr = MatSetType(*B,A->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*B,0,lens);CHKERRQ(ierr);
  ierr = PetscFree(lens);CHKERRQ(ierr);

  ierr = PetscMalloc(n*sizeof(PetscInt),&cnew);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    ierr = MatGetRow_SeqAIJ(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
    for (j=0; j<nz; j++) { cnew[j] = col[cwork[j]];}
    ierr = MatSetValues_SeqAIJ(*B,1,&row[i],nz,cnew,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqAIJ(A,i,&nz,&cwork,&vwork);CHKERRQ(ierr);
  }
  ierr = PetscFree(cnew);CHKERRQ(ierr);
  (*B)->assembled     = PETSC_FALSE;
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
PetscErrorCode MatPrintHelp_SeqAIJ(Mat A)
{
  static PetscTruth called = PETSC_FALSE; 
  MPI_Comm          comm = A->comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = PETSC_TRUE;
  ierr = (*PetscHelpPrintf)(comm," Options for MATSEQAIJ and MATMPIAIJ matrix formats (the defaults):\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_lu_pivotthreshold <threshold>: Set pivoting threshold\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_oneindex: internal indices begin at 1 instead of the default 0.\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_no_inode: Do not use inodes\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_aij_inode_limit <limit>: Set inode limit (max limit=5)\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(comm,"  -mat_no_compressedrow: Do not use compressedrow\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatCopy_SeqAIJ"
PetscErrorCode MatCopy_SeqAIJ(Mat A,Mat B,MatStructure str)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the two matrices have the same copy implementation, use fast copy. */
  if (str == SAME_NONZERO_PATTERN && (A->ops->copy == B->ops->copy)) {
    Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
    Mat_SeqAIJ *b = (Mat_SeqAIJ*)B->data; 

    if (a->i[A->m] != b->i[B->m]) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,"Number of nonzeros in two matrices are different");
    }
    ierr = PetscMemcpy(b->a,a->a,(a->i[A->m])*sizeof(PetscScalar));CHKERRQ(ierr);
  } else {
    ierr = MatCopy_Basic(A,B,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetUpPreallocation_SeqAIJ"
PetscErrorCode MatSetUpPreallocation_SeqAIJ(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr =  MatSeqAIJSetPreallocation(A,PETSC_DEFAULT,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetArray_SeqAIJ"
PetscErrorCode MatGetArray_SeqAIJ(Mat A,PetscScalar *array[])
{
  Mat_SeqAIJ *a = (Mat_SeqAIJ*)A->data; 
  PetscFunctionBegin;
  *array = a->a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreArray_SeqAIJ"
PetscErrorCode MatRestoreArray_SeqAIJ(Mat A,PetscScalar *array[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatFDColoringApply_SeqAIJ"
PetscErrorCode MatFDColoringApply_SeqAIJ(Mat J,MatFDColoring coloring,Vec x1,MatStructure *flag,void *sctx)
{
  PetscErrorCode (*f)(void*,Vec,Vec,void*) = (PetscErrorCode (*)(void*,Vec,Vec,void *))coloring->f;
  PetscErrorCode ierr;
  PetscInt       k,N,start,end,l,row,col,srow,**vscaleforrow,m1,m2;
  PetscScalar    dx,mone = -1.0,*y,*xx,*w3_array;
  PetscScalar    *vscale_array;
  PetscReal      epsilon = coloring->error_rel,umin = coloring->umin; 
  Vec            w1,w2,w3;
  void           *fctx = coloring->fctx;
  PetscTruth     flg;

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
    PetscTruth assembled;
    ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
    if (assembled) {
      ierr = MatZeroEntries(J);CHKERRQ(ierr);
    }
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
    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,x1,w1,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
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
    coloring->currentcolor = k;
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
      if (!PetscAbsScalar(dx)) SETERRQ(PETSC_ERR_PLIB,"Computed 0 differencing parameter");
      w3_array[col] += dx;
    } 
    w3_array = w3_array + start; ierr = VecRestoreArray(w3,&w3_array);CHKERRQ(ierr);

    /*
       Evaluate function at x1 + dx (here dx is a vector of perturbations)
    */

    ierr = PetscLogEventBegin(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
    ierr = (*f)(sctx,w3,w2,fctx);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_FDColoringFunction,0,0,0,0);CHKERRQ(ierr);
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
  coloring->currentcolor = k;
  ierr = VecRestoreArray(coloring->vscale,&vscale_array);CHKERRQ(ierr);
  xx = xx + start; ierr  = VecRestoreArray(x1,&xx);CHKERRQ(ierr);
  ierr  = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr  = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "petscblaslapack.h"
#undef __FUNCT__  
#define __FUNCT__ "MatAXPY_SeqAIJ"
PetscErrorCode MatAXPY_SeqAIJ(const PetscScalar a[],Mat X,Mat Y,MatStructure str)
{
  PetscErrorCode ierr;
  PetscInt       i;
  Mat_SeqAIJ     *x  = (Mat_SeqAIJ *)X->data,*y = (Mat_SeqAIJ *)Y->data;
  PetscBLASInt   one=1,bnz = (PetscBLASInt)x->nz;

  PetscFunctionBegin;
  if (str == SAME_NONZERO_PATTERN) {
    BLASaxpy_(&bnz,(PetscScalar*)a,x->a,&one,y->a,&one);
  } else if (str == SUBSET_NONZERO_PATTERN) { /* nonzeros of X is a subset of Y's */
    if (y->xtoy && y->XtoY != X) {
      ierr = PetscFree(y->xtoy);CHKERRQ(ierr);
      ierr = MatDestroy(y->XtoY);CHKERRQ(ierr);
    }
    if (!y->xtoy) { /* get xtoy */
      ierr = MatAXPYGetxtoy_Private(X->m,x->i,x->j,PETSC_NULL, y->i,y->j,PETSC_NULL, &y->xtoy);CHKERRQ(ierr);
      y->XtoY = X;
    }
    for (i=0; i<x->nz; i++) y->a[y->xtoy[i]] += (*a)*(x->a[i]); 
    PetscLogInfo(0,"MatAXPY_SeqAIJ: ratio of nnz(X)/nnz(Y): %d/%d = %g\n",x->nz,y->nz,(PetscReal)(x->nz)/y->nz);
  } else {
    ierr = MatAXPY_Basic(a,X,Y,str);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetBlockSize_SeqAIJ"
PetscErrorCode MatSetBlockSize_SeqAIJ(Mat A,PetscInt bs)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps_Values = {MatSetValues_SeqAIJ,
       MatGetRow_SeqAIJ,
       MatRestoreRow_SeqAIJ,
       MatMult_SeqAIJ,
/* 4*/ MatMultAdd_SeqAIJ,
       MatMultTranspose_SeqAIJ,
       MatMultTransposeAdd_SeqAIJ,
       MatSolve_SeqAIJ,
       MatSolveAdd_SeqAIJ,
       MatSolveTranspose_SeqAIJ,
/*10*/ MatSolveTransposeAdd_SeqAIJ,
       MatLUFactor_SeqAIJ,
       0,
       MatRelax_SeqAIJ,
       MatTranspose_SeqAIJ,
/*15*/ MatGetInfo_SeqAIJ,
       MatEqual_SeqAIJ,
       MatGetDiagonal_SeqAIJ,
       MatDiagonalScale_SeqAIJ,
       MatNorm_SeqAIJ,
/*20*/ 0,
       MatAssemblyEnd_SeqAIJ,
       MatCompress_SeqAIJ,
       MatSetOption_SeqAIJ,
       MatZeroEntries_SeqAIJ,
/*25*/ MatZeroRows_SeqAIJ,
       MatLUFactorSymbolic_SeqAIJ,
       MatLUFactorNumeric_SeqAIJ,
       MatCholeskyFactorSymbolic_SeqAIJ,
       MatCholeskyFactorNumeric_SeqAIJ,
/*30*/ MatSetUpPreallocation_SeqAIJ,
       MatILUFactorSymbolic_SeqAIJ,
       MatICCFactorSymbolic_SeqAIJ,
       MatGetArray_SeqAIJ,
       MatRestoreArray_SeqAIJ,
/*35*/ MatDuplicate_SeqAIJ,
       0,
       0,
       MatILUFactor_SeqAIJ,
       0,
/*40*/ MatAXPY_SeqAIJ,
       MatGetSubMatrices_SeqAIJ,
       MatIncreaseOverlap_SeqAIJ,
       MatGetValues_SeqAIJ,
       MatCopy_SeqAIJ,
/*45*/ MatPrintHelp_SeqAIJ,
       MatScale_SeqAIJ,
       0,
       0,
       MatILUDTFactor_SeqAIJ,
/*50*/ MatSetBlockSize_SeqAIJ,
       MatGetRowIJ_SeqAIJ,
       MatRestoreRowIJ_SeqAIJ,
       MatGetColumnIJ_SeqAIJ,
       MatRestoreColumnIJ_SeqAIJ,
/*55*/ MatFDColoringCreate_SeqAIJ,
       0,
       0,
       MatPermute_SeqAIJ,
       0,
/*60*/ 0,
       MatDestroy_SeqAIJ,
       MatView_SeqAIJ,
       MatGetPetscMaps_Petsc,
       0,
/*65*/ 0,
       0,
       0,
       0,
       0,
/*70*/ 0,
       0,
       MatSetColoring_SeqAIJ,
#if defined(PETSC_HAVE_ADIC)
       MatSetValuesAdic_SeqAIJ,
#else
       0,
#endif
       MatSetValuesAdifor_SeqAIJ,
/*75*/ MatFDColoringApply_SeqAIJ,
       0,
       0,
       0,
       0,
/*80*/ 0,
       0,
       0,
       0,
       MatLoad_SeqAIJ,
/*85*/ MatIsSymmetric_SeqAIJ,
       0,
       0,
       0,
       0,
/*90*/ MatMatMult_SeqAIJ_SeqAIJ,  
       MatMatMultSymbolic_SeqAIJ_SeqAIJ,  
       MatMatMultNumeric_SeqAIJ_SeqAIJ,   
       MatPtAP_SeqAIJ_SeqAIJ,
       MatPtAPSymbolic_SeqAIJ_SeqAIJ,
/*95*/ MatPtAPNumeric_SeqAIJ_SeqAIJ,
       MatMatMultTranspose_SeqAIJ_SeqAIJ,  
       MatMatMultTransposeSymbolic_SeqAIJ_SeqAIJ,  
       MatMatMultTransposeNumeric_SeqAIJ_SeqAIJ,  
};

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJSetColumnIndices_SeqAIJ"
PetscErrorCode MatSeqAIJSetColumnIndices_SeqAIJ(Mat mat,PetscInt *indices)
{
  Mat_SeqAIJ *aij = (Mat_SeqAIJ *)mat->data;
  PetscInt   i,nz,n;

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
PetscErrorCode MatSeqAIJSetColumnIndices(Mat mat,PetscInt *indices)
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  PetscValidPointer(indices,2);
  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatSeqAIJSetColumnIndices_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat,indices);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Wrong type of matrix to set column indices");
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatStoreValues_SeqAIJ"
PetscErrorCode MatStoreValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ *)mat->data;
  PetscErrorCode ierr;
  size_t         nz = aij->i[mat->m];

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
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
PetscErrorCode MatStoreValues(Mat mat)
{
  PetscErrorCode ierr,(*f)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatStoreValues_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Wrong type of matrix to store values");
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatRetrieveValues_SeqAIJ"
PetscErrorCode MatRetrieveValues_SeqAIJ(Mat mat)
{
  Mat_SeqAIJ     *aij = (Mat_SeqAIJ *)mat->data;
  PetscErrorCode ierr;
  PetscInt       nz = aij->i[mat->m];

  PetscFunctionBegin;
  if (aij->nonew != 1) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatSetOption(A,MAT_NO_NEW_NONZERO_LOCATIONS);first");
  }
  if (!aij->saved_values) {
    SETERRQ(PETSC_ERR_ORDER,"Must call MatStoreValues(A);first");
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
PetscErrorCode MatRetrieveValues(Mat mat)
{
  PetscErrorCode ierr,(*f)(Mat);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE,1);
  if (!mat->assembled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  if (mat->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Not for factored matrix"); 

  ierr = PetscObjectQueryFunction((PetscObject)mat,"MatRetrieveValues_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(mat);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"Wrong type of matrix to retrieve values");
  }
  PetscFunctionReturn(0);
}


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
   If nnz is given then nz is ignored

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
PetscErrorCode MatCreateSeqAIJ(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt nz,const PetscInt nnz[],Mat *A)
{
  PetscErrorCode ierr;

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
     If nnz is given then nz is ignored

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
PetscErrorCode MatSeqAIJSetPreallocation(Mat B,PetscInt nz,const PetscInt nnz[])
{
  PetscErrorCode ierr,(*f)(Mat,PetscInt,const PetscInt[]);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)B,"MatSeqAIJSetPreallocation_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(B,nz,nnz);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJSetPreallocation_SeqAIJ"
PetscErrorCode MatSeqAIJSetPreallocation_SeqAIJ(Mat B,PetscInt nz,PetscInt *nnz)
{
  Mat_SeqAIJ     *b;
  size_t         len = 0;
  PetscTruth     skipallocation = PETSC_FALSE;
  PetscErrorCode ierr;
  PetscInt       i;

  PetscFunctionBegin;
  
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

  ierr = PetscMalloc((B->m+1)*sizeof(PetscInt),&b->imax);CHKERRQ(ierr);
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
    len             = ((size_t) nz)*(sizeof(PetscInt) + sizeof(PetscScalar)) + (B->m+1)*sizeof(PetscInt);
    ierr            = PetscMalloc(len,&b->a);CHKERRQ(ierr);
    b->j            = (PetscInt*)(b->a + nz);
    ierr            = PetscMemzero(b->j,nz*sizeof(PetscInt));CHKERRQ(ierr);
    b->i            = b->j + nz;
    b->i[0] = 0;
    for (i=1; i<B->m+1; i++) {
      b->i[i] = b->i[i-1] + b->imax[i-1];
    }
    b->singlemalloc = PETSC_TRUE;
    b->freedata     = PETSC_TRUE;
  } else {
    b->freedata     = PETSC_FALSE;
  }

  /* b->ilen will count nonzeros in each row so far. */
  ierr = PetscMalloc((B->m+1)*sizeof(PetscInt),&b->ilen);CHKERRQ(ierr);
  PetscLogObjectMemory(B,len+2*(B->m+1)*sizeof(PetscInt)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));
  for (i=0; i<B->m; i++) { b->ilen[i] = 0;}

  b->nz                = 0;
  b->maxnz             = nz;
  B->info.nz_unneeded  = (double)b->maxnz;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
   MATSEQAIJ - MATSEQAIJ = "seqaij" - A matrix type to be used for sequential sparse matrices, 
   based on compressed sparse row format.

   Options Database Keys:
. -mat_type seqaij - sets the matrix type to "seqaij" during a call to MatSetFromOptions()

  Level: beginner

.seealso: MatCreateSeqAIJ(), MatSetFromOptions(), MatSetType(), MatCreate(), MatType
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatCreate_SeqAIJ"
PetscErrorCode MatCreate_SeqAIJ(Mat B)
{
  Mat_SeqAIJ     *b;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(B->comm,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Comm must be of size 1");

  B->m = B->M = PetscMax(B->m,B->M);
  B->n = B->N = PetscMax(B->n,B->N);

  ierr = PetscNew(Mat_SeqAIJ,&b);CHKERRQ(ierr);
  B->data             = (void*)b;
  ierr = PetscMemcpy(B->ops,&MatOps_Values,sizeof(struct _MatOps));CHKERRQ(ierr);
  B->factor           = 0;
  B->lupivotthreshold = 1.0;
  B->mapping          = 0;
  ierr = PetscOptionsGetReal(B->prefix,"-mat_lu_pivotthreshold",&B->lupivotthreshold,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(B->prefix,"-pc_ilu_preserve_row_sums",&b->ilu_preserve_row_sums);CHKERRQ(ierr);
  b->row              = 0;
  b->col              = 0;
  b->icol             = 0;
  b->reallocs         = 0;
  b->lu_shift         = PETSC_FALSE;
 
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
  b->xtoy              = 0;
  b->XtoY              = 0;
  b->compressedrow.use     = PETSC_FALSE;
  b->compressedrow.nrows   = B->m;
  b->compressedrow.i       = PETSC_NULL;
  b->compressedrow.rindex  = PETSC_NULL;
  b->compressedrow.checked = PETSC_FALSE;
  B->same_nonzero          = PETSC_FALSE;

  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetColumnIndices_C",
                                     "MatSeqAIJSetColumnIndices_SeqAIJ",
                                     MatSeqAIJSetColumnIndices_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatStoreValues_C",
                                     "MatStoreValues_SeqAIJ",
                                     MatStoreValues_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatRetrieveValues_C",
                                     "MatRetrieveValues_SeqAIJ",
                                     MatRetrieveValues_SeqAIJ);CHKERRQ(ierr);
#if !defined(PETSC_USE_64BIT_INT)
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_seqsbaij_C",
                                     "MatConvert_SeqAIJ_SeqSBAIJ",
                                      MatConvert_SeqAIJ_SeqSBAIJ);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_seqbaij_C",
                                     "MatConvert_SeqAIJ_SeqBAIJ",
                                      MatConvert_SeqAIJ_SeqBAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatIsTranspose_C",
                                     "MatIsTranspose_SeqAIJ",
                                      MatIsTranspose_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJSetPreallocation_C",
                                     "MatSeqAIJSetPreallocation_SeqAIJ",
                                      MatSeqAIJSetPreallocation_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatReorderForNonzeroDiagonal_C",
                                     "MatReorderForNonzeroDiagonal_SeqAIJ",
                                      MatReorderForNonzeroDiagonal_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatAdjustForInodes_C",
                                     "MatAdjustForInodes_SeqAIJ",
                                      MatAdjustForInodes_SeqAIJ);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSeqAIJGetInodeSizes_C",
                                     "MatSeqAIJGetInodeSizes_SeqAIJ",
                                      MatSeqAIJGetInodeSizes_SeqAIJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDuplicate_SeqAIJ"
PetscErrorCode MatDuplicate_SeqAIJ(Mat A,MatDuplicateOption cpvalues,Mat *B)
{
  Mat            C;
  Mat_SeqAIJ     *c,*a = (Mat_SeqAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       i,m = A->m;
  size_t         len;

  PetscFunctionBegin;
  *B = 0;
  ierr = MatCreate(A->comm,A->m,A->n,A->m,A->n,&C);CHKERRQ(ierr);
  ierr = MatSetType(C,A->type_name);CHKERRQ(ierr);
  ierr = PetscMemcpy(C->ops,A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);
  
  c = (Mat_SeqAIJ*)C->data;

  C->factor           = A->factor;
  C->lupivotthreshold = A->lupivotthreshold;

  c->row            = 0;
  c->col            = 0;
  c->icol           = 0;
  c->reallocs       = 0;
  c->lu_shift       = PETSC_FALSE;

  C->assembled      = PETSC_TRUE;
 
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&c->imax);CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&c->ilen);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    c->imax[i] = a->imax[i];
    c->ilen[i] = a->ilen[i]; 
  }

  /* allocate the matrix space */
  c->singlemalloc = PETSC_TRUE;
  len   = ((size_t) (m+1))*sizeof(PetscInt)+(a->i[m])*(sizeof(PetscScalar)+sizeof(PetscInt));
  ierr  = PetscMalloc(len,&c->a);CHKERRQ(ierr);
  c->j  = (PetscInt*)(c->a + a->i[m] );
  c->i  = c->j + a->i[m];
  ierr = PetscMemcpy(c->i,a->i,(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
  if (m > 0) {
    ierr = PetscMemcpy(c->j,a->j,(a->i[m])*sizeof(PetscInt));CHKERRQ(ierr);
    if (cpvalues == MAT_COPY_VALUES) {
      ierr = PetscMemcpy(c->a,a->a,(a->i[m])*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMemzero(c->a,(a->i[m])*sizeof(PetscScalar));CHKERRQ(ierr);
    }
  }

  PetscLogObjectMemory(C,len+2*(m+1)*sizeof(PetscInt)+sizeof(struct _p_Mat)+sizeof(Mat_SeqAIJ));  
  c->sorted            = a->sorted;
  c->ignorezeroentries = a->ignorezeroentries;
  c->roworiented       = a->roworiented;
  c->nonew             = a->nonew;
  if (a->diag) {
    ierr = PetscMalloc((m+1)*sizeof(PetscInt),&c->diag);CHKERRQ(ierr);
    PetscLogObjectMemory(C,(m+1)*sizeof(PetscInt));
    for (i=0; i<m; i++) {
      c->diag[i] = a->diag[i];
    }
  } else c->diag        = 0;
  c->solve_work         = 0;
  c->inode.use          = a->inode.use;
  c->inode.limit        = a->inode.limit;
  c->inode.max_limit    = a->inode.max_limit;
  if (a->inode.size){
    ierr                = PetscMalloc((m+1)*sizeof(PetscInt),&c->inode.size);CHKERRQ(ierr);
    c->inode.node_count = a->inode.node_count;
    ierr                = PetscMemcpy(c->inode.size,a->inode.size,(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
  } else {
    c->inode.size       = 0;
    c->inode.node_count = 0;
  }
  c->saved_values          = 0;
  c->idiag                 = 0;
  c->ilu_preserve_row_sums = a->ilu_preserve_row_sums;
  c->ssor                  = 0;
  c->keepzeroedrows        = a->keepzeroedrows;
  c->freedata              = PETSC_TRUE;
  c->xtoy                  = 0;
  c->XtoY                  = 0;

  c->nz                 = a->nz;
  c->maxnz              = a->maxnz;
  C->preallocated       = PETSC_TRUE;

  c->compressedrow.use     = a->compressedrow.use;
  c->compressedrow.nrows   = a->compressedrow.nrows;
  c->compressedrow.checked = a->compressedrow.checked;
  if ( a->compressedrow.checked && a->compressedrow.use){
    i = a->compressedrow.nrows;
    ierr = PetscMalloc((2*i+1)*sizeof(PetscInt),&c->compressedrow.i);CHKERRQ(ierr);
    c->compressedrow.rindex = c->compressedrow.i + i + 1;
    ierr = PetscMemcpy(c->compressedrow.i,a->compressedrow.i,(i+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(c->compressedrow.rindex,a->compressedrow.rindex,i*sizeof(PetscInt));CHKERRQ(ierr); 
  } else {
    c->compressedrow.use    = PETSC_FALSE;
    c->compressedrow.i      = PETSC_NULL;
    c->compressedrow.rindex = PETSC_NULL;
  }
  C->same_nonzero = A->same_nonzero;
  
  *B = C;
  ierr = PetscFListDuplicate(A->qlist,&C->qlist);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLoad_SeqAIJ"
PetscErrorCode MatLoad_SeqAIJ(PetscViewer viewer,const MatType type,Mat *A)
{
  Mat_SeqAIJ     *a;
  Mat            B;
  PetscErrorCode ierr;
  PetscInt       i,nz,header[4],*rowlengths = 0,M,N;
  int            fd;
  PetscMPIInt    size;
  MPI_Comm       comm;
  
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
  ierr = PetscMalloc(M*sizeof(PetscInt),&rowlengths);CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT);CHKERRQ(ierr);

  /* create our matrix */
  ierr = MatCreate(comm,PETSC_DECIDE,PETSC_DECIDE,M,N,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,type);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,rowlengths);CHKERRQ(ierr);
  a = (Mat_SeqAIJ*)B->data;

  /* read in column indices and adjust for Fortran indexing*/
  ierr = PetscBinaryRead(fd,a->j,nz,PETSC_INT);CHKERRQ(ierr);

  /* read in nonzero values */
  ierr = PetscBinaryRead(fd,a->a,nz,PETSC_SCALAR);CHKERRQ(ierr);

  /* set matrix "i" values */
  a->i[0] = 0;
  for (i=1; i<= M; i++) {
    a->i[i]      = a->i[i-1] + rowlengths[i-1];
    a->ilen[i-1] = rowlengths[i-1];
  }
  ierr = PetscFree(rowlengths);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatEqual_SeqAIJ"
PetscErrorCode MatEqual_SeqAIJ(Mat A,Mat B,PetscTruth* flg)
{
  Mat_SeqAIJ     *a = (Mat_SeqAIJ *)A->data,*b = (Mat_SeqAIJ *)B->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* If the  matrix dimensions are not equal,or no of nonzeros */
  if ((A->m != B->m) || (A->n != B->n) ||(a->nz != b->nz)) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0); 
  }
  
  /* if the a->i are the same */
  ierr = PetscMemcmp(a->i,b->i,(A->m+1)*sizeof(PetscInt),flg);CHKERRQ(ierr);
  if (*flg == PETSC_FALSE) PetscFunctionReturn(0);
  
  /* if a->j are the same */
  ierr = PetscMemcmp(a->j,b->j,(a->nz)*sizeof(PetscInt),flg);CHKERRQ(ierr);
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

       The i and j indices are 0 based

.seealso: MatCreate(), MatCreateMPIAIJ(), MatCreateSeqAIJ()

@*/
PetscErrorCode MatCreateSeqAIJWithArrays(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt* i,PetscInt*j,PetscScalar *a,Mat *mat)
{
  PetscErrorCode ierr;
  PetscInt       ii;
  Mat_SeqAIJ     *aij;

  PetscFunctionBegin;
  ierr = MatCreate(comm,m,n,m,n,mat);CHKERRQ(ierr);
  ierr = MatSetType(*mat,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(*mat,SKIP_ALLOCATION,0);CHKERRQ(ierr);
  aij  = (Mat_SeqAIJ*)(*mat)->data;

  if (i[0] != 0) {
    SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"i (row indices) must start with 0");
  }
  aij->i = i;
  aij->j = j;
  aij->a = a;
  aij->singlemalloc = PETSC_FALSE;
  aij->nonew        = -1;             /*this indicates that inserting a new value in the matrix that generates a new nonzero is an error*/
  aij->freedata     = PETSC_FALSE;

  for (ii=0; ii<m; ii++) {
    aij->ilen[ii] = aij->imax[ii] = i[ii+1] - i[ii];
#if defined(PETSC_USE_DEBUG)
    if (i[ii+1] - i[ii] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Negative row length in i (row indices) row = %d length = %d",ii,i[ii+1] - i[ii]);
#endif    
  }
#if defined(PETSC_USE_DEBUG)
  for (ii=0; ii<aij->i[m]; ii++) {
    if (j[ii] < 0) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Negative column index at location = %d index = %d",ii,j[ii]);
    if (j[ii] > n - 1) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column index to large at location = %d index = %d",ii,j[ii]);
  }
#endif    

  ierr = MatAssemblyBegin(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSetColoring_SeqAIJ"
PetscErrorCode MatSetColoring_SeqAIJ(Mat A,ISColoring coloring)
{
  PetscErrorCode ierr;
  Mat_SeqAIJ     *a = (Mat_SeqAIJ*)A->data;  

  PetscFunctionBegin;
  if (coloring->ctype == IS_COLORING_LOCAL) {
    ierr        = ISColoringReference(coloring);CHKERRQ(ierr);
    a->coloring = coloring;
  } else if (coloring->ctype == IS_COLORING_GHOSTED) {
    PetscInt             i,*larray;
    ISColoring      ocoloring;
    ISColoringValue *colors;

    /* set coloring for diagonal portion */
    ierr = PetscMalloc((A->n+1)*sizeof(PetscInt),&larray);CHKERRQ(ierr);
    for (i=0; i<A->n; i++) {
      larray[i] = i;
    }
    ierr = ISGlobalToLocalMappingApply(A->mapping,IS_GTOLM_MASK,A->n,larray,PETSC_NULL,larray);CHKERRQ(ierr);
    ierr = PetscMalloc((A->n+1)*sizeof(ISColoringValue),&colors);CHKERRQ(ierr);
    for (i=0; i<A->n; i++) {
      colors[i] = coloring->colors[larray[i]];
    }
    ierr = PetscFree(larray);CHKERRQ(ierr);
    ierr = ISColoringCreate(PETSC_COMM_SELF,A->n,colors,&ocoloring);CHKERRQ(ierr);
    a->coloring = ocoloring;
  }
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_ADIC)
EXTERN_C_BEGIN
#include "adic/ad_utils.h"
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdic_SeqAIJ"
PetscErrorCode MatSetValuesAdic_SeqAIJ(Mat A,void *advalues)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;  
  PetscInt        m = A->m,*ii = a->i,*jj = a->j,nz,i,j,nlen;
  PetscScalar     *v = a->a,*values = ((PetscScalar*)advalues)+1;
  ISColoringValue *color;

  PetscFunctionBegin;
  if (!a->coloring) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Coloring not set for matrix");
  nlen  = PetscADGetDerivTypeSize()/sizeof(PetscScalar);
  color = a->coloring->colors;
  /* loop over rows */
  for (i=0; i<m; i++) {
    nz = ii[i+1] - ii[i];
    /* loop over columns putting computed value into matrix */
    for (j=0; j<nz; j++) {
      *v++ = values[color[*jj++]];
    }
    values += nlen; /* jump to next row of derivatives */
  }
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatSetValuesAdifor_SeqAIJ"
PetscErrorCode MatSetValuesAdifor_SeqAIJ(Mat A,PetscInt nl,void *advalues)
{
  Mat_SeqAIJ      *a = (Mat_SeqAIJ*)A->data;  
  PetscInt             m = A->m,*ii = a->i,*jj = a->j,nz,i,j;
  PetscScalar     *v = a->a,*values = (PetscScalar *)advalues;
  ISColoringValue *color;

  PetscFunctionBegin;
  if (!a->coloring) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Coloring not set for matrix");
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

