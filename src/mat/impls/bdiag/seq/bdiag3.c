/*$Id: bdiag3.c,v 1.33 2001/08/07 03:02:53 balay Exp $*/

/* Block diagonal matrix format */

#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"
#include "petscsys.h"

EXTERN int MatSetValues_SeqBDiag_1(Mat,int,int *,int,int *,PetscScalar *,InsertMode);
EXTERN int MatSetValues_SeqBDiag_N(Mat,int,int *,int,int *,PetscScalar *,InsertMode);
EXTERN int MatGetValues_SeqBDiag_1(Mat,int,int *,int,int *,PetscScalar *);
EXTERN int MatGetValues_SeqBDiag_N(Mat,int,int *,int,int *,PetscScalar *);
EXTERN int MatMult_SeqBDiag_1(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_2(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_3(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_4(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_5(Mat,Vec,Vec);
EXTERN int MatMult_SeqBDiag_N(Mat,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_2(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_3(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_4(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_5(Mat,Vec,Vec,Vec);
EXTERN int MatMultAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
EXTERN int MatMultTranspose_SeqBDiag_1(Mat,Vec,Vec);
EXTERN int MatMultTranspose_SeqBDiag_N(Mat,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
EXTERN int MatRelax_SeqBDiag_N(Mat,Vec,PetscReal,MatSORType,PetscReal,int,Vec);
EXTERN int MatRelax_SeqBDiag_1(Mat,Vec,PetscReal,MatSORType,PetscReal,int,Vec);


#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_SeqBDiag"
int MatGetInfo_SeqBDiag(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;

  PetscFunctionBegin;
  info->rows_global       = (double)A->m;
  info->columns_global    = (double)A->n;
  info->rows_local        = (double)A->m;
  info->columns_local     = (double)A->n;
  info->block_size        = a->bs;
  info->nz_allocated      = (double)a->maxnz;
  info->nz_used           = (double)a->nz;
  info->nz_unneeded       = (double)(a->maxnz - a->nz);
  info->assemblies        = (double)A->num_ass;
  info->mallocs           = (double)a->reallocs;
  info->memory            = A->mem;
  info->fill_ratio_given  = 0; /* supports ILU(0) only */
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(0);
}

/*
     Note: this currently will generate different answers if you request
 all items or a subset. If you request all items it checks if the value is
 nonzero and only includes it if it is nonzero; if you check a subset of items
 it returns a list of all active columns in the row (some which may contain
 a zero)
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetRow_SeqBDiag"
int MatGetRow_SeqBDiag(Mat A,int row,int *nz,int **col,PetscScalar **v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  int          nd = a->nd,bs = a->bs;
  int          nc = A->n,*diag = a->diag,pcol,shift,i,j,k;

  PetscFunctionBegin;
  /* For efficiency, if ((nz) && (col) && (v)) then do all at once */
  if ((nz) && (col) && (v)) {
    *col = a->colloc;
    *v   = a->dvalue;
    k    = 0;
    if (bs == 1) { 
      for (j=0; j<nd; j++) {
        pcol = row - diag[j];
#if defined(PETSC_USE_COMPLEX)
        if (pcol > -1 && pcol < nc && PetscAbsScalar((a->diagv[j])[row])) {
#else
        if (pcol > -1 && pcol < nc && (a->diagv[j])[row]) {
#endif
	  (*v)[k]   = (a->diagv[j])[row];
          (*col)[k] = pcol; 
          k++;
	}
      }
      *nz = k;
    } else {
      shift = (row/bs)*bs*bs + row%bs;
      for (j=0; j<nd; j++) {
        pcol = bs * (row/bs - diag[j]);
        if (pcol > -1 && pcol < nc) {
          for (i=0; i<bs; i++) {
#if defined(PETSC_USE_COMPLEX)
            if (PetscAbsScalar((a->diagv[j])[shift + i*bs])) {
#else
            if ((a->diagv[j])[shift + i*bs]) {
#endif
              (*v)[k]   = (a->diagv[j])[shift + i*bs];
              (*col)[k] = pcol + i; 
              k++;
            }
	  }
        } 
      }
      *nz = k;
    }
  } else {
    if (bs == 1) { 
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) k++; 
        }
        *nz = k;
      }
      if (col) {
        *col = a->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
            (*col)[k] = pcol;  k++;
          }
        }
      }
      if (v) {
        *v = a->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
	    (*v)[k] = (a->diagv[j])[row]; k++;
          }
        }
      }
    } else {
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = bs * (row/bs- diag[j]);
          if (pcol > -1 && pcol < nc) k += bs; 
        }
        *nz = k;
      }
      if (col) {
        *col = a->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = bs * (row/bs - diag[j]);
          if (pcol > -1 && pcol < nc) {
            for (i=0; i<bs; i++) {
	      (*col)[k+i] = pcol + i;
            }
	    k += bs;
          }
        }
      }
      if (v) {
        shift = (row/bs)*bs*bs + row%bs;
        *v = a->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
	  pcol = bs * (row/bs - diag[j]);
	  if (pcol > -1 && pcol < nc) {
	    for (i=0; i<bs; i++) {
	     (*v)[k+i] = (a->diagv[j])[shift + i*bs];
            }
	    k += bs;
	  }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatRestoreRow_SeqBDiag"
int MatRestoreRow_SeqBDiag(Mat A,int row,int *ncols,int **cols,PetscScalar **vals)
{
  PetscFunctionBegin;
  /* Work space is allocated during matrix creation and freed
     when matrix is destroyed */
  PetscFunctionReturn(0);
}

/* 
   MatNorm_SeqBDiag_Columns - Computes the column norms of a block diagonal
   matrix.  We code this separately from MatNorm_SeqBDiag() so that the
   routine can be used for the parallel version as well.
 */
#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqBDiag_Columns"
int MatNorm_SeqBDiag_Columns(Mat A,PetscReal *tmp,int n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  int          d,i,j,k,nd = a->nd,bs = a->bs,diag,kshift,kloc,len,ierr;
  PetscScalar  *dv;

  PetscFunctionBegin;
  ierr = PetscMemzero(tmp,A->n*sizeof(PetscReal));CHKERRQ(ierr);
  if (bs == 1) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
      if (diag > 0) {	/* lower triangle */
        for (i=0; i<len; i++) {
          tmp[i] += PetscAbsScalar(dv[i+diag]); 
        }
      } else {	/* upper triangle */
        for (i=0; i<len; i++) {
          tmp[i-diag] += PetscAbsScalar(dv[i]); 
        }
      }
    }
  } else { 
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];

      if (diag > 0) {	/* lower triangle */
        for (k=0; k<len; k++) {
          kloc = k*bs; kshift = kloc*bs + diag*bs; 
          for (i=0; i<bs; i++) {	/* i = local row */
            for (j=0; j<bs; j++) {	/* j = local column */
              tmp[kloc + j] += PetscAbsScalar(dv[kshift + j*bs + i]);
            }
          }
        }
      } else {	/* upper triangle */
        for (k=0; k<len; k++) {
          kloc = k*bs; kshift = kloc*bs; 
          for (i=0; i<bs; i++) {	/* i = local row */
            for (j=0; j<bs; j++) {	/* j = local column */
              tmp[kloc + j - bs*diag] += PetscAbsScalar(dv[kshift + j*bs + i]);
            }
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatNorm_SeqBDiag"
int MatNorm_SeqBDiag(Mat A,NormType type,PetscReal *nrm)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  PetscReal    sum = 0.0,*tmp;
  int          ierr,d,i,j,k,nd = a->nd,bs = a->bs,diag,kshift,kloc,len;
  PetscScalar  *dv;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*bs*bs;
      diag = a->diag[d];
      if (diag > 0) {
        for (i=0; i<len; i++) {
#if defined(PETSC_USE_COMPLEX)
          sum += PetscRealPart(PetscConj(dv[i+diag])*dv[i+diag]);
#else
          sum += dv[i+diag]*dv[i+diag];
#endif
        }
      } else {
        for (i=0; i<len; i++) {
#if defined(PETSC_USE_COMPLEX)
          sum += PetscRealPart(PetscConj(dv[i])*dv[i]);
#else
          sum += dv[i]*dv[i];
#endif
        }
      }
    }
    *nrm = sqrt(sum);
  } else if (type == NORM_1) { /* max column norm */
    ierr = PetscMalloc((A->n+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = MatNorm_SeqBDiag_Columns(A,tmp,A->n);CHKERRQ(ierr);
    *nrm = 0.0;
    for (j=0; j<A->n; j++) {
      if (tmp[j] > *nrm) *nrm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else if (type == NORM_INFINITY) { /* max row norm */
    ierr = PetscMalloc((A->m+1)*sizeof(PetscReal),&tmp);CHKERRQ(ierr);
    ierr = PetscMemzero(tmp,A->m*sizeof(PetscReal));CHKERRQ(ierr);
    *nrm = 0.0;
    if (bs == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) {	/* lower triangle */
          for (i=0; i<len; i++) {
            tmp[i+diag] += PetscAbsScalar(dv[i+diag]); 
          }
        } else {	/* upper triangle */
          for (i=0; i<len; i++) {
            tmp[i] += PetscAbsScalar(dv[i]); 
          }
        }
      }
    } else { 
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) {
          for (k=0; k<len; k++) {
            kloc = k*bs; kshift = kloc*bs + bs*diag; 
            for (i=0; i<bs; i++) {	/* i = local row */
              for (j=0; j<bs; j++) {	/* j = local column */
                tmp[kloc + i + bs*diag] += PetscAbsScalar(dv[kshift+j*bs+i]);
              }
            }
          }
        } else {
          for (k=0; k<len; k++) {
            kloc = k*bs; kshift = kloc*bs; 
            for (i=0; i<bs; i++) {	/* i = local row */
              for (j=0; j<bs; j++) {	/* j = local column */
                tmp[kloc + i] += PetscAbsScalar(dv[kshift + j*bs + i]);
              }
            }
          }
        }
      }
    }
    for (j=0; j<A->m; j++) {
      if (tmp[j] > *nrm) *nrm = tmp[j];
    }
    ierr = PetscFree(tmp);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_SUP,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatTranspose_SeqBDiag"
int MatTranspose_SeqBDiag(Mat A,Mat *matout)
{ 
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data,*anew;
  Mat          tmat;
  int          i,j,k,d,ierr,nd = a->nd,*diag = a->diag,*diagnew;
  int          bs = a->bs,kshift,shifto,shiftn;
  PetscScalar  *dwork,*dvnew;

  PetscFunctionBegin;
  ierr = PetscMalloc((nd+1)*sizeof(int),&diagnew);CHKERRQ(ierr);
  for (i=0; i<nd; i++) {
    diagnew[i] = -diag[nd-i-1]; /* assume sorted in descending order */
  }
  ierr = MatCreateSeqBDiag(A->comm,A->n,A->m,nd,bs,diagnew,0,&tmat);CHKERRQ(ierr);
  ierr = PetscFree(diagnew);CHKERRQ(ierr);
  anew = (Mat_SeqBDiag*)tmat->data;
  for (d=0; d<nd; d++) {
    dvnew = anew->diagv[d];
    dwork = a->diagv[nd-d-1];
    if (anew->bdlen[d] != a->bdlen[nd-d-1]) SETERRQ(PETSC_ERR_ARG_SIZ,"Incompatible diagonal lengths");
    shifto = a->diag[nd-d-1];
    shiftn = anew->diag[d];
    if (shifto > 0)  shifto = bs*bs*shifto; else shifto = 0;
    if (shiftn > 0)  shiftn = bs*bs*shiftn; else shiftn = 0;
    if (bs == 1) {
      for (k=0; k<anew->bdlen[d]; k++) dvnew[shiftn+k] = dwork[shifto+k];
    } else {
      for (k=0; k<anew->bdlen[d]; k++) {
        kshift = k*bs*bs;
        for (i=0; i<bs; i++) {	/* i = local row */
          for (j=0; j<bs; j++) {	/* j = local column */
            dvnew[shiftn + kshift + j + i*bs] = dwork[shifto + kshift + j*bs + i];
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (matout) {
    *matout = tmat;
  } else {
    /* This isn't really an in-place transpose ... but free data 
       structures from a.  We should fix this. */
    if (!a->user_alloc) { /* Free the actual diagonals */
      for (i=0; i<a->nd; i++) {
        if (a->diag[i] > 0) {
          ierr = PetscFree(a->diagv[i] + bs*bs*a->diag[i]);CHKERRQ(ierr);
        } else {
          ierr = PetscFree(a->diagv[i]);CHKERRQ(ierr);
        }
      }
    }
    if (a->pivot) {ierr = PetscFree(a->pivot);CHKERRQ(ierr);}
    ierr = PetscFree(a->diagv);CHKERRQ(ierr);
    ierr = PetscFree(a->diag);CHKERRQ(ierr);
    ierr = PetscFree(a->colloc);CHKERRQ(ierr);
    ierr = PetscFree(a->dvalue);CHKERRQ(ierr);
    ierr = PetscFree(a);CHKERRQ(ierr);
    ierr = PetscMemcpy(A,tmat,sizeof(struct _p_Mat));CHKERRQ(ierr);
    PetscHeaderDestroy(tmat);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/


#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBDiag_Binary"
int MatView_SeqBDiag_Binary(Mat A,PetscViewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  int          i,ict,fd,*col_lens,*cval,*col,ierr,nz;
  PetscScalar  *anonz,*val;

  PetscFunctionBegin;
  ierr = PetscViewerBinaryGetDescriptor(viewer,&fd);CHKERRQ(ierr);

  /* For MATSEQBDIAG format,maxnz = nz */
  ierr        = PetscMalloc((4+A->m)*sizeof(int),&col_lens);CHKERRQ(ierr);
  col_lens[0] = MAT_FILE_COOKIE;
  col_lens[1] = A->m;
  col_lens[2] = A->n;
  col_lens[3] = a->maxnz;

  /* Should do translation using less memory; this is just a quick initial version */
  ierr = PetscMalloc((a->maxnz)*sizeof(int),&cval);CHKERRQ(ierr);
  ierr = PetscMalloc((a->maxnz)*sizeof(PetscScalar),&anonz);CHKERRQ(ierr);

  ict = 0;
  for (i=0; i<A->m; i++) {
    ierr = MatGetRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
    col_lens[4+i] = nz;
    ierr = PetscMemcpy(&cval[ict],col,nz*sizeof(int));CHKERRQ(ierr);
    ierr = PetscMemcpy(&anonz[ict],anonz,nz*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = MatRestoreRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
    ict += nz;
  }
  if (ict != a->maxnz) SETERRQ(PETSC_ERR_PLIB,"Error in nonzero count");

  /* Store lengths of each row and write (including header) to file */
  ierr = PetscBinaryWrite(fd,col_lens,4+A->m,PETSC_INT,1);CHKERRQ(ierr);
  ierr = PetscFree(col_lens);CHKERRQ(ierr);

  /* Store column indices (zero start index) */
  ierr = PetscBinaryWrite(fd,cval,a->maxnz,PETSC_INT,0);CHKERRQ(ierr);

  /* Store nonzero values */
  ierr = PetscBinaryWrite(fd,anonz,a->maxnz,PETSC_SCALAR,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBDiag_ASCII"
int MatView_SeqBDiag_ASCII(Mat A,PetscViewer viewer)
{
  Mat_SeqBDiag      *a = (Mat_SeqBDiag*)A->data;
  char              *name;
  int               ierr,*col,i,j,len,diag,nr = A->m,bs = a->bs,iprint,nz;
  PetscScalar       *val,*dv,zero = 0.0;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_LONG) {
    int nline = PetscMin(10,a->nd),k,nk,np;
    if (a->user_alloc) {
      ierr = PetscViewerASCIIPrintf(viewer,"block size=%d, number of diagonals=%d, user-allocated storage\n",bs,a->nd);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"block size=%d, number of diagonals=%d, PETSc-allocated storage\n",bs,a->nd);CHKERRQ(ierr);
    }
    nk = (a->nd-1)/nline + 1;
    for (k=0; k<nk; k++) {
      ierr = PetscViewerASCIIPrintf(viewer,"diag numbers:");CHKERRQ(ierr);
      np = PetscMin(nline,a->nd - nline*k);
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
      for (i=0; i<np; i++) {
        ierr = PetscViewerASCIIPrintf(viewer,"  %d",a->diag[i+nline*k]);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);        
      ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
    }
  } else if (format == PETSC_VIEWER_ASCII_MATLAB) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Size = %d %d \n",nr,A->n);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%% Nonzeros = %d \n",a->nz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = zeros(%d,3);\n",a->nz);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"zzz = [\n");CHKERRQ(ierr);
    for (i=0; i<A->m; i++) {
      ierr = MatGetRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
        if (val[j] != zero) {
#if defined(PETSC_USE_COMPLEX)
          ierr = PetscViewerASCIIPrintf(viewer,"%d %d  %18.16e  %18.16ei \n",
             i+1,col[j]+1,PetscRealPart(val[j]),PetscImaginaryPart(val[j]));CHKERRQ(ierr);
#else
          ierr = PetscViewerASCIIPrintf(viewer,"%d %d  %18.16ei \n",i+1,col[j]+1,val[j]);CHKERRQ(ierr);
#endif
        }
      }
      ierr = MatRestoreRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"];\n %s = spconvert(zzz);\n",name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else if (format == PETSC_VIEWER_ASCII_IMPL) {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    if (bs == 1) { /* diagonal format */
      for (i=0; i<a->nd; i++) {
        dv   = a->diagv[i];
        diag = a->diag[i];
        ierr = PetscViewerASCIIPrintf(viewer,"\n<diagonal %d>\n",diag);CHKERRQ(ierr);
        /* diag[i] is (row-col)/bs */
        if (diag > 0) {  /* lower triangle */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[diag+j] != zero) {
#if defined(PETSC_USE_COMPLEX)
              if (PetscImaginaryPart(dv[diag+j]) != 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %e + %e i\n",
  	                j+diag,j,PetscRealPart(dv[diag+j]),PetscImaginaryPart(dv[diag+j]));CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %e\n",j+diag,j,PetscRealPart(dv[diag+j]));CHKERRQ(ierr);
              }
#else
              ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %e\n",j+diag,j,dv[diag+j]);CHKERRQ(ierr);

#endif
            }
          }
        } else {         /* upper triangle, including main diagonal */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[j] != zero) {
#if defined(PETSC_USE_COMPLEX)
              if (PetscImaginaryPart(dv[j]) != 0.0) {
                ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %g + %g i\n",
                                         j,j-diag,PetscRealPart(dv[j]),PetscImaginaryPart(dv[j]));CHKERRQ(ierr);
              } else {
                ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %g\n",j,j-diag,PetscRealPart(dv[j]));CHKERRQ(ierr);
              }
#else
              ierr = PetscViewerASCIIPrintf(viewer,"A[ %d , %d ] = %g\n",j,j-diag,dv[j]);CHKERRQ(ierr);
#endif
            }
          }
        }
      }
    } else {  /* Block diagonals */
      int d,k,kshift;
      for (d=0; d< a->nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
	ierr = PetscViewerASCIIPrintf(viewer,"\n<diagonal %d>\n", diag);CHKERRQ(ierr);
	if (diag > 0) {		/* lower triangle */
	  for (k=0; k<len; k++) {
	    kshift = (diag+k)*bs*bs;
	    for (i=0; i<bs; i++) {
              iprint = 0;
	      for (j=0; j<bs; j++) {
		if (dv[kshift + j*bs + i] != zero) {
                  iprint = 1;
#if defined(PETSC_USE_COMPLEX)
                  if (PetscImaginaryPart(dv[kshift + j*bs + i])){
                    ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e + %5.2e i  ",(k+diag)*bs+i,k*bs+j,
                      PetscRealPart(dv[kshift + j*bs + i]),PetscImaginaryPart(dv[kshift + j*bs + i]));CHKERRQ(ierr);
                  } else {
                    ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e   ",(k+diag)*bs+i,k*bs+j,
                      PetscRealPart(dv[kshift + j*bs + i]));CHKERRQ(ierr);
                  }
#else
		  ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e   ",(k+diag)*bs+i,k*bs+j,
                      dv[kshift + j*bs + i]);CHKERRQ(ierr);
#endif
                }
              }
              if (iprint) {ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);}
            }
          }
        } else {		/* upper triangle, including main diagonal */
	  for (k=0; k<len; k++) {
	    kshift = k*bs*bs;
            for (i=0; i<bs; i++) {
              iprint = 0;
              for (j=0; j<bs; j++) {
                if (dv[kshift + j*bs + i] != zero) {
                  iprint = 1;
#if defined(PETSC_USE_COMPLEX)
                  if (PetscImaginaryPart(dv[kshift + j*bs + i])){
                    ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e + %5.2e i  ",k*bs+i,(k-diag)*bs+j,
                       PetscRealPart(dv[kshift + j*bs + i]),PetscImaginaryPart(dv[kshift + j*bs + i]));CHKERRQ(ierr);
                  } else {
                    ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e   ",k*bs+i,(k-diag)*bs+j,
                       PetscRealPart(dv[kshift + j*bs + i]));CHKERRQ(ierr);
                  }
#else
                  ierr = PetscViewerASCIIPrintf(viewer,"A[%d,%d]=%5.2e   ",k*bs+i,(k-diag)*bs+j,
                     dv[kshift + j*bs + i]);CHKERRQ(ierr);
#endif
                }
              }
              if (iprint) {ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);}
            }
          }
        }
      }
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_NO);CHKERRQ(ierr);
    /* the usual row format (PETSC_VIEWER_ASCII_NONZERO_ONLY) */
    for (i=0; i<A->m; i++) {
      ierr = PetscViewerASCIIPrintf(viewer,"row %d:",i);CHKERRQ(ierr);
      ierr = MatGetRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
#if defined(PETSC_USE_COMPLEX)
        if (PetscImaginaryPart(val[j]) != 0.0 && PetscRealPart(val[j]) != 0.0) {
          ierr = PetscViewerASCIIPrintf(viewer," %d %g + %g i ",col[j],PetscRealPart(val[j]),PetscImaginaryPart(val[j]));CHKERRQ(ierr);
        } else if (PetscRealPart(val[j]) != 0.0) {
	  ierr = PetscViewerASCIIPrintf(viewer," %d %g ",col[j],PetscRealPart(val[j]));CHKERRQ(ierr);
        }
#else
        if (val[j] != 0.0) {ierr = PetscViewerASCIIPrintf(viewer," %d %g ",col[j],val[j]);CHKERRQ(ierr);}
#endif
      }
      ierr = PetscViewerASCIIPrintf(viewer,"\n");CHKERRQ(ierr);
      ierr = MatRestoreRow_SeqBDiag(A,i,&nz,&col,&val);CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIUseTabs(viewer,PETSC_YES);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBDiag_Draw"
static int MatView_SeqBDiag_Draw(Mat A,PetscViewer viewer)
{
  PetscDraw     draw;
  PetscReal     xl,yl,xr,yr,w,h;
  int           ierr,nz,*col,i,j,nr = A->m;
  PetscTruth    isnull;

  PetscFunctionBegin;
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawIsNull(draw,&isnull);CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  xr = A->n; yr = A->m; h = yr/10.0; w = xr/10.0;
  xr += w; yr += h; xl = -w; yl = -h;
  ierr = PetscDrawSetCoordinates(draw,xl,yl,xr,yr);CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes; we really should do this
     by diagonals.  What do we really want to draw here: nonzeros, 
     allocated space? */
  for (i=0; i<nr; i++) {
    yl = nr - i - 1.0; yr = yl + 1.0;
    ierr = MatGetRow_SeqBDiag(A,i,&nz,&col,0);CHKERRQ(ierr);
    for (j=0; j<nz; j++) {
      xl = col[j]; xr = xl + 1.0;
      ierr = PetscDrawRectangle(draw,xl,yl,xr,yr,PETSC_DRAW_BLACK,PETSC_DRAW_BLACK,
			   PETSC_DRAW_BLACK,PETSC_DRAW_BLACK);CHKERRQ(ierr);
    }
    ierr = MatRestoreRow_SeqBDiag(A,i,&nz,&col,0);CHKERRQ(ierr);
  }
  ierr = PetscDrawFlush(draw);CHKERRQ(ierr);
  ierr = PetscDrawPause(draw);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatView_SeqBDiag"
int MatView_SeqBDiag(Mat A,PetscViewer viewer)
{
  int        ierr;
  PetscTruth isascii,isbinary,isdraw;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_DRAW,&isdraw);CHKERRQ(ierr);
  if (isascii) {
    ierr = MatView_SeqBDiag_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (isbinary) {
    ierr = MatView_SeqBDiag_Binary(A,viewer);CHKERRQ(ierr);
  } else if (isdraw) {
    ierr = MatView_SeqBDiag_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported by BDiag matrices",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

