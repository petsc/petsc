#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: bdiag.c,v 1.159 1998/06/03 21:31:09 balay Exp balay $";
#endif

/* Block diagonal matrix format */

#include "pinclude/pviewer.h"
#include "sys.h"
#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/vec/vecimpl.h"
#include "src/inline/ilu.h"

extern int MatSetValues_SeqBDiag_1(Mat,int,int *,int,int *,Scalar *,InsertMode);
extern int MatSetValues_SeqBDiag_N(Mat,int,int *,int,int *,Scalar *,InsertMode);
extern int MatGetValues_SeqBDiag_1(Mat,int,int *,int,int *,Scalar *);
extern int MatGetValues_SeqBDiag_N(Mat,int,int *,int,int *,Scalar *);
extern int MatMult_SeqBDiag_1(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_2(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_3(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_4(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_5(Mat,Vec,Vec);
extern int MatMult_SeqBDiag_N(Mat,Vec,Vec);
extern int MatMultAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_2(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_3(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_4(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_5(Mat,Vec,Vec,Vec);
extern int MatMultAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
extern int MatMultTrans_SeqBDiag_1(Mat,Vec,Vec);
extern int MatMultTrans_SeqBDiag_N(Mat,Vec,Vec);
extern int MatMultTransAdd_SeqBDiag_1(Mat,Vec,Vec,Vec);
extern int MatMultTransAdd_SeqBDiag_N(Mat,Vec,Vec,Vec);
extern int MatRelax_SeqBDiag_N(Mat,Vec,double,MatSORType,double,int,Vec);
extern int MatRelax_SeqBDiag_1(Mat,Vec,double,MatSORType,double,int,Vec);


#undef __FUNC__  
#define __FUNC__ "MatGetInfo_SeqBDiag"
int MatGetInfo_SeqBDiag(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  info->rows_global       = (double)a->m;
  info->columns_global    = (double)a->n;
  info->rows_local        = (double)a->m;
  info->columns_local     = (double)a->n;
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

#undef __FUNC__  
#define __FUNC__ "MatGetOwnershipRange_SeqBDiag" 
int MatGetOwnershipRange_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *m = 0; *n = a->m;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetRow_SeqBDiag"
int MatGetRow_SeqBDiag(Mat A,int row,int *nz,int **col,Scalar **v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nd = a->nd, bs = a->bs;
  int          nc = a->n, *diag = a->diag, pcol, shift, i, j, k;

  PetscFunctionBegin;
  /* For efficiency, if ((nz) && (col) && (v)) then do all at once */
  if ((nz) && (col) && (v)) {
    *col = a->colloc;
    *v   = a->dvalue;
    k    = 0;
    if (bs == 1) { 
      for (j=0; j<nd; j++) {
        pcol = row - diag[j];
        if (pcol > -1 && pcol < nc) {
	  (*v)[k]   = (a->diagv[j])[row];
          (*col)[k] = pcol;  k++;
	}
      }
      *nz = k;
    } else {
      shift = (row/bs)*bs*bs + row%bs;
      for (j=0; j<nd; j++) {
        pcol = bs * (row/bs - diag[j]);
        if (pcol > -1 && pcol < nc) {
          for (i=0; i<bs; i++) {
	    (*v)[k+i]   = (a->diagv[j])[shift + i*bs];
	    (*col)[k+i] = pcol + i;
	  }
          k += bs;
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

#undef __FUNC__  
#define __FUNC__ "MatRestoreRow_SeqBDiag"
int MatRestoreRow_SeqBDiag(Mat A,int row,int *ncols,int **cols,Scalar **vals)
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
#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqBDiag_Columns"
int MatNorm_SeqBDiag_Columns(Mat A,double *tmp,int n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          d, i, j, k, nd = a->nd, bs = a->bs, diag, kshift, kloc, len;
  Scalar       *dv;

  PetscFunctionBegin;
  PetscMemzero(tmp,a->n*sizeof(double));
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

#undef __FUNC__  
#define __FUNC__ "MatNorm_SeqBDiag"
int MatNorm_SeqBDiag(Mat A,NormType type,double *norm)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  double       sum = 0.0, *tmp;
  int          ierr,d,i,j,k,nd = a->nd,bs = a->bs,diag, kshift, kloc, len;
  Scalar       *dv;

  PetscFunctionBegin;
  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*bs*bs;
      diag = a->diag[d];
      if (diag > 0) {
        for (i=0; i<len; i++) {
#if defined(USE_PETSC_COMPLEX)
          sum += PetscReal(PetscConj(dv[i+diag])*dv[i+diag]);
#else
          sum += dv[i+diag]*dv[i+diag];
#endif
        }
      } else {
        for (i=0; i<len; i++) {
#if defined(USE_PETSC_COMPLEX)
          sum += PetscReal(PetscConj(dv[i])*dv[i]);
#else
          sum += dv[i]*dv[i];
#endif
        }
      }
    }
    *norm = sqrt(sum);
  } else if (type == NORM_1) { /* max column norm */
    tmp = (double *) PetscMalloc( (a->n+1)*sizeof(double) ); CHKPTRQ(tmp);
    ierr = MatNorm_SeqBDiag_Columns(A,tmp,a->n); CHKERRQ(ierr);
    *norm = 0.0;
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  } else if (type == NORM_INFINITY) { /* max row norm */
    tmp = (double *) PetscMalloc( (a->m+1)*sizeof(double) ); CHKPTRQ(tmp);
    PetscMemzero(tmp,a->m*sizeof(double));
    *norm = 0.0;
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
    for ( j=0; j<a->m; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PetscFree(tmp);
  } else {
    SETERRQ(PETSC_ERR_SUP,0,"No support for two norm");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatTranspose_SeqBDiag"
int MatTranspose_SeqBDiag(Mat A,Mat *matout)
{ 
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data, *anew;
  Mat          tmat;
  int          i, j, k, d, ierr, nd = a->nd, *diag = a->diag, *diagnew;
  int          bs = a->bs, kshift,shifto,shiftn;
  Scalar       *dwork, *dvnew;

  PetscFunctionBegin;
  diagnew = (int *) PetscMalloc((nd+1)*sizeof(int)); CHKPTRQ(diagnew);
  for (i=0; i<nd; i++) {
    diagnew[i] = -diag[nd-i-1]; /* assume sorted in descending order */
  }
  ierr = MatCreateSeqBDiag(A->comm,a->n,a->m,nd,bs,diagnew,0,&tmat);CHKERRQ(ierr);
  PetscFree(diagnew);
  anew = (Mat_SeqBDiag *) tmat->data;
  for (d=0; d<nd; d++) {
    dvnew = anew->diagv[d];
    dwork = a->diagv[nd-d-1];
    if (anew->bdlen[d] != a->bdlen[nd-d-1]) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Incompatible diagonal lengths");
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
  ierr = MatAssemblyBegin(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  if (matout != PETSC_NULL) {
    *matout = tmat;
  } else {
    /* This isn't really an in-place transpose ... but free data 
       structures from a.  We should fix this. */
    if (!a->user_alloc) { /* Free the actual diagonals */
      for (i=0; i<a->nd; i++) {
        if (a->diag[i] > 0) {
          PetscFree( a->diagv[i] + bs*bs*a->diag[i]  );
        } else {
          PetscFree( a->diagv[i] );
        }
      }
    }
    if (a->pivot) PetscFree(a->pivot);
    PetscFree(a->diagv); PetscFree(a->diag);
    PetscFree(a->colloc); PetscFree(a->dvalue);
    PetscFree(a);
    PetscMemcpy(A,tmat,sizeof(struct _p_Mat)); 
    PetscHeaderDestroy(tmat);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------*/


#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_Binary"
static int MatView_SeqBDiag_Binary(Mat A,Viewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ict, fd, *col_lens, *cval, *col, ierr, nz;
  Scalar       *anonz, *val;

  PetscFunctionBegin;
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);

  /* For MATSEQBDIAG format, maxnz = nz */
  col_lens    = (int *) PetscMalloc( (4+a->m)*sizeof(int) ); CHKPTRQ(col_lens);
  col_lens[0] = MAT_COOKIE;
  col_lens[1] = a->m;
  col_lens[2] = a->n;
  col_lens[3] = a->maxnz;

  /* Should do translation using less memory; this is just a quick initial version */
  cval  = (int *) PetscMalloc( (a->maxnz)*sizeof(int) ); CHKPTRQ(cval);
  anonz = (Scalar *) PetscMalloc( (a->maxnz)*sizeof(Scalar) ); CHKPTRQ(anonz);

  ict = 0;
  for (i=0; i<a->m; i++) {
    ierr = MatGetRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    col_lens[4+i] = nz;
    PetscMemcpy(&cval[ict],col,nz*sizeof(int)); CHKERRQ(ierr);
    PetscMemcpy(&anonz[ict],anonz,nz*sizeof(Scalar)); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    ict += nz;
  }
  if (ict != a->maxnz) SETERRQ(PETSC_ERR_PLIB,0,"Error in nonzero count");

  /* Store lengths of each row and write (including header) to file */
  ierr = PetscBinaryWrite(fd,col_lens,4+a->m,PETSC_INT,1); CHKERRQ(ierr);
  PetscFree(col_lens);

  /* Store column indices (zero start index) */
  ierr = PetscBinaryWrite(fd,cval,a->maxnz,PETSC_INT,0); CHKERRQ(ierr);

  /* Store nonzero values */
  ierr = PetscBinaryWrite(fd,anonz,a->maxnz,PETSC_SCALAR,0); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_ASCII"
static int MatView_SeqBDiag_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  FILE         *fd;
  char         *outputname;
  int          ierr, *col, i, j, len, diag, nr = a->m, bs = a->bs, format, iprint, nz;
  Scalar       *val, *dv, zero = 0.0;

  PetscFunctionBegin;
  ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerGetFormat(viewer,&format); CHKERRQ(ierr);
  if (format == VIEWER_FORMAT_ASCII_INFO || format == VIEWER_FORMAT_ASCII_INFO_LONG) {
    int nline = PetscMin(10,a->nd), k, nk, np;
    if (a->user_alloc)
      fprintf(fd,"  block size=%d, number of diagonals=%d, user-allocated storage\n",bs,a->nd);
    else
      fprintf(fd,"  block size=%d, number of diagonals=%d, PETSc-allocated storage\n",bs,a->nd);
    nk = (a->nd-1)/nline + 1;
    for (k=0; k<nk; k++) {
      fprintf(fd,"  diag numbers:");
      np = PetscMin(nline,a->nd - nline*k);
      for (i=0; i<np; i++) 
        fprintf(fd,"  %d",a->diag[i+nline*k]);
      fprintf(fd,"\n");        
    }
  } else if (format == VIEWER_FORMAT_ASCII_MATLAB) {
    fprintf(fd,"%% Size = %d %d \n",nr, a->n);
    fprintf(fd,"%% Nonzeros = %d \n",a->nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",a->nz);
    fprintf(fd,"zzz = [\n");
    for ( i=0; i<a->m; i++ ) {
      ierr = MatGetRow( A, i, &nz, &col, &val ); CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
        if (val[j] != zero)
#if defined(USE_PETSC_COMPLEX)
          fprintf(fd,"%d %d  %18.16e  %18.16e \n",
             i+1, col[j]+1, PetscReal(val[j]), PetscImaginary(val[j]) );
#else
          fprintf(fd,"%d %d  %18.16e\n", i+1, col[j]+1, val[j]);
#endif
      }
      ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    }
    fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
  } else if (format == VIEWER_FORMAT_ASCII_IMPL) {
    if (bs == 1) { /* diagonal format */
      for (i=0; i<a->nd; i++) {
        dv   = a->diagv[i];
        diag = a->diag[i];
        fprintf(fd,"\n<diagonal %d>\n",diag);
        /* diag[i] is (row-col)/bs */
        if (diag > 0) {  /* lower triangle */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[diag+j] != zero) {
#if defined(USE_PETSC_COMPLEX)
              if (PetscImaginary(dv[diag+j]) != 0.0) fprintf(fd,"A[ %d , %d ] = %e + %e i\n",
                                     j+diag,j,PetscReal(dv[diag+j]),PetscImaginary(dv[diag+j]));
              else fprintf(fd,"A[ %d , %d ] = %e\n",j+diag,j,PetscReal(dv[diag+j]));
#else
              fprintf(fd,"A[ %d , %d ] = %e\n",j+diag,j,dv[diag+j]);

#endif
            }
          }
        } else {         /* upper triangle, including main diagonal */
          len  = a->bdlen[i];
          for (j=0; j<len; j++) {
            if (dv[j] != zero) {
#if defined(USE_PETSC_COMPLEX)
              if (PetscImaginary(dv[j]) != 0.0) fprintf(fd,"A[ %d , %d ] = %e + %e i\n",
                                         j,j-diag,PetscReal(dv[j]),PetscImaginary(dv[j]));
              else fprintf(fd,"A[ %d , %d ] = %e\n",j,j-diag,PetscReal(dv[j]));
#else
              fprintf(fd,"A[ %d , %d ] = %e\n",j,j-diag,dv[j]);
#endif
            }
          }
        }
      }
    } else {  /* Block diagonals */
      int d, k, kshift;
      for (d=0; d< a->nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
	fprintf(fd,"\n<diagonal %d>\n", diag);
	if (diag > 0) {		/* lower triangle */
	  for (k=0; k<len; k++) {
	    kshift = (diag+k)*bs*bs;
	    for (i=0; i<bs; i++) {
              iprint = 0;
	      for (j=0; j<bs; j++) {
		if (dv[kshift + j*bs + i] != zero) {
                  iprint = 1;
#if defined(USE_PETSC_COMPLEX)
                  if (PetscImaginary(dv[kshift + j*bs + i]))
                    fprintf(fd,"A[%d,%d]=%5.2e + %5.2e i  ",(k+diag)*bs+i,k*bs+j,
                      PetscReal(dv[kshift + j*bs + i]),PetscImaginary(dv[kshift + j*bs + i]));
                  else
                    fprintf(fd,"A[%d,%d]=%5.2e   ",(k+diag)*bs+i,k*bs+j,
                      PetscReal(dv[kshift + j*bs + i]));
#else
		  fprintf(fd,"A[%d,%d]=%5.2e   ", (k+diag)*bs+i,k*bs+j,
                      dv[kshift + j*bs + i]);
#endif
                }
              }
              if (iprint) fprintf(fd,"\n");
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
#if defined(USE_PETSC_COMPLEX)
                  if (PetscImaginary(dv[kshift + j*bs + i]))
                    fprintf(fd,"A[%d,%d]=%5.2e + %5.2e i  ", k*bs+i,(k-diag)*bs+j,
                       PetscReal(dv[kshift + j*bs + i]),PetscImaginary(dv[kshift + j*bs + i]));
                  else
                    fprintf(fd,"A[%d,%d]=%5.2e   ", k*bs+i,(k-diag)*bs+j,
                       PetscReal(dv[kshift + j*bs + i]));
#else
                  fprintf(fd,"A[%d,%d]=%5.2e   ", k*bs+i,(k-diag)*bs+j,
                     dv[kshift + j*bs + i]);
#endif
                }
              }
              if (iprint) fprintf(fd,"\n");
            }
          }
        }
      }
    }
  } else {
    /* the usual row format (VIEWER_FORMAT_ASCII_NONZERO_ONLY) */
    for (i=0; i<a->m; i++) {
      fprintf(fd,"row %d:",i);
      ierr = MatGetRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
      for (j=0; j<nz; j++) {
#if defined(USE_PETSC_COMPLEX)
        if (PetscImaginary(val[j]) != 0.0 && PetscReal(val[j]) != 0.0)
          fprintf(fd," %d %g + %g i ",col[j],PetscReal(val[j]),PetscImaginary(val[j]));
        else if (PetscReal(val[j]) != 0.0)
	  fprintf(fd," %d %g ",col[j],PetscReal(val[j]));
#else
        if (val[j] != 0.0) fprintf(fd," %d %g ",col[j],val[j]);
#endif
      }
      fprintf(fd,"\n");
      ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    }
  }
  fflush(fd);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag_Draw"
static int MatView_SeqBDiag_Draw(Mat A,Viewer viewer)
{
  Mat_SeqBDiag  *a = (Mat_SeqBDiag *) A->data;
  Draw          draw;
  double        xl, yl, xr, yr, w, h;
  int           ierr, nz, *col, i, j, nr = a->m;
  PetscTruth    isnull;

  PetscFunctionBegin;
  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRQ(ierr);
  ierr = DrawIsNull(draw,&isnull); CHKERRQ(ierr); if (isnull) PetscFunctionReturn(0);

  xr = a->n; yr = a->m; h = yr/10.0; w = xr/10.0;
  xr += w; yr += h; xl = -w; yl = -h;
  ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);

  /* loop over matrix elements drawing boxes; we really should do this
     by diagonals.  What do we really want to draw here: nonzeros, 
     allocated space? */
  for ( i=0; i<nr; i++ ) {
    yl = nr - i - 1.0; yr = yl + 1.0;
    ierr = MatGetRow(A,i,&nz,&col,0); CHKERRQ(ierr);
    for ( j=0; j<nz; j++ ) {
      xl = col[j]; xr = xl + 1.0;
      ierr = DrawRectangle(draw,xl,yl,xr,yr,DRAW_BLACK,DRAW_BLACK,
			   DRAW_BLACK,DRAW_BLACK); CHKERRQ(ierr);
    }
    ierr = MatRestoreRow(A,i,&nz,&col,0); CHKERRQ(ierr);
  }
  ierr = DrawFlush(draw); CHKERRQ(ierr);
  DrawPause(draw); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatView_SeqBDiag"
int MatView_SeqBDiag(Mat A,Viewer viewer)
{
  ViewerType  vtype;
  int         ierr;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == MATLAB_VIEWER) {
    SETERRQ(PETSC_ERR_SUP,0,"Matlab viewer");
  } else if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER){
    ierr = MatView_SeqBDiag_ASCII(A,viewer);CHKERRQ(ierr);
  } else if (vtype == BINARY_FILE_VIEWER) {
    ierr = MatView_SeqBDiag_Binary(A,viewer);CHKERRQ(ierr);
  } else if (vtype == DRAW_VIEWER) {
    ierr = MatView_SeqBDiag_Draw(A,viewer);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Viewer type not supported by PETSc object");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDestroy_SeqBDiag"
int MatDestroy_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, bs = a->bs,ierr;

  PetscFunctionBegin;
  if (--A->refct > 0) PetscFunctionReturn(0);

  if (A->mapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->mapping); CHKERRQ(ierr);
  }
  if (A->bmapping) {
    ierr = ISLocalToGlobalMappingDestroy(A->bmapping); CHKERRQ(ierr);
  }
#if defined(USE_PETSC_LOG)
  PLogObjectState((PetscObject)A,"Rows=%d, Cols=%d, NZ=%d, BSize=%d, NDiag=%d",
                       a->m,a->n,a->nz,a->bs,a->nd);
#endif
  if (!a->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<a->nd; i++) {
      if (a->diag[i] > 0) {
        PetscFree( a->diagv[i] + bs*bs*a->diag[i]  );
      } else {
        PetscFree( a->diagv[i] );
      }
    }
  }
  if (a->pivot) PetscFree(a->pivot);
  PetscFree(a->diagv); PetscFree(a->diag);
  PetscFree(a->colloc);
  PetscFree(a->dvalue);
  PetscFree(a);
  PLogObjectDestroy(A);
  PetscHeaderDestroy(A);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatAssemblyEnd_SeqBDiag"
int MatAssemblyEnd_SeqBDiag(Mat A,MatAssemblyType mode)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, k, temp, *diag = a->diag, *bdlen = a->bdlen;
  Scalar       *dtemp, **dv = a->diagv;

  PetscFunctionBegin;
  if (mode == MAT_FLUSH_ASSEMBLY) PetscFunctionReturn(0);

  /* Sort diagonals */
  for (i=0; i<a->nd; i++) {
    for (k=i+1; k<a->nd; k++) {
      if (diag[i] < diag[k]) {
        temp     = diag[i];   
        diag[i]  = diag[k];
        diag[k]  = temp;
        temp     = bdlen[i];   
        bdlen[i] = bdlen[k];
        bdlen[k] = temp;
        dtemp    = dv[i];
        dv[i]    = dv[k];
        dv[k]    = dtemp;
      }
    }
  }

  /* Set location of main diagonal */
  for (i=0; i<a->nd; i++) {
    if (a->diag[i] == 0) {a->mainbd = i; break;}
  }
  PLogInfo(A,"MatAssemblyEnd_SeqBDiag:Number diagonals %d, memory used %d, block size %d\n", 
           a->nd,a->maxnz,a->bs);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatSetOption_SeqBDiag"
int MatSetOption_SeqBDiag(Mat A,MatOption op)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  if (op == MAT_NO_NEW_NONZERO_LOCATIONS)       a->nonew       = 1;
  else if (op == MAT_YES_NEW_NONZERO_LOCATIONS) a->nonew       = 0;
  else if (op == MAT_NO_NEW_DIAGONALS)          a->nonew_diag  = 1;
  else if (op == MAT_YES_NEW_DIAGONALS)         a->nonew_diag  = 0;
  else if (op == MAT_COLUMN_ORIENTED)           a->roworiented = 0;
  else if (op == MAT_ROW_ORIENTED)              a->roworiented = 1;
  else if (op == MAT_ROWS_SORTED || 
           op == MAT_ROWS_UNSORTED || 
           op == MAT_COLUMNS_SORTED || 
           op == MAT_COLUMNS_UNSORTED || 
           op == MAT_SYMMETRIC ||
           op == MAT_IGNORE_OFF_PROC_ENTRIES ||
           op == MAT_STRUCTURALLY_SYMMETRIC ||
           op == MAT_NEW_NONZERO_LOCATION_ERROR ||
           op == MAT_NEW_NONZERO_ALLOCATION_ERROR ||
           op == MAT_USE_HASH_TABLE)
    PLogInfo(A,"MatSetOption_SeqBDiag:Option ignored\n");
  else { 
    SETERRQ(PETSC_ERR_SUP,0,"unknown option");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatPrintHelp_SeqBDiag"
int MatPrintHelp_SeqBDiag(Mat A)
{
  static int called = 0; 
  MPI_Comm   comm = A->comm;

  PetscFunctionBegin;
  if (called) {PetscFunctionReturn(0);} else called = 1;
  (*PetscHelpPrintf)(comm," Options for MATSEQBDIAG and MATMPIBDIAG matrix formats:\n");
  (*PetscHelpPrintf)(comm,"  -mat_block_size <block_size>\n");
  (*PetscHelpPrintf)(comm,"  -mat_bdiag_diags <d1,d2,d3,...> (diagonal numbers)\n"); 
  (*PetscHelpPrintf)(comm,"   (for example) -mat_bdiag_diags -5,-1,0,1,5\n"); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqBDiag_N"
static int MatGetDiagonal_SeqBDiag_N(Mat A,Vec v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr,i, j, n, len, ibase, bs = a->bs, iloc;
  Scalar       *x, *dd, zero = 0.0;

  PetscFunctionBegin;
  if (A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Not for factored matrix");  
  ierr = VecSet(&zero,v);CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal not set");
  len = PetscMin(a->mblock,a->nblock);
  dd = a->diagv[a->mainbd];
  for (i=0; i<len; i++) {
    ibase = i*bs*bs;  iloc = i*bs;
    for (j=0; j<bs; j++) x[j + iloc] = dd[ibase + j*(bs+1)];
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetDiagonal_SeqBDiag_1"
static int MatGetDiagonal_SeqBDiag_1(Mat A,Vec v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          ierr,i, n, len;
  Scalar       *x, *dd, zero = 0.0;

  PetscFunctionBegin;
  ierr = VecSet(&zero,v); CHKERRQ(ierr);
  ierr = VecGetArray(v,&x); CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&n);CHKERRQ(ierr);
  if (n != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Nonconforming mat and vec");
  if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal not set");
  dd = a->diagv[a->mainbd];
  len = PetscMin(a->m,a->n);
  for (i=0; i<len; i++) x[i] = dd[i];
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroEntries_SeqBDiag"
int MatZeroEntries_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          d, i, len, bs = a->bs;
  Scalar       *dv;

  PetscFunctionBegin;
  for (d=0; d<a->nd; d++) {
    dv  = a->diagv[d];
    if (a->diag[d] > 0) {
      dv += bs*bs*a->diag[d];
    }
    len = a->bdlen[d]*bs*bs;
    for (i=0; i<len; i++) dv[i] = 0.0;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetBlockSize_SeqBDiag"
int MatGetBlockSize_SeqBDiag(Mat A,int *bs)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *bs = a->bs;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatZeroRows_SeqBDiag"
int MatZeroRows_SeqBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, N, *rows, m = a->m - 1, nz, *col;
  Scalar       *dd, *val;

  PetscFunctionBegin;
  ierr = ISGetSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  for ( i=0; i<N; i++ ) {
    if (rows[i]<0 || rows[i]>m) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,0,"row out of range");
    ierr = MatGetRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
    PetscMemzero(val,nz*sizeof(Scalar));
    ierr = MatSetValues(A,1,&rows[i],nz,col,val,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
  }
  if (diag) {
    if (a->mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Main diagonal does not exist");
    dd = a->diagv[a->mainbd];
    for ( i=0; i<N; i++ ) dd[rows[i]] = *diag;
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSize_SeqBDiag"
int MatGetSize_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;

  PetscFunctionBegin;
  *m = a->m; *n = a->n;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrix_SeqBDiag"
int MatGetSubMatrix_SeqBDiag(Mat A,IS isrow,IS iscol,MatGetSubMatrixCall scall,
                                    Mat *submat)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nznew, *smap, i, j, ierr, oldcols = a->n;
  int          *irow, *icol, newr, newc, *cwork, *col,nz, bs;
  Scalar       *vwork, *val;
  Mat          newmat;

  PetscFunctionBegin;
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&newr); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&newc); CHKERRQ(ierr);

  smap  = (int *) PetscMalloc((oldcols+1)*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PetscMalloc((newc+1)*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PetscMalloc((newc+1)*sizeof(Scalar)); CHKPTRQ(vwork);
  PetscMemzero((char*)smap,oldcols*sizeof(int));
  for ( i=0; i<newc; i++ ) smap[icol[i]] = i+1;

  /* Determine diagonals; then create submatrix */
  bs = a->bs; /* Default block size remains the same */
  ierr = MatCreateSeqBDiag(A->comm,newr,newc,0,bs,0,0,&newmat); CHKERRQ(ierr); 

  /* Fill new matrix */
  for (i=0; i<newr; i++) {
    ierr = MatGetRow(A,irow[i],&nz,&col,&val); CHKERRQ(ierr);
    nznew = 0;
    for (j=0; j<nz; j++) {
      if (smap[col[j]]) {
        cwork[nznew]   = smap[col[j]] - 1;
        vwork[nznew++] = val[j];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PetscFree(smap); PetscFree(cwork); PetscFree(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatGetSubMatrices_SeqBDiag"
int MatGetSubMatrices_SeqBDiag(Mat A,int n, IS *irow,IS *icol,MatGetSubMatrixCall scall,
                                    Mat **B)
{
  int ierr,i;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX) {
    *B = (Mat *) PetscMalloc( (n+1)*sizeof(Mat) ); CHKPTRQ(*B);
  }

  for ( i=0; i<n; i++ ) {
    ierr = MatGetSubMatrix_SeqBDiag(A,irow[i],icol[i],scall,&(*B)[i]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatScale_SeqBDiag"
int MatScale_SeqBDiag(Scalar *alpha,Mat inA)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) inA->data;
  int          one = 1, i, len, bs = a->bs;

  PetscFunctionBegin;
  for (i=0; i<a->nd; i++) {
    len = bs*bs*a->bdlen[i];
    if (a->diag[i] > 0) {
      BLscal_( &len, alpha, a->diagv[i] + bs*bs*a->diag[i], &one );
    } else {
      BLscal_( &len, alpha, a->diagv[i], &one );
    }
  }
  PLogFlops(a->nz);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatDiagonalScale_SeqBDiag"
int MatDiagonalScale_SeqBDiag(Mat A,Vec ll,Vec rr)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  Scalar       *l,*r, *dv;
  int          d, j, len;
  int          nd = a->nd, bs = a->bs, diag, m, n;

  PetscFunctionBegin;
  if (ll) {
    VecGetArray(ll,&l); VecGetSize(ll,&m);
    if (m != a->m) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Left scaling vector wrong length");
    if (bs == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= l[j+diag];
        else          for (j=0; j<len; j++) dv[j]      *= l[j];
      }
      PLogFlops(a->nz);
    } else SETERRQ(PETSC_ERR_SUP,0,"Not yet done for bs>1");
  }
  if (rr) {
    VecGetArray(rr,&r); VecGetSize(rr,&n);
    if (n != a->n) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Right scaling vector wrong length");
    if (bs == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
        if (diag > 0) for (j=0; j<len; j++) dv[j+diag] *= r[j];
        else          for (j=0; j<len; j++) dv[j]      *= r[j-diag];
      }
      PLogFlops(a->nz);
    } else SETERRQ(PETSC_ERR_SUP,0,"Not yet done for bs>1");
  }
  PetscFunctionReturn(0);
}

static int MatConvertSameType_SeqBDiag(Mat,Mat *,int);
extern int MatLUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,Mat*);
extern int MatILUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,int,Mat*);
extern int MatILUFactor_SeqBDiag(Mat,IS,IS,double,int);
extern int MatLUFactorNumeric_SeqBDiag_N(Mat,Mat*);
extern int MatLUFactorNumeric_SeqBDiag_1(Mat,Mat*);
extern int MatSolve_SeqBDiag_1(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_2(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_3(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_4(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_5(Mat,Vec,Vec);
extern int MatSolve_SeqBDiag_N(Mat,Vec,Vec);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqBDiag_N,
       MatGetRow_SeqBDiag,MatRestoreRow_SeqBDiag,
       MatMult_SeqBDiag_N,MatMultAdd_SeqBDiag_N, 
       MatMultTrans_SeqBDiag_N,MatMultTransAdd_SeqBDiag_N, 
       MatSolve_SeqBDiag_N,0,
       0,0,0,0,
       MatRelax_SeqBDiag_N,MatTranspose_SeqBDiag,
       MatGetInfo_SeqBDiag,0,
       MatGetDiagonal_SeqBDiag_N,MatDiagonalScale_SeqBDiag,MatNorm_SeqBDiag,
       0,MatAssemblyEnd_SeqBDiag,
       0,MatSetOption_SeqBDiag,MatZeroEntries_SeqBDiag,MatZeroRows_SeqBDiag,
       0,MatLUFactorNumeric_SeqBDiag_N,0,0,
       MatGetSize_SeqBDiag,MatGetSize_SeqBDiag,MatGetOwnershipRange_SeqBDiag,
       MatILUFactorSymbolic_SeqBDiag,0,
       0,0,
       MatConvertSameType_SeqBDiag,0,0,
       MatILUFactor_SeqBDiag,0,0,
       MatGetSubMatrices_SeqBDiag,0,MatGetValues_SeqBDiag_N,0,
       MatPrintHelp_SeqBDiag,MatScale_SeqBDiag,
       0,0,0,MatGetBlockSize_SeqBDiag};

#undef __FUNC__  
#define __FUNC__ "MatCreateSeqBDiag"
/*@C
   MatCreateSeqBDiag - Creates a sequential block diagonal matrix.

   Collective on MPI_Comm

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals (optional)
.  bs - each element of a diagonal is an bs x bs dense matrix
.  diag - optional array of block diagonal numbers (length nd).
   For a matrix element A[i,j], where i=row and j=column, the
   diagonal number is
$     diag = i/bs - j/bs  (integer division)
   Set diag=PETSC_NULL on input for PETSc to dynamically allocate memory as 
   needed (expensive).
-  diagv - pointer to actual diagonals (in same order as diag array), 
   if allocated by user.  Otherwise, set diagv=PETSC_NULL on input for PETSc
   to control memory allocation.

   Output Parameters:
.  A - the matrix

   Options Database Keys:
.  -mat_block_size <bs> - Sets blocksize
.  -mat_bdiag_diags <s1,s2,s3,...> - Sets diagonal numbers

   Notes:
   See the users manual for further details regarding this storage format.

   Fortran Note:
   Fortran programmers cannot set diagv; this value is ignored.

.keywords: matrix, block, diagonal, sparse

.seealso: MatCreate(), MatCreateMPIBDiag(), MatSetValues()
@*/
int MatCreateSeqBDiag(MPI_Comm comm,int m,int n,int nd,int bs,int *diag,Scalar **diagv,Mat *A)
{
  Mat          B;
  Mat_SeqBDiag *b;
  int          i, nda, sizetot, ierr,  nd2 = 128,flg1,idiag[128],size;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_WRONG,0,"Comm must be of size 1");

  *A = 0;
  if (bs == PETSC_DEFAULT) bs = 1;
  if (nd == PETSC_DEFAULT) nd = 0;
  ierr = OptionsGetInt(PETSC_NULL,"-mat_block_size",&bs,&flg1);CHKERRQ(ierr);
  ierr = OptionsGetIntArray(PETSC_NULL,"-mat_bdiag_diags",idiag,&nd2,&flg1);CHKERRQ(ierr);
  if (flg1) {
    diag = idiag;
    nd   = nd2;
  }

  if ((n%bs) || (m%bs)) SETERRQ(PETSC_ERR_ARG_SIZ,0,"Invalid block size");
  if (!nd) nda = nd + 1;
  else     nda = nd;
  PetscHeaderCreate(B,_p_Mat,struct _MatOps,MAT_COOKIE,MATSEQBDIAG,comm,MatDestroy,MatView);
  PLogObjectCreate(B);
  B->data    = (void *) (b = PetscNew(Mat_SeqBDiag)); CHKPTRQ(b);
  PetscMemzero(b,sizeof(Mat_SeqBDiag));
  PetscMemcpy(B->ops,&MatOps,sizeof(struct _MatOps));
  B->ops->destroy = MatDestroy_SeqBDiag;
  B->ops->view    = MatView_SeqBDiag;
  B->factor  = 0;
  B->mapping = 0;

  ierr = OptionsHasName(PETSC_NULL,"-mat_no_unroll",&flg1); CHKERRQ(ierr);
  if (!flg1) {
    switch (bs) {
      case 1:
        B->ops->setvalues       = MatSetValues_SeqBDiag_1;
        B->ops->getvalues       = MatGetValues_SeqBDiag_1;
        B->ops->getdiagonal     = MatGetDiagonal_SeqBDiag_1;
        B->ops->mult            = MatMult_SeqBDiag_1;
        B->ops->multadd         = MatMultAdd_SeqBDiag_1;
        B->ops->multtrans       = MatMultTrans_SeqBDiag_1;
        B->ops->multtransadd    = MatMultTransAdd_SeqBDiag_1;
        B->ops->relax           = MatRelax_SeqBDiag_1;
        B->ops->solve           = MatSolve_SeqBDiag_1;
        B->ops->lufactornumeric = MatLUFactorNumeric_SeqBDiag_1;
        break;
      case 2:
	B->ops->mult            = MatMult_SeqBDiag_2; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_2;
        B->ops->solve           = MatSolve_SeqBDiag_2;
        break;
      case 3:
	B->ops->mult            = MatMult_SeqBDiag_3; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_3;
	B->ops->solve           = MatSolve_SeqBDiag_3; 
        break;
      case 4:
	B->ops->mult            = MatMult_SeqBDiag_4; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_4;
	B->ops->solve           = MatSolve_SeqBDiag_4; 
        break;
      case 5:
	B->ops->mult            = MatMult_SeqBDiag_5; 
        B->ops->multadd         = MatMultAdd_SeqBDiag_5;
	B->ops->solve           = MatSolve_SeqBDiag_5; 
        break;
   }
  }

  b->m      = m; B->m = m; B->M = m;
  b->n      = n; B->n = n; B->N = n;
  b->mblock = m/bs;
  b->nblock = n/bs;
  b->nd     = nd;
  b->bs     = bs;
  b->ndim   = 0;
  b->mainbd = -1;
  b->pivot  = 0;

  b->diag   = (int *)PetscMalloc(2*nda*sizeof(int)); CHKPTRQ(b->diag);
  b->bdlen  = b->diag + nda;
  b->colloc = (int *)PetscMalloc((n+1)*sizeof(int)); CHKPTRQ(b->colloc);
  b->diagv  = (Scalar**)PetscMalloc(nda*sizeof(Scalar*)); CHKPTRQ(b->diagv);
  sizetot   = 0;

  if (diagv != PETSC_NULL) { /* user allocated space */
    b->user_alloc = 1;
    for (i=0; i<nd; i++) b->diagv[i] = diagv[i];
  } else b->user_alloc = 0;

  for (i=0; i<nd; i++) {
    b->diag[i] = diag[i];
    if (diag[i] > 0) { /* lower triangular */
      b->bdlen[i] = PetscMin(b->nblock,b->mblock - diag[i]);
    } else {           /* upper triangular */
      b->bdlen[i] = PetscMin(b->mblock,b->nblock + diag[i]);
    }
    sizetot += b->bdlen[i];
  }
  sizetot   *= bs*bs;
  b->maxnz  =  sizetot;
  b->dvalue = (Scalar *) PetscMalloc((n+1)*sizeof(Scalar)); CHKPTRQ(b->dvalue);
  PLogObjectMemory(B,(nda*(bs+2))*sizeof(int) + bs*nda*sizeof(Scalar)
                    + nda*sizeof(Scalar*) + sizeof(Mat_SeqBDiag)
                    + sizeof(struct _p_Mat) + sizetot*sizeof(Scalar));

  if (!b->user_alloc) {
    for (i=0; i<nd; i++) {
      b->diagv[i] = (Scalar*)PetscMalloc(bs*bs*b->bdlen[i]*sizeof(Scalar));CHKPTRQ(b->diagv[i]);
      PetscMemzero(b->diagv[i],bs*bs*b->bdlen[i]*sizeof(Scalar));
    }
    b->nonew = 0; b->nonew_diag = 0;
  } else { /* diagonals are set on input; don't allow dynamic allocation */
    b->nonew = 1; b->nonew_diag = 1;
  }

  /* adjust diagv so one may access rows with diagv[diag][row] for all rows */
  for (i=0; i<nd; i++) {
    if (diag[i] > 0) {
      b->diagv[i] -= bs*bs*diag[i];
    }
  }

  b->nz          = b->maxnz; /* Currently not keeping track of exact count */
  b->roworiented = 1;
  ierr = OptionsHasName(PETSC_NULL,"-help",&flg1); CHKERRQ(ierr);
  if (flg1) {ierr = MatPrintHelp(B); CHKERRQ(ierr);}
  B->info.nz_unneeded = (double)b->maxnz;

  *A = B;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatConvertSameType_SeqBDiag"
static int MatConvertSameType_SeqBDiag(Mat A,Mat *matout,int cpvalues)
{ 
  Mat_SeqBDiag *newmat, *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, len,diag,bs = a->bs;
  Mat          mat;

  PetscFunctionBegin;
  ierr = MatCreateSeqBDiag(A->comm,a->m,a->n,a->nd,bs,a->diag,PETSC_NULL,matout);CHKERRQ(ierr);

  /* Copy contents of diagonals */
  mat = *matout;
  newmat = (Mat_SeqBDiag *) mat->data;
  if (cpvalues == COPY_VALUES) {
    for (i=0; i<a->nd; i++) {
      len = a->bdlen[i] * bs * bs * sizeof(Scalar);
      diag = a->diag[i];
      if (diag > 0) {
        PetscMemcpy(newmat->diagv[i]+bs*bs*diag,a->diagv[i]+bs*bs*diag,len);
      } else {
        PetscMemcpy(newmat->diagv[i],a->diagv[i],len);
      }
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatLoad_SeqBDiag"
int MatLoad_SeqBDiag(Viewer viewer,MatType type,Mat *A)
{
  Mat          B;
  int          *scols, i, nz, ierr, fd, header[4], size,nd = 128;
  int          bs, *rowlengths = 0,M,N,*cols,flg,extra_rows,*diag = 0;
  int          idiag[128];
  Scalar       *vals, *svals;
  MPI_Comm     comm;
  
  PetscFunctionBegin;
  PetscObjectGetComm((PetscObject)viewer,&comm);
  MPI_Comm_size(comm,&size);
  if (size > 1) SETERRQ(PETSC_ERR_ARG_SIZ,0,"view must have one processor");
  ierr = ViewerBinaryGetDescriptor(viewer,&fd); CHKERRQ(ierr);
  ierr = PetscBinaryRead(fd,header,4,PETSC_INT); CHKERRQ(ierr);
  if (header[0] != MAT_COOKIE) SETERRQ(PETSC_ERR_FILE_UNEXPECTED,0,"Not matrix object");
  M = header[1]; N = header[2]; nz = header[3];
  if (M != N) SETERRQ(PETSC_ERR_SUP,0,"Can only load square matrices");
  if (header[3] < 0) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,1,"Matrix stored in special format, cannot load as SeqBDiag");
  }

  /* 
     This code adds extra rows to make sure the number of rows is 
    divisible by the blocksize
  */
  bs = 1;
  ierr = OptionsGetInt(PETSC_NULL,"-matload_block_size",&bs,&flg);CHKERRQ(ierr);
  extra_rows = bs - M + bs*(M/bs);
  if (extra_rows == bs) extra_rows = 0;
  if (extra_rows) {
    PLogInfo(0,"MatLoad_SeqBDiag:Padding loaded matrix to match blocksize\n");
  }

  /* read row lengths */
  rowlengths = (int*) PetscMalloc((M+extra_rows)*sizeof(int));CHKPTRQ(rowlengths);
  ierr = PetscBinaryRead(fd,rowlengths,M,PETSC_INT); CHKERRQ(ierr);
  for ( i=0; i<extra_rows; i++ ) rowlengths[M+i] = 1;

  /* load information about diagonals */
  ierr = OptionsGetIntArray(PETSC_NULL,"-matload_bdiag_diags",idiag,&nd,&flg);
         CHKERRQ(ierr);
  if (flg) {
    diag = idiag;
  }

  /* create our matrix */
  ierr = MatCreateSeqBDiag(comm,M+extra_rows,M+extra_rows,nd,bs,diag,
                           PETSC_NULL,A); CHKERRQ(ierr);
  B = *A;

  /* read column indices and nonzeros */
  cols = scols = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(cols);
  ierr = PetscBinaryRead(fd,cols,nz,PETSC_INT); CHKERRQ(ierr);
  vals = svals = (Scalar *) PetscMalloc( nz*sizeof(Scalar) ); CHKPTRQ(vals);
  ierr = PetscBinaryRead(fd,vals,nz,PETSC_SCALAR); CHKERRQ(ierr);
  /* insert into matrix */

  for ( i=0; i<M; i++ ) {
    ierr = MatSetValues(B,1,&i,rowlengths[i],scols,svals,INSERT_VALUES);CHKERRQ(ierr);
    scols += rowlengths[i]; svals += rowlengths[i];
  }
  vals[0] = 1.0;
  for ( i=M; i<M+extra_rows; i++ ) {
    ierr = MatSetValues(B,1,&i,1,&i,vals,INSERT_VALUES);CHKERRQ(ierr);
  }

  PetscFree(cols);
  PetscFree(vals);
  PetscFree(rowlengths);   

  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
