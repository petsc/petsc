#define PETSCMAT_DLL

/* Block diagonal matrix format */

#include "src/mat/impls/bdiag/seq/bdiag.h"
#include "src/inline/ilu.h"

#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqBDiag_1"
PetscErrorCode MatSetValues_SeqBDiag_1(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       kk,ldiag,row,newnz,*bdlen_new;
  PetscErrorCode ierr;
  PetscInt       j,k, *diag_new;
  PetscTruth     roworiented = a->roworiented,dfound;
  PetscScalar    value,**diagv_new;

  PetscFunctionBegin;
  for (kk=0; kk<m; kk++) { /* loop over added rows */
    row = im[kk];   
    if (row < 0) continue;
    if (row >= A->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap.N-1);
    for (j=0; j<n; j++) {
      if (in[j] < 0) continue;
      if (in[j] >= A->cmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Column too large: col %D max %D",in[j],A->cmap.N-1);
      ldiag  = row - in[j]; /* diagonal number */
      dfound = PETSC_FALSE;
      if (roworiented) {
        value = v[j + kk*n]; 
      } else {
        value = v[kk + j*m];
      }
      /* search diagonals for required one */
      for (k=0; k<a->nd; k++) {
	if (a->diag[k] == ldiag) {
          dfound = PETSC_TRUE;
          if (is == ADD_VALUES) a->diagv[k][row] += value;
	  else                  a->diagv[k][row]  = value;
          break;
        }
      }
      if (!dfound) {
        if (a->nonew || a->nonew_diag) {
#if !defined(PETSC_USE_COMPLEX)
          if (a->user_alloc && value) {
#else
          if (a->user_alloc && PetscRealPart(value) || PetscImaginaryPart(value)) {
#endif
            ierr = PetscInfo1(A,"Nonzero in diagonal %D that user did not allocate\n",ldiag);CHKERRQ(ierr);
          }
        } else {
          ierr = PetscInfo1(A,"Allocating new diagonal: %D\n",ldiag);CHKERRQ(ierr);
          a->reallocs++;
          /* free old bdiag storage info and reallocate */
          ierr      = PetscMalloc(2*(a->nd+1)*sizeof(PetscInt),&diag_new);CHKERRQ(ierr);
          bdlen_new = diag_new + a->nd + 1;
          ierr      = PetscMalloc((a->nd+1)*sizeof(PetscScalar*),&diagv_new);CHKERRQ(ierr);
          for (k=0; k<a->nd; k++) {
            diag_new[k]  = a->diag[k];
            diagv_new[k] = a->diagv[k];
            bdlen_new[k] = a->bdlen[k];
          }
          diag_new[a->nd]  = ldiag;
          if (ldiag > 0) { /* lower triangular */
            bdlen_new[a->nd] = PetscMin(a->nblock,a->mblock - ldiag);
          } else {         /* upper triangular */
            bdlen_new[a->nd] = PetscMin(a->mblock,a->nblock + ldiag);
          }
          newnz = bdlen_new[a->nd];
          ierr = PetscMalloc(newnz*sizeof(PetscScalar),&diagv_new[a->nd]);CHKERRQ(ierr);
          ierr = PetscMemzero(diagv_new[a->nd],newnz*sizeof(PetscScalar));CHKERRQ(ierr);
          /* adjust pointers so that dv[diag][row] works for all diagonals*/
          if (diag_new[a->nd] > 0) {
            diagv_new[a->nd] -= diag_new[a->nd];
          }
          a->maxnz += newnz;
          a->nz    += newnz;
          ierr = PetscFree(a->diagv);CHKERRQ(ierr);
          ierr = PetscFree(a->diag);CHKERRQ(ierr);
          a->diag  = diag_new; 
          a->bdlen = bdlen_new;
          a->diagv = diagv_new;

          /* Insert value */
          if (is == ADD_VALUES) a->diagv[a->nd][row] += value;
          else                  a->diagv[a->nd][row] = value;
          a->nd++;
          ierr = PetscLogObjectMemory(A,newnz*sizeof(PetscScalar)+2*sizeof(PetscInt)+sizeof(PetscScalar*));CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSetValues_SeqBDiag_N"
PetscErrorCode MatSetValues_SeqBDiag_N(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],const PetscScalar v[],InsertMode is)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       kk,ldiag,shift,row,newnz,*bdlen_new;
  PetscInt       j,k,bs = A->rmap.bs,*diag_new,idx=0;
  PetscTruth     roworiented = a->roworiented,dfound;
  PetscScalar    value,**diagv_new;

  PetscFunctionBegin;
  for (kk=0; kk<m; kk++) { /* loop over added rows */
    row = im[kk];   
    if (row < 0) SETERRQ1(PETSC_ERR_ARG_OUTOFRANGE,"Negative row: %D",row);
    if (row >= A->rmap.N) SETERRQ2(PETSC_ERR_ARG_OUTOFRANGE,"Row too large: row %D max %D",row,A->rmap.N-1);
    shift = (row/bs)*bs*bs + row%bs;
    for (j=0; j<n; j++) {
      ldiag  = row/bs - in[j]/bs; /* block diagonal */
      dfound = PETSC_FALSE;
      if (roworiented) {
        value = v[idx++]; 
      } else {
        value = v[kk + j*m];
      }
      /* seach for appropriate diagonal */
      for (k=0; k<a->nd; k++) {
        if (a->diag[k] == ldiag) {
          dfound = PETSC_TRUE;
          if (is == ADD_VALUES) a->diagv[k][shift + (in[j]%bs)*bs] += value;
	  else                  a->diagv[k][shift + (in[j]%bs)*bs] = value;
          break;
        }
      }
      if (!dfound) {
        if (a->nonew || a->nonew_diag) {
#if !defined(PETSC_USE_COMPLEX)
          if (a->user_alloc && value) {
#else
          if (a->user_alloc && PetscRealPart(value) || PetscImaginaryPart(value)) {
#endif
            ierr = PetscInfo1(A,"Nonzero in diagonal %D that user did not allocate\n",ldiag);CHKERRQ(ierr);
          }
        } else {
          ierr = PetscInfo1(A,"Allocating new diagonal: %D\n",ldiag);CHKERRQ(ierr);
          a->reallocs++;
          /* free old bdiag storage info and reallocate */
          ierr      = PetscMalloc(2*(a->nd+1)*sizeof(PetscInt),&diag_new);CHKERRQ(ierr);
          bdlen_new = diag_new + a->nd + 1;
          ierr      = PetscMalloc((a->nd+1)*sizeof(PetscScalar*),&diagv_new);CHKERRQ(ierr);
          for (k=0; k<a->nd; k++) {
            diag_new[k]  = a->diag[k];
            diagv_new[k] = a->diagv[k];
            bdlen_new[k] = a->bdlen[k];
          }
          diag_new[a->nd]  = ldiag;
          if (ldiag > 0) {/* lower triangular */
            bdlen_new[a->nd] = PetscMin(a->nblock,a->mblock - ldiag);
          } else {         /* upper triangular */
            bdlen_new[a->nd] = PetscMin(a->mblock,a->nblock + ldiag);
          }
          newnz = bs*bs*bdlen_new[a->nd];
          ierr = PetscMalloc(newnz*sizeof(PetscScalar),&diagv_new[a->nd]);CHKERRQ(ierr);
          ierr = PetscMemzero(diagv_new[a->nd],newnz*sizeof(PetscScalar));CHKERRQ(ierr);
          /* adjust pointer so that dv[diag][row] works for all diagonals */
          if (diag_new[a->nd] > 0) {
            diagv_new[a->nd] -= bs*bs*diag_new[a->nd];
          }
          a->maxnz += newnz; a->nz += newnz;
          ierr = PetscFree(a->diagv);CHKERRQ(ierr);
          ierr = PetscFree(a->diag);CHKERRQ(ierr);
          a->diag  = diag_new; 
          a->bdlen = bdlen_new;
          a->diagv = diagv_new;

          /* Insert value */
          if (is == ADD_VALUES) a->diagv[k][shift + (in[j]%bs)*bs] += value;
          else                  a->diagv[k][shift + (in[j]%bs)*bs] = value;
          a->nd++;
          ierr = PetscLogObjectMemory(A,newnz*sizeof(PetscScalar)+2*sizeof(PetscInt)+sizeof(PetscScalar*));CHKERRQ(ierr);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqBDiag_1"
PetscErrorCode MatGetValues_SeqBDiag_1(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  PetscInt     kk,ldiag,row,j,k;
  PetscScalar  zero = 0.0;
  PetscTruth   dfound;

  PetscFunctionBegin;
  for (kk=0; kk<m; kk++) { /* loop over rows */
    row = im[kk];   
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (row >= A->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    for (j=0; j<n; j++) {
      if (in[j] < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative column");
      if (in[j] >= A->cmap.n) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Column too large");
      ldiag = row - in[j]; /* diagonal number */
      dfound = PETSC_FALSE;
      for (k=0; k<a->nd; k++) {
        if (a->diag[k] == ldiag) {
          dfound = PETSC_TRUE;
          *v++ = a->diagv[k][row];
          break;
        }
      }
      if (!dfound) *v++ = zero;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatGetValues_SeqBDiag_N"
PetscErrorCode MatGetValues_SeqBDiag_N(Mat A,PetscInt m,const PetscInt im[],PetscInt n,const PetscInt in[],PetscScalar v[])
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag*)A->data;
  PetscInt     kk,ldiag,shift,row,j,k,bs = A->rmap.bs;
  PetscScalar  zero = 0.0;
  PetscTruth   dfound;

  PetscFunctionBegin;
  for (kk=0; kk<m; kk++) { /* loop over rows */
    row = im[kk];   
    if (row < 0) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Negative row");
    if (row >= A->rmap.N) SETERRQ(PETSC_ERR_ARG_OUTOFRANGE,"Row too large");
    shift = (row/bs)*bs*bs + row%bs;
    for (j=0; j<n; j++) {
      ldiag  = row/bs - in[j]/bs; /* block diagonal */
      dfound = PETSC_FALSE;
      for (k=0; k<a->nd; k++) {
        if (a->diag[k] == ldiag) {
          dfound = PETSC_TRUE;
          *v++ = a->diagv[k][shift + (in[j]%bs)*bs ];
          break;
        }
      }
      if (!dfound) *v++ = zero;
    }
  }
  PetscFunctionReturn(0);
}

/*
    MatMults for blocksize 1 to 5 and N -------------------------------
 */
#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_1"
PetscErrorCode MatMult_SeqBDiag_1(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,diag,*a_diag = a->diag,*a_bdlen = a->bdlen;
  PetscErrorCode ierr;
  PetscInt       d,j,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv   = a_diagv[d];
    diag = a_diag[d];
    len  = a_bdlen[d];
    if (diag > 0) {	     /* lower triangle */
      pvin  = vin;
      pvout = vout + diag;
      dv    = dv   + diag;
    } else {		     /* upper triangle,including main diagonal */
      pvin  = vin - diag;
      pvout = vout;
    }
    for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
    ierr = PetscLogFlops(2*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_2"
PetscErrorCode MatMult_SeqBDiag_2(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,nb_diag;
  PetscErrorCode ierr;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr); 
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 2*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 2*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0     = pvin[0]; pvin1 = pvin[1];

      pvout[0] += dv[0]*pvin0 + dv[2]*pvin1;
      pvout[1] += dv[1]*pvin0 + dv[3]*pvin1;

      pvout += 2; pvin += 2; dv += 4; 
    }
    ierr = PetscLogFlops(8*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr); 
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_3"
PetscErrorCode MatMult_SeqBDiag_3(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,nb_diag;
  PetscErrorCode ierr;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 3*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 3*nb_diag;
    } else {		       /* upper triangle,including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2];

      pvout[0] += dv[0]*pvin0 + dv[3]*pvin1  + dv[6]*pvin2;
      pvout[1] += dv[1]*pvin0 + dv[4]*pvin1  + dv[7]*pvin2;
      pvout[2] += dv[2]*pvin0 + dv[5]*pvin1  + dv[8]*pvin2;

      pvout += 3; pvin += 3; dv += 9; 
    }
    ierr = PetscLogFlops(18*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_4"
PetscErrorCode MatMult_SeqBDiag_4(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,nb_diag;
  PetscErrorCode ierr;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2,pvin3;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 4*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 4*nb_diag;
    } else {		       /* upper triangle,including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2]; pvin3 = pvin[3];

      pvout[0] += dv[0]*pvin0 + dv[4]*pvin1  + dv[8]*pvin2 + dv[12]*pvin3;
      pvout[1] += dv[1]*pvin0 + dv[5]*pvin1  + dv[9]*pvin2 + dv[13]*pvin3;
      pvout[2] += dv[2]*pvin0 + dv[6]*pvin1  + dv[10]*pvin2 + dv[14]*pvin3;
      pvout[3] += dv[3]*pvin0 + dv[7]*pvin1  + dv[11]*pvin2 + dv[15]*pvin3;

      pvout += 4; pvin += 4; dv += 16; 
    }
    ierr = PetscLogFlops(32*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_5"
PetscErrorCode MatMult_SeqBDiag_5(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,nb_diag;
  PetscErrorCode ierr;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2,pvin3,pvin4;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 5*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 5*nb_diag;
    } else {		       /* upper triangle,including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2]; pvin3 = pvin[3]; pvin4 = pvin[4];

      pvout[0] += dv[0]*pvin0 + dv[5]*pvin1  + dv[10]*pvin2 + dv[15]*pvin3 + dv[20]*pvin4;
      pvout[1] += dv[1]*pvin0 + dv[6]*pvin1  + dv[11]*pvin2 + dv[16]*pvin3 + dv[21]*pvin4;
      pvout[2] += dv[2]*pvin0 + dv[7]*pvin1  + dv[12]*pvin2 + dv[17]*pvin3 + dv[22]*pvin4;
      pvout[3] += dv[3]*pvin0 + dv[8]*pvin1  + dv[13]*pvin2 + dv[18]*pvin3 + dv[23]*pvin4;
      pvout[4] += dv[4]*pvin0 + dv[9]*pvin1  + dv[14]*pvin2 + dv[19]*pvin3 + dv[24]*pvin4;

      pvout += 5; pvin += 5; dv += 25; 
    }
    ierr = PetscLogFlops(50*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMult_SeqBDiag_N"
PetscErrorCode MatMult_SeqBDiag_N(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscInt       nd = a->nd,bs = A->rmap.bs,nb_diag,bs2 = bs*bs;
  PetscErrorCode ierr;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv;

  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->rmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = bs*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + bs*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      Kernel_v_gets_v_plus_A_times_w(bs,pvout,dv,pvin);
      pvout += bs; pvin += bs; dv += bs2; 
    }
    ierr = PetscLogFlops(2*bs2*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    MatMultAdds for blocksize 1 to 5 and N -------------------------------
 */
#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_1"
PetscErrorCode MatMultAdd_SeqBDiag_1(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,diag,*a_diag = a->diag,*a_bdlen = a->bdlen,d,j,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr); 
  for (d=0; d<nd; d++) {
    dv   = a_diagv[d];
    diag = a_diag[d];
    len  = a_bdlen[d];
    if (diag > 0) {	     /* lower triangle */
      pvin  = vin;
      pvout = vout + diag;
      dv    = dv   + diag;
    } else {		     /* upper triangle, including main diagonal */
      pvin  = vin - diag;
      pvout = vout;
    }
    for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
    ierr = PetscLogFlops(2*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_2"
PetscErrorCode MatMultAdd_SeqBDiag_2(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,nb_diag;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr); 
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 2*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 2*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1];

      pvout[0] += dv[0]*pvin0 + dv[2]*pvin1;
      pvout[1] += dv[1]*pvin0 + dv[3]*pvin1;

      pvout += 2; pvin += 2; dv += 4; 
    }
    ierr = PetscLogFlops(8*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_3"
PetscErrorCode MatMultAdd_SeqBDiag_3(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,nb_diag;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr); 
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 3*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 3*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2];

      pvout[0] += dv[0]*pvin0 + dv[3]*pvin1  + dv[6]*pvin2;
      pvout[1] += dv[1]*pvin0 + dv[4]*pvin1  + dv[7]*pvin2;
      pvout[2] += dv[2]*pvin0 + dv[5]*pvin1  + dv[8]*pvin2;

      pvout += 3; pvin += 3; dv += 9; 
    }
    ierr = PetscLogFlops(18*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_4"
PetscErrorCode MatMultAdd_SeqBDiag_4(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,nb_diag;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2,pvin3;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr); 
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 4*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 4*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2]; pvin3 = pvin[3];

      pvout[0] += dv[0]*pvin0 + dv[4]*pvin1  + dv[8]*pvin2 + dv[12]*pvin3;
      pvout[1] += dv[1]*pvin0 + dv[5]*pvin1  + dv[9]*pvin2 + dv[13]*pvin3;
      pvout[2] += dv[2]*pvin0 + dv[6]*pvin1  + dv[10]*pvin2 + dv[14]*pvin3;
      pvout[3] += dv[3]*pvin0 + dv[7]*pvin1  + dv[11]*pvin2 + dv[15]*pvin3;

      pvout += 4; pvin += 4; dv += 16; 
    }
    ierr = PetscLogFlops(32*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_5"
PetscErrorCode MatMultAdd_SeqBDiag_5(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,nb_diag;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv,pvin0,pvin1,pvin2,pvin3,pvin4;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = 5*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + 5*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      pvin0 = pvin[0]; pvin1 = pvin[1]; pvin2 = pvin[2]; pvin3 = pvin[3]; pvin4 = pvin[4];

      pvout[0] += dv[0]*pvin0 + dv[5]*pvin1  + dv[10]*pvin2 + dv[15]*pvin3 + dv[20]*pvin4;
      pvout[1] += dv[1]*pvin0 + dv[6]*pvin1  + dv[11]*pvin2 + dv[16]*pvin3 + dv[21]*pvin4;
      pvout[2] += dv[2]*pvin0 + dv[7]*pvin1  + dv[12]*pvin2 + dv[17]*pvin3 + dv[22]*pvin4;
      pvout[3] += dv[3]*pvin0 + dv[8]*pvin1  + dv[13]*pvin2 + dv[18]*pvin3 + dv[23]*pvin4;
      pvout[4] += dv[4]*pvin0 + dv[9]*pvin1  + dv[14]*pvin2 + dv[19]*pvin3 + dv[24]*pvin4;

      pvout += 5; pvin += 5; dv += 25; 
    }
    ierr = PetscLogFlops(50*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultAdd_SeqBDiag_N"
PetscErrorCode MatMultAdd_SeqBDiag_N(Mat A,Vec xx,Vec zz,Vec yy)
{ 
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,bs = A->rmap.bs,nb_diag,bs2 = bs*bs;
  PetscInt       *a_diag = a->diag,*a_bdlen = a->bdlen,d,k,len;
  PetscScalar    *vin,*vout,**a_diagv = a->diagv;
  PetscScalar    *pvin,*pvout,*dv;

  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv      = a_diagv[d];
    nb_diag = bs*a_diag[d];
    len     = a_bdlen[d];
    if (nb_diag > 0) {	        /* lower triangle */
      pvin  = vin;
      pvout = vout + nb_diag;
      dv    = dv   + bs*nb_diag;
    } else {		       /* upper triangle, including main diagonal */
      pvin  = vin - nb_diag;
      pvout = vout;
    }
    for (k=0; k<len; k++) {
      Kernel_v_gets_v_plus_A_times_w(bs,pvout,dv,pvin);
      pvout += bs; pvin += bs; dv += bs2; 
    }
    ierr = PetscLogFlops(2*bs2*len);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatMultTranspose ----------------------------------------------
 */
#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBDiag_1"
PetscErrorCode MatMultTranspose_SeqBDiag_1(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,diag,d,j,len;
  PetscScalar    *pvin,*pvout,*dv;
  PetscScalar    *vin,*vout;
  
  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->cmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv   = a->diagv[d];
    diag = a->diag[d];
    len  = a->bdlen[d];
      /* diag of original matrix is (row/bs - col/bs) */
      /* diag of transpose matrix is (col/bs - row/bs) */
    if (diag < 0) {	/* transpose is lower triangle */
      pvin  = vin;
      pvout = vout - diag;
    } else {	/* transpose is upper triangle, including main diagonal */
      pvin  = vin + diag;
      pvout = vout;
      dv    = dv + diag;
    }
    for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTranspose_SeqBDiag_N"
PetscErrorCode MatMultTranspose_SeqBDiag_N(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,bs = A->rmap.bs,diag,kshift,kloc,d,i,j,k,len;
  PetscScalar    *pvin,*pvout,*dv;
  PetscScalar    *vin,*vout;
  
  PetscFunctionBegin;
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  ierr = PetscMemzero(vout,A->cmap.n*sizeof(PetscScalar));CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv   = a->diagv[d];
    diag = a->diag[d];
    len  = a->bdlen[d];
      /* diag of original matrix is (row/bs - col/bs) */
      /* diag of transpose matrix is (col/bs - row/bs) */
    if (diag < 0) {	/* transpose is lower triangle */
      pvin  = vin;
      pvout = vout - bs*diag;
    } else {	/* transpose is upper triangle, including main diagonal */
      pvin  = vin + bs*diag;
      pvout = vout;
      dv    = dv + diag;
    }
    for (k=0; k<len; k++) {
      kloc = k*bs; kshift = kloc*bs;
      for (i=0; i<bs; i++) {	 /* i = local column of transpose */
        for (j=0; j<bs; j++) {   /* j = local row of transpose */
          pvout[kloc + j] += dv[kshift + j*bs + i] * pvin[kloc + i];
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     MatMultTransposeAdd ----------------------------------------------
 */
#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqBDiag_1"
PetscErrorCode MatMultTransposeAdd_SeqBDiag_1(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,diag,d,j,len;
  PetscScalar    *pvin,*pvout,*dv;
  PetscScalar    *vin,*vout;
  
  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv   = a->diagv[d];
    diag = a->diag[d];
    len  = a->bdlen[d];
      /* diag of original matrix is (row/bs - col/bs) */
      /* diag of transpose matrix is (col/bs - row/bs) */
    if (diag < 0) {	/* transpose is lower triangle */
      pvin  = vin;
      pvout = vout - diag;
    } else {	/* transpose is upper triangle, including main diagonal */
      pvin  = vin + diag;
      pvout = vout;
      dv    = dv + diag;
    }
    for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMultTransposeAdd_SeqBDiag_N"
PetscErrorCode MatMultTransposeAdd_SeqBDiag_N(Mat A,Vec xx,Vec zz,Vec yy)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscErrorCode ierr;
  PetscInt       nd = a->nd,bs = A->rmap.bs,diag,kshift,kloc,d,i,j,k,len;
  PetscScalar    *pvin,*pvout,*dv;
  PetscScalar    *vin,*vout;
  
  PetscFunctionBegin;
  if (zz != yy) {ierr = VecCopy(zz,yy);CHKERRQ(ierr);}
  ierr = VecGetArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecGetArray(yy,&vout);CHKERRQ(ierr);
  for (d=0; d<nd; d++) {
    dv   = a->diagv[d];
    diag = a->diag[d];
    len  = a->bdlen[d];
      /* diag of original matrix is (row/bs - col/bs) */
      /* diag of transpose matrix is (col/bs - row/bs) */
    if (diag < 0) {	/* transpose is lower triangle */
      pvin  = vin;
      pvout = vout - bs*diag;
    } else {	/* transpose is upper triangle, including main diagonal */
      pvin  = vin + bs*diag;
      pvout = vout;
      dv    = dv + diag;
    }
    for (k=0; k<len; k++) {
      kloc = k*bs; kshift = kloc*bs;
      for (i=0; i<bs; i++) {	 /* i = local column of transpose */
        for (j=0; j<bs; j++) {   /* j = local row of transpose */
          pvout[kloc + j] += dv[kshift + j*bs + i] * pvin[kloc + i];
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&vin);CHKERRQ(ierr);
  ierr = VecRestoreArray(yy,&vout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#undef __FUNCT__  
#define __FUNCT__ "MatRelax_SeqBDiag_N"
PetscErrorCode MatRelax_SeqBDiag_N(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscScalar    *x,*b,*xb,*dd,*dv,dval,sum;
  PetscErrorCode ierr;
  PetscInt       i,j,k,d,kbase,bs = A->rmap.bs,kloc;
  PetscInt       mainbd = a->mainbd,diag,mblock = a->mblock,bloc;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);

  /* Currently this code doesn't use wavefront orderings, although
     we should eventually incorporate that option, whatever wavefront
     ordering maybe :-) */

  if (mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Main diagonal not set");

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  if (xx != bb) {
    ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  } else {
    b = x;
  }
  dd = a->diagv[mainbd];
  if (flag == SOR_APPLY_UPPER) {
    /* apply (U + D/omega) to the vector */
    for (k=0; k<mblock; k++) {
      kloc = k*bs; kbase = kloc*bs;
      for (i=0; i<bs; i++) {
        sum = b[i+kloc] * (shift + dd[i*(bs+1)+kbase]) / omega;
        for (j=i+1; j<bs; j++) sum += dd[kbase + j*bs + i] * b[kloc + j];
        for (d=mainbd+1; d<a->nd; d++) {
          diag = a->diag[d];
          dv   = a->diagv[d];
          if (k-diag < mblock) {
            for (j=0; j<bs; j++) {
              sum += dv[kbase + j*bs + i] * b[(k-diag)*bs + j];
            }
          }
        }
        x[kloc+i] = sum;
      }
    }
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    if (xx != bb) {ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);}
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (k=0; k<mblock; k++) {
        kloc = k*bs; kbase = kloc*bs;
        for (i=0; i<bs; i++) {
          sum  = b[i+kloc];
          dval = shift + dd[i*(bs+1)+kbase];
          for (d=0; d<mainbd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            if (k >= diag) {
              for (j=0; j<bs; j++)
                sum -= dv[k*bs*bs + j*bs + i] * x[(k-diag)*bs + j];
            }
          }
          for (j=0; j<i; j++){
            sum -= dd[kbase + j*bs + i] * x[kloc + j];
          }
          x[kloc+i] = omega*sum/dval;
        }
      }
      xb = x;
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (k=0; k<mblock; k++) {
        kloc = k*bs; kbase = kloc*bs;
        for (i=0; i<bs; i++)
          x[kloc+i] *= dd[i*(bs+1)+kbase];
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (k=mblock-1; k>=0; k--) {
        kloc = k*bs; kbase = kloc*bs;
        for (i=bs-1; i>=0; i--) {
          sum  = xb[i+kloc];
          dval = shift + dd[i*(bs+1)+kbase];
          for (j=i+1; j<bs; j++)
            sum -= dd[kbase + j*bs + i] * x[kloc + j];
          for (d=mainbd+1; d<a->nd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            bloc = k - diag;
            if (bloc < mblock) {
              for (j=0; j<bs; j++)
                sum -= dv[kbase + j*bs + i] * x[(k-diag)*bs + j];
            }
          }
          x[kloc+i] = omega*sum/dval;
        }
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (k=0; k<mblock; k++) {
        kloc = k*bs; kbase = kloc*bs;
        for (i=0; i<bs; i++) {
          sum  = b[i+kloc];
          dval = shift + dd[i*(bs+1)+kbase];
          for (d=0; d<mainbd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            bloc = k - diag;
            if (bloc >= 0) {
              for (j=0; j<bs; j++) {
                sum -= dv[k*bs*bs + j*bs + i] * x[bloc*bs + j];
              }
            }
          }
          for (d=mainbd; d<a->nd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            bloc = k - diag;
            if (bloc < mblock) {
              for (j=0; j<bs; j++) {
                sum -= dv[kbase + j*bs + i] * x[(k-diag)*bs + j];
              }
            }
	  }
          x[kloc+i] = (1.-omega)*x[kloc+i]+omega*(sum+dd[i*(bs+1)+kbase]*x[kloc+i])/dval;
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (k=mblock-1; k>=0; k--) {
        kloc = k*bs; kbase = kloc*bs;
        for (i=bs-1; i>=0; i--) {
          sum  = b[i+kloc];
          dval = shift + dd[i*(bs+1)+kbase];
          for (d=0; d<mainbd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            bloc = k - diag;
            if (bloc >= 0) {
              for (j=0; j<bs; j++) {
                sum -= dv[k*bs*bs + j*bs + i] * x[bloc*bs + j];
              }
            }
          }
          for (d=mainbd; d<a->nd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
            bloc = k - diag;
            if (bloc < mblock) {
              for (j=0; j<bs; j++) {
                sum -= dv[kbase + j*bs + i] * x[(k-diag)*bs + j];
              }
            }
          }
          x[kloc+i] = (1.-omega)*x[kloc+i]+omega*(sum+dd[i*(bs+1)+kbase]*x[kloc+i])/dval;
        }
      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  if (xx != bb) ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

#undef __FUNCT__  
#define __FUNCT__ "MatRelax_SeqBDiag_1"
PetscErrorCode MatRelax_SeqBDiag_1(Mat A,Vec bb,PetscReal omega,MatSORType flag,PetscReal shift,PetscInt its,PetscInt lits,Vec xx)
{
  Mat_SeqBDiag   *a = (Mat_SeqBDiag*)A->data;
  PetscScalar    *x,*b,*xb,*dd,dval,sum;
  PetscErrorCode ierr;
  PetscInt       m = A->rmap.n,i,d,loc;
  PetscInt       mainbd = a->mainbd,diag;

  PetscFunctionBegin;
  its = its*lits;
  if (its <= 0) SETERRQ2(PETSC_ERR_ARG_WRONG,"Relaxation requires global its %D and local its %D both positive",its,lits);
  /* Currently this code doesn't use wavefront orderings,although
     we should eventually incorporate that option, whatever wavefront
     ordering maybe :-) */

  ierr = VecGetArray(xx,&x);CHKERRQ(ierr);
  ierr = VecGetArray(bb,&b);CHKERRQ(ierr);
  if (mainbd == -1) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Main diagonal not set");
  dd = a->diagv[mainbd];
  if (flag == SOR_APPLY_UPPER) {
    /* apply (U + D/omega) to the vector */
    for (i=0; i<m; i++) {
      sum = b[i] * (shift + dd[i]) / omega;
      for (d=mainbd+1; d<a->nd; d++) {
        diag = a->diag[d];
        if (i-diag < m) sum += a->diagv[d][i] * x[i-diag];
      }
      x[i] = sum;
    }
    ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        sum  = b[i];
        for (d=0; d<mainbd; d++) {
          if (i >= a->diag[d]) sum -= a->diagv[d][i] * x[i-a->diag[d]];
        }
        x[i] = omega*(sum/(shift + dd[i]));
      }
      xb = x;
    } else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      for (i=0; i<m; i++) x[i] *= dd[i];
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      for (i=m-1; i>=0; i--) {
        sum = xb[i];
        for (d=mainbd+1; d<a->nd; d++) {
          diag = a->diag[d];
          if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
        }
        x[i] = omega*(sum/(shift + dd[i]));
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      for (i=0; i<m; i++) {
        sum  = b[i];
        dval = shift + dd[i];
        for (d=0; d<mainbd; d++) {
          if (i >= a->diag[d]) sum -= a->diagv[d][i] * x[i-a->diag[d]];
        }
        for (d=mainbd; d<a->nd; d++) {
          diag = a->diag[d];
          if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
        }
        x[i] = (1. - omega)*x[i] + omega*(sum + dd[i]*x[i])/dval;
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      for (i=m-1; i>=0; i--) {
        sum = b[i];
        for (d=0; d<mainbd; d++) {
          loc = i - a->diag[d];
          if (loc >= 0) sum -= a->diagv[d][i] * x[loc];
        }
        for (d=mainbd; d<a->nd; d++) {
          diag = a->diag[d];
          if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
        }
        x[i] = (1. - omega)*x[i] + omega*(sum + dd[i]*x[i])/(shift + dd[i]);
      }
    }
  }
  ierr = VecRestoreArray(xx,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(bb,&b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
