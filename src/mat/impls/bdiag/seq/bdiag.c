#ifndef lint
static char vcid[] = "$Id: bdiag.c,v 1.54 1995/09/30 19:29:09 bsmith Exp bsmith $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

static int MatSetValues_SeqBDiag(Mat matin,int m,int *idxm,int n,
                            int *idxn,Scalar *v,InsertMode  addv)
{
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag *) matin->data;
  int          i, kk, j, k, loc, ldiag, shift, row, nz = n, dfound, temp;
  int          nb = dmat->nb, nd = dmat->nd, *diag = dmat->diag, *diag_new;
  int          newnz, *bdlen_new;
  Scalar       *valpt, **diagv_new, *dtemp;

  if (m!=1) SETERRQ(1,"MatSetValues_SeqBDiag:Can set only 1 row at a time");
  if (nb == 1) {
    for ( kk=0; kk<m; kk++ ) { /* loop over added rows */
      row  = idxm[kk];   
      if (row < 0) SETERRQ(1,"MatSetValues_SeqBDiag:Negative row");
      if (row >= dmat->m) SETERRQ(1,"MatSetValues_SeqBDiag:Row too large");
      for (j=0; j<nz; j++) {
        ldiag = row - idxn[j]; /* diagonal number */
        dfound = 0;
        for (k=0; k<nd; k++) {
	  if (diag[k] == ldiag) {
            dfound = 1;
	    if (ldiag > 0) loc = row - ldiag; /* lower triangle */
	    else           loc = row;
	    if ((valpt = &((dmat->diagv[k])[loc]))) {
	      if (addv == ADD_VALUES) *valpt += v[j];
	      else                    *valpt = v[j];
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            break;
          }
        }
        if (!dfound) {
          if (dmat->nonew) {
#if !defined(PETSC_COMPLEX)
            if (dmat->user_alloc && v[j]) {
#else
            if (dmat->user_alloc && real(v[j]) || imag(v[j])) {
#endif
              PLogInfo((PetscObject)matin,
                "MatSetValues_SeqBDiag: Nonzero in diagonal %d that user did not allocate\n",ldiag);
            }
          } else {
            nd++; 
            PLogInfo((PetscObject)matin,"MatSetValues_SeqBDiag: Allocating new diagonal: %d\n",ldiag);
            /* free old bdiag storage info and reallocate */
            diag_new = (int *)PETSCMALLOC(2*nd*sizeof(int)); CHKPTRQ(diag_new);
            bdlen_new = diag_new + nd;
            diagv_new = (Scalar**)PETSCMALLOC(nd*sizeof(Scalar*)); CHKPTRQ(diagv_new);
            for (k=0; k<dmat->nd; k++) {
              diag_new[k]  = dmat->diag[k];
              diagv_new[k] = dmat->diagv[k];
              bdlen_new[k] = dmat->bdlen[k];
            }
            diag_new[dmat->nd]  = ldiag;
            if (ldiag > 0) /* lower triangular */
              bdlen_new[dmat->nd] = PETSCMIN(dmat->nblock,dmat->mblock - ldiag);
            else {         /* upper triangular */
              if (dmat->mblock - ldiag > dmat->nblock)
                bdlen_new[dmat->nd] = dmat->nblock + ldiag;
              else
                bdlen_new[dmat->nd] = dmat->mblock;
            }
            newnz = bdlen_new[dmat->nd];
            diagv_new[dmat->nd] = (Scalar*)PETSCMALLOC(newnz*sizeof(Scalar));
            CHKPTRQ(diagv_new[dmat->nd]);
            dmat->maxnz += newnz;
            dmat->nz += newnz;
            PETSCFREE(dmat->diagv); PETSCFREE(dmat->diag); 
            dmat->diag  = diag_new; 
            dmat->bdlen = bdlen_new;
            dmat->diagv = diagv_new;

            /* Insert value */
	    if (ldiag > 0) loc = row - ldiag; /* lower triangle */
	    else           loc = row;
	    if ((valpt = &((dmat->diagv[dmat->nd])[loc]))) {
	      if (addv == ADD_VALUES) *valpt += v[j];
	      else                    *valpt = v[j];
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");

            for (i=0; i<nd; i++) {  /* Sort diagonals */
              for (k=i+1; k<nd; k++) {
                if (diag_new[i] < diag_new[k]) {
                  temp         = diag_new[i];   
                  diag_new[i]  = diag_new[k];
                  diag_new[k]  = temp;
                  temp         = bdlen_new[i];   
                  bdlen_new[i] = bdlen_new[k];
                  bdlen_new[k] = temp;
                  dtemp        = diagv_new[i];
                  diagv_new[i] = diagv_new[k];
                  diagv_new[k] = dtemp;
                }
              }
            }
            dmat->nd = nd; 
          }
        }
      }
    }
  } else {

    for ( kk=0; kk<m; kk++ ) { /* loop over added rows */
      row    = idxm[kk];   
      if (row < 0) SETERRQ(1,"MatSetValues_SeqBDiag:Negative row");
      if (row >= dmat->m) SETERRQ(1,"MatSetValues_SeqBDiag:Row too large");
      shift = (row/nb)*nb*nb + row%nb;
      for (j=0; j<nz; j++) {
        ldiag = row/nb - idxn[j]/nb; /* block diagonal */
        dfound = 0;
        for (k=0; k<nd; k++) {
          if (diag[k] == ldiag) {
            dfound = 1;
	    if (ldiag > 0) /* lower triangle */
	      loc = shift - ldiag*nb*nb;
             else
	      loc = shift;
	    if ((valpt = &((dmat->diagv[k])[loc + (idxn[j]%nb)*nb ]))) {
	      if (addv == ADD_VALUES) *valpt += v[j];
	      else                    *valpt = v[j];
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            break;
          }
        }
        if (!dfound) {
          if (dmat->nonew) {
#if !defined(PETSC_COMPLEX)
            if (dmat->user_alloc && v[j]) {
#else
            if (dmat->user_alloc && real(v[j]) || imag(v[j])) {
#endif
              PLogInfo((PetscObject)matin,
                "MatSetValues_SeqBDiag: Nonzero in diagonal %d that user did not allocate\n",ldiag);
            }
          } else {
            nd++; 
            PLogInfo((PetscObject)matin,"MatSetValues_SeqBDiag: Allocating new diagonal: %d\n",ldiag);
            /* free old bdiag storage info and reallocate */
            diag_new = (int *)PETSCMALLOC(2*nd*sizeof(int)); CHKPTRQ(diag_new);
            bdlen_new = diag_new + nd;
            diagv_new = (Scalar**)PETSCMALLOC(nd*sizeof(Scalar*)); CHKPTRQ(diagv_new);
            for (k=0; k<dmat->nd; k++) {
              diag_new[k]  = dmat->diag[k];
              diagv_new[k] = dmat->diagv[k];
              bdlen_new[k] = dmat->bdlen[k];
            }
            diag_new[dmat->nd]  = ldiag;
            if (ldiag > 0) /* lower triangular */
              bdlen_new[dmat->nd] = PETSCMIN(dmat->nblock,dmat->mblock - ldiag);
            else {         /* upper triangular */
              if (dmat->mblock - ldiag > dmat->nblock)
                bdlen_new[dmat->nd] = dmat->nblock + ldiag;
              else
                bdlen_new[dmat->nd] = dmat->mblock;
            }
            newnz = nb*nb*bdlen_new[dmat->nd];
            diagv_new[dmat->nd] = (Scalar*)PETSCMALLOC(newnz*sizeof(Scalar));
            CHKPTRQ(diagv_new[dmat->nd]);
            dmat->maxnz += newnz; dmat->nz += newnz;
            PETSCFREE(dmat->diagv); PETSCFREE(dmat->diag); 
            dmat->diag  = diag_new; 
            dmat->bdlen = bdlen_new;
            dmat->diagv = diagv_new;

            /* Insert value */
	    if (ldiag > 0) /* lower triangle */
	      loc = shift - ldiag*nb*nb;
             else
	      loc = shift;
	    if ((valpt = &((dmat->diagv[k])[loc + (idxn[j]%nb)*nb ]))) {
	      if (addv == ADD_VALUES) *valpt += v[j];
	      else                    *valpt = v[j];
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");

            for (i=0; i<nd; i++) {  /* Sort diagonals */
              for (k=i+1; k<nd; k++) {
                if (diag_new[i] < diag_new[k]) {
                  temp         = diag_new[i];   
                  diag_new[i]  = diag_new[k];
                  diag_new[k]  = temp;
                  temp         = bdlen_new[i];   
                  bdlen_new[i] = bdlen_new[k];
                  bdlen_new[k] = temp;
                  dtemp        = diagv_new[i];
                  diagv_new[i] = diagv_new[k];
                  diagv_new[k] = dtemp;
                }
              }
            }
            dmat->nd = nd; 
          }
        }
      }
    }
  }
  return 0;
}

/*
  MatMult_SeqBDiag_base - This routine is intended for use with 
  MatMult_SeqBDiag() and MatMultAdd_SeqBDiag().  It computes yy += mat * xx.
 */
static int MatMult_SeqBDiag_base(Mat matin,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag    *mat= (Mat_SeqBDiag *) matin->data;
  int             nd = mat->nd, nb = mat->nb, diag, kshift, kloc;
  Scalar          *vin, *vout;
  register Scalar *pvin, *pvout, *dv;
  register int    d, i, j, k, len;

  VecGetArray(xx,&vin); VecGetArray(yy,&vout);
  if (nb == 1) {
    for (d=0; d<nd; d++) {
      dv   = mat->diagv[d];
      diag = mat->diag[d];
      len  = mat->bdlen[d];
      if (diag > 0) {	/* lower triangle */
        pvin = vin;
	pvout = vout + diag;
      } else {		/* upper triangle, including main diagonal */
        pvin  = vin - diag;
        pvout = vout;
      }
      for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
    }
  } else { /* Block diagonal approach, assuming storage within dense blocks 
              in column-major order */
    for (d=0; d<nd; d++) {
      dv   = mat->diagv[d];
      diag = mat->diag[d];
      len  = mat->bdlen[d];
      if (diag > 0) {	/* lower triangle */
        pvin = vin;
	pvout = vout + nb*diag;
      } else {		/* upper triangle, including main diagonal */
        pvin  = vin - nb*diag;
        pvout = vout;
      }
      for (k=0; k<len; k++) {
        kloc = k*nb; kshift = kloc*nb; 
        for (i=0; i<nb; i++) {	/* i = local row */
          for (j=0; j<nb; j++) {	/* j = local column */
            pvout[kloc + i] += dv[kshift + j*nb + i] * pvin[kloc + j];
          }
        }
      }
    }
  }
  return 0;
}

/*
  MatMultTrans_SeqBDiag_base - This routine is intended for use with 
  MatMultTrans_SeqBDiag() and MatMultTransAdd_SeqBDiag().  It computes 
            yy += mat^T * xx.
 */
static int MatMultTrans_SeqBDiag_base(Mat matin,Vec xx,Vec yy)
{
  Mat_SeqBDiag    *mat = (Mat_SeqBDiag *) matin->data;
  int             nd = mat->nd, nb = mat->nb, diag,kshift, kloc;
  register Scalar *pvin, *pvout, *dv;
  register int    d, i, j, k, len;
  Scalar          *vin, *vout;
  
  VecGetArray(xx,&vin); VecGetArray(yy,&vout);
  if (nb == 1) {
    for (d=0; d<nd; d++) {
      dv   = mat->diagv[d];
      diag = mat->diag[d];
      len  = mat->bdlen[d];
      /* diag of original matrix is (row/nb - col/nb) */
      /* diag of transpose matrix is (col/nb - row/nb) */
      if (diag < 0) {	/* transpose is lower triangle */
        pvin  = vin;
	pvout = vout - diag;
      } else {	/* transpose is upper triangle, including main diagonal */
        pvin  = vin + diag;
        pvout = vout;
      }
      for (j=0; j<len; j++) pvout[j] += dv[j] * pvin[j];
    }
  } else { /* Block diagonal approach, assuming storage within dense blocks
              in column-major order */
    for (d=0; d<nd; d++) {
      dv   = mat->diagv[d];
      diag = mat->diag[d];
      len  = mat->bdlen[d];
      /* diag of original matrix is (row/nb - col/nb) */
      /* diag of transpose matrix is (col/nb - row/nb) */
      if (diag < 0) {	/* transpose is lower triangle */
        pvin  = vin;
        pvout = vout - nb*diag;
      } else {	/* transpose is upper triangle, including main diagonal */
        pvin  = vin + nb*diag;
        pvout = vout;
      }
      for (k=0; k<len; k++) {
        kloc = k*nb; kshift = kloc*nb;
        for (i=0; i<nb; i++) {	 /* i = local column of transpose */
          for (j=0; j<nb; j++) { /* j = local row of transpose */
            pvout[kloc + j] += dv[kshift + j*nb + i] * pvin[kloc + i];
          }
        }
      }
    }
  }
  return 0;
}
static int MatMult_SeqBDiag(Mat matin,Vec xx,Vec yy)
{
  Scalar zero = 0.0;
  int    ierr;
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  return MatMult_SeqBDiag_base(matin,xx,yy);
}
static int MatMultTrans_SeqBDiag(Mat matin,Vec xx,Vec yy)
{
  Scalar zero = 0.0;
  int    ierr;
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  return MatMultTrans_SeqBDiag_base(matin,xx,yy);
}
static int MatMultAdd_SeqBDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  int ierr;
  ierr = VecCopy(zz,yy); CHKERRQ(ierr);
  return MatMult_SeqBDiag_base(matin,xx,yy);
}
static int MatMultTransAdd_SeqBDiag(Mat matin,Vec xx,Vec zz,Vec yy)
{
  int ierr;
  ierr = VecCopy(zz,yy); CHKERRQ(ierr);
  return MatMultTrans_SeqBDiag_base(matin,xx,yy);
}

static int MatRelax_SeqBDiag(Mat matin,Vec bb,double omega,MatSORType flag,
                             double shift,int its,Vec xx)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  Scalar       *x, *b, *xb, *dvmain, *dv, dval, sum;
  int          m = mat->m, i, j, k, d, kbase, nb = mat->nb, loc, kloc;
  int          mainbd = mat->mainbd, diag, mblock = mat->mblock, bloc;

  /* Currently this code doesn't use wavefront orderings, although
     we should eventually incorporate that option */
  VecGetArray(xx,&x); VecGetArray(bb,&b);
  if (mainbd == -1) SETERRQ(1,"MatRelax_SeqBDiag:Main diagonal not set");
  dvmain = mat->diagv[mainbd];
  if (flag == SOR_APPLY_UPPER) {
    /* apply ( U + D/omega) to the vector */
    if (nb == 1) {
      for ( i=0; i<m; i++ ) {
        sum = b[i] * (shift + dvmain[i]) / omega;
        for (d=mainbd+1; d<mat->nd; d++) {
          diag = mat->diag[d];
          if (i-diag < m) sum += mat->diagv[d][i] * x[i-diag];
        }
        x[i] = sum;
      }
    } else {
      for ( k=0; k<mblock; k++ ) {
        kloc = k*nb; kbase = kloc*nb;
        for (i=0; i<nb; i++) {
          sum = b[i+kloc] * (shift + dvmain[i*(nb+1)+kbase]) / omega;
          for (j=i+1; j<nb; j++)
            sum += dvmain[kbase + j*nb + i] * b[kloc + j];
          for (d=mainbd+1; d<mat->nd; d++) {
            diag = mat->diag[d];
            dv   = mat->diagv[d];
            if (k-diag < mblock) {
              for (j=0; j<nb; j++)
                sum += dv[kbase + j*nb + i] * b[(k-diag)*nb + j];
            }
	  }
          x[kloc+i] = sum;
        }
      }
    }
    return 0;
  }
  if (flag & SOR_ZERO_INITIAL_GUESS) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      if (nb == 1) {
        for (i=0; i<m; i++) {
          sum  = b[i];
          for (d=0; d<mainbd; d++) {
            loc = i - mat->diag[d];
            if (loc >= 0) sum -= mat->diagv[d][loc] * x[loc];
          }
          x[i] = omega*(sum/(shift + dvmain[i]));
        }
      } else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++) {
            sum  = b[i+kloc];
            dval = shift + dvmain[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (j=0; j<i; j++)
              sum -= dvmain[kbase + j*nb + i] * x[kloc + j];
            x[kloc+i] = omega*sum/dval;
          }
        }
      }
      xb = x;
    }
    else xb = b;
    if ((flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) && 
        (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP)) {
      if (nb == 1) {
        for ( i=0; i<m; i++ ) x[i] *= dvmain[i];
      } 
      else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++)
            x[kloc+i] *= dvmain[i*(nb+1)+kbase];
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      if (nb == 1) {
        for ( i=m-1; i>=0; i-- ) {
          sum = xb[i];
          for (d=mainbd+1; d<mat->nd; d++) {
            diag = mat->diag[d];
            if (i-diag < m) sum -= mat->diagv[d][i] * x[i-diag];
          }
          x[i] = omega*(sum/(shift + dvmain[i]));
        }
      } 
      else {
        for ( k=mblock-1; k>=0; k-- ) {
          kloc = k*nb; kbase = kloc*nb;
          for ( i=nb-1; i>=0; i-- ) {
            sum  = xb[i+kloc];
            dval = shift + dvmain[i*(nb+1)+kbase];
            for ( j=i+1; j<nb; j++ )
              sum -= dvmain[kbase + j*nb + i] * x[kloc + j];
            for (d=mainbd+1; d<mat->nd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc < mblock) {
                for (j=0; j<nb; j++)
                  sum -= dv[kbase + j*nb + i] * x[(k-diag)*nb + j];
              }
	    }
            x[kloc+i] = omega*sum/dval;
          }
        }
      }
    }
    its--;
  }
  while (its--) {
    if (flag & SOR_FORWARD_SWEEP || flag & SOR_LOCAL_FORWARD_SWEEP) {
      if (nb == 1) {
        for (i=0; i<m; i++) {
          sum  = b[i];
          dval = shift + dvmain[i];
          for (d=0; d<mainbd; d++) {
            loc = i - mat->diag[d];
            if (loc >= 0) sum -= mat->diagv[d][loc] * x[loc];
          }
          for (d=mainbd; d<mat->nd; d++) {
            diag = mat->diag[d];
            if (i-diag < m) sum -= mat->diagv[d][i] * x[i-diag];
          }
          x[i] = (1. - omega)*x[i] + omega*(sum/dval + x[i]);
        }
      } else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++) {
            sum  = b[i+kloc];
            dval = shift + dvmain[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (d=mainbd; d<mat->nd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc < mblock) {
                for (j=0; j<nb; j++)
                  sum -= dv[kbase + j*nb + i] * x[(k-diag)*nb + j];
              }
	    }
            x[kloc+i] = (1. - omega)*x[kloc+i] + omega*(sum/dval + x[kloc+i]);
          }
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP){
      if (nb == 1) {
        for ( i=m-1; i>=0; i-- ) {
          sum = b[i];
          for (d=0; d<mainbd; d++) {
            loc = i - mat->diag[d];
            if (loc >= 0) sum -= mat->diagv[d][loc] * x[loc];
          }
          for (d=mainbd; d<mat->nd; d++) {
            diag = mat->diag[d];
            if (i-diag < m) sum -= mat->diagv[d][i] * x[i-diag];
          }
          x[i] = (1. - omega)*x[i] + omega*(sum/(shift + dvmain[i]) + x[i]);
        }
      } 
      else {
        for ( k=mblock-1; k>=0; k-- ) {
          kloc = k*nb; kbase = kloc*nb;
          for ( i=nb-1; i>=0; i-- ) {
            sum  = b[i+kloc];
            dval = shift + dvmain[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (d=mainbd; d<mat->nd; d++) {
              diag = mat->diag[d];
              dv   = mat->diagv[d];
              bloc = k - diag;
              if (bloc < mblock) {
                for (j=0; j<nb; j++)
                  sum -= dv[kbase + j*nb + i] * x[(k-diag)*nb + j];
              }
	    }
            x[kloc+i] = (1. - omega)*x[kloc+i] + omega*(sum/dval + x[kloc+i]);
          }
        }
      }
    }
  }
  return 0;
} 

static int MatGetInfo_SeqBDiag(Mat matin,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  *nz      = mat->nz;
  *nzalloc = mat->maxnz;
  *mem     = (int)matin->mem;
  return 0;
}

static int MatGetOwnershipRange_SeqBDiag(Mat matin,int *m,int *n)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  *m = 0; *n = mat->m;
  return 0;
}

static int MatGetRow_SeqBDiag(Mat matin,int row,int *nz,int **col,Scalar **v)
{
  Mat_SeqBDiag *dmat = (Mat_SeqBDiag *) matin->data;
  int          nd = dmat->nd, nb = dmat->nb, loc;
  int          nc = dmat->n, *diag = dmat->diag, pcol, shift, i, j, k;

/* For efficiency, if ((nz) && (col) && (v)) then do all at once */
  if ((nz) && (col) && (v)) {
    *col = dmat->colloc;
    *v   = dmat->dvalue;
    k    = 0;
    if (nb == 1) { 
      for (j=0; j<nd; j++) {
        pcol = row - diag[j];
        if (pcol > -1 && pcol < nc) {
	  if (diag[j] > 0)
	    loc = row - diag[j];
	  else
	    loc = row;
	  (*v)[k]   = (dmat->diagv[j])[loc];
          (*col)[k] = pcol;  k++;
	}
      }
      *nz = k;
    } else {
      shift = (row/nb)*nb*nb + row%nb;
      for (j=0; j<nd; j++) {
        pcol = nb * (row/nb - diag[j]);
        if (pcol > -1 && pcol < nc) {
          if (diag[j] > 0)
	    loc = shift - diag[j]*nb*nb;
	  else 
	    loc = shift;
          for (i=0; i<nb; i++) {
	    (*v)[k+i]   = (dmat->diagv[j])[loc + i*nb];
	    (*col)[k+i] = pcol + i;
	  }
          k += nb;
        } 
      }
      *nz = k;
    }
  }
  else {
    if (nb == 1) { 
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) k++; 
        }
        *nz = k;
      }
      if (col) {
        *col = dmat->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
            (*col)[k] = pcol;  k++;
          }
        }
      }
      if (v) {
        *v = dmat->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = row - diag[j];
          if (pcol > -1 && pcol < nc) {
	    if (diag[j] > 0)
	      loc = row - diag[j];
	    else
	      loc = row;
	    (*v)[k] = (dmat->diagv[j])[loc]; k++;
          }
        }
      }
    } 
    else {
      if (nz) {
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = nb * (row/nb- diag[j]);
          if (pcol > -1 && pcol < nc) k += nb; 
        }
        *nz = k;
      }
      if (col) {
        *col = dmat->colloc;
        k = 0;
        for (j=0; j<nd; j++) {
          pcol = nb * (row/nb - diag[j]);
          if (pcol > -1 && pcol < nc) {
            for (i=0; i<nb; i++) {
	      (*col)[k+i] = pcol + i;
            }
	    k += nb;
          }
        }
      }
      if (v) {
        shift = (row/nb)*nb*nb + row%nb;
        *v = dmat->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
	  pcol = nb * (row/nb - diag[j]);
	  if (pcol > -1 && pcol < nc) {
	    if (diag[j] > 0)
	      loc = shift - diag[j]*nb*nb;
	    else 
	      loc = shift;
	    for (i=0; i<nb; i++) {
	     (*v)[k+i] = (dmat->diagv[j])[loc + i*nb];
            }
	    k += nb;
	  }
        }
      }
    }
  }
  return 0;
}

static int MatRestoreRow_SeqBDiag(Mat matin,int row,int *ncols,int **cols,Scalar **vals)
{
  /* Work space is allocated once during matrix creation and then freed
     when matrix is destroyed */
  return 0;
}

static int MatNorm_SeqBDiag(Mat matin,MatNormType type,double *norm)
{
  Mat_SeqBDiag *mat= (Mat_SeqBDiag *) matin->data;
  double       sum = 0.0, *tmp;
  int          d, i, j, k, nd = mat->nd, nb = mat->nb, diag, kshift, kloc, len;
  Scalar       *dv;

  if (!mat->assembled) SETERRQ(1,"MatNorm_SeqBDiag:Must assemble mat");

  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = mat->diagv[d];
      len  = mat->bdlen[d]*nb*nb;
      for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
        sum += real(conj(dv[i])*dv[i]);
#else
        sum += dv[i]*dv[i];
#endif
      }
    }
    *norm = sqrt(sum);
  }
  else if (type == NORM_1) { /* max column norm */
    tmp = (double *) PETSCMALLOC( mat->n*sizeof(double) ); CHKPTRQ(tmp);
    PetscZero(tmp,mat->n*sizeof(double));
    *norm = 0.0;
    if (nb == 1) {
      for (d=0; d<nd; d++) {
        dv   = mat->diagv[d];
        diag = mat->diag[d];
        len  = mat->bdlen[d];
        if (diag > 0) {	/* lower triangle: row = loc+diag, col = loc */
          for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
            tmp[i] += abs(dv[i]); 
#else
            tmp[i] += fabs(dv[i]); 
#endif
          }
        } else {	/* upper triangle: row = loc, col = loc-diag */
          for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
            tmp[i-diag] += abs(dv[i]); 
#else
            tmp[i-diag] += fabs(dv[i]); 
#endif
          }
        }
      }
    } else { 
      for (d=0; d<nd; d++) {
        dv   = mat->diagv[d];
        diag = mat->diag[d];
        len  = mat->bdlen[d];

        if (diag > 0) {	/* lower triangle: row = loc+diag, col = loc */
          for (k=0; k<len; k++) {
            kloc = k*nb; kshift = kloc*nb; 
            for (i=0; i<nb; i++) {	/* i = local row */
              for (j=0; j<nb; j++) {	/* j = local column */
#if defined(PETSC_COMPLEX)
                tmp[kloc + j] += abs(dv[kshift + j*nb + i]);
#else
                tmp[kloc + j] += fabs(dv[kshift + j*nb + i]);
#endif
              }
            }
          }
        } else {	/* upper triangle: row = loc, col = loc-diag */
          for (k=0; k<len; k++) {
            kloc = k*nb; kshift = kloc*nb; 
            for (i=0; i<nb; i++) {	/* i = local row */
              for (j=0; j<nb; j++) {	/* j = local column */
#if defined(PETSC_COMPLEX)
                tmp[kloc + j - nb*diag] += abs(dv[kshift + j*nb + i]);
#else
                tmp[kloc + j - nb*diag] += fabs(dv[kshift + j*nb + i]);
#endif
              }
            }
          }
        }
      }
    }
    for ( j=0; j<mat->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PETSCFREE(tmp);
  }
  else if (type == NORM_INFINITY) { /* max row norm */
    tmp = (double *) PETSCMALLOC( mat->m*sizeof(double) ); CHKPTRQ(tmp);
    PetscZero(tmp,mat->m*sizeof(double));
    *norm = 0.0;
    if (nb == 1) {
      for (d=0; d<nd; d++) {
        dv   = mat->diagv[d];
        diag = mat->diag[d];
        len  = mat->bdlen[d];
        if (diag > 0) {	/* lower triangle: row = loc+diag, col = loc */
          for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
            tmp[i+diag] += abs(dv[i]); 
#else
            tmp[i+diag] += fabs(dv[i]); 
#endif
          }
        } else {	/* upper triangle: row = loc, col = loc-diag */
          for (i=0; i<len; i++) {
#if defined(PETSC_COMPLEX)
            tmp[i] += abs(dv[i]); 
#else
            tmp[i] += fabs(dv[i]); 
#endif
          }
        }
      }
    } else { 
      for (d=0; d<nd; d++) {
        dv   = mat->diagv[d];
        diag = mat->diag[d];
        len  = mat->bdlen[d];
        if (diag > 0) {
          for (k=0; k<len; k++) {
            kloc = k*nb; kshift = kloc*nb; 
            for (i=0; i<nb; i++) {	/* i = local row */
              for (j=0; j<nb; j++) {	/* j = local column */
#if defined(PETSC_COMPLEX)
                tmp[kloc + i + nb*diag] += abs(dv[kshift + j*nb + i]);
#else
                tmp[kloc + i + nb*diag] += fabs(dv[kshift + j*nb + i]);
#endif
              }
            }
          }
        } else {
          for (k=0; k<len; k++) {
            kloc = k*nb; kshift = kloc*nb; 
            for (i=0; i<nb; i++) {	/* i = local row */
              for (j=0; j<nb; j++) {	/* j = local column */
#if defined(PETSC_COMPLEX)
                tmp[kloc + i] += abs(dv[kshift + j*nb + i]);
#else
                tmp[kloc + i] += fabs(dv[kshift + j*nb + i]);
#endif
              }
            }
          }
        }
      }
    }
    for ( j=0; j<mat->m; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PETSCFREE(tmp);
  }
  else {
    SETERRQ(1,"MatNorm_SeqBDiag:No support for two norm");
  }
  return 0;
}

static int MatTranspose_SeqBDiag(Mat A,Mat *matout)
{ 
  Mat_SeqBDiag *mbd = (Mat_SeqBDiag *) A->data, *mbdnew;
  Mat          tmat;
  int          i, j, k, d, ierr, nd = mbd->nd, *diag = mbd->diag, *diagnew;
  int          nb = mbd->nb, kshift;
  Scalar       *dwork, *dvnew;

  diagnew = (int *) PETSCMALLOC(nd*sizeof(int)); CHKPTRQ(diagnew);
  for (i=0; i<nd; i++) {
    diagnew[i] = -diag[nd-i-1]; /* assume sorted in descending order */
  }
  ierr = MatCreateSeqBDiag(A->comm,mbd->n,mbd->m,nd,nb,diagnew,
                                    0,&tmat); CHKERRQ(ierr);
  PETSCFREE(diagnew);
  mbdnew = (Mat_SeqBDiag *) tmat->data;
  for (d=0; d<nd; d++) {
    dvnew = mbdnew->diagv[d];
    dwork = mbd->diagv[nd-d-1];
    if (mbdnew->bdlen[d] != mbd->bdlen[nd-d-1])
      SETERRQ(1,"MatTranspose_SeqBDiag:Incompatible diagonal lengths");
    if (nb == 1) {
      for (k=0; k<mbdnew->bdlen[d]; k++) dvnew[k] = dwork[k];
    } else {
      for (k=0; k<mbdnew->bdlen[d]; k++) {
        kshift = k*nb*nb;
        for (i=0; i<nb; i++) {	/* i = local row */
          for (j=0; j<nb; j++) {	/* j = local column */
            dvnew[kshift + j + i*nb] = dwork[kshift + j*nb + i];
          }
        }
      }
    }
  }
  ierr = MatAssemblyBegin(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(tmat,FINAL_ASSEMBLY); CHKERRQ(ierr);

  if (matout) {
    *matout = tmat;
  } else {
    /* This isn't really an in-place transpose ... but free data 
       structures from mbd.  We should fix this. */
    if (!mbd->user_alloc) { /* Free the actual diagonals */
      for (i=0; i<mbd->nd; i++) PETSCFREE( mbd->diagv[i] );
    }
    if (mbd->pivots) PETSCFREE(mbd->pivots);
    PETSCFREE(mbd->diagv);
    PETSCFREE(mbd->diag);
    PETSCFREE(mbd->colloc);
    PETSCFREE(mbd->dvalue);
    PETSCFREE(mbd);
    PetscMemcpy(A,tmat,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(tmat);
  }
  return 0;
}

/* ----------------------------------------------------------------*/
#include "draw.h"
#include "pinclude/pviewer.h"

int MatView_SeqBDiag(PetscObject obj,Viewer ptr)
{
  Mat          matin = (Mat) obj;
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int          ierr, *col, i, j, len, diag, nr = mat->m, nb = mat->nb;
  int          nz, nzalloc, mem;
  Scalar       *val, *dv, zero = 0.0;
  PetscObject  vobj = (PetscObject) ptr;

  if (!mat->assembled) SETERRQ(1,"MatView_SeqBDiag:Not for unassembled matrix");
  if (!ptr) { 
    ptr = STDOUT_VIEWER_SELF; vobj = (PetscObject) ptr;
  }
  if (vobj->cookie == DRAW_COOKIE && vobj->type == NULLWINDOW) return 0;
  if (vobj && vobj->cookie == VIEWER_COOKIE && vobj->type == MATLAB_VIEWER) {
    SETERRQ(1,"MatView_SeqBDiag:Matlab viewer not supported for MATSEQBDIAG format");
  }
  if (vobj && vobj->cookie == DRAW_COOKIE) {
    DrawCtx draw = (DrawCtx) ptr;
    double  xl,yl,xr,yr,w,h;
    xr = mat->n; yr = mat->m; h = yr/10.0; w = xr/10.0;
    xr += w; yr += h; xl = -w; yl = -h;
    ierr = DrawSetCoordinates(draw,xl,yl,xr,yr); CHKERRQ(ierr);
    /* loop over matrix elements drawing boxes; we really should do this
       by diagonals. */
    /* What do we really want to draw here?  nonzeros, allocated space? */
    for ( i=0; i<nr; i++ ) {
      yl = nr - i - 1.0; yr = yl + 1.0;
      ierr = MatGetRow(matin,i,&nz,&col,0); CHKERRQ(ierr);
      for ( j=0; j<nz; j++ ) {
        xl = col[j]; xr = xl + 1.0;
        DrawRectangle(draw,xl,yl,xr,yr,DRAW_BLACK,DRAW_BLACK,DRAW_BLACK,
                      DRAW_BLACK);
      }
    ierr = MatRestoreRow(matin,i,&nz,&col,0); CHKERRQ(ierr);
    }
    return 0;
  }
  else {
    FILE *fd;
    char *outputname;
    int format;

    ierr = ViewerFileGetPointer_Private(ptr,&fd); CHKERRQ(ierr);
    ierr = ViewerFileGetOutputname_Private(ptr,&outputname);
    ierr = ViewerFileGetFormat_Private(ptr,&format);
    if (format == FILE_FORMAT_INFO) {
      fprintf(fd,"  block size=%d, number of diagonals=%d\n",nb,mat->nd);
    }
    else if (format == FILE_FORMAT_MATLAB) {
      MatGetInfo(matin,MAT_LOCAL,&nz,&nzalloc,&mem);
      fprintf(fd,"%% Size = %d %d \n",nr, mat->n);
      fprintf(fd,"%% Nonzeros = %d \n",nz);
      fprintf(fd,"zzz = zeros(%d,3);\n",nz);
      fprintf(fd,"zzz = [\n");
      for ( i=0; i<mat->m; i++ ) {
        ierr = MatGetRow( matin, i, &nz, &col, &val ); CHKERRQ(ierr);
        for (j=0; j<nz; j++) {
          if (val[j] != zero)
#if defined(PETSC_COMPLEX)
            fprintf(fd,"%d %d  %18.16e  %18.16e \n",
               i+1, col[j]+1, real(val[j]), imag(val[j]) );
#else
            fprintf(fd,"%d %d  %18.16e\n", i+1, col[j]+1, val[j]);
#endif
        }
      }
      fprintf(fd,"];\n %s = spconvert(zzz);\n",outputname);
    } 
    else if (format == FILE_FORMAT_IMPL) {
#if !defined(PETSC_COMPLEX)
      if (nb == 1) { /* diagonal format */
        for (i=0; i< mat->nd; i++) {
          dv   = mat->diagv[i];
          diag = mat->diag[i];
          fprintf(fd,"\n<diagonal %d>\n",diag);
          /* diag[i] is (row-col)/nb */
          if (diag > 0) {  /* lower triangle */
            len = nr - diag;
            for (j=0; j<len; j++)
               if (dv[j]) fprintf(fd,"A[ %d , %d ] = %e\n", j+diag, j, dv[j]);
          }
          else {         /* upper triangle, including main diagonal */
            len = nr + diag;
            for (j=0; j<len; j++)
              if (dv[j]) fprintf(fd,"A[ %d , %d ] = %e\n", j, j-diag, dv[j]);
          }
        }
      } else {  /* Block diagonals */
        int d, k, kshift;
        for (d=0; d< mat->nd; d++) {
          dv   = mat->diagv[d];
          diag = mat->diag[d];
          len  = mat->bdlen[d];
          fprintf(fd,"\n<diagonal %d>\n", diag);
          if (diag > 0) {  /* lower triangle */
            for (k=0; k<len; k++) {
              kshift = k*nb*nb;
              for (i=0; i<nb; i++) {
                for (j=0; j<nb; j++) {
                  if (dv[kshift + j*nb + i])
                    fprintf(fd,"A[%d,%d]=%5.2e   ", (k+diag)*nb + i, 
	                k*nb + j, dv[kshift + j*nb + i] );
                }
                fprintf(fd,"\n");
              }
            }
         } else {  /* upper triangle, including main diagonal */
            for (k=0; k<len; k++) {
              kshift = k*nb*nb;
              for (i=0; i<nb; i++) {
                for (j=0; j<nb; j++) {
                  if (dv[kshift + j*nb + i])
                    fprintf(fd,"A[%d,%d]=%5.2e   ", k*nb + i, 
                          (k-diag)*nb + j, dv[kshift + j*nb + i] );
                }
                fprintf(fd,"\n");
              }
            }
          }
        }
      }
#endif
    } else {
      for (i=0; i<mat->m; i++) { /* the usual row format */
        fprintf(fd,"row %d:",i);
        ierr = MatGetRow( matin, i, &nz, &col, &val ); CHKERRQ(ierr);
        for (j=0; j<nz; j++) {
          if (val[j] != zero)
#if defined(PETSC_COMPLEX)
            if (imag(val[j]) != 0.0) {
              fprintf(fd," %d %g ", col[j], real(val[j]), imag(val[j]) );
            }
            else {
              fprintf(fd," %d %g ", col[j], real(val[j]) );
            }
#else
            fprintf(fd," %d %g ", col[j], val[j] );
#endif
        }
        fprintf(fd,"\n");
        ierr = MatRestoreRow( matin, i, &nz, &col, &val ); CHKERRQ(ierr);
      }
    }
    fflush(fd);
  }
  return 0;
}

static int MatDestroy_SeqBDiag(PetscObject obj)
{
  Mat          bmat = (Mat) obj;
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) bmat->data;
  int          i;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d, BSize=%d, NDiag=%d",
                       mat->m,mat->n,mat->nz,mat->nb,mat->nd);
#endif
  if (!mat->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<mat->nd; i++) PETSCFREE( mat->diagv[i] );
  }
  if (mat->pivots) PETSCFREE(mat->pivots);
  PETSCFREE(mat->diagv);
  PETSCFREE(mat->diag);
  PETSCFREE(mat->colloc);
  PETSCFREE(mat->dvalue);
  PETSCFREE(mat);
  PLogObjectDestroy(bmat);
  PETSCHEADERDESTROY(bmat);
  return 0;
}

static int MatAssemblyEnd_SeqBDiag(Mat matin,MatAssemblyType mode)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  if (mode == FLUSH_ASSEMBLY) return 0;
  mat->assembled = 1;
  return 0;
}

static int MatSetOption_SeqBDiag(Mat mat,MatOption op)
{
  Mat_SeqBDiag *mbd = (Mat_SeqBDiag *) mat->data;
  if (op == NO_NEW_NONZERO_LOCATIONS)       mbd->nonew = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) mbd->nonew = 0;
  return 0;
}

static int MatGetDiagonal_SeqBDiag(Mat matin,Vec v)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int          i, j, n, ibase, nb = mat->nb, iloc;
  Scalar       *x, *dvmain;
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != mat->m) 
     SETERRQ(1,"MatGetDiagonal_SeqBDiag:Nonconforming matrix and vector");
  if (mat->mainbd == -1) 
     SETERRQ(1,"MatGetDiagonal_SeqBDiag:Main diagonal is not set");
  dvmain = mat->diagv[mat->mainbd];
  if (mat->nb == 1) {
    for (i=0; i<mat->m; i++) x[i] = dvmain[i];
  } else {
    for (i=0; i<mat->mblock; i++) {
      ibase = i*nb*nb;  iloc = i*nb;
      for (j=0; j<nb; j++) x[j + iloc] = dvmain[ibase + j*(nb+1)];
    }
  }
  return 0;
}

static int MatZeroEntries_SeqBDiag(Mat matin)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int          d, i, len, nb = mat->nb;
  Scalar       *dv;

  for (d=0; d<mat->nd; d++) {
    dv  = mat->diagv[d];
    len = mat->bdlen[d]*nb*nb;
    for (i=0; i<len; i++) dv[i] = 0.0;
  }
  return 0;
}

static int MatZeroRows_SeqBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_SeqBDiag *l = (Mat_SeqBDiag *) A->data;
  int          i, ierr, N, *rows, m = l->m - 1, nz, *col;
  Scalar       *dvmain, *val;

  ierr = ISGetLocalSize(is,&N); CHKERRQ(ierr);
  ierr = ISGetIndices(is,&rows); CHKERRQ(ierr);
  for ( i=0; i<N; i++ ) {
    if (rows[i] < 0 || rows[i] > m) SETERRQ(1,"MatZeroRows_SeqBDiag:row out of range");
    ierr = MatGetRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
    PetscZero(val,nz*sizeof(Scalar));
    ierr = MatSetValues(A,1,&rows[i],nz,col,val,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatRestoreRow(A,rows[i],&nz,&col,&val); CHKERRQ(ierr);
  }
  if (diag) {
    if (l->mainbd == -1) SETERRQ(1,"MatZeroRows_SeqBDiag:Main diagonal does not exist");
    dvmain = l->diagv[l->mainbd];
    for ( i=0; i<N; i++ ) dvmain[rows[i]] = *diag;
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatGetSize_SeqBDiag(Mat matin,int *m,int *n)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  *m = mat->m; *n = mat->n;
  return 0;
}

extern int MatDetermineDiagonals_Private(Mat,int,int,int,int*,int*,int*,int**);

static int MatGetSubMatrix_SeqBDiag(Mat matin,IS isrow,IS iscol,Mat *submat)
{
  Mat_SeqBDiag *mat = (Mat_SeqBDiag *) matin->data;
  int          nznew, *smap, i, j, ierr, oldcols = mat->n;
  int          *irow, *icol, newr, newc, *cwork, *col,nz, nb, ndiag, *diag;
  Scalar       *vwork, *val;
  Mat          newmat;

  if (!mat->assembled) SETERRQ(1,"MatGetSubMatrix_SeqBDiag:Not for unassembled matrix");
  ierr = ISGetIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISGetIndices(iscol,&icol); CHKERRQ(ierr);
  ierr = ISGetSize(isrow,&newr); CHKERRQ(ierr);
  ierr = ISGetSize(iscol,&newc); CHKERRQ(ierr);

  smap  = (int *) PETSCMALLOC(oldcols*sizeof(int)); CHKPTRQ(smap);
  cwork = (int *) PETSCMALLOC(newc*sizeof(int)); CHKPTRQ(cwork);
  vwork = (Scalar *) PETSCMALLOC(newc*sizeof(Scalar)); CHKPTRQ(vwork);
  PetscZero((char*)smap,oldcols*sizeof(int));
  for ( i=0; i<newc; i++ ) smap[icol[i]] = i+1;

  /* Determine diagonals; then create submatrix */
  nb = mat->nb; /* Default block size remains the same */
  ierr = MatDetermineDiagonals_Private(matin,nb,newr,newc,irow,icol,
         &ndiag,&diag); CHKERRQ(ierr); 
  ierr = MatCreateSeqBDiag(matin->comm,newr,newc,ndiag,nb,diag,
         0,&newmat); CHKERRQ(ierr); 
  PETSCFREE(diag);

  /* Fill new matrix */
  for (i=0; i<newr; i++) {
    ierr = MatGetRow(matin,irow[i],&nz,&col,&val); CHKERRQ(ierr);
    nznew = 0;
    for (j=0; j<nz; j++) {
      if (smap[col[j]]) {
        cwork[nznew]   = smap[col[j]] - 1;
        vwork[nznew++] = val[j];
      }
    }
    ierr = MatSetValues(newmat,1,&i,nznew,cwork,vwork,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatRestoreRow(matin,i,&nz,&col,&val); CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(newmat,FINAL_ASSEMBLY); CHKERRQ(ierr);

  /* Free work space */
  PETSCFREE(smap); PETSCFREE(cwork); PETSCFREE(vwork);
  ierr = ISRestoreIndices(isrow,&irow); CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscol,&icol); CHKERRQ(ierr);
  *submat = newmat;
  return 0;
}

static int MatCopyPrivate_SeqBDiag(Mat,Mat *);
extern int MatLUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,Mat*);
extern int MatLUFactorNumeric_SeqBDiag(Mat,Mat*);
extern int MatLUFactor_SeqBDiag(Mat,IS,IS,double);
extern int MatSolve_SeqBDiag(Mat,Vec,Vec);
extern int MatSolveAdd_SeqBDiag(Mat,Vec,Vec,Vec);
extern int MatSolveTrans_SeqBDiag(Mat,Vec,Vec);
extern int MatSolveTransAdd_SeqBDiag(Mat,Vec,Vec,Vec);

/* -------------------------------------------------------------------*/
static struct _MatOps MatOps = {MatSetValues_SeqBDiag,
       MatGetRow_SeqBDiag, MatRestoreRow_SeqBDiag,
       MatMult_SeqBDiag, MatMultAdd_SeqBDiag, 
       MatMultTrans_SeqBDiag, MatMultTransAdd_SeqBDiag, 
       MatSolve_SeqBDiag,MatSolveAdd_SeqBDiag,
       MatSolveTrans_SeqBDiag,MatSolveTransAdd_SeqBDiag,
       MatLUFactor_SeqBDiag, 0,
       MatRelax_SeqBDiag, MatTranspose_SeqBDiag,
       MatGetInfo_SeqBDiag, 0,
       MatGetDiagonal_SeqBDiag, 0, MatNorm_SeqBDiag,
       0,MatAssemblyEnd_SeqBDiag,
       0, MatSetOption_SeqBDiag, MatZeroEntries_SeqBDiag,MatZeroRows_SeqBDiag, 0,
       MatLUFactorSymbolic_SeqBDiag,MatLUFactorNumeric_SeqBDiag, 0, 0,
       MatGetSize_SeqBDiag,MatGetSize_SeqBDiag,MatGetOwnershipRange_SeqBDiag,
       0, 0,
       0, 0, 0,
       MatGetSubMatrix_SeqBDiag, 0,
       MatCopyPrivate_SeqBDiag, 0, 0 };

/*@C
   MatCreateSeqBDiag - Creates a sequential block diagonal matrix.

   Input Parameters:
.  comm - MPI communicator, set to MPI_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nd - number of block diagonals (optional)
.  nb - each element of a diagonal is an nb x nb dense matrix
.  diag - optional array of block diagonal numbers (length nd),
$     where for a matrix element A[i,j], 
$     where i=row and j=column, the diagonal number is
$     diag = i/nb - j/nb  (integer division)
$     Set diag=0 on input for PETSc to dynamically allocate memory
$     as needed.
.  diagv - pointer to actual diagonals (in same order as diag array), 
   if allocated by user.  Otherwise, set diagv=0 on input for PETSc to 
   control memory allocation.

   Output Parameters:
.  newmat - the matrix

   Notes:
   See the users manual for further details regarding this storage format.

   The case nb=1 (conventional diagonal storage) is implemented as
   a special case. 

   Fortran programmers cannot set diagv; It is ignored.

.keywords: matrix, block, diagonal, sparse

.seealso: MatCreate(), MatCreateMPIBDiag(), MatSetValues()
@*/
int MatCreateSeqBDiag(MPI_Comm comm,int m,int n,int nd,int nb,
                             int *diag,Scalar **diagv,Mat *newmat)
{
  Mat          bmat;
  Mat_SeqBDiag *mat;
  int          i, j, nda, temp, sizetot;
  Scalar       *dtemp;

  *newmat       = 0;
  if ((n%nb) || (m%nb)) SETERRQ(1,"MatCreateSeqBDiag:Invalid block size");
  if (!nd) nda = nd + 1;
  else nda = nd;
  PETSCHEADERCREATE(bmat,_Mat,MAT_COOKIE,MATSEQBDIAG,comm);
  PLogObjectCreate(bmat);
  bmat->data    = (void *) (mat = PETSCNEW(Mat_SeqBDiag)); CHKPTRQ(mat);
  PetscMemcpy(&bmat->ops,&MatOps,sizeof(struct _MatOps));
  bmat->destroy = MatDestroy_SeqBDiag;
  bmat->view    = MatView_SeqBDiag;
  bmat->factor  = 0;

  mat->m      = m;
  mat->n      = n;
  mat->mblock = m/nb;
  mat->nblock = n/nb;
  mat->nd     = nd;
  mat->nb     = nb;
  mat->ndim   = 0;
  mat->mainbd = -1;
  mat->pivots = 0;

  mat->diag   = (int *)PETSCMALLOC(2*nda*sizeof(int)); CHKPTRQ(mat->diag);
  mat->bdlen  = mat->diag + nda;
  mat->colloc = (int *)PETSCMALLOC(n*sizeof(int)); CHKPTRQ(mat->colloc);
  mat->diagv  = (Scalar**)PETSCMALLOC(nda*sizeof(Scalar*)); CHKPTRQ(mat->diagv);
  sizetot = 0;

  if (diagv) { /* user allocated space */
    mat->user_alloc = 1;
    for (i=0; i<nd; i++) mat->diagv[i] = diagv[i];
  }
  else mat->user_alloc = 0;

  /* Sort diagonals in decreasing order. */
  for (i=0; i<nd; i++) {
    for (j=i+1; j<nd; j++) {
      if (diag[i] < diag[j]) {
        temp = diag[i];   
        diag[i] = diag[j];
        diag[j] = temp;
        if (diagv) {
          dtemp = mat->diagv[i];
          mat->diagv[i] = mat->diagv[j];
          mat->diagv[j] = dtemp;
        }
      }
    }
  }

  for (i=0; i<nd; i++) {
    mat->diag[i] = diag[i];
    if (diag[i] > 0) /* lower triangular */
      mat->bdlen[i] = PETSCMIN(mat->nblock,mat->mblock - diag[i]);
    else {           /* upper triangular */
      if (mat->mblock - diag[i] > mat->nblock)
        mat->bdlen[i] = mat->nblock + diag[i];
  /*    mat->bdlen[i] = mat->mblock + diag[i] + (mat->nblock - mat->mblock); */
      else
        mat->bdlen[i] = mat->mblock;
    }
    sizetot += mat->bdlen[i];
    if (diag[i] == 0) mat->mainbd = i;
  }
  sizetot *= nb*nb;
  if (nda != nd) sizetot += 1;
  mat->maxnz  = sizetot;
  mat->dvalue = (Scalar *)PETSCMALLOC(n*sizeof(Scalar)); CHKPTRQ(mat->dvalue);
  PLogObjectMemory(bmat,(nda*(nb+2))*sizeof(int) + nb*nda*sizeof(Scalar)
                    + nda*sizeof(Scalar*) + sizeof(Mat_SeqBDiag)
                    + sizeof(struct _Mat) + sizetot*sizeof(Scalar));

  if (!mat->user_alloc) {
    for (i=0; i<nd; i++) {
      mat->diagv[i] = (Scalar*)PETSCMALLOC(nb*nb*mat->bdlen[i]*sizeof(Scalar));
      CHKPTRQ(mat->diagv[i]);
      PetscZero(mat->diagv[i],nb*nb*mat->bdlen[i]*sizeof(Scalar));
    }
    mat->nonew = 0;
  } else { /* diagonals are set on input; don't allow dynamic allocation */
    mat->nonew = 1;
  }

  mat->nz        = mat->maxnz; /* Currently not keeping track of exact count */
  mat->assembled = 0;
  *newmat        = bmat;
  return 0;
}

static int MatCopyPrivate_SeqBDiag(Mat matin,Mat *matout)
{ 
  Mat_SeqBDiag *newmat, *oldmat = (Mat_SeqBDiag *) matin->data;
  int          i, ierr, len;
  Mat          mat;

  if (!oldmat->assembled) SETERRQ(1,"MatCopyPrivate_SeqBDiag:Assemble matrix");

  ierr = MatCreateSeqBDiag(matin->comm,oldmat->m,oldmat->n,
                         oldmat->nd,oldmat->nb,oldmat->diag,0,matout); CHKERRQ(ierr);

  /* Copy contents of diagonals */
  mat = *matout;
  newmat = (Mat_SeqBDiag *) mat->data;
  for (i=0; i<oldmat->nd; i++) {
    len = oldmat->bdlen[i] * oldmat->nb * oldmat->nb * sizeof(Scalar);
    PetscMemcpy(newmat->diagv[i],oldmat->diagv[i],len);
  }
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
