#ifndef lint
static char vcid[] = "$Id: bdiag.c,v 1.61 1995/10/13 19:40:06 curfman Exp curfman $";
#endif

/* Block diagonal matrix format */

#include "bdiag.h"
#include "vec/vecimpl.h"
#include "inline/spops.h"

static int MatSetValues_SeqBDiag(Mat A,int m,int *im,int n,int *in,
                                 Scalar *v,InsertMode is)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          kk, loc, ldiag, shift, row, dfound, newnz, *bdlen_new;
  int          j, k, nb = a->nb, *diag_new;
  Scalar       value, *valpt, **diagv_new;

  if (nb == 1) {
    for ( kk=0; kk<m; kk++ ) { /* loop over added rows */
      row = im[kk];   
      if (row < 0) SETERRQ(1,"MatSetValues_SeqBDiag:Negative row");
      if (row >= a->m) SETERRQ(1,"MatSetValues_SeqBDiag:Row too large");
      for (j=0; j<n; j++) {
        if (in[j] < 0) SETERRQ(1,"MatSetValues_SeqBDiag:Negative column");
        if (in[j] >= a->n) SETERRQ(1,"MatSetValues_SeqBDiag:Column too large");
        ldiag = row - in[j]; /* diagonal number */
        dfound = 0;
        value = *v++;
        for (k=0; k<a->nd; k++) {
	  if (a->diag[k] == ldiag) {
            dfound = 1;
	    if (ldiag > 0) loc = row - ldiag; /* lower triangle */
	    else           loc = row;
	    if ((valpt = &((a->diagv[k])[loc]))) {
	      if (is == ADD_VALUES) *valpt += value;
	      else                  *valpt = value;
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            break;
          }
        }
        if (!dfound) {
          if (a->nonew || a->nonew_diag) {
#if !defined(PETSC_COMPLEX)
            if (a->user_alloc && value) {
#else
            if (a->user_alloc && real(value) || imag(value)) {
#endif
              PLogInfo((PetscObject)A,
                "MatSetValues_SeqBDiag: Nonzero in diagonal %d that user did not allocate\n",ldiag);
            }
          } else {
            PLogInfo((PetscObject)A,"MatSetValues_SeqBDiag: Allocating new diagonal: %d\n",ldiag);
            /* free old bdiag storage info and reallocate */
            diag_new = (int *)PETSCMALLOC(2*(a->nd+1)*sizeof(int)); CHKPTRQ(diag_new);
            bdlen_new = diag_new + a->nd + 1;
            diagv_new = (Scalar**)PETSCMALLOC((a->nd+1)*sizeof(Scalar*)); CHKPTRQ(diagv_new);
            for (k=0; k<a->nd; k++) {
              diag_new[k]  = a->diag[k];
              diagv_new[k] = a->diagv[k];
              bdlen_new[k] = a->bdlen[k];
            }
            diag_new[a->nd]  = ldiag;
            if (ldiag > 0) /* lower triangular */
              bdlen_new[a->nd] = PETSCMIN(a->nblock,a->mblock - ldiag);
            else {         /* upper triangular */
              if (a->mblock - ldiag > a->nblock)
                bdlen_new[a->nd] = a->nblock + ldiag;
              else
                bdlen_new[a->nd] = a->mblock;
            }
            newnz = bdlen_new[a->nd];
            diagv_new[a->nd] = (Scalar*)PETSCMALLOC(newnz*sizeof(Scalar));
            CHKPTRQ(diagv_new[a->nd]);
            PetscZero(diagv_new[a->nd],newnz*sizeof(Scalar));
            a->maxnz += newnz;
            a->nz += newnz;
            PETSCFREE(a->diagv); PETSCFREE(a->diag); 
            a->diag  = diag_new; 
            a->bdlen = bdlen_new;
            a->diagv = diagv_new;

            /* Insert value */
	    if (ldiag > 0) loc = row - ldiag; /* lower triangle */
	    else           loc = row;
	    if ((valpt = &((a->diagv[a->nd])[loc]))) {
	      if (is == ADD_VALUES) *valpt += value;
	      else                  *valpt = value;
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            a->nd++;
            PLogObjectMemory(A,newnz*sizeof(Scalar) + 2*sizeof(int) + sizeof(Scalar*));
          }
        }
      }
    }
  } else {

    for ( kk=0; kk<m; kk++ ) { /* loop over added rows */
      row = im[kk];   
      if (row < 0) SETERRQ(1,"MatSetValues_SeqBDiag:Negative row");
      if (row >= a->m) SETERRQ(1,"MatSetValues_SeqBDiag:Row too large");
      shift = (row/nb)*nb*nb + row%nb;
      for (j=0; j<n; j++) {
        ldiag = row/nb - in[j]/nb; /* block diagonal */
        dfound = 0;
        value = *v++;
        for (k=0; k<a->nd; k++) {
          if (a->diag[k] == ldiag) {
            dfound = 1;
	    if (ldiag > 0) /* lower triangle */
	      loc = shift - ldiag*nb*nb;
             else
	      loc = shift;
	    if ((valpt = &((a->diagv[k])[loc + (in[j]%nb)*nb ]))) {
	      if (is == ADD_VALUES) *valpt += value;
	      else                  *valpt = value;
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            break;
          }
        }
        if (!dfound) {
          if (a->nonew || a->nonew_diag) {
#if !defined(PETSC_COMPLEX)
            if (a->user_alloc && value) {
#else
            if (a->user_alloc && real(value) || imag(value)) {
#endif
              PLogInfo((PetscObject)A,
                "MatSetValues_SeqBDiag: Nonzero in diagonal %d that user did not allocate\n",ldiag);
            }
          } else {
            PLogInfo((PetscObject)A,"MatSetValues_SeqBDiag: Allocating new diagonal: %d\n",ldiag);
            /* free old bdiag storage info and reallocate */
            diag_new = (int *)PETSCMALLOC(2*(a->nd+1)*sizeof(int)); CHKPTRQ(diag_new);
            bdlen_new = diag_new + a->nd + 1;
            diagv_new = (Scalar**)PETSCMALLOC((a->nd+1)*sizeof(Scalar*)); CHKPTRQ(diagv_new);
            for (k=0; k<a->nd; k++) {
              diag_new[k]  = a->diag[k];
              diagv_new[k] = a->diagv[k];
              bdlen_new[k] = a->bdlen[k];
            }
            diag_new[a->nd]  = ldiag;
            if (ldiag > 0) /* lower triangular */
              bdlen_new[a->nd] = PETSCMIN(a->nblock,a->mblock - ldiag);
            else {         /* upper triangular */
              if (a->mblock - ldiag > a->nblock)
                bdlen_new[a->nd] = a->nblock + ldiag;
              else
                bdlen_new[a->nd] = a->mblock;
            }
            newnz = nb*nb*bdlen_new[a->nd];
            diagv_new[a->nd] = (Scalar*)PETSCMALLOC(newnz*sizeof(Scalar));
            CHKPTRQ(diagv_new[a->nd]);
            PetscZero(diagv_new[a->nd],newnz*sizeof(Scalar));
            a->maxnz += newnz; a->nz += newnz;
            PETSCFREE(a->diagv); PETSCFREE(a->diag); 
            a->diag  = diag_new; 
            a->bdlen = bdlen_new;
            a->diagv = diagv_new;

            /* Insert value */
	    if (ldiag > 0) /* lower triangle */
	      loc = shift - ldiag*nb*nb;
             else
	      loc = shift;
	    if ((valpt = &((a->diagv[k])[loc + (in[j]%nb)*nb ]))) {
	      if (is == ADD_VALUES) *valpt += value;
	      else                  *valpt = value;
            } else SETERRQ(1,"MatSetValues_SeqBDiag:Invalid data location");
            a->nd++;
            PLogObjectMemory(A,newnz*sizeof(Scalar) + 2*sizeof(int) + sizeof(Scalar*));
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
static int MatMult_SeqBDiag_base(Mat A,Vec xx,Vec yy)
{ 
  Mat_SeqBDiag    *a = (Mat_SeqBDiag *) A->data;
  int             nd = a->nd, nb = a->nb, diag, kshift, kloc;
  Scalar          *vin, *vout;
  register Scalar *pvin, *pvout, *dv;
  register int    d, i, j, k, len;

  VecGetArray(xx,&vin); VecGetArray(yy,&vout);
  if (nb == 1) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
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
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
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
static int MatMultTrans_SeqBDiag_base(Mat A,Vec xx,Vec yy)
{
  Mat_SeqBDiag    *a = (Mat_SeqBDiag *) A->data;
  int             nd = a->nd, nb = a->nb, diag,kshift, kloc;
  register Scalar *pvin, *pvout, *dv;
  register int    d, i, j, k, len;
  Scalar          *vin, *vout;
  
  VecGetArray(xx,&vin); VecGetArray(yy,&vout);
  if (nb == 1) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
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
      dv   = a->diagv[d];
      diag = a->diag[d];
      len  = a->bdlen[d];
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
static int MatMult_SeqBDiag(Mat A,Vec xx,Vec yy)
{
  Scalar zero = 0.0;
  int    ierr;
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  return MatMult_SeqBDiag_base(A,xx,yy);
}
static int MatMultTrans_SeqBDiag(Mat A,Vec xx,Vec yy)
{
  Scalar zero = 0.0;
  int    ierr;
  ierr = VecSet(&zero,yy); CHKERRQ(ierr);
  return MatMultTrans_SeqBDiag_base(A,xx,yy);
}
static int MatMultAdd_SeqBDiag(Mat A,Vec xx,Vec zz,Vec yy)
{
  int ierr;
  ierr = VecCopy(zz,yy); CHKERRQ(ierr);
  return MatMult_SeqBDiag_base(A,xx,yy);
}
static int MatMultTransAdd_SeqBDiag(Mat A,Vec xx,Vec zz,Vec yy)
{
  int ierr;
  ierr = VecCopy(zz,yy); CHKERRQ(ierr);
  return MatMultTrans_SeqBDiag_base(A,xx,yy);
}

static int MatRelax_SeqBDiag(Mat A,Vec bb,double omega,MatSORType flag,
                             double shift,int its,Vec xx)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  Scalar       *x, *b, *xb, *dd, *dv, dval, sum;
  int          m = a->m, i, j, k, d, kbase, nb = a->nb, loc, kloc;
  int          mainbd = a->mainbd, diag, mblock = a->mblock, bloc;

  /* Currently this code doesn't use wavefront orderings, although
     we should eventually incorporate that option */
  VecGetArray(xx,&x); VecGetArray(bb,&b);
  if (mainbd == -1) SETERRQ(1,"MatRelax_SeqBDiag:Main diagonal not set");
  dd = a->diagv[mainbd];
  if (flag == SOR_APPLY_UPPER) {
    /* apply ( U + D/omega) to the vector */
    if (nb == 1) {
      for ( i=0; i<m; i++ ) {
        sum = b[i] * (shift + dd[i]) / omega;
        for (d=mainbd+1; d<a->nd; d++) {
          diag = a->diag[d];
          if (i-diag < m) sum += a->diagv[d][i] * x[i-diag];
        }
        x[i] = sum;
      }
    } else {
      for ( k=0; k<mblock; k++ ) {
        kloc = k*nb; kbase = kloc*nb;
        for (i=0; i<nb; i++) {
          sum = b[i+kloc] * (shift + dd[i*(nb+1)+kbase]) / omega;
          for (j=i+1; j<nb; j++)
            sum += dd[kbase + j*nb + i] * b[kloc + j];
          for (d=mainbd+1; d<a->nd; d++) {
            diag = a->diag[d];
            dv   = a->diagv[d];
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
            loc = i - a->diag[d];
            if (loc >= 0) sum -= a->diagv[d][loc] * x[loc];
          }
          x[i] = omega*(sum/(shift + dd[i]));
        }
      } else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++) {
            sum  = b[i+kloc];
            dval = shift + dd[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (j=0; j<i; j++)
              sum -= dd[kbase + j*nb + i] * x[kloc + j];
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
        for ( i=0; i<m; i++ ) x[i] *= dd[i];
      } 
      else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++)
            x[kloc+i] *= dd[i*(nb+1)+kbase];
        }
      }
    }
    if (flag & SOR_BACKWARD_SWEEP || flag & SOR_LOCAL_BACKWARD_SWEEP) {
      if (nb == 1) {
        for ( i=m-1; i>=0; i-- ) {
          sum = xb[i];
          for (d=mainbd+1; d<a->nd; d++) {
            diag = a->diag[d];
            if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
          }
          x[i] = omega*(sum/(shift + dd[i]));
        }
      } 
      else {
        for ( k=mblock-1; k>=0; k-- ) {
          kloc = k*nb; kbase = kloc*nb;
          for ( i=nb-1; i>=0; i-- ) {
            sum  = xb[i+kloc];
            dval = shift + dd[i*(nb+1)+kbase];
            for ( j=i+1; j<nb; j++ )
              sum -= dd[kbase + j*nb + i] * x[kloc + j];
            for (d=mainbd+1; d<a->nd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
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
          dval = shift + dd[i];
          for (d=0; d<mainbd; d++) {
            loc = i - a->diag[d];
            if (loc >= 0) sum -= a->diagv[d][loc] * x[loc];
          }
          for (d=mainbd; d<a->nd; d++) {
            diag = a->diag[d];
            if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
          }
          x[i] = (1. - omega)*x[i] + omega*(sum/dval + x[i]);
        }
      } else {
        for ( k=0; k<mblock; k++ ) {
          kloc = k*nb; kbase = kloc*nb;
          for (i=0; i<nb; i++) {
            sum  = b[i+kloc];
            dval = shift + dd[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (d=mainbd; d<a->nd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
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
            loc = i - a->diag[d];
            if (loc >= 0) sum -= a->diagv[d][loc] * x[loc];
          }
          for (d=mainbd; d<a->nd; d++) {
            diag = a->diag[d];
            if (i-diag < m) sum -= a->diagv[d][i] * x[i-diag];
          }
          x[i] = (1. - omega)*x[i] + omega*(sum/(shift + dd[i]) + x[i]);
        }
      } 
      else {
        for ( k=mblock-1; k>=0; k-- ) {
          kloc = k*nb; kbase = kloc*nb;
          for ( i=nb-1; i>=0; i-- ) {
            sum  = b[i+kloc];
            dval = shift + dd[i*(nb+1)+kbase];
            for (d=0; d<mainbd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
              bloc = k - diag;
              if (bloc >= 0) {
                for (j=0; j<nb; j++)
                  sum -= dv[bloc*nb*nb + j*nb + i] * x[bloc*nb + j];
              }
	    }
            for (d=mainbd; d<a->nd; d++) {
              diag = a->diag[d];
              dv   = a->diagv[d];
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

static int MatGetInfo_SeqBDiag(Mat A,MatInfoType flag,int *nz,int *nzalloc,int *mem)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  *nz      = a->nz;
  *nzalloc = a->maxnz;
  *mem     = (int)A->mem;
  return 0;
}

static int MatGetOwnershipRange_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  *m = 0; *n = a->m;
  return 0;
}

static int MatGetRow_SeqBDiag(Mat A,int row,int *nz,int **col,Scalar **v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nd = a->nd, nb = a->nb, loc;
  int          nc = a->n, *diag = a->diag, pcol, shift, i, j, k;

/* For efficiency, if ((nz) && (col) && (v)) then do all at once */
  if ((nz) && (col) && (v)) {
    *col = a->colloc;
    *v   = a->dvalue;
    k    = 0;
    if (nb == 1) { 
      for (j=0; j<nd; j++) {
        pcol = row - diag[j];
        if (pcol > -1 && pcol < nc) {
	  if (diag[j] > 0)
	    loc = row - diag[j];
	  else
	    loc = row;
	  (*v)[k]   = (a->diagv[j])[loc];
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
	    (*v)[k+i]   = (a->diagv[j])[loc + i*nb];
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
	    if (diag[j] > 0)
	      loc = row - diag[j];
	    else
	      loc = row;
	    (*v)[k] = (a->diagv[j])[loc]; k++;
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
        *col = a->colloc;
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
        *v = a->dvalue;
        k = 0;
        for (j=0; j<nd; j++) {
	  pcol = nb * (row/nb - diag[j]);
	  if (pcol > -1 && pcol < nc) {
	    if (diag[j] > 0)
	      loc = shift - diag[j]*nb*nb;
	    else 
	      loc = shift;
	    for (i=0; i<nb; i++) {
	     (*v)[k+i] = (a->diagv[j])[loc + i*nb];
            }
	    k += nb;
	  }
        }
      }
    }
  }
  return 0;
}

static int MatRestoreRow_SeqBDiag(Mat A,int row,int *ncols,int **cols,Scalar **vals)
{
  /* Work space is allocated once during matrix creation and then freed
     when matrix is destroyed */
  return 0;
}

static int MatNorm_SeqBDiag(Mat A,MatNormType type,double *norm)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  double       sum = 0.0, *tmp;
  int          d, i, j, k, nd = a->nd, nb = a->nb, diag, kshift, kloc, len;
  Scalar       *dv;

  if (!a->assembled) SETERRQ(1,"MatNorm_SeqBDiag:Must assemble mat");

  if (type == NORM_FROBENIUS) {
    for (d=0; d<nd; d++) {
      dv   = a->diagv[d];
      len  = a->bdlen[d]*nb*nb;
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
    tmp = (double *) PETSCMALLOC( a->n*sizeof(double) ); CHKPTRQ(tmp);
    PetscZero(tmp,a->n*sizeof(double));
    *norm = 0.0;
    if (nb == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
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
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];

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
    for ( j=0; j<a->n; j++ ) {
      if (tmp[j] > *norm) *norm = tmp[j];
    }
    PETSCFREE(tmp);
  }
  else if (type == NORM_INFINITY) { /* max row norm */
    tmp = (double *) PETSCMALLOC( a->m*sizeof(double) ); CHKPTRQ(tmp);
    PetscZero(tmp,a->m*sizeof(double));
    *norm = 0.0;
    if (nb == 1) {
      for (d=0; d<nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
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
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
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
    for ( j=0; j<a->m; j++ ) {
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
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data, *anew;
  Mat          tmat;
  int          i, j, k, d, ierr, nd = a->nd, *diag = a->diag, *diagnew;
  int          nb = a->nb, kshift;
  Scalar       *dwork, *dvnew;

  diagnew = (int *) PETSCMALLOC(nd*sizeof(int)); CHKPTRQ(diagnew);
  for (i=0; i<nd; i++) {
    diagnew[i] = -diag[nd-i-1]; /* assume sorted in descending order */
  }
  ierr = MatCreateSeqBDiag(A->comm,a->n,a->m,nd,nb,diagnew,
                                    0,&tmat); CHKERRQ(ierr);
  PETSCFREE(diagnew);
  anew = (Mat_SeqBDiag *) tmat->data;
  for (d=0; d<nd; d++) {
    dvnew = anew->diagv[d];
    dwork = a->diagv[nd-d-1];
    if (anew->bdlen[d] != a->bdlen[nd-d-1])
      SETERRQ(1,"MatTranspose_SeqBDiag:Incompatible diagonal lengths");
    if (nb == 1) {
      for (k=0; k<anew->bdlen[d]; k++) dvnew[k] = dwork[k];
    } else {
      for (k=0; k<anew->bdlen[d]; k++) {
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
       structures from a.  We should fix this. */
    if (!a->user_alloc) { /* Free the actual diagonals */
      for (i=0; i<a->nd; i++) PETSCFREE( a->diagv[i] );
    }
    if (a->pivot) PETSCFREE(a->pivot);
    PETSCFREE(a->diagv); PETSCFREE(a->diag);
    PETSCFREE(a->colloc); PETSCFREE(a->dvalue);
    PETSCFREE(a);
    PetscMemcpy(A,tmat,sizeof(struct _Mat)); 
    PETSCHEADERDESTROY(tmat);
  }
  return 0;
}

/* ----------------------------------------------------------------*/

#include "draw.h"
#include "pinclude/pviewer.h"
#include "sysio.h"

static int MatView_SeqBDiag_Binary(Mat A,Viewer viewer)
{
  SETERRQ(1,"MatView_SeqBDiag_Binary: Routine not finished yet.");
  return 0;
}

static int MatView_SeqBDiag_ASCII(Mat A,Viewer viewer)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  FILE         *fd;
  char         *outputname;
  int          ierr, *col, i, j, len, diag, nr = a->m, nb = a->nb;
  int          format, nz, nzalloc, mem;
  Scalar       *val, *dv, zero = 0.0;

  ierr = ViewerFileGetPointer_Private(viewer,&fd); CHKERRQ(ierr);
  ierr = ViewerFileGetOutputname_Private(viewer,&outputname); CHKERRQ(ierr);
  ierr = ViewerFileGetFormat_Private(viewer,&format); CHKERRQ(ierr);
  if (format == FILE_FORMAT_INFO) {
    fprintf(fd,"  block size=%d, number of diagonals=%d\n",nb,a->nd);
  }
  else if (format == FILE_FORMAT_MATLAB) {
    MatGetInfo(A,MAT_LOCAL,&nz,&nzalloc,&mem);
    fprintf(fd,"%% Size = %d %d \n",nr, a->n);
    fprintf(fd,"%% Nonzeros = %d \n",nz);
    fprintf(fd,"zzz = zeros(%d,3);\n",nz);
    fprintf(fd,"zzz = [\n");
    for ( i=0; i<a->m; i++ ) {
      ierr = MatGetRow( A, i, &nz, &col, &val ); CHKERRQ(ierr);
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
      for (i=0; i< a->nd; i++) {
        dv   = a->diagv[i];
        diag = a->diag[i];
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
      for (d=0; d< a->nd; d++) {
        dv   = a->diagv[d];
        diag = a->diag[d];
        len  = a->bdlen[d];
	fprintf(fd,"\n<diagonal %d>\n", diag);
	if (diag > 0) {		/* lower triangle */
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
	} else {		/* upper triangle, including main diagonal */
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
    for (i=0; i<a->m; i++) { /* the usual row format */
      fprintf(fd,"row %d:",i);
      ierr = MatGetRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
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
      ierr = MatRestoreRow(A,i,&nz,&col,&val); CHKERRQ(ierr);
    }
  }
  fflush(fd);
  return 0;
}

static int MatView_SeqBDiag_Draw(Mat A,Viewer viewer)
{
  Mat_SeqBDiag  *a = (Mat_SeqBDiag *) A->data;
  DrawCtx draw = (DrawCtx) viewer;
  double  xl, yl, xr, yr, w, h;
  int     ierr, nz, *col, i, j, nr = a->m;

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
  return 0;
}

static int MatView_SeqBDiag(PetscObject obj,Viewer viewer)
{
  Mat         A = (Mat) obj;
  Mat_SeqBDiag  *a = (Mat_SeqBDiag*) A->data;
  PetscObject vobj = (PetscObject) viewer;

  if (!a->assembled) SETERRQ(1,"MatView_SeqBDiag:Not for unassembled matrix");
  if (!viewer) { 
    viewer = STDOUT_VIEWER_SELF; vobj = (PetscObject) viewer;
  }
  if (vobj->cookie == VIEWER_COOKIE) {
    if (vobj->type == MATLAB_VIEWER) {
      SETERRQ(PETSC_ERR_SUP,"MatView_SeqBDiag:Matlab viewer");
    }
    else if (vobj->type == ASCII_FILE_VIEWER || vobj->type == ASCII_FILES_VIEWER){
      return MatView_SeqBDiag_ASCII(A,viewer);
    }
    else if (vobj->type == BINARY_FILE_VIEWER) {
      return MatView_SeqBDiag_Binary(A,viewer);
    }
  }
  else if (vobj->cookie == DRAW_COOKIE) {
    if (vobj->type == NULLWINDOW) return 0;
    else return MatView_SeqBDiag_Draw(A,viewer);
  }
  return 0;
}

static int MatDestroy_SeqBDiag(PetscObject obj)
{
  Mat          A = (Mat) obj;
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i;

#if defined(PETSC_LOG)
  PLogObjectState(obj,"Rows=%d, Cols=%d, NZ=%d, BSize=%d, NDiag=%d",
                       a->m,a->n,a->nz,a->nb,a->nd);
#endif
  if (!a->user_alloc) { /* Free the actual diagonals */
    for (i=0; i<a->nd; i++) PETSCFREE( a->diagv[i] );
  }
  if (a->pivot) PETSCFREE(a->pivot);
  PETSCFREE(a->diagv); PETSCFREE(a->diag);
  PETSCFREE(a->colloc); PETSCFREE(a->dvalue);
  PETSCFREE(a);
  PLogObjectDestroy(A);
  PETSCHEADERDESTROY(A);
  return 0;
}

static int MatAssemblyEnd_SeqBDiag(Mat A,MatAssemblyType mode)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, k, temp, *diag = a->diag, *bdlen = a->bdlen;
  Scalar       *dtemp, **dv = a->diagv;

  if (mode == FLUSH_ASSEMBLY) return 0;

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
  a->assembled = 1;
  return 0;
}

static int MatSetOption_SeqBDiag(Mat A,MatOption op)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  if (op == NO_NEW_NONZERO_LOCATIONS)       a->nonew = 1;
  else if (op == YES_NEW_NONZERO_LOCATIONS) a->nonew = 0;
  else if (op == NO_NEW_DIAGONALS)          a->nonew_diag = 1;
  else if (op == YES_NEW_DIAGONALS)         a->nonew_diag = 0;
  else if (op == ROWS_SORTED || 
           op == ROW_ORIENTED ||
           op == SYMMETRIC_MATRIX ||
           op == STRUCTURALLY_SYMMETRIC_MATRIX)
    PLogInfo((PetscObject)A,"Info:MatSetOption_SeqBDiag:Option ignored\n");
  else if (op == COLUMN_ORIENTED)
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqBDiag:COLUMN_ORIENTED");}
  else 
    {SETERRQ(PETSC_ERR_SUP,"MatSetOption_SeqBDiag:unknown option");}
  return 0;
}

static int MatGetDiagonal_SeqBDiag(Mat A,Vec v)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, j, n, ibase, nb = a->nb, iloc;
  Scalar       *x, *dd;
  VecGetArray(v,&x); VecGetLocalSize(v,&n);
  if (n != a->m) 
     SETERRQ(1,"MatGetDiagonal_SeqBDiag:Nonconforming matrix and vector");
  if (a->mainbd == -1) 
     SETERRQ(1,"MatGetDiagonal_SeqBDiag:Main diagonal is not set");
  dd = a->diagv[a->mainbd];
  if (a->nb == 1) {
    for (i=0; i<a->m; i++) x[i] = dd[i];
  } else {
    for (i=0; i<a->mblock; i++) {
      ibase = i*nb*nb;  iloc = i*nb;
      for (j=0; j<nb; j++) x[j + iloc] = dd[ibase + j*(nb+1)];
    }
  }
  return 0;
}

static int MatZeroEntries_SeqBDiag(Mat A)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          d, i, len, nb = a->nb;
  Scalar       *dv;

  for (d=0; d<a->nd; d++) {
    dv  = a->diagv[d];
    len = a->bdlen[d]*nb*nb;
    for (i=0; i<len; i++) dv[i] = 0.0;
  }
  return 0;
}

static int MatZeroRows_SeqBDiag(Mat A,IS is,Scalar *diag)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, N, *rows, m = a->m - 1, nz, *col;
  Scalar       *dd, *val;

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
    if (a->mainbd == -1) SETERRQ(1,"MatZeroRows_SeqBDiag:Main diagonal does not exist");
    dd = a->diagv[a->mainbd];
    for ( i=0; i<N; i++ ) dd[rows[i]] = *diag;
  }
  ISRestoreIndices(is,&rows);
  ierr = MatAssemblyBegin(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}

static int MatGetSize_SeqBDiag(Mat A,int *m,int *n)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  *m = a->m; *n = a->n;
  return 0;
}

static int MatGetSubMatrix_SeqBDiag(Mat A,IS isrow,IS iscol,Mat *submat)
{
  Mat_SeqBDiag *a = (Mat_SeqBDiag *) A->data;
  int          nznew, *smap, i, j, ierr, oldcols = a->n;
  int          *irow, *icol, newr, newc, *cwork, *col,nz, nb;
  Scalar       *vwork, *val;
  Mat          newmat;

  if (!a->assembled) SETERRQ(1,"MatGetSubMatrix_SeqBDiag:Not for unassembled matrix");
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
  nb = a->nb; /* Default block size remains the same */
  ierr = MatCreateSeqBDiag(A->comm,newr,newc,0,nb,0,0,&newmat); CHKERRQ(ierr); 

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
extern int MatILUFactorSymbolic_SeqBDiag(Mat,IS,IS,double,int,Mat*);
extern int MatLUFactorNumeric_SeqBDiag(Mat,Mat*);
extern int MatLUFactor_SeqBDiag(Mat,IS,IS,double);
extern int MatILUFactor_SeqBDiag(Mat,IS,IS,double,int);
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
       MatILUFactorSymbolic_SeqBDiag, 0,
       0, 0, 0,
       MatGetSubMatrix_SeqBDiag, 0,
       MatCopyPrivate_SeqBDiag, 0, 0,
       MatILUFactor_SeqBDiag, 0, 0 };

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
int MatCreateSeqBDiag(MPI_Comm comm,int m,int n,int nd,int nb,int *diag,
                      Scalar **diagv,Mat *newmat)
{
  Mat          A;
  Mat_SeqBDiag *a;
  int          i, nda, sizetot;

  *newmat       = 0;
  if ((n%nb) || (m%nb)) SETERRQ(1,"MatCreateSeqBDiag:Invalid block size");
  if (!nd) nda = nd + 1;
  else nda = nd;
  PETSCHEADERCREATE(A,_Mat,MAT_COOKIE,MATSEQBDIAG,comm);
  PLogObjectCreate(A);
  A->data    = (void *) (a = PETSCNEW(Mat_SeqBDiag)); CHKPTRQ(a);
  PetscMemcpy(&A->ops,&MatOps,sizeof(struct _MatOps));
  A->destroy = MatDestroy_SeqBDiag;
  A->view    = MatView_SeqBDiag;
  A->factor  = 0;

  a->m      = m;
  a->n      = n;
  a->mblock = m/nb;
  a->nblock = n/nb;
  a->nd     = nd;
  a->nb     = nb;
  a->ndim   = 0;
  a->mainbd = -1;
  a->pivot  = 0;

  a->diag   = (int *)PETSCMALLOC(2*nda*sizeof(int)); CHKPTRQ(a->diag);
  a->bdlen  = a->diag + nda;
  a->colloc = (int *)PETSCMALLOC(n*sizeof(int)); CHKPTRQ(a->colloc);
  a->diagv  = (Scalar**)PETSCMALLOC(nda*sizeof(Scalar*)); CHKPTRQ(a->diagv);
  sizetot = 0;

  if (diagv) { /* user allocated space */
    a->user_alloc = 1;
    for (i=0; i<nd; i++) a->diagv[i] = diagv[i];
  }
  else a->user_alloc = 0;

  for (i=0; i<nd; i++) {
    a->diag[i] = diag[i];
    if (diag[i] > 0) /* lower triangular */
      a->bdlen[i] = PETSCMIN(a->nblock,a->mblock - diag[i]);
    else {           /* upper triangular */
      if (a->mblock - diag[i] > a->nblock)
        a->bdlen[i] = a->nblock + diag[i];
  /*    a->bdlen[i] = a->mblock + diag[i] + (a->nblock - a->mblock); */
      else
        a->bdlen[i] = a->mblock;
    }
    sizetot += a->bdlen[i];
  }
  sizetot *= nb*nb;
  a->maxnz  = sizetot;
  a->dvalue = (Scalar *)PETSCMALLOC(n*sizeof(Scalar)); CHKPTRQ(a->dvalue);
  PLogObjectMemory(A,(nda*(nb+2))*sizeof(int) + nb*nda*sizeof(Scalar)
                    + nda*sizeof(Scalar*) + sizeof(Mat_SeqBDiag)
                    + sizeof(struct _Mat) + sizetot*sizeof(Scalar));

  if (!a->user_alloc) {
    for (i=0; i<nd; i++) {
      a->diagv[i] = (Scalar*)PETSCMALLOC(nb*nb*a->bdlen[i]*sizeof(Scalar));
      CHKPTRQ(a->diagv[i]);
      PetscZero(a->diagv[i],nb*nb*a->bdlen[i]*sizeof(Scalar));
    }
    a->nonew = 0; a->nonew_diag = 0;
  } else { /* diagonals are set on input; don't allow dynamic allocation */
    a->nonew = 1; a->nonew_diag = 1;
  }

  a->nz         = a->maxnz; /* Currently not keeping track of exact count */
  a->assembled  = 0;
  *newmat       = A;
  return 0;
}

static int MatCopyPrivate_SeqBDiag(Mat A,Mat *matout)
{ 
  Mat_SeqBDiag *newmat, *a = (Mat_SeqBDiag *) A->data;
  int          i, ierr, len;
  Mat          mat;

  if (!a->assembled) SETERRQ(1,"MatCopyPrivate_SeqBDiag:Assemble matrix");

  ierr = MatCreateSeqBDiag(A->comm,a->m,a->n,a->nd,a->nb,a->diag,0,matout);
  CHKERRQ(ierr);

  /* Copy contents of diagonals */
  mat = *matout;
  newmat = (Mat_SeqBDiag *) mat->data;
  for (i=0; i<a->nd; i++) {
    len = a->bdlen[i] * a->nb * a->nb * sizeof(Scalar);
    PetscMemcpy(newmat->diagv[i],a->diagv[i],len);
  }
  ierr = MatAssemblyBegin(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,FINAL_ASSEMBLY); CHKERRQ(ierr);
  return 0;
}
