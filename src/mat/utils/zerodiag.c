#ifndef lint
static char vcid[] = "$Id: zerodiag.c,v 1.6 1995/11/01 19:11:16 bsmith Exp bsmith $";
#endif

/*
    This file contains routines to reorder a matrix so that the diagonal
    elements are nonzero.
 */

#include "matimpl.h"       /*I  "mat.h"  I*/
#include <math.h>

#define SWAP(a,b) {int _t; _t = a; a = b; b = _t; }

/* Given a current row and current permutation, find a column permutation
   that removes a zero diagonal */
int MatZeroFindPre_Private(Mat mat,int prow,int* row,int* col,double repla,
                           double atol,int* rc,double* rcv )
{
  int      k, nz, repl, *j, kk, nnz, *jj;
  Scalar   *v, *vv;

  MatGetRow( mat, row[prow], &nz, &j, &v );
  for (k=0; k<nz; k++) {
    if (col[j[k]] < prow && PetscAbsScalar(v[k]) > repla) {
      /* See if this one will work */
      repl  = col[j[k]];
      MatGetRow( mat, row[repl], &nnz, &jj, &vv );
      for (kk=0; kk<nnz; kk++) {
	if (col[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > atol) {
	  *rcv = PetscAbsScalar(v[k]);
	  *rc  = repl;
          MatRestoreRow( mat, row[repl], &nnz, &jj, &vv );
          MatRestoreRow( mat, row[prow], &nz, &j, &v );
	  return 1;
	}
      }
      MatRestoreRow( mat, row[repl], &nnz, &jj, &vv );
    }
  }
  MatRestoreRow( mat, row[prow], &nz, &j, &v );
  return 0;
}

/*@
    MatReorderForNonzeroDiagonal - Changes matrix ordering to remove
        zeros from diagonal. This may help in the LU factorization to 
        prevent a zero pivot.

    Input Parameters:
.   mat  - matrix to reorder
.   rmap,cmap - row and column permutations.  Usually obtained from 
.               MatGetReordering().

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure. It is only a stop-gap measure.

    Algorithm:
    Column pivoting is used.  Choice of column is made by looking at the
    non-zero elements in the row.  This algorithm is simple and fast but
    does NOT guarentee that a non-singular or well conditioned
    principle submatrix will be produced.
@*/
int MatReorderForNonzeroDiagonal(Mat mat,double atol,IS ris,IS cis )
{
  int      ierr, prow, k, nz, n, repl, *j, *col, *row, m;
  Scalar   *v;
  double   repla;

  ierr = ISGetIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISGetIndices(cis,&col); CHKERRQ(ierr);
  ierr = MatGetSize(mat,&m,&n); CHKERRQ(ierr);

  for (prow=0; prow<n; prow++) {
    MatGetRow( mat, row[prow], &nz, &j, &v );
    for (k=0; k<nz; k++) {if (col[j[k]] == prow) break;}
    if (k >= nz || PetscAbsScalar(v[k]) <= atol) {
      /* Element too small or zero; find the best candidate */
      repl  = prow;
      repla = (k >= nz) ? 0.0 : PetscAbsScalar(v[k]);
      for (k=0; k<nz; k++) {
	if (col[j[k]] > prow && PetscAbsScalar(v[k]) > repla) {
	  repl = col[j[k]];
	  repla = PetscAbsScalar(v[k]);
        }
      }
      if (prow == repl) {
	    /* Now we need to look for an element that allows us
	       to pivot with a previous column.  To do this, we need
	       to be sure that we don't introduce a zero in a previous
	       diagonal */
        if (!MatZeroFindPre_Private(mat,prow,row,col,repla,atol,&repl,&repla)){
	  SETERRQ(1,"MatReorderForNonzeroDiagonal:Can not reorder matrix");
	}
      }
      SWAP(col[prow],col[repl]); 
    }
    MatRestoreRow( mat, row[prow], &nz, &j, &v );
  }
  ierr = ISRestoreIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISRestoreIndices(cis,&col); CHKERRQ(ierr);
  return 0;
}



