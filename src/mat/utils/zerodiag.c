#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zerodiag.c,v 1.25 1998/11/04 16:23:24 bsmith Exp bsmith $";
#endif

/*
    This file contains routines to reorder a matrix so that the diagonal
    elements are nonzero.
 */

#include "src/mat/matimpl.h"       /*I  "mat.h"  I*/
#include <math.h>

#define SWAP(a,b) {int _t; _t = a; a = b; b = _t; }

#undef __FUNC__  
#define __FUNC__ "MatZeroFindPre_Private"
/* 
   Given a current row and current permutation, find a column permutation
   that removes a zero diagonal.
*/
int MatZeroFindPre_Private(Mat mat,int prow,int* row,int* col,int *icol,double repla,
                           double atol,int* rc,double* rcv,int nz, int *j, Scalar *v)
{
  int      k, repl, kk, nnz, *jj,ierr;
  Scalar   *vv;

  PetscFunctionBegin;
   /*
      Here one could sort the col[j[k]] to try to select the column closest
     to the diagonal (in the new ordering) that satisfies the criteria
  */
  for (k=0; k<nz; k++) {
    if (icol[j[k]] < prow && PetscAbsScalar(v[k]) > repla) {
      /* See if this one will work */
      repl  = icol[j[k]];
      ierr = MatGetRow( mat, irow[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
      for (kk=0; kk<nnz; kk++) {
	if (icol[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > atol) {
	  *rcv = PetscAbsScalar(v[k]);
	  *rc  = repl;
          ierr = MatRestoreRow( mat, irow[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
	  PetscFunctionReturn(1);
	}
      }
      ierr = MatRestoreRow( mat, irow[repl], &nnz, &jj, &vv ); CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MatReorderForNonzeroDiagonal"
/*@
    MatReorderForNonzeroDiagonal - Changes matrix ordering to remove
    zeros from diagonal. This may help in the LU factorization to 
    prevent a zero pivot.

    Collective on Mat

    Input Parameters:
+   mat  - matrix to reorder
-   rmap,cmap - row and column permutations.  Usually obtained from 
               MatGetReordering().

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure. It is only a stop-gap measure. Should be called
    after a call to MatGetReordering(), this routine changes the column 
    ordering defined in cis.

    Options Database Keys (When using SLES):
+      -pc_ilu_nonzeros_along_diagonal
-      -pc_lu_nonzeros_along_diagonal

    Algorithm Notes:
    Column pivoting is used. 

    1) Choice of column is made by looking at the
       non-zero elements in the troublesome row for columns that are not yet 
       included (moving from left to right).
 
    2) If (1) fails we check all the columns to the left of the current row
       and see if we can 


@*/
int MatReorderForNonzeroDiagonal(Mat mat,double atol,IS ris,IS cis )
{
  int      ierr, prow, k, nz, n, repl, *j, *col, *row, m, *irow, *icol;
  Scalar   *v;
  double   repla;
  IS       icis,iris;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mat,MAT_COOKIE);
  PetscValidHeaderSpecific(ris,IS_COOKIE);
  PetscValidHeaderSpecific(cis,IS_COOKIE);
  
  ierr = ISGetIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISGetIndices(cis,&col); CHKERRQ(ierr);
  ierr = ISInvertPermutation(cis,&icis);CHKERRQ(ierr);
  ierr = ISInvertPermutation(ris,&iris);CHKERRQ(ierr);
  ierr = ISGetIndices(icis,&icol); CHKERRQ(ierr);
  ierr = MatGetSize(mat,&m,&n); CHKERRQ(ierr);

  for (prow=0; prow<n; prow++) {
    ierr = MatGetRow( mat, row[prow], &nz, &j, &v ); CHKERRQ(ierr);
    for (k=0; k<nz; k++) {if (icol[j[k]] == prow) break;}
    if (k >= nz || PetscAbsScalar(v[k]) <= atol) {
      /* Element too small or zero; find the best candidate */
      repl  = prow;
      repla = (k >= nz) ? 0.0 : PetscAbsScalar(v[k]);
      /*
        Here one could sort the icol[j[k]] list to try to select the 
        column closest to the diagonal in the new ordering. (Note have
        to permute the v[k] values as well, and use a fixed bound on the
        quality of repla rather then looking for the absolute largest.
      */
      for (k=0; k<nz; k++) {
	if (icol[j[k]] > prow && PetscAbsScalar(v[k]) > repla) {
	  repl  = icol[j[k]];
	  repla = PetscAbsScalar(v[k]);
        }
      }
      if (prow == repl) {
	/* 
           Look for an element that allows us
	   to pivot with a previous column.  To do this, we need
	   to be sure that we don't introduce a zero in a previous
	   diagonal 
        */
        if (!MatZeroFindPre_Private(mat,prow,row,col,icol,repla,atol,&repl,&repla,nz,j,v)){
          (*PetscErrorPrintf)("Permuted row number %d\n",prow);
	  SETERRQ(PETSC_ERR_MAT_LU_ZRPVT,0,"Cannot reorder matrix to eliminate zero diagonal entry");
	}
      }
      SWAP(icol[col[prow]],icol[col[repl]]); 
      SWAP(col[prow],col[repl]); 
    }
    ierr = MatRestoreRow( mat, row[prow], &nz, &j, &v ); CHKERRQ(ierr);
  }
  ierr = ISRestoreIndices(ris,&row); CHKERRQ(ierr);
  ierr = ISRestoreIndices(cis,&col); CHKERRQ(ierr);
  ierr = ISRestoreIndices(icis,&icol); CHKERRQ(ierr);
  ierr = ISDestroy(icis); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



