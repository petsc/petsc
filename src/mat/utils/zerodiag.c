
/*
    This file contains routines to reorder a matrix so that the diagonal
    elements are nonzero.
 */

#include <petsc-private/matimpl.h>       /*I  "petscmat.h"  I*/

#define SWAP(a,b) {PetscInt _t; _t = a; a = b; b = _t; }

#undef __FUNCT__  
#define __FUNCT__ "MatReorderForNonzeroDiagonal"
/*@
    MatReorderForNonzeroDiagonal - Changes matrix ordering to remove
    zeros from diagonal. This may help in the LU factorization to 
    prevent a zero pivot.

    Collective on Mat

    Input Parameters:
+   mat  - matrix to reorder
-   rmap,cmap - row and column permutations.  Usually obtained from 
               MatGetOrdering().

    Level: intermediate

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure. It is only a stop-gap measure. Should be called
    after a call to MatGetOrdering(), this routine changes the column 
    ordering defined in cis.

    Only works for SeqAIJ matrices

    Options Database Keys (When using KSP):
.      -pc_factor_nonzeros_along_diagonal

    Algorithm Notes:
    Column pivoting is used. 

    1) Choice of column is made by looking at the
       non-zero elements in the troublesome row for columns that are not yet 
       included (moving from left to right).
 
    2) If (1) fails we check all the columns to the left of the current row
       and see if one of them has could be swapped. It can be swapped if
       its corresponding row has a non-zero in the column it is being 
       swapped with; to make sure the previous nonzero diagonal remains 
       nonzero


@*/
PetscErrorCode  MatReorderForNonzeroDiagonal(Mat mat,PetscReal abstol,IS ris,IS cis)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(mat,"MatReorderForNonzeroDiagonal_C",(Mat,PetscReal,IS,IS),(mat,abstol,ris,cis));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

extern PetscErrorCode MatGetRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);
extern PetscErrorCode MatRestoreRow_SeqAIJ(Mat,PetscInt,PetscInt*,PetscInt**,PetscScalar**);

#include <../src/vec/is/impls/general/general.h>

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatReorderForNonzeroDiagonal_SeqAIJ"
PetscErrorCode  MatReorderForNonzeroDiagonal_SeqAIJ(Mat mat,PetscReal abstol,IS ris,IS cis)
{
  PetscErrorCode ierr;
  PetscInt       prow,k,nz,n,repl,*j,*col,*row,m,*icol,nnz,*jj,kk;
  PetscScalar    *v,*vv;
  PetscReal      repla;
  IS             icis;

  PetscFunctionBegin;
  /* access the indices of the IS directly, because it changes them */
  row  = ((IS_General*)ris->data)->idx;
  col  = ((IS_General*)cis->data)->idx;
  ierr = ISInvertPermutation(cis,PETSC_DECIDE,&icis);CHKERRQ(ierr);
  icol  = ((IS_General*)icis->data)->idx;
  ierr = MatGetSize(mat,&m,&n);CHKERRQ(ierr);

  for (prow=0; prow<n; prow++) {
    ierr = MatGetRow_SeqAIJ(mat,row[prow],&nz,&j,&v);CHKERRQ(ierr);
    for (k=0; k<nz; k++) {if (icol[j[k]] == prow) break;}
    if (k >= nz || PetscAbsScalar(v[k]) <= abstol) {
      /* Element too small or zero; find the best candidate */
      repla = (k >= nz) ? 0.0 : PetscAbsScalar(v[k]);
      /*
          Look for a later column we can swap with this one
      */
      for (k=0; k<nz; k++) {
	if (icol[j[k]] > prow && PetscAbsScalar(v[k]) > repla) {
          /* found a suitable later column */
	  repl  = icol[j[k]];   
          SWAP(icol[col[prow]],icol[col[repl]]); 
          SWAP(col[prow],col[repl]); 
          goto found;
        }
      }
      /* 
           Did not find a suitable later column so look for an earlier column
	   We need to be sure that we don't introduce a zero in a previous
	   diagonal 
      */
      for (k=0; k<nz; k++) {
        if (icol[j[k]] < prow && PetscAbsScalar(v[k]) > repla) {
          /* See if this one will work */
          repl  = icol[j[k]];
          ierr = MatGetRow_SeqAIJ(mat,row[repl],&nnz,&jj,&vv);CHKERRQ(ierr);
          for (kk=0; kk<nnz; kk++) {
            if (icol[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > abstol) {
              ierr = MatRestoreRow_SeqAIJ(mat,row[repl],&nnz,&jj,&vv);CHKERRQ(ierr);
              SWAP(icol[col[prow]],icol[col[repl]]); 
              SWAP(col[prow],col[repl]); 
              goto found;
	    }
          }
          ierr = MatRestoreRow_SeqAIJ(mat,row[repl],&nnz,&jj,&vv);CHKERRQ(ierr);
        }
      }
      /* 
          No column  suitable; instead check all future rows 
          Note: this will be very slow 
      */
      for (k=prow+1; k<n; k++) {
        ierr = MatGetRow_SeqAIJ(mat,row[k],&nnz,&jj,&vv);CHKERRQ(ierr);
        for (kk=0; kk<nnz; kk++) {
          if (icol[jj[kk]] == prow && PetscAbsScalar(vv[kk]) > abstol) {
            /* found a row */
            SWAP(row[prow],row[k]);
            goto found;
          }
        }
        ierr = MatRestoreRow_SeqAIJ(mat,row[k],&nnz,&jj,&vv);CHKERRQ(ierr);
      }

      found:;
    }
    ierr = MatRestoreRow_SeqAIJ(mat,row[prow],&nz,&j,&v);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&icis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END


