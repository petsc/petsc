#ifndef lint
static char vcid[] = "$Id: ij.c,v 1.14 1995/11/02 04:28:29 bsmith Exp bsmith $";
#endif

#include "aij.h"

/*
  MatToSymmetricIJ_SeqAIJ - Convert a sparse AIJ matrix to IJ format 
           (ignore the "A" part) Allocates the space needed. Uses only 
           the lower triangular part of the matrix.

    Description:
    Take the data in the row-oriented sparse storage and build the
    IJ data for the Matrix.  Return 0 on success, row + 1 on failure
    at that row. Produces the ij for a symmetric matrix by only using
    the lower triangular part of the matrix.

    Input Parameters:
.   Matrix - matrix to convert

    Output Parameters:
.   ia     - ia part of IJ representation (row information)
.   ja     - ja part (column indices)

    Notes:
$    Both ia and ja may be freed with PetscFree();
$    This routine is provided for ordering routines that require a 
$    symmetric structure.  It is used in SpOrder (and derivatives) since
$    those routines call SparsePak routines that expect a symmetric 
$    matrix.
*/
int MatToSymmetricIJ_SeqAIJ( Mat_SeqAIJ *A, int **iia, int **jja )
{
  int *work,*ia,*ja,*j,i, nz, n = A->m, row, col, shift = A->indexshift;
  int *ai = A->i, *aj = A->j + shift;

  /* allocate space for row pointers */
  *iia = ia = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(ia);
  PetscMemzero(ia,(n+1)*sizeof(int));
  work = (int *) PetscMalloc( (n+1)*sizeof(int) ); CHKPTRQ(work);

  /* determine the number of columns in each row */
  ia[0] = 1;
  for (row = 0; row < n; row++) {
    nz = ai[row+1] - ai[row];
    j  = aj + ai[row];
    while (nz--) {
       col = *j++ + shift;
       if (col > row) { break;}
       if (col != row) ia[row+1]++;
       ia[col+1]++;
    }
  }

  /* shift ia[i] to point to next row */
  for ( i=1; i<n+1; i++ ) {
    row       = ia[i-1];
    ia[i]     += row;
    work[i-1] = row - 1;
  }

  /* allocate space for column pointers */
  nz = ia[n] + (!shift);
  *jja = ja = (int *) PetscMalloc( nz*sizeof(int) ); CHKPTRQ(ja);

  /* loop over lower triangular part putting into ja */ 
  for (row = 0; row < n; row++) {
    nz = ai[row+1] - ai[row];
    j  = aj + ai[row];
    while (nz--) {
      col = *j++ + shift;
      if (col > row) { break;}
      if (col != row) {ja[work[col]++] = row + 1; }
      ja[work[row]++] = col + 1;
    }
  }
  PetscFree(work);
  return 0;
}



