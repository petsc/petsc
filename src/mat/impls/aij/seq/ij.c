#ifndef lint
static char vcid[] = "$Id: ij.c,v 1.10 1995/09/21 20:10:16 bsmith Exp bsmith $";
#endif

#include "aij.h"

/*
  MatToSymmetricIJ_SeqAIJ - Convert a sparse AIJ matrix to IJ format 
           (ignore the "A" part)
           Allocates the space needed. Uses only the lower triangular 
           part of the matrix.

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
$    Both ia and ja maybe freed with PETSCFREE();
$    This routine is provided for ordering routines that require a 
$    symmetric structure.  It is used in SpOrder (and derivatives) since
$    those routines call SparsePak routines that expect a symmetric 
$    matrix.
*/
int MatToSymmetricIJ_SeqAIJ( Mat_SeqAIJ *A, int **iia, int **jja )
{
  int *work,*ia,*ja,*j,i, nz, n = A->m, row, wr, col, shift = A->indexshift;

  /* allocate space for row pointers */
  *iia = ia = (int *) PETSCMALLOC( (n+1)*sizeof(int) ); CHKPTRQ(ia);
  PetscZero(ia,(n+1)*sizeof(int));
  work = (int *) PETSCMALLOC( (n+1)*sizeof(int) ); CHKPTRQ(work);

  /* determine the number of columns in each row */
  ia[0] = 1;
  for (row = 0; row < n; row++) {
    nz = A->i[row+1] - A->i[row];
    j  = A->j + A->i[row] + shift;
    while (nz--) {
       col = *j++ + shift;
       if ( col > row ) { break;}
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
  *jja = ja = (int *) PETSCMALLOC( nz*sizeof(int) ); CHKPTRQ(ja);

  /* loop over lower triangular part putting into ja */ 
  for (row = 0; row < n; row++) {
    nz = A->i[row+1] - A->i[row];
    j  = A->j + A->i[row] + shift;
    while (nz--) {
       col = *j++ + shift;
       if ( col > row ) { break;}
       if (col != row) {wr = work[col]; work[col] = wr + 1; ja[wr] = row + 1; }
       wr = work[row]; work[row] = wr + 1;
       ja[wr] = col + 1;
    }
  }
  PETSCFREE(work);
  return 0;
}

