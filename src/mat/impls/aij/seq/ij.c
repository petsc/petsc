#ifndef lint
static char vcid[] = "$Id: ij.c,v 1.5 1995/03/23 22:01:19 curfman Exp curfman $";
#endif

#include "aij.h"

/*
  SpToSymmetricIJ_AIJ - Convert a sparse AIJ matrix to IJ format 
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
    This routine is provided for ordering routines that require a 
    symmetric structure.  It is used in SpOrder (and derivatives) since
    those routines call SparsePak routines that expect a symmetric 
    matrix.
*/
int SpToSymmetricIJ_AIJ( Mat_AIJ *Matrix, int **iia, int **jja )
{
  int          *work,*ia,*ja,*j,i, nz, n, row, wr;
  register int col;

  n  = Matrix->m;

  /* allocate space for row pointers */
  *iia = ia = (int *) MALLOC( (n+1)*sizeof(int) ); CHKPTR(ia);
  MEMSET(ia,0,(n+1)*sizeof(int));
  work = (int *) MALLOC( (n+1)*sizeof(int) ); CHKPTR(work);

  /* determine the number of columns in each row */
  ia[0] = 1;
  for (row = 0; row < n; row++) {
    nz = Matrix->i[row+1] - Matrix->i[row];
    j  = Matrix->j + Matrix->i[row] - 1;
    while (nz--) {
       col = *j++ - 1;
       if ( col > row ) {
          break;
       }
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
 nz = ia[n];
 *jja = ja = (int *) MALLOC( nz*sizeof(int) ); CHKPTR(ja);

 /* loop over lower triangular part putting into ja */ 
 for (row = 0; row < n; row++) {
    nz = Matrix->i[row+1] - Matrix->i[row];
    j  = Matrix->j + Matrix->i[row] - 1;
    while (nz--) {
       col = *j++ - 1;
       if ( col > row ) {
          break;
       }
       if (col != row) {wr = work[col]; work[col] = wr + 1;
			ja[wr] = row + 1; }
       wr = work[row]; work[row] = wr + 1;
       ja[wr] = col + 1;
    }

  }
  FREE(work);
  return 0;
}

