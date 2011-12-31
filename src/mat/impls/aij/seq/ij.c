
#include <../src/mat/impls/aij/seq/aij.h>

#undef __FUNCT__  
#define __FUNCT__ "MatToSymmetricIJ_SeqAIJ"
/*
  MatToSymmetricIJ_SeqAIJ - Convert a (generally nonsymmetric) sparse AIJ matrix
           to IJ format (ignore the "A" part) Allocates the space needed. Uses only 
           the lower triangular part of the matrix.

    Description:
    Take the data in the row-oriented sparse storage and build the
    IJ data for the Matrix.  Return 0 on success,row + 1 on failure
    at that row. Produces the ij for a symmetric matrix by only using
    the lower triangular part of the matrix.

    Input Parameters:
.   Matrix - matrix to convert
.   shiftin - the shift for the original matrix (0 or 1)
.   shiftout - the shift required for the ordering routine (0 or 1)

    Output Parameters:
.   ia     - ia part of IJ representation (row information)
.   ja     - ja part (column indices)

    Notes:
    Both ia and ja may be freed with PetscFree();
    This routine is provided for ordering routines that require a 
    symmetric structure.  It is required since those routines call 
    SparsePak routines that expect a symmetric  matrix.
*/
PetscErrorCode MatToSymmetricIJ_SeqAIJ(PetscInt m,PetscInt *ai,PetscInt *aj,PetscInt shiftin,PetscInt shiftout,PetscInt **iia,PetscInt **jja)
{
  PetscErrorCode ierr;
  PetscInt       *work,*ia,*ja,*j,i,nz,row,col;

  PetscFunctionBegin;
  /* allocate space for row pointers */
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&ia);CHKERRQ(ierr);
  *iia = ia;
  ierr = PetscMemzero(ia,(m+1)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = PetscMalloc((m+1)*sizeof(PetscInt),&work);CHKERRQ(ierr);

  /* determine the number of columns in each row */
  ia[0] = shiftout;
  for (row = 0; row < m; row++) {
    nz = ai[row+1] - ai[row];
    j  = aj + ai[row] + shiftin;
    while (nz--) {
       col = *j++ + shiftin;
       if (col > row) { break;}
       if (col != row) ia[row+1]++;
       ia[col+1]++;
    }
  }

  /* shiftin ia[i] to point to next row */
  for (i=1; i<m+1; i++) {
    row       = ia[i-1];
    ia[i]     += row;
    work[i-1] = row - shiftout;
  }

  /* allocate space for column pointers */
  nz   = ia[m] + (!shiftin);
  ierr = PetscMalloc(nz*sizeof(PetscInt),&ja);CHKERRQ(ierr);
  *jja = ja;

  /* loop over lower triangular part putting into ja */ 
  for (row = 0; row < m; row++) {
    nz = ai[row+1] - ai[row];
    j  = aj + ai[row] + shiftin;
    while (nz--) {
      col = *j++ + shiftin;
      if (col > row) { break;}
      if (col != row) {ja[work[col]++] = row + shiftout; }
      ja[work[row]++] = col + shiftout;
    }
  }
  ierr = PetscFree(work);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}



