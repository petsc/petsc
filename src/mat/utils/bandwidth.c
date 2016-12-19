#include <petsc/private/matimpl.h>       /*I  "petscmat.h"  I*/

/*@
  MatComputeBandwidth - Calculate the full bandwidth of the matrix, meaning the width 2k+1 where k diagonals on either side are sufficient to contain all the matrix nonzeros.

  Collective on Mat

  Input Parameters:
+ A - The Mat
- fraction - An optional percentage of the Frobenius norm of the matrix that the bandwidth should enclose

  Output Parameter:
. bw - The matrix bandwidth

  Level: beginner

.seealso: DMPlexCreate(), DMPlexSetConeSize(), DMPlexSetChart()
@*/
PetscErrorCode MatComputeBandwidth(Mat A, PetscReal fraction, PetscInt *bw)
{
  PetscInt       lbw[2] = {0, 0}, gbw[2];
  PetscInt       rStart, rEnd, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(A,fraction,2);
  PetscValidPointer(bw, 3);
  if ((fraction > 0.0) && (fraction < 1.0)) SETERRQ(PetscObjectComm((PetscObject) A), PETSC_ERR_SUP, "We do not yet support a fractional bandwidth");
  ierr = MatGetOwnershipRange(A, &rStart, &rEnd);CHKERRQ(ierr);
  for (r = rStart; r < rEnd; ++r) {
    const PetscInt *cols;
    PetscInt        ncols;

    ierr = MatGetRow(A, r, &ncols, &cols, NULL);CHKERRQ(ierr);
    if (ncols) {
      lbw[0] = PetscMax(lbw[0], r - cols[0]);
      lbw[1] = PetscMax(lbw[1], cols[ncols-1] - r);
    }
    ierr = MatRestoreRow(A, r, &ncols, &cols, NULL);CHKERRQ(ierr);
  }
  ierr = MPIU_Allreduce(lbw, gbw, 2, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject) A));CHKERRQ(ierr);
  *bw = 2*PetscMax(gbw[0], gbw[1]) + 1;
  PetscFunctionReturn(0);
}
