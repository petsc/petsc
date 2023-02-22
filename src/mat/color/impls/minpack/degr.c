
/* degr.f -- translated by f2c (version of 25 March 1992  12:58:56). */

#include <../src/mat/color/impls/minpack/color.h>

PetscErrorCode MINPACKdegr(PetscInt *n, const PetscInt *indrow, const PetscInt *jpntr, const PetscInt *indcol, const PetscInt *ipntr, PetscInt *ndeg, PetscInt *iwa)
{
  /* System generated locals */
  PetscInt i__1, i__2, i__3;

  /* Local variables */
  PetscInt jcol, ic, ip, jp, ir;

  /*     subroutine degr */
  /*     Given the sparsity pattern of an m by n matrix A, */
  /*     this subroutine determines the degree sequence for */
  /*     the intersection graph of the columns of A. */
  /*     In graph-theory terminology, the intersection graph of */
  /*     the columns of A is the loopless graph G with vertices */
  /*     a(j), j = 1,2,...,n where a(j) is the j-th column of A */
  /*     and with edge (a(i),a(j)) if and only if columns i and j */
  /*     have a non-zero in the same row position. */
  /*     Note that the value of m is not needed by degr and is */
  /*     therefore not present in the subroutine statement. */
  /*     The subroutine statement is */
  /*       subroutine degr(n,indrow,jpntr,indcol,ipntr,ndeg,iwa) */
  /*     where */
  /*       n is a positive integer input variable set to the number */
  /*         of columns of A. */
  /*       indrow is an integer input array which contains the row */
  /*         indices for the non-zeroes in the matrix A. */
  /*       jpntr is an integer input array of length n + 1 which */
  /*         specifies the locations of the row indices in indrow. */
  /*         The row indices for column j are */
  /*               indrow(k), k = jpntr(j),...,jpntr(j+1)-1. */
  /*         Note that jpntr(n+1)-1 is then the number of non-zero */
  /*         elements of the matrix A. */
  /*       indcol is an integer input array which contains the */
  /*         column indices for the non-zeroes in the matrix A. */
  /*       ipntr is an integer input array of length m + 1 which */
  /*         specifies the locations of the column indices in indcol. */
  /*         The column indices for row i are */
  /*               indcol(k), k = ipntr(i),...,ipntr(i+1)-1. */
  /*         Note that ipntr(m+1)-1 is then the number of non-zero */
  /*         elements of the matrix A. */
  /*       ndeg is an integer output array of length n which */
  /*         specifies the degree sequence. The degree of the */
  /*         j-th column of A is ndeg(j). */
  /*       iwa is an integer work array of length n. */
  /*     Argonne National Laboratory. MINPACK Project. July 1983. */
  /*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

  PetscFunctionBegin;
  /* Parameter adjustments */
  --iwa;
  --ndeg;
  --ipntr;
  --indcol;
  --jpntr;
  --indrow;

  /* Function Body */
  i__1 = *n;
  for (jp = 1; jp <= i__1; ++jp) {
    ndeg[jp] = 0;
    iwa[jp]  = 0;
  }

  /*     Compute the degree sequence by determining the contributions */
  /*     to the degrees from the current(jcol) column and further */
  /*     columns which have not yet been considered. */

  i__1 = *n;
  for (jcol = 2; jcol <= i__1; ++jcol) {
    iwa[jcol] = *n;

    /*        Determine all positions (ir,jcol) which correspond */
    /*        to non-zeroes in the matrix. */

    i__2 = jpntr[jcol + 1] - 1;
    for (jp = jpntr[jcol]; jp <= i__2; ++jp) {
      ir = indrow[jp];

      /*           For each row ir, determine all positions (ir,ic) */
      /*           which correspond to non-zeroes in the matrix. */

      i__3 = ipntr[ir + 1] - 1;
      for (ip = ipntr[ir]; ip <= i__3; ++ip) {
        ic = indcol[ip];

        /*              Array iwa marks columns which have contributed to */
        /*              the degree count of column jcol. Update the degree */
        /*              counts of these columns as well as column jcol. */

        if (iwa[ic] < jcol) {
          iwa[ic] = jcol;
          ++ndeg[ic];
          ++ndeg[jcol];
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
