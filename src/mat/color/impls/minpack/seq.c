
/* seq.f -- translated by f2c (version of 25 March 1992  12:58:56). */

#include <../src/mat/color/impls/minpack/color.h>

PetscErrorCode MINPACKseq(PetscInt *n, const PetscInt *indrow, const PetscInt *jpntr, const PetscInt *indcol, const PetscInt *ipntr, PetscInt *list, PetscInt *ngrp, PetscInt *maxgrp, PetscInt *iwa)
{
  /* System generated locals */
  PetscInt i__1, i__2, i__3;

  /* Local variables */
  PetscInt jcol, j, ic, ip, jp, ir;

  /*     Given the sparsity pattern of an m by n matrix A, this */
  /*     subroutine determines a consistent partition of the */
  /*     columns of A by a sequential algorithm. */
  /*     A consistent partition is defined in terms of the loopless */
  /*     graph G with vertices a(j), j = 1,2,...,n where a(j) is the */
  /*     j-th column of A and with edge (a(i),a(j)) if and only if */
  /*     columns i and j have a non-zero in the same row position. */
  /*     A partition of the columns of A into groups is consistent */
  /*     if the columns in any group are not adjacent in the graph G. */
  /*     In graph-theory terminology, a consistent partition of the */
  /*     columns of A corresponds to a coloring of the graph G. */
  /*     The subroutine examines the columns in the order specified */
  /*     by the array list, and assigns the current column to the */
  /*     group with the smallest possible number. */
  /*     Note that the value of m is not needed by seq and is */
  /*     therefore not present in the subroutine statement. */
  /*     The subroutine statement is */
  /*       subroutine seq(n,indrow,jpntr,indcol,ipntr,list,ngrp,maxgrp, */
  /*                      iwa) */
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
  /*       list is an integer input array of length n which specifies */
  /*         the order to be used by the sequential algorithm. */
  /*         The j-th column in this order is list(j). */
  /*       ngrp is an integer output array of length n which specifies */
  /*         the partition of the columns of A. Column jcol belongs */
  /*         to group ngrp(jcol). */
  /*       maxgrp is an integer output variable which specifies the */
  /*         number of groups in the partition of the columns of A. */
  /*       iwa is an integer work array of length n. */
  /*     Argonne National Laboratory. MINPACK Project. July 1983. */
  /*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

  PetscFunctionBegin;
  /* Parameter adjustments */
  --iwa;
  --ngrp;
  --list;
  --ipntr;
  --indcol;
  --jpntr;
  --indrow;

  /* Function Body */
  *maxgrp = 0;
  i__1    = *n;
  for (jp = 1; jp <= i__1; ++jp) {
    ngrp[jp] = *n;
    iwa[jp]  = 0;
  }

  /*     Beginning of iteration loop. */

  i__1 = *n;
  for (j = 1; j <= i__1; ++j) {
    jcol = list[j];

    /*        Find all columns adjacent to column jcol. */

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

        /*              Array iwa marks the group numbers of the */
        /*              columns which are adjacent to column jcol. */

        iwa[ngrp[ic]] = j;
      }
    }

    /*        Assign the smallest un-marked group number to jcol. */

    i__2 = *maxgrp;
    for (jp = 1; jp <= i__2; ++jp) {
      if (iwa[jp] != j) goto L50;
    }
    ++(*maxgrp);
  L50:
    ngrp[jcol] = jp;
  }

  /*        End of iteration loop. */
  PetscFunctionReturn(PETSC_SUCCESS);
}
