
/* setr.f -- translated by f2c (version of 25 March 1992  12:58:56). */

#include <../src/mat/color/color.h>

#undef __FUNCT__  
#define __FUNCT__ "MINPACKsetr" 
PetscErrorCode MINPACKsetr(PetscInt*m,PetscInt* n,PetscInt* indrow,PetscInt* jpntr,PetscInt* indcol, PetscInt*ipntr,PetscInt* iwa)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt jcol, jp, ir;

/*     Given a column-oriented definition of the sparsity pattern */
/*     of an m by n matrix A, this subroutine determines a */
/*     row-oriented definition of the sparsity pattern of A. */
/*     On input the column-oriented definition is specified by */
/*     the arrays indrow and jpntr. On output the row-oriented */
/*     definition is specified by the arrays indcol and ipntr. */
/*     The subroutine statement is */
/*       subroutine setr(m,n,indrow,jpntr,indcol,ipntr,iwa) */
/*     where */
/*       m is a positive integer input variable set to the number */
/*         of rows of A. */
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
/*       indcol is an integer output array which contains the */
/*         column indices for the non-zeroes in the matrix A. */
/*       ipntr is an integer output array of length m + 1 which */
/*         specifies the locations of the column indices in indcol. */
/*         The column indices for row i are */
/*               indcol(k), k = ipntr(i),...,ipntr(i+1)-1. */
/*         Note that ipntr(1) is set to 1 and that ipntr(m+1)-1 is */
/*         then the number of non-zero elements of the matrix A. */
/*       iwa is an integer work array of length m. */
/*     Argonne National Laboratory. MINPACK Project. July 1983. */
/*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

    /*     Store in array iwa the counts of non-zeroes in the rows. */

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iwa;
    --ipntr;
    --indcol;
    --jpntr;
    --indrow;

    /* Function Body */
    i__1 = *m;
    for (ir = 1; ir <= i__1; ++ir) {
	iwa[ir] = 0;
    }
    i__1 = jpntr[*n + 1] - 1;
    for (jp = 1; jp <= i__1; ++jp) {
	++iwa[indrow[jp]];
    }

    /*     Set pointers to the start of the rows in indcol. */

    ipntr[1] = 1;
    i__1 = *m;
    for (ir = 1; ir <= i__1; ++ir) {
	ipntr[ir + 1] = ipntr[ir] + iwa[ir];
	iwa[ir] = ipntr[ir];
    }

    /*     Fill indcol. */

    i__1 = *n;
    for (jcol = 1; jcol <= i__1; ++jcol) {
	i__2 = jpntr[jcol + 1] - 1;
	for (jp = jpntr[jcol]; jp <= i__2; ++jp) {
	    ir = indrow[jp];
	    indcol[iwa[ir]] = jcol;
	    ++iwa[ir];
	}
    }
    PetscFunctionReturn(0);
}

