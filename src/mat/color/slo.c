#define PETSCMAT_DLL

/* slo.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

#include "../src/mat/color/color.h"

#undef __FUNCT__  
#define __FUNCT__ "MINPACKslo" 
PetscErrorCode MINPACKslo(PetscInt *n,PetscInt * indrow,PetscInt * jpntr,PetscInt * indcol, PetscInt *ipntr, PetscInt *ndeg,PetscInt * list,
                          PetscInt * maxclq,PetscInt *iwa1,PetscInt * iwa2,PetscInt * iwa3,PetscInt * iwa4)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    PetscInt jcol, ic, ip, jp, ir, mindeg, numdeg, numord;

/*     Given the sparsity pattern of an m by n matrix A, this */
/*     subroutine determines the smallest-last ordering of the */
/*     columns of A. */
/*     The smallest-last ordering is defined for the loopless */
/*     graph G with vertices a(j), j = 1,2,...,n where a(j) is the */
/*     j-th column of A and with edge (a(i),a(j)) if and only if */
/*     columns i and j have a non-zero in the same row position. */
/*     The smallest-last ordering is determined recursively by */
/*     letting list(k), k = n,...,1 be a column with least degree */
/*     in the subgraph spanned by the un-ordered columns. */
/*     Note that the value of m is not needed by slo and is */
/*     therefore not present in the subroutine statement. */
/*     The subroutine statement is */
/*       subroutine slo(n,indrow,jpntr,indcol,ipntr,ndeg,list, */
/*                      maxclq,iwa1,iwa2,iwa3,iwa4) */
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
/*       ndeg is an integer input array of length n which specifies */
/*         the degree sequence. The degree of the j-th column */
/*         of A is ndeg(j). */
/*       list is an integer output array of length n which specifies */
/*         the smallest-last ordering of the columns of A. The j-th */
/*         column in this order is list(j). */
/*       maxclq is an integer output variable set to the size */
/*         of the largest clique found during the ordering. */
/*       iwa1,iwa2,iwa3, and iwa4 are integer work arrays of length n. */
/*     Subprograms called */
/*       FORTRAN-supplied ... min */
/*     Argonne National Laboratory. MINPACK Project. August 1984. */
/*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iwa4;
    --iwa3;
    --iwa2;
    --list;
    --ndeg;
    --ipntr;
    --indcol;
    --jpntr;
    --indrow;

    /* Function Body */
    mindeg = *n;
    i__1 = *n;
    for (jp = 1; jp <= i__1; ++jp) {
	iwa1[jp - 1] = 0;
	iwa4[jp] = *n;
	list[jp] = ndeg[jp];
        /* Computing MIN */
	i__2 = mindeg, i__3 = ndeg[jp];
	mindeg = PetscMin(i__2,i__3);
    }

    /*     Create a doubly-linked list to access the degrees of the */
    /*     columns. The pointers for the linked list are as follows. */

    /*     Each un-ordered column ic is in a list (the degree list) */
    /*     of columns with the same degree. */

    /*     iwa1(numdeg) is the first column in the numdeg list */
    /*     unless iwa1(numdeg) = 0. In this case there are */
    /*     no columns in the numdeg list. */

    /*     iwa2(ic) is the column before ic in the degree list */
    /*     unless iwa2(ic) = 0. In this case ic is the first */
    /*     column in this degree list. */

    /*     iwa3(ic) is the column after ic in the degree list */
    /*     unless iwa3(ic) = 0. In this case ic is the last */
    /*     column in this degree list. */

    /*     If ic is an un-ordered column, then list(ic) is the */
    /*     degree of ic in the graph induced by the un-ordered */
    /*     columns. If jcol is an ordered column, then list(jcol) */
    /*     is the smallest-last order of column jcol. */

    i__1 = *n;
    for (jp = 1; jp <= i__1; ++jp) {
	numdeg = ndeg[jp];
	iwa2[jp] = 0;
	iwa3[jp] = iwa1[numdeg];
	if (iwa1[numdeg] > 0) {
	    iwa2[iwa1[numdeg]] = jp;
	}
	iwa1[numdeg] = jp;
    }
    *maxclq = 0;
    numord = *n;

    /*     Beginning of iteration loop. */

L30:

    /*        Choose a column jcol of minimal degree mindeg. */

L40:
    jcol = iwa1[mindeg];
    if (jcol > 0) {
	goto L50;
    }
    ++mindeg;
    goto L40;
L50:
    list[jcol] = numord;

    /*        Mark the size of the largest clique */
    /*        found during the ordering. */

    if (mindeg + 1 == numord && !*maxclq) {
	*maxclq = numord;
    }

    /*        Termination test. */

    --numord;
    if (!numord) {
	goto L80;
    }

    /*        Delete column jcol from the mindeg list. */

    iwa1[mindeg] = iwa3[jcol];
    if (iwa3[jcol] > 0) {
	iwa2[iwa3[jcol]] = 0;
    }

    /*        Find all columns adjacent to column jcol. */

    iwa4[jcol] = 0;

    /*        Determine all positions (ir,jcol) which correspond */
    /*        to non-zeroes in the matrix. */

    i__1 = jpntr[jcol + 1] - 1;
    for (jp = jpntr[jcol]; jp <= i__1; ++jp) {
	ir = indrow[jp];

        /*           For each row ir, determine all positions (ir,ic) */
        /*           which correspond to non-zeroes in the matrix. */

	i__2 = ipntr[ir + 1] - 1;
	for (ip = ipntr[ir]; ip <= i__2; ++ip) {
	    ic = indcol[ip];

            /*              Array iwa4 marks columns which are adjacent to */
            /*              column jcol. */

	    if (iwa4[ic] > numord) {
		iwa4[ic] = numord;

                /*                 Update the pointers to the current degree lists. */

		numdeg = list[ic];
		--list[ic];
                /* Computing MIN */
		i__3 = mindeg, i__4 = list[ic];
		mindeg = PetscMin(i__3,i__4);

                /*                 Delete column ic from the numdeg list. */

		if (!iwa2[ic]) {
		    iwa1[numdeg] = iwa3[ic];
		} else {
		    iwa3[iwa2[ic]] = iwa3[ic];
		}
		if (iwa3[ic] > 0) {
		    iwa2[iwa3[ic]] = iwa2[ic];
		}

                /*                 Add column ic to the numdeg-1 list. */

		iwa2[ic] = 0;
		iwa3[ic] = iwa1[numdeg - 1];
		if (iwa1[numdeg - 1] > 0) {
		    iwa2[iwa1[numdeg - 1]] = ic;
		}
		iwa1[numdeg - 1] = ic;
	    }
	}
    }

    /*        End of iteration loop. */

    goto L30;
L80:

    /*     Invert the array list. */

    i__1 = *n;
    for (jcol = 1; jcol <= i__1; ++jcol) {
	iwa2[list[jcol]] = jcol;
    }
    i__1 = *n;
    for (jp = 1; jp <= i__1; ++jp) {
	list[jp] = iwa2[jp];
    }
    PetscFunctionReturn(0);
}

