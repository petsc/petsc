/* ido.f -- translated by f2c (version of 25 March 1992  12:58:56).*/

#include <../src/mat/color/color.h>

static PetscInt c_n1 = -1;

#undef __FUNCT__
#define __FUNCT__ "MINPACKido"
PetscErrorCode MINPACKido(PetscInt *m,PetscInt * n,const PetscInt * indrow,const PetscInt * jpntr,const PetscInt * indcol,const PetscInt * ipntr,PetscInt * ndeg,
               PetscInt *list,PetscInt *maxclq, PetscInt *iwa1, PetscInt *iwa2, PetscInt *iwa3, PetscInt *iwa4)
{
    /* System generated locals */
    PetscInt i__1, i__2, i__3, i__4;

    /* Local variables */
    PetscInt jcol = 0, ncomp = 0, ic, ip, jp, ir, maxinc, numinc, numord, maxlst, numwgt, numlst;

/*     Given the sparsity pattern of an m by n matrix A, this */
/*     subroutine determines an incidence-degree ordering of the */
/*     columns of A. */
/*     The incidence-degree ordering is defined for the loopless */
/*     graph G with vertices a(j), j = 1,2,...,n where a(j) is the */
/*     j-th column of A and with edge (a(i),a(j)) if and only if */
/*     columns i and j have a non-zero in the same row position. */
/*     The incidence-degree ordering is determined recursively by */
/*     letting list(k), k = 1,...,n be a column with maximal */
/*     incidence to the subgraph spanned by the ordered columns. */
/*     Among all the columns of maximal incidence, ido chooses a */
/*     column of maximal degree. */
/*     The subroutine statement is */
/*       subroutine ido(m,n,indrow,jpntr,indcol,ipntr,ndeg,list, */
/*                      maxclq,iwa1,iwa2,iwa3,iwa4) */
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
/*         the incidence-degree ordering of the columns of A. The j-th */
/*         column in this order is list(j). */
/*       maxclq is an integer output variable set to the size */
/*         of the largest clique found during the ordering. */
/*       iwa1,iwa2,iwa3, and iwa4 are integer work arrays of length n. */
/*     Subprograms called */
/*       MINPACK-supplied ... numsrt */
/*       FORTRAN-supplied ... max */
/*     Argonne National Laboratory. MINPACK Project. August 1984. */
/*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

/*     Sort the degree sequence. */

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
    i__1 = *n - 1;
    MINPACKnumsrt(n, &i__1, &ndeg[1], &c_n1, &iwa4[1], &iwa2[1], &iwa3[1]);

/*     Initialization block. */
/*     Create a doubly-linked list to access the incidences of the */
/*     columns. The pointers for the linked list are as follows. */
/*     Each un-ordered column ic is in a list (the incidence list) */
/*     of columns with the same incidence. */
/*     iwa1(numinc) is the first column in the numinc list */
/*     unless iwa1(numinc) = 0. In this case there are */
/*     no columns in the numinc list. */
/*     iwa2(ic) is the column before ic in the incidence list */
/*     unless iwa2(ic) = 0. In this case ic is the first */
/*     column in this incidence list. */
/*     iwa3(ic) is the column after ic in the incidence list */
/*     unless iwa3(ic) = 0. In this case ic is the last */
/*     column in this incidence list. */
/*     If ic is an un-ordered column, then list(ic) is the */
/*     incidence of ic to the graph induced by the ordered */
/*     columns. If jcol is an ordered column, then list(jcol) */
/*     is the incidence-degree order of column jcol. */

    maxinc = 0;
    for (jp = *n; jp >= 1; --jp) {
	ic = iwa4[jp];
	iwa1[*n - jp] = 0;
	iwa2[ic] = 0;
	iwa3[ic] = iwa1[0];
	if (iwa1[0] > 0) {
	    iwa2[iwa1[0]] = ic;
	}
	iwa1[0] = ic;
	iwa4[jp] = 0;
	list[jp] = 0;
    }

/*     Determine the maximal search length for the list */
/*     of columns of maximal incidence. */

    maxlst = 0;
    i__1 = *m;
    for (ir = 1; ir <= i__1; ++ir) {
/* Computing 2nd power */
	i__2 = ipntr[ir + 1] - ipntr[ir];
	maxlst += i__2 * i__2;
    }
    maxlst /= *n;
    *maxclq = 0;
    numord = 1;

/*     Beginning of iteration loop. */

L30:

/*        Choose a column jcol of maximal degree among the */
/*        columns of maximal incidence maxinc. */

L40:
    jp = iwa1[maxinc];
    if (jp > 0) {
	goto L50;
    }
    --maxinc;
    goto L40;
L50:
    numwgt = -1;
    i__1 = maxlst;
    for (numlst = 1; numlst <= i__1; ++numlst) {
	if (ndeg[jp] > numwgt) {
	    numwgt = ndeg[jp];
	    jcol = jp;
	}
	jp = iwa3[jp];
	if (jp <= 0) {
	    goto L70;
	}
    }
L70:
    list[jcol] = numord;

/*        Update the size of the largest clique */
/*        found during the ordering. */

    if (!maxinc) {
	ncomp = 0;
    }
    ++ncomp;
    if (maxinc + 1 == ncomp) {
	*maxclq = PetscMax(*maxclq,ncomp);
    }

/*        Termination test. */

    ++numord;
    if (numord > *n) {
	goto L100;
    }

/*        Delete column jcol from the maxinc list. */

    if (!iwa2[jcol]) {
	iwa1[maxinc] = iwa3[jcol];
    } else {
	iwa3[iwa2[jcol]] = iwa3[jcol];
    }
    if (iwa3[jcol] > 0) {
	iwa2[iwa3[jcol]] = iwa2[jcol];
    }

/*        Find all columns adjacent to column jcol. */

    iwa4[jcol] = *n;

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

	    if (iwa4[ic] < numord) {
		iwa4[ic] = numord;

/*                 Update the pointers to the current incidence lists. */

		numinc = list[ic];
		++list[ic];
/* Computing MAX */
		i__3 = maxinc, i__4 = list[ic];
		maxinc = PetscMax(i__3,i__4);

/*                 Delete column ic from the numinc list. */

		if (!iwa2[ic]) {
		    iwa1[numinc] = iwa3[ic];
		} else {
		    iwa3[iwa2[ic]] = iwa3[ic];
		}
		if (iwa3[ic] > 0) {
		    iwa2[iwa3[ic]] = iwa2[ic];
		}

/*                 Add column ic to the numinc+1 list. */

		iwa2[ic] = 0;
		iwa3[ic] = iwa1[numinc + 1];
		if (iwa1[numinc + 1] > 0) {
		    iwa2[iwa1[numinc + 1]] = ic;
		}
		iwa1[numinc + 1] = ic;
	    }
	}
    }

/*        End of iteration loop. */

    goto L30;
L100:

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

