/*$Id: ilut.c,v 1.4 1999/12/19 04:22:00 bsmith Exp bsmith $*/
/* ilut.f -- translated by f2c (version of 25 March 1992  12:58:56).

        This code is protected by the GNU copyright. See the file 
     gnu in this directory. See below for the Author.
*/
#include "petsc.h"

static int SPARSEKIT2qsplit(Scalar *a,int *ind,int *n,int *ncut)
{
    /* System generated locals */
    int i__1;
    Scalar d__1;

    /* Local variables */
    int last,itmp,j,first;
    PetscReal abskey;
    int mid;
    Scalar tmp;

/* -----------------------------------------------------------------------
 */
/*     does a quick-sort split of a real array. */
/*     on input a(1:n). is a real array */
/*     on output a(1:n) is permuted such that its elements satisfy: */

/*     abs(a(i)) .ge. abs(a(ncut)) for i .lt. ncut and */
/*     abs(a(i)) .le. abs(a(ncut)) for i .gt. ncut */

/*    ind(1:n) is an integer array which permuted in the same way as a(*).
*/
/* -----------------------------------------------------------------------
 */
/* ----- */
    /* Parameter adjustments */
    --ind;
    --a;

    /* Function Body */
    first = 1;
    last = *n;
    if (*ncut < first || *ncut > last) {
	return 0;
    }

/*     outer loop -- while mid .ne. ncut do */

L1:
    mid = first;
    abskey = (d__1 = a[mid],PetscAbsScalar(d__1));
    i__1 = last;
    for (j = first + 1; j <= i__1; ++j) {
	if ((d__1 = a[j],PetscAbsScalar(d__1)) > abskey) {
	    ++mid;
/*     interchange */
	    tmp = a[mid];
	    itmp = ind[mid];
	    a[mid] = a[j];
	    ind[mid] = ind[j];
	    a[j] = tmp;
	    ind[j] = itmp;
	}
/* L2: */
    }

/*     interchange */

    tmp = a[mid];
    a[mid] = a[first];
    a[first] = tmp;

    itmp = ind[mid];
    ind[mid] = ind[first];
    ind[first] = itmp;

/*     test for while loop */

    if (mid == *ncut) {
	return 0;
    }
    if (mid > *ncut) {
	last = mid - 1;
    } else {
	first = mid + 1;
    }
    goto L1;
/* ----------------end-of-qsplit------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* qsplit_ */


/* ---------------------------------------------------------------------- */
int SPARSEKIT2ilutp(int *n,Scalar *a,int *ja,int * ia,int *lfil,PetscReal *droptol,PetscReal *permtol,int *mbloc,Scalar *alu,
	int *jlu,int *ju,int *iwk,Scalar *w,int *jw,  int *iperm,int *ierr)
{
    /* System generated locals */
    int i__1,i__2;
    Scalar d__1;

    /* Local variables */
    Scalar fact;
    int lenl,imax,lenu,icut,jpos;
    PetscReal xmax;
    int jrow;
    PetscReal xmax0;
    int i,j,k;
    Scalar s,t;
    int j_1,j2;
    PetscReal tnorm,t1;
    int ii,jj;
    int ju0,len;
    Scalar tmp;

/* -----------------------------------------------------------------------
 */
/*     implicit none */
/* ----------------------------------------------------------------------*
 */
/*       *** ILUTP preconditioner -- ILUT with pivoting  ***            * 
*/
/*      incomplete LU factorization with dual truncation mechanism      * 
*/
/* ----------------------------------------------------------------------*
 */
/* author Yousef Saad *Sep 8, 1993 -- Latest revision, August 1996.     * 
*/
/* ----------------------------------------------------------------------*
 */
/* on entry: */
/* ========== */
/* n       = integer. The dimension of the matrix A. */

/* a,ja,ia = matrix stored in Compressed Sparse Row format. */
/*           ON RETURN THE COLUMNS OF A ARE PERMUTED. SEE BELOW FOR */
/*           DETAILS. */

/* lfil    = integer. The fill-in parameter. Each row of L and each row */

/*           of U will have a maximum of lfil elements (excluding the */
/*           diagonal element). lfil must be .ge. 0. */
/*           ** WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO 
*/
/*           EARLIER VERSIONS. */

/* droptol = real*8. Sets the threshold for dropping small terms in the */

/*           factorization. See below for details on dropping strategy. */


/* lfil    = integer. The fill-in parameter. Each row of L and */
/*           each row of U will have a maximum of lfil elements. */
/*           WARNING: THE MEANING OF LFIL HAS CHANGED WITH RESPECT TO */
/*           EARLIER VERSIONS. */
/*           lfil must be .ge. 0. */

/* permtol = tolerance ratio used to  determne whether or not to permute 
*/
/*           two columns.  At step i columns i and j are permuted when */

/*                     abs(a(i,j))*permtol .gt. abs(a(i,i)) */

/*           [0 --> never permute; good values 0.1 to 0.01] */

/* mbloc   = if desired, permuting can be done only within the diagonal */

/*           blocks of size mbloc. Useful for PDE problems with several */

/*           degrees of freedom.. If feature not wanted take mbloc=n. */


/* iwk     = integer. The lengths of arrays alu and jlu. If the arrays */
/*           are not big enough to store the ILU factorizations, ilut */
/*           will stop with an error message. */

/* On return: */
/* =========== */

/* alu,jlu = matrix stored in Modified Sparse Row (MSR) format containing 
*/
/*           the L and U factors together. The diagonal (stored in */
/*           alu(1:n)) is inverted. Each i-th row of the alu,jlu matrix 
*/
/*           contains the i-th row of L (excluding the diagonal entry=1) 
*/
/*           followed by the i-th row of U. */

/* ju      = integer array of length n containing the pointers to */
/*           the beginning of each row of U in the matrix alu,jlu. */

/* iperm   = contains the permutation arrays. */
/*           iperm(1:n) = old numbers of unknowns */
/*           iperm(n+1:2*n) = reverse permutation = new unknowns. */

/* ierr    = integer. Error message with the following meaning. */
/*           ierr  = 0    --> successful return. */
/*           ierr .gt. 0  --> zero pivot encountered at step number ierr. 
*/
/*           ierr  = -1   --> Error. input matrix may be wrong. */
/*                            (The elimination process has generated a */
/*                            row in L or U whose length is .gt.  n.) */
/*           ierr  = -2   --> The matrix L overflows the array al. */
/*           ierr  = -3   --> The matrix U overflows the array alu. */
/*           ierr  = -4   --> Illegal value for lfil. */
/*           ierr  = -5   --> zero row encountered. */

/* work arrays: */
/* ============= */
/* jw      = integer work array of length 2*n. */
/* w       = real work array of length n */

/* IMPORTANR NOTE: */
/* -------------- */
/* TO AVOID PERMUTING THE SOLUTION VECTORS ARRAYS FOR EACH LU-SOLVE, */
/* THE MATRIX A IS PERMUTED ON RETURN. [all column indices are */
/* changed]. SIMILARLY FOR THE U MATRIX. */
/* To permute the matrix back to its original state use the loop: */

/*      do k=ia(1), ia(n+1)-1 */
/*         ja(k) = iperm(ja(k)) */
/*      enddo */

/* -----------------------------------------------------------------------
 */
/*     local variables */


    /* Parameter adjustments */
    --iperm;
    --jw;
    --w;
    --ju;
    --jlu;
    --alu;
    --ia;
    --ja;
    --a;

    /* Function Body */
    if (*lfil < 0) {
	goto L998;
    }
/* -----------------------------------------------------------------------
 */
/*     initialize ju0 (points to next element to be added to alu,jlu) */
/*     and pointer array. */
/* -----------------------------------------------------------------------
 */
    ju0 = *n + 2;
    jlu[1] = ju0;

/*  integer PetscReal pointer array. */

    i__1 = *n;
    for (j = 1; j <= i__1; ++j) {
	jw[*n + j] = 0;
	iperm[j] = j;
	iperm[*n + j] = j;
/* L1: */
    }
/* -----------------------------------------------------------------------
 */
/*     beginning of main loop. */
/* -----------------------------------------------------------------------
 */
    i__1 = *n;
    for (ii = 1; ii <= i__1; ++ii) {
	j_1 = ia[ii];
	j2 = ia[ii + 1] - 1;
	tnorm = 0.;
	i__2 = j2;
	for (k = j_1; k <= i__2; ++k) {
	    tnorm += (d__1 = a[k], PetscAbsScalar(d__1));
/* L501: */
	}
	if (tnorm == 0.) {
	    goto L999;
	}
	tnorm /= j2 - j_1 + 1;

/*     unpack L-part and U-part of row of A in arrays  w  -- */

	lenu = 1;
	lenl = 0;
	jw[ii] = ii;
	w[ii] = (float)0.;
	jw[*n + ii] = ii;

	i__2 = j2;
	for (j = j_1; j <= i__2; ++j) {
	    k = iperm[*n + ja[j]];
	    t = a[j];
	    if (k < ii) {
		++lenl;
		jw[lenl] = k;
		w[lenl] = t;
		jw[*n + k] = lenl;
	    } else if (k == ii) {
		w[ii] = t;
	    } else {
		++lenu;
		jpos = ii + lenu - 1;
		jw[jpos] = k;
		w[jpos] = t;
		jw[*n + k] = jpos;
	    }
/* L170: */
	}
	jj = 0;
	len = 0;

/*     eliminate previous rows */

L150:
	++jj;
	if (jj > lenl) {
	    goto L160;
	}
/* ------------------------------------------------------------------
----- */
/*     in order to do the elimination in the correct order we must sel
ect */
/*     the smallest column index among jw(k), k=jj+1, ..., lenl. */
/* ------------------------------------------------------------------
----- */
	jrow = jw[jj];
	k = jj;

/*     determine smallest column index */

	i__2 = lenl;
	for (j = jj + 1; j <= i__2; ++j) {
	    if (jw[j] < jrow) {
		jrow = jw[j];
		k = j;
	    }
/* L151: */
	}

	if (k != jj) {
/*     exchange in jw */
	    j = jw[jj];
	    jw[jj] = jw[k];
	    jw[k] = j;
/*     exchange in jr */
	    jw[*n + jrow] = jj;
	    jw[*n + j] = k;
/*     exchange in w */
	    s = w[jj];
	    w[jj] = w[k];
	    w[k] = s;
	}

/*     zero out element in row by resetting jw(n+jrow) to zero. */

	jw[*n + jrow] = 0;

/*     get the multiplier for row to be eliminated: jrow */

	fact = w[jj] * alu[jrow];

/*     drop term if small */

	if (PetscAbsScalar(fact) <= *droptol) {
	    goto L150;
	}

/*     combine current row and row jrow */

	i__2 = jlu[jrow + 1] - 1;
	for (k = ju[jrow]; k <= i__2; ++k) {
	    s = fact * alu[k];
/*     new column number */
	    j = iperm[*n + jlu[k]];
	    jpos = jw[*n + j];
	    if (j >= ii) {

/*     dealing with upper part. */

		if (jpos == 0) {

/*     this is a fill-in element */

		    ++lenu;
		    i = ii + lenu - 1;
		    if (lenu > *n) {
			goto L995;
		    }
		    jw[i] = j;
		    jw[*n + j] = i;
		    w[i] = -s;
		} else {
/*     no fill-in element -- */
		    w[jpos] -= s;
		}
	    } else {

/*     dealing with lower part. */

		if (jpos == 0) {

/*     this is a fill-in element */

		    ++lenl;
		    if (lenl > *n) {
			goto L995;
		    }
		    jw[lenl] = j;
		    jw[*n + j] = lenl;
		    w[lenl] = -s;
		} else {

/*     this is not a fill-in element */

		    w[jpos] -= s;
		}
	    }
/* L203: */
	}

/*     store this pivot element -- (from left to right -- no danger of
 */
/*     overlap with the working elements in L (pivots). */

	++len;
	w[len] = fact;
	jw[len] = jrow;
	goto L150;
L160:

/*     reset double-pointer to zero (U-part) */

	i__2 = lenu;
	for (k = 1; k <= i__2; ++k) {
	    jw[*n + jw[ii + k - 1]] = 0;
/* L308: */
	}

/*     update L-matrix */

	lenl = len;
	len = PetscMin(lenl,*lfil);

/*     sort by quick-split */

	SPARSEKIT2qsplit(&w[1], &jw[1], &lenl, &len);

/*     store L-part -- in original coordinates .. */

	i__2 = len;
	for (k = 1; k <= i__2; ++k) {
	    if (ju0 > *iwk) {
		goto L996;
	    }
	    alu[ju0] = w[k];
	    jlu[ju0] = iperm[jw[k]];
	    ++ju0;
/* L204: */
	}

/*     save pointer to beginning of row ii of U */

	ju[ii] = ju0;

/*     update U-matrix -- first apply dropping strategy */

	len = 0;
	i__2 = lenu - 1;
	for (k = 1; k <= i__2; ++k) {
	    if ((d__1 = w[ii + k], PetscAbsScalar(d__1)) > *droptol * tnorm) {
		++len;
		w[ii + len] = w[ii + k];
		jw[ii + len] = jw[ii + k];
	    }
	}
	lenu = len + 1;
	len = PetscMin(lenu,*lfil);
	i__2 = lenu - 1;
	SPARSEKIT2qsplit(&w[ii + 1], &jw[ii + 1], &i__2, &len);

/*     determine next pivot -- */

	imax = ii;
	xmax = (d__1 = w[imax], PetscAbsScalar(d__1));
	xmax0 = xmax;
	icut = ii - 1 + *mbloc - (ii - 1) % *mbloc;
	i__2 = ii + len - 1;
	for (k = ii + 1; k <= i__2; ++k) {
	    t1 = (d__1 = w[k], PetscAbsScalar(d__1));
	    if (t1 > xmax && t1 * *permtol > xmax0 && jw[k] <= icut) {
		imax = k;
		xmax = t1;
	    }
	}

/*     exchange w's */

	tmp = w[ii];
	w[ii] = w[imax];
	w[imax] = tmp;

/*     update iperm and reverse iperm */

	j = jw[imax];
	i = iperm[ii];
	iperm[ii] = iperm[j];
	iperm[j] = i;

/*     reverse iperm */

	iperm[*n + iperm[ii]] = ii;
	iperm[*n + iperm[j]] = j;
/* ------------------------------------------------------------------
----- */

	if (len + ju0 > *iwk) {
	    goto L997;
	}

/*     copy U-part in original coordinates */

	i__2 = ii + len - 1;
	for (k = ii + 1; k <= i__2; ++k) {
	    jlu[ju0] = iperm[jw[k]];
	    alu[ju0] = w[k];
	    ++ju0;
/* L302: */
	}

/*     store inverse of diagonal element of u */

	if (w[ii] == 0.) {
	    w[ii] = (*droptol + 1e-4) * tnorm;
	}
	alu[ii] = 1. / w[ii];

/*     update pointer to beginning of next row of U. */

	jlu[ii + 1] = ju0;
/* ------------------------------------------------------------------
----- */
/*     end main loop */
/* ------------------------------------------------------------------
----- */
/* L500: */
    }

/*     permute all column indices of LU ... */

    i__1 = jlu[*n + 1] - 1;
    for (k = jlu[1]; k <= i__1; ++k) {
	jlu[k] = iperm[*n + jlu[k]];
    }

/*     ...and of A */

    i__1 = ia[*n + 1] - 1;
    for (k = ia[1]; k <= i__1; ++k) {
	ja[k] = iperm[*n + ja[k]];
    }

    *ierr = 0;
    return 0;

/*     incomprehensible error. Matrix must be wrong. */

L995:
    *ierr = -1;
    return 0;

/*     insufficient storage in L. */

L996:
    *ierr = -2;
    return 0;

/*     insufficient storage in U. */

L997:
    *ierr = -3;
    return 0;

/*     illegal lfil entered. */

L998:
    *ierr = -4;
    return 0;

/*     zero row encountered */

L999:
    *ierr = -5;
    return 0;
/* ----------------end-of-ilutp-------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* ilutp_ */

