#define PETSCMAT_DLL
/* dsm.f -- translated by f2c (version of 25 March 1992  12:58:56). */

#include "../src/mat/color/color.h"

static PetscInt c_n1 = -1;

#undef __FUNCT__  
#define __FUNCT__ "MINPACKdsm" 
PetscErrorCode MINPACKdsm(PetscInt *m,PetscInt *n,PetscInt *npairs,PetscInt *indrow,PetscInt *indcol,PetscInt *ngrp,PetscInt *maxgrp,
                          PetscInt *mingrp,PetscInt *info,PetscInt *ipntr,PetscInt *jpntr,PetscInt *iwa,PetscInt *liwa)
{
    /* System generated locals */
    PetscInt i__1,i__2,i__3;

    /* Local variables */
    PetscInt i,j,maxclq,numgrp;

/*     Given the sparsity pattern of an m by n matrix A, this */
/*     subroutine determines a partition of the columns of A */
/*     consistent with the direct determination of A. */
/*     The sparsity pattern of the matrix A is specified by */
/*     the arrays indrow and indcol. On input the indices */
/*     for the non-zero elements of A are */
/*           indrow(k),indcol(k), k = 1,2,...,npairs. */
/*     The (indrow,indcol) pairs may be specified in any order. */
/*     Duplicate input pairs are permitted, but the subroutine */
/*     eliminates them. */
/*     The subroutine partitions the columns of A into groups */
/*     such that columns in the same group do not have a */
/*     non-zero in the same row position. A partition of the */
/*     columns of A with this property is consistent with the */
/*     direct determination of A. */
/*     The subroutine statement is */
/*       subroutine dsm(m,n,npairs,indrow,indcol,ngrp,maxgrp,mingrp, */
/*                      info,ipntr,jpntr,iwa,liwa) */
/*     where */
/*       m is a positive integer input variable set to the number */
/*         of rows of A. */
/*       n is a positive integer input variable set to the number */
/*         of columns of A. */
/*       npairs is a positive integer input variable set to the */
/*         number of (indrow,indcol) pairs used to describe the */
/*         sparsity pattern of A. */
/*       indrow is an integer array of length npairs. On input indrow */
/*         must contain the row indices of the non-zero elements of A. */
/*         On output indrow is permuted so that the corresponding */
/*         column indices are in non-decreasing order. The column */
/*         indices can be recovered from the array jpntr. */
/*       indcol is an integer array of length npairs. On input indcol */
/*         must contain the column indices of the non-zero elements of */
/*         A. On output indcol is permuted so that the corresponding */
/*         row indices are in non-decreasing order. The row indices */
/*         can be recovered from the array ipntr. */
/*       ngrp is an integer output array of length n which specifies */
/*         the partition of the columns of A. Column jcol belongs */
/*         to group ngrp(jcol). */
/*       maxgrp is an integer output variable which specifies the */
/*         number of groups in the partition of the columns of A. */
/*       mingrp is an integer output variable which specifies a lower */
/*         bound for the number of groups in any consistent partition */
/*         of the columns of A. */
/*       info is an integer output variable set as follows. For */
/*         normal termination info = 1. If m, n, or npairs is not */
/*         positive or liwa is less than max(m,6*n), then info = 0. */
/*         If the k-th element of indrow is not an integer between */
/*         1 and m or the k-th element of indcol is not an integer */
/*         between 1 and n, then info = -k. */
/*       ipntr is an integer output array of length m + 1 which */
/*         specifies the locations of the column indices in indcol. */
/*         The column indices for row i are */
/*               indcol(k), k = ipntr(i),...,ipntr(i+1)-1. */
/*         Note that ipntr(m+1)-1 is then the number of non-zero */
/*         elements of the matrix A. */
/*       jpntr is an integer output array of length n + 1 which */
/*         specifies the locations of the row indices in indrow. */
/*         The row indices for column j are */
/*               indrow(k), k = jpntr(j),...,jpntr(j+1)-1. */
/*         Note that jpntr(n+1)-1 is then the number of non-zero */
/*         elements of the matrix A. */
/*       iwa is an integer work array of length liwa. */
/*       liwa is a positive integer input variable not less than */
/*         max(m,6*n). */
/*     Subprograms called */
/*       MINPACK-supplied ... degr,ido,numsrt,seq,setr,slo,srtdat */
/*       FORTRAN-supplied ... max */
/*     Argonne National Laboratory. MINPACK Project. December 1984. */
/*     Thomas F. Coleman, Burton S. Garbow, Jorge J. More' */

    PetscFunctionBegin;
    /* Parameter adjustments */
    --iwa;
    --jpntr;
    --ipntr;
    --ngrp;
    --indcol;
    --indrow;

    *info = 0;

/*     Determine a lower bound for the number of groups. */

    *mingrp = 0;
    i__1 = *m;
    for (i = 1; i <= i__1; ++i) {
/* Computing MAX */
	i__2 = *mingrp,i__3 = ipntr[i + 1] - ipntr[i];
	*mingrp = PetscMax(i__2,i__3);
    }

/*     Determine the degree sequence for the intersection */
/*     graph of the columns of A. */

    MINPACKdegr(n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[*n * 5 + 1],&
	    iwa[*n + 1]);

/*     Color the intersection graph of the columns of A */
/*     with the smallest-last (SL) ordering. */

    MINPACKslo(n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[*n * 5 + 1],&
	    iwa[(*n << 2) + 1],&maxclq,&iwa[1],&iwa[*n + 1],&iwa[(*n << 1)
	     + 1],&iwa[*n * 3 + 1]);
    MINPACKseq(n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[(*n << 2) + 1],
	     &ngrp[1],maxgrp,&iwa[*n + 1]);
    *mingrp = PetscMax(*mingrp,maxclq);

/*     Exit if the smallest-last ordering is optimal. */

    if (*maxgrp == *mingrp) {
	PetscFunctionReturn(0);
    }

/*     Color the intersection graph of the columns of A */
/*     with the incidence-degree (ID) ordering. */

    MINPACKido(m,n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[*n * 5 + 1],
	     &iwa[(*n << 2) + 1],&maxclq,&iwa[1],&iwa[*n + 1],&iwa[(*n << 
	    1) + 1],&iwa[*n * 3 + 1]);
    MINPACKseq(n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[(*n << 2) + 1],
	     &iwa[1],&numgrp,&iwa[*n + 1]);
    *mingrp = PetscMax(*mingrp,maxclq);

/*     Retain the better of the two orderings so far. */

    if (numgrp < *maxgrp) {
	*maxgrp = numgrp;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    ngrp[j] = iwa[j];
	}

/*        Exit if the incidence-degree ordering is optimal. */

	if (*maxgrp == *mingrp) {
	    PetscFunctionReturn(0);
	}
    }

/*     Color the intersection graph of the columns of A */
/*     with the largest-first (LF) ordering. */

    i__1 = *n - 1;
    MINPACKnumsrt(n,&i__1,&iwa[*n * 5 + 1],&c_n1,&iwa[(*n << 2) + 1],&iwa[(*n 
	    << 1) + 1],&iwa[*n + 1]);
    MINPACKseq(n,&indrow[1],&jpntr[1],&indcol[1],&ipntr[1],&iwa[(*n << 2) + 1],
	     &iwa[1],&numgrp,&iwa[*n + 1]);

/*     Retain the best of the three orderings and exit. */

    if (numgrp < *maxgrp) {
	*maxgrp = numgrp;
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    ngrp[j] = iwa[j];
	}
    }
    PetscFunctionReturn(0);
}
