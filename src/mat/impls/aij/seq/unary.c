#define PETSCMAT_DLL

/* unary.f -- translated by f2c (version of 25 March 1992  12:58:56).

        This code is protected by the GNU copyright. See the file 
     ilut.c in this directory for the full copyright. See below for the Author.
*/
#include "petsc.h"
/* ----------------------------------------------------------------------- */
static PetscErrorCode SPARSEKIT2rperm(PetscInt *nrow,PetscScalar *a,PetscInt *ja,PetscInt *ia,PetscScalar *ao,PetscInt *jao,PetscInt *iao,PetscInt *perm,PetscInt *job)
{
    /* System generated locals */
    PetscInt i__1,i__2;

    /* Local variables */
    PetscInt i,j,k,ii,ko;
    PetscInt values;

/* -----------------------------------------------------------------------
 */
/* this subroutine permutes the rows of a matrix in CSR format. */
/* rperm  computes B = P A  where P is a permutation matrix. */
/* the permutation P is defined through the array perm: for each j, */
/* perm(j) represents the destination row number of row number j. */
/* Youcef Saad -- recoded Jan 28, 1991. */
/* -----------------------------------------------------------------------
 */
/* on entry: */
/* ---------- */
/* n 	= dimension of the matrix */
/* a, ja, ia = input matrix in csr format */
/* perm 	= integer array of length nrow containing the permutation arrays 
*/
/* 	  for the rows: perm(i) is the destination of row i in the */
/*         permuted matrix. */
/*         ---> a(i,j) in the original matrix becomes a(perm(i),j) */
/*         in the output  matrix. */

/* job	= integer indicating the work to be done: */
/* 		job = 1	permute a, ja, ia into ao, jao, iao */
/*                       (including the copying of real values ao and */
/*                       the array iao). */
/* 		job .ne. 1 :  ignore real values. */
/*                     (in which case arrays a and ao are not needed nor 
*/
/*                      used). */

/* ------------ */
/* on return: */
/* ------------ */
/* ao, jao, iao = input matrix in a, ja, ia format */
/* note : */
/*        if (job.ne.1)  then the arrays a and ao are not used. */
/* ----------------------------------------------------------------------c
 */
/*           Y. Saad, May  2, 1990                                      c 
*/
/* ----------------------------------------------------------------------c
 */
    /* Parameter adjustments */
    --perm;
    --iao;
    --jao;
    --ao;
    --ia;
    --ja;
    --a;

    /* Function Body */
    values = *job == 1;

/*     determine pointers for output matix. */

    i__1 = *nrow;
    for (j = 1; j <= i__1; ++j) {
	i = perm[j];
	iao[i + 1] = ia[j + 1] - ia[j];
/* L50: */
    }

/* get pointers from lengths */

    iao[1] = 1;
    i__1 = *nrow;
    for (j = 1; j <= i__1; ++j) {
	iao[j + 1] += iao[j];
/* L51: */
    }

/* copying */

    i__1 = *nrow;
    for (ii = 1; ii <= i__1; ++ii) {

/* old row = ii  -- new row = iperm(ii) -- ko = new pointer */

	ko = iao[perm[ii]];
	i__2 = ia[ii + 1] - 1;
	for (k = ia[ii]; k <= i__2; ++k) {
	    jao[ko] = ja[k];
	    if (values) {
		ao[ko] = a[k];
	    }
	    ++ko;
/* L60: */
	}
/* L100: */
    }

    return 0;
/* ---------end-of-rperm -------------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* rperm_ */

/* ----------------------------------------------------------------------- */
static PetscErrorCode SPARSEKIT2cperm(PetscInt *nrow,PetscScalar * a,PetscInt * ja,PetscInt * ia,PetscScalar * ao,PetscInt * jao,PetscInt * iao,PetscInt * perm,PetscInt * job)
{
    /* System generated locals */
    PetscInt i__1;

    /* Local variables */
    PetscInt i,k,nnz;

/* -----------------------------------------------------------------------
 */
/* this subroutine permutes the columns of a matrix a, ja, ia. */
/* the result is written in the output matrix  ao, jao, iao. */
/* cperm computes B = A P, where  P is a permutation matrix */
/* that maps column j into column perm(j), i.e., on return */
/*      a(i,j) becomes a(i,perm(j)) in new matrix */
/* Y. Saad, May 2, 1990 / modified Jan. 28, 1991. */
/* -----------------------------------------------------------------------
 */
/* on entry: */
/* ---------- */
/* nrow 	= row dimension of the matrix */

/* a, ja, ia = input matrix in csr format. */

/* perm	= integer array of length ncol (number of columns of A */
/*         containing the permutation array  the columns: */
/*         a(i,j) in the original matrix becomes a(i,perm(j)) */
/*         in the output matrix. */

/* job	= integer indicating the work to be done: */
/* 		job = 1	permute a, ja, ia into ao, jao, iao */
/*                       (including the copying of real values ao and */
/*                       the array iao). */
/* 		job .ne. 1 :  ignore real values ao and ignore iao. */

/* ------------ */
/* on return: */
/* ------------ */
/* ao, jao, iao = input matrix in a, ja, ia format (array ao not needed) 
*/

/* Notes: */
/* ------- */
/* 1. if job=1 then ao, iao are not used. */
/* 2. This routine is in place: ja, jao can be the same. */
/* 3. If the matrix is initially sorted (by increasing column number) */
/*    then ao,jao,iao  may not be on return. */

/* ----------------------------------------------------------------------c
 */
/* local parameters: */

    /* Parameter adjustments */
    --perm;
    --iao;
    --jao;
    --ao;
    --ia;
    --ja;
    --a;

    /* Function Body */
    nnz = ia[*nrow + 1] - 1;
    i__1 = nnz;
    for (k = 1; k <= i__1; ++k) {
	jao[k] = perm[ja[k]];
/* L100: */
    }

/*     done with ja array. return if no need to touch values. */

    if (*job != 1) {
	return 0;
    }

/* else get new pointers -- and copy values too. */

    i__1 = *nrow + 1;
    for (i = 1; i <= i__1; ++i) {
	iao[i] = ia[i];
/* L1: */
    }

    i__1 = nnz;
    for (k = 1; k <= i__1; ++k) {
	ao[k] = a[k];
/* L2: */
    }

    return 0;
/* ---------end-of-cperm--------------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* cperm_ */

/* ----------------------------------------------------------------------- */
PetscErrorCode SPARSEKIT2dperm(PetscInt *nrow,PetscScalar *a,PetscInt *ja,PetscInt *ia,PetscScalar *ao,PetscInt *jao,PetscInt *iao,PetscInt *perm,PetscInt *qperm,PetscInt *job)
{
    PetscInt locjob;

/* -----------------------------------------------------------------------
 */
/* This routine permutes the rows and columns of a matrix stored in CSR */

/* format. i.e., it computes P A Q, where P, Q are permutation matrices. 
*/
/* P maps row i into row perm(i) and Q maps column j into column qperm(j):
 */
/*      a(i,j)    becomes   a(perm(i),qperm(j)) in new matrix */
/* In the particular case where Q is the transpose of P (symmetric */
/* permutation of A) then qperm is not needed. */
/* note that qperm should be of length ncol (number of columns) but this 
*/
/* is not checked. */
/* -----------------------------------------------------------------------
 */
/* Y. Saad, Sep. 21 1989 / recoded Jan. 28 1991. */
/* -----------------------------------------------------------------------
 */
/* on entry: */
/* ---------- */
/* n 	= dimension of the matrix */
/* a, ja, */
/*    ia = input matrix in a, ja, ia format */
/* perm 	= integer array of length n containing the permutation arrays */
/* 	  for the rows: perm(i) is the destination of row i in the */
/*         permuted matrix -- also the destination of column i in case */
/*         permutation is symmetric (job .le. 2) */

/* qperm	= same thing for the columns. This should be provided only */
/*         if job=3 or job=4, i.e., only in the case of a nonsymmetric */
/* 	  permutation of rows and columns. Otherwise qperm is a dummy */

/* job	= integer indicating the work to be done: */
/* * job = 1,2 permutation is symmetric  Ao :== P * A * transp(P) */
/* 		job = 1	permute a, ja, ia into ao, jao, iao */
/* 		job = 2 permute matrix ignoring real values. */
/* * job = 3,4 permutation is non-symmetric  Ao :== P * A * Q */
/* 		job = 3	permute a, ja, ia into ao, jao, iao */
/* 		job = 4 permute matrix ignoring real values. */

/* on return: */
/* ----------- */
/* ao, jao, iao = input matrix in a, ja, ia format */

/* in case job .eq. 2 or job .eq. 4, a and ao are never referred to */
/* and can be dummy arguments. */
/* Notes: */
/* ------- */
/*  1) algorithm is in place */
/*  2) column indices may not be sorted on return even  though they may be
 */
/*     on entry. */
/* ----------------------------------------------------------------------c
 */
/* local variables */

/*     locjob indicates whether or not real values must be copied. */

    /* Parameter adjustments */
    --qperm;
    --perm;
    --iao;
    --jao;
    --ao;
    --ia;
    --ja;
    --a;

    /* Function Body */
    locjob = *job % 2;

/* permute rows first */

    SPARSEKIT2rperm(nrow, &a[1], &ja[1], &ia[1], &ao[1], &jao[1], &iao[1], &perm[1], &locjob);

/* then permute columns */

    locjob = 0;

    if (*job <= 2) {
	SPARSEKIT2cperm(nrow, &ao[1], &jao[1], &iao[1], &ao[1], &jao[1], &iao[1], &perm[1], &locjob);
    } else {
	SPARSEKIT2cperm(nrow, &ao[1], &jao[1], &iao[1], &ao[1], &jao[1], &iao[1], &qperm[1], &locjob);
    }

    return 0;
/* -------end-of-dperm----------------------------------------------------
 */
/* -----------------------------------------------------------------------
 */
} /* dperm_ */

/* ----------------------------------------------------------------------- */
PetscErrorCode SPARSEKIT2msrcsr(PetscInt *n,PetscScalar * a,PetscInt * ja,PetscScalar * ao,PetscInt * jao,PetscInt * iao,PetscScalar * wk,PetscInt * iwk)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt iptr;
    PetscInt added;
    PetscInt i, j, k, idiag, ii;

/* -----------------------------------------------------------------------
 */
/*       Modified - Sparse Row  to   Compressed Sparse Row */

/* -----------------------------------------------------------------------
 */
/* converts a compressed matrix using a separated diagonal */
/* (modified sparse row format) in the Compressed Sparse Row */
/* format. */
/* does not check for zero elements in the diagonal. */


/* on entry : */
/* --------- */
/* n          = row dimension of matrix */
/* a, ja      = sparse matrix in msr sparse storage format */
/*              see routine csrmsr for details on data structure */

/* on return : */
/* ----------- */

/* ao,jao,iao = output matrix in csr format. */

/* work arrays: */
/* ------------ */
/* wk       = real work array of length n */
/* iwk      = integer work array of length n+1 */

/* notes: */
/*   The original version of this was NOT in place, but has */
/*   been modified by adding the vector iwk to be in place. */
/*   The original version had ja instead of iwk everywhere in */
/*   loop 500.  Modified  Sun 29 May 1994 by R. Bramley (Indiana). */

/* -----------------------------------------------------------------------
 */
    /* Parameter adjustments */
    --iwk;
    --wk;
    --iao;
    --jao;
    --ao;
    --ja;
    --a;

    /* Function Body */
    i__1 = *n;
    for (i = 1; i <= i__1; ++i) {
	wk[i] = a[i];
	iwk[i] = ja[i];
/* L1: */
    }
    iwk[*n + 1] = ja[*n + 1];
    iao[1] = 1;
    iptr = 1;
/* --------- */
    i__1 = *n;
    for (ii = 1; ii <= i__1; ++ii) {
	added = 0;
	idiag = iptr + (iwk[ii + 1] - iwk[ii]);
	i__2 = iwk[ii + 1] - 1;
	for (k = iwk[ii]; k <= i__2; ++k) {
	    j = ja[k];
	    if (j < ii) {
		ao[iptr] = a[k];
		jao[iptr] = j;
		++iptr;
	    } else if (added) {
		ao[iptr] = a[k];
		jao[iptr] = j;
		++iptr;
	    } else {
/* add diag element - only reserve a position for it. */
		idiag = iptr;
		++iptr;
		added = 1;
/*     then other element */
		ao[iptr] = a[k];
		jao[iptr] = j;
		++iptr;
	    }
/* L100: */
	}
	ao[idiag] = wk[ii];
	jao[idiag] = ii;
	if (! added) {
	    ++iptr;
	}
	iao[ii + 1] = iptr;
/* L500: */
    }
    return 0;
} /* msrcsr_ */

