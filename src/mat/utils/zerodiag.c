#ifndef lint
static char vcid[] = "$Id: zerodiag.c,v 1.1 1994/03/18 00:27:07 gropp Exp $";
#endif

/*
    This file contains routines to reorder a matrix so that the diagonal
    elements meet a simple criteria.  The most common use is expected to
    be reordering matrices with zero diagonal elements for incomplete
    factorizations.  A pivoting (partial or pairwise) routine is planned.
 */

#include "tools.h"
#include <math.h>
#include "sparse/spmat.h"
#include "sparse/sppriv.h"
#define SWAP(a,b) {int _t; _t = a; a = b; b = _t; }
#define SWAPD(a,b) {double _t; _t = a; a = b; b = _t; }

int SpiZeroFindPre( );

/*@
    SpUnSymmetricReorderForZeroDiagonal - Reorder the matrix so that no
    zeros (or small elements) are on the diagonal

    Input Parameters:
.   mat  - matrix to reorder
.   atol - elements smaller than this in magnitude are candidates for 
           reordering
.   rmap,cmap - row and column permutations.  On entrance, they should be
                either the identity ordering or copies of the matrix's
		row and column permutations.  On exit, they will be the 
		modified permutations.

    Error Conditions:
$   No memory 
$   Unable to find suitable reording

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure; use the pivoting factorization routine in that
    case.

    After this routine has been used, you MUST recompute the icolmap
    mapping (this is the inverse column mapping; since this routine
    modifies the column mapping, the inverse column mapping becomes
    invalid.  The reason that it is not "corrected" here is because
    additional orderings can be applied before the inverse column 
    mapping is needed.  THIS DECISION MAY CHANGE if it proves a poor
    choice.  Use the routine SpInverse to recompute the inverse column mapping.

    Algorithm:
    Column pivoting is used.  Choice of column is made by looking at the
    non-zero elements in the row.  This algorithm is simple and fast but
    does NOT guarentee that a non-singular or well conditioned
    principle submatrix will be produced.
@*/
void SpUnSymmetricReorderForZeroDiagonal( mat, atol, rmap, cmap )
SpMat  *mat;
double atol;
int    *rmap, *cmap;
{
int      prow, k, nz, n, repl, *j, *col, *row;
double   *v, repla;

SPLITTOMAT(mat);

col = cmap;
row = rmap;
n   = mat->rows;

for (prow=0; prow<n; prow++) {
    SpScatterFromRow( mat, row[prow], &nz, &j, &v );
    for (k=0; k<nz; k++) 
	if (col[j[k]] == prow) break;
    if (k >= nz || fabs(v[k]) <= atol) {
	/* Element too small or zero; find the best candidate */
	repl  = prow;
	repla = (k >= nz) ? 0.0 : fabs(v[k]);
	for (k=0; k<nz; k++) 
	    if (col[j[k]] > prow && fabs(v[k]) > repla) {
		repl = col[j[k]];
		repla = fabs(v[k]);
                }
	if (prow == repl) {
	    /* Now we need to look for an element that allows us
	       to pivot with a previous column.  To do this, we need
	       to be sure that we don't introduce a zero in a previous
	       diagonal */
	    if (!SpiZeroFindPre( mat, prow, row, col, repla, atol, 
				 &repl, &repla )) {
		SETERRC(1,"Can not reorder matrix");
		break;
		}
	    }
	SWAP(col[prow],col[repl]); 
        }
    }
}

/*@
    SpSymmetricReorderForZeroDiagonal - Reorder the matrix so that no
    zeros (or small elements) are on the diagonal

    Input Parameters:
.   mat  - matrix to reorder
.   atol - elements smaller than this in magnitude are candidates for 
           reordering
.   rmap,cmap - row and column permutations.  On entrance, they should be
                either the identity ordering or copies of the matrix's
		row and column permutations.  On exit, they will be the 
		moidified permutations.

    Error Conditions:
$   No memory 
$   Unable to find suitable reording

    Notes:
    This is not intended as a replacement for pivoting for matrices that
    have ``bad'' structure; use the pivoting factorization routine in that
    case (those routines are not yet available).

    After this routine has been used, you MUST recompute the icolmap
    mapping (this is the inverse column mapping; since this routine
    modifies the column mapping, the inverse column mapping becomes
    invalid.  The reason that it is not "corrected" here is because
    additional orderings can be applied before the inverse column 
    mapping is needed.  THIS DECISION MAY CHANGE if it proves a poor
    choice.  Use the routine SpInverse to recompute the inverse column 
    mapping.

    Algorithm:
    This is more complicated than the unsymmetric version, since a zero
    diagonal element can not be moved off the diagonal by symmetric
    permutations.
    Rather, we try to insure that a zero diagonal will be subject to some
    fill before it is encountered.  We do this as follows

    For each zero diagonal element, look down that row for a non-zero entry
    such that the diagonal under (in the same column) is non-zero.  The
    criteria is to pick the largest corresponding diagonal (other criteria
    are possible, such as a balance or a "no-fill" estimate of the
    resulting pivot sizes).  To speed this computation up, we gather some
    auxillery data: location of the diagonal elements and their values.
    We have to update these structures as we choose a reordering.
@*/
void SpSymmetricReorderForZeroDiagonal( mat, atol, rmap, cmap )
SpMat  *mat;
double atol;
int    *rmap, *cmap;
{
int      prow, k, nz, n, repl, *j, *col, *row, cloc;
double   *v, repla;
double   *diagv, offd, rval;

SPLITTOMAT(mat);

n     = mat->rows;
col   = cmap;
row   = rmap;
diagv = (double *)MALLOC( n * sizeof(double) );       CHKPTR(diagv);
/* load the diagonal values */
for (prow=0; prow<n; prow++) {
    SpScatterFromRow( mat, row[prow], &nz, &j, &v );
    rval     = 0.0;
    for (k=0; k<nz; k++)
    	if (col[j[k]] == prow) {
    	    rval = v[k];
	    break;
	    }
    if (k >= nz) {
    	/* No diagonal element.  Add it */
    	SpAddValue( mat, 0.0, prow, prow );
        }
    diagv[row[prow]] = fabs(rval);
    }
    
for (prow=0; prow<n; prow++) {
    if (diagv[prow] > atol) continue;
    /* If we knew that the column indices in a row were sorted, we'd only
       have to look at the values after the diagonal's location.  Since
       we are not making that assumption, we have to look at every element.
       This lets us use this routine after applying a fill-reducing ordering
       (though in that case, the diagonal elements should already have been
       introduced into the structure). */
    /* For each column, look for the largest diagonal element */
    SpScatterFromRow( mat, row[prow], &nz, &j, &v );
    repl     = prow;
    repla    = diagv[row[prow]];
    for (k=0; k<nz; k++) 
        if ((cloc = col[j[k]]) > prow && diagv[cloc] > repla) {
      	    repl  = j[k];
	    repla = diagv[cloc];
	    offd  = v[k];
            }
    if (prow == repl) {
        SETERRC(1,"Can not reorder matrix");
	return;
        }
    /* Compute an estimate of the replaced diagonal */
    diagv[row[prow]] = fabs( offd * offd / repla );
    SWAPD(diagv[row[prow]],diagv[row[repl]]);

    SWAP(col[prow],col[repl]);
    if (col != row) 
        SWAP(row[prow],row[repl]);
    }
FREE(diagv);
}

/* Given a current row and current permutation, find a column permutation
   that removes a zero diagonal */
int SpiZeroFindPre( mat, prow, row, col, repla, atol, rc, rcv )
SpMat  *mat;
int    prow, *row, *col, *rc;
double repla, atol, *rcv;
{
int      k, nz, repl, *j, kk, nnz, *jj;
double   *v, *vv;

SpScatterFromRow( mat, row[prow], &nz, &j, &v );
for (k=0; k<nz; k++) {
    if (col[j[k]] < prow && fabs(v[k]) > repla) {
	/* See if this one will work */
	repl  = col[j[k]];
	SpScatterFromRow( mat, row[repl], &nnz, &jj, &vv );
	for (kk=0; kk<nnz; kk++) {
	    if (col[jj[kk]] == prow && fabs(vv[kk]) > atol) {
		*rcv = fabs(v[k]);
		*rc  = repl;
		return 1;
		}
	    }
	}
    }
return 0;
}
