
/* rcm.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <../src/mat/order/order.h>

/*****************************************************************/
/*********     RCM ..... REVERSE CUTHILL-MCKEE ORDERING   *******/
/*****************************************************************/
/*    PURPOSE - RCM NUMBERS A CONNECTED COMPONENT SPECIFIED BY    */
/*       MASK AND ROOT, USING THE RCM ALGORITHM.                  */
/*       THE NUMBERING IS TO BE STARTED AT THE NODE ROOT.         */
/*                                                               */
/*    INPUT PARAMETERS -                                         */
/*       ROOT - IS THE NODE THAT DEFINES THE CONNECTED           */
/*              COMPONENT AND IT IS USED AS THE STARTING         */
/*              NODE FOR THE RCM ORDERING.                       */
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR           */
/*              THE GRAPH.                                       */
/*                                                               */
/*    UPDATED PARAMETERS -                                       */
/*       MASK - ONLY THOSE NODES WITH NONZERO INPUT MASK         */
/*              VALUES ARE CONSIDERED BY THE ROUTINE.  THE       */
/*              NODES NUMBERED BY RCM WILL HAVE THEIR            */
/*              MASK VALUES SET TO ZERO.                         */
/*                                                               */
/*    OUTPUT PARAMETERS -                                        */
/*       PERM - WILL CONTAIN THE RCM ORDERING.                   */
/*       CCSIZE - IS THE SIZE OF THE CONNECTED COMPONENT         */
/*              THAT HAS BEEN NUMBERED BY RCM.                  */
/*                                                              */
/*    WORKING PARAMETER -                                       */
/*       DEG - IS A TEMPORARY VECTOR USED TO HOLD THE DEGREE    */
/*              OF THE NODES IN THE SECTION GRAPH SPECIFIED     */
/*              BY MASK AND ROOT.                               */
/*                                                              */
/*    PROGRAM SUBROUTINES -                                     */
/*       DEGREE.                                                */
/*                                                              */
/****************************************************************/
#undef __FUNCT__
#define __FUNCT__ "SPARSEPACKrcm"
PetscErrorCode SPARSEPACKrcm(const PetscInt *root,const PetscInt *xadj,const PetscInt *adjncy,
	PetscInt *mask, PetscInt *perm, PetscInt *ccsize, PetscInt *deg)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt node, fnbr, lnbr, i, j, k, l, lperm, jstop, jstrt;
    PetscInt lbegin, lvlend, nbr;

/*       FIND THE DEGREES OF THE NODES IN THE                  */
/*       COMPONENT SPECIFIED BY MASK AND ROOT.                 */
/*       -------------------------------------                 */


    PetscFunctionBegin;
    /* Parameter adjustments */
    --deg;
    --perm;
    --mask;
    --adjncy;
    --xadj;


    SPARSEPACKdegree(root, &xadj[1], &adjncy[1], &mask[1], &deg[1], ccsize, &perm[1]);
    mask[*root] = 0;
    if (*ccsize <= 1) {
	PetscFunctionReturn(0);
    }
    lvlend = 0;
    lnbr = 1;
/*       LBEGIN AND LVLEND POINT TO THE BEGINNING AND */
/*       THE END OF THE CURRENT LEVEL RESPECTIVELY.  */
L100:
    lbegin = lvlend + 1;
    lvlend = lnbr;
    i__1 = lvlend;
    for (i = lbegin; i <= i__1; ++i) {
/*          FOR EACH NODE IN CURRENT LEVEL ...     */
	node = perm[i];
	jstrt = xadj[node];
	jstop = xadj[node + 1] - 1;

/*          FIND THE UNNUMBERED NEIGHBORS OF NODE.   */
/*          FNBR AND LNBR POINT TO THE FIRST AND LAST  */
/*          UNNUMBERED NEIGHBORS RESPECTIVELY OF THE CURRENT  */
/*          NODE IN PERM. */
	fnbr = lnbr + 1;
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    nbr = adjncy[j];
	    if (!mask[nbr]) {
		goto L200;
	    }
	    ++lnbr;
	    mask[nbr] = 0;
	    perm[lnbr] = nbr;
L200:
	    ;
	}
	if (fnbr >= lnbr) {
	    goto L600;
	}
/*             SORT THE NEIGHBORS OF NODE IN INCREASING    */
/*             ORDER BY DEGREE. LINEAR INSERTION IS USED.*/
	k = fnbr;
L300:
	l = k;
	++k;
	nbr = perm[k];
L400:
	if (l < fnbr) {
	    goto L500;
	}
	lperm = perm[l];
	if (deg[lperm] <= deg[nbr]) {
	    goto L500;
	}
	perm[l + 1] = lperm;
	--l;
	goto L400;
L500:
	perm[l + 1] = nbr;
	if (k < lnbr) {
	    goto L300;
	}
L600:
	;
    }
    if (lnbr > lvlend) {
	goto L100;
    }
/*       WE NOW HAVE THE CUTHILL MCKEE ORDERING.*/
/*       REVERSE IT BELOW ...*/
    k = *ccsize / 2;
    l = *ccsize;
    i__1 = k;
    for (i = 1; i <= i__1; ++i) {
	lperm = perm[l];
	perm[l] = perm[i];
	perm[i] = lperm;
	--l;
    }
    PetscFunctionReturn(0);
}

