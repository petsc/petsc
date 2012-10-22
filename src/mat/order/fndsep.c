
/* fndsep.f -- translated by f2c (version 19931217).
*/

#include <../src/mat/order/order.h>

/*****************************************************************/
/*************     FNDSEP ..... FIND SEPARATOR       *************/
/*****************************************************************/
/*    PURPOSE - THIS ROUTINE IS USED TO FIND A SMALL             */
/*              SEPARATOR FOR A CONNECTED COMPONENT SPECIFIED    */
/*              BY MASK IN THE GIVEN GRAPH.                      */
/*                                                               */
/*    INPUT PARAMETERS -                                         */
/*       ROOT - IS THE NODE THAT DETERMINES THE MASKED           */
/*              COMPONENT.                                       */
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.          */
/*                                                               */
/*    OUTPUT PARAMETERS -                                        */
/*       NSEP - NUMBER OF VARIABLES IN THE SEPARATOR.            */
/*       SEP - VECTOR CONTAINING THE SEPARATOR NODES.            */
/*                                                               */
/*    UPDATED PARAMETER -                                        */
/*       MASK - NODES IN THE SEPARATOR HAVE THEIR MASK           */
/*              VALUES SET TO ZERO.                              */
/*                                                               */
/*    WORKING PARAMETERS -                                       */
/*       (XLS, LS) - LEVEL STRUCTURE PAIR FOR LEVEL STRUCTURE    */
/*              FOUND BY FNROOT.                                 */
/*                                                               */
/*    PROGRAM SUBROUTINES -                                      */
/*       FNROOT.                                                 */
/*                                                               */
/*****************************************************************/
#undef __FUNCT__
#define __FUNCT__ "SPARSEPACKfndsep"
PetscErrorCode SPARSEPACKfndsep(PetscInt *root,const PetscInt *inxadj,const PetscInt *adjncy,
	                        PetscInt *mask, PetscInt *nsep, PetscInt *sep, PetscInt *xls, PetscInt *ls)
{
    PetscInt *xadj = (PetscInt*)inxadj; /* Used as temporary and reset within this function */

    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt node, nlvl, i, j, jstop, jstrt, mp1beg, mp1end, midbeg, midend, midlvl;
    PetscInt nbr;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ls;
    --xls;
    --sep;
    --mask;
    --adjncy;
    --xadj;

    SPARSEPACKfnroot(root, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &ls[1]);
/*       IF THE NUMBER OF LEVELS IS LESS THAN 3, RETURN */
/*       THE WHOLE COMPONENT AS THE SEPARATOR.*/
    if (nlvl >= 3) {
	goto L200;
    }
    *nsep = xls[nlvl + 1] - 1;
    i__1 = *nsep;
    for (i = 1; i <= i__1; ++i) {
	node = ls[i];
	sep[i] = node;
	mask[node] = 0;
    }
    PetscFunctionReturn(0);
/*       FIND THE MIDDLE LEVEL OF THE ROOTED LEVEL STRUCTURE.*/
L200:
    midlvl = (nlvl + 2) / 2;
    midbeg = xls[midlvl];
    mp1beg = xls[midlvl + 1];
    midend = mp1beg - 1;
    mp1end = xls[midlvl + 2] - 1;
/*       THE SEPARATOR IS OBTAINED BY INCLUDING ONLY THOSE*/
/*       MIDDLE-LEVEL NODES WITH NEIGHBORS IN THE MIDDLE+1*/
/*       LEVEL. XADJ IS USED TEMPORARILY TO MARK THOSE*/
/*       NODES IN THE MIDDLE+1 LEVEL.*/
    i__1 = mp1end;
    for (i = mp1beg; i <= i__1; ++i) {
	node = ls[i];
	xadj[node] = -xadj[node];
    }
    *nsep = 0;
    i__1 = midend;
    for (i = midbeg; i <= i__1; ++i) {
	node = ls[i];
	jstrt = xadj[node];
	jstop = (i__2 = xadj[node + 1], (PetscInt)PetscAbsInt(i__2)) - 1;
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    nbr = adjncy[j];
	    if (xadj[nbr] > 0) {
		goto L400;
	    }
	    ++(*nsep);
	    sep[*nsep] = node;
	    mask[node] = 0;
	    goto L500;
L400:
	    ;
	}
L500:
	;
    }
/*       RESET XADJ TO ITS CORRECT SIGN.*/
    i__1 = mp1end;
    for (i = mp1beg; i <= i__1; ++i) {
	node = ls[i];
	xadj[node] = -xadj[node];
    }
    PetscFunctionReturn(0);
}
