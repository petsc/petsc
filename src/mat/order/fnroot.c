
/* fnroot.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <../src/mat/order/order.h>

/*****************************************************************/
/********     FNROOT ..... FIND PSEUDO-PERIPHERAL NODE    ********/
/*****************************************************************/
/*   PURPOSE - FNROOT IMPLEMENTS A MODIFIED VERSION OF THE       */
/*      SCHEME BY GIBBS, POOLE, AND STOCKMEYER TO FIND PSEUDO-   */
/*      PERIPHERAL NODES.  IT DETERMINES SUCH A NODE FOR THE     */
/*      SECTION SUBGRAPH SPECIFIED BY MASK AND ROOT.             */
/*   INPUT PARAMETERS -                                          */
/*      (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE GRAPH. */
/*      MASK - SPECIFIES A SECTION SUBGRAPH. NODES FOR WHICH     */
/*             MASK IS ZERO ARE IGNORED BY FNROOT.              */
/*   UPDATED PARAMETER -                                        */
/*      ROOT - ON INPUT, IT (ALONG WITH MASK) DEFINES THE       */
/*             COMPONENT FOR WHICH A PSEUDO-PERIPHERAL NODE IS  */
/*             TO BE FOUND. ON OUTPUT, IT IS THE NODE OBTAINED. */
/*                                                              */
/*   OUTPUT PARAMETERS -                                        */
/*      NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE   */
/*             ROOTED AT THE NODE ROOT.                         */
/*      (XLS,LS) - THE LEVEL STRUCTURE ARRAY PAIR CONTAINING    */
/*                 THE LEVEL STRUCTURE FOUND.                   */
/*                                                              */
/*   PROGRAM SUBROUTINES -                                      */
/*      ROOTLS.                                                 */
/*                                                              */
/****************************************************************/
#undef __FUNCT__
#define __FUNCT__ "SPARSEPACKfnroot"
PetscErrorCode SPARSEPACKfnroot(PetscInt *root,const PetscInt *xadj,const PetscInt *adjncy,
                                PetscInt *mask, PetscInt *nlvl, PetscInt *xls, PetscInt *ls)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt ndeg, node, j, k, nabor, kstop, jstrt, kstrt, mindeg, ccsize, nunlvl;
/*       DETERMINE THE LEVEL STRUCTURE ROOTED AT ROOT. */

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ls;
    --xls;
    --mask;
    --adjncy;
    --xadj;

    SPARSEPACKrootls(root, &xadj[1], &adjncy[1], &mask[1], nlvl, &xls[1], &ls[1]);
    ccsize = xls[*nlvl + 1] - 1;
    if (*nlvl == 1 || *nlvl == ccsize) {
	PetscFunctionReturn(0);
    }
/*       PICK A NODE WITH MINIMUM DEGREE FROM THE LAST LEVEL.*/
L100:
    jstrt = xls[*nlvl];
    mindeg = ccsize;
    *root = ls[jstrt];
    if (ccsize == jstrt) {
	goto L400;
    }
    i__1 = ccsize;
    for (j = jstrt; j <= i__1; ++j) {
	node = ls[j];
	ndeg = 0;
	kstrt = xadj[node];
	kstop = xadj[node + 1] - 1;
	i__2 = kstop;
	for (k = kstrt; k <= i__2; ++k) {
	    nabor = adjncy[k];
	    if (mask[nabor] > 0) {
		++ndeg;
	    }
	}
	if (ndeg >= mindeg) {
	    goto L300;
	}
	*root = node;
	mindeg = ndeg;
L300:
	;
    }
/*       AND GENERATE ITS ROOTED LEVEL STRUCTURE.*/
L400:
    SPARSEPACKrootls(root, &xadj[1], &adjncy[1], &mask[1], &nunlvl, &xls[1], &ls[1]);
    if (nunlvl <= *nlvl) {
	PetscFunctionReturn(0);
    }
    *nlvl = nunlvl;
    if (*nlvl < ccsize) {
	goto L100;
    }
    PetscFunctionReturn(0);
}

