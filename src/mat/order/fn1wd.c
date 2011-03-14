
/* fn1wd.f -- translated by f2c (version 19931217).*/

#include <../src/mat/order/order.h>
extern PetscErrorCode SPARSEPACKfnroot(PetscInt*, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

/*****************************************************************/
/********     FN1WD ..... FIND ONE-WAY DISSECTORS        *********/
/*****************************************************************/
/*    PURPOSE - THIS SUBROUTINE FINDS ONE-WAY DISSECTORS OF      */
/*       A CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT.       */
/*                                                               */
/*    INPUT PARAMETERS -                                         */
/*       ROOT - A NODE THAT DEFINES (ALONG WITH MASK) THE        */
/*              COMPONENT TO BE PROCESSED.                       */
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.               */
/*                                                               */
/*    OUTPUT PARAMETERS -                                        */
/*       NSEP - NUMBER OF NODES IN THE ONE-WAY DISSECTORS.       */
/*       SEP - VECTOR CONTAINING THE DISSECTOR NODES.            */
/*                                                               */
/*    UPDATED PARAMETER -                                        */
/*       MASK - NODES IN THE DISSECTOR HAVE THEIR MASK VALUES    */
/*              SET TO ZERO.                                     */
/*                                                               */
/*    WORKING PARAMETERS-                                        */
/*       (XLS, LS) - LEVEL STRUCTURE USED BY THE ROUTINE FNROOT. */
/*                                                               */
/*    PROGRAM SUBROUTINE -                                       */
/*       FNROOT.                                                 */
/*****************************************************************/
#undef __FUNCT__  
#define __FUNCT__ "SPARSEPACKfn1wd" 
PetscErrorCode SPARSEPACKfn1wd(PetscInt *root, PetscInt *xadj, PetscInt *adjncy, 
                               PetscInt *mask, PetscInt *nsep, PetscInt *sep, PetscInt *nlvl, PetscInt *
                               xls, PetscInt *ls)
{
    /* System generated locals */
    PetscInt  i__1, i__2;

    /* Local variables */
    PetscInt  node, i, j, k;
    PetscReal width, fnlvl;
    PetscInt  kstop, kstrt, lp1beg, lp1end;
    PetscReal deltp1;
    PetscInt  lvlbeg, lvlend;
    PetscInt  nbr, lvl;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ls;
    --xls;
    --sep;
    --mask;
    --adjncy;
    --xadj;

    SPARSEPACKfnroot(root, &xadj[1], &adjncy[1], &mask[1], nlvl, &xls[1], &ls[1]);
    fnlvl = (PetscReal) (*nlvl);
    *nsep = xls[*nlvl + 1] - 1;
    width = (PetscReal) (*nsep) / fnlvl;
    deltp1 = sqrt((width * 3.f + 13.f) / 2.f) + 1.f;
    if (*nsep >= 50 && deltp1 <= fnlvl * .5f) {
        goto L300;
    }
/*       THE COMPONENT IS TOO SMALL, OR THE LEVEL STRUCTURE */
/*       IS VERY LONG AND NARROW. RETURN THE WHOLE COMPONENT.*/
    i__1 = *nsep;
    for (i = 1; i <= i__1; ++i) {
        node = ls[i];
        sep[i] = node;
        mask[node] = 0;
    }
    PetscFunctionReturn(0);
/*       FIND THE PARALLEL DISSECTORS.*/
L300:
    *nsep = 0;
    i = 0;
L400:
    ++i;
    lvl = (PetscInt)((PetscReal) i * deltp1 + .5f);
    if (lvl >= *nlvl) {
        PetscFunctionReturn(0);
    }
    lvlbeg = xls[lvl];
    lp1beg = xls[lvl + 1];
    lvlend = lp1beg - 1;
    lp1end = xls[lvl + 2] - 1;
    i__1 = lp1end;
    for (j = lp1beg; j <= i__1; ++j) {
        node = ls[j];
        xadj[node] = -xadj[node];
    }
/*          NODES IN LEVEL LVL ARE CHOSEN TO FORM DISSECTOR. */
/*          INCLUDE ONLY THOSE WITH NEIGHBORS IN LVL+1 LEVEL. */
/*          XADJ IS USED TEMPORARILY TO MARK NODES IN LVL+1.  */
    i__1 = lvlend;
    for (j = lvlbeg; j <= i__1; ++j) {
        node = ls[j];
        kstrt = xadj[node];
        kstop = (i__2 = xadj[node + 1], (PetscInt)PetscAbsInt(i__2)) - 1;
	i__2 = kstop;
	for (k = kstrt; k <= i__2; ++k) {
	    nbr = adjncy[k];
	    if (xadj[nbr] > 0) {
		goto L600;
	    }
	    ++(*nsep);
	    sep[*nsep] = node;
	    mask[node] = 0;
	    goto L700;
L600:
	    ;
	}
L700:
	;
    }
    i__1 = lp1end;
    for (j = lp1beg; j <= i__1; ++j) {
	node = ls[j];
	xadj[node] = -xadj[node];
    }
    goto L400;
}

