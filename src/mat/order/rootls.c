
/* rootls.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <../src/mat/order/order.h>

/*****************************************************************/
/*********     ROOTLS ..... ROOTED LEVEL STRUCTURE      **********/
/*****************************************************************/
/*    PURPOSE - ROOTLS GENERATES THE LEVEL STRUCTURE ROOTED */
/*       AT THE INPUT NODE CALLED ROOT. ONLY THOSE NODES FOR*/
/*       WHICH MASK IS NONZERO WILL BE CONSIDERED.*/
/*                                                */
/*    INPUT PARAMETERS -                          */
/*       ROOT - THE NODE AT WHICH THE LEVEL STRUCTURE IS TO*/
/*              BE ROOTED.*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE*/
/*              GIVEN GRAPH.*/
/*       MASK - IS USED TO SPECIFY A SECTION SUBGRAPH. NODES*/
/*              WITH MASK(I)=0 ARE IGNORED.*/
/*    OUTPUT PARAMETERS -*/
/*       NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE.*/
/*       (XLS, LS) - ARRAY PAIR FOR THE ROOTED LEVEL STRUCTURE.*/
/*****************************************************************/
#undef __FUNCT__
#define __FUNCT__ "SPARSEPACKrootls"
PetscErrorCode SPARSEPACKrootls(const PetscInt *root,const PetscInt *xadj,const PetscInt *adjncy,
	PetscInt *mask, PetscInt *nlvl, PetscInt *xls, PetscInt *ls)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt node, i, j, jstop, jstrt, lbegin, ccsize, lvlend, lvsize,
	    nbr;

/*       INITIALIZATION ...*/


    PetscFunctionBegin;
    /* Parameter adjustments */
    --ls;
    --xls;
    --mask;
    --adjncy;
    --xadj;

    mask[*root] = 0;
    ls[1] = *root;
    *nlvl = 0;
    lvlend = 0;
    ccsize = 1;
/*       LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT*/
/*       LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.*/
L200:
    lbegin = lvlend + 1;
    lvlend = ccsize;
    ++(*nlvl);
    xls[*nlvl] = lbegin;
/*       GENERATE THE NEXT LEVEL BY FINDING ALL THE MASKED */
/*       NEIGHBORS OF NODES IN THE CURRENT LEVEL.*/
    i__1 = lvlend;
    for (i = lbegin; i <= i__1; ++i) {
	node = ls[i];
	jstrt = xadj[node];
	jstop = xadj[node + 1] - 1;
	if (jstop < jstrt) {
	    goto L400;
	}
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    nbr = adjncy[j];
	    if (!mask[nbr]) {
		goto L300;
	    }
	    ++ccsize;
	    ls[ccsize] = nbr;
	    mask[nbr] = 0;
L300:
	    ;
	}
L400:
	;
    }
/*       COMPUTE THE CURRENT LEVEL WIDTH.*/
/*       IF IT IS NONZERO, GENERATE THE NEXT LEVEL.*/
    lvsize = ccsize - lvlend;
    if (lvsize > 0) {
	goto L200;
    }
/*       RESET MASK TO ONE FOR THE NODES IN THE LEVEL STRUCTURE.*/
    xls[*nlvl + 1] = lvlend + 1;
    i__1 = ccsize;
    for (i = 1; i <= i__1; ++i) {
	node = ls[i];
	mask[node] = 1;
    }
    PetscFunctionReturn(0);
}

