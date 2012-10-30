
/* qmdrch.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <../src/mat/order/order.h>

/*****************************************************************/
/**********     QMDRCH ..... QUOT MIN DEG REACH SET    ***********/
/*****************************************************************/

/*    PURPOSE - THIS SUBROUTINE DETERMINES THE REACHABLE SET OF*/
/*       A NODE THROUGH A GIVEN SUBSET.  THE ADJACENCY STRUCTURE*/
/*       IS ASSUMED TO BE STORED IN A QUOTIENT GRAPH FORMAT.*/

/*    INPUT PARAMETERS -*/
/*       ROOT - THE GIVEN NODE NOT IN THE SUBSET.*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.*/
/*       DEG - THE DEGREE VECTOR.  DEG(I) LT 0 MEANS THE NODE*/
/*              BELONGS TO THE GIVEN SUBSET.*/

/*    OUTPUT PARAMETERS -*/
/*       (RCHSZE, RCHSET) - THE REACHABLE SET.*/
/*       (NHDSZE, NBRHD) - THE NEIGHBORHOOD SET.*/

/*    UPDATED PARAMETERS -*/
/*       MARKER - THE MARKER VECTOR FOR REACH AND NBRHD SETS.*/
/*              GT 0 MEANS THE NODE IS IN REACH SET.*/
/*              LT 0 MEANS THE NODE HAS BEEN MERGED WITH*/
/*              OTHERS IN THE QUOTIENT OR IT IS IN NBRHD SET.*/
/*****************************************************************/
#undef __FUNCT__
#define __FUNCT__ "SPARSEPACKqmdrch"
PetscErrorCode SPARSEPACKqmdrch(const PetscInt *root,const PetscInt *xadj,const PetscInt *adjncy,
	PetscInt *deg, PetscInt *marker, PetscInt *rchsze, PetscInt *rchset,
	PetscInt *nhdsze, PetscInt *nbrhd)
{
    /* System generated locals */
    PetscInt i__1, i__2;

    /* Local variables */
    PetscInt node, i, j, nabor, istop, jstop, istrt, jstrt;

/*       LOOP THROUGH THE NEIGHBORS OF ROOT IN THE*/
/*       QUOTIENT GRAPH.*/


    PetscFunctionBegin;
    /* Parameter adjustments */
    --nbrhd;
    --rchset;
    --marker;
    --deg;
    --adjncy;
    --xadj;

    *nhdsze = 0;
    *rchsze = 0;
    istrt = xadj[*root];
    istop = xadj[*root + 1] - 1;
    if (istop < istrt) {
	PetscFunctionReturn(0);
    }
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
	nabor = adjncy[i];
	if (!nabor) {
	    PetscFunctionReturn(0);
	}
	if (marker[nabor] != 0) {
	    goto L600;
	}
	if (deg[nabor] < 0) {
	    goto L200;
	}
/*                   INCLUDE NABOR INTO THE REACHABLE SET.*/
	++(*rchsze);
	rchset[*rchsze] = nabor;
	marker[nabor] = 1;
	goto L600;
/*                NABOR HAS BEEN ELIMINATED. FIND NODES*/
/*                REACHABLE FROM IT.*/
L200:
	marker[nabor] = -1;
	++(*nhdsze);
	nbrhd[*nhdsze] = nabor;
L300:
	jstrt = xadj[nabor];
	jstop = xadj[nabor + 1] - 1;
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
	    node = adjncy[j];
	    nabor = -node;
	    if (node < 0) {
		goto L300;
	    } else if (!node) {
		goto L600;
	    } else {
		goto L400;
	    }
L400:
	    if (marker[node] != 0) {
		goto L500;
	    }
	    ++(*rchsze);
	    rchset[*rchsze] = node;
	    marker[node] = 1;
L500:
	    ;
	}
L600:
	;
    }
    PetscFunctionReturn(0);
}

