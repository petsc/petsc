/* gennd.f -- translated by f2c (version 19931217).*/

#include "petsc.h"

#undef __FUNCT__  
#define __FUNCT__ "SPARSEPACKrevrse" 
int SPARSEPACKrevrse(int *n,int *perm)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    int swap,i,m,in;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --perm;

    in = *n;
    m = *n / 2;
    i__1 = m;
    for (i = 1; i <= i__1; ++i) {
	swap = perm[i];
	perm[i] = perm[in];
	perm[in] = swap;
	--in;
    }
    PetscFunctionReturn(0);
}


/*****************************************************************/
/*********     GENND ..... GENERAL NESTED DISSECTION     *********/
/*****************************************************************/

/*    PURPOSE - SUBROUTINE GENND FINDS A NESTED DISSECTION*/
/*       ORDERING FOR A GENERAL GRAPH.*/

/*    INPUT PARAMETERS -*/
/*       NEQNS - NUMBER OF EQUATIONS.*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR.*/

/*    OUTPUT PARAMETERS -*/
/*       PERM - THE NESTED DISSECTION ORDERING.*/

/*    WORKING PARAMETERS -*/
/*       MASK - IS USED TO MASK OFF VARIABLES THAT HAVE*/
/*              BEEN NUMBERED DURING THE ORDERNG PROCESS.*/
/*       (XLS, LS) - THIS LEVEL STRUCTURE PAIR IS USED AS*/
/*              TEMPORARY STORAGE BY FNROOT.*/

/*    PROGRAM SUBROUTINES -*/
/*       FNDSEP, REVRSE.*/
/*****************************************************************/

#undef __FUNCT__  
#define __FUNCT__ "SPARSEPACKgennd" 
int SPARSEPACKgennd(int *neqns,int *xadj,int *adjncy,int *mask,int *perm,int *xls,int *ls)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    int nsep,root,i;
    EXTERN int SPARSEPACKfndsep(int*,int *,int *,int *,int *,int *,int *,int *);
    int num;

    PetscFunctionBegin;
    /* Parameter adjustments */
    --ls;
    --xls;
    --perm;
    --mask;
    --adjncy;
    --xadj;

    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
	mask[i] = 1;
    }
    num = 0;
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*           FOR EACH MASKED COMPONENT ...*/
L200:
	if (!mask[i]) {
	    goto L300;
	}
	root = i;
/*              FIND A SEPARATOR AND NUMBER THE NODES NEXT.*/
	SPARSEPACKfndsep(&root,&xadj[1],&adjncy[1],&mask[1],&nsep,&perm[num + 1],
		&xls[1],&ls[1]);
	num += nsep;
	if (num >= *neqns) {
	    goto L400;
	}
	goto L200;
L300:
	;
    }
/*        SINCE SEPARATORS FOUND FIRST SHOULD BE ORDERED*/
/*        LAST, ROUTINE REVRSE IS CALLED TO ADJUST THE*/
/*        ORDERING VECTOR.*/
L400:
    SPARSEPACKrevrse(neqns,&perm[1]);
    PetscFunctionReturn(0);
}
