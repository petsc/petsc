/* genrcm.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE GENRCM */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*********   GENRCM ..... GENERAL REVERSE CUTHILL MCKEE   *******           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - GENRCM FINDS THE REVERSE CUTHILL-MCKEE                     7.
*/
/*       ORDERING FOR A GENERAL GRAPH. FOR EACH CONNECTED                  8.
*/
/*       COMPONENT IN THE GRAPH, GENRCM OBTAINS THE ORDERING               9.
*/
/*       BY CALLING THE SUBROUTINE RCM.                                   10.
*/
/*                                                                        11.
*/
/*    INPUT PARAMETERS -                                                  12.
*/
/*       NEQNS - NUMBER OF EQUATIONS                                      13.
*/
/*       (XADJ, ADJNCY) - ARRAY PAIR CONTAINING THE ADJACENCY             14.
*/
/*              STRUCTURE OF THE GRAPH OF THE MATRIX.                     15.
*/
/*                                                                        16.
*/
/*    OUTPUT PARAMETER -                                                  17.
*/
/*       PERM - VECTOR THAT CONTAINS THE RCM ORDERING.                    18.
*/
/*                                                                        19.
*/
/*    WORKING PARAMETERS -                                                20.
*/
/*       MASK - IS USED TO MARK VARIABLES THAT HAVE BEEN                  21.
*/
/*              NUMBERED DURING THE ORDERING PROCESS. IT IS               22.
*/
/*              INITIALIZED TO 1, AND SET TO ZERO AS EACH NODE            23.
*/
/*              IS NUMBERED.                                              24.
*/
/*       XLS - THE INDEX VECTOR FOR A LEVEL STRUCTURE.  THE               25.
*/
/*              LEVEL STRUCTURE IS STORED IN THE CURRENTLY                26.
*/
/*              UNUSED SPACES IN THE PERMUTATION VECTOR PERM.             27.
*/
/*                                                                        28.
*/
/*    PROGRAM SUBROUTINES -                                               29.
*/
/*       FNROOT, RCM.                                                     30.
*/
/*                                                                        31.
*/
/****************************************************************          32.
*/
/*                                                                        33.
*/
/*<       SUBROUTINE  GENRCM ( NEQNS, XADJ, ADJNCY, PERM, MASK, XLS )        >*/
/* Subroutine */ int genrcm(int *neqns, int *xadj, int *adjncy, 
	int *perm, int *mask, int *xls)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int nlvl, root, i, ccsize;
    extern /* Subroutine */ int fnroot(int *, int *, int *, 
	    int *, int *, int *, int *), rcm(int *, 
	    int *, int *, int *, int *, int *, int *);
    static int num;

/*                                                                        
35.*/
/****************************************************************         
 36.*/
/*                                                                        
37.*/
/*<          INT ADJNCY(1), MASK(1), PERM(1), XLS(1)                     >*/
/*<    >*/
/*                                                                        
41.*/
/****************************************************************         
 42.*/
/*                                                                        
43.*/
/*<          DO 100 I = 1, NEQNS                                             >*/
    /* Parameter adjustments */
    --xls;
    --mask;
    --perm;
    --adjncy;
    --xadj;

    /* Function Body */
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*<             MASK(I) = 1                                                  >*/
	mask[i] = 1;
/*<   100    CONTINUE                                                        >*/
/* L100: */
    }
/*<          NUM = 1                                                         >*/
    num = 1;
/*<          DO 200 I = 1, NEQNS                                             >*/
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*          ---------------------------------------                   
    49.*/
/*          FOR EACH MASKED CONNECTED COMPONENT ...                   
    50.*/
/*          ---------------------------------------                   
    51.*/
/*<             IF (MASK(I) .EQ. 0) GO TO 200                                >*/
	if (mask[i] == 0) {
	    goto L200;
	}
/*<                ROOT = I                                                  >*/
	root = i;
/*             -----------------------------------------              
    54.*/
/*             FIRST FIND A PSEUDO-PERIPHERAL NODE ROOT.              
    55.*/
/*             NOTE THAT THE LEVEL STRUCTURE FOUND BY                 
    56.*/
/*             FNROOT IS STORED STARTING AT PERM(NUM).                
    57.*/
/*             THEN RCM IS CALLED TO ORDER THE COMPONENT              
    58.*/
/*             USING ROOT AS THE STARTING NODE.                       
    59.*/
/*             -----------------------------------------              
    60.*/
/*<    >*/
	fnroot(&root, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &perm[
		num]);
/*<    >*/
	rcm(&root, &xadj[1], &adjncy[1], &mask[1], &perm[num], &ccsize, &xls[
		1]);
/*<                NUM = NUM + CCSIZE                                        >*/
	num += ccsize;
/*<                IF (NUM .GT. NEQNS) RETURN                                >*/
	if (num > *neqns) {
	    return 0;
	}
/*<   200    CONTINUE                                                        >*/
L200:
	;
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* genrcm_ */

