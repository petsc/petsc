/* gen1wd.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE GEN1WD */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/***********     GEN1WD ..... GENERAL ONE-WAY DISSECTION  *******           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - GEN1WD FINDS A ONE-WAY DISSECTION PARTITIONING             7.
*/
/*       FOR A GENERAL GRAPH.  FN1WD IS USED FOR EACH CONNECTED            8.
*/
/*       COMPONENT.                                                        9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       NEQNS - NUMBER OF EQUATIONS.                                     12.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.                   13.
*/
/*                                                                        14.
*/
/*    OUTPUT PARAMETERS -                                                 15.
*/
/*       (NBLKS, XBLK) - THE PARTITIONING FOUND.                          16.
*/
/*       PERM - THE ONE-WAY DISSECTION ORDERING.                          17.
*/
/*                                                                        18.
*/
/*    WORKING VECTORS -                                                   19.
*/
/*       MASK - IS USED TO MARK VARIABLES THAT HAVE                       20.
*/
/*              BEEN NUMBERED DURING THE ORDERING PROCESS.                21.
*/
/*       (XLS, LS) - LEVEL STRUCTURE USED BY ROOTLS.                      22.
*/
/*                                                                        23.
*/
/*    PROGRAM SUBROUTINES -                                               24.
*/
/*       FN1WD, REVRSE, ROOTLS.                                           25.
*/
/*                                                                        26.
*/
/****************************************************************          27.
*/
/*                                                                        28.
*/
/*<    >*/
/* Subroutine */ int gen1wd(int *neqns, int *xadj, int *adjncy, 
	int *mask, int *nblks, int *xblk, int *perm, int *
	xls, int *ls)
{
    /* System generated locals */
    int i__1, i__2, i__3;

    /* Local variables */
    static int node, nsep, lnum, nlvl, root;
    extern /* Subroutine */ int fn1wd(int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *);
    static int i, j, k, ccsize;
    extern /* Subroutine */ int revrse(int *, int *), rootls(
	    int *, int *, int *, int *, int *, int *, 
	    int *);
    static int num;

/*                                                                        
31.*/
/****************************************************************         
 32.*/

/*<          int neqns >*/
/*<    >*/
/*<    >*/
/*                                                                        
39.*/
/****************************************************************         
 40.*/
/*                                                                        
41.*/
/*<          DO 100 I = 1, NEQNS                                             >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --perm;
    --xblk;
    --mask;
    --xadj;
    --adjncy;

    /* Function Body */
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*<             MASK(I) = 1                                                  >*/
	mask[i] = 1;
/*<   100    CONTINUE                                                        >*/
/* L100: */
    }
/*<          NBLKS = 0                                                       >*/
    *nblks = 0;
/*<          NUM   = 0                                                       >*/
    num = 0;
/*<          DO 400 I = 1, NEQNS                                             >*/
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*<             IF ( MASK(I) .EQ. 0 )  GO TO 400                             >*/
	if (mask[i] == 0) {
	    goto L400;
	}
/*             --------------------------------------------           
    49.*/
/*             FIND A ONE-WAY DISSECTOR FOR EACH COMPONENT.           
    50.*/
/*             --------------------------------------------           
    51.*/
/*<                ROOT = I                                                  >*/
	root = i;
/*<    >*/
	fn1wd(&root, &xadj[1], &adjncy[1], &mask[1], &nsep, &perm[num + 1], &
		nlvl, &xls[1], &ls[1]);
/*<                NUM = NUM + NSEP                                          >*/
	num += nsep;
/*<                NBLKS = NBLKS + 1                                         >*/
	++(*nblks);
/*<                XBLK(NBLKS) = NEQNS - NUM + 1                             >*/
	xblk[*nblks] = *neqns - num + 1;
/*<                CCSIZE = XLS(NLVL+1) - 1                                  >*/
	ccsize = xls[nlvl + 1] - 1;
/*             ----------------------------------------------         
    59.*/
/*             NUMBER THE REMAINING NODES IN THE COMPONENT.           
    60.*/
/*             EACH COMPONENT IN THE REMAINING SUBGRAPH FORMS         
    61.*/
/*             A NEW BLOCK IN THE PARTITIONING.                       
    62.*/
/*             ----------------------------------------------         
    63.*/
/*<                DO 300 J = 1, CCSIZE                                      >*/
	i__2 = ccsize;
	for (j = 1; j <= i__2; ++j) {
/*<                   NODE = LS(J)                                           >*/
	    node = ls[j];
/*<                   IF ( MASK(NODE) .EQ. 0 )  GO TO 300                    >*/
	    if (mask[node] == 0) {
		goto L300;
	    }
/*<    >*/
	    rootls(&node, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &
		    perm[num + 1]);
/*<                      LNUM = NUM + 1                                      >*/
	    lnum = num + 1;
/*<                      NUM  = NUM + XLS(NLVL+1) - 1                        >*/
	    num = num + xls[nlvl + 1] - 1;
/*<                      NBLKS = NBLKS + 1                                   >*/
	    ++(*nblks);
/*<                      XBLK(NBLKS) = NEQNS - NUM + 1                       >*/
	    xblk[*nblks] = *neqns - num + 1;
/*<                      DO 200 K = LNUM, NUM                                >*/
	    i__3 = num;
	    for (k = lnum; k <= i__3; ++k) {
/*<                         NODE = PERM(K)                                   >*/
		node = perm[k];
/*<                         MASK(NODE) = 0                                   >*/
		mask[node] = 0;
/*<   200                CONTINUE                                            >*/
/* L200: */
	    }
/*<                      IF ( NUM .GT. NEQNS )  GO TO 500                    >*/
	    if (num > *neqns) {
		goto L500;
	    }
/*<   300          CONTINUE                                                  >*/
L300:
	    ;
	}
/*<   400    CONTINUE                                                        >*/
L400:
	;
    }
/*       ----------------------------------------------------             
80.*/
/*       SINCE DISSECTORS FOUND FIRST SHOULD BE ORDERED LAST,             
81.*/
/*       ROUTINE REVRSE IS CALLED TO ADJUST THE ORDERING                  
82.*/
/*       VECTOR, AND THE BLOCK INDEX VECTOR.                              
83.*/
/*       ----------------------------------------------------             
84.*/
/*<   500    CALL  REVRSE ( NEQNS, PERM )                                    >*/
L500:
    revrse(neqns, &perm[1]);
/*<          CALL  REVRSE ( NBLKS, XBLK )                                    >*/
    revrse(nblks, &xblk[1]);
/*<          XBLK(NBLKS+1) = NEQNS + 1                                       >*/
    xblk[*nblks + 1] = *neqns + 1;
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* gen1wd_ */

