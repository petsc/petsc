/* fnroot.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE FNROOT */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/********     FNROOT ..... FIND PSEUDO-PERIPHERAL NODE    *******           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*   PURPOSE - FNROOT IMPLEMENTS A MODIFIED VERSION OF THE                 7.
*/
/*      SCHEME BY GIBBS, POOLE, AND STOCKMEYER TO FIND PSEUDO-             8.
*/
/*      PERIPHERAL NODES.  IT DETERMINES SUCH A NODE FOR THE               9.
*/
/*      SECTION SUBGRAPH SPECIFIED BY MASK AND ROOT.                      10.
*/
/*                                                                        11.
*/
/*   INPUT PARAMETERS -                                                   12.
*/
/*      (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE GRAPH.          13.
*/
/*      MASK - SPECIFIES A SECTION SUBGRAPH. NODES FOR WHICH              14.
*/
/*             MASK IS ZERO ARE IGNORED BY FNROOT.                        15.
*/
/*                                                                        16.
*/
/*   UPDATED PARAMETER -                                                  17.
*/
/*      ROOT - ON INPUT, IT (ALONG WITH MASK) DEFINES THE                 18.
*/
/*             COMPONENT FOR WHICH A PSEUDO-PERIPHERAL NODE IS            19.
*/
/*             TO BE FOUND. ON OUTPUT, IT IS THE NODE OBTAINED.           20.
*/
/*                                                                        21.
*/
/*   OUTPUT PARAMETERS -                                                  22.
*/
/*      NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE             23.
*/
/*             ROOTED AT THE NODE ROOT.                                   24.
*/
/*      (XLS,LS) - THE LEVEL STRUCTURE ARRAY PAIR CONTAINING              25.
*/
/*                 THE LEVEL STRUCTURE FOUND.                             26.
*/
/*                                                                        27.
*/
/*   PROGRAM SUBROUTINES -                                                28.
*/
/*      ROOTLS.                                                           29.
*/
/*                                                                        30.
*/
/****************************************************************          31.
*/
/*                                                                        32.
*/
/*<       SUBROUTINE  FNROOT ( ROOT, XADJ, ADJNCY, MASK, NLVL, XLS, LS )     >*/
/* Subroutine */ int fnroot(int *root, int *xadj, int *adjncy, 
	int *mask, int *nlvl, int *xls, int *ls)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int ndeg, node, j, k, nabor, kstop, jstrt, kstrt, mindeg, 
	    ccsize, nunlvl;
    extern /* Subroutine */ int rootls(int *, int *, int *, 
	    int *, int *, int *, int *);

/*                                                                        
34.*/
/****************************************************************         
 35.*/
/*                                                                        
36.*/
/*<          INT ADJNCY(1), LS(1), MASK(1), XLS(1)                       >*/
/*<    >*/
/*                                                                        
41.*/
/****************************************************************         
 42.*/
/*                                                                        
43.*/
/*       ---------------------------------------------                    
44.*/
/*       DETERMINE THE LEVEL STRUCTURE ROOTED AT ROOT.                    
45.*/
/*       ---------------------------------------------                    
46.*/
/*<          CALL  ROOTLS ( ROOT, XADJ, ADJNCY, MASK, NLVL, XLS, LS )        >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    rootls(root, &xadj[1], &adjncy[1], &mask[1], nlvl, &xls[1], &ls[1]);
/*<          CCSIZE = XLS(NLVL+1) - 1                                        >*/
    ccsize = xls[*nlvl + 1] - 1;
/*<          IF ( NLVL .EQ. 1 .OR. NLVL .EQ. CCSIZE ) RETURN                 >*/
    if (*nlvl == 1 || *nlvl == ccsize) {
	return 0;
    }
/*       ----------------------------------------------------             
50.*/
/*       PICK A NODE WITH MINIMUM DEGREE FROM THE LAST LEVEL.             
51.*/
/*       ----------------------------------------------------             
52.*/
/*<   100    JSTRT = XLS(NLVL)                                               >*/
L100:
    jstrt = xls[*nlvl];
/*<          MINDEG = CCSIZE                                                 >*/
    mindeg = ccsize;
/*<          ROOT = LS(JSTRT)                                                >*/
    *root = ls[jstrt];
/*<          IF ( CCSIZE .EQ. JSTRT )  GO TO 400                             >*/
    if (ccsize == jstrt) {
	goto L400;
    }
/*<             DO 300 J = JSTRT, CCSIZE                                     >*/
    i__1 = ccsize;
    for (j = jstrt; j <= i__1; ++j) {
/*<                NODE = LS(J)                                              >*/
	node = ls[j];
/*<                NDEG = 0                                                  >*/
	ndeg = 0;
/*<                KSTRT = XADJ(NODE)                                        >*/
	kstrt = xadj[node];
/*<                KSTOP = XADJ(NODE+1) - 1                                  >*/
	kstop = xadj[node + 1] - 1;
/*<                DO 200 K = KSTRT, KSTOP                                   >*/
	i__2 = kstop;
	for (k = kstrt; k <= i__2; ++k) {
/*<                   NABOR = ADJNCY(K)                                      >*/
	    nabor = adjncy[k];
/*<                   IF ( MASK(NABOR) .GT. 0 )  NDEG = NDEG + 1             >*/
	    if (mask[nabor] > 0) {
		++ndeg;
	    }
/*<   200          CONTINUE                                                  >*/
/* L200: */
	}
/*<                IF ( NDEG .GE. MINDEG ) GO TO 300                         >*/
	if (ndeg >= mindeg) {
	    goto L300;
	}
/*<                   ROOT = NODE                                            >*/
	*root = node;
/*<                   MINDEG = NDEG                                          >*/
	mindeg = ndeg;
/*<   300       CONTINUE                                                     >*/
L300:
	;
    }
/*       ----------------------------------------                         
70.*/
/*       AND GENERATE ITS ROOTED LEVEL STRUCTURE.                         
71.*/
/*       ----------------------------------------                         
72.*/
/*<   400    CALL  ROOTLS ( ROOT, XADJ, ADJNCY, MASK, NUNLVL, XLS, LS )      >*/
L400:
    rootls(root, &xadj[1], &adjncy[1], &mask[1], &nunlvl, &xls[1], &ls[1]);
/*<          IF (NUNLVL .LE. NLVL)  RETURN                                   >*/
    if (nunlvl <= *nlvl) {
	return 0;
    }
/*<             NLVL = NUNLVL                                                >*/
    *nlvl = nunlvl;
/*<             IF ( NLVL .LT. CCSIZE )  GO TO 100                           >*/
    if (*nlvl < ccsize) {
	goto L100;
    }
/*<             RETURN                                                       >*/
    return 0;
/*<       END                                                                >*/
} /* fnroot_ */

