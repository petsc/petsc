/* degree.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE DEGREE */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*********     DEGREE ..... DEGREE IN MASKED COMPONENT   ********           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS ROUTINE COMPUTES THE DEGREES OF THE NODES             7.
*/
/*       IN THE CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT.            8.
*/
/*       NODES FOR WHICH MASK IS ZERO ARE IGNORED.                         9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETER -                                                   11.
*/
/*       ROOT - IS THE INPUT NODE THAT DEFINES THE COMPONENT.             12.
*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR.                       13.
*/
/*       MASK - SPECIFIES A SECTION SUBGRAPH.                             14.
*/
/*                                                                        15.
*/
/*    OUTPUT PARAMETERS -                                                 16.
*/
/*       DEG - ARRAY CONTAINING THE DEGREES OF THE NODES IN               17.
*/
/*             THE COMPONENT.                                             18.
*/
/*       CCSIZE-SIZE OF THE COMPONENT SPECIFED BY MASK AND ROOT           19.
*/
/*                                                                        20.
*/
/*    WORKING PARAMETER -                                                 21.
*/
/*       LS - A TEMPORARY VECTOR USED TO STORE THE NODES OF THE           22.
*/
/*              COMPONENT LEVEL BY LEVEL.                                 23.
*/
/*                                                                        24.
*/
/****************************************************************          25.
*/
/*                                                                        26.
*/
/*<    >*/
/* Subroutine */ int degree(int *root, int *xadj, int *adjncy, 
	int *mask, int *deg, int *ccsize, int *ls)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int ideg, node, i, j, jstop, jstrt, lbegin, lvlend, lvsize, 
	    nbr;

/*                                                                        
29.*/
/****************************************************************         
 30.*/
/*                                                                        
31.*/
/*<          INT ADJNCY(1), DEG(1), LS(1), MASK(1)                       >*/
/*<    >*/
/*                                                                        
35.*/
/****************************************************************         
 36.*/
/*                                                                        
37.*/
/*       -------------------------------------------------                
38.*/
/*       INITIALIZATION ...                                               
39.*/
/*       THE ARRAY XADJ IS USED AS A TEMPORARY MARKER TO                  
40.*/
/*       INDICATE WHICH NODES HAVE BEEN CONSIDERED SO FAR.                
41.*/
/*       -------------------------------------------------                
42.*/
/*<          LS(1) = ROOT                                                    >*/
    /* Parameter adjustments */
    --ls;
    --deg;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    ls[1] = *root;
/*<          XADJ(ROOT) = -XADJ(ROOT)                                        >*/
    xadj[*root] = -xadj[*root];
/*<          LVLEND = 0                                                      >*/
    lvlend = 0;
/*<          CCSIZE = 1                                                      >*/
    *ccsize = 1;
/*       -----------------------------------------------------            
47.*/
/*       LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT            
48.*/
/*       LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.               
49.*/
/*       -----------------------------------------------------            
50.*/
/*<   100    LBEGIN = LVLEND + 1                                             >*/
L100:
    lbegin = lvlend + 1;
/*<          LVLEND = CCSIZE                                                 >*/
    lvlend = *ccsize;
/*       -----------------------------------------------                  
53.*/
/*       FIND THE DEGREES OF NODES IN THE CURRENT LEVEL,                  
54.*/
/*       AND AT THE SAME TIME, GENERATE THE NEXT LEVEL.                   
55.*/
/*       -----------------------------------------------                  
56.*/
/*<          DO 400 I = LBEGIN, LVLEND                                       >*/
    i__1 = lvlend;
    for (i = lbegin; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             JSTRT = -XADJ(NODE)                                          >*/
	jstrt = -xadj[node];
/*<             JSTOP = IABS(XADJ(NODE + 1)) - 1                             >*/
	jstop = (i__2 = xadj[node + 1], (int)abs(i__2)) - 1;
/*<             IDEG = 0                                                     >*/
	ideg = 0;
/*<             IF ( JSTOP .LT. JSTRT ) GO TO 300                            >*/
	if (jstop < jstrt) {
	    goto L300;
	}
/*<                DO 200 J = JSTRT, JSTOP                                   >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                   NBR = ADJNCY(J)                                        >*/
	    nbr = adjncy[j];
/*<                   IF ( MASK(NBR) .EQ. 0 )  GO TO  200                    >*/
	    if (mask[nbr] == 0) {
		goto L200;
	    }
/*<                      IDEG = IDEG + 1                                     >*/
	    ++ideg;
/*<                      IF ( XADJ(NBR) .LT. 0 ) GO TO 200                   >*/
	    if (xadj[nbr] < 0) {
		goto L200;
	    }
/*<                         XADJ(NBR) = -XADJ(NBR)                           >*/
	    xadj[nbr] = -xadj[nbr];
/*<                         CCSIZE = CCSIZE + 1                              >*/
	    ++(*ccsize);
/*<                         LS(CCSIZE) = NBR                                 >*/
	    ls[*ccsize] = nbr;
/*<   200          CONTINUE                                                  >*/
L200:
	    ;
	}
/*<   300       DEG(NODE) = IDEG                                             >*/
L300:
	deg[node] = ideg;
/*<   400    CONTINUE                                                        >*/
/* L400: */
    }
/*       ------------------------------------------                       
74.*/
/*       COMPUTE THE CURRENT LEVEL WIDTH.                                 
75.*/
/*       IF IT IS NONZERO , GENERATE ANOTHER LEVEL.                       
76.*/
/*       ------------------------------------------                       
77.*/
/*<          LVSIZE = CCSIZE - LVLEND                                        >*/
    lvsize = *ccsize - lvlend;
/*<          IF ( LVSIZE .GT. 0 ) GO TO 100                                  >*/
    if (lvsize > 0) {
	goto L100;
    }
/*       ------------------------------------------                       
80.*/
/*       RESET XADJ TO ITS CORRECT SIGN AND RETURN.                       
81.*/
/*       ------------------------------------------                       
82.*/
/*<          DO 500 I = 1, CCSIZE                                            >*/
    i__1 = *ccsize;
    for (i = 1; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             XADJ(NODE) = -XADJ(NODE)                                     >*/
	xadj[node] = -xadj[node];
/*<   500    CONTINUE                                                        >*/
/* L500: */
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* degree_ */

