/* rootls.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE ROOTLS */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*********     ROOTLS ..... ROOTED LEVEL STRUCTURE      *********           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - ROOTLS GENERATES THE LEVEL STRUCTURE ROOTED                7.
*/
/*       AT THE INPUT NODE CALLED ROOT. ONLY THOSE NODES FOR               8.
*/
/*       WHICH MASK IS NONZERO WILL BE CONSIDERED.                         9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       ROOT - THE NODE AT WHICH THE LEVEL STRUCTURE IS TO               12.
*/
/*              BE ROOTED.                                                13.
*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR THE                14.
*/
/*              GIVEN GRAPH.                                              15.
*/
/*       MASK - IS USED TO SPECIFY A SECTION SUBGRAPH. NODES              16.
*/
/*              WITH MASK(I)=0 ARE IGNORED.                               17.
*/
/*                                                                        18.
*/
/*    OUTPUT PARAMETERS -                                                 19.
*/
/*       NLVL - IS THE NUMBER OF LEVELS IN THE LEVEL STRUCTURE.           20.
*/
/*       (XLS, LS) - ARRAY PAIR FOR THE ROOTED LEVEL STRUCTURE.           21.
*/
/*                                                                        22.
*/
/****************************************************************          23.
*/
/*                                                                        24.
*/
/*<       SUBROUTINE  ROOTLS ( ROOT, XADJ, ADJNCY, MASK, NLVL, XLS, LS )     >*/
/* Subroutine */ int rootls(int *root, int *xadj, int *adjncy, 
	int *mask, int *nlvl, int *xls, int *ls)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int node, i, j, jstop, jstrt, lbegin, ccsize, lvlend, lvsize, 
	    nbr;

/*                                                                        
26.*/
/****************************************************************         
 27.*/
/*                                                                        
28.*/
/*<          INT ADJNCY(1), LS(1), MASK(1), XLS(1)                       >*/
/*<    >*/
/*                                                                        
33.*/
/****************************************************************         
 34.*/
/*                                                                        
35.*/
/*       ------------------                                               
36.*/
/*       INITIALIZATION ...                                               
37.*/
/*       ------------------                                               
38.*/
/*<          MASK(ROOT) = 0                                                  >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    mask[*root] = 0;
/*<          LS(1) = ROOT                                                    >*/
    ls[1] = *root;
/*<          NLVL = 0                                                        >*/
    *nlvl = 0;
/*<          LVLEND = 0                                                      >*/
    lvlend = 0;
/*<          CCSIZE = 1                                                      >*/
    ccsize = 1;
/*       -----------------------------------------------------            
44.*/
/*       LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT            
45.*/
/*       LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.               
46.*/
/*       -----------------------------------------------------            
47.*/
/*<   200    LBEGIN = LVLEND + 1                                             >*/
L200:
    lbegin = lvlend + 1;
/*<          LVLEND = CCSIZE                                                 >*/
    lvlend = ccsize;
/*<          NLVL = NLVL + 1                                                 >*/
    ++(*nlvl);
/*<          XLS(NLVL) = LBEGIN                                              >*/
    xls[*nlvl] = lbegin;
/*       -------------------------------------------------                
52.*/
/*       GENERATE THE NEXT LEVEL BY FINDING ALL THE MASKED                
53.*/
/*       NEIGHBORS OF NODES IN THE CURRENT LEVEL.                         
54.*/
/*       -------------------------------------------------                
55.*/
/*<          DO 400 I = LBEGIN, LVLEND                                       >*/
    i__1 = lvlend;
    for (i = lbegin; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             JSTRT = XADJ(NODE)                                           >*/
	jstrt = xadj[node];
/*<             JSTOP = XADJ(NODE + 1) - 1                                   >*/
	jstop = xadj[node + 1] - 1;
/*<             IF ( JSTOP .LT. JSTRT )  GO TO 400                           >*/
	if (jstop < jstrt) {
	    goto L400;
	}
/*<                DO 300 J = JSTRT, JSTOP                                   >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                   NBR = ADJNCY(J)                                        >*/
	    nbr = adjncy[j];
/*<                   IF (MASK(NBR) .EQ. 0) GO TO 300                        >*/
	    if (mask[nbr] == 0) {
		goto L300;
	    }
/*<                      CCSIZE = CCSIZE + 1                                 >*/
	    ++ccsize;
/*<                      LS(CCSIZE) = NBR                                    >*/
	    ls[ccsize] = nbr;
/*<                      MASK(NBR) = 0                                       >*/
	    mask[nbr] = 0;
/*<   300          CONTINUE                                                  >*/
L300:
	    ;
	}
/*<   400    CONTINUE                                                        >*/
L400:
	;
    }
/*       ------------------------------------------                       
69.*/
/*       COMPUTE THE CURRENT LEVEL WIDTH.                                 
70.*/
/*       IF IT IS NONZERO, GENERATE THE NEXT LEVEL.                       
71.*/
/*       ------------------------------------------                       
72.*/
/*<          LVSIZE = CCSIZE - LVLEND                                        >*/
    lvsize = ccsize - lvlend;
/*<          IF (LVSIZE .GT. 0 ) GO TO 200                                   >*/
    if (lvsize > 0) {
	goto L200;
    }
/*       -------------------------------------------------------          
75.*/
/*       RESET MASK TO ONE FOR THE NODES IN THE LEVEL STRUCTURE.          
76.*/
/*       -------------------------------------------------------          
77.*/
/*<          XLS(NLVL+1) = LVLEND + 1                                        >*/
    xls[*nlvl + 1] = lvlend + 1;
/*<          DO 500 I = 1, CCSIZE                                            >*/
    i__1 = ccsize;
    for (i = 1; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             MASK(NODE) = 1                                               >*/
	mask[node] = 1;
/*<   500    CONTINUE                                                        >*/
/* L500: */
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* rootls_ */

