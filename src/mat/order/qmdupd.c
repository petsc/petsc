/* qmdupd.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE QMDUPD */
/*****************************************************************          1.
*/
/*****************************************************************          2.
*/
/***********     QMDUPD ..... QUOT MIN DEG UPDATE      ***********          3.
*/
/*****************************************************************          4.
*/
/*****************************************************************          5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS ROUTINE PERFORMS DEGREE UPDATE FOR A SET              7.
*/
/*       OF NODES IN THE MINIMUM DEGREE ALGORITHM.                         8.
*/
/*                                                                         9.
*/
/*    INPUT PARAMETERS -                                                  10.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                        11.
*/
/*       (NLIST, LIST) - THE LIST OF NODES WHOSE DEGREE HAS TO            12.
*/
/*              BE UPDATED.                                               13.
*/
/*                                                                        14.
*/
/*    UPDATED PARAMETERS -                                                15.
*/
/*       DEG - THE DEGREE VECTOR.                                         16.
*/
/*       QSIZE - SIZE OF INDISTINGUISHABLE SUPERNODES.                    17.
*/
/*       QLINK - LINKED LIST FOR INDISTINGUISHABLE NODES.                 18.
*/
/*       MARKER - USED TO MARK THOSE NODES IN REACH/NBRHD SETS.           19.
*/
/*                                                                        20.
*/
/*    WORKING PARAMETERS -                                                21.
*/
/*       RCHSET - THE REACHABLE SET.                                      22.
*/
/*       NBRHD -  THE NEIGHBORHOOD SET.                                   23.
*/
/*                                                                        24.
*/
/*    PROGRAM SUBROUTINES -                                               25.
*/
/*       QMDMRG.                                                          26.
*/
/*                                                                        27.
*/
/*****************************************************************         28.
*/
/*                                                                        29.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define qmdupd_ QMDUPD
#elif !defined(FORTRANUNDERSCORE)
#define qmdupd_ qmdupd
#endif
/* Subroutine */ int qmdupd_(int *xadj, int *adjncy, int *nlist, 
	int *list, int *deg, int *qsize, int *qlink, int *
	marker, int *rchset, int *nbrhd)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int inhd, irch, node, mark, j, inode, nabor, jstop, jstrt, il;
    extern /* Subroutine */ int qmdrch_(int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *),
	     qmdmrg_(int *, int *, int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *);
    static int nhdsze, rchsze, deg0, deg1;

/*                                                                        
32.*/
/*****************************************************************        
 33.*/
/*                                                                        
34.*/
/*<    >*/
/*<    >*/
/*                                                                        
40.*/
/*****************************************************************        
 41.*/
/*                                                                        
42.*/
/*       ------------------------------------------------                 
43.*/
/*       FIND ALL ELIMINATED SUPERNODES THAT ARE ADJACENT                 
44.*/
/*       TO SOME NODES IN THE GIVEN LIST. PUT THEM INTO                   
45.*/
/*       (NHDSZE, NBRHD). DEG0 CONTAINS THE NUMBER OF                     
46.*/
/*       NODES IN THE LIST.                                               
47.*/
/*       ------------------------------------------------                 
48.*/
/*<          IF ( NLIST .LE. 0 )  RETURN                                     >*/
    /* Parameter adjustments */
    --nbrhd;
    --rchset;
    --marker;
    --qlink;
    --qsize;
    --deg;
    --list;
    --adjncy;
    --xadj;

    /* Function Body */
    if (*nlist <= 0) {
	return 0;
    }
/*<          DEG0 = 0                                                        >*/
    deg0 = 0;
/*<          NHDSZE = 0                                                      >*/
    nhdsze = 0;
/*<          DO 200 IL = 1, NLIST                                            >*/
    i__1 = *nlist;
    for (il = 1; il <= i__1; ++il) {
/*<             NODE = LIST(IL)                                              >*/
	node = list[il];
/*<             DEG0 = DEG0 + QSIZE(NODE)                                    >*/
	deg0 += qsize[node];
/*<             JSTRT = XADJ(NODE)                                           >*/
	jstrt = xadj[node];
/*<             JSTOP = XADJ(NODE+1) - 1                                     >*/
	jstop = xadj[node + 1] - 1;
/*<             DO 100 J = JSTRT, JSTOP                                      >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                NABOR = ADJNCY(J)                                         >*/
	    nabor = adjncy[j];
/*<    >*/
	    if (marker[nabor] != 0 || deg[nabor] >= 0) {
		goto L100;
	    }
/*<                   MARKER(NABOR) = - 1                                    >*/
	    marker[nabor] = -1;
/*<                   NHDSZE = NHDSZE + 1                                    >*/
	    ++nhdsze;
/*<                   NBRHD(NHDSZE) = NABOR                                  >*/
	    nbrhd[nhdsze] = nabor;
/*<   100       CONTINUE                                                     >*/
L100:
	    ;
	}
/*<   200    CONTINUE                                                        >*/
/* L200: */
    }
/*       --------------------------------------------                     
66.*/
/*       MERGE INDISTINGUISHABLE NODES IN THE LIST BY                     
67.*/
/*       CALLING THE SUBROUTINE QMDMRG.                                   
68.*/
/*       --------------------------------------------                     
69.*/
/*<    >*/
    if (nhdsze > 0) {
	qmdmrg_(&xadj[1], &adjncy[1], &deg[1], &qsize[1], &qlink[1], &marker[
		1], &deg0, &nhdsze, &nbrhd[1], &rchset[1], &nbrhd[nhdsze + 1])
		;
    }
/*       ----------------------------------------------------             
74.*/
/*       FIND THE NEW DEGREES OF THE NODES THAT HAVE NOT BEEN             
75.*/
/*       MERGED.                                                          
76.*/
/*       ----------------------------------------------------             
77.*/
/*<          DO 600 IL = 1, NLIST                                            >*/
    i__1 = *nlist;
    for (il = 1; il <= i__1; ++il) {
/*<             NODE = LIST(IL)                                              >*/
	node = list[il];
/*<             MARK = MARKER(NODE)                                          >*/
	mark = marker[node];
/*<             IF ( MARK .GT. 1  .OR.  MARK .LT. 0 )  GO TO 600             >*/
	if (mark > 1 || mark < 0) {
	    goto L600;
	}
/*<                MARKER(NODE) = 2                                          >*/
	marker[node] = 2;
/*<    >*/
	qmdrch_(&node, &xadj[1], &adjncy[1], &deg[1], &marker[1], &rchsze, &
		rchset[1], &nhdsze, &nbrhd[1]);
/*<                DEG1 = DEG0                                               >*/
	deg1 = deg0;
/*<                IF ( RCHSZE .LE. 0 )  GO TO 400                           >*/
	if (rchsze <= 0) {
	    goto L400;
	}
/*<                   DO 300 IRCH = 1, RCHSZE                                >*/
	i__2 = rchsze;
	for (irch = 1; irch <= i__2; ++irch) {
/*<                      INODE = RCHSET(IRCH)                                >*/
	    inode = rchset[irch];
/*<                      DEG1 = DEG1 + QSIZE(INODE)                          >*/
	    deg1 += qsize[inode];
/*<                      MARKER(INODE) = 0                                   >*/
	    marker[inode] = 0;
/*<   300             CONTINUE                                               >*/
/* L300: */
	}
/*<   400          DEG(NODE) = DEG1 - 1                                      >*/
L400:
	deg[node] = deg1 - 1;
/*<                IF ( NHDSZE .LE. 0 )  GO TO 600                           >*/
	if (nhdsze <= 0) {
	    goto L600;
	}
/*<                   DO 500 INHD = 1, NHDSZE                                >*/
	i__2 = nhdsze;
	for (inhd = 1; inhd <= i__2; ++inhd) {
/*<                      INODE = NBRHD(INHD)                                 >*/
	    inode = nbrhd[inhd];
/*<                      MARKER(INODE) = 0                                   >*/
	    marker[inode] = 0;
/*<   500             CONTINUE                                               >*/
/* L500: */
	}
/*<   600    CONTINUE                                                        >*/
L600:
	;
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* qmdupd_ */

