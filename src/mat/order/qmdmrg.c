/* qmdmrg.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE QMDMRG */
/*****************************************************************          1.
*/
/*****************************************************************          2.
*/
/***********     QMDMRG ..... QUOT MIN DEG MERGE       ***********          3.
*/
/*****************************************************************          4.
*/
/*****************************************************************          5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS ROUTINE MERGES INDISTINGUISHABLE NODES IN             7.
*/
/*              THE MINIMUM DEGREE ORDERING ALGORITHM.                     8.
*/
/*              IT ALSO COMPUTES THE NEW DEGREES OF THESE                  9.
*/
/*              NEW SUPERNODES.                                           10.
*/
/*                                                                        11.
*/
/*    INPUT PARAMETERS -                                                  12.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                        13.
*/
/*       DEG0 - THE NUMBER OF NODES IN THE GIVEN SET.                     14.
*/
/*       (NHDSZE, NBRHD) - THE SET OF ELIMINATED SUPERNODES               15.
*/
/*              ADJACENT TO SOME NODES IN THE SET.                        16.
*/
/*                                                                        17.
*/
/*    UPDATED PARAMETERS -                                                18.
*/
/*       DEG - THE DEGREE VECTOR.                                         19.
*/
/*       QSIZE - SIZE OF INDISTINGUISHABLE NODES.                         20.
*/
/*       QLINK - LINKED LIST FOR INDISTINGUISHABLE NODES.                 21.
*/
/*       MARKER - THE GIVEN SET IS GIVEN BY THOSE NODES WITH              22.
*/
/*              MARKER VALUE SET TO 1.  THOSE NODES WITH DEGREE           23.
*/
/*              UPDATED WILL HAVE MARKER VALUE SET TO 2.                  24.
*/
/*                                                                        25.
*/
/*    WORKING PARAMETERS -                                                26.
*/
/*       RCHSET - THE REACHABLE SET.                                      27.
*/
/*       OVRLP -  TEMP VECTOR TO STORE THE INTERSECTION OF TWO            28.
*/
/*              REACHABLE SETS.                                           29.
*/
/*                                                                        30.
*/
/*****************************************************************         31.
*/
/*                                                                        32.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define qmdmrg_ QMDMRG
#elif !defined(FORTRANUNDERSCORE)
#define qmdmrg_ qmdmrg
#endif
/* Subroutine */ int qmdmrg_(int *xadj, int *adjncy, int *deg, 
	int *qsize, int *qlink, int *marker, int *deg0, 
	int *nhdsze, int *nbrhd, int *rchset, int *ovrlp)
{
    /* System generated locals */
    int i__1, i__2, i__3;

    /* Local variables */
    static int head, inhd, irch, node, mark, link, root, j, lnode, nabor, 
	    jstop, jstrt, rchsze, mrgsze, novrlp, iov, deg1;

/*                                                                        
36.*/
/*****************************************************************        
 37.*/
/*                                                                        
38.*/
/*<    >*/
/*<    >*/
/*                                                                        
44.*/
/*****************************************************************        
 45.*/
/*                                                                        
46.*/
/*       ------------------                                               
47.*/
/*       INITIALIZATION ...                                               
48.*/
/*       ------------------                                               
49.*/
/*<          IF ( NHDSZE .LE. 0 )  RETURN                                    >*/
    /* Parameter adjustments */
    --ovrlp;
    --rchset;
    --nbrhd;
    --marker;
    --qlink;
    --qsize;
    --deg;
    --adjncy;
    --xadj;

    /* Function Body */
    if (*nhdsze <= 0) {
	return 0;
    }
/*<          DO 100 INHD = 1, NHDSZE                                         >*/
    i__1 = *nhdsze;
    for (inhd = 1; inhd <= i__1; ++inhd) {
/*<             ROOT = NBRHD(INHD)                                           >*/
	root = nbrhd[inhd];
/*<             MARKER(ROOT) = 0                                             >*/
	marker[root] = 0;
/*<   100    CONTINUE                                                        >*/
/* L100: */
    }
/*       -------------------------------------------------                
55.*/
/*       LOOP THROUGH EACH ELIMINATED SUPERNODE IN THE SET                
56.*/
/*       (NHDSZE, NBRHD).                                                 
57.*/
/*       -------------------------------------------------                
58.*/
/*<          DO 1400 INHD = 1, NHDSZE                                        >*/
    i__1 = *nhdsze;
    for (inhd = 1; inhd <= i__1; ++inhd) {
/*<             ROOT = NBRHD(INHD)                                           >*/
	root = nbrhd[inhd];
/*<             MARKER(ROOT) = - 1                                           >*/
	marker[root] = -1;
/*<             RCHSZE = 0                                                   >*/
	rchsze = 0;
/*<             NOVRLP = 0                                                   >*/
	novrlp = 0;
/*<             DEG1   = 0                                                   >*/
	deg1 = 0;
/*<   200       JSTRT  = XADJ(ROOT)                                          >*/
L200:
	jstrt = xadj[root];
/*<             JSTOP  = XADJ(ROOT+1) - 1                                    >*/
	jstop = xadj[root + 1] - 1;
/*          ----------------------------------------------            
    67.*/
/*          DETERMINE THE REACHABLE SET AND ITS INTERSECT-            
    68.*/
/*          ION WITH THE INPUT REACHABLE SET.                         
    69.*/
/*          ----------------------------------------------            
    70.*/
/*<             DO 600 J = JSTRT, JSTOP                                      >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                NABOR = ADJNCY(J)                                         >*/
	    nabor = adjncy[j];
/*<                ROOT  = - NABOR                                           >*/
	    root = -nabor;
/*<                IF (NABOR)  200, 700, 300                                 >*/
	    if (nabor < 0) {
		goto L200;
	    } else if (nabor == 0) {
		goto L700;
	    } else {
		goto L300;
	    }
/*                                                                
        75.*/
/*<   300          MARK = MARKER(NABOR)                                      >*/
L300:
	    mark = marker[nabor];
/*<                IF ( MARK ) 600, 400, 500                                 >*/
	    if (mark < 0) {
		goto L600;
	    } else if (mark == 0) {
		goto L400;
	    } else {
		goto L500;
	    }
/*<   400             RCHSZE = RCHSZE + 1                                    >*/
L400:
	    ++rchsze;
/*<                   RCHSET(RCHSZE) = NABOR                                 >*/
	    rchset[rchsze] = nabor;
/*<                   DEG1 = DEG1 + QSIZE(NABOR)                             >*/
	    deg1 += qsize[nabor];
/*<                   MARKER(NABOR) = 1                                      >*/
	    marker[nabor] = 1;
/*<                   GOTO 600                                               >*/
	    goto L600;
/*<   500          IF ( MARK .GT. 1 )  GOTO 600                              >*/
L500:
	    if (mark > 1) {
		goto L600;
	    }
/*<                   NOVRLP = NOVRLP + 1                                    >*/
	    ++novrlp;
/*<                   OVRLP(NOVRLP) = NABOR                                  >*/
	    ovrlp[novrlp] = nabor;
/*<                   MARKER(NABOR) = 2                                      >*/
	    marker[nabor] = 2;
/*<   600       CONTINUE                                                     >*/
L600:
	    ;
	}
/*          --------------------------------------------              
    88.*/
/*          FROM THE OVERLAPPED SET, DETERMINE THE NODES              
    89.*/
/*          THAT CAN BE MERGED TOGETHER.                              
    90.*/
/*          --------------------------------------------              
    91.*/
/*<   700       HEAD = 0                                                     >*/
L700:
	head = 0;
/*<             MRGSZE = 0                                                   >*/
	mrgsze = 0;
/*<             DO 1100 IOV = 1, NOVRLP                                      >*/
	i__2 = novrlp;
	for (iov = 1; iov <= i__2; ++iov) {
/*<                NODE = OVRLP(IOV)                                         >*/
	    node = ovrlp[iov];
/*<                JSTRT = XADJ(NODE)                                        >*/
	    jstrt = xadj[node];
/*<                JSTOP = XADJ(NODE+1) - 1                                  >*/
	    jstop = xadj[node + 1] - 1;
/*<                DO 800 J = JSTRT, JSTOP                                   >*/
	    i__3 = jstop;
	    for (j = jstrt; j <= i__3; ++j) {
/*<                   NABOR = ADJNCY(J)                                      >*/
		nabor = adjncy[j];
/*<                   IF ( MARKER(NABOR) .NE. 0 )  GOTO 800                  >*/
		if (marker[nabor] != 0) {
		    goto L800;
		}
/*<                      MARKER(NODE) = 1                                    >*/
		marker[node] = 1;
/*<                      GOTO 1100                                           >*/
		goto L1100;
/*<   800          CONTINUE                                                  >*/
L800:
		;
	    }
/*             -----------------------------------------          
       104.*/
/*             NODE BELONGS TO THE NEW MERGED SUPERNODE.          
       105.*/
/*             UPDATE THE VECTORS QLINK AND QSIZE.                
       106.*/
/*             -----------------------------------------          
       107.*/
/*<                MRGSZE = MRGSZE + QSIZE(NODE)                             >*/
	    mrgsze += qsize[node];
/*<                MARKER(NODE) = - 1                                        >*/
	    marker[node] = -1;
/*<                LNODE = NODE                                              >*/
	    lnode = node;
/*<   900          LINK  = QLINK(LNODE)                                      >*/
L900:
	    link = qlink[lnode];
/*<                IF ( LINK .LE. 0 )  GOTO 1000                             >*/
	    if (link <= 0) {
		goto L1000;
	    }
/*<                   LNODE = LINK                                           >*/
	    lnode = link;
/*<                   GOTO 900                                               >*/
	    goto L900;
/*<  1000          QLINK(LNODE) = HEAD                                       >*/
L1000:
	    qlink[lnode] = head;
/*<                HEAD = NODE                                               >*/
	    head = node;
/*<  1100       CONTINUE                                                     >*/
L1100:
	    ;
	}
/*<             IF ( HEAD .LE. 0 )  GOTO 1200                                >*/
	if (head <= 0) {
	    goto L1200;
	}
/*<                QSIZE(HEAD) = MRGSZE                                      >*/
	qsize[head] = mrgsze;
/*<                DEG(HEAD) = DEG0 + DEG1 - 1                               >*/
	deg[head] = *deg0 + deg1 - 1;
/*<                MARKER(HEAD) = 2                                          >*/
	marker[head] = 2;
/*          --------------------                                      
   122.*/
/*          RESET MARKER VALUES.                                      
   123.*/
/*          --------------------                                      
   124.*/
/*<  1200       ROOT = NBRHD(INHD)                                           >*/
L1200:
	root = nbrhd[inhd];
/*<             MARKER(ROOT) = 0                                             >*/
	marker[root] = 0;
/*<             IF ( RCHSZE .LE. 0 )  GOTO 1400                              >*/
	if (rchsze <= 0) {
	    goto L1400;
	}
/*<                DO 1300 IRCH = 1, RCHSZE                                  >*/
	i__2 = rchsze;
	for (irch = 1; irch <= i__2; ++irch) {
/*<                   NODE = RCHSET(IRCH)                                    >*/
	    node = rchset[irch];
/*<                   MARKER(NODE) = 0                                       >*/
	    marker[node] = 0;
/*<  1300          CONTINUE                                                  >*/
/* L1300: */
	}
/*<  1400    CONTINUE                                                        >*/
L1400:
	;
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* qmdmrg_ */

