/* qmdqt.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE QMDQT */
/**************************************************************             1.
*/
/**************************************************************             2.
*/
/********     QMDQT  ..... QUOT MIN DEG QUOT TRANSFORM  *******             3.
*/
/**************************************************************             4.
*/
/**************************************************************             5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS SUBROUTINE PERFORMS THE QUOTIENT GRAPH                7.
*/
/*       TRANSFORMATION AFTER A NODE HAS BEEN ELIMINATED.                  8.
*/
/*                                                                         9.
*/
/*    INPUT PARAMETERS -                                                  10.
*/
/*       ROOT - THE NODE JUST ELIMINATED. IT BECOMES THE                  11.
*/
/*              REPRESENTATIVE OF THE NEW SUPERNODE.                      12.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                        13.
*/
/*       (RCHSZE, RCHSET) - THE REACHABLE SET OF ROOT IN THE              14.
*/
/*              OLD QUOTIENT GRAPH.                                       15.
*/
/*       NBRHD - THE NEIGHBORHOOD SET WHICH WILL BE MERGED                16.
*/
/*              WITH ROOT TO FORM THE NEW SUPERNODE.                      17.
*/
/*       MARKER - THE MARKER VECTOR.                                      18.
*/
/*                                                                        19.
*/
/*    UPDATED PARAMETER -                                                 20.
*/
/*       ADJNCY - BECOMES THE ADJNCY OF THE QUOTIENT GRAPH.               21.
*/
/*                                                                        22.
*/
/**************************************************************            23.
*/
/*                                                                        24.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define qmdqt_ QMDQT
#elif !defined(FORTRANUNDERSCORE)
#define qmdqt_ qmdqt
#endif
/* Subroutine */ int qmdqt_(int *root, int *xadj, int *adjncy, 
	int *marker, int *rchsze, int *rchset, int *nbrhd)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int inhd, irch, node, link, j, nabor, jstop, jstrt;

/*                                                                        
27.*/
/**************************************************************           
 28.*/
/*                                                                        
29.*/
/*<          INT ADJNCY(1), MARKER(1), RCHSET(1), NBRHD(1)               >*/
/*<    >*/
/*                                                                        
33.*/
/**************************************************************           
 34.*/
/*                                                                        
35.*/
/*<          IRCH = 0                                                        >*/
    /* Parameter adjustments */
    --nbrhd;
    --rchset;
    --marker;
    --adjncy;
    --xadj;

    /* Function Body */
    irch = 0;
/*<          INHD = 0                                                        >*/
    inhd = 0;
/*<          NODE = ROOT                                                     >*/
    node = *root;
/*<   100    JSTRT = XADJ(NODE)                                              >*/
L100:
    jstrt = xadj[node];
/*<          JSTOP = XADJ(NODE+1) - 2                                        >*/
    jstop = xadj[node + 1] - 2;
/*<          IF ( JSTOP .LT. JSTRT )  GO TO 300                              >*/
    if (jstop < jstrt) {
	goto L300;
    }
/*          ------------------------------------------------              
42.*/
/*          PLACE REACH NODES INTO THE ADJACENT LIST OF NODE              
43.*/
/*          ------------------------------------------------              
44.*/
/*<             DO 200 J = JSTRT, JSTOP                                      >*/
    i__1 = jstop;
    for (j = jstrt; j <= i__1; ++j) {
/*<                IRCH = IRCH + 1                                           >*/
	++irch;
/*<                ADJNCY(J) = RCHSET(IRCH)                                  >*/
	adjncy[j] = rchset[irch];
/*<                IF ( IRCH .GE. RCHSZE )  GOTO 400                         >*/
	if (irch >= *rchsze) {
	    goto L400;
	}
/*<   200       CONTINUE                                                     >*/
/* L200: */
    }
/*       ----------------------------------------------                   
50.*/
/*       LINK TO OTHER SPACE PROVIDED BY THE NBRHD SET.                   
51.*/
/*       ----------------------------------------------                   
52.*/
/*<   300    LINK = ADJNCY(JSTOP+1)                                          >*/
L300:
    link = adjncy[jstop + 1];
/*<          NODE = - LINK                                                   >*/
    node = -link;
/*<          IF ( LINK .LT. 0 )  GOTO 100                                    >*/
    if (link < 0) {
	goto L100;
    }
/*<             INHD = INHD + 1                                              >*/
    ++inhd;
/*<             NODE = NBRHD(INHD)                                           >*/
    node = nbrhd[inhd];
/*<             ADJNCY(JSTOP+1) = - NODE                                     >*/
    adjncy[jstop + 1] = -node;
/*<             GO TO 100                                                    >*/
    goto L100;
/*       -------------------------------------------------------          
60.*/
/*       ALL REACHABLE NODES HAVE BEEN SAVED.  END THE ADJ LIST.          
61.*/
/*       ADD ROOT TO THE NBR LIST OF EACH NODE IN THE REACH SET.          
62.*/
/*       -------------------------------------------------------          
63.*/
/*<   400    ADJNCY(J+1) = 0                                                 >*/
L400:
    adjncy[j + 1] = 0;
/*<          DO 600 IRCH = 1, RCHSZE                                         >*/
    i__1 = *rchsze;
    for (irch = 1; irch <= i__1; ++irch) {
/*<             NODE = RCHSET(IRCH)                                          >*/
	node = rchset[irch];
/*<             IF ( MARKER(NODE) .LT. 0 )  GOTO 600                         >*/
	if (marker[node] < 0) {
	    goto L600;
	}
/*<                JSTRT = XADJ(NODE)                                        >*/
	jstrt = xadj[node];
/*<                JSTOP = XADJ(NODE+1) - 1                                  >*/
	jstop = xadj[node + 1] - 1;
/*<                DO 500 J = JSTRT, JSTOP                                   >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                   NABOR = ADJNCY(J)                                      >*/
	    nabor = adjncy[j];
/*<                   IF ( MARKER(NABOR) .GE. 0 ) GO TO 500                  >*/
	    if (marker[nabor] >= 0) {
		goto L500;
	    }
/*<                      ADJNCY(J) = ROOT                                    >*/
	    adjncy[j] = *root;
/*<                      GOTO 600                                            >*/
	    goto L600;
/*<   500          CONTINUE                                                  >*/
L500:
	    ;
	}
/*<   600    CONTINUE                                                        >*/
L600:
	;
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* qmdqt_ */

