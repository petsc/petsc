/* qmdrch.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE QMDRCH */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/**********     QMDRCH ..... QUOT MIN DEG REACH SET    **********           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS SUBROUTINE DETERMINES THE REACHABLE SET OF            7.
*/
/*       A NODE THROUGH A GIVEN SUBSET.  THE ADJACENCY STRUCTURE           8.
*/
/*       IS ASSUMED TO BE STORED IN A QUOTIENT GRAPH FORMAT.               9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       ROOT - THE GIVEN NODE NOT IN THE SUBSET.                         12.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.                   13.
*/
/*       DEG - THE DEGREE VECTOR.  DEG(I) LT 0 MEANS THE NODE             14.
*/
/*              BELONGS TO THE GIVEN SUBSET.                              15.
*/
/*                                                                        16.
*/
/*    OUTPUT PARAMETERS -                                                 17.
*/
/*       (RCHSZE, RCHSET) - THE REACHABLE SET.                            18.
*/
/*       (NHDSZE, NBRHD) - THE NEIGHBORHOOD SET.                          19.
*/
/*                                                                        20.
*/
/*    UPDATED PARAMETERS -                                                21.
*/
/*       MARKER - THE MARKER VECTOR FOR REACH AND NBRHD SETS.             22.
*/
/*              GT 0 MEANS THE NODE IS IN REACH SET.                      23.
*/
/*              LT 0 MEANS THE NODE HAS BEEN MERGED WITH                  24.
*/
/*              OTHERS IN THE QUOTIENT OR IT IS IN NBRHD SET.             25.
*/
/*                                                                        26.
*/
/****************************************************************          27.
*/
/*                                                                        28.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define qmdrch_ QMDRCH
#elif !defined(FORTRANUNDERSCORE)
#define qmdrch_ qmdrch
#endif
/* Subroutine */ int qmdrch_(int *root, int *xadj, int *adjncy, 
	int *deg, int *marker, int *rchsze, int *rchset, 
	int *nhdsze, int *nbrhd)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int node, i, j, nabor, istop, jstop, istrt, jstrt;

/*                                                                        
31.*/
/****************************************************************         
 32.*/
/*                                                                        
33.*/
/*<    >*/
/*<    >*/
/*                                                                        
38.*/
/****************************************************************         
 39.*/
/*                                                                        
40.*/
/*       -----------------------------------------                        
41.*/
/*       LOOP THROUGH THE NEIGHBORS OF ROOT IN THE                        
42.*/
/*       QUOTIENT GRAPH.                                                  
43.*/
/*       -----------------------------------------                        
44.*/
/*<          NHDSZE = 0                                                      >*/
    /* Parameter adjustments */
    --nbrhd;
    --rchset;
    --marker;
    --deg;
    --adjncy;
    --xadj;

    /* Function Body */
    *nhdsze = 0;
/*<          RCHSZE = 0                                                      >*/
    *rchsze = 0;
/*<          ISTRT = XADJ(ROOT)                                              >*/
    istrt = xadj[*root];
/*<          ISTOP = XADJ(ROOT+1) - 1                                        >*/
    istop = xadj[*root + 1] - 1;
/*<          IF ( ISTOP .LT. ISTRT )  RETURN                                 >*/
    if (istop < istrt) {
	return 0;
    }
/*<             DO 600 I = ISTRT, ISTOP                                      >*/
    i__1 = istop;
    for (i = istrt; i <= i__1; ++i) {
/*<                NABOR =  ADJNCY(I)                                        >*/
	nabor = adjncy[i];
/*<                IF ( NABOR .EQ. 0 ) RETURN                                >*/
	if (nabor == 0) {
	    return 0;
	}
/*<                IF ( MARKER(NABOR) .NE. 0 )  GO TO 600                    >*/
	if (marker[nabor] != 0) {
	    goto L600;
	}
/*<                   IF ( DEG(NABOR) .LT. 0 )     GO TO 200                 >*/
	if (deg[nabor] < 0) {
	    goto L200;
	}
/*                   -------------------------------------            
    55.*/
/*                   INCLUDE NABOR INTO THE REACHABLE SET.            
    56.*/
/*                   -------------------------------------            
    57.*/
/*<                      RCHSZE = RCHSZE + 1                                 >*/
	++(*rchsze);
/*<                      RCHSET(RCHSZE) = NABOR                              >*/
	rchset[*rchsze] = nabor;
/*<                      MARKER(NABOR) = 1                                   >*/
	marker[nabor] = 1;
/*<                      GO TO 600                                           >*/
	goto L600;
/*                -------------------------------------               
    62.*/
/*                NABOR HAS BEEN ELIMINATED. FIND NODES               
    63.*/
/*                REACHABLE FROM IT.                                  
    64.*/
/*                -------------------------------------               
    65.*/
/*<   200             MARKER(NABOR) = -1                                     >*/
L200:
	marker[nabor] = -1;
/*<                   NHDSZE = NHDSZE +  1                                   >*/
	++(*nhdsze);
/*<                   NBRHD(NHDSZE) = NABOR                                  >*/
	nbrhd[*nhdsze] = nabor;
/*<   300             JSTRT = XADJ(NABOR)                                    >*/
L300:
	jstrt = xadj[nabor];
/*<                   JSTOP = XADJ(NABOR+1) - 1                              >*/
	jstop = xadj[nabor + 1] - 1;
/*<                   DO 500 J = JSTRT, JSTOP                                >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                      NODE = ADJNCY(J)                                    >*/
	    node = adjncy[j];
/*<                      NABOR = - NODE                                      >*/
	    nabor = -node;
/*<                      IF (NODE) 300, 600, 400                             >*/
	    if (node < 0) {
		goto L300;
	    } else if (node == 0) {
		goto L600;
	    } else {
		goto L400;
	    }
/*<   400                IF ( MARKER(NODE) .NE. 0 )  GO TO 500               >*/
L400:
	    if (marker[node] != 0) {
		goto L500;
	    }
/*<                         RCHSZE = RCHSZE + 1                              >*/
	    ++(*rchsze);
/*<                         RCHSET(RCHSZE) = NODE                            >*/
	    rchset[*rchsze] = node;
/*<                         MARKER(NODE) = 1                                 >*/
	    marker[node] = 1;
/*<   500             CONTINUE                                               >*/
L500:
	    ;
	}
/*<   600       CONTINUE                                                     >*/
L600:
	;
    }
/*<             RETURN                                                       >*/
    return 0;
/*<       END                                                                >*/
} /* qmdrch_ */

