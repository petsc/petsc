/* rcm.f -- translated by f2c (version 19931217).
*/

#include "petsc.h"

/* ----- SUBROUTINE RCM */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*********     RCM ..... REVERSE CUTHILL-MCKEE ORDERING   *******           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - RCM NUMBERS A CONNECTED COMPONENT SPECIFIED BY             7.
*/
/*       MASK AND ROOT, USING THE RCM ALGORITHM.                           8.
*/
/*       THE NUMBERING IS TO BE STARTED AT THE NODE ROOT.                  9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       ROOT - IS THE NODE THAT DEFINES THE CONNECTED                    12.
*/
/*              COMPONENT AND IT IS USED AS THE STARTING                  13.
*/
/*              NODE FOR THE RCM ORDERING.                                14.
*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR FOR                    15.
*/
/*              THE GRAPH.                                                16.
*/
/*                                                                        17.
*/
/*    UPDATED PARAMETERS -                                                18.
*/
/*       MASK - ONLY THOSE NODES WITH NONZERO INPUT MASK                  19.
*/
/*              VALUES ARE CONSIDERED BY THE ROUTINE.  THE                20.
*/
/*              NODES NUMBERED BY RCM WILL HAVE THEIR                     21.
*/
/*              MASK VALUES SET TO ZERO.                                  22.
*/
/*                                                                        23.
*/
/*    OUTPUT PARAMETERS -                                                 24.
*/
/*       PERM - WILL CONTAIN THE RCM ORDERING.                            25.
*/
/*       CCSIZE - IS THE SIZE OF THE CONNECTED COMPONENT                  26.
*/
/*              THAT HAS BEEN NUMBERED BY RCM.                            27.
*/
/*                                                                        28.
*/
/*    WORKING PARAMETER -                                                 29.
*/
/*       DEG - IS A TEMPORARY VECTOR USED TO HOLD THE DEGREE              30.
*/
/*              OF THE NODES IN THE SECTION GRAPH SPECIFIED               31.
*/
/*              BY MASK AND ROOT.                                         32.
*/
/*                                                                        33.
*/
/*    PROGRAM SUBROUTINES -                                               34.
*/
/*       DEGREE.                                                          35.
*/
/*                                                                        36.
*/
/****************************************************************          37.
*/
/*                                                                        38.
*/
/*<    >*/
/* Subroutine */ int rcm(int *root, int *xadj, int *adjncy, 
	int *mask, int *perm, int *ccsize, int *deg)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int node, fnbr, lnbr, i, j, k, l, lperm, jstop, jstrt;
    extern /* Subroutine */ int degree(int *, int *, int *, 
	    int *, int *, int *, int *);
    static int lbegin, lvlend, nbr;

/*                                                                        
41.*/
/****************************************************************         
 42.*/
/*                                                                        
43.*/
/*<          INT ADJNCY(1), DEG(1), MASK(1), PERM(1)                     >*/
/*<    >*/
/*                                                                        
48.*/
/****************************************************************         
 49.*/
/*                                                                        
50.*/
/*       -------------------------------------                            
51.*/
/*       FIND THE DEGREES OF THE NODES IN THE                             
52.*/
/*       COMPONENT SPECIFIED BY MASK AND ROOT.                            
53.*/
/*       -------------------------------------                            
54.*/
/*<    >*/
    /* Parameter adjustments */
    --deg;
    --perm;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    degree(root, &xadj[1], &adjncy[1], &mask[1], &deg[1], ccsize, &perm[1]);
/*<          MASK(ROOT) = 0                                                  >*/
    mask[*root] = 0;
/*<          IF ( CCSIZE .LE. 1 ) RETURN                                     >*/
    if (*ccsize <= 1) {
	return 0;
    }
/*<          LVLEND = 0                                                      >*/
    lvlend = 0;
/*<          LNBR = 1                                                        >*/
    lnbr = 1;
/*       --------------------------------------------                     
61.*/
/*       LBEGIN AND LVLEND POINT TO THE BEGINNING AND                     
62.*/
/*       THE END OF THE CURRENT LEVEL RESPECTIVELY.                       
63.*/
/*       --------------------------------------------                     
64.*/
/*<   100    LBEGIN = LVLEND + 1                                             >*/
L100:
    lbegin = lvlend + 1;
/*<          LVLEND = LNBR                                                   >*/
    lvlend = lnbr;
/*<          DO 600 I = LBEGIN, LVLEND                                       >*/
    i__1 = lvlend;
    for (i = lbegin; i <= i__1; ++i) {
/*          ----------------------------------                        
    68.*/
/*          FOR EACH NODE IN CURRENT LEVEL ...                        
    69.*/
/*          ----------------------------------                        
    70.*/
/*<             NODE = PERM(I)                                               >*/
	node = perm[i];
/*<             JSTRT = XADJ(NODE)                                           >*/
	jstrt = xadj[node];
/*<             JSTOP = XADJ(NODE+1) - 1                                     >*/
	jstop = xadj[node + 1] - 1;
/*          ------------------------------------------------          
    74.*/
/*          FIND THE UNNUMBERED NEIGHBORS OF NODE.                    
    75.*/
/*          FNBR AND LNBR POINT TO THE FIRST AND LAST                 
    76.*/
/*          UNNUMBERED NEIGHBORS RESPECTIVELY OF THE CURRENT          
    77.*/
/*          NODE IN PERM.                                             
    78.*/
/*          ------------------------------------------------          
    79.*/
/*<             FNBR = LNBR + 1                                              >*/
	fnbr = lnbr + 1;
/*<             DO 200 J = JSTRT, JSTOP                                      >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                NBR = ADJNCY(J)                                           >*/
	    nbr = adjncy[j];
/*<                IF ( MASK(NBR) .EQ. 0 )  GO TO 200                        >*/
	    if (mask[nbr] == 0) {
		goto L200;
	    }
/*<                   LNBR = LNBR + 1                                        >*/
	    ++lnbr;
/*<                   MASK(NBR) = 0                                          >*/
	    mask[nbr] = 0;
/*<                   PERM(LNBR) = NBR                                       >*/
	    perm[lnbr] = nbr;
/*<   200       CONTINUE                                                     >*/
L200:
	    ;
	}
/*<             IF ( FNBR .GE. LNBR )  GO TO 600                             >*/
	if (fnbr >= lnbr) {
	    goto L600;
	}
/*             ------------------------------------------             
    89.*/
/*             SORT THE NEIGHBORS OF NODE IN INCREASING               
    90.*/
/*             ORDER BY DEGREE. LINEAR INSERTION IS USED.             
    91.*/
/*             ------------------------------------------             
    92.*/
/*<                K = FNBR                                                  >*/
	k = fnbr;
/*<   300          L = K                                                     >*/
L300:
	l = k;
/*<                   K = K + 1                                              >*/
	++k;
/*<                   NBR = PERM(K)                                          >*/
	nbr = perm[k];
/*<   400             IF ( L .LT. FNBR )  GO TO 500                          >*/
L400:
	if (l < fnbr) {
	    goto L500;
	}
/*<                      LPERM = PERM(L)                                     >*/
	lperm = perm[l];
/*<                      IF ( DEG(LPERM) .LE. DEG(NBR) )  GO TO 500          >*/
	if (deg[lperm] <= deg[nbr]) {
	    goto L500;
	}
/*<                         PERM(L+1) = LPERM                                >*/
	perm[l + 1] = lperm;
/*<                         L = L - 1                                        >*/
	--l;
/*<                         GO TO 400                                        >*/
	goto L400;
/*<   500             PERM(L+1) = NBR                                        >*/
L500:
	perm[l + 1] = nbr;
/*<                   IF ( K .LT. LNBR )  GO TO 300                          >*/
	if (k < lnbr) {
	    goto L300;
	}
/*<   600    CONTINUE                                                        >*/
L600:
	;
    }
/*<          IF (LNBR .GT. LVLEND) GO TO 100                                 >*/
    if (lnbr > lvlend) {
	goto L100;
    }
/*       ---------------------------------------                         1
07.*/
/*       WE NOW HAVE THE CUTHILL MCKEE ORDERING.                         1
08.*/
/*       REVERSE IT BELOW ...                                            1
09.*/
/*       ---------------------------------------                         1
10.*/
/*<          K = CCSIZE/2                                                    >*/
    k = *ccsize / 2;
/*<          L = CCSIZE                                                      >*/
    l = *ccsize;
/*<          DO 700 I = 1, K                                                 >*/
    i__1 = k;
    for (i = 1; i <= i__1; ++i) {
/*<             LPERM = PERM(L)                                              >*/
	lperm = perm[l];
/*<             PERM(L) = PERM(I)                                            >*/
	perm[l] = perm[i];
/*<             PERM(I) = LPERM                                              >*/
	perm[i] = lperm;
/*<             L = L - 1                                                    >*/
	--l;
/*<   700    CONTINUE                                                        >*/
/* L700: */
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* rcm_ */

