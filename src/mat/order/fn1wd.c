/* fn1wd.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"
#include <math.h>

/* ----- SUBROUTINE FN1WD */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/********     FN1WD ..... FIND ONE-WAY DISSECTORS        ********           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS SUBROUTINE FINDS ONE-WAY DISSECTORS OF                7.
*/
/*       A CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT.                 8.
*/
/*                                                                         9.
*/
/*    INPUT PARAMETERS -                                                  10.
*/
/*       ROOT - A NODE THAT DEFINES (ALONG WITH MASK) THE                 11.
*/
/*              COMPONENT TO BE PROCESSED.                                12.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                        13.
*/
/*                                                                        14.
*/
/*    OUTPUT PARAMETERS -                                                 15.
*/
/*       NSEP - NUMBER OF NODES IN THE ONE-WAY DISSECTORS.                16.
*/
/*       SEP - VECTOR CONTAINING THE DISSECTOR NODES.                     17.
*/
/*                                                                        18.
*/
/*    UPDATED PARAMETER -                                                 19.
*/
/*       MASK - NODES IN THE DISSECTOR HAVE THEIR MASK VALUES             20.
*/
/*              SET TO ZERO.                                              21.
*/
/*                                                                        22.
*/
/*    WORKING PARAMETERS-                                                 23.
*/
/*       (XLS, LS) - LEVEL STRUCTURE USED BY THE ROUTINE FNROOT.          24.
*/
/*                                                                        25.
*/
/*    PROGRAM SUBROUTINE -                                                26.
*/
/*       FNROOT.                                                          27.
*/
/*                                                                        28.
*/
/****************************************************************          29.
*/
/*                                                                        30.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define fn1wd_ FN1WD
#elif !defined(FORTRANUNDERSCORE)
#define fn1wd_ fn1wd
#endif
/* Subroutine */ int fn1wd_(int *root, int *xadj, int *adjncy, 
	int *mask, int *nsep, int *sep, int *nlvl, int *
	xls, int *ls)
{
    /* System generated locals */
    int i__1, i__2;

    /* Builtin functions */

    /* Local variables */
    static int node, i, j, k;
    static double width, fnlvl;
    static int kstop, kstrt, lp1beg, lp1end;
    static double deltp1;
    static int lvlbeg, lvlend;
    extern /* Subroutine */ int fnroot_(int *, int *, int *, 
	    int *, int *, int *, int *);
    static int nbr, lvl;

/*                                                                        
33.*/
/****************************************************************         
 34.*/

/*<          INT ADJNCY(1), LS(1), MASK(1), SEP(1), XLS(1)               >*/
/*<    >*/
/*<          DOUBLE DELTP1, FNLVL, WIDTH                                       >*/
/*                                                                        
41.*/
/****************************************************************         
 42.*/
/*                                                                        
43.*/
/*<    >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --sep;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    fnroot_(root, &xadj[1], &adjncy[1], &mask[1], nlvl, &xls[1], &ls[1]);
/*<          FNLVL = FLOAT(NLVL)                                             >*/
    fnlvl = (double) (*nlvl);
/*<          NSEP  = XLS(NLVL + 1) - 1                                       >*/
    *nsep = xls[*nlvl + 1] - 1;
/*<          WIDTH = FLOAT(NSEP) / FNLVL                                     >*/
    width = (double) (*nsep) / fnlvl;
/*<          DELTP1 = 1.0 + SQRT((3.0*WIDTH+13.0)/2.0)                       >*/
    deltp1 = sqrt((width * 3.f + 13.f) / 2.f) + 1.f;
/*<          IF  (NSEP .GE. 50 .AND. DELTP1 .LE. 0.5*FNLVL) GO TO 300        >*/
    if (*nsep >= 50 && deltp1 <= fnlvl * .5f) {
	goto L300;
    }
/*       ----------------------------------------------------             
51.*/
/*       THE COMPONENT IS TOO SMALL, OR THE LEVEL STRUCTURE               
52.*/
/*       IS VERY LONG AND NARROW. RETURN THE WHOLE COMPONENT.             
53.*/
/*       ----------------------------------------------------             
54.*/
/*<             DO 200 I = 1, NSEP                                           >*/
    i__1 = *nsep;
    for (i = 1; i <= i__1; ++i) {
/*<                NODE = LS(I)                                              >*/
	node = ls[i];
/*<                SEP(I) = NODE                                             >*/
	sep[i] = node;
/*<                MASK(NODE) = 0                                            >*/
	mask[node] = 0;
/*<   200       CONTINUE                                                     >*/
/* L200: */
    }
/*<             RETURN                                                       >*/
    return 0;
/*       -----------------------------                                    
61.*/
/*       FIND THE PARALLEL DISSECTORS.                                    
62.*/
/*       -----------------------------                                    
63.*/
/*<   300    NSEP = 0                                                        >*/
L300:
    *nsep = 0;
/*<          I = 0                                                           >*/
    i = 0;
/*<   400    I = I + 1                                                       >*/
L400:
    ++i;
/*<             LVL = IFIX (FLOAT(I)*DELTP1 + 0.5)                           >*/
    lvl = (int) ((double) i * deltp1 + .5f);
/*<             IF ( LVL .GE. NLVL )  RETURN                                 >*/
    if (lvl >= *nlvl) {
	return 0;
    }
/*<             LVLBEG = XLS(LVL)                                            >*/
    lvlbeg = xls[lvl];
/*<             LP1BEG = XLS(LVL + 1)                                        >*/
    lp1beg = xls[lvl + 1];
/*<             LVLEND = LP1BEG - 1                                          >*/
    lvlend = lp1beg - 1;
/*<             LP1END = XLS(LVL + 2) - 1                                    >*/
    lp1end = xls[lvl + 2] - 1;
/*<             DO 500 J = LP1BEG, LP1END                                    >*/
    i__1 = lp1end;
    for (j = lp1beg; j <= i__1; ++j) {
/*<                NODE = LS(J)                                              >*/
	node = ls[j];
/*<                XADJ(NODE) =  - XADJ(NODE)                                >*/
	xadj[node] = -xadj[node];
/*<   500       CONTINUE                                                     >*/
/* L500: */
    }
/*          -------------------------------------------------             
77.*/
/*          NODES IN LEVEL LVL ARE CHOSEN TO FORM DISSECTOR.              
78.*/
/*          INCLUDE ONLY THOSE WITH NEIGHBORS IN LVL+1 LEVEL.             
79.*/
/*          XADJ IS USED TEMPORARILY TO MARK NODES IN LVL+1.              
80.*/
/*          -------------------------------------------------             
81.*/
/*<             DO 700 J = LVLBEG, LVLEND                                    >*/
    i__1 = lvlend;
    for (j = lvlbeg; j <= i__1; ++j) {
/*<                NODE = LS(J)                                              >*/
	node = ls[j];
/*<                KSTRT = XADJ(NODE)                                        >*/
	kstrt = xadj[node];
/*<                KSTOP = IABS(XADJ(NODE+1)) - 1                            >*/
	kstop = (i__2 = xadj[node + 1], abs(i__2)) - 1;
/*<                DO 600 K = KSTRT, KSTOP                                   >*/
	i__2 = kstop;
	for (k = kstrt; k <= i__2; ++k) {
/*<                   NBR = ADJNCY(K)                                        >*/
	    nbr = adjncy[k];
/*<                   IF ( XADJ(NBR) .GT. 0 )  GO TO 600                     >*/
	    if (xadj[nbr] > 0) {
		goto L600;
	    }
/*<                      NSEP = NSEP + 1                                     >*/
	    ++(*nsep);
/*<                      SEP(NSEP) = NODE                                    >*/
	    sep[*nsep] = node;
/*<                      MASK(NODE) = 0                                      >*/
	    mask[node] = 0;
/*<                      GO TO 700                                           >*/
	    goto L700;
/*<   600          CONTINUE                                                  >*/
L600:
	    ;
	}
/*<   700       CONTINUE                                                     >*/
L700:
	;
    }
/*<             DO 800 J = LP1BEG, LP1END                                    >*/
    i__1 = lp1end;
    for (j = lp1beg; j <= i__1; ++j) {
/*<                NODE = LS(J)                                              >*/
	node = ls[j];
/*<                XADJ(NODE) = - XADJ(NODE)                                 >*/
	xadj[node] = -xadj[node];
/*<   800       CONTINUE                                                     >*/
/* L800: */
    }
/*<           GO TO 400                                                      >*/
    goto L400;
/*<        END                                                               >*/
} /* fn1wd_ */

