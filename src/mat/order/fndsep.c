/* fndsep.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE FNDSEP */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*************     FNDSEP ..... FIND SEPARATOR       ************           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS ROUTINE IS USED TO FIND A SMALL                       7.
*/
/*              SEPARATOR FOR A CONNECTED COMPONENT SPECIFIED              8.
*/
/*              BY MASK IN THE GIVEN GRAPH.                                9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       ROOT - IS THE NODE THAT DETERMINES THE MASKED                    12.
*/
/*              COMPONENT.                                                13.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.                   14.
*/
/*                                                                        15.
*/
/*    OUTPUT PARAMETERS -                                                 16.
*/
/*       NSEP - NUMBER OF VARIABLES IN THE SEPARATOR.                     17.
*/
/*       SEP - VECTOR CONTAINING THE SEPARATOR NODES.                     18.
*/
/*                                                                        19.
*/
/*    UPDATED PARAMETER -                                                 20.
*/
/*       MASK - NODES IN THE SEPARATOR HAVE THEIR MASK                    21.
*/
/*              VALUES SET TO ZERO.                                       22.
*/
/*                                                                        23.
*/
/*    WORKING PARAMETERS -                                                24.
*/
/*       (XLS, LS) - LEVEL STRUCTURE PAIR FOR LEVEL STRUCTURE             25.
*/
/*              FOUND BY FNROOT.                                          26.
*/
/*                                                                        27.
*/
/*    PROGRAM SUBROUTINES -                                               28.
*/
/*       FNROOT.                                                          29.
*/
/*                                                                        30.
*/
/****************************************************************          31.
*/
/*                                                                        32.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define fndsep_ FNDSEP
#elif !defined(FORTRANUNDERSCORE)
#define fndsep_ fndsep
#endif
/* Subroutine */ int fndsep_(int *root, int *xadj, int *adjncy, 
	int *mask, int *nsep, int *sep, int *xls, int *ls)
{
    /* System generated locals */
    int i__1, i__2;

    /* Local variables */
    static int node, nlvl, i, j, jstop, jstrt, mp1beg, mp1end, midbeg, 
	    midend, midlvl;
    extern /* Subroutine */ int fnroot_(int *, int *, int *, 
	    int *, int *, int *, int *);
    static int nbr;

/*                                                                        
35.*/
/****************************************************************         
 36.*/
/*                                                                        
37.*/
/*<          INT ADJNCY(1), LS(1), MASK(1), SEP(1), XLS(1)               >*/
/*<    >*/
/*                                                                        
42.*/
/****************************************************************         
 43.*/
/*                                                                        
44.*/
/*<    >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --sep;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    fnroot_(root, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &ls[1]);
/*       ----------------------------------------------                   
47.*/
/*       IF THE NUMBER OF LEVELS IS LESS THAN 3, RETURN                   
48.*/
/*       THE WHOLE COMPONENT AS THE SEPARATOR.                            
49.*/
/*       ----------------------------------------------                   
50.*/
/*<          IF ( NLVL .GE. 3 )  GO TO 200                                   >*/
    if (nlvl >= 3) {
	goto L200;
    }
/*<             NSEP = XLS(NLVL+1) - 1                                       >*/
    *nsep = xls[nlvl + 1] - 1;
/*<             DO 100 I = 1, NSEP                                           >*/
    i__1 = *nsep;
    for (i = 1; i <= i__1; ++i) {
/*<                NODE = LS(I)                                              >*/
	node = ls[i];
/*<                SEP(I) = NODE                                             >*/
	sep[i] = node;
/*<                MASK(NODE) = 0                                            >*/
	mask[node] = 0;
/*<   100       CONTINUE                                                     >*/
/* L100: */
    }
/*<             RETURN                                                       >*/
    return 0;
/*       ----------------------------------------------------             
59.*/
/*       FIND THE MIDDLE LEVEL OF THE ROOTED LEVEL STRUCTURE.             
60.*/
/*       ----------------------------------------------------             
61.*/
/*<   200    MIDLVL = (NLVL + 2)/2                                           >*/
L200:
    midlvl = (nlvl + 2) / 2;
/*<          MIDBEG = XLS(MIDLVL)                                            >*/
    midbeg = xls[midlvl];
/*<          MP1BEG = XLS(MIDLVL + 1)                                        >*/
    mp1beg = xls[midlvl + 1];
/*<          MIDEND = MP1BEG - 1                                             >*/
    midend = mp1beg - 1;
/*<          MP1END = XLS(MIDLVL+2) - 1                                      >*/
    mp1end = xls[midlvl + 2] - 1;
/*       -------------------------------------------------                
67.*/
/*       THE SEPARATOR IS OBTAINED BY INCLUDING ONLY THOSE                
68.*/
/*       MIDDLE-LEVEL NODES WITH NEIGHBORS IN THE MIDDLE+1                
69.*/
/*       LEVEL. XADJ IS USED TEMPORARILY TO MARK THOSE                    
70.*/
/*       NODES IN THE MIDDLE+1 LEVEL.                                     
71.*/
/*       -------------------------------------------------                
72.*/
/*<          DO 300 I = MP1BEG, MP1END                                       >*/
    i__1 = mp1end;
    for (i = mp1beg; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             XADJ(NODE) = - XADJ(NODE)                                    >*/
	xadj[node] = -xadj[node];
/*<   300    CONTINUE                                                        >*/
/* L300: */
    }
/*<          NSEP  = 0                                                       >*/
    *nsep = 0;
/*<          DO 500 I = MIDBEG, MIDEND                                       >*/
    i__1 = midend;
    for (i = midbeg; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             JSTRT = XADJ(NODE)                                           >*/
	jstrt = xadj[node];
/*<             JSTOP = IABS(XADJ(NODE+1)) - 1                               >*/
	jstop = (i__2 = xadj[node + 1], (int)abs(i__2)) - 1;
/*<             DO 400 J = JSTRT, JSTOP                                      >*/
	i__2 = jstop;
	for (j = jstrt; j <= i__2; ++j) {
/*<                NBR = ADJNCY(J)                                           >*/
	    nbr = adjncy[j];
/*<                IF ( XADJ(NBR) .GT. 0 )  GO TO 400                        >*/
	    if (xadj[nbr] > 0) {
		goto L400;
	    }
/*<                   NSEP = NSEP + 1                                        >*/
	    ++(*nsep);
/*<                   SEP(NSEP) = NODE                                       >*/
	    sep[*nsep] = node;
/*<                   MASK(NODE) = 0                                         >*/
	    mask[node] = 0;
/*<                   GO TO 500                                              >*/
	    goto L500;
/*<   400       CONTINUE                                                     >*/
L400:
	    ;
	}
/*<   500    CONTINUE                                                        >*/
L500:
	;
    }
/*       -------------------------------                                  
91.*/
/*       RESET XADJ TO ITS CORRECT SIGN.                                  
92.*/
/*       -------------------------------                                  
93.*/
/*<          DO 600 I = MP1BEG, MP1END                                       >*/
    i__1 = mp1end;
    for (i = mp1beg; i <= i__1; ++i) {
/*<             NODE = LS(I)                                                 >*/
	node = ls[i];
/*<             XADJ(NODE) = - XADJ(NODE)                                    >*/
	xadj[node] = -xadj[node];
/*<   600    CONTINUE                                                        >*/
/* L600: */
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* fndsep_ */

