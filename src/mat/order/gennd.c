/* gennd.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/*<       subroutine revrse(n,perm) >*/
#if defined(FORTRANCAPS)
#define revrse_ REVRSE
#elif !defined(FORTRANUNDERSCORE)
#define revrse_ revrse
#endif
/* Subroutine */ int revrse_(int *n, int *perm)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int swap, i, m, in;

/*<       int n,perm(*) >*/
/*<       int i,in,m,swap >*/
/*<       in = n >*/
    /* Parameter adjustments */
    --perm;

    /* Function Body */
    in = *n;
/*<       m = n/2 >*/
    m = *n / 2;
/*<       do 10 i=1,m >*/
    i__1 = m;
    for (i = 1; i <= i__1; ++i) {
/*<         swap = perm(i) >*/
	swap = perm[i];
/*<         perm(i) = perm(in) >*/
	perm[i] = perm[in];
/*<         perm(in) = swap >*/
	perm[in] = swap;
/*<         in = in - 1 >*/
	--in;
/*<  10   continue >*/
/* L10: */
    }
/*<       return  >*/
    return 0;
/*<       end >*/
} /* revrse_ */

/* ----- SUBROUTINE GENND */
/****************************************************************           1.
*/
/****************************************************************           2.
*/
/*********     GENND ..... GENERAL NESTED DISSECTION     ********           3.
*/
/****************************************************************           4.
*/
/****************************************************************           5.
*/
/*                                                                         6.
*/
/*    PURPOSE - SUBROUTINE GENND FINDS A NESTED DISSECTION                 7.
*/
/*       ORDERING FOR A GENERAL GRAPH.                                     8.
*/
/*                                                                         9.
*/
/*                                                                        10.
*/
/*    INPUT PARAMETERS -                                                  11.
*/
/*       NEQNS - NUMBER OF EQUATIONS.                                     12.
*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR.                       13.
*/
/*                                                                        14.
*/
/*    OUTPUT PARAMETERS -                                                 15.
*/
/*       PERM - THE NESTED DISSECTION ORDERING.                           16.
*/
/*                                                                        17.
*/
/*    WORKING PARAMETERS -                                                18.
*/
/*       MASK - IS USED TO MASK OFF VARIABLES THAT HAVE                   19.
*/
/*              BEEN NUMBERED DURING THE ORDERNG PROCESS.                 20.
*/
/*       (XLS, LS) - THIS LEVEL STRUCTURE PAIR IS USED AS                 21.
*/
/*              TEMPORARY STORAGE BY FNROOT.                              22.
*/
/*                                                                        23.
*/
/*    PROGRAM SUBROUTINES -                                               24.
*/
/*       FNDSEP, REVRSE.                                                  25.
*/
/*                                                                        26.
*/
/****************************************************************          27.
*/
/*                                                                        28.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define gennd_ GENND
#elif !defined(FORTRANUNDERSCORE)
#define gennd_ gennd
#endif
/* Subroutine */ int gennd_(int *neqns, int *xadj, int *adjncy, 
	int *mask, int *perm, int *xls, int *ls)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int nsep, root, i;
    extern /* Subroutine */ int fndsep_(int *, int *, int *, 
	    int *, int *, int *, int *, int *), revrse_(
	    int *, int *);
    static int num;

/*                                                                        
31.*/
/****************************************************************         
 32.*/
/*                                                                        
33.*/
/*<    >*/
/*<           INT XADJ(1), I, NEQNS, NSEP, NUM, ROOT                     >*/
/*                                                                        
37.*/
/****************************************************************         
 38.*/
/*                                                                        
39.*/
/*<           DO 100 I = 1, NEQNS                                            >*/
    /* Parameter adjustments */
    --ls;
    --xls;
    --perm;
    --mask;
    --adjncy;
    --xadj;

    /* Function Body */
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*<              MASK(I) = 1                                                 >*/
	mask[i] = 1;
/*<   100     CONTINUE                                                       >*/
/* L100: */
    }
/*<           NUM   = 0                                                      >*/
    num = 0;
/*<           DO 300 I = 1, NEQNS                                            >*/
    i__1 = *neqns;
    for (i = 1; i <= i__1; ++i) {
/*           -----------------------------                            
    45.*/
/*           FOR EACH MASKED COMPONENT ...                            
    46.*/
/*           -----------------------------                            
    47.*/
/*<   200        IF ( MASK(I) .EQ. 0 )  GO TO 300                            >*/
L200:
	if (mask[i] == 0) {
	    goto L300;
	}
/*<                 ROOT = I                                                 >*/
	root = i;
/*              -------------------------------------------           
    50.*/
/*              FIND A SEPARATOR AND NUMBER THE NODES NEXT.           
    51.*/
/*              -------------------------------------------           
    52.*/
/*<    >*/
	fndsep_(&root, &xadj[1], &adjncy[1], &mask[1], &nsep, &perm[num + 1], 
		&xls[1], &ls[1]);
/*<                 NUM  = NUM + NSEP                                        >*/
	num += nsep;
/*<                 IF ( NUM .GE. NEQNS )  GO TO 400                         >*/
	if (num >= *neqns) {
	    goto L400;
	}
/*<                 GO TO 200                                                >*/
	goto L200;
/*<   300     CONTINUE                                                       >*/
L300:
	;
    }
/*        ----------------------------------------------                  
59.*/
/*        SINCE SEPARATORS FOUND FIRST SHOULD BE ORDERED                  
60.*/
/*        LAST, ROUTINE REVRSE IS CALLED TO ADJUST THE                    
61.*/
/*        ORDERING VECTOR.                                                
62.*/
/*        ----------------------------------------------                  
63.*/
/*<   400     CALL  REVRSE ( NEQNS, PERM )                                   >*/
L400:
    revrse_(neqns, &perm[1]);
/*<           RETURN                                                         >*/
    return 0;
/*<        END                                                               >*/
} /* gennd_ */

