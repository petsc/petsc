/* genqmd.f -- translated by f2c (version 19931217).
   You must link the resulting object file with the libraries:
	-lf2c -lm   (in that order)
*/

#include "petsc.h"

/* ----- SUBROUTINE GENQMD */
/*****************************************************************          1.
*/
/*****************************************************************          2.
*/
/***********    GENQMD ..... QUOT MIN DEGREE ORDERING    *********          3.
*/
/*****************************************************************          4.
*/
/*****************************************************************          5.
*/
/*                                                                         6.
*/
/*    PURPOSE - THIS ROUTINE IMPLEMENTS THE MINIMUM DEGREE                 7.
*/
/*       ALGORITHM.  IT MAKES USE OF THE IMPLICIT REPRESENT-               8.
*/
/*       ATION OF THE ELIMINATION GRAPHS BY QUOTIENT GRAPHS,               9.
*/
/*       AND THE NOTION OF INDISTINGUISHABLE NODES.                       10.
*/
/*       CAUTION - THE ADJACENCY VECTOR ADJNCY WILL BE                    11.
*/
/*       DESTROYED.                                                       12.
*/
/*                                                                        13.
*/
/*    INPUT PARAMETERS -                                                  14.
*/
/*       NEQNS - NUMBER OF EQUATIONS.                                     15.
*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                        16.
*/
/*                                                                        17.
*/
/*    OUTPUT PARAMETERS -                                                 18.
*/
/*       PERM - THE MINIMUM DEGREE ORDERING.                              19.
*/
/*       INVP - THE INVERSE OF PERM.                                      20.
*/
/*                                                                        21.
*/
/*    WORKING PARAMETERS -                                                22.
*/
/*       DEG - THE DEGREE VECTOR. DEG(I) IS NEGATIVE MEANS                23.
*/
/*              NODE I HAS BEEN NUMBERED.                                 24.
*/
/*       MARKER - A MARKER VECTOR, WHERE MARKER(I) IS                     25.
*/
/*              NEGATIVE MEANS NODE I HAS BEEN MERGED WITH                26.
*/
/*              ANOTHER NODE AND THUS CAN BE IGNORED.                     27.
*/
/*       RCHSET - VECTOR USED FOR THE REACHABLE SET.                      28.
*/
/*       NBRHD - VECTOR USED FOR THE NEIGHBORHOOD SET.                    29.
*/
/*       QSIZE - VECTOR USED TO STORE THE SIZE OF                         30.
*/
/*              INDISTINGUISHABLE SUPERNODES.                             31.
*/
/*       QLINK - VECTOR TO STORE INDISTINGUISHABLE NODES,                 32.
*/
/*              I, QLINK(I), QLINK(QLINK(I)) ... ARE THE                  33.
*/
/*              MEMBERS OF THE SUPERNODE REPRESENTED BY I.                34.
*/
/*                                                                        35.
*/
/*    PROGRAM SUBROUTINES -                                               36.
*/
/*       QMDRCH, QMDQT, QMDUPD.                                           37.
*/
/*                                                                        38.
*/
/*****************************************************************         39.
*/
/*                                                                        40.
*/
/*                                                                        41.
*/
/*<    >*/
#if defined(FORTRANCAPS)
#define genqmd_ GENQMD
#elif !defined(FORTRANUNDERSCORE)
#define genqmd_ genqmd
#endif
/* Subroutine */ int genqmd_(int *neqns, int *xadj, int *adjncy, 
	int *perm, int *invp, int *deg, int *marker, int *
	rchset, int *nbrhd, int *qsize, int *qlink, int *
	nofsub)
{
    /* System generated locals */
    int i__1;

    /* Local variables */
    static int ndeg, irch, node, nump1, j, inode;
    extern /* Subroutine */ int qmdqt_(int *, int *, int *, 
	    int *, int *, int *, int *);
    static int ip, np, mindeg, search;
    extern /* Subroutine */ int qmdrch_(int *, int *, int *, 
	    int *, int *, int *, int *, int *, int *),
	     qmdupd_(int *, int *, int *, int *, int *, 
	    int *, int *, int *, int *, int *);
    static int nhdsze, nxnode, rchsze, thresh, num;

/*                                                                        
45.*/
/*****************************************************************        
 46.*/
/*                                                                        
47.*/
/*<    >*/
/*<    >*/
/*                                                                        
53.*/
/*****************************************************************        
 54.*/
/*                                                                        
55.*/
/*       -----------------------------------------------------            
56.*/
/*       INITIALIZE DEGREE VECTOR AND OTHER WORKING VARIABLES.            
57.*/
/*       -----------------------------------------------------            
58.*/
/*<          MINDEG = NEQNS                                                  >*/
    /* Parameter adjustments */
    --qlink;
    --qsize;
    --nbrhd;
    --rchset;
    --marker;
    --deg;
    --invp;
    --perm;
    --adjncy;
    --xadj;

    /* Function Body */
    mindeg = *neqns;
/*<          NOFSUB = 0                                                      >*/
    *nofsub = 0;
/*<          DO 100 NODE = 1, NEQNS                                          >*/
    i__1 = *neqns;
    for (node = 1; node <= i__1; ++node) {
/*<             PERM(NODE) = NODE                                            >*/
	perm[node] = node;
/*<             INVP(NODE) = NODE                                            >*/
	invp[node] = node;
/*<             MARKER(NODE) = 0                                             >*/
	marker[node] = 0;
/*<             QSIZE(NODE)  = 1                                             >*/
	qsize[node] = 1;
/*<             QLINK(NODE)  = 0                                             >*/
	qlink[node] = 0;
/*<             NDEG = XADJ(NODE+1) - XADJ(NODE)                             >*/
	ndeg = xadj[node + 1] - xadj[node];
/*<             DEG(NODE) = NDEG                                             >*/
	deg[node] = ndeg;
/*<             IF ( NDEG .LT. MINDEG )  MINDEG = NDEG                       >*/
	if (ndeg < mindeg) {
	    mindeg = ndeg;
	}
/*<   100    CONTINUE                                                        >*/
/* L100: */
    }
/*<          NUM = 0                                                         >*/
    num = 0;
/*       -----------------------------------------------------            
72.*/
/*       PERFORM THRESHOLD SEARCH TO GET A NODE OF MIN DEGREE.            
73.*/
/*       VARIABLE SEARCH POINTS TO WHERE SEARCH SHOULD START.             
74.*/
/*       -----------------------------------------------------            
75.*/
/*<   200    SEARCH = 1                                                      >*/
L200:
    search = 1;
/*<             THRESH = MINDEG                                              >*/
    thresh = mindeg;
/*<             MINDEG = NEQNS                                               >*/
    mindeg = *neqns;
/*<   300       NUMP1 = NUM + 1                                              >*/
L300:
    nump1 = num + 1;
/*<                IF ( NUMP1 .GT. SEARCH )  SEARCH = NUMP1                  >*/
    if (nump1 > search) {
	search = nump1;
    }
/*<                DO 400 J = SEARCH, NEQNS                                  >*/
    i__1 = *neqns;
    for (j = search; j <= i__1; ++j) {
/*<                   NODE = PERM(J)                                         >*/
	node = perm[j];
/*<                   IF ( MARKER(NODE) .LT. 0 )  GOTO 400                   >*/
	if (marker[node] < 0) {
	    goto L400;
	}
/*<                      NDEG = DEG(NODE)                                    >*/
	ndeg = deg[node];
/*<                      IF ( NDEG .LE. THRESH )  GO TO 500                  >*/
	if (ndeg <= thresh) {
	    goto L500;
	}
/*<                      IF ( NDEG .LT. MINDEG )  MINDEG =  NDEG             >*/
	if (ndeg < mindeg) {
	    mindeg = ndeg;
	}
/*<   400          CONTINUE                                                  >*/
L400:
	;
    }
/*<             GO TO 200                                                    >*/
    goto L200;
/*          ---------------------------------------------------           
89.*/
/*          NODE HAS MINIMUM DEGREE. FIND ITS REACHABLE SETS BY           
90.*/
/*          CALLING QMDRCH.                                               
91.*/
/*          ---------------------------------------------------           
92.*/
/*<   500       SEARCH = J                                                   >*/
L500:
    search = j;
/*<             NOFSUB = NOFSUB + DEG(NODE)                                  >*/
    *nofsub += deg[node];
/*<             MARKER(NODE) = 1                                             >*/
    marker[node] = 1;
/*<    >*/
    qmdrch_(&node, &xadj[1], &adjncy[1], &deg[1], &marker[1], &rchsze, &
	    rchset[1], &nhdsze, &nbrhd[1]);
/*          ------------------------------------------------              
98.*/
/*          ELIMINATE ALL NODES INDISTINGUISHABLE FROM NODE.              
99.*/
/*          THEY ARE GIVEN BY NODE, QLINK(NODE), ....                    1
00.*/
/*          ------------------------------------------------             1
01.*/
/*<             NXNODE = NODE                                                >*/
    nxnode = node;
/*<   600       NUM = NUM + 1                                                >*/
L600:
    ++num;
/*<                NP  = INVP(NXNODE)                                        >*/
    np = invp[nxnode];
/*<                IP  = PERM(NUM)                                           >*/
    ip = perm[num];
/*<                PERM(NP) = IP                                             >*/
    perm[np] = ip;
/*<                INVP(IP) = NP                                             >*/
    invp[ip] = np;
/*<                PERM(NUM) = NXNODE                                        >*/
    perm[num] = nxnode;
/*<                INVP(NXNODE) = NUM                                        >*/
    invp[nxnode] = num;
/*<                DEG(NXNODE) = - 1                                         >*/
    deg[nxnode] = -1;
/*<                NXNODE = QLINK(NXNODE)                                    >*/
    nxnode = qlink[nxnode];
/*<             IF (NXNODE .GT. 0) GOTO 600                                  >*/
    if (nxnode > 0) {
	goto L600;
    }
/*                                                                       1
13.*/
/*<             IF ( RCHSZE .LE. 0 )  GO TO 800                              >*/
    if (rchsze <= 0) {
	goto L800;
    }
/*             ------------------------------------------------          1
15.*/
/*             UPDATE THE DEGREES OF THE NODES IN THE REACHABLE          1
16.*/
/*             SET AND IDENTIFY INDISTINGUISHABLE NODES.                 1
17.*/
/*             ------------------------------------------------          1
18.*/
/*<    >*/
    qmdupd_(&xadj[1], &adjncy[1], &rchsze, &rchset[1], &deg[1], &qsize[1], &
	    qlink[1], &marker[1], &rchset[rchsze + 1], &nbrhd[nhdsze + 1]);
/*             -------------------------------------------               1
22.*/
/*             RESET MARKER VALUE OF NODES IN REACH SET.                 1
23.*/
/*             UPDATE THRESHOLD VALUE FOR CYCLIC SEARCH.                 1
24.*/
/*             ALSO CALL QMDQT TO FORM NEW QUOTIENT GRAPH.               1
25.*/
/*             -------------------------------------------               1
26.*/
/*<                MARKER(NODE) = 0                                          >*/
    marker[node] = 0;
/*<                DO 700 IRCH = 1, RCHSZE                                   >*/
    i__1 = rchsze;
    for (irch = 1; irch <= i__1; ++irch) {
/*<                   INODE = RCHSET(IRCH)                                   >*/
	inode = rchset[irch];
/*<                   IF ( MARKER(INODE) .LT. 0 )  GOTO 700                  >*/
	if (marker[inode] < 0) {
	    goto L700;
	}
/*<                      MARKER(INODE) = 0                                   >*/
	marker[inode] = 0;
/*<                      NDEG = DEG(INODE)                                   >*/
	ndeg = deg[inode];
/*<                      IF ( NDEG .LT. MINDEG )  MINDEG = NDEG              >*/
	if (ndeg < mindeg) {
	    mindeg = ndeg;
	}
/*<                      IF ( NDEG .GT. THRESH )  GOTO 700                   >*/
	if (ndeg > thresh) {
	    goto L700;
	}
/*<                         MINDEG = THRESH                                  >*/
	mindeg = thresh;
/*<                         THRESH = NDEG                                    >*/
	thresh = ndeg;
/*<                         SEARCH = INVP(INODE)                             >*/
	search = invp[inode];
/*<   700          CONTINUE                                                  >*/
L700:
	;
    }
/*<    >*/
    if (nhdsze > 0) {
	qmdqt_(&node, &xadj[1], &adjncy[1], &marker[1], &rchsze, &rchset[1], &
		nbrhd[1]);
    }
/*<   800    IF ( NUM .LT. NEQNS )  GO TO 300                                >*/
L800:
    if (num < *neqns) {
	goto L300;
    }
/*<          RETURN                                                          >*/
    return 0;
/*<       END                                                                >*/
} /* genqmd_ */

