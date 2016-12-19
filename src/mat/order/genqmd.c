
/* genqmd.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <petsc/private/matorderimpl.h>

/******************************************************************/
/***********    GENQMD ..... QUOT MIN DEGREE ORDERING    **********/
/******************************************************************/
/*    PURPOSE - THIS ROUTINE IMPLEMENTS THE MINIMUM DEGREE        */
/*       ALGORITHM.  IT MAKES USE OF THE IMPLICIT REPRESENT-      */
/*       ATION OF THE ELIMINATION GRAPHS BY QUOTIENT GRAPHS,      */
/*       AND THE NOTION OF INDISTINGUISHABLE NODES.               */
/*       CAUTION - THE ADJACENCY VECTOR ADJNCY WILL BE            */
/*       DESTROYED.                                               */
/*                                                                */
/*    INPUT PARAMETERS -                                          */
/*       NEQNS - NUMBER OF EQUATIONS.                             */
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.                */
/*                                                                */
/*    OUTPUT PARAMETERS -                                         */
/*       PERM - THE MINIMUM DEGREE ORDERING.                      */
/*       INVP - THE INVERSE OF PERM.                              */
/*                                                                */
/*    WORKING PARAMETERS -                                        */
/*       DEG - THE DEGREE VECTOR. DEG(I) IS NEGATIVE MEANS        */
/*              NODE I HAS BEEN NUMBERED.                         */
/*       MARKER - A MARKER VECTOR, WHERE MARKER(I) IS             */
/*              NEGATIVE MEANS NODE I HAS BEEN MERGED WITH        */
/*              ANOTHER NODE AND THUS CAN BE IGNORED.             */
/*       RCHSET - VECTOR USED FOR THE REACHABLE SET.              */
/*       NBRHD - VECTOR USED FOR THE NEIGHBORHOOD SET.            */
/*       QSIZE - VECTOR USED TO STORE THE SIZE OF                 */
/*              INDISTINGUISHABLE SUPERNODES.                     */
/*       QLINK - VECTOR TO STORE INDISTINGUISHABLE NODES,         */
/*              I, QLINK(I), QLINK(QLINK(I)) ... ARE THE          */
/*              MEMBERS OF THE SUPERNODE REPRESENTED BY I.        */
/*                                                                */
/*    PROGRAM SUBROUTINES -                                       */
/*       QMDRCH, QMDQT, QMDUPD.                                   */
/*                                                                */
/******************************************************************/
/*                                                                */
/*                                                                */
PetscErrorCode SPARSEPACKgenqmd(const PetscInt *neqns,const PetscInt *xadj,const PetscInt *adjncy,
                                PetscInt *perm, PetscInt *invp, PetscInt *deg, PetscInt *marker,
                                PetscInt *rchset, PetscInt *nbrhd, PetscInt *qsize, PetscInt *qlink, PetscInt *nofsub)
{
  /* System generated locals */
  PetscInt i__1;

  /* Local variables */
  PetscInt ndeg, irch, node, nump1, j, inode;
  PetscInt ip, np, mindeg, search;
  PetscInt nhdsze, nxnode, rchsze, thresh, num;

/*       INITIALIZE DEGREE VECTOR AND OTHER WORKING VARIABLES.   */

  PetscFunctionBegin;
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

  mindeg  = *neqns;
  *nofsub = 0;
  i__1    = *neqns;
  for (node = 1; node <= i__1; ++node) {
    perm[node]   = node;
    invp[node]   = node;
    marker[node] = 0;
    qsize[node]  = 1;
    qlink[node]  = 0;
    ndeg         = xadj[node + 1] - xadj[node];
    deg[node]    = ndeg;
    if (ndeg < mindeg) mindeg = ndeg;
  }
  num = 0;
/*       PERFORM THRESHOLD SEARCH TO GET A NODE OF MIN DEGREE.   */
/*       VARIABLE SEARCH POINTS TO WHERE SEARCH SHOULD START.    */
L200:
  search = 1;
  thresh = mindeg;
  mindeg = *neqns;
L300:
  nump1 = num + 1;
  if (nump1 > search) search = nump1;
  i__1 = *neqns;
  for (j = search; j <= i__1; ++j) {
    node = perm[j];
    if (marker[node] < 0) goto L400;
    ndeg = deg[node];
    if (ndeg <= thresh) goto L500;
    if (ndeg < mindeg) mindeg = ndeg;
L400:
    ;
  }
  goto L200;
/*          NODE HAS MINIMUM DEGREE. FIND ITS REACHABLE SETS BY    */
/*          CALLING QMDRCH.                                        */
L500:
  search       = j;
  *nofsub     += deg[node];
  marker[node] = 1;
  SPARSEPACKqmdrch(&node, &xadj[1], &adjncy[1], &deg[1], &marker[1], &rchsze, &
                   rchset[1], &nhdsze, &nbrhd[1]);
/*          ELIMINATE ALL NODES INDISTINGUISHABLE FROM NODE.       */
/*          THEY ARE GIVEN BY NODE, QLINK(NODE), ....              */
  nxnode = node;
L600:
  ++num;
  np           = invp[nxnode];
  ip           = perm[num];
  perm[np]     = ip;
  invp[ip]     = np;
  perm[num]    = nxnode;
  invp[nxnode] = num;
  deg[nxnode]  = -1;
  nxnode       = qlink[nxnode];
  if (nxnode > 0) goto L600;
  if (rchsze <= 0) goto L800;

/*             UPDATE THE DEGREES OF THE NODES IN THE REACHABLE     */
/*             SET AND IDENTIFY INDISTINGUISHABLE NODES.            */
  SPARSEPACKqmdupd(&xadj[1], &adjncy[1], &rchsze, &rchset[1], &deg[1], &qsize[1], &
                   qlink[1], &marker[1], &rchset[rchsze + 1], &nbrhd[nhdsze + 1]);

/*             RESET MARKER VALUE OF NODES IN REACH SET.            */
/*             UPDATE THRESHOLD VALUE FOR CYCLIC SEARCH.            */
/*             ALSO CALL QMDQT TO FORM NEW QUOTIENT GRAPH.          */
  marker[node] = 0;
  i__1         = rchsze;
  for (irch = 1; irch <= i__1; ++irch) {
    inode = rchset[irch];
    if (marker[inode] < 0) goto L700;

    marker[inode] = 0;
    ndeg          = deg[inode];
    if (ndeg < mindeg) mindeg = ndeg;
    if (ndeg > thresh) goto L700;
    mindeg = thresh;
    thresh = ndeg;
    search = invp[inode];
L700:
    ;
  }
  if (nhdsze > 0) {
    SPARSEPACKqmdqt(&node, &xadj[1], &adjncy[1], &marker[1], &rchsze, &rchset[1], &
                    nbrhd[1]);
  }
L800:
  if (num < *neqns) goto L300;
  PetscFunctionReturn(0);
}
