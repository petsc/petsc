
/* qmdupd.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <petsc/private/matorderimpl.h>

/******************************************************************/
/***********     QMDUPD ..... QUOT MIN DEG UPDATE      ************/
/******************************************************************/
/******************************************************************/

/*    PURPOSE - THIS ROUTINE PERFORMS DEGREE UPDATE FOR A SET*/
/*       OF NODES IN THE MINIMUM DEGREE ALGORITHM.*/

/*    INPUT PARAMETERS -*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.*/
/*       (NLIST, LIST) - THE LIST OF NODES WHOSE DEGREE HAS TO*/
/*              BE UPDATED.*/

/*    UPDATED PARAMETERS -*/
/*       DEG - THE DEGREE VECTOR.*/
/*       QSIZE - SIZE OF INDISTINGUISHABLE SUPERNODES.*/
/*       QLINK - LINKED LIST FOR INDISTINGUISHABLE NODES.*/
/*       MARKER - USED TO MARK THOSE NODES IN REACH/NBRHD SETS.*/

/*    WORKING PARAMETERS -*/
/*       RCHSET - THE REACHABLE SET.*/
/*       NBRHD -  THE NEIGHBORHOOD SET.*/

/*    PROGRAM SUBROUTINES -*/
/*       QMDMRG.*/
/******************************************************************/
PetscErrorCode SPARSEPACKqmdupd(const PetscInt *xadj,const PetscInt *adjncy,const PetscInt *nlist,const PetscInt *list, PetscInt *deg,
                                PetscInt *qsize, PetscInt *qlink, PetscInt *marker, PetscInt *rchset, PetscInt *nbrhd)
{
  /* System generated locals */
  PetscInt i__1, i__2;

  /* Local variables */
  PetscInt inhd, irch, node, mark, j, inode, nabor, jstop, jstrt, il;
  PetscInt nhdsze, rchsze, deg0, deg1;

/*       FIND ALL ELIMINATED SUPERNODES THAT ARE ADJACENT*/
/*       TO SOME NODES IN THE GIVEN LIST. PUT THEM INTO.*/
/*       (NHDSZE, NBRHD). DEG0 CONTAINS THE NUMBER OF*/
/*       NODES IN THE LIST.*/


  PetscFunctionBegin;
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

  if (*nlist <= 0) PetscFunctionReturn(0);
  deg0   = 0;
  nhdsze = 0;
  i__1   = *nlist;
  for (il = 1; il <= i__1; ++il) {
    node  = list[il];
    deg0 += qsize[node];
    jstrt = xadj[node];
    jstop = xadj[node + 1] - 1;
    i__2  = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nabor = adjncy[j];
      if (marker[nabor] != 0 || deg[nabor] >= 0) goto L100;
      marker[nabor] = -1;
      ++nhdsze;
      nbrhd[nhdsze] = nabor;
L100:
      ;
    }
  }
/*       MERGE INDISTINGUISHABLE NODES IN THE LIST BY*/
/*       CALLING THE SUBROUTINE QMDMRG.*/
  if (nhdsze > 0) {
    SPARSEPACKqmdmrg(&xadj[1], &adjncy[1], &deg[1], &qsize[1], &qlink[1], &marker[1], &deg0, &nhdsze, &nbrhd[1], &rchset[1], &nbrhd[nhdsze + 1]);
  }
/*       FIND THE NEW DEGREES OF THE NODES THAT HAVE NOT BEEN*/
/*       MERGED.*/
  i__1 = *nlist;
  for (il = 1; il <= i__1; ++il) {
    node = list[il];
    mark = marker[node];
    if (mark > 1 || mark < 0) goto L600;
    marker[node] = 2;
    SPARSEPACKqmdrch(&node, &xadj[1], &adjncy[1], &deg[1], &marker[1], &rchsze, &rchset[1], &nhdsze, &nbrhd[1]);
    deg1 = deg0;
    if (rchsze <= 0) goto L400;
    i__2 = rchsze;
    for (irch = 1; irch <= i__2; ++irch) {
      inode         = rchset[irch];
      deg1         += qsize[inode];
      marker[inode] = 0;
    }
L400:
    deg[node] = deg1 - 1;
    if (nhdsze <= 0) goto L600;
    i__2 = nhdsze;
    for (inhd = 1; inhd <= i__2; ++inhd) {
      inode         = nbrhd[inhd];
      marker[inode] = 0;
    }
L600:
    ;
  }
  PetscFunctionReturn(0);
}

