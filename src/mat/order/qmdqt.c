
/* qmdqt.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <petsc/private/matorderimpl.h>

/***************************************************************/
/********     QMDQT  ..... QUOT MIN DEG QUOT TRANSFORM  ********/
/***************************************************************/

/*    PURPOSE - THIS SUBROUTINE PERFORMS THE QUOTIENT GRAPH  */
/*       TRANSFORMATION AFTER A NODE HAS BEEN ELIMINATED.*/

/*    INPUT PARAMETERS -*/
/*       ROOT - THE NODE JUST ELIMINATED. IT BECOMES THE*/
/*              REPRESENTATIVE OF THE NEW SUPERNODE.*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE.*/
/*       (RCHSZE, RCHSET) - THE REACHABLE SET OF ROOT IN THE*/
/*              OLD QUOTIENT GRAPH.*/
/*       NBRHD - THE NEIGHBORHOOD SET WHICH WILL BE MERGED*/
/*              WITH ROOT TO FORM THE NEW SUPERNODE.*/
/*       MARKER - THE MARKER VECTOR.*/

/*    UPDATED PARAMETER -*/
/*       ADJNCY - BECOMES THE ADJNCY OF THE QUOTIENT GRAPH.*/
/***************************************************************/
PetscErrorCode SPARSEPACKqmdqt(const PetscInt *root, const PetscInt *xadj, const PetscInt *inadjncy, PetscInt *marker, PetscInt *rchsze, PetscInt *rchset, PetscInt *nbrhd)
{
  PetscInt *adjncy = (PetscInt *)inadjncy; /* Used as temporary and reset within this function */
  /* System generated locals */
  PetscInt i__1, i__2;

  /* Local variables */
  PetscInt inhd, irch, node, ilink, j, nabor, jstop, jstrt;

  PetscFunctionBegin;
  /* Parameter adjustments */
  --nbrhd;
  --rchset;
  --marker;
  --adjncy;
  --xadj;

  irch = 0;
  inhd = 0;
  node = *root;
L100:
  jstrt = xadj[node];
  jstop = xadj[node + 1] - 2;
  if (jstop < jstrt) goto L300;

  /*          PLACE REACH NODES INTO THE ADJACENT LIST OF NODE*/
  i__1 = jstop;
  for (j = jstrt; j <= i__1; ++j) {
    ++irch;
    adjncy[j] = rchset[irch];
    if (irch >= *rchsze) goto L400;
  }
/*       LINK TO OTHER SPACE PROVIDED BY THE NBRHD SET.*/
L300:
  ilink = adjncy[jstop + 1];
  node  = -ilink;
  if (ilink < 0) goto L100;
  ++inhd;
  node              = nbrhd[inhd];
  adjncy[jstop + 1] = -node;
  goto L100;
/*       ALL REACHABLE NODES HAVE BEEN SAVED.  END THE ADJ LIST.*/
/*       ADD ROOT TO THE NBR LIST OF EACH NODE IN THE REACH SET.*/
L400:
  adjncy[j + 1] = 0;
  i__1          = *rchsze;
  for (irch = 1; irch <= i__1; ++irch) {
    node = rchset[irch];
    if (marker[node] < 0) goto L600;

    jstrt = xadj[node];
    jstop = xadj[node + 1] - 1;
    i__2  = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nabor = adjncy[j];
      if (marker[nabor] >= 0) goto L500;
      adjncy[j] = *root;
      goto L600;
    L500:;
    }
  L600:;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
