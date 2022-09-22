
/* degree.f -- translated by f2c (version 19931217).*/

#include <petsc/private/matorderimpl.h>

/*****************************************************************/
/*********     DEGREE ..... DEGREE IN MASKED COMPONENT   *********/
/*****************************************************************/

/*    PURPOSE - THIS ROUTINE COMPUTES THE DEGREES OF THE NODES*/
/*       IN THE CONNECTED COMPONENT SPECIFIED BY MASK AND ROOT*/
/*       NODES FOR WHICH MASK IS ZERO ARE IGNORED.*/

/*    INPUT PARAMETER -*/
/*       ROOT - IS THE INPUT NODE THAT DEFINES THE COMPONENT.*/
/*       (XADJ, ADJNCY) - ADJACENCY STRUCTURE PAIR.*/
/*       MASK - SPECIFIES A SECTION SUBGRAPH.*/

/*    OUTPUT PARAMETERS -*/
/*       DEG - ARRAY CONTAINING THE DEGREES OF THE NODES IN*/
/*             THE COMPONENT.*/
/*       CCSIZE-SIZE OF THE COMPONENT SPECIFED BY MASK AND ROOT*/
/*    WORKING PARAMETER -*/
/*       LS - A TEMPORARY VECTOR USED TO STORE THE NODES OF THE*/
/*              COMPONENT LEVEL BY LEVEL.*/
/*****************************************************************/
PetscErrorCode SPARSEPACKdegree(const PetscInt *root, const PetscInt *inxadj, const PetscInt *adjncy, PetscInt *mask, PetscInt *deg, PetscInt *ccsize, PetscInt *ls)
{
  PetscInt *xadj = (PetscInt *)inxadj; /* Used as temporary and reset within this function */
  /* System generated locals */
  PetscInt i__1, i__2;

  /* Local variables */
  PetscInt ideg, node, i, j, jstop, jstrt, lbegin, lvlend, lvsize, nbr;
  /*       INITIALIZATION ...*/
  /*       THE ARRAY XADJ IS USED AS A TEMPORARY MARKER TO*/
  /*       INDICATE WHICH NODES HAVE BEEN CONSIDERED SO FAR.*/

  PetscFunctionBegin;
  /* Parameter adjustments */
  --ls;
  --deg;
  --mask;
  --adjncy;
  --xadj;

  ls[1]       = *root;
  xadj[*root] = -xadj[*root];
  lvlend      = 0;
  *ccsize     = 1;
/*       LBEGIN IS THE POINTER TO THE BEGINNING OF THE CURRENT*/
/*       LEVEL, AND LVLEND POINTS TO THE END OF THIS LEVEL.*/
L100:
  lbegin = lvlend + 1;
  lvlend = *ccsize;
  /*       FIND THE DEGREES OF NODES IN THE CURRENT LEVEL,*/
  /*       AND AT THE SAME TIME, GENERATE THE NEXT LEVEL.*/
  i__1 = lvlend;
  for (i = lbegin; i <= i__1; ++i) {
    node  = ls[i];
    jstrt = -xadj[node];
    i__2  = xadj[node + 1];
    jstop = (PetscInt)PetscAbsInt(i__2) - 1;
    ideg  = 0;
    if (jstop < jstrt) goto L300;
    i__2 = jstop;
    for (j = jstrt; j <= i__2; ++j) {
      nbr = adjncy[j];
      if (!mask[nbr]) goto L200;
      ++ideg;
      if (xadj[nbr] < 0) goto L200;
      xadj[nbr] = -xadj[nbr];
      ++(*ccsize);
      ls[*ccsize] = nbr;
    L200:;
    }
  L300:
    deg[node] = ideg;
  }
  /*       COMPUTE THE CURRENT LEVEL WIDTH. */
  /*       IF IT IS NONZERO, GENERATE ANOTHER LEVEL.*/
  lvsize = *ccsize - lvlend;
  if (lvsize > 0) goto L100;
  /*       RESET XADJ TO ITS CORRECT SIGN AND RETURN. */
  /*       ------------------------------------------*/
  i__1 = *ccsize;
  for (i = 1; i <= i__1; ++i) {
    node       = ls[i];
    xadj[node] = -xadj[node];
  }
  PetscFunctionReturn(0);
}
