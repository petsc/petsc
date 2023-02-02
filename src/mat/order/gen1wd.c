
/* gen1wd.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <petsc/private/matorderimpl.h>

/*****************************************************************/
/***********     GEN1WD ..... GENERAL ONE-WAY DISSECTION  ********/
/*****************************************************************/

/*    PURPOSE - GEN1WD FINDS A ONE-WAY DISSECTION PARTITIONING*/
/*       FOR A GENERAL GRAPH.  FN1WD IS USED FOR EACH CONNECTED*/
/*       COMPONENT.*/

/*    INPUT PARAMETERS -*/
/*       NEQNS - NUMBER OF EQUATIONS.*/
/*       (XADJ, ADJNCY) - THE ADJACENCY STRUCTURE PAIR.*/

/*    OUTPUT PARAMETERS -*/
/*       (NBLKS, XBLK) - THE PARTITIONING FOUND.*/
/*       PERM - THE ONE-WAY DISSECTION ORDERING.*/

/*    WORKING VECTORS -*/
/*       MASK - IS USED TO MARK VARIABLES THAT HAVE*/
/*              BEEN NUMBERED DURING THE ORDERING PROCESS.*/
/*       (XLS, LS) - LEVEL STRUCTURE USED BY ROOTLS.*/

/*    PROGRAM SUBROUTINES -*/
/*       FN1WD, REVRSE, ROOTLS.*/
/****************************************************************/
PetscErrorCode SPARSEPACKgen1wd(const PetscInt *neqns, const PetscInt *xadj, const PetscInt *adjncy, PetscInt *mask, PetscInt *nblks, PetscInt *xblk, PetscInt *perm, PetscInt *xls, PetscInt *ls)
{
  /* System generated locals */
  PetscInt i__1, i__2, i__3;

  /* Local variables */
  PetscInt node, nsep, lnum, nlvl, root;
  PetscInt i, j, k, ccsize;
  PetscInt num;

  PetscFunctionBegin;
  /* Parameter adjustments */
  --ls;
  --xls;
  --perm;
  --xblk;
  --mask;
  --xadj;
  --adjncy;

  i__1 = *neqns;
  for (i = 1; i <= i__1; ++i) mask[i] = 1;
  *nblks = 0;
  num    = 0;
  i__1   = *neqns;
  for (i = 1; i <= i__1; ++i) {
    if (!mask[i]) goto L400;
    /*             FIND A ONE-WAY DISSECTOR FOR EACH COMPONENT.*/
    root = i;
    PetscCall(SPARSEPACKfn1wd(&root, &xadj[1], &adjncy[1], &mask[1], &nsep, &perm[num + 1], &nlvl, &xls[1], &ls[1]));
    num += nsep;
    ++(*nblks);
    xblk[*nblks] = *neqns - num + 1;
    ccsize       = xls[nlvl + 1] - 1;
    /*             NUMBER THE REMAINING NODES IN THE COMPONENT.*/
    /*             EACH COMPONENT IN THE REMAINING SUBGRAPH FORMS*/
    /*             A NEW BLOCK IN THE PARTITIONING.*/
    i__2 = ccsize;
    for (j = 1; j <= i__2; ++j) {
      node = ls[j];
      if (!mask[node]) goto L300;
      PetscCall(SPARSEPACKrootls(&node, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &perm[num + 1]));
      lnum = num + 1;
      num  = num + xls[nlvl + 1] - 1;
      ++(*nblks);
      xblk[*nblks] = *neqns - num + 1;
      i__3         = num;
      for (k = lnum; k <= i__3; ++k) {
        node       = perm[k];
        mask[node] = 0;
      }
      if (num > *neqns) goto L500;
    L300:;
    }
  L400:;
  }
/*       SINCE DISSECTORS FOUND FIRST SHOULD BE ORDERED LAST,*/
/*       ROUTINE REVRSE IS CALLED TO ADJUST THE ORDERING*/
/*       VECTOR, AND THE BLOCK INDEX VECTOR.*/
L500:
  PetscCall(SPARSEPACKrevrse(neqns, &perm[1]));
  PetscCall(SPARSEPACKrevrse(nblks, &xblk[1]));
  xblk[*nblks + 1] = *neqns + 1;
  PetscFunctionReturn(PETSC_SUCCESS);
}
