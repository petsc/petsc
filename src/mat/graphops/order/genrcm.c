/* genrcm.f -- translated by f2c (version 19931217).*/

#include <petscsys.h>
#include <petsc/private/matorderimpl.h>

/*****************************************************************/
/*****************************************************************/
/*********   GENRCM ..... GENERAL REVERSE CUTHILL MCKEE   ********/
/*****************************************************************/

/*    PURPOSE - GENRCM FINDS THE REVERSE CUTHILL-MCKEE*/
/*       ORDERING FOR A GENERAL GRAPH. FOR EACH CONNECTED*/
/*       COMPONENT IN THE GRAPH, GENRCM OBTAINS THE ORDERING*/
/*       BY CALLING THE SUBROUTINE RCM.*/

/*    INPUT PARAMETERS -*/
/*       NEQNS - NUMBER OF EQUATIONS*/
/*       (XADJ, ADJNCY) - ARRAY PAIR CONTAINING THE ADJACENCY*/
/*              STRUCTURE OF THE GRAPH OF THE MATRIX.*/

/*    OUTPUT PARAMETER -*/
/*       PERM - VECTOR THAT CONTAINS THE RCM ORDERING.*/

/*    WORKING PARAMETERS -*/
/*       MASK - IS USED TO MARK VARIABLES THAT HAVE BEEN*/
/*              NUMBERED DURING THE ORDERING PROCESS. IT IS*/
/*              INITIALIZED TO 1, AND SET TO ZERO AS EACH NODE*/
/*              IS NUMBERED.*/
/*       XLS - THE INDEX VECTOR FOR A LEVEL STRUCTURE.  THE*/
/*              LEVEL STRUCTURE IS STORED IN THE CURRENTLY*/
/*              UNUSED SPACES IN THE PERMUTATION VECTOR PERM.*/

/*    PROGRAM SUBROUTINES -*/
/*       FNROOT, RCM.*/
/*****************************************************************/
PetscErrorCode SPARSEPACKgenrcm(const PetscInt *neqns, const PetscInt *xadj, const PetscInt *adjncy, PetscInt *perm, PetscInt *mask, PetscInt *xls)
{
  /* System generated locals */
  PetscInt i__1;

  /* Local variables */
  PetscInt nlvl, root, i, ccsize;
  PetscInt num;

  PetscFunctionBegin;
  if (!*neqns) PetscFunctionReturn(PETSC_SUCCESS);
  if (*neqns == 1) {
    perm[0] = 1;
    mask[0] = 1;
    xls[0]  = 1;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Parameter adjustments */
  --xls;
  --mask;
  --perm;
  --adjncy;
  --xadj;

  i__1 = *neqns;
  for (i = 1; i <= i__1; ++i) mask[i] = 1;
  num  = 1;
  i__1 = *neqns;
  for (i = 1; i <= i__1; ++i) {
    /*          FOR EACH MASKED CONNECTED COMPONENT ...*/
    if (!mask[i]) goto L200;
    root = i;
    /*             FIRST FIND A PSEUDO-PERIPHERAL NODE ROOT.*/
    /*             NOTE THAT THE LEVEL STRUCTURE FOUND BY*/
    /*             FNROOT IS STORED STARTING AT PERM(NUM).*/
    /*             THEN RCM IS CALLED TO ORDER THE COMPONENT*/
    /*             USING ROOT AS THE STARTING NODE.*/
    PetscCall(SPARSEPACKfnroot(&root, &xadj[1], &adjncy[1], &mask[1], &nlvl, &xls[1], &perm[num]));
    PetscCall(SPARSEPACKrcm(&root, &xadj[1], &adjncy[1], &mask[1], &perm[num], &ccsize, &xls[1]));
    num += ccsize;
    if (num > *neqns) PetscFunctionReturn(PETSC_SUCCESS);
  L200:;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
