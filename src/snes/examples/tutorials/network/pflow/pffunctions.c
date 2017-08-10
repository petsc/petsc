/* function subroutines used by pf.c */

#include "pf.h"

PetscErrorCode GetListofEdges_Power(PetscInt nbranches, EDGE_Power branch,int edges[])
{
  PetscInt       i,fbus,tbus;

  PetscFunctionBegin;
  for (i=0; i<nbranches; i++) {
    fbus = branch[i].internal_i;
    tbus = branch[i].internal_j;
    edges[2*i]   = fbus;
    edges[2*i+1] = tbus;
    /* printf("branch %d, bus[%d] -> bus[%d]\n",i,fbus,tbus); */
  }
  PetscFunctionReturn(0);
}
