/* function subroutines used by pf.c */

#include "pf.h"

PetscErrorCode GetListofEdges_Power(PFDATA *pfdata,int *edgelist)
{
  PetscErrorCode ierr;
  PetscInt       i,fbus,tbus,nbranches=pfdata->nbranch;
  EDGE_Power     branch=pfdata->branch;
  PetscBool      netview=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHasName(NULL,NULL, "-powernet_view",&netview);CHKERRQ(ierr);
  for (i=0; i<nbranches; i++) {
    fbus = branch[i].internal_i;
    tbus = branch[i].internal_j;
    edgelist[2*i]   = fbus;
    edgelist[2*i+1] = tbus;
    if (netview) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"branch %d, bus[%d] -> bus[%d]\n",i,fbus,tbus);CHKERRQ(ierr);
    }
  }
  if (netview) {
    for (i=0; i<pfdata->nbus; i++) {
      if (pfdata->bus[i].ngen) {
        printf(" bus %d: gen\n",i);
      } else if (pfdata->bus[i].nload) {
        printf(" bus %d: load\n",i);
      }
    }
  }
  PetscFunctionReturn(0);
}
