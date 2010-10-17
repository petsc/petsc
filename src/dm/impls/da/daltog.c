#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscdm.h"   I*/

PetscErrorCode PETSCDM_DLLEXPORT DMLocalToGlobal_DA(DM da,Vec l,InsertMode mode,Vec g)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  if (mode == INSERT_VALUES) {
    ierr = VecScatterBegin(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(dd->ltog,l,g,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  } else {
  }
  PetscFunctionReturn(0);
}








