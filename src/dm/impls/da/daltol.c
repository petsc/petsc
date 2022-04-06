
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc/private/dmdaimpl.h>    /*I   "petscdmda.h"   I*/

/*
   DMLocalToLocalCreate_DA - Creates the local to local scatter

   Collective on da

   Input Parameter:
.  da - the distributed array

*/
PetscErrorCode  DMLocalToLocalCreate_DA(DM da)
{
  PetscInt       *idx,left,j,count,up,down,i,bottom,top,k,dim=da->dim;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);

  if (dd->ltol) PetscFunctionReturn(0);
  /*
     We simply remap the values in the from part of
     global to local to read from an array with the ghost values
     rather then from the plain array.
  */
  PetscCall(VecScatterCopy(dd->gtol,&dd->ltol));
  PetscCall(PetscLogObjectParent((PetscObject)da,(PetscObject)dd->ltol));
  if (dim == 1) {
    left = dd->xs - dd->Xs;
    PetscCall(PetscMalloc1(dd->xe-dd->xs,&idx));
    for (j=0; j<dd->xe-dd->xs; j++) idx[j] = left + j;
  } else if (dim == 2) {
    left  = dd->xs - dd->Xs; down  = dd->ys - dd->Ys; up    = down + dd->ye-dd->ys;
    PetscCall(PetscMalloc1((dd->xe-dd->xs)*(up - down),&idx));
    count = 0;
    for (i=down; i<up; i++) {
      for (j=0; j<dd->xe-dd->xs; j++) {
        idx[count++] = left + i*(dd->Xe-dd->Xs) + j;
      }
    }
  } else if (dim == 3) {
    left   = dd->xs - dd->Xs;
    bottom = dd->ys - dd->Ys; top = bottom + dd->ye-dd->ys;
    down   = dd->zs - dd->Zs; up  = down + dd->ze-dd->zs;
    count  = (dd->xe-dd->xs)*(top-bottom)*(up-down);
    PetscCall(PetscMalloc1(count,&idx));
    count  = 0;
    for (i=down; i<up; i++) {
      for (j=bottom; j<top; j++) {
        for (k=0; k<dd->xe-dd->xs; k++) {
          idx[count++] = (left+j*(dd->Xe-dd->Xs))+i*(dd->Xe-dd->Xs)*(dd->Ye-dd->Ys) + k;
        }
      }
    }
  } else SETERRQ(PetscObjectComm((PetscObject)da),PETSC_ERR_ARG_CORRUPT,"DMDA has invalid dimension %" PetscInt_FMT,dim);

  PetscCall(VecScatterRemap(dd->ltol,idx,NULL));
  PetscCall(PetscFree(idx));
  PetscFunctionReturn(0);
}

/*
   DMLocalToLocalBegin_DA - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be followed by DMLocalToLocalEnd_DA().

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Notes:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

*/
PetscErrorCode  DMLocalToLocalBegin_DA(DM da,Vec g,InsertMode mode,Vec l)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (!dd->ltol) {
    PetscCall(DMLocalToLocalCreate_DA(da));
  }
  PetscCall(VecScatterBegin(dd->ltol,g,l,mode,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}

/*
   DMLocalToLocalEnd_DA - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly.  Must be preceded by
   DMLocalToLocalBegin_DA().

   Neighbor-wise Collective on da

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Note:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

*/
PetscErrorCode  DMLocalToLocalEnd_DA(DM da,Vec g,InsertMode mode,Vec l)
{
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscCall(VecScatterEnd(dd->ltol,g,l,mode,SCATTER_FORWARD));
  PetscFunctionReturn(0);
}
