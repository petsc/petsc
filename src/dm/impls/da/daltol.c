
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include <petsc-private/daimpl.h>    /*I   "petscdmda.h"   I*/

#undef __FUNCT__
#define __FUNCT__ "DMDALocalToLocalCreate"
/*
   DMDALocalToLocalCreate - Creates the local to local scatter

   Collective on DMDA

   Input Parameter:
.  da - the distributed array

*/
PetscErrorCode  DMDALocalToLocalCreate(DM da)
{
  PetscErrorCode ierr;
  PetscInt       *idx,left,j,count,up,down,i,bottom,top,k;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);

  if (dd->ltol) PetscFunctionReturn(0);
  /*
     We simply remap the values in the from part of
     global to local to read from an array with the ghost values
     rather then from the plain array.
  */
  ierr = VecScatterCopy(dd->gtol,&dd->ltol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,dd->ltol);CHKERRQ(ierr);
  if (dd->dim == 1) {
    left = dd->xs - dd->Xs;
    ierr = PetscMalloc((dd->xe-dd->xs)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    for (j=0; j<dd->xe-dd->xs; j++) {
      idx[j] = left + j;
    }
  } else if (dd->dim == 2) {
    left  = dd->xs - dd->Xs; down  = dd->ys - dd->Ys; up    = down + dd->ye-dd->ys;
    ierr = PetscMalloc((dd->xe-dd->xs)*(up - down)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    count = 0;
    for (i=down; i<up; i++) {
      for (j=0; j<dd->xe-dd->xs; j++) {
	idx[count++] = left + i*(dd->Xe-dd->Xs) + j;
      }
    }
  } else if (dd->dim == 3) {
    left   = dd->xs - dd->Xs;
    bottom = dd->ys - dd->Ys; top = bottom + dd->ye-dd->ys ;
    down   = dd->zs - dd->Zs; up  = down + dd->ze-dd->zs;
    count  = (dd->xe-dd->xs)*(top-bottom)*(up-down);
    ierr = PetscMalloc(count*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    count  = 0;
    for (i=down; i<up; i++) {
      for (j=bottom; j<top; j++) {
	for (k=0; k<dd->xe-dd->xs; k++) {
	  idx[count++] = (left+j*(dd->Xe-dd->Xs))+i*(dd->Xe-dd->Xs)*(dd->Ye-dd->Ys) + k;
	}
      }
    }
  } else SETERRQ1(((PetscObject)da)->comm,PETSC_ERR_ARG_CORRUPT,"DMDA has invalid dimension %D",dd->dim);

  ierr = VecScatterRemap(dd->ltol,idx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscFree(idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDALocalToLocalBegin"
/*@
   DMDALocalToLocalBegin - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be followed by DMDALocalToLocalEnd().

   Neighbor-wise Collective on DMDA and Vec

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Notes:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.keywords: distributed array, local-to-local, begin

.seealso: DMDALocalToLocalEnd(), DMLocalToGlobalBegin()
@*/
PetscErrorCode  DMDALocalToLocalBegin(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  if (!dd->ltol) {
    ierr = DMDALocalToLocalCreate(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(dd->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDALocalToLocalEnd"
/*@
   DMDALocalToLocalEnd - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly.  Must be preceeded by
   DMDALocalToLocalBegin().

   Neighbor-wise Collective on DMDA and Vec

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Note:
   The local vectors used here need not be the same as those
   obtained from DMCreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be
   obtained with VecDuplicate() from the DMDA originating vectors.

.keywords: distributed array, local-to-local, end

.seealso: DMDALocalToLocalBegin(), DMLocalToGlobalBegin()
@*/
PetscErrorCode  DMDALocalToLocalEnd(DM da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;
  DM_DA          *dd = (DM_DA*)da->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_CLASSID,1);
  PetscValidHeaderSpecific(g,VEC_CLASSID,2);
  PetscValidHeaderSpecific(g,VEC_CLASSID,4);
  ierr = VecScatterEnd(dd->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

