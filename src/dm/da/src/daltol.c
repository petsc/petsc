/*$Id: daltol.c,v 1.24 2001/03/23 23:25:00 balay Exp $*/
 
/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "src/dm/da/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DALocalToLocalCreate"
/*
   DALocalToLocalCreate - Creates the local to local scatter

   Collective on DA

   Input Parameter:
.  da - the distributed array

*/
int DALocalToLocalCreate(DA da)
{
  int *idx,left,j,ierr,count,up,down,i,bottom,top,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);

  if (da->ltol) PetscFunctionReturn(0);
  /* 
     We simply remap the values in the from part of 
     global to local to read from an array with the ghost values 
     rather then from the plain array.
  */
  ierr = VecScatterCopy(da->gtol,&da->ltol);CHKERRQ(ierr);
  PetscLogObjectParent(da,da->ltol);
  if (da->dim == 1) {
    left = da->xs - da->Xs;
    ierr = PetscMalloc((da->xe-da->xs)*sizeof(int),&idx);CHKERRQ(ierr);
    for (j=0; j<da->xe-da->xs; j++) {
      idx[j] = left + j;
    }  
  } else if (da->dim == 2) {
    left  = da->xs - da->Xs; down  = da->ys - da->Ys; up    = down + da->ye-da->ys;
    ierr = PetscMalloc((da->xe-da->xs)*(up - down)*sizeof(int),&idx);CHKERRQ(ierr);
    count = 0;
    for (i=down; i<up; i++) {
      for (j=0; j<da->xe-da->xs; j++) {
	idx[count++] = left + i*(da->Xe-da->Xs) + j;
      }
    }
  } else if (da->dim == 3) {
    left   = da->xs - da->Xs; 
    bottom = da->ys - da->Ys; top = bottom + da->ye-da->ys ;
    down   = da->zs - da->Zs; up  = down + da->ze-da->zs;
    count  = (da->xe-da->xs)*(top-bottom)*(up-down);
    ierr = PetscMalloc(count*sizeof(int),&idx);CHKERRQ(ierr);
    count  = 0;
    for (i=down; i<up; i++) {
      for (j=bottom; j<top; j++) {
	for (k=0; k<da->xe-da->xs; k++) {
	  idx[count++] = (left+j*(da->Xe-da->Xs))+i*(da->Xe-da->Xs)*(da->Ye-da->Ys) + k;
	}
      }
    }
  } else SETERRQ1(1,"DA has invalid dimension %d",da->dim);

  ierr = VecScatterRemap(da->ltol,idx,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscFree(idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DALocalToLocalBegin"
/*@
   DALocalToLocalBegin - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly. Must be followed by DALocalToLocalEnd().

   Collective on DA and Vec

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Notes:
   The local vectors used here need not be the same as those
   obtained from DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, local-to-local, begin

.seealso: DALocalToLocalEnd(), DALocalToGlobal()
@*/
int DALocalToLocalBegin(DA da,Vec g,InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  if (!da->ltol) {
    ierr = DALocalToLocalCreate(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(g,l,mode,SCATTER_FORWARD,da->ltol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "DALocalToLocalEnd"
/*@
   DALocalToLocalEnd - Maps from a local vector (including ghost points
   that contain irrelevant values) to another local vector where the ghost
   points in the second are set correctly.  Must be preceeded by 
   DALocalToLocalBegin().

   Collective on DA and Vec

   Input Parameters:
+  da - the distributed array context
.  g - the original local vector
-  mode - one of INSERT_VALUES or ADD_VALUES

   Output Parameter:
.  l  - the local vector with correct ghost values

   Level: intermediate

   Note:
   The local vectors used here need not be the same as those
   obtained from DACreateLocalVector(), BUT they
   must have the same parallel data layout; they could, for example, be 
   obtained with VecDuplicate() from the DA originating vectors.

.keywords: distributed array, local-to-local, end

.seealso: DALocalToLocalBegin(), DALocalToGlobal()
@*/
int DALocalToLocalEnd(DA da,Vec g,InsertMode mode,Vec l)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DA_COOKIE);
  ierr = VecScatterEnd(g,l,mode,SCATTER_FORWARD,da->ltol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

