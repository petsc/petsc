#define PETSCDM_DLL

/*
  Code for manipulating distributed regular arrays in parallel.
*/

#include "private/daimpl.h"    /*I   "petscda.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "DALocalToLocalCreate"
/*
   DALocalToLocalCreate - Creates the local to local scatter

   Collective on DA

   Input Parameter:
.  da - the distributed array

*/
PetscErrorCode PETSCDM_DLLEXPORT DALocalToLocalCreate(DA da)
{
  PetscErrorCode ierr;
  PetscInt *idx,left,j,count,up,down,i,bottom,top,k;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);

  if (da->ltol) PetscFunctionReturn(0);
  /* 
     We simply remap the values in the from part of 
     global to local to read from an array with the ghost values 
     rather then from the plain array.
  */
  ierr = VecScatterCopy(da->gtol,&da->ltol);CHKERRQ(ierr);
  ierr = PetscLogObjectParent(da,da->ltol);CHKERRQ(ierr);
  if (da->dim == 1) {
    left = da->xs - da->Xs;
    ierr = PetscMalloc((da->xe-da->xs)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    for (j=0; j<da->xe-da->xs; j++) {
      idx[j] = left + j;
    }  
  } else if (da->dim == 2) {
    left  = da->xs - da->Xs; down  = da->ys - da->Ys; up    = down + da->ye-da->ys;
    ierr = PetscMalloc((da->xe-da->xs)*(up - down)*sizeof(PetscInt),&idx);CHKERRQ(ierr);
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
    ierr = PetscMalloc(count*sizeof(PetscInt),&idx);CHKERRQ(ierr);
    count  = 0;
    for (i=down; i<up; i++) {
      for (j=bottom; j<top; j++) {
	for (k=0; k<da->xe-da->xs; k++) {
	  idx[count++] = (left+j*(da->Xe-da->Xs))+i*(da->Xe-da->Xs)*(da->Ye-da->Ys) + k;
	}
      }
    }
  } else SETERRQ1(PETSC_ERR_ARG_CORRUPT,"DA has invalid dimension %D",da->dim);

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
PetscErrorCode PETSCDM_DLLEXPORT DALocalToLocalBegin(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  if (!da->ltol) {
    ierr = DALocalToLocalCreate(da);CHKERRQ(ierr);
  }
  ierr = VecScatterBegin(da->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
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
PetscErrorCode PETSCDM_DLLEXPORT DALocalToLocalEnd(DA da,Vec g,InsertMode mode,Vec l)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  PetscValidHeaderSpecific(g,VEC_COOKIE,2);
  PetscValidHeaderSpecific(g,VEC_COOKIE,4);
  ierr = VecScatterEnd(da->ltol,g,l,mode,SCATTER_FORWARD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

