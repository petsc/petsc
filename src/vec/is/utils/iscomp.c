/*$Id: iscomp.c,v 1.22 1999/10/24 14:01:49 bsmith Exp bsmith $*/

#include "sys.h"   /*I "sys.h" I*/
#include "is.h"    /*I "is.h"  I*/

#undef __FUNC__  
#define __FUNC__ "ISEqual"
/*@C
   ISEqual  - Compares if two index sets have the same set of indices.

   Not Collective (Hmmm, that is a little strange, some processors may return true, others false)

   Input Parameters:
.  is1, is2 - The index sets being compared

   Output Parameters:
.  flg - output flag, either PETSC_TRUE (if both index sets have the
         same indices), or PETSC_FALSE if the index sets differ by size 
         or by the set of indices)

   Level: intermediate

   Note: 
   This routine sorts the contents of the index sets before
   the comparision is made, so the order of the indices is immaterial.

.keywords: IS, index set, equal
@*/
int ISEqual(IS is1,IS is2,PetscTruth *flg)
{
  int sz,sz1,sz2,ierr,*ptr1,*ptr2,*a1,*a2;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_COOKIE);
  PetscValidHeaderSpecific(is2,IS_COOKIE);
  PetscValidIntPointer(flg);
  
  ierr = ISGetSize(is1,&sz1);CHKERRQ(ierr);
  ierr = ISGetSize(is2,&sz2);CHKERRQ(ierr);
  if(sz1 != sz2) { *flg = PETSC_FALSE; PetscFunctionReturn(0);}
  
  ierr = ISGetIndices(is1,&ptr1);CHKERRQ(ierr);
  ierr = ISGetIndices(is2,&ptr2);CHKERRQ(ierr);
  
  sz   = sz1*sizeof(int);
  a1   = (int*)PetscMalloc((sz1+1)* sizeof(int));
  a2   = (int*)PetscMalloc((sz2+1)* sizeof(int));

  ierr = PetscMemcpy(a1,ptr1,sz);CHKERRQ(ierr);
  ierr = PetscMemcpy(a2,ptr2,sz);CHKERRQ(ierr);

  ierr = PetscSortInt(sz1,a1);CHKERRQ(ierr);
  ierr = PetscSortInt(sz2,a2);CHKERRQ(ierr);
  ierr = PetscMemcmp(a1,a2,sz,flg);CHKERRQ(ierr);

  ierr = ISRestoreIndices(is1,&ptr1);CHKERRQ(ierr);
  ierr = ISRestoreIndices(is2,&ptr2);CHKERRQ(ierr);
  
  ierr = PetscFree(a1);CHKERRQ(ierr);
  ierr = PetscFree(a2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
  
