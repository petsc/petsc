#ifndef lint
static char vcid[] = "$Id: iscomp.c,v 1.5 1996/02/29 22:35:12 balay Exp bsmith $";
#endif

#include "sys.h"   /*I "sys.h" I*/
#include "is.h"    /*I "is.h"  I*/

/*@
  ISEqual  - Compares if two index sets have the same set of indices.

   Input Parameters:
.  is1, is2 - The index sets being compared

   Output Parameters:
.  flg - output flag, either
$     PETSC_TRUE if both index sets have the same indices;
$     PETSC_FALSE if either the index sets differ by size or 
$          by the set of indices.

   Note: 
   This routine sorts the contents of the index sets before
   the comparision is made, so the order of the indices is immaterial.

.keywords: IS, index set, equal
@*/
int ISEqual(IS is1, IS is2, PetscTruth *flg)
{

  int sz, sz1, sz2, ierr, *ptr1, *ptr2, *a1, *a2;
  
  ierr = ISGetSize(is1, &sz1); CHKERRQ(ierr);
  ierr = ISGetSize(is2, &sz2); CHKERRQ(ierr);
  if( sz1 != sz2) { *flg = PETSC_FALSE; return 0;}
  
  ierr = ISGetIndices(is1, &ptr1); CHKERRQ(ierr);
  ierr = ISGetIndices(is2, &ptr2); CHKERRQ(ierr);
  
  sz   = sz1*sizeof(int);
  a1   = (int *) PetscMalloc((sz1+1)* sizeof(int));
  a2   = (int *) PetscMalloc((sz2+1)* sizeof(int));

  PetscMemcpy(a1, ptr1, sz);
  PetscMemcpy(a2, ptr2, sz);

  ierr = PetscSortInt(sz1,a1); CHKERRQ(ierr);
  ierr = PetscSortInt(sz2,a2); CHKERRQ(ierr);
  if(!PetscMemcmp(a1, a2, sz)) {*flg = PETSC_TRUE;}
  else {*flg = PETSC_FALSE;}

  ierr = ISRestoreIndices(is1, &ptr1); CHKERRQ(ierr);
  ierr = ISRestoreIndices(is2, &ptr2); CHKERRQ(ierr);
  
  PetscFree(a1);
  PetscFree(a2);
  return 0;
}
  
