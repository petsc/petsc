
#include <petscis.h>    /*I "petscis.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "ISEqual"
/*@
   ISEqual  - Compares if two index sets have the same set of indices.

   Collective on IS

   Input Parameters:
.  is1, is2 - The index sets being compared

   Output Parameters:
.  flg - output flag, either PETSC_TRUE (if both index sets have the
         same indices), or PETSC_FALSE if the index sets differ by size 
         or by the set of indices)

   Level: intermediate

   Note: 
   This routine sorts the contents of the index sets before
   the comparision is made, so the order of the indices on a processor is immaterial.

   Each processor has to have the same indices in the two sets, for example,
$           Processor 
$             0      1
$    is1 = {0, 1} {2, 3}
$    is2 = {2, 3} {0, 1}
   will return false.

    Concepts: index sets^equal
    Concepts: IS^equal

@*/
PetscErrorCode  ISEqual(IS is1,IS is2,PetscBool  *flg)
{
  PetscInt       sz1,sz2,*a1,*a2;
  const PetscInt *ptr1,*ptr2;
  PetscBool      flag;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscMPIInt    mflg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscValidIntPointer(flg,3);

  if (is1 == is2) {
    *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
  }  

  ierr = MPI_Comm_compare(((PetscObject)is1)->comm,((PetscObject)is2)->comm,&mflg);CHKERRQ(ierr);
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = ISGetSize(is1,&sz1);CHKERRQ(ierr);
  ierr = ISGetSize(is2,&sz2);CHKERRQ(ierr);
  if (sz1 != sz2) { 
    *flg = PETSC_FALSE;
  } else {
    ierr = ISGetLocalSize(is1,&sz1);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is2,&sz2);CHKERRQ(ierr);

    if (sz1 != sz2) {
      flag = PETSC_FALSE;
    } else {
      ierr = ISGetIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISGetIndices(is2,&ptr2);CHKERRQ(ierr);
  
      ierr = PetscMalloc(sz1*sizeof(PetscInt),&a1);CHKERRQ(ierr);
      ierr = PetscMalloc(sz2*sizeof(PetscInt),&a2);CHKERRQ(ierr);

      ierr = PetscMemcpy(a1,ptr1,sz1*sizeof(PetscInt));CHKERRQ(ierr);
      ierr = PetscMemcpy(a2,ptr2,sz2*sizeof(PetscInt));CHKERRQ(ierr);

      ierr = PetscSortInt(sz1,a1);CHKERRQ(ierr);
      ierr = PetscSortInt(sz2,a2);CHKERRQ(ierr);
      ierr = PetscMemcmp(a1,a2,sz1*sizeof(PetscInt),&flag);CHKERRQ(ierr);

      ierr = ISRestoreIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is2,&ptr2);CHKERRQ(ierr);
  
      ierr = PetscFree(a1);CHKERRQ(ierr);
      ierr = PetscFree(a2);CHKERRQ(ierr);
    }
    ierr = PetscObjectGetComm((PetscObject)is1,&comm);CHKERRQ(ierr);  
    ierr = MPI_Allreduce(&flag,flg,1,MPI_INT,MPI_MIN,comm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
  
