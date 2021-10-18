
#include <petsc/private/isimpl.h>    /*I "petscis.h"  I*/
#include <petscviewer.h>

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
   Unlike ISEqualUnsorted(), this routine sorts the contents of the index sets before
   the comparison is made, so the order of the indices on a processor is immaterial.

   Each processor has to have the same indices in the two sets, for example,
$           Processor
$             0      1
$    is1 = {0, 1} {2, 3}
$    is2 = {2, 3} {0, 1}
   will return false.

.seealso: ISEqualUnsorted()
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
  PetscValidBoolPointer(flg,3);

  if (is1 == is2) {
    *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)is1),PetscObjectComm((PetscObject)is2),&mflg);CHKERRMPI(ierr);
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = ISGetSize(is1,&sz1);CHKERRQ(ierr);
  ierr = ISGetSize(is2,&sz2);CHKERRQ(ierr);
  if (sz1 != sz2) *flg = PETSC_FALSE;
  else {
    ierr = ISGetLocalSize(is1,&sz1);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is2,&sz2);CHKERRQ(ierr);

    if (sz1 != sz2) flag = PETSC_FALSE;
    else {
      ierr = ISGetIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISGetIndices(is2,&ptr2);CHKERRQ(ierr);

      ierr = PetscMalloc1(sz1,&a1);CHKERRQ(ierr);
      ierr = PetscMalloc1(sz2,&a2);CHKERRQ(ierr);

      ierr = PetscArraycpy(a1,ptr1,sz1);CHKERRQ(ierr);
      ierr = PetscArraycpy(a2,ptr2,sz2);CHKERRQ(ierr);

      ierr = PetscIntSortSemiOrdered(sz1,a1);CHKERRQ(ierr);
      ierr = PetscIntSortSemiOrdered(sz2,a2);CHKERRQ(ierr);
      ierr = PetscArraycmp(a1,a2,sz1,&flag);CHKERRQ(ierr);

      ierr = ISRestoreIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is2,&ptr2);CHKERRQ(ierr);

      ierr = PetscFree(a1);CHKERRQ(ierr);
      ierr = PetscFree(a2);CHKERRQ(ierr);
    }
    ierr = PetscObjectGetComm((PetscObject)is1,&comm);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&flag,flg,1,MPIU_BOOL,MPI_MIN,comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   ISEqualUnsorted  - Compares if two index sets have the same indices.

   Collective on IS

   Input Parameters:
.  is1, is2 - The index sets being compared

   Output Parameters:
.  flg - output flag, either PETSC_TRUE (if both index sets have the
         same indices), or PETSC_FALSE if the index sets differ by size
         or by the set of indices)

   Level: intermediate

   Note:
   Unlike ISEqual(), this routine does NOT sort the contents of the index sets before
   the comparison is made, i.e., the order of indices is important.

.seealso: ISEqual()
@*/
PetscErrorCode  ISEqualUnsorted(IS is1,IS is2,PetscBool  *flg)
{
  PetscInt       sz1,sz2;
  const PetscInt *ptr1,*ptr2;
  PetscBool      flag;
  MPI_Comm       comm;
  PetscErrorCode ierr;
  PetscMPIInt    mflg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1,IS_CLASSID,1);
  PetscValidHeaderSpecific(is2,IS_CLASSID,2);
  PetscValidBoolPointer(flg,3);

  if (is1 == is2) {
    *flg = PETSC_TRUE;
    PetscFunctionReturn(0);
  }

  ierr = MPI_Comm_compare(PetscObjectComm((PetscObject)is1),PetscObjectComm((PetscObject)is2),&mflg);CHKERRMPI(ierr);
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(0);
  }

  ierr = ISGetSize(is1,&sz1);CHKERRQ(ierr);
  ierr = ISGetSize(is2,&sz2);CHKERRQ(ierr);
  if (sz1 != sz2) *flg = PETSC_FALSE;
  else {
    ierr = ISGetLocalSize(is1,&sz1);CHKERRQ(ierr);
    ierr = ISGetLocalSize(is2,&sz2);CHKERRQ(ierr);

    if (sz1 != sz2) flag = PETSC_FALSE;
    else {
      ierr = ISGetIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISGetIndices(is2,&ptr2);CHKERRQ(ierr);

      ierr = PetscArraycmp(ptr1,ptr2,sz1,&flag);CHKERRQ(ierr);

      ierr = ISRestoreIndices(is1,&ptr1);CHKERRQ(ierr);
      ierr = ISRestoreIndices(is2,&ptr2);CHKERRQ(ierr);
    }
    ierr = PetscObjectGetComm((PetscObject)is1,&comm);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(&flag,flg,1,MPIU_BOOL,MPI_MIN,comm);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

