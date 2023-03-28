
#include <petsc/private/isimpl.h> /*I "petscis.h"  I*/
#include <petscviewer.h>

/*@
   ISEqual  - Compares if two index sets have the same set of indices.

   Collective

   Input Parameters:
+  is1 - first index set to compare
-  is2 - second index set to compare

   Output Parameter:
.  flg - output flag, either `PETSC_TRUE` (if both index sets have the
         same indices), or `PETSC_FALSE` if the index sets differ by size
         or by the set of indices)

   Level: intermediate

   Note:
   Unlike `ISEqualUnsorted()`, this routine sorts the contents of the index sets (only within each MPI rank) before
   the comparison is made, so the order of the indices on a processor is immaterial.

   Each processor has to have the same indices in the two sets, for example,
.vb
           Processor
             0      1
    is1 = {0, 1} {2, 3}
    is2 = {2, 3} {0, 1}
.ve
    will return false.

.seealso: [](sec_scatter), `IS`, `ISEqualUnsorted()`
@*/
PetscErrorCode ISEqual(IS is1, IS is2, PetscBool *flg)
{
  PetscInt        sz1, sz2, *a1, *a2;
  const PetscInt *ptr1, *ptr2;
  PetscBool       flag;
  MPI_Comm        comm;
  PetscMPIInt     mflg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1, IS_CLASSID, 1);
  PetscValidHeaderSpecific(is2, IS_CLASSID, 2);
  PetscValidBoolPointer(flg, 3);

  if (is1 == is2) {
    *flg = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)is1), PetscObjectComm((PetscObject)is2), &mflg));
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(ISGetSize(is1, &sz1));
  PetscCall(ISGetSize(is2, &sz2));
  if (sz1 != sz2) *flg = PETSC_FALSE;
  else {
    PetscCall(ISGetLocalSize(is1, &sz1));
    PetscCall(ISGetLocalSize(is2, &sz2));

    if (sz1 != sz2) flag = PETSC_FALSE;
    else {
      PetscCall(ISGetIndices(is1, &ptr1));
      PetscCall(ISGetIndices(is2, &ptr2));

      PetscCall(PetscMalloc1(sz1, &a1));
      PetscCall(PetscMalloc1(sz2, &a2));

      PetscCall(PetscArraycpy(a1, ptr1, sz1));
      PetscCall(PetscArraycpy(a2, ptr2, sz2));

      PetscCall(PetscIntSortSemiOrdered(sz1, a1));
      PetscCall(PetscIntSortSemiOrdered(sz2, a2));
      PetscCall(PetscArraycmp(a1, a2, sz1, &flag));

      PetscCall(ISRestoreIndices(is1, &ptr1));
      PetscCall(ISRestoreIndices(is2, &ptr2));

      PetscCall(PetscFree(a1));
      PetscCall(PetscFree(a2));
    }
    PetscCall(PetscObjectGetComm((PetscObject)is1, &comm));
    PetscCall(MPIU_Allreduce(&flag, flg, 1, MPIU_BOOL, MPI_MIN, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   ISEqualUnsorted  - Compares if two index sets have the same indices.

   Collective

   Input Parameters:
+  is1 - first index set to compare
-  is2 - second index set to compare

   Output Parameter:
.  flg - output flag, either `PETSC_TRUE` (if both index sets have the
         same indices), or `PETSC_FALSE` if the index sets differ by size
         or by the set of indices)

   Level: intermediate

   Note:
   Unlike `ISEqual()`, this routine does NOT sort the contents of the index sets before
   the comparison is made, i.e., the order of indices is important.

   Each MPI rank must have the same indices.

.seealso: [](sec_scatter), `IS`, `ISEqual()`
@*/
PetscErrorCode ISEqualUnsorted(IS is1, IS is2, PetscBool *flg)
{
  PetscInt        sz1, sz2;
  const PetscInt *ptr1, *ptr2;
  PetscBool       flag;
  MPI_Comm        comm;
  PetscMPIInt     mflg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(is1, IS_CLASSID, 1);
  PetscValidHeaderSpecific(is2, IS_CLASSID, 2);
  PetscValidBoolPointer(flg, 3);

  if (is1 == is2) {
    *flg = PETSC_TRUE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCallMPI(MPI_Comm_compare(PetscObjectComm((PetscObject)is1), PetscObjectComm((PetscObject)is2), &mflg));
  if (mflg != MPI_CONGRUENT && mflg != MPI_IDENT) {
    *flg = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(ISGetSize(is1, &sz1));
  PetscCall(ISGetSize(is2, &sz2));
  if (sz1 != sz2) *flg = PETSC_FALSE;
  else {
    PetscCall(ISGetLocalSize(is1, &sz1));
    PetscCall(ISGetLocalSize(is2, &sz2));

    if (sz1 != sz2) flag = PETSC_FALSE;
    else {
      PetscCall(ISGetIndices(is1, &ptr1));
      PetscCall(ISGetIndices(is2, &ptr2));

      PetscCall(PetscArraycmp(ptr1, ptr2, sz1, &flag));

      PetscCall(ISRestoreIndices(is1, &ptr1));
      PetscCall(ISRestoreIndices(is2, &ptr2));
    }
    PetscCall(PetscObjectGetComm((PetscObject)is1, &comm));
    PetscCall(MPIU_Allreduce(&flag, flg, 1, MPIU_BOOL, MPI_MIN, comm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
