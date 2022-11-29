/*
       Formatted test for ISGeneral routines.
*/

static char help[] = "Tests IS general routines.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  PetscMPIInt     rank, size;
  PetscInt        i, n, *indices;
  const PetscInt *ii;
  IS              is, newis;
  PetscBool       flg;
  PetscBool       permanent = PETSC_FALSE;
  PetscBool       compute   = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));

  /*
     Test IS of size 0
  */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 0, &n, PETSC_COPY_VALUES, &is));
  PetscCall(ISGetSize(is, &n));
  PetscCheck(n == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetSize");
  PetscCall(ISDestroy(&is));

  /*
     Create large IS and test ISGetIndices()
  */
  n = 10000 + rank;
  PetscCall(PetscMalloc1(n, &indices));
  for (i = 0; i < n; i++) indices[i] = rank + i;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, indices, PETSC_COPY_VALUES, &is));
  PetscCall(ISGetIndices(is, &ii));
  for (i = 0; i < n; i++) PetscCheck(ii[i] == indices[i], PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetIndices");
  PetscCall(ISRestoreIndices(is, &ii));

  /*
     Check identity and permutation
  */
  /* ISPermutation doesn't check if not set */
  PetscCall(ISPermutation(is, &flg));
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISPermutation");
  PetscCall(ISGetInfo(is, IS_PERMUTATION, IS_LOCAL, compute, &flg));
  PetscCheck(rank != 0 || flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  PetscCheck(rank == 0 || !flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  PetscCall(ISIdentity(is, &flg));
  PetscCheck(rank != 0 || flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISIdentity");
  PetscCheck(rank == 0 || !flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISIdentity");
  PetscCall(ISGetInfo(is, IS_IDENTITY, IS_LOCAL, compute, &flg));
  PetscCheck(rank != 0 || flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  PetscCheck(rank == 0 || !flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  /* we can override the computed values with ISSetInfo() */
  PetscCall(ISSetInfo(is, IS_PERMUTATION, IS_LOCAL, permanent, PETSC_TRUE));
  PetscCall(ISSetInfo(is, IS_IDENTITY, IS_LOCAL, permanent, PETSC_TRUE));
  PetscCall(ISGetInfo(is, IS_PERMUTATION, IS_LOCAL, compute, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  PetscCall(ISGetInfo(is, IS_IDENTITY, IS_LOCAL, compute, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_IDENTITY,IS_LOCAL)");

  PetscCall(ISClearInfoCache(is, PETSC_TRUE));

  /*
     Check equality of index sets
  */
  PetscCall(ISEqual(is, is, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISEqual");

  /*
     Sorting
  */
  PetscCall(ISSort(is));
  PetscCall(ISSorted(is, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISSort");
  PetscCall(ISGetInfo(is, IS_SORTED, IS_LOCAL, compute, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_SORTED,IS_LOCAL)");
  PetscCall(ISSorted(is, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISSort");
  PetscCall(ISGetInfo(is, IS_SORTED, IS_LOCAL, compute, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISGetInfo(IS_SORTED,IS_LOCAL)");

  /*
     Thinks it is a different type?
  */
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISSTRIDE, &flg));
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISStride");
  PetscCall(PetscObjectTypeCompare((PetscObject)is, ISBLOCK, &flg));
  PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISBlock");

  PetscCall(ISDestroy(&is));

  /*
     Inverting permutation
  */
  for (i = 0; i < n; i++) indices[i] = n - i - 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, n, indices, PETSC_COPY_VALUES, &is));
  PetscCall(PetscFree(indices));
  PetscCall(ISSetPermutation(is));
  PetscCall(ISInvertPermutation(is, PETSC_DECIDE, &newis));
  PetscCall(ISGetIndices(newis, &ii));
  for (i = 0; i < n; i++) PetscCheck(ii[i] == n - i - 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "ISInvertPermutation");
  PetscCall(ISRestoreIndices(newis, &ii));
  PetscCall(ISDestroy(&newis));
  PetscCall(ISDestroy(&is));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}

TEST*/
