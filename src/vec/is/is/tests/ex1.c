/*
       Formatted test for ISGeneral routines.
*/

static char help[] = "Tests IS general routines.\n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       i,n,*indices;
  const PetscInt *ii;
  IS             is,newis;
  PetscBool      flg;
  PetscBool      permanent = PETSC_FALSE;
  PetscBool      compute = PETSC_TRUE;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  /*
     Test IS of size 0
  */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,0,&n,PETSC_COPY_VALUES,&is));
  CHKERRQ(ISGetSize(is,&n));
  PetscCheckFalse(n != 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetSize");
  CHKERRQ(ISDestroy(&is));

  /*
     Create large IS and test ISGetIndices()
  */
  n    = 10000 + rank;
  CHKERRQ(PetscMalloc1(n,&indices));
  for (i=0; i<n; i++) indices[i] = rank + i;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is));
  CHKERRQ(ISGetIndices(is,&ii));
  for (i=0; i<n; i++) {
    PetscCheckFalse(ii[i] != indices[i],PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetIndices");
  }
  CHKERRQ(ISRestoreIndices(is,&ii));

  /*
     Check identity and permutation
  */
  /* ISPermutation doesn't check if not set */
  CHKERRQ(ISPermutation(is,&flg));
  PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISPermutation");
  CHKERRQ(ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,&flg));
  PetscCheckFalse(rank == 0 && !flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  PetscCheckFalse(rank && flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  CHKERRQ(ISIdentity(is,&flg));
  PetscCheckFalse(rank == 0 && !flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISIdentity");
  PetscCheckFalse(rank && flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISIdentity");
  CHKERRQ(ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,&flg));
  PetscCheckFalse(rank == 0 && !flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  PetscCheckFalse(rank && flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  /* we can override the computed values with ISSetInfo() */
  CHKERRQ(ISSetInfo(is,IS_PERMUTATION,IS_LOCAL,permanent,PETSC_TRUE));
  CHKERRQ(ISSetInfo(is,IS_IDENTITY,IS_LOCAL,permanent,PETSC_TRUE));
  CHKERRQ(ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  CHKERRQ(ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");

  CHKERRQ(ISClearInfoCache(is,PETSC_TRUE));

  /*
     Check equality of index sets
  */
  CHKERRQ(ISEqual(is,is,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISEqual");

  /*
     Sorting
  */
  CHKERRQ(ISSort(is));
  CHKERRQ(ISSorted(is,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISSort");
  CHKERRQ(ISGetInfo(is,IS_SORTED,IS_LOCAL,compute,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_SORTED,IS_LOCAL)");
  CHKERRQ(ISSorted(is,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISSort");
  CHKERRQ(ISGetInfo(is,IS_SORTED,IS_LOCAL,compute,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_SORTED,IS_LOCAL)");

  /*
     Thinks it is a different type?
  */
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg));
  PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStride");
  CHKERRQ(PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&flg));
  PetscCheck(!flg,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISBlock");

  CHKERRQ(ISDestroy(&is));

  /*
     Inverting permutation
  */
  for (i=0; i<n; i++) indices[i] = n - i - 1;
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is));
  CHKERRQ(PetscFree(indices));
  CHKERRQ(ISSetPermutation(is));
  CHKERRQ(ISInvertPermutation(is,PETSC_DECIDE,&newis));
  CHKERRQ(ISGetIndices(newis,&ii));
  for (i=0; i<n; i++) {
    PetscCheckFalse(ii[i] != n - i - 1,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISInvertPermutation");
  }
  CHKERRQ(ISRestoreIndices(newis,&ii));
  CHKERRQ(ISDestroy(&newis));
  CHKERRQ(ISDestroy(&is));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}

TEST*/
