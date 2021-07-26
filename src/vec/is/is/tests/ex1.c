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
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);

  /*
     Test IS of size 0
  */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,0,&n,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  if (n != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetSize");
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     Create large IS and test ISGetIndices()
  */
  n    = 10000 + rank;
  ierr = PetscMalloc1(n,&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) indices[i] = rank + i;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ii[i] != indices[i]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetIndices");
  }
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);

  /*
     Check identity and permutation
  */
  /* ISPermutation doesn't check if not set */
  ierr = ISPermutation(is,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISPermutation");
  ierr = ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (rank == 0 && !flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  if (rank && flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  ierr = ISIdentity(is,&flg);CHKERRQ(ierr);
  if (rank == 0 && !flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISIdentity");
  if (rank && flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISIdentity");
  ierr = ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (rank == 0 && !flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  if (rank && flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");
  /* we can override the computed values with ISSetInfo() */
  ierr = ISSetInfo(is,IS_PERMUTATION,IS_LOCAL,permanent,PETSC_TRUE);CHKERRQ(ierr);
  ierr = ISSetInfo(is,IS_IDENTITY,IS_LOCAL,permanent,PETSC_TRUE);CHKERRQ(ierr);
  ierr = ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)");
  ierr = ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_IDENTITY,IS_LOCAL)");

  ierr = ISClearInfoCache(is,PETSC_TRUE);CHKERRQ(ierr);

  /*
     Check equality of index sets
  */
  ierr = ISEqual(is,is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISEqual");

  /*
     Sorting
  */
  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = ISSorted(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISSort");
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_SORTED,IS_LOCAL)");
  ierr = ISSorted(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISSort");
  ierr = ISGetInfo(is,IS_SORTED,IS_LOCAL,compute,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_SORTED,IS_LOCAL)");

  /*
     Thinks it is a different type?
  */
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISStride");
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISBlock");

  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     Inverting permutation
  */
  for (i=0; i<n; i++) indices[i] = n - i - 1;
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = ISSetPermutation(is);CHKERRQ(ierr);
  ierr = ISInvertPermutation(is,PETSC_DECIDE,&newis);CHKERRQ(ierr);
  ierr = ISGetIndices(newis,&ii);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ii[i] != n - i - 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISInvertPermutation");
  }
  ierr = ISRestoreIndices(newis,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(&newis);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: {{1 2 3 4 5}}

TEST*/
