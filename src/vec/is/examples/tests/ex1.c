/*
       Formatted test for ISGeneral routines.
*/

static char help[] = "Tests IS general routines.\n\n";

#include <petscis.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  PetscInt       i,n,*indices;
  const PetscInt *ii;
  IS             is,newis;
  PetscBool      flg;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  /*
     Test IS of size 0 
  */
  ierr = ISCreateGeneral(PETSC_COMM_SELF,0,&n,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISGetSize(is,&n);CHKERRQ(ierr);
  if (n != 0) SETERRQ(PETSC_COMM_SELF,1,"ISGetSize");
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     Create large IS and test ISGetIndices()
  */
  n = 10000 + rank;
  ierr = PetscMalloc(n*sizeof(PetscInt),&indices);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    indices[i] = rank + i;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = ISGetIndices(is,&ii);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ii[i] != indices[i]) SETERRQ(PETSC_COMM_SELF,1,"ISGetIndices");
  }
  ierr = ISRestoreIndices(is,&ii);CHKERRQ(ierr);

  /* 
     Check identity and permutation 
  */
  ierr = ISPermutation(is,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,1,"ISPermutation");
  ierr = ISIdentity(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"ISIdentity");
  ierr = ISSetPermutation(is);CHKERRQ(ierr);
  ierr = ISSetIdentity(is);CHKERRQ(ierr);
  ierr = ISPermutation(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"ISPermutation");
  ierr = ISIdentity(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"ISIdentity");

  /*
     Check equality of index sets 
  */
  ierr = ISEqual(is,is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"ISEqual");

  /*
     Sorting 
  */
  ierr = ISSort(is);CHKERRQ(ierr);
  ierr = ISSorted(is,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,1,"ISSort");

  /*
     Thinks it is a different type?
  */
  ierr = PetscObjectTypeCompare((PetscObject)is,ISSTRIDE,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,1,"ISStride");
  ierr = PetscObjectTypeCompare((PetscObject)is,ISBLOCK,&flg);CHKERRQ(ierr);
  if (flg) SETERRQ(PETSC_COMM_SELF,1,"ISBlock");

  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /*
     Inverting permutation
  */
  for (i=0; i<n; i++) {
    indices[i] = n - i - 1;
  }
  ierr = ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,&is);CHKERRQ(ierr);
  ierr = PetscFree(indices);CHKERRQ(ierr);
  ierr = ISSetPermutation(is);CHKERRQ(ierr);
  ierr = ISInvertPermutation(is,PETSC_DECIDE,&newis);CHKERRQ(ierr);
  ierr = ISGetIndices(newis,&ii);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    if (ii[i] != n - i - 1) SETERRQ(PETSC_COMM_SELF,1,"ISInvertPermutation");
  }
  ierr = ISRestoreIndices(newis,&ii);CHKERRQ(ierr);
  ierr = ISDestroy(&newis);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 






