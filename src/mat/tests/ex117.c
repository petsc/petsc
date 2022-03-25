
static char help[] = "Tests Cholesky factorization for a SBAIJ matrix, (bs=2).\n";
/*
  This code is modified from the code contributed by JUNWANG@uwm.edu on Apr 13, 2007
*/

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            mat,fact,B;
  PetscInt       ind1[2],ind2[2];
  PetscScalar    temp[4];
  PetscInt       nnz[3];
  IS             perm,colp;
  MatFactorInfo  info;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&args,0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  nnz[0]=2;nnz[1]=1;nnz[2]=1;
  PetscCall(MatCreateSeqSBAIJ(PETSC_COMM_SELF,2,6,6,0,nnz,&mat));

  ind1[0]=0;ind1[1]=1;
  temp[0]=3;temp[1]=2;temp[2]=0;temp[3]=3;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));
  ind2[0]=4;ind2[1]=5;
  temp[0]=1;temp[1]=1;temp[2]=2;temp[3]=1;
  PetscCall(MatSetValues(mat,2,ind1,2,ind2,temp,INSERT_VALUES));
  ind1[0]=2;ind1[1]=3;
  temp[0]=4;temp[1]=1;temp[2]=1;temp[3]=5;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));
  ind1[0]=4;ind1[1]=5;
  temp[0]=5;temp[1]=1;temp[2]=1;temp[3]=6;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  PetscCall(MatDuplicate(mat,MAT_SHARE_NONZERO_PATTERN,&B));
  ind1[0]=0;ind1[1]=1;
  temp[0]=3;temp[1]=2;temp[2]=0;temp[3]=3;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));
  ind2[0]=4;ind2[1]=5;
  temp[0]=1;temp[1]=1;temp[2]=2;temp[3]=1;
  PetscCall(MatSetValues(mat,2,ind1,2,ind2,temp,INSERT_VALUES));
  ind1[0]=2;ind1[1]=3;
  temp[0]=4;temp[1]=1;temp[2]=1;temp[3]=5;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));
  ind1[0]=4;ind1[1]=5;
  temp[0]=5;temp[1]=1;temp[2]=1;temp[3]=6;
  PetscCall(MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"mat: \n"));
  PetscCall(MatView(mat,PETSC_VIEWER_STDOUT_SELF));

  /* begin cholesky factorization */
  PetscCall(MatGetOrdering(mat,MATORDERINGNATURAL,&perm,&colp));
  PetscCall(ISDestroy(&colp));

  info.fill=1.0;
  PetscCall(MatGetFactor(mat,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&fact));
  PetscCall(MatCholeskyFactorSymbolic(fact,mat,perm,&info));
  PetscCall(MatCholeskyFactorNumeric(fact,mat,&info));

  PetscCall(ISDestroy(&perm));
  PetscCall(MatDestroy(&mat));
  PetscCall(MatDestroy(&fact));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}
