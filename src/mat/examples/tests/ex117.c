
static char help[] = "Tests Cholesky factorization and Matview() for a SBAIJ matrix, (bs=2).\n";
/*
  This code is modified from the code contributed by JUNWANG@uwm.edu on Apr 13, 2007
*/

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            mat,fact,B;
  PetscInt       ind1[2],ind2[2];
  PetscScalar    temp[2*2];
  PetscInt       nnz[3];
  IS             perm,colp;
  MatFactorInfo  info;

  PetscInitialize(&argc,&args,0,help);
  nnz[0]=2;nnz[1]=1;nnz[2]=1;

  ierr = MatCreateSeqSBAIJ(PETSC_COMM_SELF,2,6,6,0,nnz,&mat);CHKERRQ(ierr);
  ind1[0]=0;ind1[1]=1;
  temp[0]=3;temp[1]=2;temp[2]=0;temp[3]=3;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind2[0]=4;ind2[1]=5;
  temp[0]=1;temp[1]=1;temp[2]=2;temp[3]=1;
  ierr = MatSetValues(mat,2,ind1,2,ind2,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind1[0]=2;ind1[1]=3;
  temp[0]=4;temp[1]=1;temp[2]=1;temp[3]=5;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind1[0]=4;ind1[1]=5;
  temp[0]=5;temp[1]=1;temp[2]=1;temp[3]=6;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatDuplicate(mat,MAT_SHARE_NONZERO_PATTERN,&B);CHKERRQ(ierr);
  ind1[0]=0;ind1[1]=1;
  temp[0]=3;temp[1]=2;temp[2]=0;temp[3]=3;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind2[0]=4;ind2[1]=5;
  temp[0]=1;temp[1]=1;temp[2]=2;temp[3]=1;
  ierr = MatSetValues(mat,2,ind1,2,ind2,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind1[0]=2;ind1[1]=3;
  temp[0]=4;temp[1]=1;temp[2]=1;temp[3]=5;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);
  ind1[0]=4;ind1[1]=5;
  temp[0]=5;temp[1]=1;temp[2]=1;temp[3]=6;
  ierr = MatSetValues(mat,2,ind1,2,ind1,temp,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"mat: \n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

   // begin cholesky factorization
  ierr = MatGetOrdering(mat,MATORDERINGNATURAL,&perm,&colp);CHKERRQ(ierr);
  ierr = ISDestroy(&colp);CHKERRQ(ierr);
  info.fill=1.0;
  ierr = MatGetFactor(mat,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&fact);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(fact,mat,perm,&info);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(fact,mat,&info);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Chol factor: \n");
  ierr = MatView(fact, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = MatDestroy(&fact);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
