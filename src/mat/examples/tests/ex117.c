
static char help[] = "Tests Cholesky factorization and Matview() for a SBAIJ matrix, (bs=2).\n";
/*
  This code is modified from the code contributed by JUNWANG@uwm.edu on Apr 13, 2007
*/

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            mat,fact;
  int            ind1[2],ind2[2];
  PetscScalar    temp[2*2];
  PetscInt       *nnz=new PetscInt[3];
  IS             perm,colp;
  MatFactorInfo  info;

  PetscInitialize(&argc,&args,0,0);
  nnz[0]=2;nnz[1]=1;nnz[1]=1;

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

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

   printf("mat: \n");
   MatView(mat,PETSC_VIEWER_STDOUT_SELF);

   // begin cholesky factorization
   MatGetOrdering(mat,MATORDERING_NATURAL,&perm,&colp);
   ierr = ISDestroy(colp);CHKERRQ(ierr);    
   info.fill=1.0; 
   ierr = MatGetFactor(mat,"petsc",MAT_FACTOR_CHOLESKY,&fact);CHKERRQ(ierr);
   ierr = MatCholeskyFactorSymbolic(mat,perm,&info,&fact); CHKERRQ(ierr);
   ierr = MatCholeskyFactorNumeric(mat,&info,&fact);CHKERRQ(ierr);
   printf("Chol factor: \n");
   ierr = MatView(fact, PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

   ierr = ISDestroy(perm);CHKERRQ(ierr);
   ierr= MatDestroy(mat);CHKERRQ(ierr);
   ierr = MatDestroy(fact);CHKERRQ(ierr);
   PetscFinalize();
   return 0;
}
