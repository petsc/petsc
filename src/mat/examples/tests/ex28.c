
static char help[] = "Tests MatReorderForNonzeroDiagonal()\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,LU;
  Vec            x,y;
  PetscInt       nnz[4]={2,1,1,1},col[4],i;
  PetscErrorCode ierr;
  PetscScalar    values[4];
  IS             rowperm,colperm;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,4,4,2,nnz,&A);CHKERRQ(ierr);

  /* build test matrix */
  values[0]=1.0;values[1]=-1.0;
  col[0]=0;col[1]=2; i=0;
  ierr = MatSetValues(A,1,&i,2,col,values,INSERT_VALUES);CHKERRQ(ierr);
  values[0]=1.0;
  col[0]=1;i=1;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES);CHKERRQ(ierr);
  values[0]=-1.0;
  col[0]=3;i=2;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES);CHKERRQ(ierr);
  values[0]=1.0;
  col[0]=2;i=3;
  ierr = MatSetValues(A,1,&i,1,col,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatGetOrdering(A,MATORDERINGNATURAL,&rowperm,&colperm);CHKERRQ(ierr);
  ierr = MatReorderForNonzeroDiagonal(A,1.e-12,rowperm,colperm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"column and row perms\n");CHKERRQ(ierr);
  ierr = ISView(rowperm,0);CHKERRQ(ierr);
  ierr = ISView(colperm,0);CHKERRQ(ierr);
  ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&LU);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(LU,A,rowperm,colperm,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(LU,A,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatView(LU,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,4);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  values[0]=0;values[1]=1.0;values[2]=-1.0;values[3]=1.0;
  for (i=0; i<4; i++) col[i]=i;
  ierr = VecSetValues(x,4,col,values,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatSolve(LU,x,y);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(&rowperm);CHKERRQ(ierr);
  ierr = ISDestroy(&colperm);CHKERRQ(ierr);
  ierr = MatDestroy(&LU);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}


