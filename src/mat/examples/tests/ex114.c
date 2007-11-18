
static char help[] = "Tests sequential MatGetRowMax(), MatGetRowMin(), MatGetRowMaxAbs()\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            min,max,maxabs;
  PetscInt       imin[5],imax[5],imaxabs[5],indices[6],row;
  PetscScalar    values[6];
  PetscErrorCode ierr;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,5,6,0,PETSC_NULL,&A);CHKERRQ(ierr);  
  row  = 0; 
  indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 3; indices[4] = 4; indices[5] = 5;
  values[0] = -1.0; values[1] = 0.0; values[2] = 1.0; values[3] = 3.0; values[4] = 4.0; values[5] = -5.0;
  ierr = MatSetValues(A,1,&row,6,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  row = 1;
  ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  row = 4;
  ierr = MatSetValues(A,1,&row,1,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
  row = 4;
  ierr = MatSetValues(A,1,&row,2,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecCreateSeq(PETSC_COMM_SELF,5,&min);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,5,&max);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,5,&maxabs);CHKERRQ(ierr);

  ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
  ierr = VecView(min,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
  ierr = VecView(max,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
  ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatConvert(A,MATSEQDENSE,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

  ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
  ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
  ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
  ierr = VecView(min,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
  ierr = VecView(max,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
  ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = VecDestroy(min);CHKERRQ(ierr);
  ierr = VecDestroy(max);CHKERRQ(ierr);
  ierr = VecDestroy(maxabs);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

