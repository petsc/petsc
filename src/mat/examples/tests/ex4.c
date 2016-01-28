
static char help[] = "Creates a matrix, inserts some values, and tests MatGetSubMatrices() and MatZeroEntries().\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat,submat,*submatrices;
  PetscInt       m = 10,n = 10,i = 4,tmp,rstart,rend;
  PetscErrorCode ierr;
  IS             irkeep,ickeep;
  PetscScalar    value = 1.0;
  PetscViewer    sviewer;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_COMMON);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&mat);CHKERRQ(ierr);
  ierr = MatSetSizes(mat,PETSC_DECIDE,PETSC_DECIDE,m,n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(mat);CHKERRQ(ierr);
  ierr = MatSetUp(mat);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(mat,&rstart,&rend);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    value = (PetscReal)i+1; tmp = i % 5;
    ierr  = MatSetValues(mat,1,&tmp,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original matrix\n");CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Form submatrix with rows 2-4 and columns 4-8 */
  ierr   = ISCreateStride(PETSC_COMM_SELF,3,2,1,&irkeep);CHKERRQ(ierr);
  ierr   = ISCreateStride(PETSC_COMM_SELF,5,4,1,&ickeep);CHKERRQ(ierr);
  ierr   = MatGetSubMatrices(mat,1,&irkeep,&ickeep,MAT_INITIAL_MATRIX,&submatrices);CHKERRQ(ierr);
  submat = *submatrices;
  ierr   = PetscFree(submatrices);CHKERRQ(ierr);
  /*
     sviewer will cause the submatrices (one per processor) to be printed in the correct order
  */
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Submatrices\n");CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = MatView(submat,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Zero the original matrix */
  ierr = PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"Original zeroed matrix\n");CHKERRQ(ierr);
  ierr = MatZeroEntries(mat);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = ISDestroy(&irkeep);CHKERRQ(ierr);
  ierr = ISDestroy(&ickeep);CHKERRQ(ierr);
  ierr = MatDestroy(&submat);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

